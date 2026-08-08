"""Model registry: construct, load and run the four architectures.

Reconstruction configs come from
/home/jokubasb/protein_protein/all_class/training/program/architectures.py
(SHA256 26e5d650eafa091a5237d0f1c8811e908e9012036a8e9f061a9616c5a9281d42) because the
``gnn`` and ``gnn_mlp`` checkpoints carry no ``model_config`` — only ``model_state_dict``,
``arch``, ``setup`` and ``label_names``. Getting a config wrong here loads silently and
predicts nonsense, so ``tests/test_parity_models.py`` checks all 20 combinations against
the original classes.

The four architectures differ in three ways that must not be conflated:

  arch      node features x        extra input      forward
  --------  ---------------------  ---------------  ---------------------------
  mlp       (none — no graph)      ESM (n,1280)     model(esm)
  gnn       5-d structural         none             model(batch)
  gnn_mlp   5 ‖ 64 embedder = 69   ESM via embedder model(batch)
  joint     5-d structural         ESM (n,1280)     model(esm, batch)[0]

``gnn_mlp`` needs a second checkpoint. ``GNNMLPArch.prepare()`` (architectures.py:157-180)
trains a prerequisite MLP whose penultimate 64-d activations become 64 of the GNN's 69
input features, and ``save_checkpoint`` (:71) then saves only the GNN. Verified by
``scripts/spike_gnn_mlp_embedder.py``: the separately-saved ``runs/<setup>/mlp/model.pt``
IS that discarded MLP (reproduces the recorded ``predictions.npz`` for dna_rna to
max|diff| = 1.19e-07, i.e. float32 epsilon), because ``main.py:60-62`` seeds identically
in both jobs and nothing consumes the torch RNG before ``trainer.train_mlp``. So the
registry pairs each gnn_mlp checkpoint with the setup's mlp checkpoint as ``embedder.pt``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch_geometric.data import Batch, Data

from .models import BindingSiteGNN, BindingSiteMLP, JointMLPGNN, mlp_embed
from .setups import SETUPS, LabelSetup

ARCHS = ("gnn_mlp", "joint", "mlp", "gnn")
DEFAULT_ARCH = "gnn_mlp"

ARCH_DISPLAY = {
    "mlp": "MLP",
    "gnn": "GNN",
    "gnn_mlp": "GNN+MLP",
    "joint": "Joint",
}

ARCH_DESCRIPTION = {
    "mlp": "Sequence only — an MLP over ESM-2 embeddings. No structure.",
    "gnn": "Structure only — GATv2 over the Voronoi contact graph. No sequence.",
    "gnn_mlp": "Structure + sequence — GATv2 over structural features concatenated with "
               "a pretrained MLP's 64-d ESM embedding.",
    "joint": "Structure + sequence trained end to end, with an auxiliary sequence loss.",
}

#: Architectures that need an ESM-2 embedding of the chain.
NEEDS_ESM = {"mlp", "gnn_mlp", "joint"}
#: Architectures that need the Voronoi contact graph.
NEEDS_GRAPH = {"gnn", "gnn_mlp", "joint"}


def build_model(arch: str, k: int, checkpoint: dict) -> torch.nn.Module:
    """Construct an architecture with the exact training-time hyper-parameters."""
    if arch == "mlp":
        # MLPArch.prepare, architectures.py:98-101 (also stored in the checkpoint).
        cfg = checkpoint.get("model_config") or {
            "input_dim": 1280, "hidden_dims": [1024, 512, 128, 64],
            "output_dim": k, "dropout": 0.4,
        }
        return BindingSiteMLP(**cfg)
    if arch == "gnn":
        # StructuralGNNArch.build_model, architectures.py:137-139.
        return BindingSiteGNN(input_dim=5, hidden_dim=256, heads=4, dropout=0.2,
                              output_dim=k)
    if arch == "gnn_mlp":
        # GNNMLPArch.build_model, architectures.py:183-185.
        return BindingSiteGNN(input_dim=69, hidden_dim=512, heads=4, dropout=0.5,
                              output_dim=k)
    if arch == "joint":
        # JointArch.build_model, architectures.py:212-219.
        mlp_config = {"input_dim": 1280, "hidden_dims": [512, 128, 64],
                      "output_dim": k, "dropout": 0.4}
        gnn_config = {"input_dim": 5 + 64, "hidden_dim": 512, "heads": 4, "dropout": 0.3}
        mlw = float(checkpoint.get("mlp_loss_weight", 0.5))
        return JointMLPGNN(mlp_config, gnn_config, mlp_loss_weight=mlw)
    raise ValueError(f"unknown architecture: {arch!r}")


@dataclass
class LoadedModel:
    setup: LabelSetup
    arch: str
    model: torch.nn.Module
    embedder: Optional[BindingSiteMLP]  # gnn_mlp only
    label_names: list[str]

    @torch.inference_mode()
    def predict(self, graph: Optional[Data], esm: Optional[torch.Tensor],
                device: torch.device) -> torch.Tensor:
        """Per-residue probabilities, shape (n_residues, k).

        ``graph.x`` must be the 5-d structural block for every graph architecture; this
        method appends the 64-d embedder features for gnn_mlp itself, so callers build the
        graph once and reuse it across architectures.
        """
        self.model.eval()
        if self.arch == "mlp":
            logits = self.model(esm.to(device))
            return torch.sigmoid(logits).cpu()

        if self.arch == "gnn_mlp":
            self.embedder.to(device)
            emb = mlp_embed(self.embedder, esm.to(device))
            graph = graph.clone()
            graph.x = torch.cat([graph.x.to(device), emb], dim=-1)

        batch = Batch.from_data_list([graph]).to(device)
        if self.arch == "joint":
            # JointArch.predict_test, architectures.py:234 — the GNN head is the
            # prediction; mlp_logits is the auxiliary head.
            gnn_logits, _ = self.model(esm.to(device), batch)
        else:
            gnn_logits = self.model(batch)
        return torch.sigmoid(gnn_logits).cpu()


class ModelRegistry:
    """Loads ``models/<setup>/<arch>/model.pt`` lazily and caches them."""

    def __init__(self, models_dir: Path, device: torch.device):
        self.models_dir = Path(models_dir)
        self.device = device
        self._cache: Dict[tuple[str, str], LoadedModel] = {}
        manifest_path = self.models_dir / "MANIFEST.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def manifest_digest(self) -> str:
        """Identifies the exact set of weights in use, so prediction caches invalidate
        when a checkpoint is swapped."""
        import hashlib
        h = hashlib.sha256()
        for key in sorted(self.manifest):
            h.update(key.encode())
            h.update(str(self.manifest[key].get("sha256", "")).encode())
        return h.hexdigest()[:16]

    def available(self) -> list[dict]:
        """Every (setup, arch) pair with a checkpoint on disk."""
        out = []
        for setup_name, setup in SETUPS.items():
            for arch in ARCHS:
                path = self.models_dir / setup_name / arch / "model.pt"
                if not path.exists():
                    continue
                entry = self.manifest.get(f"{setup_name}/{arch}", {})
                out.append({
                    "setup": setup_name,
                    "arch": arch,
                    "arch_display": ARCH_DISPLAY[arch],
                    "arch_description": ARCH_DESCRIPTION[arch],
                    "label_names": setup.label_names,
                    "description": setup.description,
                    "hint": setup.hint,
                    "needs_esm": arch in NEEDS_ESM,
                    "needs_graph": arch in NEEDS_GRAPH,
                    "metrics": entry.get("metrics", {}),
                    "sha256": entry.get("sha256"),
                })
        return out

    def get(self, setup_name: str, arch: str) -> LoadedModel:
        key = (setup_name, arch)
        if key in self._cache:
            return self._cache[key]

        if setup_name not in SETUPS:
            raise KeyError(f"unknown setup {setup_name!r}")
        if arch not in ARCHS:
            raise KeyError(f"unknown arch {arch!r}")
        setup = SETUPS[setup_name]

        path = self.models_dir / setup_name / arch / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint at {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)

        label_names = list(ck.get("label_names") or setup.label_names)
        if label_names != setup.label_names:
            raise ValueError(
                f"checkpoint {path} declares labels {label_names}, "
                f"but setup '{setup_name}' expects {setup.label_names}")

        model = build_model(arch, setup.num_labels, ck)
        model.load_state_dict(ck["model_state_dict"])
        model.to(self.device).eval()

        embedder = None
        if arch == "gnn_mlp":
            emb_path = self.models_dir / setup_name / arch / "embedder.pt"
            if not emb_path.exists():
                raise FileNotFoundError(
                    f"{arch} needs the paired 64-d embedder at {emb_path}. "
                    f"Run scripts/export_models.py to produce it.")
            eck = torch.load(emb_path, map_location="cpu", weights_only=False)
            embedder = BindingSiteMLP(**eck["model_config"])
            embedder.load_state_dict(eck["model_state_dict"])
            embedder.to(self.device).eval()

        loaded = LoadedModel(setup=setup, arch=arch, model=model,
                             embedder=embedder, label_names=label_names)
        self._cache[key] = loaded
        return loaded
