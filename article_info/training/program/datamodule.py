"""Data discovery, filtering, splitting and loading.

Wraps the loaders in ``train_multilabel`` and adds, per :class:`LabelSetup`:

* protein filtering -- keep only chains with >= 1 binding residue of the setup's
  type (cheap pre-scan of ``graph_nodes.csv``);
* label transformation -- map the loaded base 4-label tensors to the setup's
  ``k`` labels;
* a post-load safety filter -- drop anything whose derived labels are all-zero.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import train_multilabel as tm
from cluster_weights import build_chain_weights

from config import Config
from setups import BASE_LABEL_COLUMNS, LabelSetup


class ProteinDataModule:
    """Discovers, filters, splits and loads protein data for one setup."""

    def __init__(self, config: Config, setup: LabelSetup, reweight: str = "none"):
        assert list(tm.LABEL_COLUMNS) == BASE_LABEL_COLUMNS, (
            "train_multilabel.LABEL_COLUMNS changed; update setups.BASE_LABEL_COLUMNS"
        )
        self.cfg = config
        self.setup = setup
        self.reweight = reweight

        self.available: List[str] = []
        self.qualifying: List[str] = []
        self.train_proteins: List[str] = []
        self.val_proteins: List[str] = []
        self.test_proteins: List[str] = []
        self.weight_map: Dict[str, float] = {}

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        """Discover, filter by binding type, split, and build sample weights."""
        self._discover()
        self._filter_by_binding()
        self._split()
        self._build_weights()

    def _discover(self) -> None:
        graph_dir, emb_dir = self.cfg.graph_dir, self.cfg.emb_dir

        graph_proteins = set()
        if os.path.isdir(graph_dir):
            for d in os.listdir(graph_dir):
                if os.path.isfile(os.path.join(graph_dir, d, "graph_nodes.csv")):
                    graph_proteins.add(d)

        emb_proteins = set()
        if os.path.isdir(emb_dir):
            for f in os.listdir(emb_dir):
                if f.endswith(".pt"):
                    emb_proteins.add(f[:-3])

        self.available = sorted(graph_proteins & emb_proteins)
        print(f"Proteins with graphs: {len(graph_proteins)}")
        print(f"Proteins with embeddings: {len(emb_proteins)}")
        print(f"Available (both): {len(self.available)}")
        if not self.available:
            raise RuntimeError("No proteins with both graph and embedding data found.")

    def _protein_qualifies(self, name: str) -> bool:
        """True if the chain has >= 1 surface binding residue for this setup.

        Mirrors ``load_protein_base``'s surface rule (residue ``sas_area`` summed
        over atoms > 0) but only reads the few needed columns.
        """
        nodes_path = os.path.join(self.cfg.graph_dir, name, "graph_nodes.csv")
        cols = ["ID_resSeq", "sas_area"] + self.setup.trigger_columns
        try:
            df = pd.read_csv(nodes_path, usecols=cols)
        except (ValueError, OSError):
            return False  # missing column / unreadable -> won't load anyway
        agg = {"sas_area": "sum", **{c: "max" for c in self.setup.trigger_columns}}
        res = df.groupby("ID_resSeq").agg(agg)
        res = res[res["sas_area"] > 0]
        if len(res) == 0:
            return False
        return bool((res[self.setup.trigger_columns].to_numpy() > 0).any())

    def _filter_by_binding(self) -> None:
        self.qualifying = [
            name for name in tqdm(self.available, desc=f"Filter[{self.setup.name}]")
            if self._protein_qualifies(name)
        ]
        print(f"Qualifying for setup '{self.setup.name}': "
              f"{len(self.qualifying)}/{len(self.available)} chains "
              f"(>= 1 binding residue in {self.setup.trigger_columns})")
        if not self.qualifying:
            raise RuntimeError(f"No proteins qualify for setup '{self.setup.name}'.")

    def _split(self) -> None:
        s = self.cfg.train_val_test_split
        train_val, self.test_proteins = train_test_split(
            self.qualifying, test_size=s[2], random_state=self.cfg.seed)
        self.train_proteins, self.val_proteins = train_test_split(
            train_val, test_size=s[1] / (s[0] + s[1]), random_state=self.cfg.seed)
        print(f"Split: {len(self.train_proteins)} train / "
              f"{len(self.val_proteins)} val / {len(self.test_proteins)} test")

    def _build_weights(self) -> None:
        self.weight_map = build_chain_weights(
            self.cfg.csv_path, self.train_proteins, self.reweight)
        if self.weight_map:
            ws = np.array(list(self.weight_map.values()))
            print(f"Reweight='{self.reweight}': {len(self.weight_map)}/"
                  f"{len(self.train_proteins)} train chains mapped "
                  f"(mean={ws.mean():.3f} min={ws.min():.4f} max={ws.max():.3f})")
        else:
            print(f"Reweight='{self.reweight}': uniform (no per-sample weighting)")

    # --------------------------------------------------------------- loaders
    def _split_names(self, split: str) -> List[str]:
        return {
            "train": self.train_proteins,
            "val": self.val_proteins,
            "test": self.test_proteins,
        }[split]

    def load_mlp(self, split: str) -> List[Tuple[torch.Tensor, torch.Tensor, dict]]:
        """ESM-embedding tensors + setup labels: list of (emb, y_k, meta)."""
        out = []
        for name in tqdm(self._split_names(split), desc=f"MLP[{split}]"):
            r = tm.load_protein_mlp(name, self.cfg.graph_dir, self.cfg.emb_dir)
            if r is None:
                continue
            emb, y4, meta = r
            yk = self.setup.derive_labels(y4)
            if yk.sum() == 0:
                continue
            out.append((emb, yk, meta))
        return out

    def load_structural_graphs(self, split: str, with_weights: bool) -> list:
        """Structural-only PyG graphs (5D node features) for the pure GNN."""
        out = []
        for name in tqdm(self._split_names(split), desc=f"GNN[{split}]"):
            residue_df, edges_df, _ = tm.load_protein_base(name, self.cfg.graph_dir)
            if residue_df is None:
                continue
            g = tm.create_pyg_graph(residue_df, edges_df)  # 5D structural, 4-label y
            g.y = self.setup.derive_labels(g.y)
            if g.y.sum() == 0:
                continue
            g.protein_name = name
            if with_weights and self.weight_map:
                w = float(self.weight_map.get(name, 1.0))
                g.node_weight = torch.full((g.num_nodes,), w, dtype=torch.float32)
            out.append(g)
        return out

    def load_gnn_mlp_graphs(self, split: str, mlp_model, with_weights: bool) -> list:
        """Graphs with structural + MLP-embedding features (input_dim 69)."""
        cache, _ = tm.load_proteins_gnn_mlp(
            self._split_names(split), self.cfg.graph_dir, mlp_model, self.cfg.emb_dir,
            desc=f"GNN+MLP[{split}]", attach_esm=False,
            weight_map=self.weight_map if with_weights else None)
        graphs = []
        for g in cache.values():
            g.y = self.setup.derive_labels(g.y)
            if g.y.sum() == 0:
                continue
            graphs.append(g)
        return graphs

    def load_joint_graphs(self, split: str, with_weights: bool) -> list:
        """Graphs with structural (5D) node features + raw ESM attached.

        Built by reusing ``load_proteins_gnn_mlp`` with a throwaway MLP (only used
        to attach the ESM tensor + node count); its MLP-embedding columns are then
        sliced away, leaving 5D structural features. The Joint model trains its own
        MLP branch on the attached ESM embeddings.
        """
        dummy_mlp = tm.BindingSiteMLP(
            input_dim=1280, hidden_dims=[64],
            output_dim=self.setup.num_labels, dropout=0.0).to(tm.device)
        cache, _ = tm.load_proteins_gnn_mlp(
            self._split_names(split), self.cfg.graph_dir, dummy_mlp, self.cfg.emb_dir,
            desc=f"Joint[{split}]", attach_esm=True,
            weight_map=self.weight_map if with_weights else None)
        graphs = []
        for g in cache.values():
            g.x = g.x[:, :5]  # keep structural; discard dummy MLP embeddings
            g.y = self.setup.derive_labels(g.y)
            if g.y.sum() == 0:
                continue
            graphs.append(g)
        return graphs

    # ----------------------------------------------------------- criterion
    def make_criterion(self, train_labels: torch.Tensor) -> torch.nn.Module:
        """Focal loss with per-label pos_weight computed over the given labels."""
        n = len(train_labels)
        pos_weights = []
        print(f"Training label distribution (per residue, n={n}):")
        for i, name in enumerate(self.setup.label_names):
            n_pos = float(train_labels[:, i].sum())
            n_neg = n - n_pos
            pos_weights.append(n_neg / max(1.0, n_pos))
            print(f"  {name}: {int(n_pos)} ({100 * n_pos / max(1, n):.2f}%)")
        pw = torch.tensor(pos_weights, dtype=torch.float32).to(tm.device)
        print(f"pos_weights: {[f'{w:.1f}' for w in pos_weights]}")
        return tm.MultilabelFocalLoss(pos_weight=pw, gamma=self.cfg.focal_gamma)
