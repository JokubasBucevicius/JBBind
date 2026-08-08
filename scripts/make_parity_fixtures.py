#!/usr/bin/env python3
"""Generate golden parity fixtures from the research repo.

Runs in the RESEARCH environment (needs train_multilabel + the 802 GB data tree) and emits
small fixtures that let tests/ verify, with no access to that tree, that JBBind's copied
code produces bit-identical tensors and model outputs.

Three chains, chosen to span the failure modes:
  12as_A   clean; 327 observed residues, 330 SEQRES
  6oax_E   579 observed of 867 SEQRES -- heavy gaps, where rank-based ESM indexing
           (instead of SEQRES-index) would be catastrophically wrong
  3cqz_A   1733 SEQRES -> 1022 ESM embedding -- exercises the truncation path

For each chain it writes the voronota CSVs (the pipeline inputs), the layer-33 ESM slice,
and the reference tensors/probabilities computed by the ORIGINAL code.

Usage (from the research repo root):
    cd /home/jokubasb/protein_protein
    python /home/jokubasb/JBBind/scripts/make_parity_fixtures.py
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "/home/jokubasb/protein_protein"
PROGRAM = os.path.join(REPO, "all_class", "training", "program")
for p in (REPO, PROGRAM, os.path.join(REPO, "all_class", "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_multilabel as tm  # noqa: E402

GRAPH_DIR = os.path.join(REPO, "all_class", "graphs", "unified", "unified")
EMB_DIR = os.path.join(REPO, "emb")
MODELS = Path(__file__).resolve().parents[1] / "models"
OUT = Path(__file__).resolve().parents[1] / "tests" / "data"

CHAINS = ["12as_A", "6oax_E", "3cqz_A"]
SETUPS = ["protein_nucleic", "homo_hetero", "protein", "dna_rna", "nucleic"]
ARCHS = ["mlp", "gnn", "gnn_mlp", "joint"]
NUM_LABELS = {"protein_nucleic": 2, "homo_hetero": 2, "protein": 1, "dna_rna": 2, "nucleic": 1}


def esm_layer33(chain: str) -> torch.Tensor:
    d = torch.load(os.path.join(EMB_DIR, f"{chain}.pt"), map_location="cpu",
                   weights_only=False)
    layer_keys = sorted(d["representations"].keys())
    return d["representations"][layer_keys[-1]]


def reference_graph(chain: str):
    """Reproduce the original loader exactly: load_protein_base + the ESM indexing of
    _load_protein_gnn_mlp_base (:326-344) + create_pyg_graph."""
    residue_df, edges_df, _ = tm.load_protein_base(chain, GRAPH_DIR)
    assert residue_df is not None, f"{chain}: load_protein_base returned None"

    embeddings = esm_layer33(chain)
    emb_size = embeddings.shape[0]
    id_res_seq = residue_df["ID_resSeq"].values
    min_res_seq = id_res_seq.min()
    embedding_indices = id_res_seq - 1 if min_res_seq >= 1 else id_res_seq

    valid_mask = (embedding_indices >= 0) & (embedding_indices < emb_size)
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        embedding_indices = embedding_indices[valid_mask]
        residue_df = residue_df[valid_mask].reset_index(drop=True)
        if edges_df is not None:
            surface_ids = set(residue_df["ID_resSeq"])
            edge_mask = (edges_df["ID1_resSeq"].isin(surface_ids) &
                         edges_df["ID2_resSeq"].isin(surface_ids))
            edges_df = edges_df[edge_mask].reset_index(drop=True)

    esm = embeddings[embedding_indices, :].clone()
    graph = tm.create_pyg_graph(residue_df, edges_df)  # 5-d structural, 4-label y
    return graph, esm, residue_df, n_dropped, emb_size


def build_reference_model(arch: str, setup: str, k: int):
    """Mirror architectures.py construction, using the ORIGINAL tm classes."""
    ck = torch.load(MODELS / setup / arch / "model.pt", map_location="cpu",
                    weights_only=False)
    if arch == "mlp":
        model = tm.BindingSiteMLP(**ck["model_config"])
    elif arch == "gnn":
        model = tm.BindingSiteGNN(input_dim=5, hidden_dim=256, heads=4, dropout=0.2,
                                  output_dim=k)
    elif arch == "gnn_mlp":
        model = tm.BindingSiteGNN(input_dim=69, hidden_dim=512, heads=4, dropout=0.5,
                                  output_dim=k)
    elif arch == "joint":
        model = tm.JointMLPGNN(
            {"input_dim": 1280, "hidden_dims": [512, 128, 64], "output_dim": k,
             "dropout": 0.4},
            {"input_dim": 69, "hidden_dim": 512, "heads": 4, "dropout": 0.3},
            mlp_loss_weight=float(ck.get("mlp_loss_weight", 0.5)))
    else:
        raise ValueError(arch)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    embedder = None
    if arch == "gnn_mlp":
        eck = torch.load(MODELS / setup / arch / "embedder.pt", map_location="cpu",
                         weights_only=False)
        embedder = tm.BindingSiteMLP(**eck["model_config"])
        embedder.load_state_dict(eck["model_state_dict"])
        embedder.eval()
    return model, embedder


@torch.no_grad()
def reference_probs(arch, model, embedder, graph, esm):
    from torch_geometric.data import Batch
    if arch == "mlp":
        return torch.sigmoid(model(esm))

    g = graph.clone()
    if arch == "gnn_mlp":
        x = esm
        for layer in list(embedder.network.children())[:-1]:
            x = layer(x)
        g.x = torch.cat([g.x, x], dim=-1)

    batch = Batch.from_data_list([g])
    if arch == "joint":
        logits, _ = model(esm, batch)
    else:
        logits = model(batch)
    return torch.sigmoid(logits)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}

    for chain in CHAINS:
        d = OUT / chain
        d.mkdir(exist_ok=True)
        print(f"\n=== {chain}")

        for fname in ("graph_nodes.csv", "graph_links.csv"):
            src = Path(GRAPH_DIR) / chain / fname
            with open(src, "rb") as fi, gzip.open(d / f"{fname}.gz", "wb", 6) as fo:
                shutil.copyfileobj(fi, fo)

        graph, esm, residue_df, n_dropped, emb_size = reference_graph(chain)
        print(f"  residues={graph.num_nodes}  edges={graph.edge_index.shape[1]}  "
              f"esm={tuple(esm.shape)}  emb_len={emb_size}  dropped={n_dropped}")

        np.savez_compressed(
            d / "reference_graph.npz",
            x=graph.x.numpy(),
            residue_type=graph.residue_type.numpy(),
            edge_index=graph.edge_index.numpy(),
            edge_attr=graph.edge_attr.numpy(),
            id_resseq=residue_df["ID_resSeq"].values.astype(np.int64),
        )
        np.save(d / "esm_layer33.npy", esm.numpy())

        probs = {}
        for setup in SETUPS:
            for arch in ARCHS:
                model, embedder = build_reference_model(arch, setup, NUM_LABELS[setup])
                p = reference_probs(arch, model, embedder, graph, esm)
                probs[f"{setup}/{arch}"] = p.numpy()
        np.savez_compressed(d / "reference_probs.npz", **probs)
        print(f"  reference probs for {len(probs)} setup/arch combinations")

        index[chain] = {
            "n_residues": int(graph.num_nodes),
            "n_edges": int(graph.edge_index.shape[1]),
            "esm_len": int(emb_size),
            "n_dropped_by_truncation": n_dropped,
            "min_resseq": int(residue_df["ID_resSeq"].min()),
            "max_resseq": int(residue_df["ID_resSeq"].max()),
        }

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\nfixtures: {total/1e6:.1f} MB in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
