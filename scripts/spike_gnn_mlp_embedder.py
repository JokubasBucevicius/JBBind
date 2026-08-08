#!/usr/bin/env python3
"""P0.5 spike: is ``runs/<setup>/mlp/model.pt`` the embedder that ``gnn_mlp`` needs?

``GNNMLPArch.prepare()`` (all_class/training/program/architectures.py:157-180) trains a
prerequisite MLP whose 64-d penultimate activations become 64 of the GNN's 69 input
features -- then ``save_checkpoint`` (:71) saves only the GNN. The embedder is gone, so
the 5 ``gnn_mlp`` checkpoints cannot be loaded as-is.

Hypothesis: the separately-saved ``runs/<setup>/mlp/model.pt`` IS that embedder, because
``main.py:60-62`` seeds torch/numpy/random identically in both jobs and nothing between
seeding and ``trainer.train_mlp`` consumes the torch global RNG.

Decisive test: reconstruct the test split exactly as ``datamodule.py`` does, rebuild the
69-d graphs using the mlp checkpoint as embedder, run the gnn_mlp checkpoint, and compare
against the recorded ``predictions.npz``. ``y_true`` matching first proves the ordering
reconstruction; then ``y_prob`` matching proves the substitution.

Run from the research repo root:
    cd /home/jokubasb/protein_protein
    python /home/jokubasb/JBBind/scripts/spike_gnn_mlp_embedder.py --setup dna_rna
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

REPO = "/home/jokubasb/protein_protein"
PROGRAM = os.path.join(REPO, "all_class", "training", "program")
for p in (REPO, PROGRAM, os.path.join(REPO, "all_class", "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_multilabel as tm  # noqa: E402
from setups import SETUPS  # noqa: E402

GRAPH_DIR = os.path.join(REPO, "all_class", "graphs", "unified", "unified")
EMB_DIR = os.path.join(REPO, "emb")
RUNS_DIR = os.path.join(REPO, "all_class", "training", "runs")
CACHE = "/tmp/claude-503000028/-home-jokubasb/465a4035-0d13-4c9b-9c02-260b85ea9158/scratchpad"

_TRIGGER: list[str] = []


def _qualifies(name: str) -> bool:
    """Mirror ProteinDataModule._protein_qualifies (datamodule.py:78-95)."""
    nodes_path = os.path.join(GRAPH_DIR, name, "graph_nodes.csv")
    cols = ["ID_resSeq", "sas_area"] + _TRIGGER
    try:
        df = pd.read_csv(nodes_path, usecols=cols)
    except (ValueError, OSError):
        return False
    agg = {"sas_area": "sum", **{c: "max" for c in _TRIGGER}}
    res = df.groupby("ID_resSeq").agg(agg)
    res = res[res["sas_area"] > 0]
    if len(res) == 0:
        return False
    return bool((res[_TRIGGER].to_numpy() > 0).any())


def _init(trigger):
    global _TRIGGER
    _TRIGGER = trigger


def discover() -> list[str]:
    """Mirror ProteinDataModule._discover (datamodule.py:56-74)."""
    graph_proteins = {
        d for d in os.listdir(GRAPH_DIR)
        if os.path.isfile(os.path.join(GRAPH_DIR, d, "graph_nodes.csv"))
    }
    emb_proteins = {f[:-3] for f in os.listdir(EMB_DIR) if f.endswith(".pt")}
    available = sorted(graph_proteins & emb_proteins)
    print(f"graphs={len(graph_proteins)}  embeddings={len(emb_proteins)}  "
          f"available={len(available)}")
    return available


def qualifying_for(setup, available: list[str], workers: int) -> list[str]:
    cache_path = os.path.join(CACHE, f"qualifying_{setup.name}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            q = json.load(f)
        print(f"qualifying[{setup.name}] = {len(q)} (cached)")
        return q
    trigger = setup.trigger_columns
    print(f"prescan for {setup.name} over {len(available)} chains "
          f"(trigger={trigger}, workers={workers})...")
    with Pool(workers, initializer=_init, initargs=(trigger,)) as pool:
        flags = pool.map(_qualifies, available, chunksize=64)
    q = [n for n, ok in zip(available, flags) if ok]
    print(f"qualifying[{setup.name}] = {len(q)}/{len(available)}")
    os.makedirs(CACHE, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(q, f)
    return q


def split(qualifying: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Mirror ProteinDataModule._split (datamodule.py:108-113), split [0.7, 0.1, 0.2]."""
    s = [0.7, 0.1, 0.2]
    train_val, test = train_test_split(qualifying, test_size=s[2], random_state=42)
    train, val = train_test_split(train_val, test_size=s[1] / (s[0] + s[1]), random_state=42)
    print(f"split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


def build_embedder(setup, device):
    """Load runs/<setup>/mlp/model.pt as the candidate 64-d embedder."""
    path = os.path.join(RUNS_DIR, setup.name, "mlp", "model.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["model_config"]
    model = tm.BindingSiteMLP(**cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"embedder: {path}  config={cfg}")
    return model


def build_gnn(setup, device):
    """Reconstruct GNNMLPArch.build_model (architectures.py:182-185)."""
    path = os.path.join(RUNS_DIR, setup.name, "gnn_mlp", "model.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = tm.BindingSiteGNN(input_dim=69, hidden_dim=512, heads=4, dropout=0.5,
                              output_dim=setup.num_labels).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"gnn_mlp:  {path}  labels={ck['label_names']}")
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="dna_rna", choices=list(SETUPS))
    ap.add_argument("--workers", type=int, default=36)
    args = ap.parse_args()

    setup = SETUPS[args.setup]
    setup.activate()  # point tm's label globals at this setup
    device = tm.device
    print(f"=== P0.5 spike: setup={setup.name} labels={setup.label_names} device={device}")

    available = discover()
    qualifying = qualifying_for(setup, available, args.workers)
    _, _, test = split(qualifying)

    ref = np.load(os.path.join(RUNS_DIR, setup.name, "gnn_mlp", "predictions.npz"))
    y_true_ref, y_prob_ref = ref["y_true"], ref["y_prob"]
    print(f"reference predictions.npz: y_true={y_true_ref.shape} y_prob={y_prob_ref.shape}")

    embedder = build_embedder(setup, device)
    gnn = build_gnn(setup, device)

    # Mirror ProteinDataModule.load_gnn_mlp_graphs (datamodule.py:168-180).
    cache, failed = tm.load_proteins_gnn_mlp(
        test, GRAPH_DIR, embedder, EMB_DIR, desc=f"GNN+MLP[test/{setup.name}]",
        attach_esm=False, weight_map=None)
    graphs = []
    for g in cache.values():
        g.y = setup.derive_labels(g.y)
        if g.y.sum() == 0:
            continue
        graphs.append(g)
    print(f"loaded {len(graphs)} test graphs ({len(failed)} failed to load)")

    # Mirror architectures._predict_graphs (:30-42).
    from torch_geometric.data import Batch
    ys, ps = [], []
    with torch.no_grad():
        for i in range(0, len(graphs), 32):
            batch = Batch.from_data_list(graphs[i:i + 32]).to(device)
            logits = gnn(batch)
            ys.append(batch.y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.vstack(ys)
    y_prob = np.vstack(ps)
    print(f"reproduced:                y_true={y_true.shape} y_prob={y_prob.shape}")

    print("\n--- STEP 1: does the reconstructed ordering match? ---")
    if y_true.shape != y_true_ref.shape:
        print(f"FAIL shape mismatch {y_true.shape} vs {y_true_ref.shape}")
        return 2
    if not np.array_equal(y_true, y_true_ref):
        n = int((y_true != y_true_ref).sum())
        print(f"FAIL y_true differs in {n}/{y_true.size} entries -> ordering not reproduced")
        return 2
    print(f"PASS y_true identical ({y_true.shape[0]:,} residues) -> ordering proven")

    print("\n--- STEP 2: is runs/<setup>/mlp/model.pt the gnn_mlp embedder? ---")
    diff = np.abs(y_prob - y_prob_ref)
    print(f"max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}  "
          f"corr={np.corrcoef(y_prob.ravel(), y_prob_ref.ravel())[0,1]:.6f}")
    if np.allclose(y_prob, y_prob_ref, atol=1e-5, rtol=0):
        print("PASS -> the mlp checkpoint IS the discarded flat_mlp. gnn_mlp is deployable.")
        return 0
    print("FAIL -> the mlp checkpoint is a DIFFERENT model. Fall back to `joint`, or "
          "patch architectures.py:71 to save flat_mlp and re-run the 5 gnn_mlp jobs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
