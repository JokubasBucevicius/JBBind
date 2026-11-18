"""Final evaluation of every trained model on the held-out test set.

Loads the checkpoints under ``runs/<setup>/<arch>/model.pt`` and scores them on
chains that were never part of training -- ``all_class/test_graphs/...`` -- using
the *same* benchmarks the training runs reported (``benchmarks.Evaluator``, i.e.
per-label F1 / Precision / Recall / Balanced-Accuracy / ROC-AUC / PR-AUC, their
macro averages, Hamming Loss and Exact Match, all at threshold 0.5).

Output mirrors the ``runs/`` layout exactly -- ``<out>/<setup>/<arch>/`` with
``metrics.json``, ``predictions.npz`` and the ROC / PR / confusion plots -- so
``aggregate.py`` and ``three_class_test.py`` run against it unchanged with
``--runs-dir``. ``final_report.py`` then collects everything into one place.

Per setup, the qualifying-chain filter from training is re-applied: a chain is
scored only if it has at least one surface binding residue of that setup's type.

Chain names being disjoint is not enough to call the set held-out: PPI3D is dense
with near-duplicate and homologous structures, so a test chain under one PDB entry
can be the same protein as a training chain under another. ``--cluster-level``
(default 40) drops every test chain whose PPI3D sequence cluster appears anywhere
in the training csv, which is what makes the numbers a generalisation estimate
rather than a partly-memorised one.

One caveat, unavoidable from the saved artefacts: the GNN+MLP architecture feeds
its GNN a 64-d embedding from a prerequisite MLP that ``main.py`` trained but
never checkpointed. The standalone ``mlp`` run of the same setup is the same
architecture trained on the same chains, split, criterion and seed, so it is used
as the feature extractor (override with ``--gnn-mlp-feature-model``). Treat the
GNN+MLP numbers as a close reproduction rather than a bit-exact replay.

Usage (from the repository root)::

    python all_class/training/program/final_test.py --all
    python all_class/training/program/final_test.py --setup dna_rna --arch gnn
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

# config inserts the repo root on sys.path so ``train_multilabel`` is importable.
from config import REPO_ROOT, Config

import train_multilabel as tm

import testset
from benchmarks import Evaluator
from setups import SETUPS, LabelSetup

ARCH_ORDER = ["mlp", "gnn", "gnn_mlp", "joint"]
DISPLAY = {"mlp": "MLP", "gnn": "GNN", "gnn_mlp": "GNN+MLP", "joint": "Joint"}

DEFAULT_TEST_GRAPHS = "all_class/test_graphs/test_graphs"
DEFAULT_OUT = "all_class/training/test_results"
DEFAULT_CACHE = "all_class/training/test_cache"

CLUSTER_LEVELS = ("95", "70", "40")
DEFAULT_TRAIN_CSV = "all_class/ppi3d_filtered.csv"
DEFAULT_TEST_CSV = "all_class/ppi3d_test_data_filtered.csv"


# ------------------------------------------------------------- cluster filtering
def _chain_clusters(csv_path: str, level: str) -> pd.DataFrame:
    """``(chain, cluster)`` pairs from *both* subunit sides of a PPI3D csv.

    The unified graph builder emits a directory per protein chain from either
    ``subunit_1`` or ``subunit_2``, so a chain's cluster id can live in the
    ``s1_*`` columns of one row and the ``s2_*`` columns of another. Reading one
    side would leave chains unmapped and silently un-filterable.
    """
    cols = ["subunit_1", "subunit_2",
            f"s1_sequence_cluster_{level}", f"s2_sequence_cluster_{level}"]
    df = pd.read_csv(resolve(csv_path), usecols=cols, dtype=str)
    sides = []
    for side in (1, 2):
        part = df[[f"subunit_{side}", f"s{side}_sequence_cluster_{level}"]].copy()
        part.columns = ["chain", "cluster"]
        sides.append(part)
    out = pd.concat(sides, ignore_index=True).dropna()
    return out[(out["chain"] != "") & (out["cluster"] != "")].drop_duplicates()


def filter_by_cluster(names: List[str], level: str,
                      train_csv: str = DEFAULT_TRAIN_CSV,
                      test_csv: str = DEFAULT_TEST_CSV,
                      keep_unmapped: bool = False,
                      verbose: bool = True) -> List[str]:
    """Test chains whose sequence cluster never occurs in the training data.

    A chain is dropped if *any* of its cluster ids at ``level`` appears in the
    training csv -- chains can carry more than one id across rows when they show
    up in several complexes. Chains the test csv gives no cluster id for cannot
    be cleared, so they are dropped too unless ``keep_unmapped``.
    """
    if level in (None, "none"):
        if verbose:
            print("Cluster filter: disabled — scoring the full test set")
        return list(names)

    train = set(_chain_clusters(train_csv, level)["cluster"])
    pairs = _chain_clusters(test_csv, level)
    pairs = pairs[pairs["chain"].isin(set(names))]

    cluster_of: Dict[str, List[str]] = {}
    for chain, cluster in zip(pairs["chain"], pairs["cluster"]):
        cluster_of.setdefault(chain, []).append(cluster)

    drop = {c for c, cl in cluster_of.items() if any(x in train for x in cl)}
    unmapped = set(names) - set(cluster_of)
    removed = drop if keep_unmapped else drop | unmapped
    kept = [n for n in names if n not in removed]

    if verbose:
        n = max(1, len(names))
        print(f"Cluster filter: sequence_cluster_{level} vs {train_csv}")
        print(f"  input chains              {len(names):>7,}")
        print(f"  share a training cluster  {len(drop):>7,} "
              f"({100 * len(drop) / n:.1f}%)  -> dropped")
        if unmapped:
            print(f"  no cluster id in test csv {len(unmapped):>7,} "
                  f"({100 * len(unmapped) / n:.1f}%)  -> "
                  f"{'kept' if keep_unmapped else 'dropped'}")
        print(f"  non-redundant chains      {len(kept):>7,} "
              f"({100 * len(kept) / n:.1f}%)")
    return kept


def add_cluster_arguments(p: argparse.ArgumentParser) -> None:
    """Shared CLI surface so every entry point filters identically."""
    p.add_argument("--cluster-level", choices=("none",) + CLUSTER_LEVELS, default="40",
                   help="drop test chains sharing a training sequence cluster at "
                        "this identity level (default: 40; 'none' disables)")
    p.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    p.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    p.add_argument("--keep-unmapped", action="store_true",
                   help="keep test chains carrying no cluster id (default: drop)")


# --------------------------------------------------------------- model loading
def _gnn_kwargs(sd: Dict[str, torch.Tensor], prefix: str = "") -> dict:
    """Recover ``BindingSiteGNN`` geometry from a state dict."""
    aa = sd[f"{prefix}aa_embedding.weight"]
    att = sd[f"{prefix}conv1.att"]                    # (1, heads, hidden)
    num_aa, aa_embed_dim = aa.shape
    heads, hidden = att.shape[1], att.shape[2]
    total_in = sd[f"{prefix}conv1.lin_l.weight"].shape[1]
    return {
        "input_dim": total_in - aa_embed_dim,
        "hidden_dim": hidden,
        "heads": heads,
        "num_amino_acids": num_aa,
        "aa_embed_dim": aa_embed_dim,
        "output_dim": sd[f"{prefix}fc.weight"].shape[0],
    }


def _mlp_hidden_dims(sd: Dict[str, torch.Tensor], prefix: str) -> List[int]:
    dims, i = [], 0
    while f"{prefix}{i}.weight" in sd:
        dims.append(sd[f"{prefix}{i}.weight"].shape[0])
        i += 4                                        # Linear, ReLU, Dropout, LayerNorm
    return dims


def load_model(arch: str, path: str) -> Tuple[torch.nn.Module, dict]:
    """Rebuild a checkpointed model. Geometry comes from the weights themselves,
    so a config drift in ``architectures.py`` cannot silently mis-load a run."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    if arch == "mlp":
        cfg = ckpt.get("model_config") or {
            "input_dim": sd["network.0.weight"].shape[1],
            "hidden_dims": _mlp_hidden_dims(sd, "network."),
            "output_dim": sd[f"network.{4 * len(_mlp_hidden_dims(sd, 'network.'))}.weight"].shape[0],
        }
        model = tm.BindingSiteMLP(input_dim=cfg["input_dim"], hidden_dims=cfg["hidden_dims"],
                                  output_dim=cfg["output_dim"], dropout=0.0)
    elif arch in ("gnn", "gnn_mlp"):
        model = tm.BindingSiteGNN(dropout=0.0, **_gnn_kwargs(sd))
    elif arch == "joint":
        hidden = _mlp_hidden_dims(sd, "mlp_layers.")
        mlp_cfg = {"input_dim": sd["mlp_layers.0.weight"].shape[1],
                   "hidden_dims": hidden,
                   "output_dim": sd["mlp_head.weight"].shape[0],
                   "dropout": 0.0}
        gnn_kw = _gnn_kwargs(sd, prefix="gnn.")
        gnn_cfg = {"input_dim": gnn_kw["input_dim"], "hidden_dim": gnn_kw["hidden_dim"],
                   "heads": gnn_kw["heads"], "dropout": 0.0}
        model = tm.JointMLPGNN(mlp_cfg, gnn_cfg,
                               mlp_loss_weight=ckpt.get("mlp_loss_weight", 0.5))
    else:
        raise ValueError(f"unknown arch: {arch}")

    model.load_state_dict(sd)                          # strict: geometry must match
    return model.to(tm.device).eval(), ckpt


def load_feature_mlp(runs_dir: str, setup: str, override: Optional[str]) -> torch.nn.Module:
    """The MLP whose penultimate layer supplies GNN+MLP's 64-d node features."""
    path = override or os.path.join(runs_dir, setup, "mlp", "model.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"GNN+MLP needs a feature-extractor MLP; none at {path}. "
            f"Pass --gnn-mlp-feature-model.")
    model, _ = load_model("mlp", path)
    print(f"  GNN+MLP node features from {path} "
          f"(stand-in for the unsaved prerequisite MLP)")
    return model


# ------------------------------------------------------------------ prediction
def _predict_mlp(model, data) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    with torch.no_grad():
        for emb, y in data:
            ps.append(torch.sigmoid(model(emb.to(tm.device))).cpu().numpy())
            ys.append(y.numpy())
    return ys, ps


def _predict_graphs(model, graphs, batch_size, joint: bool) -> Tuple[list, list]:
    ys, ps = [], []
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[i:i + batch_size]).to(tm.device)
            if joint:
                logits, _ = model(batch.esm_embeddings.to(tm.device), batch)
            else:
                logits = model(batch)
            ys.append(batch.y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    return ys, ps


def evaluate_arch(arch: str, setup: LabelSetup, names: List[str], ts: testset.TestSet,
                  runs_dir: str, out_dir: str, cfg: Config,
                  feature_mlp_override: Optional[str], chain_chunk: int,
                  cluster_level: str = "none") -> Optional[dict]:
    """Score one (setup, arch) over the held-out chains and write its run dir."""
    ckpt_path = os.path.join(runs_dir, setup.name, arch, "model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"  [{arch}] skipped: no checkpoint at {ckpt_path}")
        return None

    t0 = time.time()
    model, ckpt = load_model(arch, ckpt_path)
    if list(ckpt.get("label_names", setup.label_names)) != list(setup.label_names):
        print(f"  [{arch}] skipped: checkpoint labels {ckpt.get('label_names')} "
              f"!= setup labels {setup.label_names}")
        return None

    feature_mlp = (load_feature_mlp(runs_dir, setup.name, feature_mlp_override)
                   if arch == "gnn_mlp" else None)

    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    n_chains = 0
    for block in tqdm(list(testset.chunks(names, chain_chunk)),
                      desc=f"{setup.name}/{arch}"):
        if arch == "mlp":
            data = ts.load_mlp(block, setup)
            if not data:
                continue
            n_chains += len(data)
            by, bp = _predict_mlp(model, data)
        else:
            if arch == "gnn":
                graphs = ts.load_gnn(block, setup)
            elif arch == "gnn_mlp":
                graphs = ts.load_esm_gnn(block, setup, "gnn_mlp", mlp_model=feature_mlp)
            else:
                graphs = ts.load_esm_gnn(block, setup, "joint")
            if not graphs:
                continue
            n_chains += len(graphs)
            by, bp = _predict_graphs(model, graphs, cfg.gnn_batch_size, joint=(arch == "joint"))
        ys.extend(by)
        ps.extend(bp)

    if not ys:
        print(f"  [{arch}] no scorable residues; skipped")
        return None

    y_true = np.vstack(ys)
    y_prob = np.vstack(ps)

    run_dir = os.path.join(out_dir, setup.name, arch)
    ev = Evaluator(setup, arch, DISPLAY[arch], run_dir)
    ev.evaluate(y_true, y_prob)
    ev.print_summary()
    ev.save()

    # n_test_chains is ours; the rest of metrics.json matches the training runs.
    meta_path = os.path.join(run_dir, "metrics.json")
    with open(meta_path) as fh:
        payload = json.load(fh)
    payload["n_test_chains"] = n_chains
    payload["split"] = "held_out_test"
    payload["cluster_filter"] = cluster_level
    payload["checkpoint"] = ckpt_path
    if arch == "gnn_mlp":
        payload["feature_mlp"] = (feature_mlp_override
                                  or os.path.join(runs_dir, setup.name, "mlp", "model.pt"))
    with open(meta_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  [{arch}] {n_chains} chains / {len(y_true):,} residues "
          f"in {time.time() - t0:.0f}s")
    return payload


# ------------------------------------------------------------------------ main
def resolve(path: str) -> str:
    """Accept an input path given relative to the repo root from any CWD."""
    if os.path.exists(path):
        return path
    from_root = os.path.join(REPO_ROOT, path)
    return from_root if os.path.exists(from_root) else path


def resolve_out(path: str) -> str:
    """Anchor an output path at the repo root unless it is absolute."""
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(REPO_ROOT, path))


def discover_chains(graph_dir: str, emb_dir: str, exclude_dir: Optional[str]) -> List[str]:
    """Chains with both a graph and an ESM embedding, minus anything trained on."""
    graphs = {d for d in os.listdir(graph_dir)
              if os.path.isfile(os.path.join(graph_dir, d, "graph_nodes.csv"))}
    embs = {f[:-3] for f in os.listdir(emb_dir) if f.endswith(".pt")}
    names = graphs & embs
    print(f"Test chains with graphs: {len(graphs)}")
    print(f"  ... also with embeddings: {len(names)}")
    if exclude_dir and os.path.isdir(exclude_dir):
        trained = set(os.listdir(exclude_dir))
        leaked = names & trained
        if leaked:
            print(f"  ... dropping {len(leaked)} chain(s) also present in the "
                  f"training graph dir: {sorted(leaked)[:10]}")
        names -= trained
    return sorted(names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score every trained model on the held-out test set")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--setup", choices=list(SETUPS))
    g.add_argument("--all", action="store_true", help="every setup")
    p.add_argument("--arch", choices=ARCH_ORDER, action="append",
                   help="restrict to these architectures (repeatable)")
    p.add_argument("--graph-dir", default=DEFAULT_TEST_GRAPHS)
    p.add_argument("--emb-dir", default=None, help="default: Config().emb_dir")
    p.add_argument("--runs-dir", default=None, help="where the checkpoints live")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--exclude-graph-dir", default=None,
                   help="training graph dir; its chains are excluded (default: Config().graph_dir)")
    p.add_argument("--limit", type=int, default=None, help="debug: first N chains")
    p.add_argument("--chain-chunk", type=int, default=256,
                   help="chains loaded into memory at a time (default: 256)")
    p.add_argument("--cache-workers", type=int, default=16)
    p.add_argument("--cache-only", action="store_true", help="build the cache and stop")
    p.add_argument("--gnn-mlp-feature-model", default=None,
                   help="MLP checkpoint supplying GNN+MLP node features")
    add_cluster_arguments(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    graph_dir = resolve(args.graph_dir)
    emb_dir = resolve(args.emb_dir or cfg.emb_dir)
    runs_dir = resolve(args.runs_dir or cfg.runs_dir)
    exclude_dir = resolve(args.exclude_graph_dir or cfg.graph_dir)
    out_dir = resolve_out(args.out_dir)
    cache_dir = resolve_out(args.cache_dir)

    print("=" * 70)
    print("FINAL TEST — held-out chains")
    print(f"  graphs : {graph_dir}")
    print(f"  embeds : {emb_dir}")
    print(f"  models : {runs_dir}")
    print(f"  output : {out_dir}")
    print(f"  cache  : {cache_dir}")
    print(f"  filter : sequence_cluster_{args.cluster_level}"
          if args.cluster_level != "none" else "  filter : none")
    print(f"  device : {tm.device}")
    print("=" * 70)

    names = discover_chains(graph_dir, emb_dir, exclude_dir)
    # Homology, not just name collisions: graphs and embeddings already exist for
    # every chain, so this only narrows what gets scored.
    names = filter_by_cluster(names, args.cluster_level, args.train_csv,
                              args.test_csv, args.keep_unmapped)
    if args.limit:
        names = names[:args.limit]
    print(f"Evaluating on {len(names)} chains")

    names = testset.build_cache(names, graph_dir, cache_dir, workers=args.cache_workers)
    if args.cache_only:
        return

    ts = testset.TestSet(cache_dir, emb_dir, names)
    archs = args.arch or ARCH_ORDER
    setup_names = list(SETUPS) if args.all else [args.setup]

    written = []
    for sname in setup_names:
        setup = SETUPS[sname]
        setup.activate()
        print("\n" + "=" * 70)
        print(f"SETUP {sname}  labels={setup.label_names}")
        print("=" * 70)
        qualifying = ts.qualifying(setup)
        if not qualifying:
            print("  no qualifying chains; skipping setup")
            continue
        # Logged so the held-out class balance is on the record next to the
        # training runs' own distribution block.
        dist = ts.label_distribution(setup, qualifying)
        ts.print_distribution(setup, dist)
        for arch in archs:
            setup.activate()   # Evaluator/tm globals are per-setup
            payload = evaluate_arch(arch, setup, qualifying, ts, runs_dir, out_dir,
                                    cfg, args.gnn_mlp_feature_model, args.chain_chunk,
                                    args.cluster_level)
            if payload:
                written.append((sname, arch))

    print("\n" + "=" * 70)
    print(f"Wrote {len(written)} run dir(s) under {out_dir}")
    for sname, arch in written:
        print(f"  {sname}/{arch}")
    print("\nNext: python all_class/training/program/final_report.py")


if __name__ == "__main__":
    main()
