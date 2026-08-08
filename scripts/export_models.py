#!/usr/bin/env python3
"""Export trained checkpoints + precomputed metrics from the research repo into models/.

Produces, for all 5 setups x 4 architectures:
  models/<setup>/<arch>/model.pt        the checkpoint, copied
  models/<setup>/gnn_mlp/embedder.pt    the paired 64-d embedder (see below)
  models/MANIFEST.json                  sha256, label_names, metrics, provenance
  models/METRICS.json                   dashboard data: metrics, downsampled ROC/PR
                                        curves, confusion counts, dataset composition

The gnn_mlp embedder: GNNMLPArch.prepare() (architectures.py:157-180) trains a prerequisite
MLP whose 64-d penultimate activations become 64 of the GNN's 69 input features, and
save_checkpoint (:71) discards it. scripts/spike_gnn_mlp_embedder.py proved the separately
saved runs/<setup>/mlp/model.pt is bit-equivalent to it (dna_rna reproduces the recorded
predictions.npz to 1.19e-07), so it is copied in as embedder.pt.

The curves are downsampled from predictions.npz (484 MB of npz across all runs) to ~200
points each so the whole dashboard is a single ~200 KB JSON and neither the npz files nor
the PNGs need to ship in the container image.

Usage:
    python scripts/export_models.py --runs /home/jokubasb/protein_protein/all_class/training/runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

REPO_DEFAULT = "/home/jokubasb/protein_protein"
SETUPS = ["protein_nucleic", "homo_hetero", "protein", "dna_rna", "nucleic"]
ARCHS = ["mlp", "gnn", "gnn_mlp", "joint"]
ARCH_DISPLAY = {"mlp": "MLP", "gnn": "GNN", "gnn_mlp": "GNN+MLP", "joint": "Joint"}
N_CURVE_POINTS = 200


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def downsample(xs: np.ndarray, ys: np.ndarray, n: int = N_CURVE_POINTS) -> list[list[float]]:
    """Keep ~n points, preserving the endpoints and the shape of the curve."""
    if len(xs) <= n:
        idx = np.arange(len(xs))
    else:
        # Even spacing in index space, endpoints forced in.
        idx = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return [[round(float(xs[i]), 5), round(float(ys[i]), 5)] for i in idx]


def curves_for(y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]) -> dict:
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    out = {}
    for i, name in enumerate(label_names):
        yt, yp = y_true[:, i], y_prob[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            out[name] = None
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        prec, rec, _ = precision_recall_curve(yt, yp)
        # precision_recall_curve returns descending recall; flip for plotting.
        out[name] = {
            "roc": downsample(fpr, tpr),
            "roc_auc": round(float(roc_auc_score(yt, yp)), 5),
            "pr": downsample(rec[::-1], prec[::-1]),
            "pr_auc": round(float(average_precision_score(yt, yp)), 5),
            "baseline": round(float(yt.mean()), 6),
            "n_positive": int(yt.sum()),
            "n_total": int(len(yt)),
        }
    return out


def confusion_at(y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str],
                 threshold: float = 0.5) -> dict:
    out = {}
    for i, name in enumerate(label_names):
        yt = y_true[:, i].astype(int)
        yp = (y_prob[:, i] > threshold).astype(int)
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
        out[name] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return out


def parse_results_txt(path: Path) -> dict:
    """Dataset composition per setup from runs/results.txt.

    Lines look like:
      setup=dna_rna  qualifying=1630/84646  train=1140 val=164 test=326  reweight=none
      train_residues=306427  DNA=25102(8.19%)  RNA=17530(5.72%)
    """
    import re
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    current = None
    for line in path.read_text().splitlines():
        m = re.search(r"setup=(\w+)\s+qualifying=(\d+)/(\d+)\s+train=(\d+)\s+val=(\d+)\s+test=(\d+)", line)
        if m:
            current = m.group(1)
            out[current] = {
                "qualifying": int(m.group(2)), "available": int(m.group(3)),
                "train_chains": int(m.group(4)), "val_chains": int(m.group(5)),
                "test_chains": int(m.group(6)), "class_balance": {},
            }
            continue
        m = re.search(r"train_residues=(\d+)", line)
        if m and current:
            out[current]["train_residues"] = int(m.group(1))
            for name, count, pct in re.findall(r"([A-Za-z][\w\- ]*?)=(\d+)\(([\d.]+)%\)", line):
                if name.strip() == "train_residues":
                    continue
                out[current]["class_balance"][name.strip()] = {
                    "count": int(count), "pct": float(pct)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--skip-curves", action="store_true",
                    help="Only copy checkpoints + metrics.json; skip the npz curve pass.")
    args = ap.parse_args()

    runs = Path(args.runs or Path(args.repo) / "all_class" / "training" / "runs")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict = {}
    metrics_doc: dict = {
        "generated_from": str(runs),
        "threshold": 0.5,
        "arch_display": ARCH_DISPLAY,
        "setups": {},
        "dataset": parse_results_txt(runs / "results.txt"),
        "three_class": {},
    }

    for setup in SETUPS:
        metrics_doc["setups"][setup] = {"archs": {}}
        for arch in ARCHS:
            src_dir = runs / setup / arch
            ckpt = src_dir / "model.pt"
            if not ckpt.exists():
                print(f"  skip {setup}/{arch}: no model.pt")
                continue

            dst_dir = out / setup / arch
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt, dst_dir / "model.pt")
            entry = {
                "setup": setup, "arch": arch,
                "source": str(ckpt),
                "sha256": sha256(dst_dir / "model.pt"),
                "size_bytes": (dst_dir / "model.pt").stat().st_size,
            }

            # gnn_mlp additionally needs the prerequisite MLP that generated its 64-d
            # node features; see the module docstring.
            if arch == "gnn_mlp":
                emb_src = runs / setup / "mlp" / "model.pt"
                if not emb_src.exists():
                    raise FileNotFoundError(
                        f"gnn_mlp needs {emb_src} as its embedder, but it is missing")
                shutil.copy2(emb_src, dst_dir / "embedder.pt")
                entry["embedder"] = {
                    "source": str(emb_src),
                    "sha256": sha256(dst_dir / "embedder.pt"),
                    "provenance": "runs/<setup>/mlp/model.pt; verified bit-equivalent to "
                                  "the discarded GNNMLPArch.flat_mlp by "
                                  "scripts/spike_gnn_mlp_embedder.py "
                                  "(dna_rna max|diff|=1.19e-07 vs predictions.npz)",
                }

            mj = src_dir / "metrics.json"
            if mj.exists():
                md = json.loads(mj.read_text())
                entry["metrics"] = md.get("metrics", {})
                entry["label_names"] = md.get("label_names", [])
                metrics_doc["setups"][setup]["archs"][arch] = {
                    "metrics": md.get("metrics", {}),
                    "label_names": md.get("label_names", []),
                    "n_test_residues": md.get("n_test_residues"),
                    "positives_per_label": md.get("positives_per_label", {}),
                }

            npz = src_dir / "predictions.npz"
            if npz.exists() and not args.skip_curves:
                d = np.load(npz)
                yt, yp = d["y_true"], d["y_prob"]
                names = entry.get("label_names") or [f"label{i}" for i in range(yt.shape[1])]
                node = metrics_doc["setups"][setup]["archs"].setdefault(arch, {})
                node["curves"] = curves_for(yt, yp, names)
                node["confusion"] = confusion_at(yt, yp, names)
                print(f"  {setup}/{arch}: curves from {yt.shape[0]:,} residues")

            manifest[f"{setup}/{arch}"] = entry
            print(f"  exported {setup}/{arch}")

    # The binary-vs-3-class comparison tables, if present.
    tc_dir = runs / "three_class"
    if tc_dir.is_dir():
        import csv
        for sub in sorted(p for p in tc_dir.iterdir() if p.is_dir()):
            pcm = sub / "per_class_metrics.csv"
            if not pcm.exists():
                continue
            with open(pcm) as f:
                rows = list(csv.DictReader(f))
            metrics_doc["three_class"][sub.name] = rows
            print(f"  three_class/{sub.name}: {len(rows)} rows")

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (out / "METRICS.json").write_text(json.dumps(metrics_doc, separators=(",", ":")))
    size = (out / "METRICS.json").stat().st_size
    total = sum(p.stat().st_size for p in out.rglob("*.pt"))
    print(f"\nMANIFEST.json: {len(manifest)} entries")
    print(f"METRICS.json:  {size/1024:.0f} KB")
    print(f"checkpoints:   {total/1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
