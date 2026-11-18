"""Per-setup comparison of the 4 architectures.

Reads ``runs/<setup>/<arch>/metrics.json`` (and ``predictions.npz``) for the four
architectures of one setup and writes, into ``runs/<setup>/``:

* ``comparison.csv``  -- one row per architecture, key benchmarks as columns;
* ``comparison.png``  -- per-label PR-AUC / F1 bar chart across architectures;
* ``comparison_curves.png`` -- per-label PR & ROC curves overlaid across architectures.

Scope is one setup at a time (the 4 architectures), matching the requested
comparison; there is intentionally no global cross-setup comparison.

Usage (from repo root)::

    python all_class/training/program/aggregate.py --setup dna_rna
    python all_class/training/program/aggregate.py --all
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from setups import SETUPS


class SetupComparator:
    ARCH_ORDER = ["mlp", "gnn", "gnn_mlp", "joint"]
    DISPLAY = {"mlp": "MLP", "gnn": "GNN", "gnn_mlp": "GNN+MLP", "joint": "Joint"}

    def __init__(self, setup_name: str, runs_dir: str):
        self.setup_name = setup_name
        self.setup_dir = os.path.join(runs_dir, setup_name)

    def _load(self) -> Dict[str, dict]:
        data = {}
        for arch in self.ARCH_ORDER:
            path = os.path.join(self.setup_dir, arch, "metrics.json")
            if os.path.isfile(path):
                with open(path) as fh:
                    data[arch] = json.load(fh)
        return data

    def run(self) -> None:
        data = self._load()
        if not data:
            print(f"[{self.setup_name}] no metrics.json found under {self.setup_dir}; skipping.")
            return
        label_names = data[next(iter(data))]["label_names"]
        print(f"[{self.setup_name}] comparing: {[self.DISPLAY[a] for a in data]}")
        self._write_csv(data, label_names)
        self._plot_bars(data, label_names)
        self._plot_curves(data, label_names)

    # ------------------------------------------------------------------ table
    def _write_csv(self, data: Dict[str, dict], label_names) -> None:
        macro = ["PR AUC (macro)", "ROC AUC (macro)", "F1 (macro)",
                 "Precision (macro)", "Recall (macro)", "Bal Acc (macro)",
                 "Hamming Loss", "Exact Match"]
        per_label = [f"{m} ({n})" for n in label_names for m in ("PR AUC", "F1")]
        rows = []
        for arch in self.ARCH_ORDER:
            if arch not in data:
                continue
            m = data[arch]["metrics"]
            row = {"architecture": self.DISPLAY[arch],
                   "n_test_residues": data[arch].get("n_test_residues")}
            for key in macro + per_label:
                row[key] = m.get(key)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("architecture")
        path = os.path.join(self.setup_dir, "comparison.csv")
        df.to_csv(path)
        print(f"  wrote {path}")
        print(df[macro].round(4).to_string())

    # ------------------------------------------------------------------- bars
    def _plot_bars(self, data: Dict[str, dict], label_names) -> None:
        archs = [a for a in self.ARCH_ORDER if a in data]
        k = len(label_names)
        x = np.arange(k)
        width = 0.8 / max(1, len(archs))
        fig, axes = plt.subplots(1, 2, figsize=(7 * 2, 5), squeeze=False)
        for ax_idx, metric in enumerate(["PR AUC", "F1"]):
            ax = axes[0, ax_idx]
            for j, arch in enumerate(archs):
                m = data[arch]["metrics"]
                vals = [m.get(f"{metric} ({n})", 0) for n in label_names]
                ax.bar(x + j * width, vals, width, label=self.DISPLAY[arch])
            ax.set(xlabel="Label", ylabel=metric, title=f"{metric} per label",
                   ylim=(0, 1))
            ax.set_xticks(x + width * (len(archs) - 1) / 2)
            ax.set_xticklabels(label_names, rotation=15)
            ax.legend(fontsize=8)
        fig.suptitle(f"Architecture comparison — setup '{self.setup_name}'")
        fig.tight_layout()
        path = os.path.join(self.setup_dir, "comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")

    # ----------------------------------------------------------------- curves
    def _plot_curves(self, data: Dict[str, dict], label_names) -> None:
        from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                      roc_auc_score, roc_curve)
        preds = {}
        for arch in self.ARCH_ORDER:
            npz = os.path.join(self.setup_dir, arch, "predictions.npz")
            if arch in data and os.path.isfile(npz):
                preds[arch] = np.load(npz)
        if not preds:
            return
        k = len(label_names)
        fig, axes = plt.subplots(2, k, figsize=(5 * k, 9), squeeze=False)
        for i, name in enumerate(label_names):
            for arch, d in preds.items():
                yt, yp = d["y_true"][:, i], d["y_prob"][:, i]
                if not (0 < yt.sum() < len(yt)):
                    continue
                prec, rec, _ = precision_recall_curve(yt, yp)
                axes[0, i].plot(rec, prec,
                                label=f"{self.DISPLAY[arch]} (AP={average_precision_score(yt, yp):.3f})")
                fpr, tpr, _ = roc_curve(yt, yp)
                axes[1, i].plot(fpr, tpr,
                                label=f"{self.DISPLAY[arch]} (AUC={roc_auc_score(yt, yp):.3f})")
            axes[0, i].set(title=f"PR: {name}", xlabel="Recall", ylabel="Precision")
            axes[0, i].legend(fontsize=8)
            axes[1, i].plot([0, 1], [0, 1], "k--", alpha=0.3)
            axes[1, i].set(title=f"ROC: {name}", xlabel="FPR", ylabel="TPR")
            axes[1, i].legend(fontsize=8)
        fig.suptitle(f"Architecture curves — setup '{self.setup_name}'")
        fig.tight_layout()
        path = os.path.join(self.setup_dir, "comparison_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the 4 architectures within a setup")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setup", choices=list(SETUPS))
    group.add_argument("--all", action="store_true", help="aggregate every setup")
    parser.add_argument("--runs-dir", default=Config().runs_dir)
    args = parser.parse_args()

    setups = list(SETUPS) if args.all else [args.setup]
    for name in setups:
        SetupComparator(name, args.runs_dir).run()


if __name__ == "__main__":
    main()
