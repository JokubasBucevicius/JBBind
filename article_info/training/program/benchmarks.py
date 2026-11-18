"""Benchmarking: metrics, predictions and per-run plots.

Reuses ``train_multilabel.evaluate_multilabel`` (per-label F1 / Precision / Recall /
Balanced-Accuracy / ROC-AUC / PR-AUC, macro averages, Hamming Loss, Exact Match --
the same benchmarks as before). Plots are re-implemented here so they are robust to
any label count ``k`` (the originals hard-code 4 labels).
"""

from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)

import train_multilabel as tm

from setups import LabelSetup


def _to_python(obj):
    """Recursively coerce numpy scalars to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


class Evaluator:
    """Computes, prints and persists benchmarks for one (setup, arch) run."""

    def __init__(self, setup: LabelSetup, arch_name: str, arch_display: str, run_dir: str):
        self.setup = setup
        self.arch = arch_name
        self.arch_display = arch_display
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.metrics: Dict[str, float] = {}

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        self.setup.activate()  # ensure tm.LABEL_NAMES matches this setup's k labels
        self.y_true = np.asarray(y_true)
        self.y_prob = np.asarray(y_prob)
        self.y_pred = (self.y_prob > 0.5).astype(int)
        self.metrics = tm.evaluate_multilabel(self.y_true, self.y_pred, self.y_prob)
        return self.metrics

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"BENCHMARKS  setup={self.setup.name}  arch={self.arch_display}")
        print("=" * 60)
        for name, val in self.metrics.items():
            if isinstance(val, (int, float, np.floating, np.integer)):
                print(f"  {name:<28} {float(val):.4f}")

    def save(self) -> None:
        self._save_metrics()
        self._save_predictions()
        self._plot_roc_pr()
        self._plot_confusion()

    def _save_metrics(self) -> None:
        payload = {
            "setup": self.setup.name,
            "arch": self.arch,
            "arch_display": self.arch_display,
            "label_names": list(self.setup.label_names),
            "n_test_residues": int(self.y_true.shape[0]),
            "positives_per_label": {
                name: int(self.y_true[:, i].sum())
                for i, name in enumerate(self.setup.label_names)
            },
            "metrics": _to_python(self.metrics),
        }
        path = os.path.join(self.run_dir, "metrics.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved metrics -> {path}")

    def _save_predictions(self) -> None:
        path = os.path.join(self.run_dir, "predictions.npz")
        np.savez_compressed(path, y_true=self.y_true, y_prob=self.y_prob)
        print(f"Saved predictions -> {path}")

    def _plot_roc_pr(self) -> None:
        names = self.setup.label_names
        k = len(names)
        fig_roc, ax_roc = plt.subplots(1, k, figsize=(5 * k, 4), squeeze=False)
        fig_pr, ax_pr = plt.subplots(1, k, figsize=(5 * k, 4), squeeze=False)
        for i, name in enumerate(names):
            yt, yp = self.y_true[:, i], self.y_prob[:, i]
            if 0 < yt.sum() < len(yt):
                fpr, tpr, _ = roc_curve(yt, yp)
                ax_roc[0, i].plot(fpr, tpr, label=f"AUC={roc_auc_score(yt, yp):.3f}")
                prec, rec, _ = precision_recall_curve(yt, yp)
                ax_pr[0, i].plot(rec, prec, label=f"AP={average_precision_score(yt, yp):.3f}")
            ax_roc[0, i].plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax_roc[0, i].set(title=f"ROC: {name}", xlabel="FPR", ylabel="TPR")
            ax_roc[0, i].legend(fontsize=8)
            ax_pr[0, i].set(title=f"PR: {name}", xlabel="Recall", ylabel="Precision")
            ax_pr[0, i].legend(fontsize=8)
        for fig, fname in ((fig_roc, "roc_curves.png"), (fig_pr, "pr_curves.png")):
            fig.suptitle(f"{self.setup.name} / {self.arch_display}")
            fig.tight_layout()
            fig.savefig(os.path.join(self.run_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)

    def _plot_confusion(self) -> None:
        names = self.setup.label_names
        k = len(names)
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), squeeze=False)
        for i, name in enumerate(names):
            cm = confusion_matrix(self.y_true[:, i], self.y_pred[:, i], labels=[0, 1])
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=["Non-bind", "Bind"],
                        yticklabels=["Non-bind", "Bind"], ax=axes[0, i])
            axes[0, i].set(title=name, ylabel="True", xlabel="Pred")
        fig.suptitle(f"{self.setup.name} / {self.arch_display}")
        fig.tight_layout()
        path = os.path.join(self.run_dir, "confusion_matrices.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
