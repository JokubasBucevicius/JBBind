"""Test the 2-class (binary) models on the 3-class task.

Ports section 12 of ``main-workflow.ipynb`` ("Multiclass task") to the runs
produced by this program. A 1-label setup is a *binary* model -- it only answers
"does this residue bind?" -- while the matching 2-label setup resolves *what* it
binds, i.e. a 3-class task ``{Non-binding, A, B}``. The two are trained on the
same qualifying chains and the same split, so their test residues line up
element-for-element and both can be scored against the same 3-class ground truth:

    nucleic  (bind?)  vs  dna_rna      ->  Non-binding | DNA  | RNA
    protein  (bind?)  vs  homo_hetero  ->  Non-binding | Homo | Hetero

A binary model has a single score to offer, so ``P(bind)`` is used as its score
for *both* classes in the one-vs-rest curves. Its confusion matrix, however, gets
its own two-column predicted axis (``Non-bind | Binding``), because it genuinely
cannot resolve the class: folding its positives into one of the two class columns
would print counts the model never predicted, and would make its class-level
accuracy depend on which of the two columns was picked. What the two kinds *can*
be compared on directly is detection, so ``Acc (detect)`` / ``F1 (detect)`` are
reported for both and the class-level scores are left blank for the binary model.

Unlike the notebook's softmax classes, the labels here are independent sigmoids,
so a residue may be positive for both -- and the 3-class model can say so. Rather
than forcing an ``argmax``, each label is thresholded independently (the same
decision rule ``benchmarks.Evaluator`` uses elsewhere) and ``Both`` is a real
predicted class, so the class space is ``{Non-bind, A, B, Both}`` on both axes and
dual-positive residues are scorable instead of being guaranteed errors. They are
a third of nucleic binding residues, so this is not a rounding detail. One-vs-rest
ROC/PR are unaffected: a class's positives are its own label column.

Everything is read from the saved ``predictions.npz`` files -- no checkpoints are
loaded and no data is re-run. Nothing but ``runs/`` is touched, so unlike the
training entry points this one does not care about the process CWD: a relative
``--runs-dir`` that does not exist is retried against the repository root.

Usage::

    python all_class/training/program/three_class_test.py --all
    python all_class/training/program/three_class_test.py --setup dna_rna
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)

from config import REPO_ROOT, Config
from setups import SETUPS

# 2-label setup -> the 1-label ("binary") setup trained on the same chains/split.
PAIRS: Dict[str, str] = {
    "dna_rna": "nucleic",
    "homo_hetero": "protein",
}

ARCH_ORDER = ["mlp", "gnn", "gnn_mlp", "joint"]
DISPLAY = {"mlp": "MLP", "gnn": "GNN", "gnn_mlp": "GNN+MLP", "joint": "Joint"}
ARCH_COLORS = {"mlp": "#1f77b4", "gnn": "#2ca02c", "gnn_mlp": "#9467bd", "joint": "#d62728"}

# Class codes, shared by the true and the predicted axis of the 3-class models.
NONBIND, CLASS_A, CLASS_B, BOTH = 0, 1, 2, 3
CLASSES = [NONBIND, CLASS_A, CLASS_B, BOTH]

# A binary model predicts on its own two-column axis; 1 means "binding, class
# unresolved" and is deliberately not one of the class columns above.
BINARY_NONBIND, BINARY_BIND = 0, 1


class ThreeClassComparator:
    """Scores one (binary, 3-class) setup pair on the shared 3-class task."""

    def __init__(self, multi_setup: str, runs_dir: str, threshold: float):
        self.multi_setup = multi_setup
        self.binary_setup = PAIRS[multi_setup]
        self.runs_dir = runs_dir
        self.threshold = threshold
        self.labels: List[str] = list(SETUPS[multi_setup].label_names)
        self.class_names = ["Non-bind"] + self.labels + ["Both"]
        self.binary_class_names = ["Non-bind", "Binding\n(unresolved)"]
        self.out_dir = os.path.join(runs_dir, "three_class", multi_setup)

    # ------------------------------------------------------------------- load
    def _npz(self, setup: str, arch: str) -> np.lib.npyio.NpzFile | None:
        path = os.path.join(self.runs_dir, setup, arch, "predictions.npz")
        return np.load(path) if os.path.isfile(path) else None

    def _load(self) -> Dict[str, dict]:
        """Per architecture: 3-class truth plus each model's two class scores."""
        out: Dict[str, dict] = {}
        for arch in ARCH_ORDER:
            binary, multi = self._npz(self.binary_setup, arch), self._npz(self.multi_setup, arch)
            if binary is None or multi is None:
                missing = [s for s, d in ((self.binary_setup, binary), (self.multi_setup, multi))
                           if d is None]
                print(f"  [{arch}] skipped: no predictions.npz under "
                      f"{'/'.join(missing)}/{arch}")
                continue

            y_multi, y_binary = multi["y_true"], binary["y_true"]
            if y_binary.shape[0] != y_multi.shape[0]:
                print(f"  [{arch}] skipped: {self.binary_setup} has {y_binary.shape[0]} test "
                      f"residues, {self.multi_setup} has {y_multi.shape[0]}")
                continue
            if not np.array_equal(y_binary[:, 0], y_multi.max(axis=1)):
                print(f"  [{arch}] skipped: labels are not aligned residue-for-residue "
                      f"(binary truth != OR of the two class labels)")
                continue

            p_bind = binary["y_prob"][:, 0]
            out[arch] = {
                "y3": self._three_class_truth(y_multi),
                "y_ovr": y_multi,                      # (n, 2) one-vs-rest targets
                # A binary model has one score for both classes; the 3-class model
                # has one per class.
                "binary": np.stack([p_bind, p_bind], axis=1),
                "multi": multi["y_prob"],
            }
        return out

    @staticmethod
    def _three_class_truth(y_multi: np.ndarray) -> np.ndarray:
        """(n, 2) independent labels -> codes 0=non-bind, 1=A, 2=B, 3=both."""
        a, b = y_multi[:, 0] > 0, y_multi[:, 1] > 0
        y3 = np.full(len(y_multi), NONBIND, dtype=np.int64)
        y3[a & ~b] = CLASS_A
        y3[b & ~a] = CLASS_B
        y3[a & b] = BOTH
        return y3

    # ------------------------------------------------------------ predictions
    def _predict(self, probs: np.ndarray, is_binary: bool) -> np.ndarray:
        """Class predictions from a model's per-class scores.

        Binary: ``0``/``1`` on its own ``Non-bind | Binding`` axis. It has one
        score and no way to resolve the class, so it is not given class columns.
        3-class: each label thresholded independently -- the same rule the rest of
        the pipeline scores these models with -- so a residue clearing both
        thresholds is predicted ``Both`` rather than being forced to one class.
        """
        if is_binary:
            return (probs[:, 0] >= self.threshold).astype(np.int64)

        a = probs[:, 0] >= self.threshold
        b = probs[:, 1] >= self.threshold
        preds = np.full(probs.shape[0], NONBIND, dtype=np.int64)
        preds[a & ~b] = CLASS_A
        preds[b & ~a] = CLASS_B
        preds[a & b] = BOTH
        return preds

    # ----------------------------------------------------------------- metrics
    def _metrics(self, d: dict, kind: str) -> dict:
        probs, y_ovr = d[kind], d["y_ovr"]
        row: dict = {}
        for i, name in enumerate(self.labels):
            yt, yp = y_ovr[:, i], probs[:, i]
            if 0 < yt.sum() < len(yt):
                row[f"ROC AUC ({name})"] = roc_auc_score(yt, yp)
                row[f"PR AUC ({name})"] = average_precision_score(yt, yp)
            else:
                row[f"ROC AUC ({name})"] = np.nan
                row[f"PR AUC ({name})"] = np.nan

        is_binary = kind == "binary"
        y3, y_pred = d["y3"], self._predict(probs, is_binary)

        # Class-level scores cover every residue, ``Both`` included, and describe
        # exactly the confusion matrix that gets plotted. They are undefined for a
        # binary model: it predicts on a different axis, and any mapping of its
        # positives onto the class columns would be an arbitrary choice that moves
        # the number (all-binding->A and all-binding->B disagree by several points).
        if is_binary:
            row["Acc (class)"] = np.nan
            row["F1 macro (class)"] = np.nan
        else:
            row["Acc (class)"] = accuracy_score(y3, y_pred)
            row["F1 macro (class)"] = f1_score(y3, y_pred, labels=CLASSES,
                                               average="macro", zero_division=0)

        # Detection collapses both axes to bind/non-bind, which is the one task
        # both kinds of model actually perform -- so this is the like-for-like
        # comparison the class-level columns cannot provide.
        det_true = (y3 != NONBIND).astype(np.int64)
        det_pred = (y_pred != NONBIND).astype(np.int64)
        row["Acc (detect)"] = accuracy_score(det_true, det_pred)
        row["F1 (detect)"] = f1_score(det_true, det_pred, zero_division=0)
        return row

    def _table(self, data: Dict[str, dict]) -> pd.DataFrame:
        rows, index = [], []
        for kind, tag in (("binary", "binary"), ("multi", "3-class")):
            for arch in ARCH_ORDER:
                if arch in data:
                    rows.append(self._metrics(data[arch], kind))
                    index.append(f"{DISPLAY[arch]} ({tag})")
        return pd.DataFrame(rows, index=index)

    # ------------------------------------------------------------------ curves
    def _plot_curves(self, data: Dict[str, dict], df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), squeeze=False)
        for i, name in enumerate(self.labels):
            ax_roc, ax_pr = axes[i, 0], axes[i, 1]
            baseline = None
            for kind, tag, style in (("binary", "binary", "-"), ("multi", "3-class", "--")):
                for arch, d in data.items():
                    yt, yp = d["y_ovr"][:, i], d[kind][:, i]
                    if not (0 < yt.sum() < len(yt)):
                        continue
                    baseline = yt.mean()
                    label = f"{DISPLAY[arch]} ({tag})"
                    fpr, tpr, _ = roc_curve(yt, yp)
                    ax_roc.plot(fpr, tpr, style, color=ARCH_COLORS[arch], linewidth=2,
                                label=f"{label} ({df.loc[label, f'ROC AUC ({name})']:.4f})")
                    prec, rec, _ = precision_recall_curve(yt, yp)
                    ax_pr.plot(rec, prec, style, color=ARCH_COLORS[arch], linewidth=2,
                               label=f"{label} ({df.loc[label, f'PR AUC ({name})']:.4f})")
            ax_roc.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.5)
            ax_roc.set(title=f"ROC — {name} detection", xlabel="False Positive Rate",
                       ylabel="True Positive Rate")
            ax_roc.legend(loc="lower right", fontsize=8)
            ax_roc.grid(True, alpha=0.3)
            if baseline is not None:
                ax_pr.axhline(baseline, color="gray", linestyle=":",
                              label=f"Baseline ({baseline:.3f})")
            ax_pr.set(title=f"PR — {name} detection", xlabel="Recall", ylabel="Precision")
            ax_pr.legend(loc="upper right", fontsize=8)
            ax_pr.grid(True, alpha=0.3)

        fig.suptitle(f"Per-class detection on the 3-class task — "
                     f"{self.binary_setup} binary (solid) vs {self.multi_setup} 3-class (dashed)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "roc_pr_curves.png")

    # ------------------------------------------------------- confusion matrices
    def _confusion(self, y3: np.ndarray, y_pred: np.ndarray, n_cols: int) -> np.ndarray:
        """Raw counts: 4 true rows (incl. ``Both``) x ``n_cols`` predicted columns.

        The predicted axis is narrower for a binary model, which is the point --
        it has two answers to give, not four.
        """
        cm = np.zeros((len(CLASSES), n_cols), dtype=np.int64)
        np.add.at(cm, (y3, y_pred), 1)
        return cm

    def _plot_confusion(self, data: Dict[str, dict], df: pd.DataFrame) -> pd.DataFrame:
        archs = [a for a in ARCH_ORDER if a in data]
        fig, axes = plt.subplots(2, len(archs), figsize=(5.5 * len(archs), 11), squeeze=False)
        rows, counts = self.class_names, []

        for r, (kind, tag, cmap) in enumerate((("binary", "binary", "Blues"),
                                               ("multi", "3-class", "Oranges"))):
            is_binary = kind == "binary"
            cols = self.binary_class_names if is_binary else self.class_names
            for c, arch in enumerate(archs):
                d = data[arch]
                cm = self._confusion(d["y3"], self._predict(d[kind], is_binary), len(cols))
                cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(1)
                ax = axes[r, c]
                sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=1,
                            xticklabels=cols, yticklabels=rows, cbar=False, ax=ax)
                label = f"{DISPLAY[arch]} ({tag})"
                note = (f"cannot resolve the class — "
                        f"detect Acc {df.loc[label, 'Acc (detect)']:.4f}" if is_binary
                        else f"Acc {df.loc[label, 'Acc (class)']:.4f}")
                ax.set(title=f"{label}\n({note})", xlabel="Predicted", ylabel="True")

                # Long format: the two kinds of model have different predicted
                # axes, so a wide table would be half empty in both directions.
                for i, row_name in enumerate(rows):
                    total = int(cm[i].sum())
                    for j, col_name in enumerate(cols):
                        counts.append({
                            "model": label, "kind": tag,
                            "true": row_name,
                            "predicted": col_name.replace("\n", " "),
                            "count": int(cm[i, j]),
                            "row_frac": cm[i, j] / total if total else np.nan,
                        })

        fig.suptitle(f"Confusion matrices (row-normalised) — {self.multi_setup} task, "
                     f"classes {{{', '.join(self.class_names)}}}, "
                     f"threshold {self.threshold}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "confusion_matrices.png")
        return pd.DataFrame(counts)

    # ------------------------------------------------------------------ output
    def _save(self, fig, name: str) -> None:
        path = os.path.join(self.out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")

    def _print_report(self, data: Dict[str, dict], df: pd.DataFrame) -> None:
        y3 = data[next(iter(data))]["y3"]
        n = len(y3)
        n_bind = int((y3 != NONBIND).sum())
        n_both = int((y3 == BOTH).sum())
        print(f"\nTest set class distribution ({n:,} residues):")
        for code, name in enumerate(self.class_names):
            k = int((y3 == code).sum())
            print(f"  {name:<12} {k:>10,} ({100 * k / n:5.2f}%)")
        print(f"  -> dual-positive residues are {100 * n_both / max(1, n_bind):.2f}% "
              f"of binding residues; they are a predictable class here, not an "
              f"automatic error")

        print("\nPer-class detection (one-vs-rest):")
        print(df.round(4).to_string())

        print(f"\n3-class over binary, same architecture (PR AUC):")
        for arch in ARCH_ORDER:
            if arch not in data:
                continue
            for name in self.labels:
                col = f"PR AUC ({name})"
                lo = df.loc[f"{DISPLAY[arch]} (binary)", col]
                hi = df.loc[f"{DISPLAY[arch]} (3-class)", col]
                pct = 100 * (hi - lo) / lo if lo > 0 else float("nan")
                print(f"  {DISPLAY[arch]:<8} {name:<14} {lo:.4f} -> {hi:.4f} "
                      f"({hi - lo:+.4f}, {pct:+.1f}%)")

    # --------------------------------------------------------------------- run
    def run(self) -> None:
        print("=" * 70)
        print(f"3-CLASS TEST: {self.binary_setup} (binary) vs {self.multi_setup} "
              f"— {' | '.join(self.class_names)}")
        print("=" * 70)

        if not os.path.isdir(self.runs_dir):
            print(f"  runs dir not found: {self.runs_dir}\n"
                  f"  (pass --runs-dir, or run from {REPO_ROOT})")
            return
        data = self._load()
        if not data:
            print(f"  no usable {self.binary_setup}/{self.multi_setup} run pair under "
                  f"{self.runs_dir}; skipping.")
            return
        print(f"  architectures: {', '.join(DISPLAY[a] for a in data)}")
        os.makedirs(self.out_dir, exist_ok=True)

        df = self._table(data)
        path = os.path.join(self.out_dir, "per_class_metrics.csv")
        df.to_csv(path)
        print(f"  wrote {path}")

        self._plot_curves(df=df, data=data)
        counts = self._plot_confusion(data, df)
        path = os.path.join(self.out_dir, "confusion_counts.csv")
        counts.to_csv(path, index=False)
        print(f"  wrote {path}")

        self._print_report(data, df)
        print()


def resolve_runs_dir(runs_dir: str) -> str:
    """Let a relative ``runs_dir`` be given from the repo root or from anywhere.

    The default (``all_class/training/runs``) is written relative to the repo root
    like the rest of the program, but this script reads nothing else, so there is
    no reason to demand that CWD.
    """
    if os.path.isdir(runs_dir):
        return runs_dir
    from_root = os.path.join(REPO_ROOT, runs_dir)
    return from_root if os.path.isdir(from_root) else runs_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the 1-label (binary) models on the matching 3-class task")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setup", choices=list(PAIRS),
                       help="the 2-label setup naming the 3-class task")
    group.add_argument("--all", action="store_true", help="every pair in PAIRS")
    parser.add_argument("--runs-dir", default=Config().runs_dir)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="binding threshold for the confusion matrices (default: 0.5)")
    args = parser.parse_args()

    runs_dir = resolve_runs_dir(args.runs_dir)
    for name in (list(PAIRS) if args.all else [args.setup]):
        ThreeClassComparator(name, runs_dir, args.threshold).run()


if __name__ == "__main__":
    main()
