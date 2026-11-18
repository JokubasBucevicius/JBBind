"""Collect every held-out test run into one place.

Reads the run dirs ``final_test.py`` wrote (same layout as ``runs/``) and emits,
at the root of ``--test-dir``:

``all_metrics.csv``
    One row per (setup, arch): every metric ``evaluate_multilabel`` reports,
    i.e. exactly the benchmarks printed during training.
``all_labels_metrics.csv``
    One row per (setup, arch, label), with prevalence, PR lift, the confusion
    counts at 0.5 and the best achievable F1 over all thresholds. Same schema as
    ``runs/all_runs_metrics.csv``, so the two files stack.
``train_vs_test.csv``
    Held-out metrics beside the internal-split metrics from ``--runs-dir``, with
    the delta -- the generalisation gap, per label and per macro metric.
``summary.txt``
    The same tables, formatted for reading in a terminal.
``report.html``
    Everything above plus the plots, as a single self-contained page.

Usage (from the repository root)::

    python all_class/training/program/final_report.py
"""

from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from config import REPO_ROOT, Config
from setups import SETUPS

ARCH_ORDER = ["mlp", "gnn", "gnn_mlp", "joint"]
DISPLAY = {"mlp": "MLP", "gnn": "GNN", "gnn_mlp": "GNN+MLP", "joint": "Joint"}

MACRO_KEYS = ["PR AUC (macro)", "ROC AUC (macro)", "F1 (macro)", "Precision (macro)",
              "Recall (macro)", "Bal Acc (macro)", "Hamming Loss", "Exact Match"]
PER_LABEL_KEYS = ["PR AUC", "ROC AUC", "F1", "Precision", "Recall", "Bal Acc"]

DEFAULT_TEST_DIR = "all_class/training/test_results"


# --------------------------------------------------------------------- loading
def load_runs(test_dir: str) -> Dict[Tuple[str, str], dict]:
    runs = {}
    for sname in SETUPS:
        for arch in ARCH_ORDER:
            path = os.path.join(test_dir, sname, arch, "metrics.json")
            if os.path.isfile(path):
                with open(path) as fh:
                    runs[(sname, arch)] = json.load(fh)
    return runs


# ------------------------------------------------------------- wide metrics csv
def build_all_metrics(runs: Dict[Tuple[str, str], dict]) -> pd.DataFrame:
    rows = []
    for (sname, arch), payload in runs.items():
        m = payload["metrics"]
        row = {
            "setup": sname,
            "architecture": DISPLAY[arch],
            "arch": arch,
            "labels": " | ".join(payload["label_names"]),
            "n_test_chains": payload.get("n_test_chains"),
            "n_test_residues": payload.get("n_test_residues"),
        }
        row.update({k: m.get(k) for k in MACRO_KEYS})
        for label in payload["label_names"]:
            for key in PER_LABEL_KEYS:
                row[f"{key} ({label})"] = m.get(f"{key} ({label})")
        rows.append(row)
    df = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(SETUPS)}
    df = df.sort_values(["setup", "arch"],
                        key=lambda c: c.map(order) if c.name == "setup"
                        else c.map({a: i for i, a in enumerate(ARCH_ORDER)}))
    return df.reset_index(drop=True)


# ------------------------------------------------------- per-label metrics csv
def _best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Highest F1 over every threshold, and the threshold that achieves it."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1), 0.0)
    i = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(f1[i]), float(thr[i]) if len(thr) else float("nan")


def build_label_metrics(runs: Dict[Tuple[str, str], dict], test_dir: str,
                        threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for (sname, arch), payload in runs.items():
        npz_path = os.path.join(test_dir, sname, arch, "predictions.npz")
        if not os.path.isfile(npz_path):
            continue
        with np.load(npz_path) as d:
            y_true, y_prob = d["y_true"], d["y_prob"]
        n = len(y_true)
        for i, label in enumerate(payload["label_names"]):
            yt, yp = y_true[:, i], y_prob[:, i]
            pred = (yp > threshold).astype(int)
            pos = int(yt.sum())
            prevalence = pos / n if n else float("nan")
            tp = int(((pred == 1) & (yt == 1)).sum())
            fp = int(((pred == 1) & (yt == 0)).sum())
            fn = int(((pred == 0) & (yt == 1)).sum())
            tn = int(((pred == 0) & (yt == 0)).sum())
            m = payload["metrics"]
            pr_auc = m.get(f"PR AUC ({label})")
            scorable = 0 < pos < n
            best_f1, best_thr = _best_f1(yt, yp) if scorable else (float("nan"),) * 2
            rows.append({
                "task": sname,
                "architecture": arch,
                "label": label,
                "n_test_chains": payload.get("n_test_chains"),
                "n_test_residues": n,
                "positives": pos,
                "prevalence": prevalence,
                "roc_auc": m.get(f"ROC AUC ({label})"),
                "pr_auc": pr_auc,
                "pr_lift": (pr_auc / prevalence) if prevalence else float("nan"),
                "precision_05": m.get(f"Precision ({label})"),
                "recall_05": m.get(f"Recall ({label})"),
                "f1_05": m.get(f"F1 ({label})"),
                "bal_acc_05": m.get(f"Bal Acc ({label})"),
                "best_f1": best_f1,
                "best_threshold": best_thr,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order_s = {s: i for i, s in enumerate(SETUPS)}
    order_a = {a: i for i, a in enumerate(ARCH_ORDER)}
    df["_s"], df["_a"] = df["task"].map(order_s), df["architecture"].map(order_a)
    return df.sort_values(["_s", "_a", "label"]).drop(columns=["_s", "_a"]).reset_index(drop=True)


# ------------------------------------------------------------ train vs. held-out
def build_train_vs_test(runs: Dict[Tuple[str, str], dict], runs_dir: str) -> pd.DataFrame:
    """Internal-split metrics beside held-out metrics, plus the gap."""
    rows = []
    for (sname, arch), payload in runs.items():
        train_path = os.path.join(runs_dir, sname, arch, "metrics.json")
        if not os.path.isfile(train_path):
            continue
        with open(train_path) as fh:
            train = json.load(fh)
        keys = MACRO_KEYS + [f"{k} ({l})" for l in payload["label_names"]
                             for k in PER_LABEL_KEYS]
        for key in keys:
            a, b = train["metrics"].get(key), payload["metrics"].get(key)
            if a is None or b is None:
                continue
            rows.append({
                "setup": sname, "architecture": DISPLAY[arch], "arch": arch,
                "metric": key, "internal_split": a, "held_out": b, "delta": b - a,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ text report
def write_summary(test_dir: str, wide: pd.DataFrame, labels: pd.DataFrame,
                  gap: pd.DataFrame, header: List[str],
                  logs_dir: Optional[str] = None) -> str:
    out = list(header) + [""]

    dist_path = os.path.join(test_dir, "label_distribution.txt")
    if os.path.isfile(dist_path):
        with open(dist_path) as fh:
            out.append(fh.read().rstrip())
        out.append("")
        train = training_composition(logs_dir) if logs_dir else pd.DataFrame()
        by_setup_path = os.path.join(test_dir, "label_distribution_by_setup.csv")
        if not train.empty and os.path.isfile(by_setup_path):
            merged = pd.read_csv(by_setup_path).merge(train, on=["setup", "label"],
                                                      how="left")
            merged["shift_pp"] = merged["positive_pct"] - merged["train_pct"]
            out.append("-" * 78)
            out.append("HELD-OUT vs TRAINING SPLIT — positive rate per label "
                       "(percentage points)")
            out.append("-" * 78)
            cols = ["setup", "label", "positive_pct", "train_pct", "shift_pp"]
            out.append(merged[cols].to_string(
                index=False, float_format=lambda v: f"{v:.4f}"))
            out.append("")

    out.append("=" * 78)
    out.append("HEADLINE — macro benchmarks per (setup, architecture)")
    out.append("=" * 78)
    cols = ["setup", "architecture", "n_test_chains", "n_test_residues"] + MACRO_KEYS
    out.append(wide[cols].round(4).to_string(index=False))
    out.append("")

    if not labels.empty:
        out.append("=" * 78)
        out.append("PER LABEL — detection quality (threshold 0.5 unless stated)")
        out.append("=" * 78)
        cols = ["task", "architecture", "label", "positives", "prevalence", "roc_auc",
                "pr_auc", "pr_lift", "f1_05", "best_f1", "best_threshold"]
        out.append(labels[cols].round(4).to_string(index=False))
        out.append("")

    for sname in SETUPS:
        sub = wide[wide["setup"] == sname]
        if sub.empty:
            continue
        best = sub.loc[sub["PR AUC (macro)"].idxmax()]
        out.append(f"Best on {sname}: {best['architecture']} "
                   f"(PR AUC macro {best['PR AUC (macro)']:.4f}, "
                   f"ROC AUC macro {best['ROC AUC (macro)']:.4f})")
    out.append("")

    if not gap.empty:
        out.append("=" * 78)
        out.append("GENERALISATION GAP — held-out minus internal test split (macro only)")
        out.append("=" * 78)
        macro = gap[gap["metric"].isin(MACRO_KEYS)]
        pivot = macro.pivot_table(index=["setup", "architecture"], columns="metric",
                                  values="delta")
        out.append(pivot[MACRO_KEYS].round(4).to_string())
        out.append("")

    text = "\n".join(out).rstrip() + "\n"
    path = os.path.join(test_dir, "summary.txt")
    with open(path, "w") as fh:
        fh.write(text)
    print(f"  wrote {path}")
    return text


# ------------------------------------------------------------------ html report
def _img(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


TEXT_COLS = {"setup", "architecture", "arch", "task", "label", "labels", "metric"}


def _table(df: pd.DataFrame, float_fmt: str = "{:.4f}",
           signed: Optional[List[str]] = None,
           mark: Optional[Dict[int, str]] = None,
           lower_better: Optional[List[str]] = None) -> str:
    """Render a frame. ``signed`` columns get a +/- sign and pos/neg colour,
    read against ``lower_better`` so a rising Hamming Loss is not shown as good.
    ``mark`` chips a row index with a short label."""
    signed = set(signed or [])
    lower_better = set(lower_better or [])
    mark = mark or {}

    def cell(col, v):
        cls = [] if col in TEXT_COLS else ["num"]
        if pd.isna(v):
            return f'<td class="{" ".join(cls)}"></td>', ""
        if col in signed and isinstance(v, (float, np.floating)):
            better = (v < 0) if col in lower_better else (v > 0)
            cls.append("pos" if better else "neg")
            return f'<td class="{" ".join(cls)}">{v:+.4f}</td>', ""
        if isinstance(v, (float, np.floating)):
            return f'<td class="{" ".join(cls)}">{float_fmt.format(v)}</td>', ""
        if isinstance(v, (int, np.integer, bool)):
            return f'<td class="{" ".join(cls)}">{int(v):,}</td>', ""
        return f'<td class="{" ".join(cls)}">{html.escape(str(v))}</td>', ""

    head = "".join(
        f'<th class="{"" if c in TEXT_COLS else "num"}">{html.escape(str(c))}</th>'
        for c in df.columns)

    rows = []
    prev_group = None
    for i, row in enumerate(df.itertuples(index=False)):
        group = getattr(row, "setup", None) or getattr(row, "task", None)
        tr_cls = "group" if (group is not None and group != prev_group and i) else ""
        prev_group = group
        cells = []
        for col, v in zip(df.columns, row):
            td, _ = cell(col, v)
            if col in ("architecture", "arch") and i in mark:
                td = td[:-5] + f'<span class="chip">{html.escape(mark[i])}</span></td>'
            cells.append(td)
        rows.append(f'<tr class="{tr_cls}">' + "".join(cells) + "</tr>")

    return ('<div class="scroll"><table><thead><tr>' + head
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


# Palette: cool slate neutrals pulled toward the teal of solvent-accessible-surface
# renderings; green/clay are reserved for the generalisation-gap signs so they never
# compete with the accent.
HTML_CSS = """
:root{
  --bg:#fbfcfc; --panel:#f2f5f6; --fg:#141a1c; --muted:#5a6b70; --line:#dde4e6;
  --rule:#c9d5d8; --accent:#0f6c78; --accent-soft:#e3eef0;
  --pos:#1d6f4f; --neg:#a8452e; --shadow:0 1px 2px rgba(20,26,28,.05);
  --serif:ui-serif,"Iowan Old Style","Source Serif 4",Palatino,Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Mono",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e1416; --panel:#161e21; --fg:#e4eaeb; --muted:#8fa1a6; --line:#252f33;
  --rule:#33454a; --accent:#4fb3bf; --accent-soft:#14282d;
  --pos:#5fbf95; --neg:#de8a6e; --shadow:none;
}}
:root[data-theme="dark"]{
  --bg:#0e1416; --panel:#161e21; --fg:#e4eaeb; --muted:#8fa1a6; --line:#252f33;
  --rule:#33454a; --accent:#4fb3bf; --accent-soft:#14282d;
  --pos:#5fbf95; --neg:#de8a6e; --shadow:none;
}

*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:3.5rem 1.5rem 7rem;
  font:16px/1.65 var(--sans);-webkit-font-smoothing:antialiased}
main{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:0}

h1{font:600 2.35rem/1.15 var(--serif);margin:0;letter-spacing:-.015em;
   text-wrap:balance}
.meta{color:var(--muted);font:400 .84rem/1.6 var(--mono);margin:.75rem 0 0;
  font-variant-numeric:tabular-nums}
h2{font:600 1.45rem/1.25 var(--serif);margin:3.5rem 0 0;padding-bottom:.5rem;
   border-bottom:1px solid var(--rule);text-wrap:balance}
h3{font:600 .76rem/1.4 var(--sans);margin:2.25rem 0 0;color:var(--accent);
   text-transform:uppercase;letter-spacing:.1em}
p{color:var(--muted);max-width:68ch;margin:.9rem 0 0}
h2 + p, h3 + p{margin-top:.7rem}

.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
  padding:.95rem 1.15rem;border-radius:0 4px 4px 0;margin:1.5rem 0 0}
.note p{margin:0;max-width:none;color:var(--fg);font-size:.94rem}
.note p + p{margin-top:.5rem}
.note strong{color:var(--accent)}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:.85rem;margin:2rem 0 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:4px;
  padding:1rem 1.1rem;box-shadow:var(--shadow)}
.tile .k{color:var(--muted);font:600 .68rem/1.4 var(--sans);
  text-transform:uppercase;letter-spacing:.1em}
.tile .v{font:600 2rem/1.1 var(--mono);letter-spacing:-.03em;margin:.4rem 0 .2rem;
  font-variant-numeric:tabular-nums}
.tile .s{color:var(--muted);font-size:.83rem;line-height:1.4}

.scroll{overflow-x:auto;margin:1.1rem 0 0;border:1px solid var(--line);
  border-radius:4px;background:var(--panel)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:13px}
th,td{padding:.5rem .75rem;text-align:left;white-space:nowrap;
  border-bottom:1px solid var(--line)}
td.num,th.num{text-align:right;font-family:var(--mono);
  font-variant-numeric:tabular-nums}
th{background:var(--panel);font:600 .7rem/1.5 var(--sans);color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;position:sticky;top:0;z-index:1;
  border-bottom:1px solid var(--rule)}
th.num{font-family:var(--sans)}
tbody tr:last-child td{border-bottom:none}
tbody tr.group td{border-top:1px solid var(--rule)}
tbody tr:hover td{background:var(--accent-soft)}
td.pos{color:var(--pos)}
td.neg{color:var(--neg)}
.chip{display:inline-block;margin-left:.5rem;padding:.05rem .4rem;border-radius:3px;
  background:var(--accent);color:var(--bg);font:600 .62rem/1.6 var(--sans);
  text-transform:uppercase;letter-spacing:.07em;vertical-align:1px}

figure{margin:1.25rem 0 0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;
  background:#fff;display:block}
figcaption{color:var(--muted);font-size:.83rem;margin-top:.5rem}
code{font-family:var(--mono);font-size:.88em;background:var(--panel);
  border:1px solid var(--line);padding:.05rem .3rem;border-radius:3px}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:640px){body{padding:2rem 1rem 4rem}h1{font-size:1.8rem}}
"""


_RE_DIST = re.compile(r"Training label distribution \(per residue, n=(\d+)\):")
_RE_CLASS = re.compile(r"^\s{2}(.+?):\s+(\d+)\s+\(([\d.]+)%\)\s*$")


def training_composition(logs_dir: str) -> pd.DataFrame:
    """Scrape each setup's training-split class balance out of its slurm log.

    The training runs print this block and nothing else records it, so the logs
    are the only source for the side-by-side with the held-out set.
    """
    rows = []
    for sname in SETUPS:
        cands = sorted(glob.glob(os.path.join(logs_dir, f"{sname}_mlp_*.out")))
        if not cands:
            continue
        with open(cands[-1], errors="replace") as fh:
            lines = fh.readlines()
        n_res, in_dist = None, False
        for line in lines:
            if (m := _RE_DIST.search(line)):
                if n_res is not None:
                    break                      # only the first block
                n_res, in_dist = int(m.group(1)), True
            elif in_dist:
                if (m := _RE_CLASS.match(line)):
                    rows.append({"setup": sname, "label": m.group(1),
                                 "train_residues": n_res,
                                 "train_positives": int(m.group(2)),
                                 "train_pct": float(m.group(3))})
                else:
                    in_dist = False
    return pd.DataFrame(rows)


def _composition_section(test_dir: str, logs_dir: Optional[str]) -> List[str]:
    """Dataset composition, and how it compares to what the models trained on."""
    by_setup_path = os.path.join(test_dir, "label_distribution_by_setup.csv")
    overall_path = os.path.join(test_dir, "label_distribution.csv")
    if not os.path.isfile(overall_path):
        return []

    parts = ["<h2>Dataset composition</h2>"]
    overall = pd.read_csv(overall_path)
    parts.append("<p>Counts are over surface residues (<code>sas_area &gt; 0</code>) — the "
                 "population the models score. A residue can carry more than one label, so "
                 "the base rows sum past the &ldquo;any binding&rdquo; row.</p>")
    parts.append(_table(overall[["label", "kind", "residues", "residue_pct",
                                 "chains", "chain_pct", "pos_weight"]],
                        float_fmt="{:.4f}"))

    if os.path.isfile(by_setup_path):
        by_setup = pd.read_csv(by_setup_path)
        train = training_composition(logs_dir) if logs_dir else pd.DataFrame()
        parts.append("<h3>Per setup, against what the models trained on</h3>")
        if not train.empty:
            merged = by_setup.merge(train, on=["setup", "label"], how="left")
            merged["shift_pp"] = merged["positive_pct"] - merged["train_pct"]
            cols = ["setup", "label", "qualifying_chains", "residues", "positives",
                    "positive_pct", "train_pct", "shift_pp", "pos_weight"]
            parts.append("<p>The last columns compare the held-out positive rate with the "
                         "training split's, in percentage points. Where the balance moved, "
                         "the task is genuinely different from the one the model was tuned "
                         "for — read the gap table below with that in mind.</p>")
            parts.append(_table(merged[cols], signed=["shift_pp"]))
        else:
            parts.append(_table(by_setup))
    return parts


def write_html(test_dir: str, wide: pd.DataFrame, labels: pd.DataFrame,
               gap: pd.DataFrame, header: List[str], runs: Dict[Tuple[str, str], dict],
               embed_plots: bool, logs_dir: Optional[str] = None) -> str:
    parts: List[str] = []
    parts.append(f"<title>Held-out test results — binding-site models</title>")
    parts.append(f"<style>{HTML_CSS}</style>")
    parts.append("<main>")
    parts.append("<h1>Held-out test — all trained models</h1>")
    parts.append('<p class="meta">' + " &middot; ".join(html.escape(h) for h in header)
                 + "</p>")

    n_chains = int(wide["n_test_chains"].max()) if "n_test_chains" in wide else 0
    n_res = int(wide["n_test_residues"].max()) if len(wide) else 0
    best = wide.loc[wide["PR AUC (macro)"].idxmax()] if len(wide) else None
    parts.append('<div class="grid">')
    parts.append(f'<div class="tile"><div class="k">Models scored</div>'
                 f'<div class="v">{len(wide)}</div>'
                 f'<div class="s">{wide["setup"].nunique()} setups &times; '
                 f'{wide["arch"].nunique()} architectures</div></div>')
    parts.append(f'<div class="tile"><div class="k">Test chains</div>'
                 f'<div class="v">{n_chains:,}</div>'
                 f'<div class="s">up to {n_res:,} residues</div></div>')
    if best is not None:
        parts.append(f'<div class="tile"><div class="k">Best macro PR-AUC</div>'
                     f'<div class="v">{best["PR AUC (macro)"]:.3f}</div>'
                     f'<div class="s">{best["architecture"]} on '
                     f'{html.escape(best["setup"])}</div></div>')
    parts.append("</div>")

    parts.append('<div class="note"><p><strong>Same benchmarks as training.</strong> '
                 'Every number comes from <code>evaluate_multilabel</code> at threshold '
                 '0.5 — the identical code path that produced the per-run metrics during '
                 'training — so held-out and internal-split figures are directly '
                 'comparable.</p>'
                 '<p><strong>Chains are new.</strong> Each model is scored on chains that '
                 'share no name with anything in the training graph directory, filtered '
                 'per setup by the same rule training used: keep a chain only if it has at '
                 'least one surface binding residue of that setup\'s type.</p></div>')

    comp = _composition_section(test_dir, logs_dir)
    if comp:
        parts.extend(comp)

    parts.append("<h2>Headline — macro benchmarks</h2>")
    parts.append("<p>Higher is better everywhere except Hamming Loss. The chip marks the "
                 "architecture with the best macro PR-AUC within each setup.</p>")
    cols = ["setup", "architecture", "n_test_chains", "n_test_residues"] + MACRO_KEYS
    head = wide[cols].reset_index(drop=True)
    best_rows = {}
    for sname in head["setup"].unique():
        sub = head[head["setup"] == sname]
        best_rows[int(sub["PR AUC (macro)"].idxmax())] = "best"
    parts.append(_table(head, mark=best_rows))

    if not labels.empty:
        parts.append("<h2>Per label</h2>")
        parts.append("<p>Prevalence is the positive rate among scored residues; PR lift is "
                     "PR-AUC over that prevalence, which is what makes the rare labels "
                     "comparable to the common ones. <code>best_f1</code> is the highest F1 "
                     "reachable at any threshold, with the threshold that gets there — the "
                     "gap between it and <code>f1_05</code> is what recalibrating the "
                     "threshold would buy.</p>")
        cols = ["task", "architecture", "label", "n_test_residues", "positives",
                "prevalence", "roc_auc", "pr_auc", "pr_lift", "precision_05",
                "recall_05", "f1_05", "bal_acc_05", "best_f1", "best_threshold"]
        parts.append(_table(labels[cols].reset_index(drop=True)))

        parts.append("<h3>Confusion counts at threshold 0.5</h3>")
        parts.append(_table(labels[["task", "architecture", "label",
                                    "tn", "fp", "fn", "tp"]].reset_index(drop=True)))

    if not gap.empty:
        parts.append("<h2>Generalisation gap</h2>")
        parts.append("<p>Held-out minus the internal test split reported during training. "
                     "Green is the direction you want — better on unseen chains — which for "
                     "Hamming Loss means a negative number.</p>")
        macro = gap[gap["metric"].isin(MACRO_KEYS)]
        pivot = (macro.pivot_table(index=["setup", "architecture"], columns="metric",
                                   values="delta")[MACRO_KEYS].reset_index())
        pivot = pivot.sort_values("setup", key=lambda c: c.map(
            {s: i for i, s in enumerate(SETUPS)})).reset_index(drop=True)
        parts.append(_table(pivot, signed=MACRO_KEYS, lower_better=["Hamming Loss"]))

    if embed_plots:
        parts.append("<h2>Curves and confusion matrices</h2>")
        for sname in SETUPS:
            setup_dir = os.path.join(test_dir, sname)
            imgs = [("comparison.png", "PR-AUC and F1 per label, across architectures"),
                    ("comparison_curves.png", "PR and ROC curves, across architectures")]
            shown = [(fn, cap, _img(os.path.join(setup_dir, fn))) for fn, cap in imgs]
            shown = [(fn, cap, src) for fn, cap, src in shown if src]
            if not shown:
                continue
            parts.append(f"<h3>{html.escape(sname)}</h3>")
            for fn, cap, src in shown:
                parts.append(f'<figure><img src="{src}" alt="{html.escape(cap)}">'
                             f'<figcaption>{html.escape(cap)}</figcaption></figure>')

        for pair in ("dna_rna", "homo_hetero"):
            d = os.path.join(test_dir, "three_class", pair)
            imgs = [("roc_pr_curves.png", "Binary vs. 3-class detection"),
                    ("confusion_matrices.png", "3-class confusion matrices")]
            shown = [(cap, _img(os.path.join(d, fn))) for fn, cap in imgs]
            shown = [(cap, src) for cap, src in shown if src]
            if not shown:
                continue
            parts.append(f"<h3>3-class task — {html.escape(pair)}</h3>")
            for cap, src in shown:
                parts.append(f'<figure><img src="{src}" alt="{html.escape(cap)}">'
                             f'<figcaption>{html.escape(cap)}</figcaption></figure>')

    parts.append("<h2>How to read this</h2>")
    if any("feature_mlp" in p for p in runs.values()):
        parts.append('<div class="note"><p><strong>One reproduction, not a replay.</strong> '
                     'GNN+MLP feeds its GNN a 64-d embedding from a prerequisite MLP that '
                     'the training script trained but never checkpointed. The standalone '
                     '<code>mlp</code> run of the same setup — same architecture, chains, '
                     'split, criterion and seed — stands in as the feature extractor. Its '
                     'numbers are a close reproduction; the other three architectures load '
                     'their own saved weights.</p></div>')
    parts.append("<p>Residue counts differ slightly between architectures within a setup: "
                 "the three ESM-based paths drop residues whose sequence position falls "
                 "past the end of the chain's embedding, while the structural GNN keeps "
                 "every surface residue. That is the same behaviour the training runs had.</p>")
    parts.append(f"<p>All of the above, plus per-run <code>metrics.json</code>, "
                 f"<code>predictions.npz</code> and plots, are under "
                 f"<code>{html.escape(test_dir)}</code>.</p>")
    parts.append("</main>")

    doc = "\n".join(parts)
    path = os.path.join(test_dir, "report.html")
    with open(path, "w") as fh:
        fh.write(doc)
    print(f"  wrote {path} ({len(doc) / 1e6:.1f} MB)")
    return path


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="Collect the held-out test runs into one place")
    ap.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    ap.add_argument("--runs-dir", default=None,
                    help="training runs, for the train-vs-test comparison")
    ap.add_argument("--logs-dir", default="all_class/logs",
                    help="slurm logs, for the training-split class balance")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip embedding PNGs in report.html")
    args = ap.parse_args()

    test_dir = args.test_dir if os.path.isdir(args.test_dir) else \
        os.path.join(REPO_ROOT, args.test_dir)
    runs_dir = args.runs_dir or Config().runs_dir
    if not os.path.isdir(runs_dir):
        runs_dir = os.path.join(REPO_ROOT, runs_dir)

    runs = load_runs(test_dir)
    if not runs:
        raise SystemExit(f"No metrics.json found under {test_dir}; run final_test.py first.")
    print(f"Collecting {len(runs)} run(s) from {test_dir}")

    wide = build_all_metrics(runs)
    wide.to_csv(os.path.join(test_dir, "all_metrics.csv"), index=False)
    print(f"  wrote {os.path.join(test_dir, 'all_metrics.csv')}")

    labels = build_label_metrics(runs, test_dir)
    if not labels.empty:
        labels.to_csv(os.path.join(test_dir, "all_labels_metrics.csv"), index=False)
        print(f"  wrote {os.path.join(test_dir, 'all_labels_metrics.csv')}")

    gap = build_train_vs_test(runs, runs_dir)
    if not gap.empty:
        gap.to_csv(os.path.join(test_dir, "train_vs_test.csv"), index=False)
        print(f"  wrote {os.path.join(test_dir, 'train_vs_test.csv')}")

    any_payload = next(iter(runs.values()))
    header = [
        f"{len(runs)} runs — {wide['setup'].nunique()} setups x {wide['arch'].nunique()} architectures",
        f"up to {int(wide['n_test_chains'].max()):,} held-out chains",
        f"threshold 0.5, metrics from evaluate_multilabel",
        f"split={any_payload.get('split', 'held_out_test')}",
    ]
    logs_dir = args.logs_dir if os.path.isdir(args.logs_dir) else \
        os.path.join(REPO_ROOT, args.logs_dir)
    logs_dir = logs_dir if os.path.isdir(logs_dir) else None

    write_summary(test_dir, wide, labels, gap, header, logs_dir)
    write_html(test_dir, wide, labels, gap, header, runs,
               embed_plots=not args.no_plots, logs_dir=logs_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
