"""Label composition of a graph dataset -- counts and percentages.

The training runs log a ``Training label distribution`` block per setup (that is
what ``results_to_txt.py`` scrapes), but nothing recorded the same for the
held-out test set. This computes it from the per-chain cache ``testset.py``
builds, so it costs seconds rather than another pass over the CSVs.

Three views, because "how many of each label" means different things per level:

* **residue** -- the population the models actually score (surface residues,
  ``sas_area > 0``), which is what class imbalance and ``pos_weight`` refer to;
* **chain** -- how many chains contain at least one residue of each type, which
  is what the per-setup qualifying filter selects on;
* **overlap** -- residues carrying more than one label at once, which the
  independent-sigmoid formulation allows and a softmax would not.

Usage (from the repository root)::

    python all_class/training/program/label_stats.py
    python all_class/training/program/label_stats.py --cache-dir <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import REPO_ROOT
from setups import BASE_LABEL_COLUMNS, SETUPS

import final_test
import testset

# Derived groupings worth naming, as index groups over the four base columns.
DERIVED: Dict[str, Sequence[int]] = {
    "Protein (homo|hetero)": (0, 1),
    "Nucleic acid (DNA|RNA)": (2, 3),
    "Any binding": (0, 1, 2, 3),
}

PRETTY = {
    "binds_homo_protein": "Homo-protein",
    "binds_hetero_protein": "Hetero-protein",
    "binds_dna": "DNA",
    "binds_rna": "RNA",
}


def collect(cache_dir: str, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-chain residue counts and per-label positive counts.

    Returns ``(per_chain_pos, n_res_per_chain, n_chains)`` where ``per_chain_pos``
    is ``(n_chains, 4)`` positive-residue counts and ``n_res_per_chain`` is
    ``(n_chains,)``.
    """
    pos, sizes = [], []
    for name in tqdm(names, desc="Labels"):
        entry = testset.load_entry(cache_dir, name)
        if entry is None:
            continue
        y4 = entry["y4"]
        pos.append((y4 > 0).sum(axis=0))
        sizes.append(y4.shape[0])
    return np.array(pos, dtype=np.int64), np.array(sizes, dtype=np.int64), len(sizes)


def _group_positive(cache_dir: str, names: Sequence[str],
                    groups: Dict[str, Sequence[int]]) -> Dict[str, Tuple[int, int]]:
    """Residue and chain counts for OR-ed label groups (needs per-residue OR)."""
    res = {k: 0 for k in groups}
    chains = {k: 0 for k in groups}
    exclusive_multi = 0
    total_res = 0
    for name in names:
        entry = testset.load_entry(cache_dir, name)
        if entry is None:
            continue
        y4 = entry["y4"] > 0
        total_res += y4.shape[0]
        exclusive_multi += int((y4.sum(axis=1) > 1).sum())
        for key, idxs in groups.items():
            hit = y4[:, list(idxs)].any(axis=1)
            n = int(hit.sum())
            res[key] += n
            chains[key] += int(n > 0)
    return {k: (res[k], chains[k]) for k in groups} | {
        "__multi__": (exclusive_multi, 0), "__total__": (total_res, 0)}


def build_tables(cache_dir: str, names: Sequence[str]) -> Tuple[pd.DataFrame, dict]:
    per_chain, sizes, n_chains = collect(cache_dir, names)
    total_res = int(sizes.sum())

    rows = []
    for i, col in enumerate(BASE_LABEL_COLUMNS):
        n_pos = int(per_chain[:, i].sum())
        n_ch = int((per_chain[:, i] > 0).sum())
        rows.append({
            "label": PRETTY[col],
            "column": col,
            "kind": "base",
            "residues": n_pos,
            "residue_pct": 100 * n_pos / total_res if total_res else np.nan,
            "chains": n_ch,
            "chain_pct": 100 * n_ch / n_chains if n_chains else np.nan,
            "pos_weight": (total_res - n_pos) / max(1, n_pos),
        })

    grouped = _group_positive(cache_dir, names, DERIVED)
    for key in DERIVED:
        n_pos, n_ch = grouped[key]
        rows.append({
            "label": key,
            "column": "|".join(BASE_LABEL_COLUMNS[i] for i in DERIVED[key]),
            "kind": "derived",
            "residues": n_pos,
            "residue_pct": 100 * n_pos / total_res if total_res else np.nan,
            "chains": n_ch,
            "chain_pct": 100 * n_ch / n_chains if n_chains else np.nan,
            "pos_weight": (total_res - n_pos) / max(1, n_pos),
        })

    n_any = grouped["Any binding"][0]
    non_binding = total_res - n_any
    rows.append({
        "label": "Non-binding", "column": "(all four zero)", "kind": "derived",
        "residues": non_binding,
        "residue_pct": 100 * non_binding / total_res if total_res else np.nan,
        "chains": n_chains - grouped["Any binding"][1],
        "chain_pct": 100 * (n_chains - grouped["Any binding"][1]) / n_chains
        if n_chains else np.nan,
        "pos_weight": np.nan,
    })

    summary = {
        "chains": n_chains,
        "residues": total_res,
        "residues_per_chain_mean": float(sizes.mean()) if n_chains else np.nan,
        "residues_per_chain_median": float(np.median(sizes)) if n_chains else np.nan,
        "multilabel_residues": grouped["__multi__"][0],
        "multilabel_pct_of_binding": (100 * grouped["__multi__"][0] / n_any
                                      if n_any else np.nan),
    }
    return pd.DataFrame(rows), summary


def build_setup_table(cache_dir: str, names: Sequence[str]) -> pd.DataFrame:
    """Per-setup composition over the chains that setup actually scores."""
    rows = []
    for sname, setup in SETUPS.items():
        idxs = sorted({i for g in setup.base_index_groups for i in g})
        qualifying, total_res = [], 0
        label_pos = np.zeros(setup.num_labels, dtype=np.int64)
        for name in names:
            entry = testset.load_entry(cache_dir, name)
            if entry is None:
                continue
            y4 = entry["y4"] > 0
            if not y4[:, idxs].any():
                continue
            qualifying.append(name)
            total_res += y4.shape[0]
            for j, group in enumerate(setup.base_index_groups):
                label_pos[j] += int(y4[:, list(group)].any(axis=1).sum())
        for j, label in enumerate(setup.label_names):
            rows.append({
                "setup": sname,
                "label": label,
                "qualifying_chains": len(qualifying),
                "residues": total_res,
                "positives": int(label_pos[j]),
                "positive_pct": 100 * label_pos[j] / total_res if total_res else np.nan,
                "pos_weight": (total_res - label_pos[j]) / max(1, int(label_pos[j])),
            })
    return pd.DataFrame(rows)


def format_report(name: str, table: pd.DataFrame, summary: dict,
                  setup_table: Optional[pd.DataFrame]) -> str:
    out = ["=" * 78, f"LABEL DISTRIBUTION — {name}", "=" * 78]
    out.append(f"chains    {summary['chains']:>12,}")
    out.append(f"residues  {summary['residues']:>12,}   (surface residues, sas_area > 0)")
    out.append(f"          {summary['residues_per_chain_mean']:>12.1f}   mean per chain "
               f"(median {summary['residues_per_chain_median']:.0f})")
    out.append("")
    out.append(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out.append("")
    out.append(f"Residues carrying more than one label: "
               f"{summary['multilabel_residues']:,} "
               f"({summary['multilabel_pct_of_binding']:.2f}% of binding residues)")
    if setup_table is not None and not setup_table.empty:
        out.append("")
        out.append("-" * 78)
        out.append("PER SETUP — over the chains that setup scores")
        out.append("-" * 78)
        out.append(setup_table.to_string(index=False,
                                         float_format=lambda v: f"{v:.4f}"))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Label counts and percentages for a dataset")
    ap.add_argument("--cache-dir", default="all_class/training/test_cache")
    ap.add_argument("--out", default="all_class/training/test_results",
                    help="directory for label_distribution.csv / .txt")
    ap.add_argument("--name", default="held-out test set")
    ap.add_argument("--no-setups", action="store_true")
    final_test.add_cluster_arguments(ap)   # same subset final_test.py scores
    args = ap.parse_args()

    cache_dir = args.cache_dir if os.path.isdir(args.cache_dir) else \
        os.path.join(REPO_ROOT, args.cache_dir)
    out_dir = args.out if os.path.isdir(args.out) else os.path.join(REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(f[:-3] for _, _, fs in os.walk(cache_dir)
                   for f in fs if f.endswith(".pt"))
    if not names:
        raise SystemExit(f"No cached chains under {cache_dir}")

    names = final_test.filter_by_cluster(names, args.cluster_level, args.train_csv,
                                         args.test_csv, args.keep_unmapped)
    if args.cluster_level != "none":
        args.name = f"{args.name} (sequence_cluster_{args.cluster_level} filtered)"

    table, summary = build_tables(cache_dir, names)
    setup_table = None if args.no_setups else build_setup_table(cache_dir, names)

    report = format_report(args.name, table, summary, setup_table)
    print(report)

    table.to_csv(os.path.join(out_dir, "label_distribution.csv"), index=False)
    print(f"\nwrote {os.path.join(out_dir, 'label_distribution.csv')}")
    if setup_table is not None:
        setup_table.to_csv(os.path.join(out_dir, "label_distribution_by_setup.csv"),
                           index=False)
        print(f"wrote {os.path.join(out_dir, 'label_distribution_by_setup.csv')}")
    path = os.path.join(out_dir, "label_distribution.txt")
    with open(path, "w") as fh:
        fh.write(report + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
