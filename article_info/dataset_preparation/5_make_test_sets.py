"""Write the cluster-<level> non-redundant slice of the test csv.

Reuses final_test.py's filter verbatim, so the rows written describe exactly the
chains the models were scored on -- no second, subtly different definition of
"non-redundant" floating around the repo.

The filter is homology decontamination, not deduplication: a test chain is dropped
when its PPI3D sequence cluster at ``level`` also occurs anywhere in the training
csv. Coarser clusters catch more distant homology, so the kept sets nest --
c40 (8,609) < c70 (11,369) < c95 (13,567) < unfiltered (21,516). c40 is the
strictest generalisation estimate; c95 still removes near-duplicates of training
chains but tolerates remote homology.

    python baselines/make_cluster_slice.py --level 95
"""

import argparse
import os
import sys

import pandas as pd

PROG = "/home/jokubasb/protein_protein/all_class/training/program"
sys.path.insert(0, PROG)

import final_test  # noqa: E402
from config import REPO_ROOT, Config  # noqa: E402

TEST_CSV = "all_class/ppi3d_test_data_filtered.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", required=True, choices=final_test.CLUSTER_LEVELS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    level = args.level
    out_csv = args.out or os.path.join(
        REPO_ROOT, f"all_class/ppi3d_test_data_filtered_c{level}.csv")

    cfg = Config()
    emb_dir = (cfg.emb_dir if os.path.isdir(cfg.emb_dir)
               else os.path.join(REPO_ROOT, cfg.emb_dir))
    graph_dir = final_test.resolve(final_test.DEFAULT_TEST_GRAPHS)

    names = final_test.discover_chains(graph_dir, emb_dir, None)
    kept = set(final_test.filter_by_cluster(names, level, test_csv=TEST_CSV))

    df = pd.read_csv(final_test.resolve(TEST_CSV), dtype=str, low_memory=False)
    s1_keep = df["subunit_1"].isin(kept)
    s2_keep = df["subunit_2"].isin(kept)

    out = df[s1_keep | s2_keep].copy()
    # Both sides carry cluster ids, so a row can survive on one subunit only; flag
    # which, otherwise a downstream consumer cannot tell a kept chain from its
    # still-redundant partner.
    both = s1_keep | s2_keep
    out.insert(len(out.columns), f"s1_nonredundant_{level}", s1_keep[both].astype(int))
    out.insert(len(out.columns), f"s2_nonredundant_{level}", s2_keep[both].astype(int))
    out.to_csv(out_csv, index=False)

    covered = (set(out["subunit_1"][out[f"s1_nonredundant_{level}"] == 1]) |
               set(out["subunit_2"][out[f"s2_nonredundant_{level}"] == 1]))
    print()
    print(f"rows in  {len(df):,}")
    print(f"rows out {len(out):,}   ({100 * len(out) / len(df):.1f}%)")
    print(f"  both subunits non-redundant {int((s1_keep & s2_keep).sum()):,}")
    print(f"  one side only               {int((s1_keep ^ s2_keep).sum()):,}")
    print(f"non-redundant chains kept {len(kept):,}, covered by these rows {len(covered):,}")
    assert covered == kept, f"{len(kept - covered)} kept chains have no row"
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
