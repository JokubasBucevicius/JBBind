#!/usr/bin/env python3
"""Tier 3a — prove the forked voronota script reproduces the training features exactly.

The original PPI3D structures are still on disk (all_class/structures/, 67k PDBs) and are
the exact inputs that produced all_class/graphs/unified/unified/<chain>/. So we can run
tools/describe-receptor-chain on those same files and diff against the training CSVs.

Expected result: every column identical except `bsite_area`, which the fork deliberately
leaves at 0 because it drops the interface/labelling step. Any other difference means one
of the ~120 deleted lines mattered.

Usage:
    export PATH="$PATH:/home/jokubasb/voronota_1.29.4781/expansion_js"
    python scripts/verify_tool_reproduction.py --n 20
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/jokubasb/protein_protein")
STRUCTURES = REPO / "all_class" / "structures"
GRAPHS = REPO / "all_class" / "graphs" / "unified" / "unified"
TOOL = Path(__file__).resolve().parents[1] / "tools" / "describe-receptor-chain"

NODE_KEY = ["ID_chainID", "ID_resSeq", "ID_iCode", "ID_resName", "ID_name"]
LINK_KEY = ["ID1_resSeq", "ID1_name", "ID2_resSeq", "ID2_name"]
#: The one column the fork is expected to differ on.
EXPECTED_DIFF = {"bsite_area"}


def find_structures(chain: str) -> list[Path]:
    """Every complex file containing this subunit.

    A chain typically appears in several complexes (3imf_B is in both 3imf_B-3imf_A.pdb and
    3imf_B-3imf_D.pdb). build_unified_graphs.py built the per-chain graph from ONE of them
    and merged only the binding *labels* across the rest, so picking the wrong file yields
    a structurally different copy of the same chain — same residues, different coordinates.
    ``pick_source`` below identifies the right one by matching coordinates.
    """
    return sorted(set(STRUCTURES.glob(f"{chain}-*.pdb")) |
                  set(STRUCTURES.glob(f"*-{chain}.pdb")))


def pick_source(candidates: list[Path], ref_nodes: pd.DataFrame,
                chain_letter: str) -> Path | None:
    """The candidate whose coordinates match the reference graph, if any."""
    want = ref_nodes.iloc[0]
    key = (str(want["ID_resSeq"]), str(want["ID_resName"]).strip(),
           str(want["ID_name"]).strip())
    target = (round(float(want["center_x"]), 3), round(float(want["center_y"]), 3),
              round(float(want["center_z"]), 3))
    for path in candidates:
        with open(path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[21] != chain_letter:
                    continue
                got = (line[22:26].strip(), line[17:20].strip(), line[12:16].strip())
                if got != key:
                    continue
                xyz = (round(float(line[30:38]), 3), round(float(line[38:46]), 3),
                       round(float(line[46:54]), 3))
                if xyz == target:
                    return path
                break
    return None


def actual_chain_letter(ref_nodes: pd.DataFrame) -> str:
    """The chain letter voronota actually saw, read from the reference CSV.

    The directory name (e.g. ``6yc7_E``) is a PPI3D *subunit* id, not the chain letter in
    the structure file -- PPI3D renames chains, so 6yc7_E is stored as chain A and 7sca_BM
    as chain B. build_unified_graphs.py resolved this at training time with
    ``resolve_chains()``; here the reference graph_nodes.csv's ID_chainID column records
    the answer directly.
    """
    return str(ref_nodes["ID_chainID"].iloc[0])


def compare(new: pd.DataFrame, ref: pd.DataFrame, key: list[str], tol: float = 1e-9):
    common = [c for c in new.columns if c in ref.columns]
    n = new.set_index(key).sort_index()
    r = ref.set_index(key).sort_index()
    if not n.index.equals(r.index):
        return None, f"row identity differs (new={len(n)}, ref={len(r)})"
    diffs = {}
    for c in common:
        if c in key:
            continue
        a, b = n[c], r[c]
        if a.dtype.kind in "fciu" and b.dtype.kind in "fciu":
            d = np.abs(pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce"))
            m = float(np.nanmax(d)) if len(d) else 0.0
            if m > tol:
                diffs[c] = m
        elif not a.equals(b):
            diffs[c] = "not-equal"
    return diffs, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    all_chains = [p.name for p in GRAPHS.iterdir() if (p / "graph_nodes.csv").exists()]
    random.seed(args.seed)
    random.shuffle(all_chains)

    checked, passed, skipped = 0, 0, 0
    failures = []

    for chain in all_chains:
        if checked >= args.n:
            break
        candidates = find_structures(chain)
        if not candidates:
            skipped += 1
            continue
        ref_nodes = pd.read_csv(GRAPHS / chain / "graph_nodes.csv")
        chain_letter = actual_chain_letter(ref_nodes)
        pdb = pick_source(candidates, ref_nodes, chain_letter)
        if pdb is None:
            # None of the on-disk complexes holds the coordinates this graph was built
            # from, so there is nothing to compare against.
            skipped += 1
            continue

        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [str(TOOL), "--input-chain", str(pdb), "--chain-id", chain_letter,
                 "--output-dir", tmp, "--no-faspr"],
                capture_output=True, timeout=600)
            if proc.returncode != 0:
                failures.append((chain, f"tool failed: {proc.stderr.decode()[-200:]}"))
                checked += 1
                continue

            new_nodes = pd.read_csv(Path(tmp) / "graph_nodes.csv")
            ndiff, err = compare(new_nodes, ref_nodes, NODE_KEY)
            if err:
                failures.append((chain, f"nodes: {err}"))
                checked += 1
                continue

            link_diff = {}
            new_links_p = Path(tmp) / "graph_links.csv"
            ref_links_p = GRAPHS / chain / "graph_links.csv"
            if new_links_p.exists() and ref_links_p.exists():
                nl = pd.read_csv(new_links_p)
                rl = pd.read_csv(ref_links_p)
                if len(nl) != len(rl):
                    link_diff["__count__"] = f"{len(nl)} vs {len(rl)}"
                else:
                    link_diff, lerr = compare(nl, rl, LINK_KEY)
                    if lerr:
                        link_diff = {"__rows__": lerr}

            unexpected = ({k: v for k, v in (ndiff or {}).items() if k not in EXPECTED_DIFF}
                          | (link_diff or {}))
            checked += 1
            if unexpected:
                failures.append((chain, f"unexpected diffs: {unexpected}"))
                print(f"  FAIL {chain}: {unexpected}")
            else:
                passed += 1
                print(f"  ok   {chain}  atoms={len(new_nodes)}  "
                      f"(bsite_area differs as expected: "
                      f"{'yes' if 'bsite_area' in (ndiff or {}) else 'no interface'})")

    print(f"\nTier 3a: {passed}/{checked} chains reproduce the training features exactly "
          f"({skipped} skipped for missing source PDB)")
    for chain, why in failures:
        print(f"  FAILED {chain}: {why}")
    print(json.dumps({"checked": checked, "passed": passed, "failed": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
