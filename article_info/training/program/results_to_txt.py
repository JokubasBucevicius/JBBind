"""Dump selected benchmarks (+ dataset sizes / class imbalance) to a txt.

Per setup it writes, from the slurm ``.out`` logs:
  * qualifying / split protein counts and the reweight mode;
  * training-set class imbalance (per-residue positive counts and %);
and then, per architecture (values only, tab-separated):
  * PR AUC (macro) + PR AUC per class, ROC AUC per class, F1 (macro) + F1 per class.

Usage (from repo root)::

    python all_class/training/program/results_to_txt.py
    python all_class/training/program/results_to_txt.py --runs-dir <dir> --logs-dir <dir> --out <file>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

from config import Config
from setups import SETUPS

ARCH_ORDER = ["mlp", "gnn", "gnn_mlp", "joint"]

_RE_QUAL = re.compile(r"Qualifying for setup '.*?': (\d+)/(\d+) chains")
_RE_SPLIT = re.compile(r"Split:\s*(\d+) train / (\d+) val / (\d+) test")
_RE_REWEIGHT = re.compile(r"Reweight='([^']+)'")
_RE_DIST = re.compile(r"Training label distribution \(per residue, n=(\d+)\):")
_RE_CLASS = re.compile(r"^\s{2}(.+?):\s+(\d+)\s+\(([\d.]+)%\)\s*$")


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "NA"


def _parse_log(logs_dir: str, setup: str) -> dict | None:
    """Extract protein counts + first training class distribution for a setup.

    Prefers the setup's ``mlp`` job log (a single, canonical distribution block).
    """
    cands = sorted(glob.glob(os.path.join(logs_dir, f"{setup}_mlp_*.out")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(logs_dir, f"{setup}_*.out")))
    if not cands:
        return None

    with open(cands[-1], errors="replace") as fh:
        lines = fh.readlines()

    info: dict = {"log": os.path.basename(cands[-1]), "classes": []}
    in_dist = False
    for line in lines:
        if (m := _RE_QUAL.search(line)):
            info["qualifying"], info["total"] = m.group(1), m.group(2)
        elif (m := _RE_SPLIT.search(line)):
            info["train"], info["val"], info["test"] = m.groups()
        elif (m := _RE_REWEIGHT.search(line)):
            info["reweight"] = m.group(1)
        elif (m := _RE_DIST.search(line)):
            if not info["classes"]:      # capture only the first block
                info["n_train_residues"] = m.group(1)
                in_dist = True
        elif in_dist:
            m = _RE_CLASS.match(line)
            if m:
                info["classes"].append((m.group(1), m.group(2), m.group(3)))
            else:
                in_dist = False
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=Config().runs_dir)
    ap.add_argument("--logs-dir", default="all_class/logs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs_dir, "results.txt")

    lines = []
    for sname in SETUPS:
        setup_dir = os.path.join(args.runs_dir, sname)
        data, labels = {}, None
        for arch in ARCH_ORDER:
            path = os.path.join(setup_dir, arch, "metrics.json")
            if os.path.isfile(path):
                with open(path) as fh:
                    data[arch] = json.load(fh)
                labels = data[arch]["label_names"]
        if not data:
            continue

        info = _parse_log(args.logs_dir, sname)
        if info:
            counts = (f"qualifying={info.get('qualifying', 'NA')}/{info.get('total', 'NA')}  "
                      f"train={info.get('train', 'NA')} val={info.get('val', 'NA')} "
                      f"test={info.get('test', 'NA')}  reweight={info.get('reweight', 'NA')}")
            lines.append(f"setup={sname}  {counts}")
            if info["classes"]:
                imb = "  ".join(f"{n}={c}({p}%)" for n, c, p in info["classes"])
                lines.append(f"train_residues={info.get('n_train_residues', 'NA')}  {imb}")
        else:
            lines.append(f"setup={sname}  (no log found in {args.logs_dir})")

        header = (["setup", "arch", "PR_macro"]
                  + [f"PR_{l}" for l in labels]
                  + [f"ROC_{l}" for l in labels]
                  + ["F1_macro"]
                  + [f"F1_{l}" for l in labels])
        lines.append("\t".join(header))
        for arch in ARCH_ORDER:
            if arch not in data:
                continue
            m = data[arch]["metrics"]
            row = [sname, arch, _fmt(m.get("PR AUC (macro)"))]
            row += [_fmt(m.get(f"PR AUC ({l})")) for l in labels]
            row += [_fmt(m.get(f"ROC AUC ({l})")) for l in labels]
            row.append(_fmt(m.get("F1 (macro)")))
            row += [_fmt(m.get(f"F1 ({l})")) for l in labels]
            lines.append("\t".join(row))
        lines.append("")

    with open(out, "w") as fh:
        fh.write("\n".join(lines).rstrip("\n") + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
