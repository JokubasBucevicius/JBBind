#!/usr/bin/env python3
"""Tier 3b — how much does serving from RCSB differ from what the models were trained on?

The training features were computed on PPI3D interface-coordinate files. JBBind serves
from the RCSB asymmetric unit. Same chain, different file: different atom subset, different
occupancies, sometimes a different crystal form. Because ``create_pyg_graph`` min-max
normalises every feature **per graph**, a change to the atom set shifts the inputs for
residues that did not change at all.

No unit test can catch this — the code is identical either way. The only honest answer is
to measure it, so this runs both provenances through the same model and reports the
distribution of the difference.

Usage:
    export PATH="$PATH:/home/jokubasb/voronota_1.29.4781/expansion_js"
    python scripts/verify_provenance_shift.py --n 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jbbind.core.cache import CacheSet                                     # noqa: E402
from jbbind.core.esm.embedder import EsmEmbedder                           # noqa: E402
from jbbind.core.features.build import build_chain_graph                   # noqa: E402
from jbbind.core.nn.registry import ModelRegistry                          # noqa: E402
from jbbind.core.pipeline import Pipeline                                  # noqa: E402
from jbbind.core.structure.normalize import NormalizationError             # noqa: E402
from jbbind.settings import Settings, UserSettings                         # noqa: E402

REPO = Path("/home/jokubasb/protein_protein")
GRAPHS = REPO / "all_class" / "graphs" / "unified" / "unified"
EMB = REPO / "emb"

FEATURES = ["sas_area", "voromqa_sas_energy", "ev14", "ev28", "ev56"]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--setup", default="protein_nucleic")
    ap.add_argument("--arch", default="gnn_mlp")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Settings()
    caches = CacheSet(cfg.cache_dir)
    registry = ModelRegistry(cfg.models_dir, cfg.device)
    embedder = EsmEmbedder(cfg.device, cache=caches.esm)
    pipeline = Pipeline(cfg, registry, embedder, caches)
    model = registry.get(args.setup, args.arch)
    user = UserSettings(arch=args.arch)

    chains = [p.name for p in GRAPHS.iterdir()
              if (p / "graph_nodes.csv").exists() and (EMB / f"{p.name}.pt").exists()]
    random.seed(args.seed)
    random.shuffle(chains)

    import torch

    rows = []
    checked = 0
    for chain in chains:
        if checked >= args.n:
            break
        pdb_id, _, _ = chain.rpartition("_")

        # --- training provenance: the exact CSVs the model was trained on
        try:
            nodes = pd.read_csv(GRAPHS / chain / "graph_nodes.csv")
            links_path = GRAPHS / chain / "graph_links.csv"
            links = pd.read_csv(links_path) if links_path.exists() else None
            d = torch.load(EMB / f"{chain}.pt", map_location="cpu", weights_only=False)
            esm_train = d["representations"][sorted(d["representations"])[-1]]
            cg_train = build_chain_graph(nodes, links, esm_train)
            if cg_train is None:
                continue
            p_train = model.predict(cg_train.graph, cg_train.esm, cfg.device).numpy()
        except Exception as exc:
            rows.append({"chain": chain, "status": f"training-side error: {exc}"[:120]})
            continue

        # --- serving provenance: the same chain, fetched and prepared by JBBind
        try:
            raw, sid, source = pipeline.load_structure(pdb_id=pdb_id)
            info, _ = pipeline.describe_structure(raw)
            train_seq = "".join(cg_train.residue_df["ID_resSeq"].astype(str))
            # Pick the RCSB chain whose residue set overlaps the training one the most;
            # PPI3D subunit labels are not RCSB chain labels.
            best = None
            train_idx = set(int(i) for i in cg_train.resseq)
            for c in info:
                try:
                    res = pipeline.predict(raw=raw, structure_id=sid, source=source,
                                           chain_id=c.chain_id, user=user,
                                           setups=[args.setup])
                except Exception:
                    continue
                idx = {r.seqres_index for r in res.residues}
                overlap = len(idx & train_idx) / max(1, len(train_idx))
                if best is None or overlap > best[0]:
                    best = (overlap, res)
            if best is None or best[0] < 0.5:
                rows.append({"chain": chain, "status": "no matching RCSB chain"})
                checked += 1
                continue
            overlap, res = best
        except NormalizationError as exc:
            rows.append({"chain": chain, "status": f"{exc.code}"})
            checked += 1
            continue
        except Exception as exc:
            rows.append({"chain": chain, "status": f"serving-side error: {exc}"[:120]})
            checked += 1
            continue

        # --- compare on the residues both provenances resolved
        serve_by_idx = {r.seqres_index: r for r in res.residues}
        train_by_idx = {int(i): k for k, i in enumerate(cg_train.resseq)}
        shared = sorted(set(serve_by_idx) & set(train_by_idx))
        if len(shared) < 10:
            rows.append({"chain": chain, "status": "too little overlap"})
            checked += 1
            continue

        pt = np.array([p_train[train_by_idx[i]][0] for i in shared])
        ps = np.array([serve_by_idx[i].probs[args.setup][0] for i in shared])

        feat = {}
        train_feats = cg_train.residue_df.set_index("ID_resSeq")
        for f in ("sas_area",):
            a = np.array([train_feats.loc[i, f] for i in shared], dtype=float)
            b = np.array([serve_by_idx[i].sas_area or np.nan for i in shared], dtype=float)
            feat[f"spearman_{f}"] = round(spearman(a, b), 4)

        rows.append({
            "chain": chain, "status": "ok", "n_shared": len(shared),
            "residue_overlap": round(overlap, 4),
            "pearson_prob": round(float(np.corrcoef(pt, ps)[0, 1]), 4),
            "max_abs_diff": round(float(np.abs(pt - ps).max()), 4),
            "mean_abs_diff": round(float(np.abs(pt - ps).mean()), 4),
            **feat,
        })
        checked += 1
        print(f"  {chain}: n={len(shared)} r={rows[-1]['pearson_prob']} "
              f"max|d|={rows[-1]['max_abs_diff']}", file=sys.stderr)

    ok = [r for r in rows if r.get("status") == "ok"]
    print("\n=== Tier 3b: provenance shift (PPI3D training files vs RCSB asymmetric unit)")
    print(f"chains compared        {len(ok)} of {len(rows)} attempted")
    if ok:
        for key, label in (("pearson_prob", "Pearson r (probabilities)"),
                           ("max_abs_diff", "max |Δprobability|"),
                           ("mean_abs_diff", "mean |Δprobability|"),
                           ("spearman_sas_area", "Spearman ρ (SASA)")):
            v = np.array([r[key] for r in ok if r.get(key) == r.get(key)], dtype=float)
            if len(v):
                print(f"{label:<28} median {np.median(v):.4f}   "
                      f"p10 {np.percentile(v, 10):.4f}   p90 {np.percentile(v, 90):.4f}")
        print("\nInterpretation: r near 1.0 and a small max|Δ| mean serving from RCSB "
              "reproduces the training-time inputs closely. A long tail means the "
              "per-graph min-max normalisation is sensitive to the atom set, and the "
              "biological-assembly setting is worth trying.")
    other = [r for r in rows if r.get("status") != "ok"]
    if other:
        from collections import Counter
        print("\nnot compared:")
        for status, n in Counter(r["status"] for r in other).most_common(10):
            print(f"  {status[:70]:<72}{n}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nfull results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
