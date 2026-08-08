#!/usr/bin/env python3
"""Tier 3c — does JBBind's SEQRES renumbering reproduce the training numbering?

This is the single most important verification number in the project. Everything
downstream assumes ``ID_resSeq`` is a 1-based index into the chain's SEQRES sequence,
because that is what the training data had and what ``train_multilabel.py:329``'s
``embedding_indices = ID_resSeq - 1`` relies on. If JBBind's renumbering disagrees, the
app silently pairs residues with the wrong ESM embedding — no error, just worse answers.

For each sampled training chain this:
  1. reads the training graph_nodes.csv to get the residue indices the model was trained on
  2. checks whether the TRAINING data itself satisfies
     ``one_letter(ID_resName) == s1_sequence[ID_resSeq - 1]``
  3. fetches the entry from RCSB and runs JBBind's Stage 1 only (gemmi; no voronota,
     no ESM, no GPU)
  4. compares JBBind's SEQRES indices against the training ones

Separating (2) from (4) matters: a sample of 250 training chains found ~5.6% where the
training numbering did not index its own sequence, so those chains trained against
effectively random embeddings. Those are pre-existing data defects, not app bugs, and the
report keeps the two categories apart.

Usage:
    python scripts/check_renumbering.py --n 200 --workers 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jbbind.core.cache import DiskCache                                    # noqa: E402
from jbbind.core.structure.fetch import fetch_pdb                          # noqa: E402
from jbbind.core.structure.modres import one_letter                        # noqa: E402
from jbbind.core.structure.normalize import (NormalizationError,           # noqa: E402
                                             list_chains, prepare_chain,
                                             read_structure)


def _identity(train: dict[int, str], app: dict[int, str]) -> float:
    """Fraction of shared SEQRES indices where the residue identity agrees."""
    shared = set(train) & set(app)
    if not shared:
        return 0.0
    return sum(1 for i in shared if train[i] == app[i]) / len(shared)

REPO = Path("/home/jokubasb/protein_protein")
GRAPHS = REPO / "all_class" / "graphs" / "unified" / "unified"
CSV = REPO / "ppi3d_filtered.csv"


def load_sequences() -> dict[str, str]:
    seqs: dict[str, str] = {}
    for chunk in pd.read_csv(CSV, usecols=["subunit_1", "s1_sequence",
                                           "subunit_2", "s2_sequence"], chunksize=20000):
        for name_col, seq_col in (("subunit_1", "s1_sequence"),
                                  ("subunit_2", "s2_sequence")):
            for name, seq in zip(chunk[name_col], chunk[seq_col]):
                if isinstance(seq, str):
                    seqs.setdefault(name, seq.upper())
    return seqs


def training_residues(chain: str) -> dict[int, str]:
    """SEQRES index -> one-letter code, as the training graph recorded it."""
    df = pd.read_csv(GRAPHS / chain / "graph_nodes.csv",
                     usecols=["ID_resSeq", "ID_resName"])
    g = df.groupby("ID_resSeq")["ID_resName"].first()
    return {int(k): one_letter(v) for k, v in g.items()}


def check_one(chain: str, sequences: dict[str, str], cache: DiskCache) -> dict:
    pdb_id, _, chain_letter = chain.rpartition("_")
    out = {"chain": chain, "pdb_id": pdb_id}

    try:
        train = training_residues(chain)
    except Exception as exc:
        return {**out, "verdict": "skip", "reason": f"training graph unreadable: {exc}"}

    seq = sequences.get(chain)
    if not seq:
        return {**out, "verdict": "skip", "reason": "no sequence in ppi3d_filtered.csv"}

    # (2) Is the TRAINING numbering itself a valid index into its own sequence?
    in_range = {i: aa for i, aa in train.items() if 1 <= i <= len(seq)}
    agree = sum(1 for i, aa in in_range.items() if seq[i - 1] == aa)
    train_frac = agree / max(1, len(in_range))
    training_sound = train_frac >= 0.95

    try:
        raw = fetch_pdb(pdb_id, cache)
        prepared = prepare_chain(read_structure(raw), chain_letter)
    except NormalizationError as exc:
        return {**out, "verdict": "app_error", "code": exc.code, "reason": exc.message,
                "training_sound": training_sound}
    except Exception as exc:
        return {**out, "verdict": "fetch_error", "reason": str(exc)[:160],
                "training_sound": training_sound}

    app = {r.seqres_index: r.one_letter for r in prepared.residues}
    remapped_from = None

    # PPI3D subunit labels are not RCSB chain labels -- PPI3D renames chains, so its
    # "7qhp_B" is RCSB chain T and its "5ikl_F" is RCSB chain B. Comparing label-to-label
    # would score those as misalignments when the app is in fact correct. Resolve by
    # sequence before declaring a mismatch.
    train_seq = "".join(train[i] for i in sorted(train))
    if _identity(train, app) < 0.99:
        try:
            chains, _ = list_chains(read_structure(raw))
        except Exception:
            chains = []
        for candidate in chains:
            if candidate.chain_id == chain_letter:
                continue
            try:
                alt = prepare_chain(read_structure(raw), candidate.chain_id)
            except Exception:
                continue
            alt_map = {r.seqres_index: r.one_letter for r in alt.residues}
            alt_seq = "".join(alt_map[i] for i in sorted(alt_map))
            if train_seq[:40] and (train_seq[:40] in alt_seq or alt_seq[:40] in train_seq):
                if _identity(train, alt_map) > _identity(train, app):
                    prepared, app, remapped_from = alt, alt_map, candidate.chain_id
                break

    # (4) Do JBBind's indices land on the same residues as training's?
    shared = set(train) & set(app)
    matched = sum(1 for i in shared if train[i] == app[i])
    coverage = len(shared) / max(1, len(train))
    identity = matched / max(1, len(shared))

    if training_sound:
        if identity >= 0.99 and coverage >= 0.9:
            verdict = "match_after_chain_remap" if remapped_from else "match"
        elif identity >= 0.99:
            # Every shared residue agrees; the app simply resolves fewer of them (RCSB
            # asymmetric unit vs PPI3D biounit, altloc/modres pruning). Not a
            # misalignment -- no residue is paired with the wrong embedding.
            verdict = "match_partial_coverage"
        else:
            verdict = "mismatch"
    else:
        # Training numbering was already wrong for this chain; JBBind disagreeing with it
        # is expected and is not an app defect.
        verdict = "training_defect"

    return {**out, "verdict": verdict, "training_sound": training_sound,
            "remapped_from": remapped_from,
            "training_selfconsistency": round(train_frac, 4),
            "n_training_residues": len(train), "n_app_residues": len(app),
            "coverage": round(coverage, 4), "identity": round(identity, 4),
            "numbering_source": prepared.numbering_source}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("loading sequences from ppi3d_filtered.csv …", file=sys.stderr)
    sequences = load_sequences()

    chains = [p.name for p in GRAPHS.iterdir() if (p / "graph_nodes.csv").exists()]
    random.seed(args.seed)
    random.shuffle(chains)
    sample = [c for c in chains if c in sequences][:args.n]
    print(f"checking {len(sample)} chains with {args.workers} workers …", file=sys.stderr)

    cache_root = Path("/tmp/claude-503000028/-home-jokubasb/"
                      "465a4035-0d13-4c9b-9c02-260b85ea9158/scratchpad/jbcache")
    cache = DiskCache(cache_root, "rcsb")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_one, c, sequences, cache): c for c in sample}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"chain": futures[fut], "verdict": "crash",
                                "reason": str(exc)[:160]})
            if i % 25 == 0:
                print(f"  {i}/{len(sample)}", file=sys.stderr)

    counts = Counter(r["verdict"] for r in results)
    total = len(results)
    good = (counts["match"] + counts["match_after_chain_remap"]
            + counts["match_partial_coverage"])
    comparable = good + counts["mismatch"]

    print("\n=== Tier 3c: SEQRES renumbering audit")
    print(f"chains sampled                  {total}")
    print(f"  training numbering sound      {sum(1 for r in results if r.get('training_sound'))}"
          f"  ({counts['training_defect']} pre-existing training defects excluded)")
    for verdict in ("match", "match_after_chain_remap", "match_partial_coverage",
                    "mismatch", "training_defect", "app_error",
                    "fetch_error", "skip", "crash"):
        if counts[verdict]:
            print(f"  {verdict:<28}{counts[verdict]}")
    if comparable:
        print(f"\nAGREEMENT ON SOUND CHAINS       {good}/{comparable} "
              f"= {100*good/comparable:.1f}%")
        if counts["match_after_chain_remap"]:
            print(f"  (of which {counts['match_after_chain_remap']} needed chain "
                  f"re-resolution: PPI3D subunit labels are not RCSB chain labels)")

    mism = [r for r in results if r["verdict"] == "mismatch"]
    if mism:
        print("\nmismatches (each is a chain the app would mis-embed):")
        for r in mism[:15]:
            print(f"  {r['chain']:<12} coverage={r['coverage']} identity={r['identity']} "
                  f"src={r.get('numbering_source')}")

    errs = Counter(r.get("code") or r["verdict"] for r in results
                   if r["verdict"] in ("app_error", "fetch_error"))
    if errs:
        print("\nchains the app declined to process:")
        for code, n in errs.most_common():
            print(f"  {code:<28}{n}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nfull results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
