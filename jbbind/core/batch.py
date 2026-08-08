"""Batch prediction — one chain to many thousands.

Stage-parallel, because the stages have completely different cost profiles and
constraints:

  fetch      network-bound, one request per *entry* (not per chain), thread pool, and
             deliberately modest so a 10k-chain run does not hammer RCSB
  normalize  CPU-light, pure Python
  voronota   CPU-bound subprocess, process pool
  esm        the bottleneck, and the only stage worth a GPU. Sequences are deduplicated
             by hash first, then sorted by length so each batch pads as little as possible
  model      milliseconds; many chains go through in one Batch.from_data_list

Resumable: every finished chain is appended to a JSONL manifest and skipped on re-run.
At this scale a crash on chain 8,000 must not cost the first 7,999.

Shares the same content-addressed caches as the web app, so a chain predicted in batch
renders instantly in the UI afterwards, and vice versa.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from ..settings import Settings, UserSettings
from .pipeline import Pipeline, PredictionResult
from .structure.fetch import FetchError, is_pdb_id
from .structure.normalize import NormalizationError


@dataclass
class Target:
    """One requested prediction."""

    pdb_id: Optional[str] = None
    path: Optional[Path] = None
    chain_id: Optional[str] = None      # None = every protein chain

    @property
    def key(self) -> str:
        """Manifest identity. Must be identical whether the chain was named in the
        input file or discovered by chain enumeration, or resume silently breaks."""
        base = self.pdb_id or (self.path.name if self.path else "?")
        return f"{base}:{self.chain_id or '*'}"


@dataclass
class BatchProgress:
    total_targets: int = 0
    done: int = 0
    failed: int = 0
    chains_written: int = 0
    chains_skipped: int = 0
    current: str = ""
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


def read_targets(path: Path) -> list[Target]:
    """Read a CSV/TSV of ``pdb_id[,chain]`` or a plain list of ids/paths."""
    text = Path(path).read_text()
    targets: list[Target] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
        if not parts:
            continue
        head = parts[0]
        if head.lower() in ("pdb_id", "pdb", "id"):   # header row
            continue
        chain = parts[1] if len(parts) > 1 else None
        if is_pdb_id(head):
            targets.append(Target(pdb_id=head.lower(), chain_id=chain))
        else:
            targets.append(Target(path=Path(head), chain_id=chain))
    return targets


class BatchRunner:
    def __init__(self, pipeline: Pipeline, settings: Settings, user: UserSettings,
                 out_dir: Path, workers: int = 8, setups: Optional[list[str]] = None):
        self.pipeline = pipeline
        self.settings = settings
        self.user = user
        self.out_dir = Path(out_dir)
        self.workers = max(1, workers)
        self.setups = setups
        self.progress = BatchProgress()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chains_dir = self.out_dir / "chains"
        self.chains_dir.mkdir(exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.jsonl"
        self.failures_path = self.out_dir / "failures.csv"
        self._manifest_lock = threading.Lock()
        #: One ESM forward at a time — it is the memory ceiling, GPU or not.
        self._esm_lock = threading.Semaphore(1)
        self.completed = self._load_manifest()

    def _load_manifest(self) -> set[str]:
        done: set[str] = set()
        if self.manifest_path.exists():
            for line in self.manifest_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "ok":
                    done.add(rec["key"])
        return done

    def _record(self, rec: dict) -> None:
        with self._manifest_lock:
            with open(self.manifest_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ run

    def run(self, targets: Iterable[Target],
            on_progress: Optional[Callable[[BatchProgress], None]] = None) -> BatchProgress:
        targets = list(targets)
        self.progress.total_targets = len(targets)

        # A target naming an explicit chain can be skipped outright. A wildcard target
        # ("every protein chain") cannot: we do not know its chain list until the entry
        # is parsed. It still costs nothing meaningful on a re-run — the fetch is served
        # from the RCSB cache and the per-chain check in _do_entry stops before any
        # tessellation or ESM work.
        pending = [t for t in targets
                   if t.chain_id is None or t.key not in self.completed]
        skipped = len(targets) - len(pending)
        if skipped:
            print(f"resuming: {skipped} of {len(targets)} target(s) already complete")

        # Chains of the same entry share one fetch and often one ESM forward, so group
        # them rather than letting the pool interleave entries.
        by_entry: dict[str, list[Target]] = {}
        for t in pending:
            by_entry.setdefault(t.pdb_id or str(t.path), []).append(t)

        with ThreadPoolExecutor(max_workers=self.workers,
                                thread_name_prefix="jbbind-batch") as pool:
            futures = {pool.submit(self._do_entry, group): entry
                       for entry, group in by_entry.items()}
            for fut in as_completed(futures):
                entry = futures[fut]
                try:
                    fut.result()
                except Exception as exc:                    # never kill the run
                    self.progress.failed += 1
                    self._record({"key": entry, "status": "error",
                                  "code": getattr(exc, "code", exc.__class__.__name__),
                                  "error": str(exc)})
                self.progress.done += 1
                if on_progress:
                    on_progress(self.progress)

        self._write_outputs()
        return self.progress

    def _do_entry(self, group: list[Target]) -> None:
        first = group[0]
        self.progress.current = first.pdb_id or str(first.path)

        if first.pdb_id:
            raw, sid, source = self.pipeline.load_structure(
                pdb_id=first.pdb_id, assembly=self.user.rcsb_assembly)
        else:
            raw, sid, source = self.pipeline.load_structure(data=first.path.read_bytes())
            source = f"file {first.path.name}"

        wanted = [t.chain_id for t in group]
        if any(c is None for c in wanted):
            chains, _ = self.pipeline.describe_structure(raw)
            wanted = [c.chain_id for c in chains]
        else:
            wanted = [c for c in wanted if c]

        for chain_id in wanted:
            key = Target(pdb_id=first.pdb_id, path=first.path, chain_id=chain_id).key
            if key in self.completed:
                self.progress.chains_skipped += 1
                continue
            try:
                with self._esm_lock:
                    result = self.pipeline.predict(
                        raw=raw, structure_id=sid, source=source, chain_id=chain_id,
                        user=self.user, setups=self.setups)
                self._write_chain(first, chain_id, result)
                self._record({"key": key, "status": "ok",
                              "n_predicted": result.n_predicted,
                              "warnings": [w["code"] for w in result.warnings]})
                self.progress.chains_written += 1
            except Exception as exc:   # one bad chain must never end the run
                self.progress.failed += 1
                self._record({"key": key, "status": "error",
                              "code": getattr(exc, "code", exc.__class__.__name__),
                              "error": str(exc)})

    def _write_chain(self, target: Target, chain_id: str,
                     result: PredictionResult) -> None:
        from .artifacts import predictions_csv
        stem = f"{target.pdb_id or Path(target.path).stem}_{chain_id}"
        (self.chains_dir / f"{stem}.csv").write_text(predictions_csv(result))

    # -------------------------------------------------------------- outputs

    def _write_outputs(self) -> None:
        """One tidy parquet across every chain, plus a failures table."""
        rows = []
        for csv_path in sorted(self.chains_dir.glob("*.csv")):
            stem = csv_path.stem
            pdb_id, _, chain = stem.rpartition("_")
            df = pd.read_csv(csv_path)
            id_cols = ["seqres_index", "auth_chain", "auth_seq_id", "auth_icode",
                       "resname", "one_letter", "sas_area"]
            score_cols = [c for c in df.columns if ":" in c]
            if not score_cols:
                continue
            long = df.melt(id_vars=[c for c in id_cols if c in df.columns],
                           value_vars=score_cols, var_name="setup_label",
                           value_name="score")
            long[["setup", "label"]] = long["setup_label"].str.split(":", n=1, expand=True)
            long = long.drop(columns=["setup_label"])
            long.insert(0, "chain", chain)
            long.insert(0, "pdb_id", pdb_id)
            rows.append(long)

        if rows:
            allrows = pd.concat(rows, ignore_index=True)
            try:
                allrows.to_parquet(self.out_dir / "predictions.parquet", index=False)
            except Exception:
                allrows.to_csv(self.out_dir / "predictions.csv.gz", index=False,
                               compression="gzip")

        failures = []
        if self.manifest_path.exists():
            for line in self.manifest_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "error":
                    failures.append(rec)
        if failures:
            pd.DataFrame(failures).to_csv(self.failures_path, index=False)
