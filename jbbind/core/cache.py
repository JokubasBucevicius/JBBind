"""Content-addressed disk cache.

Every expensive stage of the pipeline is a pure function of its input, so caching is both
safe and the single biggest performance lever — especially the ESM embedding, which is
keyed by sequence hash and is therefore shared across setups, architectures, re-runs, and
different PDB entries that happen to contain the same sequence.

Writes are atomic (temp file + ``os.replace``) and guarded by a per-key lock, so N
concurrent requests for the same chain collapse into one computation instead of N.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from filelock import FileLock


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


class DiskCache:
    """One namespace of the cache, e.g. ``rcsb`` or ``esm``."""

    def __init__(self, root: Path, namespace: str, max_bytes: Optional[int] = None):
        self.root = Path(root) / namespace
        self.namespace = namespace
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks = self.root.parent / ".locks"
        self._locks.mkdir(exist_ok=True)

    def path(self, key: str, suffix: str = "") -> Path:
        # Two-level fan-out; 65k directories is plenty and keeps listings fast.
        return self.root / key[:2] / f"{key}{suffix}"

    def _lock(self, key: str) -> FileLock:
        return FileLock(str(self._locks / f"{self.namespace}-{key}.lock"), timeout=1800)

    def get_or_create(self, key: str, producer: Callable[[Path], None],
                      suffix: str = "", ttl_seconds: Optional[float] = None) -> Path:
        """Return the cached path, producing it if absent or stale.

        ``producer`` receives a temporary path to write to; the result is moved into place
        atomically only if it returns without raising.
        """
        target = self.path(key, suffix)
        if target.exists() and not self._is_stale(target, ttl_seconds):
            os.utime(target, None)  # touch for LRU
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(key):
            if target.exists() and not self._is_stale(target, ttl_seconds):
                return target
            tmp = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
            if tmp.exists():
                shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()
            try:
                producer(tmp)
                if not tmp.exists():
                    raise RuntimeError(f"producer for {self.namespace}/{key} wrote nothing")
                if target.exists():
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                os.replace(tmp, target)
            except BaseException:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True) if tmp.is_dir() else tmp.unlink()
                raise
        self._evict_if_needed()
        return target

    @staticmethod
    def _is_stale(path: Path, ttl_seconds: Optional[float]) -> bool:
        if ttl_seconds is None:
            return False
        return (time.time() - path.stat().st_mtime) > ttl_seconds

    def stats(self) -> dict:
        entries, total = 0, 0
        for p in self.root.rglob("*"):
            if p.is_file():
                entries += 1
                total += p.stat().st_size
        return {"namespace": self.namespace, "entries": entries, "bytes": total,
                "max_bytes": self.max_bytes}

    def clear(self) -> int:
        n = sum(1 for p in self.root.rglob("*") if p.is_file())
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        return n

    def _evict_if_needed(self) -> None:
        """Least-recently-used eviction down to 90% of the cap."""
        if not self.max_bytes:
            return
        items = []
        total = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                st = p.stat()
                items.append((st.st_atime, st.st_size, p))
                total += st.st_size
        if total <= self.max_bytes:
            return
        items.sort()
        for _, size, p in items:
            if total <= self.max_bytes * 0.9:
                break
            try:
                p.unlink()
                total -= size
            except OSError:
                pass


class CacheSet:
    """The four cache layers the pipeline uses."""

    def __init__(self, root: Path, esm_max_bytes: int = 20 << 30,
                 chain_max_bytes: int = 10 << 30):
        self.root = Path(root)
        self.rcsb = DiskCache(root, "rcsb")
        self.esm = DiskCache(root, "esm", max_bytes=esm_max_bytes)
        self.chain = DiskCache(root, "chain", max_bytes=chain_max_bytes)
        self.pred = DiskCache(root, "pred", max_bytes=2 << 30)

    def all_stats(self) -> list[dict]:
        return [c.stats() for c in (self.rcsb, self.esm, self.chain, self.pred)]

    def write_json(self, cache: DiskCache, key: str, obj: dict) -> Path:
        def produce(tmp: Path):
            tmp.write_text(json.dumps(obj))
        return cache.get_or_create(key, produce, suffix=".json")
