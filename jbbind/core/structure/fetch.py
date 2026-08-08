"""Fetch structures from RCSB.

Uses the **asymmetric unit** (``files.rcsb.org/download/{id}.cif.gz``) rather than a
biological assembly, for three reasons: the chain ids match what the user typed;
``label_seq_id`` is unambiguous (assembly files duplicate chains across symmetry copies);
and ``voronota_restrict_atoms`` isolates a single chain anyway, so the rest of the assembly
contributes nothing to the features. The choice is exposed as a setting because Tier 3b may
show the training data (PPI3D biounit interface coordinates) is closer to assembly 1.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from ..cache import DiskCache

RCSB_ASYM = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
RCSB_ASSEMBLY = "https://files.rcsb.org/download/{pdb_id}-assembly{n}.cif.gz"
TTL_SECONDS = 30 * 24 * 3600


class FetchError(Exception):
    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def is_pdb_id(value: str) -> bool:
    v = value.strip()
    return len(v) == 4 and v[0].isdigit() and v.isalnum()


def fetch_pdb(pdb_id: str, cache: DiskCache, *, assembly: int | None = None,
              timeout: float = 60.0) -> bytes:
    """Download (or read from cache) one entry as gzipped mmCIF."""
    pdb_id = pdb_id.strip().lower()
    if not is_pdb_id(pdb_id):
        raise FetchError("UnsupportedFormat",
                         f"{pdb_id!r} is not a 4-character PDB ID", status=422)

    url = (RCSB_ASSEMBLY.format(pdb_id=pdb_id, n=assembly) if assembly
           else RCSB_ASYM.format(pdb_id=pdb_id))
    key = f"{pdb_id}" + (f"-assembly{assembly}" if assembly else "")

    def produce(tmp: Path) -> None:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url)
        except httpx.HTTPError as exc:
            raise FetchError("RcsbUnavailable", f"could not reach RCSB: {exc}") from exc
        if resp.status_code == 404:
            raise FetchError("PdbNotFound",
                             f"RCSB has no entry {pdb_id.upper()}"
                             + (f" assembly {assembly}" if assembly else ""),
                             status=404)
        if resp.status_code != 200:
            raise FetchError("RcsbUnavailable",
                             f"RCSB returned HTTP {resp.status_code} for {pdb_id.upper()}")
        tmp.write_bytes(resp.content)

    path = cache.get_or_create(key, produce, suffix=".cif.gz", ttl_seconds=TTL_SECONDS)
    return path.read_bytes()
