"""Run the forked voronota tool and load its output.

The subprocess is a pure function of (chain PDB bytes, tool script, voronota version), so
its whole output directory is content-addressed and reused.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..cache import DiskCache, sha256_file, sha256_text

TOOL = Path(__file__).resolve().parents[3] / "tools" / "describe-receptor-chain"

#: Columns the voronota export always produces; checked as a post-condition so a silent
#: upstream change shows up here rather than as a mysterious prediction.
EXPECTED_NODE_COLUMNS = {
    "ID_chainID", "ID_resSeq", "ID_iCode", "ID_resName", "ID_name",
    "atom_index", "residue_index", "atom_type", "residue_type",
    "center_x", "center_y", "center_z", "radius", "sas_area",
    "voromqa_sas_energy", "voromqa_depth", "ev14", "ev28", "ev56", "bsite_area",
}

BASE_TIMEOUT = 120.0
PER_RESIDUE_TIMEOUT = 1.0
MAX_TIMEOUT = 900.0


class VoronotaError(Exception):
    def __init__(self, code: str, message: str, stderr: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stderr = stderr


@dataclass
class VoronotaOutput:
    nodes: pd.DataFrame
    links: Optional[pd.DataFrame]
    receptor_pdb: str
    directory: Path


def voronota_version() -> str:
    """Identifies the binary so a version bump invalidates the cache."""
    exe = shutil.which("voronota-js")
    if exe is None:
        raise VoronotaError("VoronotaMissing",
                            "'voronota-js' is not on PATH; the container must provide it")
    return sha256_file(Path(exe))[:16]


def describe_chain(pdb_text: str, chain_id: str, cache: DiskCache,
                   n_residues: int = 0) -> VoronotaOutput:
    """Tessellate one chain and return its per-atom / per-contact tables."""
    if not TOOL.exists():
        raise VoronotaError("VoronotaMissing", f"tool script not found at {TOOL}")

    key = sha256_text(pdb_text, sha256_file(TOOL), voronota_version())
    timeout = min(MAX_TIMEOUT, BASE_TIMEOUT + PER_RESIDUE_TIMEOUT * max(0, n_residues))

    def produce(tmp: Path) -> None:
        tmp.mkdir(parents=True, exist_ok=True)
        chain_pdb = tmp / "chain.pdb"
        chain_pdb.write_text(pdb_text)
        env = dict(os.environ)
        try:
            proc = subprocess.run(
                [str(TOOL), "--input-chain", str(chain_pdb), "--chain-id", chain_id,
                 "--output-dir", str(tmp), "--no-faspr"],
                capture_output=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            raise VoronotaError(
                "VoronotaTimeout",
                f"tessellation exceeded {timeout:.0f}s for a {n_residues}-residue chain"
            ) from exc
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-800:]
            raise VoronotaError("VoronotaFailed",
                                "voronota could not describe this chain", stderr=tail)

    directory = cache.get_or_create(key, produce)

    nodes_path = directory / "graph_nodes.csv"
    if not nodes_path.exists() or nodes_path.stat().st_size == 0:
        raise VoronotaError("VoronotaFailed", "voronota produced no atom table")
    nodes = pd.read_csv(nodes_path)

    missing = EXPECTED_NODE_COLUMNS - set(nodes.columns)
    if missing:
        raise VoronotaError(
            "VoronotaFailed",
            f"voronota output is missing expected columns: {sorted(missing)}")
    if nodes["residue_type"].isna().any():
        raise VoronotaError(
            "VoronotaFailed",
            "some atoms have no residue_type — a non-standard residue reached voronota")
    bad = nodes.loc[(nodes["residue_type"] < 0) | (nodes["residue_type"] > 19), "ID_resName"]
    if len(bad):
        raise VoronotaError(
            "VoronotaFailed",
            f"residue_type outside [0,19] for residues: {sorted(set(bad))[:5]}")

    links_path = directory / "graph_links.csv"
    links = None
    if links_path.exists() and links_path.stat().st_size > 0:
        links = pd.read_csv(links_path)

    receptor_path = directory / "receptor.pdb"
    receptor = receptor_path.read_text() if receptor_path.exists() else ""

    return VoronotaOutput(nodes=nodes, links=links, receptor_pdb=receptor,
                          directory=directory)
