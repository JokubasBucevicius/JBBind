"""Shared fixtures: the golden chains and their reference tensors.

Fixture data is produced by ``scripts/make_parity_fixtures.py`` running in the research
environment. These tests need no access to the 802 GB research tree.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

DATA = Path(__file__).parent / "data"
MODELS = Path(__file__).parents[1] / "models"

CHAINS = ["12as_A", "6oax_E", "3cqz_A"]
SETUPS = ["protein_nucleic", "homo_hetero", "protein", "dna_rna", "nucleic"]
ARCHS = ["mlp", "gnn", "gnn_mlp", "joint"]


def _read_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f)


class Fixture:
    def __init__(self, chain: str):
        self.chain = chain
        d = DATA / chain
        self.nodes = _read_csv_gz(d / "graph_nodes.csv.gz")
        self.links = _read_csv_gz(d / "graph_links.csv.gz")
        self.esm = torch.from_numpy(np.load(d / "esm_layer33.npy"))
        ref = np.load(d / "reference_graph.npz")
        self.ref = {k: ref[k] for k in ref.files}
        self.ref_probs = np.load(d / "reference_probs.npz")

    @property
    def full_esm(self) -> torch.Tensor:
        """The *canonical-sequence* embedding, reconstructed to full SEQRES length.

        The fixture stores the already-indexed slice (one row per surviving residue), which
        is what the model consumes. To exercise ``build_chain_graph``'s own alignment we
        need the pre-indexing tensor, so scatter the slice back to SEQRES positions. Rows
        for residues that are absent (buried, or unobserved) are never read by the aligner,
        so their contents do not matter.
        """
        meta = json.loads((DATA / "index.json").read_text())[self.chain]
        length = meta["esm_len"]
        full = torch.zeros((length, 1280), dtype=torch.float32)
        resseq = self.ref["id_resseq"]
        full[resseq - 1] = self.esm
        return full


@pytest.fixture(scope="session", params=CHAINS)
def fx(request) -> Fixture:
    return Fixture(request.param)


@pytest.fixture(scope="session")
def index() -> dict:
    return json.loads((DATA / "index.json").read_text())


def pytest_configure(config):
    config.addinivalue_line("markers", "parity: compares against the original research code")

    # Single-threaded, deterministic CPU math for the parity assertions.
    #
    # GATv2Conv's message passing is a scatter-add, and multi-threaded CPU reductions
    # accumulate in nondeterministic order — so the same input can differ run to run in
    # the last bits. That made test_probabilities_match_reference occasionally exceed its
    # 1e-6 tolerance. The right fix is to remove the nondeterminism, not to loosen the
    # tolerance: a parity test that passes by luck is not a parity test.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
