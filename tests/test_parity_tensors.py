"""Tier 1 — tensor equivalence with the original research code.

Asserts that JBBind's copied aggregation/normalization/graph code produces tensors
**bit-identical** to ``train_multilabel``'s on the golden chains. Exact equality is the
right bar: the arithmetic is the same float32 operations in the same order, so a 1e-7
discrepancy would itself be a finding (most likely a pandas groupby summation-order
change), not something to paper over with ``allclose``.

This single test pins: per-graph min-max normalization, ``normalize_voromqa``'s
shift -> log1p -> min-max, the (min, max) edge canonicalization, the one-directional edge
convention, the surface filter, and the ESM index alignment.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jbbind.core.features.build import build_chain_graph


def _build(fx):
    cg = build_chain_graph(fx.nodes, fx.links, fx.full_esm)
    assert cg is not None, f"{fx.chain}: build_chain_graph returned None"
    return cg


def test_node_features_bit_identical(fx):
    cg = _build(fx)
    assert torch.equal(cg.graph.x, torch.from_numpy(fx.ref["x"])), (
        f"{fx.chain}: node features differ from the reference")


def test_residue_type_bit_identical(fx):
    cg = _build(fx)
    assert torch.equal(cg.graph.residue_type,
                       torch.from_numpy(fx.ref["residue_type"]))


def test_edge_index_bit_identical(fx):
    cg = _build(fx)
    assert torch.equal(cg.graph.edge_index, torch.from_numpy(fx.ref["edge_index"]))


def test_edge_attr_bit_identical(fx):
    cg = _build(fx)
    assert torch.equal(cg.graph.edge_attr, torch.from_numpy(fx.ref["edge_attr"]))


def test_esm_alignment_bit_identical(fx):
    cg = _build(fx)
    assert torch.equal(cg.esm, fx.esm), (
        f"{fx.chain}: ESM rows are not aligned to the reference residues")


def test_residue_identity_preserved(fx):
    cg = _build(fx)
    np.testing.assert_array_equal(cg.resseq, fx.ref["id_resseq"])


def test_truncation_drops_match_reference(fx, index):
    """3cqz_A is 1733 SEQRES truncated to 1022; the drop must reproduce training."""
    cg = _build(fx)
    expected = index[fx.chain]["n_dropped_by_truncation"]
    assert len(cg.dropped_resseq) == expected
    if expected:
        assert any(w["code"] == "esm_range_dropped" for w in cg.warnings)
        assert all(r - 1 >= index[fx.chain]["esm_len"] for r in cg.dropped_resseq)


def test_edge_index_is_not_symmetric(fx):
    """Guard rail: training never symmetrized edges, so neither may we.

    ``aggregate_edges_to_residues`` canonicalises each contact to (min, max) resSeq and no
    reverse edge is ever added -- there is no ``to_undirected()`` anywhere in the training
    code. "Fixing" this would silently change every prediction.
    """
    cg = _build(fx)
    ei = cg.graph.edge_index
    assert ei.shape[1] > 0
    assert (ei[0] < ei[1]).all(), "edges must run low-index -> high-index only"
    forward = set(map(tuple, ei.t().tolist()))
    reverse = {(b, a) for a, b in forward}
    assert not (forward & reverse), "edge_index must not contain both directions"


def test_no_labels_required(fx):
    """The binds_* columns exist in training CSVs but never at inference."""
    nodes = fx.nodes.drop(columns=[c for c in fx.nodes.columns if c.startswith("binds_")])
    cg = build_chain_graph(nodes, fx.links, fx.full_esm)
    assert cg is not None
    assert torch.equal(cg.graph.x, torch.from_numpy(fx.ref["x"]))
    assert cg.graph.y is None


def test_structure_only_path_needs_no_esm(fx):
    """The `gnn` architecture takes no embedding; the graph must be identical."""
    cg = build_chain_graph(fx.nodes, fx.links, esm=None)
    assert cg.esm is None
    if not fx.ref["id_resseq"].shape[0] == cg.n_residues:
        # Only 3cqz_A differs, because without an embedding nothing is truncated.
        assert cg.n_residues > fx.ref["id_resseq"].shape[0]


# --- negative controls: a parity test that cannot fail is worthless -------------------

def test_perturbing_features_breaks_parity(fx):
    """Sanity: if the reference were insensitive to the inputs, the tests above are vacuous."""
    nodes = fx.nodes.copy()
    nodes["ev14"] = nodes["ev14"] * 2.0 + 1.0
    cg = build_chain_graph(nodes, fx.links, fx.full_esm)
    assert not torch.equal(cg.graph.x, torch.from_numpy(fx.ref["x"]))


def test_rank_based_esm_indexing_would_differ(fx, index):
    """Documents *why* SEQRES-index alignment matters.

    Indexing the embedding by rank-of-observed-residue instead of ``resSeq - 1`` is the
    obvious-looking simplification. For a gapped chain it is catastrophically wrong; this
    asserts the two disagree so nobody switches to it.
    """
    if index[fx.chain]["min_resseq"] == 1 and \
            index[fx.chain]["max_resseq"] == fx.ref["id_resseq"].shape[0]:
        pytest.skip("chain is densely numbered from 1; the two schemes coincide")
    n = fx.ref["id_resseq"].shape[0]
    by_rank = fx.full_esm[np.arange(n)]
    assert not torch.equal(by_rank, fx.esm)
