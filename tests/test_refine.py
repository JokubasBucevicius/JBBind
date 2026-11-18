"""Neighbourhood refinement (core/refine.py) and how the pipeline applies it.

Toy graphs pin the arithmetic; the golden chains check that the contacts the pipeline
builds are the Voronoi graph the GNN sees, and that refinement never touches the raw
scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from jbbind.core.features.aggregate import load_chain_base
from jbbind.core.features.build import build_chain_graph
from jbbind.core.pipeline import Pipeline
from jbbind.core.refine import ContactGraph, refine
from jbbind.settings import UserSettings, UserSettingsStore

# 1 - 2 - 3 form a patch; 4 touches nothing; the (4, 9) contact names a residue that
# was not predicted and must be dropped.
GRAPH = ContactGraph.from_edges([1, 2, 3, 4], [1, 1, 2, 4], [2, 3, 3, 9], [10, 20, 5, 7])
P = np.array([[0.2, 0.9], [0.9, 0.1], [0.8, 0.1], [0.5, 0.5]])


def test_edges_are_symmetrised_and_foreign_contacts_dropped():
    assert len(GRAPH.src) == 6
    assert GRAPH.neighbour_lists() == [[3, 2], [1, 3], [1, 2], []]


@pytest.mark.parametrize("method", ["none", "bayes", "gumbel"])
def test_off_or_zero_strength_is_the_identity(method):
    strength = 0.0 if method != "none" else 0.5
    out = refine(P, GRAPH, method, strength=strength)
    np.testing.assert_array_equal(out.probs, P)


@pytest.mark.parametrize("method", ["bayes", "gumbel"])
def test_an_isolated_residue_keeps_its_score(method):
    out = refine(P, GRAPH, method, strength=1.0)
    np.testing.assert_allclose(out.probs[3], P[3], atol=1e-12)


def test_bayes_lifts_a_low_residue_among_high_contacts_monotonically_in_beta():
    lifted = [refine(P, GRAPH, "bayes", strength=b).probs[0, 0] for b in (0.25, 0.5, 1.0)]
    assert P[0, 0] < lifted[0] < lifted[1] < lifted[2] < 1


def test_labels_are_refined_independently():
    both = refine(P, GRAPH, "bayes", strength=0.5).probs
    alone = refine(P[:, :1], GRAPH, "bayes", strength=0.5).probs
    np.testing.assert_allclose(both[:, 0], alone[:, 0])


def test_gumbel_is_seeded_bounded_and_reports_a_spread():
    a = refine(P, GRAPH, "gumbel", strength=0.5, tau=0.5, samples=64)
    b = refine(P, GRAPH, "gumbel", strength=0.5, tau=0.5, samples=64)
    np.testing.assert_array_equal(a.probs, b.probs)
    assert np.all((a.probs >= 0) & (a.probs <= 1))
    assert np.all(a.sd >= 0) and a.sd[:3].max() > 0 and a.sd[3].max() == 0


def test_hot_gumbel_tends_to_the_plain_shell_mean():
    """τ divides prior, odds and noise alike, so a hot softmax flattens to uniform."""
    out = refine(P, GRAPH, "gumbel", strength=1.0, tau=1e3, samples=64)
    np.testing.assert_allclose(out.probs[0], P[:3].mean(axis=0), atol=0.01)


def test_cold_gumbel_follows_the_likeliest_anchor():
    """τ → 0 is the Gumbel-max trick: exact draws ∝ contact area × odds."""
    out = refine(P, GRAPH, "gumbel", strength=1.0, tau=0.01, samples=1024)
    # residue 1's shell for label 0: self 0.2 (area 20), 2 at 0.9 (10), 3 at 0.8 (20)
    w = np.array([20, 10, 20]) * np.array([0.2 / 0.8, 0.9 / 0.1, 0.8 / 0.2])
    expected = (w / w.sum()) @ P[:3, 0]
    assert out.probs[0, 0] == pytest.approx(expected, abs=0.03)


def test_samples_need_not_be_a_multiple_of_the_block():
    assert refine(P, GRAPH, "gumbel", samples=37).probs.shape == P.shape


# ------------------------------------------------------------ golden chains

def test_contacts_are_the_graph_the_gnn_sees(fx):
    cg = build_chain_graph(fx.nodes, fx.links, esm=None)
    ei = cg.graph.edge_index.numpy()
    g = ContactGraph.from_edges(cg.resseq, cg.resseq[ei[0]], cg.resseq[ei[1]],
                                cg.graph.edge_attr[:, 0].numpy())
    assert len(g.src) == 2 * ei.shape[1]

    # The MLP path reads the same contacts straight from the aggregated tables.
    _, edges = load_chain_base(fx.nodes, fx.links)
    g2 = ContactGraph.from_edges(cg.resseq, edges["ID1_resSeq"], edges["ID2_resSeq"],
                                 edges["area"])
    assert g2.neighbour_lists() == g.neighbour_lists()


def _pipeline_result(fx):
    """A PredictionResult for a golden chain, built as Pipeline.predict builds one."""
    from jbbind.core.pipeline import PredictionResult, ResiduePrediction

    cg = build_chain_graph(fx.nodes, fx.links, esm=None)
    ei = cg.graph.edge_index.numpy()
    rng = np.random.default_rng(1)
    residues = [ResiduePrediction(int(i), "A", int(i), "", "ALA", "A", 1.0,
                                  {"protein_nucleic": rng.random(2).round(5).tolist()})
                for i in cg.resseq]
    contacts = ContactGraph.from_edges(cg.resseq, cg.resseq[ei[0]], cg.resseq[ei[1]],
                                       cg.graph.edge_attr[:, 0].numpy())
    return PredictionResult(
        structure_id="x", source="t", chain_id="A", arch="gnn_mlp", sequence="A",
        numbering_source="seqres",
        label_names={"protein_nucleic": ["Protein", "Nucleic acid"]},
        residues=residues, unpredicted=[], receptor_pdb="", contacts=contacts)


@pytest.mark.parametrize("method", ["bayes", "gumbel"])
def test_pipeline_refine_never_touches_the_raw_scores(fx, method):
    res = _pipeline_result(fx)
    raw = [r.probs for r in res.residues]
    out = Pipeline.refine(res, UserSettings(refine_method=method))
    assert [r.probs for r in out.residues] == raw
    assert [r.probs for r in res.residues] == raw          # the input is not mutated
    assert all(r.refined is None for r in res.residues)
    assert out.refinement["method"] == method
    assert all(len(r.refined["protein_nucleic"]) == 2 for r in out.residues)
    assert (out.residues[0].refined_sd is None) == (method == "bayes")

    off = Pipeline.refine(out, UserSettings(refine_method="none"))
    assert off.refinement is None and all(r.refined is None for r in off.residues)


def test_refining_without_contacts_says_what_to_do(fx):
    from jbbind.core.structure.normalize import NormalizationError

    res = _pipeline_result(fx)
    res.contacts = None
    with pytest.raises(NormalizationError) as err:
        Pipeline.refine(res, UserSettings(refine_method="bayes"))
    assert err.value.code == "RefinementUnavailable"


# ----------------------------------------------------------------- settings

def test_refinement_is_off_by_default():
    assert UserSettings().refine_method == "none"


@pytest.mark.parametrize("patch", [
    {"refine_method": "softmax"},
    {"refine_strength": 1.5},
    {"refine_strength": -0.1},
    {"gumbel_tau": 0.0},
    {"gumbel_tau": 50},
    {"gumbel_samples": 4},
    {"gumbel_samples": 64.5},
])
def test_bad_refine_settings_are_rejected(tmp_path, patch):
    with pytest.raises(ValueError):
        UserSettingsStore(tmp_path / "settings.json").update(patch)


def test_refine_settings_round_trip(tmp_path):
    store = UserSettingsStore(tmp_path / "settings.json")
    store.update({"refine_method": "gumbel", "gumbel_tau": 1.5, "gumbel_samples": 128})
    s = UserSettingsStore(tmp_path / "settings.json").get()
    assert (s.refine_method, s.gumbel_tau, s.gumbel_samples) == ("gumbel", 1.5, 128)
