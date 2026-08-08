"""Tier 2 — model-output equivalence for all 5 setups x 4 architectures.

Runs each exported checkpoint through JBBind's registry and compares against probabilities
computed by the ORIGINAL ``train_multilabel`` classes on the same graph.

This is what catches the failure modes that a tensor test cannot: a wrong reconstruction
config for ``gnn``/``gnn_mlp`` (their checkpoints carry no ``model_config``), a missing
``eval()``, returning ``mlp_logits`` instead of ``gnn_logits`` for ``joint``, and above all
swapping the node-feature contract between ``gnn_mlp`` (69-d x) and ``joint`` (5-d x plus a
separate ESM argument) -- which loads without error and predicts nonsense.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jbbind.core.features.build import build_chain_graph
from jbbind.core.nn.registry import ARCHS, ModelRegistry
from jbbind.core.nn.setups import SETUPS
from tests.conftest import MODELS

DEVICE = torch.device("cpu")
COMBOS = [(s, a) for s in SETUPS for a in ARCHS]


@pytest.fixture(scope="session")
def registry() -> ModelRegistry:
    return ModelRegistry(MODELS, DEVICE)


@pytest.mark.parametrize("setup,arch", COMBOS, ids=[f"{s}-{a}" for s, a in COMBOS])
def test_probabilities_match_reference(fx, registry, setup, arch):
    cg = build_chain_graph(fx.nodes, fx.links, fx.full_esm)
    model = registry.get(setup, arch)
    probs = model.predict(cg.graph, cg.esm, DEVICE).numpy()

    ref = fx.ref_probs[f"{setup}/{arch}"]
    assert probs.shape == ref.shape, f"{setup}/{arch}: shape {probs.shape} vs {ref.shape}"
    np.testing.assert_allclose(probs, ref, rtol=0, atol=1e-6,
                               err_msg=f"{setup}/{arch} diverges from the reference")


@pytest.mark.parametrize("setup", list(SETUPS))
def test_output_width_matches_setup(fx, registry, setup):
    cg = build_chain_graph(fx.nodes, fx.links, fx.full_esm)
    model = registry.get(setup, "gnn_mlp")
    probs = model.predict(cg.graph, cg.esm, DEVICE)
    assert probs.shape[1] == SETUPS[setup].num_labels
    assert model.label_names == SETUPS[setup].label_names


def test_probabilities_are_in_range(fx, registry):
    cg = build_chain_graph(fx.nodes, fx.links, fx.full_esm)
    for setup, arch in COMBOS:
        p = registry.get(setup, arch).predict(cg.graph, cg.esm, DEVICE)
        assert torch.isfinite(p).all(), f"{setup}/{arch} produced non-finite output"
        assert (p >= 0).all() and (p <= 1).all()


def test_registry_lists_all_twenty(registry):
    avail = registry.available()
    assert len(avail) == 20
    assert {(a["setup"], a["arch"]) for a in avail} == set(COMBOS)
    assert all(a["metrics"] for a in avail), "MANIFEST is missing metrics"


def test_gnn_needs_no_embedding(fx, registry):
    """The structure-only architecture must run with esm=None."""
    cg = build_chain_graph(fx.nodes, fx.links, esm=None)
    p = registry.get("protein", "gnn").predict(cg.graph, None, DEVICE)
    assert p.shape == (cg.n_residues, 1)


# --- negative controls ---------------------------------------------------------------

def test_wrong_embedder_changes_gnn_mlp_output(fx, registry):
    """If a random embedder gave the same answer, the pairing would be unverified."""
    from jbbind.core.nn.models import BindingSiteMLP

    cg = build_chain_graph(fx.nodes, fx.links, fx.full_esm)
    model = registry.get("protein", "gnn_mlp")
    good = model.predict(cg.graph, cg.esm, DEVICE).numpy()

    torch.manual_seed(0)
    rogue = BindingSiteMLP(input_dim=1280, hidden_dims=[1024, 512, 128, 64],
                           output_dim=1, dropout=0.4).eval()
    original, model.embedder = model.embedder, rogue
    try:
        bad = model.predict(cg.graph, cg.esm, DEVICE).numpy()
    finally:
        model.embedder = original
    assert not np.allclose(good, bad, atol=1e-4), (
        "gnn_mlp output is insensitive to its embedder — the pairing is not being used")


def test_symmetrizing_edges_changes_output(fx, registry):
    """Guards the one-directional edge convention at the model level."""
    cg = build_chain_graph(fx.nodes, fx.links, fx.full_esm)
    model = registry.get("protein", "gnn_mlp")
    good = model.predict(cg.graph, cg.esm, DEVICE).numpy()

    g = cg.graph.clone()
    g.edge_index = torch.cat([g.edge_index, g.edge_index.flip(0)], dim=1)
    g.edge_attr = torch.cat([g.edge_attr, g.edge_attr], dim=0)
    bad = model.predict(g, cg.esm, DEVICE).numpy()
    assert not np.allclose(good, bad, atol=1e-4)


def test_joint_and_gnn_mlp_disagree(fx, registry):
    """They take different node features; identical output would mean one is mis-wired."""
    cg = build_chain_graph(fx.nodes, fx.links, fx.full_esm)
    a = registry.get("protein", "gnn_mlp").predict(cg.graph, cg.esm, DEVICE).numpy()
    b = registry.get("protein", "joint").predict(cg.graph, cg.esm, DEVICE).numpy()
    assert not np.allclose(a, b, atol=1e-4)


def test_predict_does_not_mutate_the_graph(fx, registry):
    """gnn_mlp appends 64 columns to x; it must not do so in place."""
    cg = build_chain_graph(fx.nodes, fx.links, fx.full_esm)
    before = cg.graph.x.clone()
    for arch in ARCHS:
        registry.get("protein", arch).predict(cg.graph, cg.esm, DEVICE)
    assert torch.equal(cg.graph.x, before), "predict() mutated the shared graph"
    assert cg.graph.x.shape[1] == 5
