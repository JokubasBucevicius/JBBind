"""Residue tables -> PyTorch Geometric ``Data``.

Copied from /home/jokubasb/protein_protein/train_multilabel.py
  create_pyg_graph  lines 267-305
Source SHA256: 5a2582a07a6db3fd72e95af341b633dc908221dd3e279040ed00c58ed6e07232
Copied: 2026-08-08

DO NOT EDIT without re-running tests/test_parity_tensors.py.

Two properties here are load-bearing and counter-intuitive; both are deliberate, both are
covered by tests, and neither should be "fixed":

1. **Node features are min-max normalized per graph.** Every chain is rescaled against its
   own extremes, so the same residue gets different features depending on what else is in
   the chain. That is what the checkpoints were trained on.
2. **Edges are stored in one direction only.** ``aggregate_edges_to_residues`` canonicalises
   each contact to (min, max) resSeq and no reverse edge is ever added — there is no
   ``to_undirected()`` anywhere in the training code. Message passing is therefore
   one-directional. ``tests/test_parity_tensors.py::test_edge_index_is_not_symmetric``
   guards this.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .aggregate import normalize_column, normalize_voromqa


def create_pyg_graph(residue_df, edges_df, extra_features=None, y=None):
    """Create PyG graph. If extra_features provided, concatenate with structural."""
    structural = np.stack([
        normalize_column(residue_df['sas_area'].values),
        normalize_voromqa(residue_df['voromqa_sas_energy'].values),
        normalize_column(residue_df['ev14'].values),
        normalize_column(residue_df['ev28'].values),
        normalize_column(residue_df['ev56'].values),
    ], axis=1).astype(np.float32)

    if extra_features is not None:
        if torch.is_tensor(extra_features):
            extra_features = extra_features.numpy()
        node_features = np.concatenate([structural, extra_features], axis=1).astype(np.float32)
    else:
        node_features = structural

    residue_types = torch.tensor(residue_df['residue_type'].values, dtype=torch.long)

    if edges_df is not None and len(edges_df) > 0:
        id_to_idx = {id_val: idx for idx, id_val in enumerate(residue_df['ID_resSeq'])}
        src = [id_to_idx[id1] for id1 in edges_df['ID1_resSeq']]
        dst = [id_to_idx[id2] for id2 in edges_df['ID2_resSeq']]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edges_df[['area', 'boundary']].values, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)

    # DEVIATION: the original always calls extract_multilabels(residue_df), which reads the
    # module-global LABEL_COLUMNS and requires the four binds_* columns. At inference there
    # are no labels, so y is an optional argument instead.
    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        residue_type=residue_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(residue_df),
    )
