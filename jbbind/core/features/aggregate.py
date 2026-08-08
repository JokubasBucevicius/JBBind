"""Atom -> residue aggregation and feature normalization.

Copied from /home/jokubasb/protein_protein/train_multilabel.py
  aggregate_atoms_to_residues  lines 101-120
  aggregate_edges_to_residues  lines 123-133
  normalize_column             lines 251-256
  normalize_voromqa            lines 259-264
  load_chain_base              mirrors load_protein_base, lines 142-176
Source SHA256: 5a2582a07a6db3fd72e95af341b633dc908221dd3e279040ed00c58ed6e07232
Copied: 2026-08-08

DO NOT EDIT without re-running tests/test_parity_tensors.py, which asserts these produce
tensors bit-identical to the originals on the golden fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The full training-time pooling map. Label columns (binds_*) only exist in training data;
# bsite_area survives at inference because the forked voronota script keeps the
# "set bsite_area = 0" line, so the CSV schema is otherwise identical.
POOLING = {
    "residue_type": "first",
    "sas_area": "sum",
    "ev14": "max",
    "ev28": "max",
    "ev56": "max",
    "voromqa_sas_energy": "sum",
    "bsite_area": "sum",
    "binds_homo_protein": "max",
    "binds_hetero_protein": "max",
    "binds_dna": "max",
    "binds_rna": "max",
}

# Columns the model actually consumes. Anything else is carried for provenance only.
REQUIRED_NODE_COLUMNS = [
    "ID_resSeq", "residue_type", "sas_area",
    "voromqa_sas_energy", "ev14", "ev28", "ev56",
]


def aggregate_atoms_to_residues(nodes_df):
    if "surface_atom" not in nodes_df.columns:
        nodes_df["surface_atom"] = (nodes_df["sas_area"] > 0.0).astype(int)

    # DEVIATION: the original hardcodes the full pooling dict, which requires the four
    # binds_* label columns. At inference those do not exist, so pool only what is present.
    pooling = {k: v for k, v in POOLING.items() if k in nodes_df.columns}

    residue_df = nodes_df.groupby("ID_resSeq").agg(pooling).reset_index()
    return residue_df


def aggregate_edges_to_residues(edges_df):
    edges_df = edges_df[edges_df["ID1_resSeq"] != edges_df["ID2_resSeq"]].copy()
    edges_df["res1"] = edges_df[["ID1_resSeq", "ID2_resSeq"]].min(axis=1)
    edges_df["res2"] = edges_df[["ID1_resSeq", "ID2_resSeq"]].max(axis=1)

    residue_edges = edges_df.groupby(["res1", "res2"]).agg({
        "area": "sum",
        "boundary": "mean"
    }).reset_index()
    residue_edges = residue_edges.rename(columns={"res1": "ID1_resSeq", "res2": "ID2_resSeq"})
    return residue_edges


def normalize_column(values):
    values = np.array(values, dtype=np.float32)
    min_val, max_val = values.min(), values.max()
    if max_val - min_val > 0:
        return (values - min_val) / (max_val - min_val)
    return np.zeros_like(values)


def normalize_voromqa(values):
    values = np.array(values, dtype=np.float32)
    if values.min() < 0:
        values = values - values.min()
    values = np.log1p(values)
    return normalize_column(values)


def load_chain_base(nodes_df: pd.DataFrame, edges_df: pd.DataFrame | None):
    """Aggregate one chain's voronota CSVs to surface residues + residue edges.

    Mirrors ``train_multilabel.load_protein_base`` (:142-176) exactly, with two deviations:
    it takes DataFrames rather than paths, and it drops the gate at :152-154 that returns
    None unless all four binds_* label columns are present.

    Returns ``(residue_df, edges_df)``; ``residue_df`` is None if no surface residues remain.
    """
    missing = [c for c in REQUIRED_NODE_COLUMNS if c not in nodes_df.columns]
    if missing:
        raise ValueError(f"graph_nodes.csv missing required columns: {missing}")

    residue_df = aggregate_atoms_to_residues(nodes_df)
    if edges_df is not None and len(edges_df) > 0:
        edges_df = aggregate_edges_to_residues(edges_df)
    else:
        edges_df = None

    surface_mask = residue_df['sas_area'] > 0
    residue_df = residue_df[surface_mask].reset_index(drop=True)

    if len(residue_df) == 0:
        return None, None

    if edges_df is not None and len(edges_df) > 0:
        surface_ids = set(residue_df['ID_resSeq'])
        edge_mask = (edges_df['ID1_resSeq'].isin(surface_ids) &
                     edges_df['ID2_resSeq'].isin(surface_ids))
        edges_df = edges_df[edge_mask].reset_index(drop=True)

    return residue_df, edges_df


def restrict_to_residues(residue_df, edges_df, keep_mask):
    """Drop residues (and their edges), mirroring train_multilabel.py:335-342.

    Used by the ESM truncation path, which drops residues whose SEQRES index exceeds the
    embedding length. Order matters: filter residues first, then re-restrict edges.
    """
    residue_df = residue_df[keep_mask].reset_index(drop=True)
    if edges_df is not None and len(edges_df) > 0:
        surface_ids = set(residue_df['ID_resSeq'])
        edge_mask = (edges_df['ID1_resSeq'].isin(surface_ids) &
                     edges_df['ID2_resSeq'].isin(surface_ids))
        edges_df = edges_df[edge_mask].reset_index(drop=True)
    return residue_df, edges_df
