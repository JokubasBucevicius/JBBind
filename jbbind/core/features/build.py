"""Build a model-ready graph from one chain's voronota CSVs + its ESM-2 embedding.

This is both the production path and the parity target: it reproduces, step for step, what
``train_multilabel._load_protein_gnn_mlp_base`` (:308-358) did at training time.

The ESM alignment is the subtle part. Training indexed embeddings with
``ID_resSeq - 1`` (:329), which is only correct because PPI3D's ``ID_resSeq`` is a 1-based
index into the full SEQRES sequence — verified on 250 random training chains, where
``one_letter(ID_resName) == s1_sequence[ID_resSeq - 1]`` holds for 94.4% of them. JBBind
therefore renumbers every incoming structure to SEQRES indices before voronota runs (see
``jbbind.core.structure.renumber``) and embeds the canonical sequence, which makes
``ID_resSeq - 1`` the correct index here too. Do not replace this with rank-of-observed-
residue indexing: for a chain like 6oax_E (579 observed of 867 SEQRES) that would shift
almost every residue onto the wrong embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .aggregate import load_chain_base, restrict_to_residues
from .pyg import create_pyg_graph


@dataclass
class ChainGraph:
    """A chain ready for inference, plus everything needed to map results back."""

    graph: Data                    # x is the 5-d structural block
    esm: Optional[torch.Tensor]    # (n_residues, 1280), row-aligned to graph nodes
    residue_df: pd.DataFrame       # row-aligned to graph nodes; carries ID_resSeq
    n_input_residues: int          # surface residues before ESM-range filtering
    dropped_resseq: list[int] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    @property
    def resseq(self) -> np.ndarray:
        return self.residue_df["ID_resSeq"].to_numpy()

    @property
    def n_residues(self) -> int:
        return int(self.graph.num_nodes)


def build_chain_graph(nodes_df: pd.DataFrame,
                      edges_df: Optional[pd.DataFrame],
                      esm: Optional[torch.Tensor] = None) -> Optional[ChainGraph]:
    """Aggregate to surface residues, align the ESM embedding, and build the PyG graph.

    ``esm`` must be the layer-33 per-token embedding of the chain's **canonical (SEQRES)**
    sequence, shape (L, 1280) with BOS/EOS already stripped. Pass None for the
    structure-only ``gnn`` architecture.

    Returns None if the chain has no surface residues, or if no residue falls inside the
    embedding (which for a correctly renumbered chain means the whole chain was past ESM's
    1022-token truncation).
    """
    residue_df, edges_df = load_chain_base(nodes_df, edges_df)
    if residue_df is None:
        return None

    n_input = len(residue_df)
    warnings: list[dict] = []
    dropped: list[int] = []
    esm_sel = None

    if esm is not None:
        emb_size = int(esm.shape[0])
        id_res_seq = residue_df["ID_resSeq"].to_numpy()
        min_res_seq = id_res_seq.min()
        # train_multilabel.py:329 — the guard exists for chains numbered from 0.
        embedding_indices = id_res_seq - 1 if min_res_seq >= 1 else id_res_seq

        valid_mask = (embedding_indices >= 0) & (embedding_indices < emb_size)
        if valid_mask.sum() == 0:
            return None

        if valid_mask.sum() < len(embedding_indices):
            dropped = [int(r) for r in id_res_seq[~valid_mask]]
            embedding_indices = embedding_indices[valid_mask]
            residue_df, edges_df = restrict_to_residues(residue_df, edges_df, valid_mask)
            warnings.append({
                "code": "esm_range_dropped",
                "detail": (f"{len(dropped)} of {n_input} surface residues fall outside the "
                           f"{emb_size}-token embedding and were dropped, matching "
                           f"training behaviour (train_multilabel.py:334-341)."),
                "n_dropped": len(dropped),
                "embedding_length": emb_size,
            })

        esm_sel = esm[embedding_indices, :].clone()

    graph = create_pyg_graph(residue_df, edges_df)
    return ChainGraph(graph=graph, esm=esm_sel, residue_df=residue_df,
                      n_input_residues=n_input, dropped_resseq=dropped,
                      warnings=warnings)
