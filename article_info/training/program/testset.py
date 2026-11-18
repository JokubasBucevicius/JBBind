"""Held-out test set: per-chain cache + loaders mirroring the training pipeline.

The final test set (``all_class/test_graphs/...``) is ~21.5k chains of ~2 MB of
atom-level CSV each, and every ``(setup, arch)`` combination needs the same
residue-level aggregation. Re-reading the CSVs once per combination costs ~17 min
a pass, so this module reads them **once** into a compact per-chain cache and
rebuilds the exact frames ``train_multilabel`` expects from it.

What is cached is the output of ``tm.load_protein_base`` -- the residue-level
aggregation and surface filter -- *before* normalisation, because
``tm.create_pyg_graph`` min-max normalises over whatever residues survive and the
ESM-embedding paths drop residues past the end of the embedding. Feature
construction itself is never re-implemented: the cached columns are handed back
to ``tm.create_pyg_graph``.

The three loaders below reproduce, residue-for-residue, the three loading paths
used during training:

===============  ==========================================================
``load_mlp``     ``tm.load_protein_mlp``            -> MLP
``load_gnn``     ``tm.load_protein_base`` + graph   -> GNN
``load_esm_gnn`` ``tm._load_protein_gnn_mlp_base``  -> GNN+MLP, Joint
===============  ==========================================================
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# config inserts the repo root on sys.path so ``train_multilabel`` is importable.
import config  # noqa: F401
import train_multilabel as tm

from setups import BASE_LABEL_COLUMNS, LabelSetup

# The structural columns ``tm.create_pyg_graph`` reads, in its own order.
STRUCT_COLS = ["sas_area", "voromqa_sas_energy", "ev14", "ev28", "ev56"]

CACHE_VERSION = 1


# ----------------------------------------------------------------- cache build
def _cache_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, name[:2], f"{name}.pt")


def _build_entry(name: str, graph_dir: str) -> Optional[dict]:
    """Residue-level aggregation of one chain, as plain numpy arrays."""
    residue_df, edges_df, _ = tm.load_protein_base(name, graph_dir)
    if residue_df is None:
        return None

    entry = {
        "v": CACHE_VERSION,
        "name": name,
        "id_res": residue_df["ID_resSeq"].to_numpy(np.int32),
        "residue_type": residue_df["residue_type"].to_numpy(np.int16),
        "struct": residue_df[STRUCT_COLS].to_numpy(np.float32),
        "y4": residue_df[BASE_LABEL_COLUMNS].to_numpy(np.uint8),
        "has_edges": edges_df is not None,
    }
    if edges_df is not None and len(edges_df) > 0:
        entry["e1"] = edges_df["ID1_resSeq"].to_numpy(np.int32)
        entry["e2"] = edges_df["ID2_resSeq"].to_numpy(np.int32)
        entry["e_attr"] = edges_df[["area", "boundary"]].to_numpy(np.float32)
    else:
        entry["e1"] = np.zeros(0, np.int32)
        entry["e2"] = np.zeros(0, np.int32)
        entry["e_attr"] = np.zeros((0, 2), np.float32)
    return entry


def _build_one(args: Tuple[str, str, str]) -> Tuple[str, bool]:
    name, graph_dir, cache_dir = args
    path = _cache_path(cache_dir, name)
    if os.path.exists(path):
        return name, True
    try:
        entry = _build_entry(name, graph_dir)
    except Exception:
        entry = None
    if entry is None:
        return name, False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(entry, tmp)
    os.replace(tmp, path)
    return name, True


def build_cache(names: Sequence[str], graph_dir: str, cache_dir: str,
                workers: int = 16) -> List[str]:
    """Populate ``cache_dir`` (resumable). Returns the chains that loaded."""
    todo = [(n, graph_dir, cache_dir) for n in names]
    ok: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, good in tqdm(pool.map(_build_one, todo, chunksize=16),
                               total=len(todo), desc="Cache"):
            if good:
                ok.append(name)
    print(f"Cached {len(ok)}/{len(names)} chains -> {cache_dir}")
    return sorted(ok)


def load_entry(cache_dir: str, name: str) -> Optional[dict]:
    path = _cache_path(cache_dir, name)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


# ------------------------------------------------------------- reconstruction
def _frames(entry: dict, keep: Optional[np.ndarray] = None
            ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Rebuild the ``(residue_df, edges_df)`` pair ``tm.create_pyg_graph`` wants.

    ``keep`` is an optional boolean mask over residues; edges are re-filtered to
    the survivors exactly as ``tm._load_protein_gnn_mlp_base`` does.
    """
    sel = slice(None) if keep is None else keep
    struct = entry["struct"][sel]
    residue_df = pd.DataFrame({
        "ID_resSeq": entry["id_res"][sel],
        "residue_type": entry["residue_type"][sel].astype(np.int64),
        **{c: struct[:, i] for i, c in enumerate(STRUCT_COLS)},
        **{c: entry["y4"][sel][:, i].astype(np.float32)
           for i, c in enumerate(BASE_LABEL_COLUMNS)},
    })

    if not entry["has_edges"]:
        return residue_df, None

    e1, e2, attr = entry["e1"], entry["e2"], entry["e_attr"]
    if keep is not None and len(e1):
        surface = set(residue_df["ID_resSeq"].tolist())
        mask = np.fromiter((a in surface and b in surface for a, b in zip(e1, e2)),
                           dtype=bool, count=len(e1))
        e1, e2, attr = e1[mask], e2[mask], attr[mask]
    edges_df = pd.DataFrame({
        "ID1_resSeq": e1, "ID2_resSeq": e2,
        "area": attr[:, 0], "boundary": attr[:, 1],
    })
    return residue_df, edges_df


def _esm(name: str, emb_dir: str) -> Optional[torch.Tensor]:
    path = os.path.join(emb_dir, f"{name}.pt")
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "representations" in data:
        layer_keys = sorted(data["representations"].keys())
        data = data["representations"][layer_keys[-1]]
    return data


def _align(entry: dict, emb_size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(embedding_indices, keep_mask)`` -- the residue<->ESM alignment rule.

    Copied from ``tm.load_protein_mlp`` / ``tm._load_protein_gnn_mlp_base``:
    1-based ``ID_resSeq`` shifts down by one, and residues indexing past the end
    of the embedding are dropped.
    """
    ids = entry["id_res"]
    idx = ids - 1 if ids.min() >= 1 else ids
    if idx.min() < 0:
        return None
    keep = idx < emb_size
    if not keep.any():
        return None
    return idx[keep], keep


# -------------------------------------------------------------------- dataset
class TestSet:
    """The held-out chains, filtered per setup, loaded in memory-bounded chunks."""

    def __init__(self, cache_dir: str, emb_dir: str, names: Sequence[str]):
        self.cache_dir = cache_dir
        self.emb_dir = emb_dir
        self.names = list(names)
        self._y4_max: Dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- filtering
    def qualifying(self, setup: LabelSetup) -> List[str]:
        """Chains with >= 1 surface binding residue of this setup's type.

        The same rule ``ProteinDataModule._protein_qualifies`` applied when the
        models were trained, read off the cached labels instead of the CSV.
        """
        idxs = sorted({i for group in setup.base_index_groups for i in group})
        out = []
        for name in tqdm(self.names, desc=f"Filter[{setup.name}]"):
            colmax = self._y4_max.get(name)
            if colmax is None:
                entry = load_entry(self.cache_dir, name)
                if entry is None:
                    continue
                colmax = entry["y4"].max(axis=0)
                self._y4_max[name] = colmax
            if colmax[idxs].any():
                out.append(name)
        print(f"Qualifying for setup '{setup.name}': {len(out)}/{len(self.names)} chains "
              f"(>= 1 binding residue in {setup.trigger_columns})")
        return out

    def label_distribution(self, setup: LabelSetup, names: Sequence[str]) -> dict:
        """Per-residue positives for this setup over ``names``.

        Mirrors the ``Training label distribution`` block the training runs log
        via ``ProteinDataModule.make_criterion``, so the two are comparable line
        for line.
        """
        all_idx = sorted({i for g in setup.base_index_groups for i in g})
        n_res = n_any = n_multi = 0
        pos = np.zeros(setup.num_labels, dtype=np.int64)
        for name in names:
            entry = load_entry(self.cache_dir, name)
            if entry is None:
                continue
            y4 = entry["y4"] > 0
            n_res += y4.shape[0]
            n_any += int(y4[:, all_idx].any(axis=1).sum())
            derived = np.stack([y4[:, list(g)].any(axis=1)
                                for g in setup.base_index_groups], axis=1)
            n_multi += int((derived.sum(axis=1) > 1).sum())
            pos += derived.sum(axis=0)
        return {"n_residues": n_res, "n_chains": len(names),
                "n_binding": n_any, "n_multilabel": n_multi,
                "positives": {name: int(pos[j])
                              for j, name in enumerate(setup.label_names)}}

    @staticmethod
    def print_distribution(setup: LabelSetup, dist: dict) -> None:
        n = max(1, dist["n_residues"])
        print(f"Test label distribution (per residue, n={dist['n_residues']}, "
              f"chains={dist['n_chains']}):")
        weights = []
        for label in setup.label_names:
            n_pos = dist["positives"][label]
            weights.append((dist["n_residues"] - n_pos) / max(1, n_pos))
            print(f"  {label}: {n_pos} ({100 * n_pos / n:.2f}%)")
        non_binding = dist["n_residues"] - dist["n_binding"]
        print(f"  Non-binding: {non_binding} ({100 * non_binding / n:.2f}%)")
        if setup.num_labels > 1:
            print(f"  Multi-label residues: {dist['n_multilabel']} "
                  f"({100 * dist['n_multilabel'] / max(1, dist['n_binding']):.2f}% "
                  f"of binding)")
        print(f"implied pos_weights: {[f'{w:.1f}' for w in weights]}")

    # ------------------------------------------------------------------ loaders
    def load_mlp(self, names: Sequence[str], setup: LabelSetup
                 ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Mirrors ``tm.load_protein_mlp``: (ESM rows, setup labels) per chain."""
        out = []
        for name in names:
            entry = load_entry(self.cache_dir, name)
            if entry is None:
                continue
            emb = _esm(name, self.emb_dir)
            if emb is None:
                continue
            aligned = _align(entry, emb.shape[0])
            if aligned is None:
                continue
            idx, keep = aligned
            y4 = torch.from_numpy(entry["y4"][keep].astype(np.float32))
            yk = setup.derive_labels(y4)
            if yk.sum() == 0:
                continue
            out.append((emb[idx, :], yk))
        return out

    def load_gnn(self, names: Sequence[str], setup: LabelSetup) -> list:
        """Mirrors ``ProteinDataModule.load_structural_graphs``: 5D structural graphs."""
        out = []
        for name in names:
            entry = load_entry(self.cache_dir, name)
            if entry is None:
                continue
            residue_df, edges_df = _frames(entry)
            g = tm.create_pyg_graph(residue_df, edges_df)
            g.y = setup.derive_labels(g.y)
            if g.y.sum() == 0:
                continue
            g.protein_name = name
            out.append(g)
        return out

    def load_esm_gnn(self, names: Sequence[str], setup: LabelSetup, mode: str,
                     mlp_model=None) -> list:
        """Mirrors ``tm._load_protein_gnn_mlp_base`` -- the ESM-aligned graph paths.

        ``mode='gnn_mlp'`` appends ``mlp_model``'s 64-d penultimate activations to
        the 5 structural features. ``mode='joint'`` keeps the 5 structural features
        and attaches raw ESM instead: training built those graphs with a throwaway
        MLP whose columns were sliced straight back off, so no MLP is involved and
        none is needed here.
        """
        out = []
        for name in names:
            entry = load_entry(self.cache_dir, name)
            if entry is None:
                continue
            emb = _esm(name, self.emb_dir)
            if emb is None:
                continue
            aligned = _align(entry, emb.shape[0])
            if aligned is None:
                continue
            idx, keep = aligned
            esm = emb[idx, :].clone()

            extra = None
            if mode == "gnn_mlp":
                with torch.no_grad():
                    x = esm.to(tm.device)
                    for layer in list(mlp_model.network.children())[:-1]:
                        x = layer(x)
                    extra = x.cpu()

            residue_df, edges_df = _frames(entry, keep=keep)
            g = tm.create_pyg_graph(residue_df, edges_df, extra_features=extra)
            if mode == "joint":
                g.esm_embeddings = esm
            g.y = setup.derive_labels(g.y)
            if g.y.sum() == 0:
                continue
            g.protein_name = name
            out.append(g)
        return out


def chunks(names: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(names), size):
        yield list(names[i:i + size])
