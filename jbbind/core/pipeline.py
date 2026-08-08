"""End-to-end inference: structure in, per-residue binding scores out.

    acquire -> normalize -> voronota -> aggregate -> ESM -> model -> map back

Every stage is cached on a content hash, so a repeat request for the same chain is nearly
free and a batch run warms the cache for the UI.

One design decision worth stating: all five label setups are predicted in a single request.
The expensive parts (tessellation, ESM) are shared, and the five GNN forwards that follow
cost milliseconds each, so predicting one setup and predicting all five differ by well
under a second.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from ..settings import Settings, UserSettings
from .cache import CacheSet, sha256_bytes, sha256_text
from .esm.embedder import EsmEmbedder
from .features.build import ChainGraph, build_chain_graph
from .features.voronota import describe_chain
from .nn.registry import NEEDS_ESM, NEEDS_GRAPH, ModelRegistry
from .nn.setups import SETUPS
from .structure import normalize as norm
from .structure.fetch import fetch_pdb
from .structure.normalize import (NormalizationError, PreparedChain, list_chains,
                                  prepare_chain, read_structure)

ProgressFn = Callable[[str, str], None]


@dataclass
class ResiduePrediction:
    seqres_index: int
    auth_chain: str
    auth_seq_id: int
    auth_icode: str
    resname: str
    one_letter: str
    sas_area: float
    probs: dict[str, list[float]]  # setup -> per-label probabilities


@dataclass
class PredictionResult:
    structure_id: str
    source: str
    chain_id: str
    arch: str
    sequence: str
    numbering_source: str
    label_names: dict[str, list[str]]
    residues: list[ResiduePrediction]
    unpredicted: list[dict]
    receptor_pdb: str
    warnings: list[dict] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def n_predicted(self) -> int:
        return len(self.residues)


class Pipeline:
    def __init__(self, settings: Settings, registry: ModelRegistry,
                 embedder: EsmEmbedder, caches: CacheSet):
        self.settings = settings
        self.registry = registry
        self.embedder = embedder
        self.caches = caches

    # -- input ------------------------------------------------------------------

    def load_structure(self, *, pdb_id: str | None = None, data: bytes | None = None,
                       assembly: int | None = None) -> tuple[bytes, str, str]:
        """Returns (raw bytes, structure_id, human-readable source)."""
        if pdb_id:
            raw = fetch_pdb(pdb_id, self.caches.rcsb, assembly=assembly)
            src = f"RCSB {pdb_id.upper()}" + (f" assembly {assembly}" if assembly else "")
            return raw, sha256_bytes(raw), src
        if data is None:
            raise ValueError("either pdb_id or data is required")
        if len(data) > self.settings.max_upload_bytes:
            raise NormalizationError(
                "PayloadTooLarge",
                f"upload is {len(data)/1e6:.1f} MB; the limit is "
                f"{self.settings.max_upload_bytes/1e6:.0f} MB")
        return data, sha256_bytes(data), "uploaded file"

    def describe_structure(self, raw: bytes) -> tuple[list, list[dict]]:
        st = read_structure(raw)
        return list_chains(st)

    # -- prediction --------------------------------------------------------------

    def predict(self, *, raw: bytes, structure_id: str, source: str, chain_id: str,
                user: UserSettings, setups: Optional[list[str]] = None,
                progress: Optional[ProgressFn] = None) -> PredictionResult:
        setups = setups or list(SETUPS)
        arch = user.arch
        device = self.settings.device if user.device == "auto" else torch.device(user.device)
        timings: dict[str, int] = {}
        warnings: list[dict] = []

        def step(name: str, message: str):
            if progress:
                progress(name, message)
            return time.perf_counter()

        t = step("normalize", f"preparing chain {chain_id}")
        st = read_structure(raw)
        prepared: PreparedChain = prepare_chain(st, chain_id)
        warnings.extend(prepared.warnings)
        if len(prepared.sequence) > self.settings.max_residues:
            raise NormalizationError(
                "TooManyResidues",
                f"chain {chain_id} has {len(prepared.sequence)} residues; this instance is "
                f"configured for at most {self.settings.max_residues}.")
        timings["normalize"] = int((time.perf_counter() - t) * 1000)

        need_graph = arch in NEEDS_GRAPH
        need_esm = arch in NEEDS_ESM

        nodes = links = None
        receptor_pdb = ""
        if need_graph:
            t = step("voronota", "computing Voronoi tessellation and surface features")
            vout = describe_chain(prepared.pdb_text, norm.VORONOTA_CHAIN_ID,
                                  self.caches.chain, n_residues=len(prepared.residues))
            nodes, links, receptor_pdb = vout.nodes, vout.links, vout.receptor_pdb
            timings["voronota"] = int((time.perf_counter() - t) * 1000)

        esm = None
        if need_esm:
            t = step("esm", f"embedding {len(prepared.sequence)} residues with ESM-2 650M")
            self.embedder.long_seq_mode = user.esm_long_seq_mode
            esm, esm_warnings = self.embedder.embed(prepared.sequence)
            warnings.extend(esm_warnings)
            timings["esm"] = int((time.perf_counter() - t) * 1000)

        t = step("features", "building the residue graph")
        if need_graph:
            cg = build_chain_graph(nodes, links, esm)
            if cg is None:
                raise NormalizationError(
                    "NoSurfaceResidues",
                    f"chain {chain_id} has no solvent-accessible residue to predict on.")
            warnings.extend(cg.warnings)
            resseq = cg.resseq
            sas = cg.residue_df["sas_area"].to_numpy()
            graph, esm_rows = cg.graph, cg.esm
        else:
            # Sequence-only architecture: every residue of the canonical sequence that we
            # actually observed gets a prediction.
            resseq = np.array([r.seqres_index for r in prepared.residues], dtype=np.int64)
            keep = resseq <= esm.shape[0]
            resseq = resseq[keep]
            esm_rows = esm[resseq - 1]
            sas = np.full(len(resseq), np.nan)
            graph = None
        timings["features"] = int((time.perf_counter() - t) * 1000)

        t = step("model", f"running {arch} for {len(setups)} label setups")
        probs_by_setup: dict[str, np.ndarray] = {}
        label_names: dict[str, list[str]] = {}
        for setup_name in setups:
            model = self.registry.get(setup_name, arch)
            p = model.predict(graph, esm_rows, device).numpy()
            probs_by_setup[setup_name] = p
            label_names[setup_name] = model.label_names
        timings["model"] = int((time.perf_counter() - t) * 1000)

        residue_map = prepared.residue_map
        residues: list[ResiduePrediction] = []
        for i, idx in enumerate(resseq):
            rec = residue_map.get(int(idx))
            if rec is None:
                continue
            residues.append(ResiduePrediction(
                seqres_index=int(idx),
                auth_chain=rec.auth_chain,
                auth_seq_id=rec.auth_seq_id,
                auth_icode=rec.auth_icode,
                resname=rec.resname,
                one_letter=rec.one_letter,
                sas_area=float(sas[i]) if not np.isnan(sas[i]) else None,
                probs={s: [round(float(x), 5) for x in probs_by_setup[s][i]]
                       for s in setups},
            ))

        predicted = {r.seqres_index for r in residues}
        unpredicted = [
            {"seqres_index": rec.seqres_index, "auth_seq_id": rec.auth_seq_id,
             "one_letter": rec.one_letter,
             "reason": "buried" if need_graph else "outside embedding"}
            for rec in prepared.residues if rec.seqres_index not in predicted
        ]

        return PredictionResult(
            structure_id=structure_id, source=source, chain_id=chain_id, arch=arch,
            sequence=prepared.sequence, numbering_source=prepared.numbering_source,
            label_names=label_names, residues=residues, unpredicted=unpredicted,
            receptor_pdb=receptor_pdb, warnings=warnings, timings_ms=timings)

    def cache_key(self, structure_id: str, chain_id: str, user: UserSettings) -> str:
        return sha256_text(structure_id, chain_id, user.arch, user.esm_long_seq_mode,
                           str(user.rcsb_assembly),
                           self.registry.manifest_digest())
