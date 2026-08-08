"""ESM-2 embeddings, reproducing exactly what the training data contains.

Training used Meta's ``esm/scripts/extract.py`` (copied to the research repo as
``extract.py``) invoked as::

    python extract.py esm2_t33_650M_UR50D ppi3d_filtered_subunits.fasta emb/ \
        --repr_layers 0 33 6 --include per_tok

which writes ``{"label": ..., "representations": {0: ..., 6: ..., 33: ...}}`` with each
tensor shaped (L, 1280), BOS stripped and EOS excluded. Only layer 33 is ever read
(``train_multilabel.py:191-192`` takes ``sorted(keys)[-1]``), so that is all we compute.

Two behaviours are inherited deliberately rather than improved:

* **1022-token truncation.** ``extract.py``'s ``--truncation_seq_length`` defaults to 1022,
  so 942 of 88,604 training chains (1.06%) were silently truncated and their tail residues
  dropped from the graph. ``truncate`` mode reproduces this exactly; ``tile`` is offered as
  an opt-in but is explicitly out of distribution.
* **Embedding the canonical (SEQRES) sequence, not the observed one.** Training embedded
  PPI3D's full ``s1_sequence``, which includes unmodelled residues; the graph then indexed
  into it with ``resSeq - 1``. Embedding only the observed residues would shift every
  residue after the first gap onto the wrong vector.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

MODEL_NAME = "esm2_t33_650M_UR50D"
LAYER = 33
EMBED_DIM = 1280
#: extract.py --truncation_seq_length default.
MAX_TOKENS = 1022


class EsmEmbedder:
    """Lazily-loaded ESM-2 650M, with a sequence-hash disk cache.

    The model is ~2.6 GB in fp32 and forwards are serialized behind a lock: holding one
    copy and queueing is the right trade for a service that also has to fit a torch runtime
    and a web server in memory.
    """

    def __init__(self, device: torch.device, cache=None,
                 long_seq_mode: str = "truncate"):
        self.device = device
        self.cache = cache
        self.long_seq_mode = long_seq_mode
        self._model = None
        self._alphabet = None
        self._converter = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import esm
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model.eval().to(self.device)
            self._model = model
            self._alphabet = alphabet
            self._converter = alphabet.get_batch_converter()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- embedding ---------------------------------------------------------------

    @staticmethod
    def cache_key(sequence: str, mode: str) -> str:
        return hashlib.sha256(f"{MODEL_NAME}|{LAYER}|{mode}|{sequence}".encode()).hexdigest()

    def embed(self, sequence: str) -> tuple[torch.Tensor, list[dict]]:
        """Per-residue layer-33 embedding of ``sequence``.

        Returns ``(tensor, warnings)``. The tensor has one row per input residue in
        ``tile`` mode, or ``min(len(sequence), 1022)`` rows in ``truncate`` mode — the
        caller drops residues that fall past the end, exactly as training did.
        """
        warnings: list[dict] = []
        key = self.cache_key(sequence, self.long_seq_mode)

        if self.cache is not None:
            path = self.cache.path(key, ".npy")
            if path.exists():
                arr = np.load(path)
                return torch.from_numpy(arr), self._length_warnings(sequence, arr.shape[0])

        tensor = self._compute(sequence)

        if self.cache is not None:
            def produce(tmp: Path) -> None:
                # Write through a file handle: np.save(path, ...) would append a second
                # ".npy" to the cache's temporary filename and the atomic rename would
                # then find nothing to move.
                with open(tmp, "wb") as fh:
                    np.save(fh, tensor.numpy())
            self.cache.get_or_create(key, produce, suffix=".npy")

        warnings.extend(self._length_warnings(sequence, tensor.shape[0]))
        return tensor, warnings

    def _length_warnings(self, sequence: str, n_rows: int) -> list[dict]:
        if len(sequence) <= MAX_TOKENS or self.long_seq_mode != "truncate":
            return []
        return [{
            "code": "esm_truncated",
            "detail": (f"the chain is {len(sequence)} residues but ESM-2 was run with a "
                       f"{MAX_TOKENS}-token limit, matching how the training embeddings "
                       f"were generated. Residues past position {MAX_TOKENS} have no "
                       f"embedding and are not predicted."),
            "sequence_length": len(sequence),
            "embedded_length": n_rows,
        }]

    @torch.inference_mode()
    def _compute(self, sequence: str) -> torch.Tensor:
        self.load()
        if len(sequence) <= MAX_TOKENS or self.long_seq_mode == "truncate":
            return self._forward(sequence[:MAX_TOKENS]).cpu()
        return self._forward_tiled(sequence).cpu()

    def _forward(self, sequence: str) -> torch.Tensor:
        with self._lock:
            _, _, tokens = self._converter([("query", sequence)])
            tokens = tokens.to(self.device)
            out = self._model(tokens, repr_layers=[LAYER], return_contacts=False)
            # Strip BOS; take exactly len(sequence) rows, dropping EOS. Mirrors
            # extract.py's `t[i, 1:truncate_len + 1]`.
            return out["representations"][LAYER][0, 1:len(sequence) + 1].float()

    def _forward_tiled(self, sequence: str) -> torch.Tensor:
        """Overlapping windows; each residue takes the window where it is most central.

        Out of distribution relative to training — the models never saw an embedding
        produced this way. Offered only because truncation silently discards the tail of
        very long chains.
        """
        stride = MAX_TOKENS // 2
        out = torch.zeros((len(sequence), EMBED_DIM), dtype=torch.float32)
        best = np.full(len(sequence), np.inf)
        for start in range(0, len(sequence), stride):
            end = min(start + MAX_TOKENS, len(sequence))
            chunk = self._forward(sequence[start:end]).cpu()
            centre = (start + end - 1) / 2.0
            for i in range(start, end):
                d = abs(i - centre)
                if d < best[i]:
                    best[i] = d
                    out[i] = chunk[i - start]
            if end == len(sequence):
                break
        return out
