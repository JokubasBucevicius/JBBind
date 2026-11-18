"""Optional post-training refinement: let each residue borrow evidence from its contacts.

Binding sites are patches, so a residue whose Voronoi neighbours all score high is more
likely to bind than its own score alone says, and a lone high score in a quiet
neighbourhood is more likely noise. Two rules are offered; both are off by default and
neither touches the model's raw output, which stays what every reported metric
describes.

``bayes`` -- log-odds pooling. Each neighbour contributes its log-odds relative to the
    chain's baseline, weighted by contact area, as extra evidence:

        z'_i = z_i + beta * sum_j w_ij (z_j - z_bar)

    ``w`` is row-normalised contact area and ``z_bar`` the logit of the chain's mean score
    for that label (an empirical-Bayes prior). ``beta`` < 1 discounts the neighbours
    because their evidence is correlated with, not independent of, the residue's own.

``gumbel`` -- Gumbel-softmax over the contact shell N(i) + {i}. The categorical is
    "which residue anchors this patch", with prior proportional to contact area (self
    weighted as the strongest contact) and likelihood the residue's odds. Each sample
    draws a relaxed one-hot ``a_i`` and reads ``sum_j a_ij p_j``; the result blends the
    sample mean into the raw score by ``lambda``. Gumbel-softmax is a training-time
    relaxation: used after training, its mean is just a temperature-weighted neighbour
    average. What the sampling adds is the spread, returned as a per-residue sd.

The labels are independent sigmoids, so every softmax here runs over neighbours, never
over labels. Pure numpy, so the offline evaluation can import it without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

METHODS = ("none", "bayes", "gumbel")

#: Scores are clipped before the logit so a 0 or 1 cannot become an infinite vote.
_EPS = 1e-4
#: Samples processed together; bounds memory at (block x edges x labels) floats.
_SAMPLE_BLOCK = 16


@dataclass
class ContactGraph:
    """Residue contacts, row-aligned to a list of predicted residues.

    Edges are stored in both directions; ``weight`` is Voronoi contact area in Å².
    """

    resseq: np.ndarray      # (n,) SEQRES index per row
    src: np.ndarray         # (E,) row index
    dst: np.ndarray         # (E,) row index
    weight: np.ndarray      # (E,) contact area

    @property
    def n(self) -> int:
        return len(self.resseq)

    @classmethod
    def from_edges(cls, resseq, id1, id2, area) -> "ContactGraph":
        """Build from residue-pair contacts keyed by SEQRES index.

        ``resseq`` fixes the rows. Contacts naming a residue outside it are dropped, and a
        row with no contact is simply isolated -- the refinement leaves it unchanged. The
        voronota edges are one-directional (min, max) pairs, so the reverse is added here.
        """
        resseq = np.asarray(resseq, dtype=np.int64)
        row = {int(r): i for i, r in enumerate(resseq)}
        src, dst, w = [], [], []
        for a, b, x in zip(np.asarray(id1), np.asarray(id2), np.asarray(area, dtype=float)):
            i, j = row.get(int(a)), row.get(int(b))
            if i is None or j is None or i == j or not x > 0:
                continue
            src += (i, j)
            dst += (j, i)
            w += (x, x)
        return cls(resseq=resseq, src=np.asarray(src, dtype=np.int64),
                   dst=np.asarray(dst, dtype=np.int64), weight=np.asarray(w, dtype=float))

    def neighbour_lists(self) -> list[list[int]]:
        """Per row, the neighbours' SEQRES indices, largest contact first."""
        out: list[list[int]] = [[] for _ in range(self.n)]
        for e in np.lexsort((-self.weight, self.src)):
            out[self.src[e]].append(int(self.resseq[self.dst[e]]))
        return out


@dataclass
class Refined:
    probs: np.ndarray                 # (n, L)
    sd: Optional[np.ndarray] = None   # (n, L), gumbel only


def refine(probs, graph: ContactGraph, method: str, *, strength: float = 0.5,
           tau: float = 0.5, samples: int = 64, seed: int = 0) -> Refined:
    """Refine ``probs`` (n residues x L labels) over ``graph``, one label at a time."""
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "none" or strength == 0 or len(graph.src) == 0:
        return Refined(p.copy(), np.zeros_like(p) if method == "gumbel" else None)
    if method == "bayes":
        return Refined(_bayes(p, graph, strength))
    return _gumbel(p, graph, strength, tau, samples, seed)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def _bayes(p: np.ndarray, g: ContactGraph, beta: float) -> np.ndarray:
    z = _logit(p)
    z_bar = _logit(p.mean(axis=0))                          # (L,)
    row_sum = np.bincount(g.src, weights=g.weight, minlength=g.n)
    w = g.weight / row_sum[g.src]
    evidence = np.zeros_like(z)
    np.add.at(evidence, g.src, w[:, None] * (z[g.dst] - z_bar))
    return 1.0 / (1.0 + np.exp(-(z + beta * evidence)))


def _gumbel(p: np.ndarray, g: ContactGraph, lam: float, tau: float, samples: int,
            seed: int) -> Refined:
    n = g.n
    # Self loop weighted as the residue's strongest contact; an isolated residue gets
    # only itself, so every sample returns its own score and it passes through.
    self_w = np.zeros(n)
    np.maximum.at(self_w, g.src, g.weight)
    self_w[self_w == 0] = 1.0
    src = np.concatenate([g.src, np.arange(n)])
    dst = np.concatenate([g.dst, np.arange(n)])
    w = np.concatenate([g.weight, self_w])

    # Sort by row so each shell is a contiguous run for reduceat; every row has its
    # self loop, so no run is empty.
    order = np.argsort(src, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    starts = np.searchsorted(src, np.arange(n))

    base = np.log(w)[:, None] + _logit(p)[dst]              # (E, L): log prior + log odds
    p_dst = p[dst]
    rng = np.random.default_rng(seed)
    draws = []
    for s0 in range(0, samples, _SAMPLE_BLOCK):
        k = min(_SAMPLE_BLOCK, samples - s0)
        logits = (base[None] + rng.gumbel(size=(k,) + base.shape)) / tau
        logits -= np.maximum.reduceat(logits, starts, axis=1)[:, src]
        e = np.exp(logits)
        a = e / np.add.reduceat(e, starts, axis=1)[:, src]
        draws.append(np.add.reduceat(a * p_dst[None], starts, axis=1))
    draws = np.concatenate(draws)                            # (S, n, L)
    mean = (1 - lam) * p + lam * draws.mean(axis=0)
    return Refined(np.clip(mean, 0.0, 1.0), lam * draws.std(axis=0))
