"""Label setups.

Each :class:`LabelSetup` defines a prediction task as a combination/subset of the
four base per-residue binding columns loaded by ``train_multilabel``::

    y4 = [binds_homo_protein, binds_hetero_protein, binds_dna, binds_rna]   # indices 0..3

A setup's labels are obtained by OR-ing (``max``) groups of those base columns.
"Non-binding" is the implicit all-zero row (never an explicit column).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

# Base per-residue label columns, in the order ``train_multilabel`` loads them.
# Kept here explicitly so this module is self-describing; datamodule asserts it
# matches ``train_multilabel.LABEL_COLUMNS``.
BASE_LABEL_COLUMNS: List[str] = [
    "binds_homo_protein",
    "binds_hetero_protein",
    "binds_dna",
    "binds_rna",
]


@dataclass(frozen=True)
class LabelSetup:
    """A prediction task derived from the four base binding columns."""

    name: str
    label_names: List[str]
    # One group of base-column indices per output label; the label is the max
    # (logical OR) over its group. e.g. [[0, 1], [2, 3]] -> [protein, nucleic].
    base_index_groups: List[Sequence[int]]

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    @property
    def trigger_columns(self) -> List[str]:
        """Base columns relevant to this setup (union over all groups).

        A protein qualifies for the setup iff it has >= 1 binding residue in any
        of these columns.
        """
        idxs = sorted({i for group in self.base_index_groups for i in group})
        return [BASE_LABEL_COLUMNS[i] for i in idxs]

    def derive_labels(self, y4: torch.Tensor) -> torch.Tensor:
        """Map a base (n, 4) label tensor to this setup's (n, k) label tensor."""
        cols = [y4[:, list(group)].amax(dim=1) for group in self.base_index_groups]
        return torch.stack(cols, dim=1).to(y4.dtype)

    def has_binding(self, y4: torch.Tensor) -> bool:
        """True if any residue is positive for any of this setup's labels."""
        return bool(self.derive_labels(y4).sum() > 0)

    def activate(self) -> None:
        """Point ``train_multilabel``'s label globals at this setup.

        ``train_multilabel``'s reused helpers (``compute_val_pr_auc``,
        ``evaluate_multilabel``, the training loops) read ``NUM_LABELS`` and
        ``LABEL_NAMES`` as module globals. This is the single, deliberate shim
        that makes those generic-over-label-count helpers operate on this setup's
        ``k`` labels. Call once before any training/evaluation.
        """
        import train_multilabel as tm

        tm.NUM_LABELS = self.num_labels
        tm.LABEL_NAMES = list(self.label_names)


SETUPS = {
    "protein_nucleic": LabelSetup(
        name="protein_nucleic",
        label_names=["Protein", "Nucleic acid"],
        base_index_groups=[[0, 1], [2, 3]],
    ),
    "homo_hetero": LabelSetup(
        name="homo_hetero",
        label_names=["Homo", "Hetero"],
        base_index_groups=[[0], [1]],
    ),
    "protein": LabelSetup(
        name="protein",
        label_names=["Protein"],
        base_index_groups=[[0, 1]],
    ),
    "dna_rna": LabelSetup(
        name="dna_rna",
        label_names=["DNA", "RNA"],
        base_index_groups=[[2], [3]],
    ),
    "nucleic": LabelSetup(
        name="nucleic",
        label_names=["Nucleic acid"],
        base_index_groups=[[2, 3]],
    ),
}
