"""Label setups — which classes each model predicts.

Adapted from /home/jokubasb/protein_protein/all_class/training/program/setups.py
Copied: 2026-08-08

DEVIATION: ``LabelSetup.activate()`` is deliberately NOT carried over. In the training
program it mutates ``train_multilabel.NUM_LABELS`` / ``LABEL_NAMES`` module globals so the
generic helpers pick up the current setup's label count. A server holding five setups in
memory at once cannot use mutable global label state, and nothing in the inference path
needs it — label count comes from the checkpoint's ``label_names``.

``derive_labels`` is also dropped: it maps training's 4-column ground truth onto a setup's
k labels, and there is no ground truth at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# Base per-residue label columns, in the order train_multilabel loads them.
BASE_LABEL_COLUMNS: List[str] = [
    "binds_homo_protein",
    "binds_hetero_protein",
    "binds_dna",
    "binds_rna",
]


@dataclass(frozen=True)
class LabelSetup:
    """A prediction task: k independent per-residue binary labels."""

    name: str
    label_names: List[str]
    description: str
    #: Short blurb shown in the UI next to the selector.
    hint: str

    @property
    def num_labels(self) -> int:
        return len(self.label_names)


SETUPS: Dict[str, LabelSetup] = {
    "protein_nucleic": LabelSetup(
        name="protein_nucleic",
        label_names=["Protein", "Nucleic acid"],
        description="Does this residue bind a protein, a nucleic acid, or neither?",
        hint="Broadest task. Weakest metrics — nucleic-acid residues are ~0.3% of this "
             "setup's training data.",
    ),
    "homo_hetero": LabelSetup(
        name="homo_hetero",
        label_names=["Homo", "Hetero"],
        description="Homomeric vs heteromeric protein-protein interface.",
        hint="Distinguishes self-association from binding a different protein.",
    ),
    "protein": LabelSetup(
        name="protein",
        label_names=["Protein"],
        description="Does this residue sit at a protein-protein interface?",
        hint="Single-label protein interface prediction.",
    ),
    "dna_rna": LabelSetup(
        name="dna_rna",
        label_names=["DNA", "RNA"],
        description="DNA-binding vs RNA-binding residue.",
        hint="Trained only on the 1,630 chains with a nucleic-acid interface. "
             "DNA is predicted markedly better than RNA.",
    ),
    "nucleic": LabelSetup(
        name="nucleic",
        label_names=["Nucleic acid"],
        description="Does this residue bind DNA or RNA?",
        hint="Best-performing setup by PR-AUC.",
    ),
}

DEFAULT_SETUP = "protein_nucleic"
