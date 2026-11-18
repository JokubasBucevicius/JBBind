"""Paths and hyper-parameters for the training program.

Defaults mirror ``train_multilabel.CONFIG`` so behaviour matches the original
pipeline. The repository root is computed from this file's location so the program
can be launched from anywhere, though data paths are relative and therefore still
expect the process CWD to be the repository root (the SLURM scripts set
``--chdir`` accordingly).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Repository root: program -> training -> all_class -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]

# Make ``train_multilabel`` importable as a library no matter the CWD.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Config:
    """Resolved configuration for a single training run."""

    # Data locations (relative to the repository root / process CWD).
    graph_dir: str = "all_class/graphs/unified/unified"
    emb_dir: str = "emb"
    csv_path: str = "ppi3d_filtered.csv"

    # Output / runs root. The per-run dir is ``runs_dir/<setup>/<arch>``.
    runs_dir: str = "all_class/training/runs"

    # Split + reproducibility.
    train_val_test_split: List[float] = field(default_factory=lambda: [0.7, 0.1, 0.2])
    seed: int = 42

    # Training hyper-parameters.
    num_epochs: int = 50
    batch_size: int = 256        # MLP (dense) batches
    gnn_batch_size: int = 32     # graph batches
    patience: int = 10
    focal_gamma: float = 2.0

    # Joint architecture: MLP auxiliary loss weight (floored to >= 0.5).
    mlp_loss_weight: float = 0.5

    def run_dir(self, setup_name: str, arch_name: str) -> str:
        return str(Path(self.runs_dir) / setup_name / arch_name)
