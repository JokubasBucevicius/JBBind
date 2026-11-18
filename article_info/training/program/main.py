"""Entry point: train one architecture under one label setup.

Run from the repository root (the SLURM scripts set ``--chdir`` to it)::

    python all_class/training/program/main.py --setup dna_rna --arch gnn

Outputs (model checkpoint, metrics.json, predictions.npz, plots) are written to
``all_class/training/runs/<setup>/<arch>/``.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

# config inserts the repo root on sys.path so ``train_multilabel`` is importable.
from config import Config
from setups import SETUPS
from datamodule import ProteinDataModule
from architectures import ARCHES
from benchmarks import Evaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one architecture x setup for "
                                            "multilabel binding-site prediction")
    p.add_argument("--setup", required=True, choices=list(SETUPS),
                   help="Label setup (which classes to predict)")
    p.add_argument("--arch", required=True, choices=list(ARCHES),
                   help="Architecture: mlp | gnn | gnn_mlp | joint")
    p.add_argument("--reweight", default="none",
                   choices=["none", "sequence", "binding_site"],
                   help="Per-residue cluster loss reweighting (default: none)")
    p.add_argument("--csv", default=None, help="Path to ppi3d_filtered.csv")
    p.add_argument("--save-dir", default=None,
                   help="Output dir; defaults to runs/<setup>/<arch>")
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    p.add_argument("--mlp-loss-weight", type=float, default=0.5,
                   help="Joint only; auxiliary MLP loss weight (floored to >= 0.5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config()
    if args.csv:
        cfg.csv_path = args.csv
    if args.epochs:
        cfg.num_epochs = args.epochs
    cfg.mlp_loss_weight = args.mlp_loss_weight

    setup = SETUPS[args.setup]
    save_dir = args.save_dir or cfg.run_dir(setup.name, args.arch)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    setup.activate()  # point train_multilabel's label globals at this setup

    print("=" * 60)
    print(f"SETUP={setup.name}  labels={setup.label_names}  ARCH={args.arch}  "
          f"reweight={args.reweight}")
    print(f"Output dir: {save_dir}")
    print("=" * 60)

    dm = ProteinDataModule(cfg, setup, reweight=args.reweight)
    dm.prepare()

    arch = ARCHES[args.arch](dm, cfg, setup)
    y_true, y_prob = arch.run()

    os.makedirs(save_dir, exist_ok=True)
    arch.save_checkpoint(os.path.join(save_dir, "model.pt"))

    evaluator = Evaluator(setup, args.arch, arch.display_name, save_dir)
    evaluator.evaluate(y_true, y_prob)
    evaluator.print_summary()
    evaluator.save()

    print("\nDone!")


if __name__ == "__main__":
    main()
