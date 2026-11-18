"""Architectures.

Four per-residue multilabel architectures, each wrapping the matching training
loop from ``train_multilabel`` behind a uniform interface::

    arch = ARCHES[name](datamodule, config, setup)
    y_true, y_prob = arch.run()      # prepare -> criterion -> model -> fit -> predict
    arch.save_checkpoint(path)

All four use the sigmoid/BCE focal-loss multilabel paradigm so their benchmarks are
directly comparable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

import train_multilabel as tm

import trainer
from config import Config
from datamodule import ProteinDataModule
from setups import LabelSetup


def _predict_graphs(model, graphs, batch_size) -> Tuple[np.ndarray, np.ndarray]:
    """Run a plain GNN model over graphs -> (y_true, y_prob)."""
    if not graphs:
        raise RuntimeError("No test graphs to evaluate.")
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[i:i + batch_size]).to(tm.device)
            logits = model(batch)
            ys.append(batch.y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(ys), np.vstack(ps)


class Architecture:
    """Template: prepare -> build_criterion -> build_model -> fit -> predict_test."""

    name = "base"
    display_name = "Base"

    def __init__(self, dm: ProteinDataModule, cfg: Config, setup: LabelSetup):
        self.dm = dm
        self.cfg = cfg
        self.setup = setup
        self.model = None
        self.criterion = None
        self._train_labels: torch.Tensor | None = None
        self.model_meta: dict = {}

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        self.prepare()
        self.build_criterion()
        self.build_model()
        self.fit()
        return self.predict_test()

    def build_criterion(self) -> None:
        assert self._train_labels is not None, "prepare() must set self._train_labels"
        self.criterion = self.dm.make_criterion(self._train_labels)

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "arch": self.name,
            "setup": self.setup.name,
            "label_names": list(self.setup.label_names),
            **self.model_meta,
        }, path)

    # Subclasses implement these.
    def prepare(self) -> None: raise NotImplementedError
    def build_model(self) -> None: raise NotImplementedError
    def fit(self) -> None: raise NotImplementedError
    def predict_test(self) -> Tuple[np.ndarray, np.ndarray]: raise NotImplementedError


class MLPArch(Architecture):
    name = "mlp"
    display_name = "MLP"

    def prepare(self) -> None:
        self.train_data = self.dm.load_mlp("train")
        self.val_data = self.dm.load_mlp("val")
        self.test_data = self.dm.load_mlp("test")
        if not self.train_data or not self.val_data or not self.test_data:
            raise RuntimeError("MLP: insufficient data after loading/filtering.")
        self._train_labels = torch.cat([d[1] for d in self.train_data])
        self.mlp_config = {
            "input_dim": 1280, "hidden_dims": [1024, 512, 128, 64],
            "output_dim": self.setup.num_labels, "dropout": 0.4,
        }
        self.model_meta = {"model_config": self.mlp_config}

    def build_model(self) -> None:
        pass  # train_mlp_multilabel constructs the model from mlp_config

    def fit(self) -> None:
        self.model, _ = trainer.train_mlp(
            self.mlp_config, self.train_data, self.val_data, self.criterion,
            self.cfg.num_epochs, self.cfg.batch_size, self.cfg.patience,
            weight_map=self.dm.weight_map)

    def predict_test(self) -> Tuple[np.ndarray, np.ndarray]:
        test_X = torch.cat([d[0] for d in self.test_data])
        test_Y = torch.cat([d[1] for d in self.test_data]).numpy()
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(test_X.to(tm.device))).cpu().numpy()
        return test_Y, probs


class StructuralGNNArch(Architecture):
    """Pure structural GNN: GATv2 over 5 structural features + aa-embedding."""

    name = "gnn"
    display_name = "GNN"

    def prepare(self) -> None:
        self.train_graphs = self.dm.load_structural_graphs("train", with_weights=True)
        self.val_graphs = self.dm.load_structural_graphs("val", with_weights=False)
        self.test_graphs = self.dm.load_structural_graphs("test", with_weights=False)
        if not self.train_graphs or not self.val_graphs or not self.test_graphs:
            raise RuntimeError("GNN: insufficient graph data after filtering.")
        self._train_labels = torch.cat([g.y for g in self.train_graphs])

    def build_model(self) -> None:
        self.model = tm.BindingSiteGNN(
            input_dim=5, hidden_dim=256, heads=4, dropout=0.2,
            output_dim=self.setup.num_labels).to(tm.device)

    def fit(self) -> None:
        trainer.train_gnn(
            self.model, self.train_graphs, self.val_graphs, self.criterion,
            self.cfg.num_epochs, self.cfg.gnn_batch_size, self.cfg.patience,
            lr=0.001, model_name="GNN")

    def predict_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return _predict_graphs(self.model, self.test_graphs, self.cfg.gnn_batch_size)


class GNNMLPArch(Architecture):
    """GNN over structural + MLP-embedding features (trains a prerequisite MLP)."""

    name = "gnn_mlp"
    display_name = "GNN+MLP"

    def prepare(self) -> None:
        mlp_train = self.dm.load_mlp("train")
        mlp_val = self.dm.load_mlp("val")
        if not mlp_train or not mlp_val:
            raise RuntimeError("GNN+MLP: insufficient MLP data for the prerequisite MLP.")

        print("\n--- Prerequisite MLP (for GNN+MLP embeddings) ---")
        mlp_config = {
            "input_dim": 1280, "hidden_dims": [1024, 512, 128, 64],
            "output_dim": self.setup.num_labels, "dropout": 0.4,
        }
        mlp_criterion = self.dm.make_criterion(torch.cat([d[1] for d in mlp_train]))
        self.flat_mlp, _ = trainer.train_mlp(
            mlp_config, mlp_train, mlp_val, mlp_criterion,
            self.cfg.num_epochs, self.cfg.batch_size, self.cfg.patience,
            weight_map=self.dm.weight_map)

        print("\n--- GNN+MLP graphs ---")
        self.train_graphs = self.dm.load_gnn_mlp_graphs("train", self.flat_mlp, with_weights=True)
        self.val_graphs = self.dm.load_gnn_mlp_graphs("val", self.flat_mlp, with_weights=False)
        self.test_graphs = self.dm.load_gnn_mlp_graphs("test", self.flat_mlp, with_weights=False)
        if not self.train_graphs or not self.val_graphs or not self.test_graphs:
            raise RuntimeError("GNN+MLP: insufficient graph data after filtering.")
        self._train_labels = torch.cat([g.y for g in self.train_graphs])

    def build_model(self) -> None:
        self.model = tm.BindingSiteGNN(
            input_dim=69, hidden_dim=512, heads=4, dropout=0.5,
            output_dim=self.setup.num_labels).to(tm.device)

    def fit(self) -> None:
        trainer.train_gnn(
            self.model, self.train_graphs, self.val_graphs, self.criterion,
            self.cfg.num_epochs, self.cfg.gnn_batch_size, self.cfg.patience,
            lr=0.001, model_name="GNN+MLP")

    def predict_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return _predict_graphs(self.model, self.test_graphs, self.cfg.gnn_batch_size)


class JointArch(Architecture):
    """Joint MLP+GNN with an auxiliary MLP loss (mlp_loss_weight >= 0.5)."""

    name = "joint"
    display_name = "Joint"

    def prepare(self) -> None:
        self.train_graphs = self.dm.load_joint_graphs("train", with_weights=True)
        self.val_graphs = self.dm.load_joint_graphs("val", with_weights=False)
        self.test_graphs = self.dm.load_joint_graphs("test", with_weights=False)
        if not self.train_graphs or not self.val_graphs or not self.test_graphs:
            raise RuntimeError("Joint: insufficient graph data after filtering.")
        self._train_labels = torch.cat([g.y for g in self.train_graphs])

    def build_model(self) -> None:
        mlw = max(0.5, float(self.cfg.mlp_loss_weight))  # requirement: >= 0.5
        joint_mlp_config = {
            "input_dim": 1280, "hidden_dims": [512, 128, 64],
            "output_dim": self.setup.num_labels, "dropout": 0.4,
        }
        joint_gnn_config = {"input_dim": 5 + 64, "hidden_dim": 512, "heads": 4, "dropout": 0.3}
        self.model = tm.JointMLPGNN(
            joint_mlp_config, joint_gnn_config, mlp_loss_weight=mlw).to(tm.device)
        self.model_meta = {"mlp_loss_weight": mlw}

    def fit(self) -> None:
        trainer.train_joint(
            self.model, self.train_graphs, self.val_graphs, self.criterion,
            self.cfg.num_epochs, self.cfg.gnn_batch_size, self.cfg.patience)

    def predict_test(self) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for i in range(0, len(self.test_graphs), self.cfg.gnn_batch_size):
                batch = Batch.from_data_list(
                    self.test_graphs[i:i + self.cfg.gnn_batch_size]).to(tm.device)
                gnn_logits, _ = self.model(batch.esm_embeddings.to(tm.device), batch)
                ys.append(batch.y.cpu().numpy())
                ps.append(torch.sigmoid(gnn_logits).cpu().numpy())
        return np.vstack(ys), np.vstack(ps)


ARCHES = {
    "mlp": MLPArch,
    "gnn": StructuralGNNArch,
    "gnn_mlp": GNNMLPArch,
    "joint": JointArch,
}
