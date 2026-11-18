"""Training loops with per-epoch train *and* validation PR-AUC.

These mirror the loops in ``train_multilabel`` but additionally report an online
**train** macro-PR-AUC each epoch -- accumulated from the forward passes already
done during the training step (model in train mode, so it includes dropout and is
a slight underestimate, but costs essentially nothing). Reporting train alongside
val makes the train/val gap visible: a large gap => overfitting (early stopping
earns its keep); both flat and close => feature/capacity limited.

Kept in the program (not in ``train_multilabel``) so the original script is
unchanged. Reuses ``train_multilabel``'s device, model classes and
``compute_val_pr_auc`` (which reads the label-count globals set by
``LabelSetup.activate()``).
"""

from __future__ import annotations

import copy
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import optim
from torch_geometric.data import Batch

import train_multilabel as tm


def _macro_pr_auc(labels_list: List[np.ndarray], probs_list: List[np.ndarray]) -> float:
    if not labels_list:
        return float("nan")
    return tm.compute_val_pr_auc(np.vstack(labels_list), np.vstack(probs_list))


def train_mlp(model_config: dict, train_data, val_data, criterion,
              num_epochs: int, batch_size: int, patience: int,
              lr: float = 0.001, weight_map=None) -> Tuple[torch.nn.Module, float]:
    """Train a BindingSiteMLP; returns (model, best_val_pr_auc)."""
    device = tm.device
    model = tm.BindingSiteMLP(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_pr_auc, patience_counter, best_state = 0.0, 0, None

    print(f"  Concatenating tensors: {len(train_data)} train / {len(val_data)} val proteins...")
    train_X = torch.cat([d[0] for d in train_data])
    train_Y = torch.cat([d[1] for d in train_data])
    val_X = torch.cat([d[0] for d in val_data])
    val_Y = torch.cat([d[1] for d in val_data])
    print(f"  train_X: {tuple(train_X.shape)}, val_X: {tuple(val_X.shape)}")

    if weight_map:
        train_W = torch.cat([
            torch.full((d[1].shape[0],), float(weight_map.get(d[2]['protein_name'], 1.0)))
            for d in train_data
        ])
    else:
        train_W = None

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(len(train_X))
        train_X, train_Y = train_X[perm], train_Y[perm]
        if train_W is not None:
            train_W = train_W[perm]

        epoch_loss, n_samples = 0.0, 0
        tr_labels, tr_probs = [], []
        for i in range(0, len(train_X), batch_size):
            bx = train_X[i:i+batch_size].to(device)
            by = train_Y[i:i+batch_size].to(device)
            bw = train_W[i:i+batch_size].to(device) if train_W is not None else None
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by, sample_weight=bw)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
            n_samples += len(bx)
            tr_labels.append(by.detach().cpu().numpy())
            tr_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

        train_loss = epoch_loss / n_samples
        train_pr_auc = _macro_pr_auc(tr_labels, tr_probs)

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(val_X.to(device))).cpu().numpy()
        val_pr_auc = tm.compute_val_pr_auc(val_Y.numpy(), val_probs)

        best_pr_auc, patience_counter, best_state, marker = _update_best(
            val_pr_auc, best_pr_auc, patience_counter, best_state, model)

        print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} "
              f"Train PR-AUC(macro): {train_pr_auc:.4f} "
              f"Val PR-AUC(macro): {val_pr_auc:.4f}{marker} "
              f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping (best PR-AUC: {best_pr_auc:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_pr_auc


def train_gnn(model, train_graphs, val_graphs, criterion,
              num_epochs: int, batch_size: int, patience: int,
              lr: float = 0.001, model_name: str = "GNN") -> float:
    """Train a graph model (forward = ``model(batch)``); returns best_val_pr_auc."""
    device = tm.device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_pr_auc, patience_counter, best_state = 0.0, 0, None

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train_graphs)
        epoch_loss, epoch_n, skipped = 0.0, 0, 0
        tr_labels, tr_probs = [], []
        for i in range(0, len(train_graphs), batch_size):
            try:
                batch = Batch.from_data_list(train_graphs[i:i+batch_size]).to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y,
                                 sample_weight=getattr(batch, 'node_weight', None))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.num_nodes
                epoch_n += batch.num_nodes
                tr_labels.append(batch.y.detach().cpu().numpy())
                tr_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1

        train_loss = epoch_loss / epoch_n if epoch_n > 0 else float('nan')
        train_pr_auc = _macro_pr_auc(tr_labels, tr_probs)
        if skipped:
            print(f"  [{model_name}] WARNING: skipped {skipped} OOM batches this epoch")

        val_pr_auc = _eval_graphs_pr_auc(model, val_graphs, batch_size, joint=False)

        best_pr_auc, patience_counter, best_state, marker = _update_best(
            val_pr_auc, best_pr_auc, patience_counter, best_state, model)

        print(f"  [{model_name}] Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} "
              f"Train PR-AUC(macro): {train_pr_auc:.4f} "
              f"Val PR-AUC(macro): {val_pr_auc:.4f}{marker} "
              f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping (best PR-AUC: {best_pr_auc:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_pr_auc


def train_joint(model, train_graphs, val_graphs, criterion,
                num_epochs: int, batch_size: int, patience: int,
                lr: float = 0.002) -> float:
    """Train a JointMLPGNN (forward = ``model(esm, batch)``); returns best_val_pr_auc."""
    device = tm.device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_pr_auc, patience_counter, best_state = 0.0, 0, None

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train_graphs)
        epoch_loss, epoch_n, skipped = 0.0, 0, 0
        tr_labels, tr_probs = [], []
        for i in range(0, len(train_graphs), batch_size):
            try:
                batch = Batch.from_data_list(train_graphs[i:i+batch_size]).to(device)
                optimizer.zero_grad()
                gnn_logits, mlp_logits = model(batch.esm_embeddings.to(device), batch)
                total_loss, _, _ = model.compute_loss(
                    gnn_logits, mlp_logits, batch.y, criterion,
                    sample_weight=getattr(batch, 'node_weight', None))
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item() * batch.num_nodes
                epoch_n += batch.num_nodes
                tr_labels.append(batch.y.detach().cpu().numpy())
                tr_probs.append(torch.sigmoid(gnn_logits).detach().cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1

        train_loss = epoch_loss / epoch_n if epoch_n > 0 else float('nan')
        train_pr_auc = _macro_pr_auc(tr_labels, tr_probs)
        if skipped:
            print(f"  [Joint] WARNING: skipped {skipped} OOM batches this epoch")

        val_pr_auc = _eval_graphs_pr_auc(model, val_graphs, batch_size, joint=True)

        best_pr_auc, patience_counter, best_state, marker = _update_best(
            val_pr_auc, best_pr_auc, patience_counter, best_state, model)

        print(f"  [Joint] Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} "
              f"Train PR-AUC(macro): {train_pr_auc:.4f} "
              f"Val PR-AUC(macro): {val_pr_auc:.4f}{marker} "
              f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping (best PR-AUC: {best_pr_auc:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_pr_auc


# --------------------------------------------------------------------- helpers
def _update_best(val_pr_auc, best_pr_auc, patience_counter, best_state, model):
    if val_pr_auc > best_pr_auc:
        return val_pr_auc, 0, copy.deepcopy(model.state_dict()), " +"
    return best_pr_auc, patience_counter + 1, best_state, ""


def _eval_graphs_pr_auc(model, graphs, batch_size, joint: bool) -> float:
    device = tm.device
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[i:i+batch_size]).to(device)
            if joint:
                logits, _ = model(batch.esm_embeddings.to(device), batch)
            else:
                logits = model(batch)
            labels.append(batch.y.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return tm.compute_val_pr_auc(np.vstack(labels), np.vstack(probs))
