#!/usr/bin/env python3
"""
Multi-Label + Hierarchical Binding Site Prediction Pipeline
============================================================
Task: Per-residue multi-label prediction of 4 binding types:
      binds_homo_protein, binds_hetero_protein, binds_dna, binds_rna

Two approaches compared:
  A) Flat multi-label: 4 sigmoid outputs trained simultaneously
  B) Hierarchical: L1(binds_any?) → L2(protein vs nucleic?) → L3(homo/hetero or DNA/RNA?)

Three model architectures each:
  1. MLP (ESM-2 embeddings only)
  2. GNN + MLP embeddings (structural + sequence)
  3. Joint MLP+GNN (end-to-end)

Usage:
    python train_multilabel.py                    # Run everything
    python train_multilabel.py --approach flat     # Flat only
    python train_multilabel.py --approach hier     # Hierarchical only
"""

import os
import sys
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import argparse
import warnings
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    hamming_loss,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cluster-based loss reweighting lives in all_class/training/cluster_weights.py
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'all_class', 'training'))
from cluster_weights import build_chain_weights
                    
# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_LABELS = 4
LABEL_NAMES = ['Homo-protein', 'Hetero-protein', 'DNA', 'RNA']
LABEL_COLUMNS = ['binds_homo_protein', 'binds_hetero_protein', 'binds_dna', 'binds_rna']


CONFIG = {
    'data': {
        'graph_dir': 'all_class/graphs/unified/unified',
        'emb_dir': 'emb',
        'csv_path': 'ppi3d_filtered.csv',
        'train_val_test_split': [0.7, 0.1, 0.2],
        'random_seed': 42,
    },
    'training': {
        'num_epochs': 50,
        'batch_size': 512,
        'gnn_batch_size': 32,
        'patience': 10,
        'focal_gamma': 2.0,
    },
    'models': {
        'save_dir': 'test_overfit',
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def graph_subdir(protein_name):
    return f"{protein_name}"


def aggregate_atoms_to_residues(nodes_df):
    if "surface_atom" not in nodes_df.columns:
        nodes_df["surface_atom"] = (nodes_df["sas_area"] > 0.0).astype(int)

    pooling = {
        "residue_type": "first",
        "sas_area": "sum",
        "ev14": "max",
        "ev28": "max",
        "ev56": "max",
        "voromqa_sas_energy": "sum",
        "bsite_area": "sum",
        "binds_homo_protein": "max",
        "binds_hetero_protein": "max",
        "binds_dna": "max",
        "binds_rna": "max",
    }

    residue_df = nodes_df.groupby("ID_resSeq").agg(pooling).reset_index()
    return residue_df


def aggregate_edges_to_residues(edges_df):
    edges_df = edges_df[edges_df["ID1_resSeq"] != edges_df["ID2_resSeq"]].copy()
    edges_df["res1"] = edges_df[["ID1_resSeq", "ID2_resSeq"]].min(axis=1)
    edges_df["res2"] = edges_df[["ID1_resSeq", "ID2_resSeq"]].max(axis=1)

    residue_edges = edges_df.groupby(["res1", "res2"]).agg({
        "area": "sum",
        "boundary": "mean"
    }).reset_index()
    residue_edges = residue_edges.rename(columns={"res1": "ID1_resSeq", "res2": "ID2_resSeq"})
    return residue_edges


def extract_multilabels(residue_df):
    """Extract 4-label binary vector per residue. Returns float32 tensor of shape (n, 4)."""
    y = residue_df[LABEL_COLUMNS].values.astype(np.float32)
    return torch.tensor(y, dtype=torch.float32)


def load_protein_base(protein_name, graph_dir):
    """Load and aggregate protein graph data. Returns residue_df, edges_df, metadata."""
    subdir = graph_subdir(protein_name)
    nodes_path = os.path.join(graph_dir, subdir, "graph_nodes.csv")
    edges_path = os.path.join(graph_dir, subdir, "graph_links.csv")

    if not os.path.exists(nodes_path):
        return None, None, None

    nodes_df = pd.read_csv(nodes_path)

    for col in LABEL_COLUMNS:
        if col not in nodes_df.columns:
            return None, None, None

    edges_df = pd.read_csv(edges_path) if os.path.exists(edges_path) else None

    residue_df = aggregate_atoms_to_residues(nodes_df)
    if edges_df is not None:
        edges_df = aggregate_edges_to_residues(edges_df)

    surface_mask = residue_df['sas_area'] > 0
    residue_df = residue_df[surface_mask].reset_index(drop=True)

    if len(residue_df) == 0:
        return None, None, None

    if edges_df is not None and len(edges_df) > 0:
        surface_ids = set(residue_df['ID_resSeq'])
        edge_mask = (edges_df['ID1_resSeq'].isin(surface_ids) &
                     edges_df['ID2_resSeq'].isin(surface_ids))
        edges_df = edges_df[edge_mask].reset_index(drop=True)

    metadata = {'protein_name': protein_name, 'n_residues': len(residue_df)}
    return residue_df, edges_df, metadata


def load_protein_mlp(protein_name, graph_dir, emb_dir):
    """Load protein data for MLP: ESM-2 embeddings + multi-labels."""
    subdir = graph_subdir(protein_name)
    nodes_path = os.path.join(graph_dir, subdir, "graph_nodes.csv")
    embedding_path = os.path.join(emb_dir, f"{protein_name}.pt")

    if not os.path.exists(nodes_path) or not os.path.exists(embedding_path):
        return None

    try:
        emb_data = torch.load(embedding_path, map_location='cpu', weights_only=False)
        if isinstance(emb_data, dict) and 'representations' in emb_data:
            layer_keys = sorted(emb_data['representations'].keys())
            embeddings = emb_data['representations'][layer_keys[-1]]
        else:
            embeddings = emb_data

        df = pd.read_csv(nodes_path)
        for col in LABEL_COLUMNS:
            if col not in df.columns:
                return None

        residue_data = df.groupby('ID_resSeq').agg({
            'sas_area': 'sum',
            'bsite_area': 'sum',
            'binds_homo_protein': 'max',
            'binds_hetero_protein': 'max',
            'binds_dna': 'max',
            'binds_rna': 'max',
            'ev14': 'max', 'ev28': 'max', 'ev56': 'max',
            'voromqa_sas_energy': 'sum',
        }).reset_index()

        surface_mask = residue_data['sas_area'] > 0
        surface_residues = residue_data[surface_mask].copy()
        if len(surface_residues) == 0:
            return None

        id_res_seq = surface_residues['ID_resSeq'].values
        min_res_seq = id_res_seq.min()
        emb_size = embeddings.shape[0]

        embedding_indices = id_res_seq - 1 if min_res_seq >= 1 else id_res_seq
        if embedding_indices.min() < 0:
            return None

        if embedding_indices.max() >= emb_size:
            valid_mask = embedding_indices < emb_size
            if valid_mask.sum() == 0:
                return None
            embedding_indices = embedding_indices[valid_mask]
            surface_residues = surface_residues[valid_mask].reset_index(drop=True)

        filtered_embeddings = embeddings[embedding_indices, :]
        y_multi = surface_residues[LABEL_COLUMNS].values.astype(np.float32)
        y_multi = torch.tensor(y_multi, dtype=torch.float32)

        metadata = {
            'protein_name': protein_name,
            'n_residues': len(y_multi),
            'embedding_size': emb_size,
        }
        return filtered_embeddings, y_multi, metadata

    except Exception:
        return None


# ============================================================================
# GNN GRAPH CREATION
# ============================================================================

def normalize_column(values):
    values = np.array(values, dtype=np.float32)
    min_val, max_val = values.min(), values.max()
    if max_val - min_val > 0:
        return (values - min_val) / (max_val - min_val)
    return np.zeros_like(values)


def normalize_voromqa(values):
    values = np.array(values, dtype=np.float32)
    if values.min() < 0:
        values = values - values.min()
    values = np.log1p(values)
    return normalize_column(values)


def create_pyg_graph(residue_df, edges_df, extra_features=None):
    """Create PyG graph. If extra_features provided, concatenate with structural."""
    structural = np.stack([
        normalize_column(residue_df['sas_area'].values),
        normalize_voromqa(residue_df['voromqa_sas_energy'].values),
        normalize_column(residue_df['ev14'].values),
        normalize_column(residue_df['ev28'].values),
        normalize_column(residue_df['ev56'].values),
    ], axis=1).astype(np.float32)

    if extra_features is not None:
        if torch.is_tensor(extra_features):
            extra_features = extra_features.numpy()
        node_features = np.concatenate([structural, extra_features], axis=1).astype(np.float32)
    else:
        node_features = structural

    residue_types = torch.tensor(residue_df['residue_type'].values, dtype=torch.long)

    if edges_df is not None and len(edges_df) > 0:
        id_to_idx = {id_val: idx for idx, id_val in enumerate(residue_df['ID_resSeq'])}
        src = [id_to_idx[id1] for id1 in edges_df['ID1_resSeq']]
        dst = [id_to_idx[id2] for id2 in edges_df['ID2_resSeq']]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edges_df[['area', 'boundary']].values, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)

    y = extract_multilabels(residue_df)

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        residue_type=residue_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(residue_df),
    )


def _load_protein_gnn_mlp_base(protein_name, graph_dir, mlp_model, emb_dir, weight_map=None):
    """Shared loader: returns (graph_without_esm, esm_embeddings) or None."""
    try:
        residue_df, edges_df, metadata = load_protein_base(protein_name, graph_dir)
        if residue_df is None:
            return None

        embedding_path = os.path.join(emb_dir, f"{protein_name}.pt")
        if not os.path.exists(embedding_path):
            return None

        emb_data = torch.load(embedding_path, map_location='cpu', weights_only=False)
        if isinstance(emb_data, dict) and 'representations' in emb_data:
            layer_keys = sorted(emb_data['representations'].keys())
            embeddings = emb_data['representations'][layer_keys[-1]]
        else:
            embeddings = emb_data

        emb_size = embeddings.shape[0]
        id_res_seq = residue_df['ID_resSeq'].values
        min_res_seq = id_res_seq.min()
        embedding_indices = id_res_seq - 1 if min_res_seq >= 1 else id_res_seq

        valid_mask = (embedding_indices >= 0) & (embedding_indices < emb_size)
        if valid_mask.sum() == 0:
            return None

        if valid_mask.sum() < len(embedding_indices):
            embedding_indices = embedding_indices[valid_mask]
            residue_df = residue_df[valid_mask].reset_index(drop=True)
            if edges_df is not None:
                surface_ids = set(residue_df['ID_resSeq'])
                edge_mask = (edges_df['ID1_resSeq'].isin(surface_ids) &
                             edges_df['ID2_resSeq'].isin(surface_ids))
                edges_df = edges_df[edge_mask].reset_index(drop=True)

        esm_embeddings = embeddings[embedding_indices, :].clone()
        mlp_model.eval()
        with torch.no_grad():
            x = esm_embeddings.to(device)
            layers = list(mlp_model.network.children())
            for layer in layers[:-1]:
                x = layer(x)
            mlp_embeddings = x.cpu()

        graph = create_pyg_graph(residue_df, edges_df, extra_features=mlp_embeddings)
        graph.protein_name = protein_name
        if weight_map:
            w = float(weight_map.get(protein_name, 1.0))
            graph.node_weight = torch.full((graph.num_nodes,), w, dtype=torch.float32)
        return graph, esm_embeddings

    except Exception:
        return None


def load_protein_gnn_with_mlp(protein_name, graph_dir, mlp_model, emb_dir, attach_esm=False,
                              weight_map=None):
    """Load protein with structural + MLP embedding features.

    attach_esm: set True only for Joint model training (adds esm_embeddings to graph).
    weight_map: optional {chain -> weight}; when set, attaches per-node sample weights.
    """
    result = _load_protein_gnn_mlp_base(protein_name, graph_dir, mlp_model, emb_dir,
                                        weight_map=weight_map)
    if result is None:
        return None
    graph, esm_embeddings = result
    if attach_esm:
        graph.esm_embeddings = esm_embeddings
    return graph


def load_proteins_gnn_mlp(protein_list, graph_dir, mlp_model, emb_dir, desc="Loading",
                           attach_esm=False, weight_map=None):
    cache = {}
    failed = []
    for name in tqdm(protein_list, desc=desc):
        g = load_protein_gnn_with_mlp(name, graph_dir, mlp_model, emb_dir,
                                      attach_esm=attach_esm, weight_map=weight_map)
        if g is not None:
            cache[name] = g
        else:
            failed.append(name)
    return cache, failed


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MultilabelFocalLoss(nn.Module):
    """Binary focal loss applied independently per label."""
    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets, sample_weight=None):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma

        if self.pos_weight is not None:
            weight = targets * self.pos_weight + (1 - targets)
            loss = weight * focal_weight * bce
        else:
            loss = focal_weight * bce

        # Per-sample (per-residue) reweighting, broadcast over the label dim.
        # sample_weight is normalized to mean ~1.0, so .mean() keeps the loss
        # scale comparable to the un-weighted baseline.
        if sample_weight is not None:
            loss = loss * sample_weight.view(-1, 1)

        return loss.mean()


class BinaryFocalLoss(nn.Module):
    """Binary focal loss for a single output (hierarchical levels)."""
    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight  # scalar

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma

        if self.pos_weight is not None:
            weight = targets * self.pos_weight + (1 - targets)
            loss = weight * focal_weight * bce
        else:
            loss = focal_weight * bce

        return loss.mean()


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class BindingSiteMLP(nn.Module):
    def __init__(self, input_dim=1280, hidden_dims=None, output_dim=4, dropout=0.4):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 128, 64]
        self.embedding_dim = hidden_dims[-1]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, return_embedding=False):
        if return_embedding:
            for layer in list(self.network.children())[:-1]:
                x = layer(x)
            embedding = x
            logits = list(self.network.children())[-1](x)
            return logits, embedding
        return self.network(x)


class BindingSiteGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, heads=4, dropout=0.2,
                 num_amino_acids=20, aa_embed_dim=32, output_dim=4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.aa_embedding = nn.Embedding(num_amino_acids, aa_embed_dim)
        total_input_dim = input_dim + aa_embed_dim

        self.conv1 = GATv2Conv(total_input_dim, hidden_dim, heads=heads, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        aa_emb = self.aa_embedding(data.residue_type)
        x = torch.cat([x, aa_emb], dim=-1)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        return self.fc(x)


class JointMLPGNN(nn.Module):
    def __init__(self, mlp_config, gnn_config, mlp_loss_weight=0.0):
        super().__init__()
        self.mlp_loss_weight = mlp_loss_weight
        self.mlp_embedding_dim = mlp_config['hidden_dims'][-1]

        self.mlp_layers = nn.ModuleList()
        prev_dim = mlp_config['input_dim']
        for hd in mlp_config['hidden_dims']:
            self.mlp_layers.append(nn.Linear(prev_dim, hd))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(mlp_config['dropout']))
            self.mlp_layers.append(nn.LayerNorm(hd))
            prev_dim = hd
        self.mlp_head = nn.Linear(prev_dim, mlp_config['output_dim'])

        self.gnn = BindingSiteGNN(
            input_dim=gnn_config['input_dim'],
            hidden_dim=gnn_config['hidden_dim'],
            heads=gnn_config['heads'],
            dropout=gnn_config['dropout'],
            output_dim=mlp_config['output_dim'],
        )

    def mlp_forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return self.mlp_head(x), x  # logits, embeddings

    def forward(self, esm_embeddings, graph_data):
        mlp_logits, mlp_emb = self.mlp_forward(esm_embeddings)
        graph_data_copy = graph_data.clone()
        graph_data_copy.x = torch.cat([graph_data.x, mlp_emb], dim=-1)
        gnn_logits = self.gnn(graph_data_copy)
        return gnn_logits, mlp_logits

    def compute_loss(self, gnn_logits, mlp_logits, labels, criterion, sample_weight=None):
        gnn_loss = criterion(gnn_logits, labels, sample_weight=sample_weight)
        if self.mlp_loss_weight > 0:
            mlp_loss = criterion(mlp_logits, labels, sample_weight=sample_weight)
            total_loss = gnn_loss + self.mlp_loss_weight * mlp_loss
        else:
            total_loss = gnn_loss
            mlp_loss = torch.tensor(0.0)
        return total_loss, gnn_loss, mlp_loss


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_multilabel(y_true, y_pred, y_prob):
    """Compute per-label and overall metrics for multi-label classification.

    Args:
        y_true: (n, 4) binary ground truth
        y_pred: (n, 4) binary predictions
        y_prob: (n, 4) predicted probabilities
    """
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Per-label metrics
    for i, name in enumerate(LABEL_NAMES):
        yt, yp, yprob = y_true[:, i], y_pred[:, i], y_prob[:, i]
        metrics[f'F1 ({name})'] = f1_score(yt, yp, zero_division=0)
        metrics[f'Precision ({name})'] = precision_score(yt, yp, zero_division=0)
        metrics[f'Recall ({name})'] = recall_score(yt, yp, zero_division=0)
        metrics[f'Bal Acc ({name})'] = balanced_accuracy_score(yt, yp)
        if yt.sum() > 0 and yt.sum() < len(yt):
            metrics[f'ROC AUC ({name})'] = roc_auc_score(yt, yprob)
            metrics[f'PR AUC ({name})'] = average_precision_score(yt, yprob)
        else:
            metrics[f'ROC AUC ({name})'] = 0.0
            metrics[f'PR AUC ({name})'] = 0.0

    # Macro averages
    for metric_name in ['F1', 'Precision', 'Recall', 'ROC AUC', 'PR AUC', 'Bal Acc']:
        vals = [metrics[f'{metric_name} ({name})'] for name in LABEL_NAMES]
        metrics[f'{metric_name} (macro)'] = np.mean(vals)

    # Overall metrics
    metrics['Hamming Loss'] = hamming_loss(y_true, y_pred)
    metrics['Exact Match'] = np.all(y_true == y_pred, axis=1).mean()

    return metrics


def compute_val_pr_auc(y_true, y_prob):
    """Compute macro PR AUC across labels for validation monitoring."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    aucs = []
    for i in range(NUM_LABELS):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            aucs.append(average_precision_score(y_true[:, i], y_prob[:, i]))
    return np.mean(aucs) if aucs else 0.0


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_mlp_multilabel(train_data, val_data, criterion, model_config,
                         num_epochs, batch_size, patience, lr=0.001, weight_map=None):
    """Train MLP for multi-label prediction. Returns model, best_val_pr_auc."""
    model = BindingSiteMLP(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_pr_auc = 0.0
    patience_counter = 0
    best_state = None

    # Prepare batched data
    print(f"  Concatenating tensors: {len(train_data)} train / {len(val_data)} val proteins...")
    train_X = torch.cat([d[0] for d in train_data])
    train_Y = torch.cat([d[1] for d in train_data])
    val_X = torch.cat([d[0] for d in val_data])
    val_Y = torch.cat([d[1] for d in val_data])
    print(f"  train_X: {tuple(train_X.shape)}, val_X: {tuple(val_X.shape)}")

    # Per-residue sample weights (chain weight repeated per residue), aligned with train_X.
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

        epoch_loss = 0.0
        n_samples = 0
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

        train_loss = epoch_loss / n_samples

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X.to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_pr_auc = compute_val_pr_auc(val_Y.numpy(), val_probs)

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " +"
        else:
            patience_counter += 1
            marker = ""

        print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} "
              f"Val PR-AUC(macro): {val_pr_auc:.4f}{marker} "
              f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping (best PR-AUC: {best_pr_auc:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_pr_auc


def train_gnn_multilabel(model, train_graphs, val_graphs, criterion,
                         num_epochs, batch_size, patience, lr=0.001,
                         model_name="GNN"):
    """Train GNN-based model for multi-label prediction."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_pr_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train_graphs)
        epoch_loss = 0.0
        epoch_n = 0

        skipped = 0
        for i in range(0, len(train_graphs), batch_size):
            try:
                batch = Batch.from_data_list(train_graphs[i:i+batch_size]).to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y, sample_weight=getattr(batch, 'node_weight', None))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.num_nodes
                epoch_n += batch.num_nodes
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1

        train_loss = epoch_loss / epoch_n if epoch_n > 0 else float('nan')
        if skipped:
            print(f"  [{model_name}] WARNING: skipped {skipped} OOM batches this epoch")

        # Validation
        model.eval()
        val_labels, val_probs_list = [], []
        with torch.no_grad():
            for i in range(0, len(val_graphs), batch_size):
                batch = Batch.from_data_list(val_graphs[i:i+batch_size]).to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits)
                val_labels.append(batch.y.cpu().numpy())
                val_probs_list.append(probs.cpu().numpy())

        val_labels = np.vstack(val_labels)
        val_probs = np.vstack(val_probs_list)
        val_pr_auc = compute_val_pr_auc(val_labels, val_probs)

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " +"
        else:
            patience_counter += 1
            marker = ""

        print(f"  [{model_name}] Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} "
              f"Val PR-AUC(macro): {val_pr_auc:.4f}{marker} "
              f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping (best PR-AUC: {best_pr_auc:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_pr_auc


def train_joint_multilabel(model, train_graphs, val_graphs, criterion,
                           num_epochs, batch_size, patience, lr=0.002):
    """Train Joint MLP+GNN for multi-label prediction."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_pr_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train_graphs)
        epoch_loss = 0.0
        epoch_n = 0

        skipped = 0
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
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1

        train_loss = epoch_loss / epoch_n if epoch_n > 0 else float('nan')
        if skipped:
            print(f"  [Joint] WARNING: skipped {skipped} OOM batches this epoch")

        # Validation
        model.eval()
        val_labels, val_probs_list = [], []
        with torch.no_grad():
            for i in range(0, len(val_graphs), batch_size):
                batch = Batch.from_data_list(val_graphs[i:i+batch_size]).to(device)
                gnn_logits, _ = model(batch.esm_embeddings.to(device), batch)
                probs = torch.sigmoid(gnn_logits)
                val_labels.append(batch.y.cpu().numpy())
                val_probs_list.append(probs.cpu().numpy())

        val_labels = np.vstack(val_labels)
        val_probs = np.vstack(val_probs_list)
        val_pr_auc = compute_val_pr_auc(val_labels, val_probs)

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " +"
        else:
            patience_counter += 1
            marker = ""

        print(f"  [Joint] Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} "
              f"Val PR-AUC(macro): {val_pr_auc:.4f}{marker} "
              f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping (best PR-AUC: {best_pr_auc:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_pr_auc


# ============================================================================
# HIERARCHICAL TRAINING
# ============================================================================

def train_binary_mlp(train_X, train_Y, val_X, val_Y, criterion,
                     input_dim=1280, hidden_dims=None, num_epochs=15,
                     batch_size=512, patience=10, lr=0.001, level_name="L1"):
    """Train a binary MLP classifier for one hierarchical level."""
    if hidden_dims is None:
        hidden_dims = [1024, 512, 128, 64]

    model = BindingSiteMLP(input_dim=input_dim, hidden_dims=hidden_dims,
                           output_dim=1, dropout=0.4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_pr_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(len(train_X))
        tX, tY = train_X[perm], train_Y[perm]

        epoch_loss, n = 0.0, 0
        for i in range(0, len(tX), batch_size):
            bx = tX[i:i+batch_size].to(device)
            by = tY[i:i+batch_size].to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
            n += len(bx)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_X.to(device)).squeeze(-1).cpu()
            val_probs = torch.sigmoid(val_logits).numpy()
            val_true = val_Y.numpy()

        if val_true.sum() > 0 and val_true.sum() < len(val_true):
            val_pr_auc = average_precision_score(val_true, val_probs)
        else:
            val_pr_auc = 0.0

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " +"
        else:
            patience_counter += 1
            marker = ""

        print(f"    [{level_name}] Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss/n:.4f} Val PR-AUC: {val_pr_auc:.4f}{marker}")

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_pr_auc


def train_hierarchical_mlp(train_data, val_data, num_epochs, batch_size, patience, gamma):
    """Train hierarchical MLP: L1 → L2 → L3a → L3b. Returns dict of models."""
    print(f"  Concatenating tensors: {len(train_data)} train / {len(val_data)} val proteins...")
    train_X = torch.cat([d[0] for d in train_data])
    train_Y = torch.cat([d[1] for d in train_data])
    val_X = torch.cat([d[0] for d in val_data])
    val_Y = torch.cat([d[1] for d in val_data])
    print(f"  train_X: {tuple(train_X.shape)}, val_X: {tuple(val_X.shape)}")

    # Derive higher-level labels
    train_protein = (train_Y[:, 0] + train_Y[:, 1]).clamp(max=1)
    train_nucleic = (train_Y[:, 2] + train_Y[:, 3]).clamp(max=1)
    train_binds = (train_protein + train_nucleic).clamp(max=1)

    val_protein = (val_Y[:, 0] + val_Y[:, 1]).clamp(max=1)
    val_nucleic = (val_Y[:, 2] + val_Y[:, 3]).clamp(max=1)
    val_binds = (val_protein + val_nucleic).clamp(max=1)

    models = {}

    # L1: binds anything?
    print("\n  --- L1: Binding vs Non-binding ---")
    pw = (train_binds == 0).sum().float() / max(1, (train_binds == 1).sum().float())
    criterion_l1 = BinaryFocalLoss(pos_weight=pw, gamma=gamma)
    models['L1'], _ = train_binary_mlp(
        train_X, train_binds, val_X, val_binds, criterion_l1,
        num_epochs=num_epochs, batch_size=batch_size, patience=patience, level_name="L1")

    # L2: protein vs nucleic (among binding residues)
    print("\n  --- L2: Protein vs Nucleic ---")
    train_bind_mask = train_binds == 1
    val_bind_mask = val_binds == 1
    if train_bind_mask.sum() > 0 and val_bind_mask.sum() > 0:
        t_l2_X, t_l2_Y = train_X[train_bind_mask], train_nucleic[train_bind_mask]
        v_l2_X, v_l2_Y = val_X[val_bind_mask], val_nucleic[val_bind_mask]
        pw = (t_l2_Y == 0).sum().float() / max(1, (t_l2_Y == 1).sum().float())
        criterion_l2 = BinaryFocalLoss(pos_weight=pw, gamma=gamma)
        models['L2'], _ = train_binary_mlp(
            t_l2_X, t_l2_Y, v_l2_X, v_l2_Y, criterion_l2,
            num_epochs=num_epochs, batch_size=batch_size, patience=patience, level_name="L2")
    else:
        models['L2'] = None

    # L3a: homo vs hetero (among protein-binding)
    print("\n  --- L3a: Homo vs Hetero protein ---")
    train_prot_mask = train_protein == 1
    val_prot_mask = val_protein == 1
    if train_prot_mask.sum() > 0 and val_prot_mask.sum() > 0:
        t_l3a_X = train_X[train_prot_mask]
        t_l3a_Y = train_Y[train_prot_mask, 1]  # hetero_protein
        v_l3a_X = val_X[val_prot_mask]
        v_l3a_Y = val_Y[val_prot_mask, 1]
        pw = (t_l3a_Y == 0).sum().float() / max(1, (t_l3a_Y == 1).sum().float())
        criterion_l3a = BinaryFocalLoss(pos_weight=pw, gamma=gamma)
        models['L3a'], _ = train_binary_mlp(
            t_l3a_X, t_l3a_Y, v_l3a_X, v_l3a_Y, criterion_l3a,
            num_epochs=num_epochs, batch_size=batch_size, patience=patience, level_name="L3a")
    else:
        models['L3a'] = None

    # L3b: DNA vs RNA (among nucleic-binding)
    print("\n  --- L3b: DNA vs RNA ---")
    train_nuc_mask = train_nucleic == 1
    val_nuc_mask = val_nucleic == 1
    if train_nuc_mask.sum() > 0 and val_nuc_mask.sum() > 0:
        t_l3b_X = train_X[train_nuc_mask]
        t_l3b_Y = train_Y[train_nuc_mask, 3]  # rna
        v_l3b_X = val_X[val_nuc_mask]
        v_l3b_Y = val_Y[val_nuc_mask, 3]
        pw = (t_l3b_Y == 0).sum().float() / max(1, (t_l3b_Y == 1).sum().float())
        criterion_l3b = BinaryFocalLoss(pos_weight=pw, gamma=gamma)
        models['L3b'], _ = train_binary_mlp(
            t_l3b_X, t_l3b_Y, v_l3b_X, v_l3b_Y, criterion_l3b,
            num_epochs=num_epochs, batch_size=batch_size, patience=patience, level_name="L3b")
    else:
        models['L3b'] = None

    return models


def _prepare_level_graphs(graphs, level):
    """Add level_y (binary target) and level_mask (which nodes participate) to each graph.

    Levels:
        L1: all nodes, target = binds_any
        L2: only binding nodes, target = binds_nucleic (1) vs binds_protein (0)
        L3a: only protein-binding nodes, target = hetero (1) vs homo (0)
        L3b: only nucleic-binding nodes, target = RNA (1) vs DNA (0)
    """
    prepared = []
    for g in graphs:
        y = g.y  # shape (n, 4): [homo_prot, hetero_prot, dna, rna]
        protein = (y[:, 0] + y[:, 1]).clamp(max=1)
        nucleic = (y[:, 2] + y[:, 3]).clamp(max=1)
        binds_any = (protein + nucleic).clamp(max=1)

        if level == 'L1':
            mask = torch.ones(g.num_nodes, dtype=torch.bool)
            target = binds_any.float()
        elif level == 'L2':
            mask = binds_any == 1
            target = nucleic.float()  # 1=nucleic, 0=protein
        elif level == 'L3a':
            mask = protein == 1
            target = y[:, 1].float()  # 1=hetero, 0=homo
        elif level == 'L3b':
            mask = nucleic == 1
            target = y[:, 3].float()  # 1=RNA, 0=DNA
        else:
            raise ValueError(f"Unknown level: {level}")

        if mask.sum() == 0:
            continue

        gc = g.clone()
        gc.level_y = target
        gc.level_mask = mask
        prepared.append(gc)
    return prepared


def train_binary_gnn(model, train_graphs, val_graphs, criterion,
                     num_epochs, batch_size, patience, lr=0.001, level_name="L1"):
    """Train a binary GNN using per-node level_mask and level_y attributes."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_pr_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train_graphs)
        epoch_loss, epoch_n = 0.0, 0

        for i in range(0, len(train_graphs), batch_size):
            batch = Batch.from_data_list(train_graphs[i:i+batch_size]).to(device)
            optimizer.zero_grad()
            logits = model(batch).squeeze(-1)
            mask = batch.level_mask
            if mask.sum() == 0:
                continue
            loss = criterion(logits[mask], batch.level_y[mask])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * mask.sum().item()
            epoch_n += mask.sum().item()

        if epoch_n == 0:
            continue
        train_loss = epoch_loss / epoch_n

        # Validation
        model.eval()
        val_labels, val_probs_list = [], []
        with torch.no_grad():
            for i in range(0, len(val_graphs), batch_size):
                batch = Batch.from_data_list(val_graphs[i:i+batch_size]).to(device)
                logits = model(batch).squeeze(-1)
                mask = batch.level_mask
                if mask.sum() > 0:
                    val_labels.append(batch.level_y[mask].cpu().numpy())
                    val_probs_list.append(torch.sigmoid(logits[mask]).cpu().numpy())

        if val_labels:
            vl = np.concatenate(val_labels)
            vp = np.concatenate(val_probs_list)
            val_pr_auc = average_precision_score(vl, vp) if vl.sum() > 0 and vl.sum() < len(vl) else 0.0
        else:
            val_pr_auc = 0.0

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " +"
        else:
            patience_counter += 1
            marker = ""

        print(f"    [{level_name}] Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {train_loss:.4f} Val PR-AUC: {val_pr_auc:.4f}{marker}")

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_pr_auc


def train_hierarchical_gnn(train_graphs, val_graphs, num_epochs, batch_size,
                           patience, gamma, input_dim=69, hidden_dim=512, heads=4):
    """Train hierarchical GNN+MLP: L1 → L2 → L3a → L3b."""
    models = {}

    for level in ['L1', 'L2', 'L3a', 'L3b']:
        print(f"\n  --- {level} ---")
        t_graphs = _prepare_level_graphs(train_graphs, level)
        v_graphs = _prepare_level_graphs(val_graphs, level)

        if not t_graphs or not v_graphs:
            print(f"    Skipping {level}: insufficient data")
            models[level] = None
            continue

        # Compute pos_weight from training data
        all_y = torch.cat([g.level_y[g.level_mask] for g in t_graphs])
        n_pos = (all_y == 1).sum().float()
        n_neg = (all_y == 0).sum().float()
        pw = n_neg / max(1, n_pos)
        criterion = BinaryFocalLoss(pos_weight=pw, gamma=gamma)

        model = BindingSiteGNN(
            input_dim=input_dim, hidden_dim=hidden_dim, heads=heads,
            dropout=0.5, output_dim=1).to(device)

        models[level], _ = train_binary_gnn(
            model, t_graphs, v_graphs, criterion,
            num_epochs=num_epochs, batch_size=batch_size,
            patience=patience, lr=0.001, level_name=level)

    return models


def predict_hierarchical_gnn(models, graphs, batch_size=32):
    """Cascaded hierarchical GNN prediction. Returns (preds, probs) both shape (n, 4)."""
    # Collect all predictions across graphs
    all_preds = []
    all_probs = []
    all_true = []

    for g in graphs:
        n = g.num_nodes
        preds = np.zeros((n, 4), dtype=np.int32)
        probs = np.zeros((n, 4), dtype=np.float32)

        with torch.no_grad():
            g_dev = g.clone().to(device)

            # L1
            if models.get('L1') is not None:
                l1_logits = models['L1'](g_dev).squeeze(-1).cpu()
                l1_prob = torch.sigmoid(l1_logits).numpy()
                l1_pred = (l1_prob > 0.5).astype(int)
            else:
                l1_prob = np.zeros(n)
                l1_pred = np.zeros(n, dtype=int)

            binding_mask = l1_pred == 1
            if binding_mask.sum() == 0:
                all_preds.append(preds)
                all_probs.append(probs)
                all_true.append(g.y.numpy())
                continue

            # L2
            if models.get('L2') is not None:
                l2_logits = models['L2'](g_dev).squeeze(-1).cpu()
                l2_prob = torch.sigmoid(l2_logits).numpy()
                l2_pred = (l2_prob > 0.5).astype(int)
            else:
                l2_prob = np.zeros(n)
                l2_pred = np.zeros(n, dtype=int)

            protein_mask = binding_mask & (l2_pred == 0)
            nucleic_mask = binding_mask & (l2_pred == 1)

            # L3a
            if models.get('L3a') is not None:
                l3a_logits = models['L3a'](g_dev).squeeze(-1).cpu()
                l3a_prob = torch.sigmoid(l3a_logits).numpy()
                l3a_pred = (l3a_prob > 0.5).astype(int)
            else:
                l3a_prob = np.zeros(n)
                l3a_pred = np.zeros(n, dtype=int)

            # L3b
            if models.get('L3b') is not None:
                l3b_logits = models['L3b'](g_dev).squeeze(-1).cpu()
                l3b_prob = torch.sigmoid(l3b_logits).numpy()
                l3b_pred = (l3b_prob > 0.5).astype(int)
            else:
                l3b_prob = np.zeros(n)
                l3b_pred = np.zeros(n, dtype=int)

        # Assemble predictions
        preds[protein_mask & (l3a_pred == 0), 0] = 1  # homo
        preds[protein_mask & (l3a_pred == 1), 1] = 1  # hetero
        preds[nucleic_mask & (l3b_pred == 0), 2] = 1  # dna
        preds[nucleic_mask & (l3b_pred == 1), 3] = 1  # rna

        # Cascaded probabilities
        probs[:, 0] = l1_prob * (1 - l2_prob) * (1 - l3a_prob)
        probs[:, 1] = l1_prob * (1 - l2_prob) * l3a_prob
        probs[:, 2] = l1_prob * l2_prob * (1 - l3b_prob)
        probs[:, 3] = l1_prob * l2_prob * l3b_prob

        all_preds.append(preds)
        all_probs.append(probs)
        all_true.append(g.y.numpy())

    return np.vstack(all_preds), np.vstack(all_probs), np.vstack(all_true)


def train_binary_joint(model, train_graphs, val_graphs, criterion,
                       num_epochs, batch_size, patience, lr=0.002, level_name="L1"):
    """Train a binary Joint MLP+GNN using per-node level_mask and level_y."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_pr_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train_graphs)
        epoch_loss, epoch_n = 0.0, 0

        for i in range(0, len(train_graphs), batch_size):
            batch = Batch.from_data_list(train_graphs[i:i+batch_size]).to(device)
            optimizer.zero_grad()
            gnn_logits, mlp_logits = model(batch.esm_embeddings.to(device), batch)
            gnn_logits = gnn_logits.squeeze(-1)
            mlp_logits = mlp_logits.squeeze(-1)
            mask = batch.level_mask
            if mask.sum() == 0:
                continue
            gnn_loss = criterion(gnn_logits[mask], batch.level_y[mask])
            if model.mlp_loss_weight > 0:
                mlp_loss = criterion(mlp_logits[mask], batch.level_y[mask])
                loss = gnn_loss + model.mlp_loss_weight * mlp_loss
            else:
                loss = gnn_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * mask.sum().item()
            epoch_n += mask.sum().item()

        if epoch_n == 0:
            continue
        train_loss = epoch_loss / epoch_n

        # Validation
        model.eval()
        val_labels, val_probs_list = [], []
        with torch.no_grad():
            for i in range(0, len(val_graphs), batch_size):
                batch = Batch.from_data_list(val_graphs[i:i+batch_size]).to(device)
                gnn_logits, _ = model(batch.esm_embeddings.to(device), batch)
                gnn_logits = gnn_logits.squeeze(-1)
                mask = batch.level_mask
                if mask.sum() > 0:
                    val_labels.append(batch.level_y[mask].cpu().numpy())
                    val_probs_list.append(torch.sigmoid(gnn_logits[mask]).cpu().numpy())

        if val_labels:
            vl = np.concatenate(val_labels)
            vp = np.concatenate(val_probs_list)
            val_pr_auc = average_precision_score(vl, vp) if vl.sum() > 0 and vl.sum() < len(vl) else 0.0
        else:
            val_pr_auc = 0.0

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " +"
        else:
            patience_counter += 1
            marker = ""

        print(f"    [{level_name}] Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {train_loss:.4f} Val PR-AUC: {val_pr_auc:.4f}{marker}")

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_pr_auc


def train_hierarchical_joint(train_graphs, val_graphs, num_epochs, batch_size,
                             patience, gamma):
    """Train hierarchical Joint MLP+GNN: L1 → L2 → L3a → L3b."""
    models = {}

    for level in ['L1', 'L2', 'L3a', 'L3b']:
        print(f"\n  --- {level} ---")
        t_graphs = _prepare_level_graphs(train_graphs, level)
        v_graphs = _prepare_level_graphs(val_graphs, level)

        if not t_graphs or not v_graphs:
            print(f"    Skipping {level}: insufficient data")
            models[level] = None
            continue

        # Compute pos_weight
        all_y = torch.cat([g.level_y[g.level_mask] for g in t_graphs])
        n_pos = (all_y == 1).sum().float()
        n_neg = (all_y == 0).sum().float()
        pw = n_neg / max(1, n_pos)
        criterion = BinaryFocalLoss(pos_weight=pw, gamma=gamma)

        mlp_config = {
            'input_dim': 1280, 'hidden_dims': [512, 128, 64],
            'output_dim': 1, 'dropout': 0.4,
        }
        gnn_config = {
            'input_dim': 5 + 64, 'hidden_dim': 512, 'heads': 4, 'dropout': 0.3,
        }
        model = JointMLPGNN(mlp_config, gnn_config, mlp_loss_weight=0.5).to(device)

        models[level], _ = train_binary_joint(
            model, t_graphs, v_graphs, criterion,
            num_epochs=num_epochs, batch_size=batch_size,
            patience=patience, lr=0.002, level_name=level)

    return models


def predict_hierarchical_joint(models, graphs, batch_size=32):
    """Cascaded hierarchical Joint prediction. Returns (preds, probs, true_labels)."""
    all_preds = []
    all_probs = []
    all_true = []

    for g in graphs:
        n = g.num_nodes
        preds = np.zeros((n, 4), dtype=np.int32)
        probs = np.zeros((n, 4), dtype=np.float32)

        with torch.no_grad():
            g_dev = g.clone().to(device)
            esm = g.esm_embeddings.to(device)

            # L1
            if models.get('L1') is not None:
                l1_gnn, _ = models['L1'](esm, g_dev)
                l1_prob = torch.sigmoid(l1_gnn.squeeze(-1)).cpu().numpy()
                l1_pred = (l1_prob > 0.5).astype(int)
            else:
                l1_prob = np.zeros(n)
                l1_pred = np.zeros(n, dtype=int)

            binding_mask = l1_pred == 1
            if binding_mask.sum() == 0:
                all_preds.append(preds)
                all_probs.append(probs)
                all_true.append(g.y.numpy())
                continue

            # L2
            if models.get('L2') is not None:
                l2_gnn, _ = models['L2'](esm, g_dev)
                l2_prob = torch.sigmoid(l2_gnn.squeeze(-1)).cpu().numpy()
                l2_pred = (l2_prob > 0.5).astype(int)
            else:
                l2_prob = np.zeros(n)
                l2_pred = np.zeros(n, dtype=int)

            protein_mask = binding_mask & (l2_pred == 0)
            nucleic_mask = binding_mask & (l2_pred == 1)

            # L3a
            if models.get('L3a') is not None:
                l3a_gnn, _ = models['L3a'](esm, g_dev)
                l3a_prob = torch.sigmoid(l3a_gnn.squeeze(-1)).cpu().numpy()
                l3a_pred = (l3a_prob > 0.5).astype(int)
            else:
                l3a_prob = np.zeros(n)
                l3a_pred = np.zeros(n, dtype=int)

            # L3b
            if models.get('L3b') is not None:
                l3b_gnn, _ = models['L3b'](esm, g_dev)
                l3b_prob = torch.sigmoid(l3b_gnn.squeeze(-1)).cpu().numpy()
                l3b_pred = (l3b_prob > 0.5).astype(int)
            else:
                l3b_prob = np.zeros(n)
                l3b_pred = np.zeros(n, dtype=int)

        preds[protein_mask & (l3a_pred == 0), 0] = 1
        preds[protein_mask & (l3a_pred == 1), 1] = 1
        preds[nucleic_mask & (l3b_pred == 0), 2] = 1
        preds[nucleic_mask & (l3b_pred == 1), 3] = 1

        probs[:, 0] = l1_prob * (1 - l2_prob) * (1 - l3a_prob)
        probs[:, 1] = l1_prob * (1 - l2_prob) * l3a_prob
        probs[:, 2] = l1_prob * l2_prob * (1 - l3b_prob)
        probs[:, 3] = l1_prob * l2_prob * l3b_prob

        all_preds.append(preds)
        all_probs.append(probs)
        all_true.append(g.y.numpy())

    return np.vstack(all_preds), np.vstack(all_probs), np.vstack(all_true)


def predict_hierarchical(models, X):
    n = X.shape[0]
    preds = np.zeros((n, 4), dtype=np.int32)
    probs = np.zeros((n, 4), dtype=np.float32)

    with torch.no_grad():
        X_dev = X.to(device)

        # L1: binds anything?
        l1_logits = models['L1'](X_dev).squeeze(-1).cpu()
        l1_prob = torch.sigmoid(l1_logits).numpy()
        l1_pred = (l1_prob > 0.5).astype(int)

        binding_mask = l1_pred == 1
        if binding_mask.sum() == 0:
            return preds, probs

        # L2: protein vs nucleic
        if models['L2'] is not None:
            l2_logits = models['L2'](X_dev).squeeze(-1).cpu()
            l2_prob = torch.sigmoid(l2_logits).numpy()
            l2_pred = (l2_prob > 0.5).astype(int)  # 1=nucleic, 0=protein
        else:
            l2_prob = np.zeros(n)
            l2_pred = np.zeros(n, dtype=int)

        protein_mask = binding_mask & (l2_pred == 0)
        nucleic_mask = binding_mask & (l2_pred == 1)

        # L3a: homo vs hetero
        if models['L3a'] is not None:
            l3a_logits = models['L3a'](X_dev).squeeze(-1).cpu()
            l3a_prob = torch.sigmoid(l3a_logits).numpy()
            l3a_pred = (l3a_prob > 0.5).astype(int)  # 1=hetero, 0=homo
        else:
            l3a_prob = np.zeros(n)
            l3a_pred = np.zeros(n, dtype=int)

        # L3b: DNA vs RNA
        if models['L3b'] is not None:
            l3b_logits = models['L3b'](X_dev).squeeze(-1).cpu()
            l3b_prob = torch.sigmoid(l3b_logits).numpy()
            l3b_pred = (l3b_prob > 0.5).astype(int)  # 1=RNA, 0=DNA
        else:
            l3b_prob = np.zeros(n)
            l3b_pred = np.zeros(n, dtype=int)

    # Assemble final predictions
    # homo_protein: binding & protein & !hetero
    preds[protein_mask & (l3a_pred == 0), 0] = 1
    # hetero_protein: binding & protein & hetero
    preds[protein_mask & (l3a_pred == 1), 1] = 1
    # dna: binding & nucleic & !rna
    preds[nucleic_mask & (l3b_pred == 0), 2] = 1
    # rna: binding & nucleic & rna
    preds[nucleic_mask & (l3b_pred == 1), 3] = 1

    # Assemble probabilities (product of cascaded probabilities)
    probs[:, 0] = l1_prob * (1 - l2_prob) * (1 - l3a_prob)  # homo
    probs[:, 1] = l1_prob * (1 - l2_prob) * l3a_prob          # hetero
    probs[:, 2] = l1_prob * l2_prob * (1 - l3b_prob)          # dna
    probs[:, 3] = l1_prob * l2_prob * l3b_prob                  # rna

    return preds, probs


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_roc_pr_curves(results_dict, save_dir):
    """Plot ROC and PR curves. results_dict: {approach_model: (y_true, y_prob)}"""
    fig_roc, axes_roc = plt.subplots(1, 4, figsize=(20, 5))
    fig_pr, axes_pr = plt.subplots(1, 4, figsize=(20, 5))

    colors = {'MLP': 'green', 'GNN+MLP': 'red', 'Joint': 'purple'}
    styles_flat = {'MLP': '-', 'GNN+MLP': '-', 'Joint': '-'}
    styles_hier = {'MLP': '--', 'GNN+MLP': '--', 'Joint': '--'}

    for i, label_name in enumerate(LABEL_NAMES):
        for key, (y_true, y_prob) in results_dict.items():
            approach, model_name = key.split('/')
            yt = y_true[:, i]
            yp = y_prob[:, i]

            color = colors.get(model_name, 'gray')
            style = styles_hier.get(model_name, '-') if approach == 'hier' else styles_flat.get(model_name, '-')
            label_prefix = f"{approach}/{model_name}"

            if yt.sum() > 0 and yt.sum() < len(yt):
                fpr, tpr, _ = roc_curve(yt, yp)
                roc_auc = roc_auc_score(yt, yp)
                axes_roc[i].plot(fpr, tpr, color=color, linestyle=style,
                                 label=f"{label_prefix} (AUC={roc_auc:.3f})")

                prec, rec, _ = precision_recall_curve(yt, yp)
                pr_auc = average_precision_score(yt, yp)
                axes_pr[i].plot(rec, prec, color=color, linestyle=style,
                                label=f"{label_prefix} (AUC={pr_auc:.3f})")

        axes_roc[i].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes_roc[i].set_title(label_name)
        axes_roc[i].set_xlabel('FPR')
        axes_roc[i].set_ylabel('TPR')
        axes_roc[i].legend(fontsize=7)

        axes_pr[i].set_title(label_name)
        axes_pr[i].set_xlabel('Recall')
        axes_pr[i].set_ylabel('Precision')
        axes_pr[i].legend(fontsize=7)

    fig_roc.suptitle('ROC Curves (solid=flat, dashed=hierarchical)', fontsize=14)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_roc)

    fig_pr.suptitle('PR Curves (solid=flat, dashed=hierarchical)', fontsize=14)
    fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(save_dir, 'pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_pr)
    print(f"Saved ROC and PR curve plots to {save_dir}")


def plot_confusion_matrices(results_dict, save_dir):
    """Plot per-label binary confusion matrices for each model."""
    model_keys = list(results_dict.keys())
    n_models = len(model_keys)

    fig, axes = plt.subplots(n_models, NUM_LABELS, figsize=(4*NUM_LABELS, 4*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, key in enumerate(model_keys):
        y_true, y_pred = results_dict[key]
        for col, label_name in enumerate(LABEL_NAMES):
            cm = confusion_matrix(y_true[:, col], y_pred[:, col], labels=[0, 1])
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=['Non-bind', 'Bind'],
                        yticklabels=['Non-bind', 'Bind'],
                        ax=axes[row, col])
            axes[row, col].set_title(f"{key}\n{label_name}")
            axes[row, col].set_ylabel('True')
            axes[row, col].set_xlabel('Pred')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confusion matrices to {save_dir}")


def plot_comparison_summary(all_metrics, save_dir):
    """Bar chart comparing flat vs hierarchical per-label PR AUC."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, metric_prefix in enumerate(['PR AUC', 'F1']):
        x = np.arange(NUM_LABELS)
        width = 0.12
        bars = []
        labels_list = []

        for i, (key, metrics) in enumerate(all_metrics.items()):
            vals = [metrics.get(f'{metric_prefix} ({name})', 0) for name in LABEL_NAMES]
            bar = axes[ax_idx].bar(x + i * width, vals, width, label=key)
            bars.append(bar)

        axes[ax_idx].set_xlabel('Label')
        axes[ax_idx].set_ylabel(metric_prefix)
        axes[ax_idx].set_title(f'{metric_prefix} per label')
        axes[ax_idx].set_xticks(x + width * len(all_metrics) / 2)
        axes[ax_idx].set_xticklabels(LABEL_NAMES, rotation=15)
        axes[ax_idx].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'comparison_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison summary to {save_dir}")


def _strip_esm(graphs):
    """Return clones of graphs with esm_embeddings removed (safe for GNN-only batching)."""
    out = []
    for g in graphs:
        gc = g.clone()
        if hasattr(gc, 'esm_embeddings'):
            del gc.esm_embeddings
        out.append(gc)
    return out


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-label binding site prediction")
    parser.add_argument('--approach', choices=['flat', 'hier', 'both'], default='both')
    parser.add_argument('--reweight', choices=['none', 'sequence', 'binding_site'],
                    default='none',
                    help="Per-residue cluster loss reweighting scheme (see cluster_weights.py)")
    parser.add_argument('--csv', type=str, default=CONFIG['data']['csv_path'],
                    help="Path to ppi3d_filtered.csv (source of cluster columns)")
    parser.add_argument('--save-dir', type=str, default=None,
                    help="Output dir; defaults to all_class/training/runs/<reweight>")
    parser.add_argument('--overfit-proteins', type=str, default=None,
                    help="Comma-separated protein IDs for overfitting test (same data for train/val/test)")
    args = parser.parse_args()

    GRAPH_DIR = CONFIG['data']['graph_dir']
    EMB_DIR = CONFIG['data']['emb_dir']
    SAVE_DIR = args.save_dir or os.path.join('all_class', 'training', 'runs', args.reweight)
    SEED = CONFIG['data']['random_seed']
    NUM_EPOCHS = CONFIG['training']['num_epochs']
    BATCH_SIZE = CONFIG['training']['batch_size']
    GNN_BATCH_SIZE = CONFIG['training']['gnn_batch_size']
    PATIENCE = CONFIG['training']['patience']
    GAMMA = CONFIG['training']['focal_gamma']

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'flat'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'hierarchical'), exist_ok=True)

    # ── Phase 1: Data Discovery ──────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: DATA DISCOVERY")
    print("="*60)

    # Find all proteins with both graph and embedding files
    graph_proteins = set()
    if os.path.isdir(GRAPH_DIR):
        for d in os.listdir(GRAPH_DIR):
            if os.path.isfile(os.path.join(GRAPH_DIR, d, "graph_nodes.csv")):
                graph_proteins.add(d)

    emb_proteins = set()
    if os.path.isdir(EMB_DIR):
        for f in os.listdir(EMB_DIR):
            if f.endswith('.pt'):
                emb_proteins.add(f.replace('.pt', ''))

    available = sorted(graph_proteins & emb_proteins)
    print(f"Proteins with graphs: {len(graph_proteins)}")
    print(f"Proteins with embeddings: {len(emb_proteins)}")
    print(f"Available (both): {len(available)}")

    if len(available) == 0:
        print("ERROR: No proteins with both graph and embedding data found.")
        return
    # Split
    if args.overfit_proteins:
        selected = [p.strip() for p in args.overfit_proteins.split(',')]
        selected = [p for p in selected if p in set(available)]
        if not selected:
            print("ERROR: None of the specified proteins found in available data.")
            return
        train_proteins = selected
        val_proteins = selected
        test_proteins = selected
        NUM_EPOCHS = 100
        PATIENCE = 50
        print(f"\nOVERFIT MODE: {len(selected)} proteins used for train/val/test")
    else:
        train_val, test_proteins = train_test_split(
            available, test_size=CONFIG['data']['train_val_test_split'][2], random_state=SEED)
        train_proteins, val_proteins = train_test_split(
            train_val,
            test_size=CONFIG['data']['train_val_test_split'][1] / (
                CONFIG['data']['train_val_test_split'][0] + CONFIG['data']['train_val_test_split'][1]),
            random_state=SEED)
        print(f"\nSplit: {len(train_proteins)} train / {len(val_proteins)} val / {len(test_proteins)} test")

    # ── Cluster reweighting: build per-chain sample weights ──────────────
    # Counts computed over the TRAINING chains only (no val/test leakage).
    weight_map = build_chain_weights(args.csv, train_proteins, args.reweight)
    if weight_map:
        ws = np.array(list(weight_map.values()))
        print(f"\nReweighting='{args.reweight}': {len(weight_map)}/{len(train_proteins)} "
              f"train chains mapped | weight mean={ws.mean():.3f} "
              f"min={ws.min():.4f} max={ws.max():.3f} "
              f"(unmapped chains default to 1.0)")
    else:
        print(f"\nReweighting='{args.reweight}': uniform (no per-sample weighting)")

    # ── Phase 2: Load MLP Data ───────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: LOADING DATA")
    print("="*60)

    def load_split(proteins, desc):
        data = []
        failed = []
        for name in tqdm(proteins, desc=desc):
            result = load_protein_mlp(name, GRAPH_DIR, EMB_DIR)
            if result is not None:
                data.append(result)
            else:
                failed.append(name)
        return data, failed

    train_data, train_failed = load_split(train_proteins, "Loading train")
    val_data, val_failed = load_split(val_proteins, "Loading val")
    test_data, test_failed = load_split(test_proteins, "Loading test")

    print(f"\nLoaded: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")
    print(f"Failed: {len(train_failed)} / {len(val_failed)} / {len(test_failed)}")

    if len(train_data) == 0:
        print("ERROR: No training data loaded.")
        return

    # Label distribution
    all_train_Y = torch.cat([d[1] for d in train_data])
    print(f"\nTraining label distribution (per residue):")
    print(f"  Total residues: {len(all_train_Y)}")
    for i, name in enumerate(LABEL_NAMES):
        n_pos = int(all_train_Y[:, i].sum())
        print(f"  {name}: {n_pos} ({100*n_pos/len(all_train_Y):.2f}%)")

    # Co-occurrence
    print(f"\nLabel co-occurrence (training):")
    for i, name_i in enumerate(LABEL_NAMES):
        for j, name_j in enumerate(LABEL_NAMES):
            if j > i:
                both = int((all_train_Y[:, i] * all_train_Y[:, j]).sum())
                if both > 0:
                    print(f"  {name_i} & {name_j}: {both}")

    # Compute pos_weights for focal loss
    pos_weights = []
    for i in range(NUM_LABELS):
        n_pos = all_train_Y[:, i].sum().item()
        n_neg = len(all_train_Y) - n_pos
        pw = n_neg / max(1, n_pos)
        pos_weights.append(pw)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    print(f"\npos_weights: {[f'{w:.1f}' for w in pos_weights]}")

    criterion_flat = MultilabelFocalLoss(pos_weight=pos_weight_tensor, gamma=GAMMA)

    # Track all results for comparison
    all_results_curves = {}  # {key: (y_true, y_prob)}
    all_results_cm = {}      # {key: (y_true, y_pred)}
    all_metrics = {}
    flat_mlp = None  # will be set by flat training, reused for hier GNN embeddings

    # ── Phase 3: Flat Multi-Label Training ───────────────────────────────
    if args.approach in ('flat', 'both'):
        print("\n" + "="*60)
        print("PHASE 3: FLAT MULTI-LABEL TRAINING")
        print("="*60)

        # 3a. MLP
        print("\n--- Flat MLP ---")
        mlp_config = {'input_dim': 1280, 'hidden_dims': [1024, 512, 128, 64],
                       'output_dim': NUM_LABELS, 'dropout': 0.4}
        flat_mlp, flat_mlp_pr_auc = train_mlp_multilabel(
            train_data, val_data, criterion_flat, mlp_config,
            NUM_EPOCHS, BATCH_SIZE, PATIENCE, weight_map=weight_map)

        torch.save({
            'model_state_dict': flat_mlp.state_dict(),
            'model_config': mlp_config,
            'label_names': LABEL_NAMES,
        }, os.path.join(SAVE_DIR, 'flat', 'MLP_flat.pt'))

        # 3b. GNN + MLP embeddings
        print("\n--- Flat GNN+MLP ---")
        # Load GNN graphs with MLP embeddings (attach_esm=True so Joint can reuse these)
        train_gnn_cache, _ = load_proteins_gnn_mlp(
            [d[2]['protein_name'] for d in train_data], GRAPH_DIR, flat_mlp, EMB_DIR,
            "Train GNN+MLP", attach_esm=True, weight_map=weight_map)
        val_gnn_cache, _ = load_proteins_gnn_mlp(
            [d[2]['protein_name'] for d in val_data], GRAPH_DIR, flat_mlp, EMB_DIR,
            "Val GNN+MLP", attach_esm=True)

        train_gnn_graphs = list(train_gnn_cache.values())
        val_gnn_graphs = list(val_gnn_cache.values())

        if train_gnn_graphs and val_gnn_graphs:
            gnn_mlp_model = BindingSiteGNN(
                input_dim=69, hidden_dim=512, heads=4, dropout=0.5,
                output_dim=NUM_LABELS).to(device)

            flat_gnn_mlp_pr_auc = train_gnn_multilabel(
                gnn_mlp_model, _strip_esm(train_gnn_graphs), _strip_esm(val_gnn_graphs),
                criterion_flat, NUM_EPOCHS, GNN_BATCH_SIZE, PATIENCE, lr=0.001,
                model_name="GNN+MLP")

            torch.save({
                'model_state_dict': gnn_mlp_model.state_dict(),
                'label_names': LABEL_NAMES,
            }, os.path.join(SAVE_DIR, 'flat', 'GNN_MLP_flat.pt'))
        else:
            gnn_mlp_model = None
            print("  Skipping GNN+MLP: insufficient graph data")

        # 3c. Joint MLP+GNN
        print("\n--- Flat Joint ---")
        if train_gnn_graphs and val_gnn_graphs:
            joint_mlp_config = {
                'input_dim': 1280, 'hidden_dims': [512, 128, 64],
                'output_dim': NUM_LABELS, 'dropout': 0.4,
            }
            joint_gnn_config = {
                'input_dim': 5 + 64, 'hidden_dim': 512, 'heads': 4, 'dropout': 0.3,
            }
            joint_model = JointMLPGNN(joint_mlp_config, joint_gnn_config,
                                       mlp_loss_weight=0.5).to(device)

            # Need graphs with ESM embeddings attached
            # Reload with structural-only features (5D) for Joint model
            def make_joint_graphs(gnn_cache):
                graphs = []
                for name, g in gnn_cache.items():
                    # Reuse the already-filtered graph; slice to structural features only.
                    # Reloading via load_protein_base gives a different node count because
                    # it doesn't apply the embedding-index valid_mask used during loading.
                    graph = g.clone()
                    graph.x = g.x[:, :5]  # first 5 cols are structural; rest are MLP emb
                    graphs.append(graph)
                return graphs

            joint_train = make_joint_graphs(train_gnn_cache)
            joint_val = make_joint_graphs(val_gnn_cache)

            flat_joint_pr_auc = train_joint_multilabel(
                joint_model, joint_train, joint_val, criterion_flat,
                NUM_EPOCHS, GNN_BATCH_SIZE, PATIENCE)

            torch.save({
                'model_state_dict': joint_model.state_dict(),
                'label_names': LABEL_NAMES,
            }, os.path.join(SAVE_DIR, 'flat', 'Joint_flat.pt'))
        else:
            joint_model = None

        # ── Flat Evaluation ──
        print("\n--- Flat Evaluation on Test Set ---")
        test_X = torch.cat([d[0] for d in test_data])
        test_Y = torch.cat([d[1] for d in test_data]).numpy()

        # MLP
        flat_mlp.eval()
        with torch.no_grad():
            mlp_probs = torch.sigmoid(flat_mlp(test_X.to(device))).cpu().numpy()
        mlp_preds = (mlp_probs > 0.5).astype(int)
        all_results_curves['flat/MLP'] = (test_Y, mlp_probs)
        all_results_cm['flat/MLP'] = (test_Y, mlp_preds)
        all_metrics['flat/MLP'] = evaluate_multilabel(test_Y, mlp_preds, mlp_probs)

        # GNN+MLP
        if gnn_mlp_model is not None:
            test_gnn_cache, _ = load_proteins_gnn_mlp(
                [d[2]['protein_name'] for d in test_data], GRAPH_DIR, flat_mlp, EMB_DIR,
                "Test GNN+MLP", attach_esm=True)
            test_gnn_graphs = list(test_gnn_cache.values())
            if test_gnn_graphs:
                gnn_mlp_model.eval()
                gnn_labels, gnn_probs_list = [], []
                test_gnn_stripped = _strip_esm(test_gnn_graphs)
                with torch.no_grad():
                    for i in range(0, len(test_gnn_stripped), GNN_BATCH_SIZE):
                        batch = Batch.from_data_list(test_gnn_stripped[i:i+GNN_BATCH_SIZE]).to(device)
                        logits = gnn_mlp_model(batch)
                        gnn_labels.append(batch.y.cpu().numpy())
                        gnn_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                gnn_test_Y = np.vstack(gnn_labels)
                gnn_test_probs = np.vstack(gnn_probs_list)
                gnn_test_preds = (gnn_test_probs > 0.5).astype(int)
                all_results_curves['flat/GNN+MLP'] = (gnn_test_Y, gnn_test_probs)
                all_results_cm['flat/GNN+MLP'] = (gnn_test_Y, gnn_test_preds)
                all_metrics['flat/GNN+MLP'] = evaluate_multilabel(gnn_test_Y, gnn_test_preds, gnn_test_probs)

        # Joint
        if joint_model is not None:
            test_joint = make_joint_graphs(test_gnn_cache) if test_gnn_graphs else []
            if test_joint:
                joint_model.eval()
                j_labels, j_probs_list = [], []
                with torch.no_grad():
                    for i in range(0, len(test_joint), GNN_BATCH_SIZE):
                        batch = Batch.from_data_list(test_joint[i:i+GNN_BATCH_SIZE]).to(device)
                        gnn_logits, _ = joint_model(batch.esm_embeddings.to(device), batch)
                        j_labels.append(batch.y.cpu().numpy())
                        j_probs_list.append(torch.sigmoid(gnn_logits).cpu().numpy())
                j_test_Y = np.vstack(j_labels)
                j_test_probs = np.vstack(j_probs_list)
                j_test_preds = (j_test_probs > 0.5).astype(int)
                all_results_curves['flat/Joint'] = (j_test_Y, j_test_probs)
                all_results_cm['flat/Joint'] = (j_test_Y, j_test_preds)
                all_metrics['flat/Joint'] = evaluate_multilabel(j_test_Y, j_test_preds, j_test_probs)

    # ── Phase 4: Hierarchical Training ───────────────────────────────────
    if args.approach in ('hier', 'both'):
        print("\n" + "="*60)
        print("PHASE 4: HIERARCHICAL TRAINING")
        print("="*60)

        print("\n--- Hierarchical MLP ---")
        hier_models_mlp = train_hierarchical_mlp(
            train_data, val_data, NUM_EPOCHS, BATCH_SIZE, PATIENCE, GAMMA)

        # Save
        for level, model in hier_models_mlp.items():
            if model is not None:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'level': level,
                }, os.path.join(SAVE_DIR, 'hierarchical', f'MLP_{level}.pt'))

        # Evaluate hierarchical MLP
        print("\n--- Hierarchical MLP Evaluation on Test Set ---")
        test_X = torch.cat([d[0] for d in test_data])
        test_Y = torch.cat([d[1] for d in test_data]).numpy()

        hier_preds, hier_probs = predict_hierarchical(hier_models_mlp, test_X)
        all_results_curves['hier/MLP'] = (test_Y, hier_probs)
        all_results_cm['hier/MLP'] = (test_Y, hier_preds)
        all_metrics['hier/MLP'] = evaluate_multilabel(test_Y, hier_preds, hier_probs)

        # ── Hierarchical GNN+MLP ──
        print("\n--- Hierarchical GNN+MLP ---")
        # Use flat MLP for embeddings if available, otherwise use hier L1 MLP
        emb_mlp = flat_mlp if flat_mlp is not None else hier_models_mlp.get('L1')
        if emb_mlp is None:
            # Train a quick flat MLP just for embeddings
            mlp_cfg = {'input_dim': 1280, 'hidden_dims': [1024, 512, 128, 64],
                       'output_dim': NUM_LABELS, 'dropout': 0.4}
            emb_mlp, _ = train_mlp_multilabel(
                train_data, val_data, criterion_flat,
                mlp_cfg, NUM_EPOCHS, BATCH_SIZE, PATIENCE)

        hier_train_gnn_cache, _ = load_proteins_gnn_mlp(
            [d[2]['protein_name'] for d in train_data], GRAPH_DIR, emb_mlp, EMB_DIR,
            "Hier Train GNN+MLP", attach_esm=True)
        hier_val_gnn_cache, _ = load_proteins_gnn_mlp(
            [d[2]['protein_name'] for d in val_data], GRAPH_DIR, emb_mlp, EMB_DIR,
            "Hier Val GNN+MLP", attach_esm=True)

        hier_train_gnn_graphs = list(hier_train_gnn_cache.values())
        hier_val_gnn_graphs = list(hier_val_gnn_cache.values())

        if hier_train_gnn_graphs and hier_val_gnn_graphs:
            hier_models_gnn = train_hierarchical_gnn(
                _strip_esm(hier_train_gnn_graphs), _strip_esm(hier_val_gnn_graphs),
                NUM_EPOCHS, GNN_BATCH_SIZE, PATIENCE, GAMMA, input_dim=69)

            for level, model in hier_models_gnn.items():
                if model is not None:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'level': level,
                    }, os.path.join(SAVE_DIR, 'hierarchical', f'GNN_MLP_{level}.pt'))

            # Evaluate
            print("\n--- Hierarchical GNN+MLP Evaluation ---")
            hier_test_gnn_cache, _ = load_proteins_gnn_mlp(
                [d[2]['protein_name'] for d in test_data], GRAPH_DIR, emb_mlp, EMB_DIR,
                "Hier Test GNN+MLP", attach_esm=True)
            hier_test_gnn_graphs = list(hier_test_gnn_cache.values())

            if hier_test_gnn_graphs:
                gnn_preds, gnn_probs, gnn_true = predict_hierarchical_gnn(
                    hier_models_gnn, _strip_esm(hier_test_gnn_graphs))
                all_results_curves['hier/GNN+MLP'] = (gnn_true, gnn_probs)
                all_results_cm['hier/GNN+MLP'] = (gnn_true, gnn_preds)
                all_metrics['hier/GNN+MLP'] = evaluate_multilabel(gnn_true, gnn_preds, gnn_probs)
        else:
            print("  Skipping: insufficient graph data")

        # ── Hierarchical Joint ──
        print("\n--- Hierarchical Joint ---")
        if hier_train_gnn_graphs and hier_val_gnn_graphs:
            # Prepare Joint graphs (5D structural + ESM embeddings, no MLP embeddings in x)
            def make_hier_joint_graphs(gnn_cache):
                graphs = []
                for name, g in gnn_cache.items():
                    graph = g.clone()
                    graph.x = g.x[:, :5]
                    graphs.append(graph)
                return graphs

            joint_hier_train = make_hier_joint_graphs(hier_train_gnn_cache)
            joint_hier_val = make_hier_joint_graphs(hier_val_gnn_cache)

            if joint_hier_train and joint_hier_val:
                hier_models_joint = train_hierarchical_joint(
                    joint_hier_train, joint_hier_val,
                    NUM_EPOCHS, GNN_BATCH_SIZE, PATIENCE, GAMMA)

                for level, model in hier_models_joint.items():
                    if model is not None:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'level': level,
                        }, os.path.join(SAVE_DIR, 'hierarchical', f'Joint_{level}.pt'))

                # Evaluate
                print("\n--- Hierarchical Joint Evaluation ---")
                joint_hier_test = make_hier_joint_graphs(hier_test_gnn_cache) if hier_test_gnn_graphs else []
                if joint_hier_test:
                    j_preds, j_probs, j_true = predict_hierarchical_joint(
                        hier_models_joint, joint_hier_test)
                    all_results_curves['hier/Joint'] = (j_true, j_probs)
                    all_results_cm['hier/Joint'] = (j_true, j_preds)
                    all_metrics['hier/Joint'] = evaluate_multilabel(j_true, j_preds, j_probs)
            else:
                print("  Skipping: insufficient Joint graph data")
        else:
            print("  Skipping: no GNN graphs available")

    # ── Phase 5: Comparison & Visualization ──────────────────────────────
    print("\n" + "="*60)
    print("PHASE 5: COMPARISON & VISUALIZATION")
    print("="*60)

    # Print metrics table
    print("\n" + "-"*80)
    print(f"{'Metric':<30}", end="")
    for key in all_metrics:
        print(f"{key:>18}", end="")
    print()
    print("-"*80)

    all_metric_names = set()
    for m in all_metrics.values():
        all_metric_names.update(m.keys())

    for metric_name in sorted(all_metric_names):
        print(f"{metric_name:<30}", end="")
        for key in all_metrics:
            val = all_metrics[key].get(metric_name, '-')
            if isinstance(val, float):
                print(f"{val:>18.4f}", end="")
            else:
                print(f"{str(val):>18}", end="")
        print()

    # Plot
    if all_results_curves:
        plot_roc_pr_curves(all_results_curves, SAVE_DIR)

    if all_results_cm:
        plot_confusion_matrices(all_results_cm, SAVE_DIR)

    if all_metrics:
        plot_comparison_summary(all_metrics, SAVE_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
