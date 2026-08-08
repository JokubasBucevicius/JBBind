"""Model architectures — copied verbatim from the research repo.

Copied from /home/jokubasb/protein_protein/train_multilabel.py
  BindingSiteMLP  lines 456-481
  BindingSiteGNN  lines 484-504
  JointMLPGNN     lines 507-551
Source SHA256: 5a2582a07a6db3fd72e95af341b633dc908221dd3e279040ed00c58ed6e07232
Copied: 2026-08-08

DO NOT EDIT. These definitions must stay bit-compatible with the trained checkpoints.
``tests/test_parity_models.py`` imports the original module and asserts the two produce
identical outputs; any edit here will fail that test.

Why copied rather than imported: ``train_multilabel`` builds a module-level ``device`` at
import (:53, forcing CUDA init), prints at import (:54), sys.path-hacks in a training-only
``cluster_weights`` module (:58-60), imports matplotlib/seaborn, and exposes NUM_LABELS /
LABEL_NAMES globals that ``setups.LabelSetup.activate()`` mutates — unusable when serving
five label setups from one process. The classes below read none of that state; they take
``output_dim`` as an argument, so they copy with zero edits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


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


@torch.no_grad()
def mlp_embed(model: BindingSiteMLP, esm: torch.Tensor) -> torch.Tensor:
    """Penultimate (64-d) activations of a BindingSiteMLP, for GNN+MLP node features.

    Copied from train_multilabel._load_protein_gnn_mlp_base lines 345-351, which runs every
    layer of ``network`` except the final Linear. Equivalent to
    ``model(esm, return_embedding=True)[1]`` but kept in the original form so the parity
    test compares like with like.
    """
    model.eval()
    x = esm
    layers = list(model.network.children())
    for layer in layers[:-1]:
        x = layer(x)
    return x
