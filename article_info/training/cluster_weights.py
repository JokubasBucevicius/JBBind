#!/usr/bin/env python3
"""
Hierarchical cluster reweighting for the multilabel training pipeline.
=====================================================================
Corrects for uneven sampling of protein sequence space by giving every
coarse (40% identity) cluster an equal total weight of 1, regardless of how
many sequences it contains. Proteins from large, heavily sampled families
(e.g. immunoglobulins) end up with small weights; proteins from
under-represented families get larger weights.

Per-chain weight (counts taken over the TRAINING chains only):

    w = 1 / N_members(C95)
        * 1 / N_95clusters_in(C70)
        * 1 / N_70clusters_in(C40)

where, for the chain's clusters (C95, C70, C40):
  - N_members(C95)      = # training chains sharing this 95% cluster
  - N_95clusters_in(C70) = # distinct 95% clusters inside this 70% cluster
  - N_70clusters_in(C40) = # distinct 70% clusters inside this 40% cluster

This makes every 40% cluster contribute total weight 1. Weights are finally
normalized so their mean over training chains is 1.0, keeping the loss scale
comparable to the un-weighted baseline.

Two clustering schemes are supported (selected by `mode`):
  - 'sequence'     : s1_sequence_cluster_{95,70,40}        (per-chain sequence clusters)
  - 'binding_site' : s1_binding_site_cluster_data_{95,70,40} (binding-mode sub-clusters;
                     full string is the cluster id; first occurrence per chain is kept)

Training is per subunit_1 chain (embeddings exist only for s1 chains), so all
columns are read from the s1 side. Chains absent from the CSV default to weight 1.0.

All computation is vectorized pandas so it scales to 40k+ chains.
"""

import pandas as pd


# Cluster-id columns per scheme, ordered finest -> coarsest (95, 70, 40).
CLUSTER_COLUMNS = {
    'sequence': [
        's1_sequence_cluster_95',
        's1_sequence_cluster_70',
        's1_sequence_cluster_40',
    ],
    'binding_site': [
        's1_binding_site_cluster_data_95',
        's1_binding_site_cluster_data_70',
        's1_binding_site_cluster_data_40',
    ],
}


def build_chain_weights(csv_path, train_chains, mode):
    """Build a {chain -> normalized weight} map for the given reweighting scheme.

    Args:
        csv_path:     path to ppi3d_filtered.csv
        train_chains: iterable of training chain ids (subunit_1 dir names), e.g. '5j41_B'
        mode:         'none' | 'sequence' | 'binding_site'

    Returns:
        dict[str, float]. Empty dict for mode='none' (signals uniform weighting).
        Chains not present in the CSV are simply absent (callers default to 1.0).
    """
    if mode == 'none':
        return {}
    if mode not in CLUSTER_COLUMNS:
        raise ValueError(
            f"Unknown reweight mode {mode!r}; expected one of "
            f"'none', {list(CLUSTER_COLUMNS)}")

    c95, c70, c40 = CLUSTER_COLUMNS[mode]
    df = pd.read_csv(csv_path, usecols=['subunit_1', c95, c70, c40])

    # Keep only training chains, on the s1 side, one row per chain (first occurrence).
    df = df[df['subunit_1'].isin(set(train_chains))]
    df = df.drop_duplicates('subunit_1', keep='first').reset_index(drop=True)
    if len(df) == 0:
        return {}

    # Treat cluster ids as strings (binding_site ids are already strings like '31891_0').
    for c in (c95, c70, c40):
        df[c] = df[c].astype(str)

    # Hierarchical nesting counts (vectorized; one row per chain after dedup).
    n_members_c95 = df.groupby(c95)['subunit_1'].transform('size')
    n95_in_c70 = df.groupby(c70)[c95].transform('nunique')
    n70_in_c40 = df.groupby(c40)[c70].transform('nunique')

    w = 1.0 / (n_members_c95 * n95_in_c70 * n70_in_c40)
    w = w / w.mean()  # normalize mean -> 1.0 over training chains

    return dict(zip(df['subunit_1'], w.astype(float)))
