"""Unified setup x architecture multilabel binding-site training program.

A modular, class-based front-end over the building blocks in ``train_multilabel.py``
(models, loss, loaders, training loops, metrics). It trains one architecture
(``mlp`` | ``gnn`` | ``gnn_mlp`` | ``joint``) under one label setup
(``protein_nucleic`` | ``homo_hetero`` | ``protein`` | ``dna_rna`` | ``nucleic``).

Run from the repository root, e.g.::

    python all_class/training/program/main.py --setup dna_rna --arch gnn
"""
