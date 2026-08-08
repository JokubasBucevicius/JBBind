# Verification results

Generated 2026-08-08T17:10:21+03:00 on gpuhv.bioinfo.lt.

## Tier 3a — forked voronota script reproduces the training features
```
  ok   6fqz_A  atoms=3719  (bsite_area differs as expected: yes)
  ok   2ilk_A  atoms=1264  (bsite_area differs as expected: yes)
  ok   2ov8_A  atoms=1768  (bsite_area differs as expected: yes)
  ok   5a6y_B  atoms=814  (bsite_area differs as expected: yes)
  ok   6b58_B  atoms=533  (bsite_area differs as expected: yes)
  ok   5cz2_A  atoms=1214  (bsite_area differs as expected: yes)
  ok   4msv_B  atoms=1216  (bsite_area differs as expected: yes)
Tier 3a: 40/40 chains reproduce the training features exactly (0 skipped for missing source PDB)
```

## Tier 3b — provenance shift (PPI3D training files vs RCSB asymmetric unit)
```
chains compared        39 of 40 attempted
Pearson r (probabilities)    median 1.0000   p10 0.9185   p90 1.0000
max |Δprobability|           median 0.0000   p10 0.0000   p90 0.0710
mean |Δprobability|          median 0.0000   p10 0.0000   p90 0.0088
Spearman ρ (SASA)            median 1.0000   p10 0.9956   p90 1.0000

Interpretation: r near 1.0 and a small max|Δ| mean serving from RCSB reproduces the training-time inputs closely. A long tail means the per-graph min-max normalisation is sensitive to the atom set, and the biological-assembly setting is worth trying.

not compared:
```

## Tier 3c — SEQRES renumbering audit
```
=== Tier 3c: SEQRES renumbering audit
chains sampled                  200
  training numbering sound      190  (10 pre-existing training defects excluded)
  match                       188
  match_after_chain_remap     1
  match_partial_coverage      1
  training_defect             10
AGREEMENT ON SOUND CHAINS       190/190 = 100.0%
  (of which 1 needed chain re-resolution: PPI3D subunit labels are not RCSB chain labels)
full results -> /tmp/claude-503000028/-home-jokubasb/465a4035-0d13-4c9b-9c02-260b85ea9158/scratchpad/tier3c2.json
```

## P0.5 — gnn_mlp embedder recovery
```
embedder: /home/jokubasb/protein_protein/all_class/training/runs/dna_rna/mlp/model.pt  config={'input_dim': 1280, 'hidden_dims': [1024, 512, 128, 64], 'output_dim': 2, 'dropout': 0.4}
gnn_mlp:  /home/jokubasb/protein_protein/all_class/training/runs/dna_rna/gnn_mlp/model.pt  labels=['DNA', 'RNA']
GNN+MLP[test/dna_rna]:   0%|          | 0/326 [00:00<?, ?it/s]GNN+MLP[test/dna_rna]:   0%|          | 1/326 [00:00<01:58,  2.75it/s]GNN+MLP[test/dna_rna]:   1%|          | 3/326 [00:00<00:42,  7.51it/s]GNN+MLP[test/dna_rna]:   2%|▏         | 5/326 [00:00<00:32,  9.89it/s]GNN+MLP[test/dna_rna]:   2%|▏         | 7/326 [00:00<00:27, 11.54it/s]GNN+MLP[test/dna_rna]:   3%|▎         | 9/326 [00:00<00:25, 12.51it/s]GNN+MLP[test/dna_rna]:   3%|▎         | 11/326 [00:01<00:26, 11.99it/s]GNN+MLP[test/dna_rna]:   4%|▍         | 13/326 [00:01<00:25, 12.07it/s]GNN+MLP[test/dna_rna]:   5%|▍         | 15/326 [00:01<00:25, 12.02it/s]GNN+MLP[test/dna_rna]:   6%|▌         | 19/326 [00:01<00:18, 16.99it/s]GNN+MLP[test/dna_rna]:   6%|▋         | 21/326 [00:01<00:20, 15.21it/s]GNN+MLP[test/dna_rna]:   7%|▋         | 23/326 [00:01<00:23, 13.13it/s]GNN+MLP[test/dna_rna]:   8%|▊         | 26/326 [00:02<00:18, 15.80it/s]GNN+MLP[test/dna_rna]:   9%|▉         | 29/326 [00:02<00:16, 18.51it/s]GNN+MLP[test/dna_rna]:  10%|▉         | 32/326 [00:02<00:17, 16.38it/s]GNN+MLP[test/dna_rna]:  10%|█         | 34/326 [00:02<00:17, 17.06it/s]GNN+MLP[test/dna_rna]:  11%|█▏        | 37/326 [00:02<00:14, 19.56it/s]GNN+MLP[test/dna_rna]:  12%|█▏        | 40/326 [00:02<00:14, 19.65it/s]GNN+MLP[test/dna_rna]:  13%|█▎        | 43/326 [00:02<00:14, 19.22it/s]GNN+MLP[test/dna_rna]:  14%|█▍        | 46/326 [00:03<00:16, 17.18it/s]GNN+MLP[test/dna_rna]:  15%|█▌        | 49/326 [00:03<00:14, 19.06it/s]GNN+MLP[test/dna_rna]:  16%|█▌        | 52/326 [00:03<00:15, 18.16it/s]GNN+MLP[test/dna_rna]:  17%|█▋        | 55/326 [00:03<00:15, 18.06it/s]GNN+MLP[test/dna_rna]:  18%|█▊        | 59/326 [00:03<00:12, 21.23it/s]GNN+MLP[test/dna_rna]:  19%|█▉        | 62/326 [00:03<00:11, 22.82it/s]GNN+MLP[test/dna_rna]:  20%|██        | 66/326 [00:03<00:09, 26.34it/s]GNN+MLP[test/dna_rna]:  21%|██        | 69/326 [00:04<00:09, 25.86it/s]GNN+MLP[test/dna_rna]:  22%|██▏       | 72/326 [00:04<00:10, 24.48it/s]GNN+MLP[test/dna_rna]:  23%|██▎       | 75/326 [00:04<00:10, 23.03it/s]GNN+MLP[test/dna_rna]:  24%|██▍       | 78/326 [00:04<00:12, 19.15it/s]GNN+MLP[test/dna_rna]:  25%|██▍       | 81/326 [00:04<00:11, 20.93it/s]GNN+MLP[test/dna_rna]:  26%|██▌       | 84/326 [00:04<00:12, 19.35it/s]GNN+MLP[test/dna_rna]:  27%|██▋       | 87/326 [00:05<00:12, 18.66it/s]GNN+MLP[test/dna_rna]:  27%|██▋       | 89/326 [00:05<00:14, 16.31it/s]GNN+MLP[test/dna_rna]:  28%|██▊       | 91/326 [00:05<00:15, 15.62it/s]GNN+MLP[test/dna_rna]:  29%|██▊       | 93/326 [00:05<00:19, 11.89it/s]GNN+MLP[test/dna_rna]:  29%|██▉       | 96/326 [00:05<00:15, 14.74it/s]GNN+MLP[test/dna_rna]:  30%|███       | 99/326 [00:05<00:12, 17.66it/s]GNN+MLP[test/dna_rna]:  31%|███▏      | 102/326 [00:06<00:13, 17.09it/s]GNN+MLP[test/dna_rna]:  32%|███▏      | 104/326 [00:06<00:13, 16.61it/s]GNN+MLP[test/dna_rna]:  33%|███▎      | 107/326 [00:06<00:12, 17.89it/s]GNN+MLP[test/dna_rna]:  34%|███▍      | 111/326 [00:06<00:13, 16.06it/s]GNN+MLP[test/dna_rna]:  35%|███▌      | 115/326 [00:06<00:11, 18.47it/s]GNN+MLP[test/dna_rna]:  36%|███▌      | 118/326 [00:06<00:10, 20.27it/s]GNN+MLP[test/dna_rna]:  37%|███▋      | 121/326 [00:07<00:11, 17.18it/s]GNN+MLP[test/dna_rna]:  38%|███▊      | 123/326 [00:07<00:11, 17.09it/s]GNN+MLP[test/dna_rna]:  38%|███▊      | 125/326 [00:07<00:12, 15.75it/s]GNN+MLP[test/dna_rna]:  40%|███▉      | 129/326 [00:07<00:10, 19.51it/s]GNN+MLP[test/dna_rna]:  40%|████      | 132/326 [00:07<00:09, 21.45it/s]GNN+MLP[test/dna_rna]:  42%|████▏     | 136/326 [00:07<00:07, 24.07it/s]GNN+MLP[test/dna_rna]:  43%|████▎     | 139/326 [00:07<00:07, 24.66it/s]GNN+MLP[test/dna_rna]:  44%|████▎     | 142/326 [00:08<00:09, 18.84it/s]GNN+MLP[test/dna_rna]:  44%|████▍     | 145/326 [00:08<00:09, 18.52it/s]GNN+MLP[test/dna_rna]:  45%|████▌     | 148/326 [00:08<00:11, 15.35it/s]GNN+MLP[test/dna_rna]:  47%|████▋     | 152/326 [00:08<00:08, 19.69it/s]GNN+MLP[test/dna_rna]:  48%|████▊     | 155/326 [00:08<00:09, 18.23it/s]GNN+MLP[test/dna_rna]:  48%|████▊     | 158/326 [00:09<00:09, 17.47it/s]GNN+MLP[test/dna_rna]:  49%|████▉     | 161/326 [00:09<00:08, 19.30it/s]GNN+MLP[test/dna_rna]:  50%|█████     | 164/326 [00:09<00:07, 20.65it/s]GNN+MLP[test/dna_rna]:  51%|█████     | 167/326 [00:09<00:08, 18.62it/s]GNN+MLP[test/dna_rna]:  52%|█████▏    | 170/326 [00:09<00:09, 16.52it/s]GNN+MLP[test/dna_rna]:  53%|█████▎    | 173/326 [00:09<00:08, 18.70it/s]GNN+MLP[test/dna_rna]:  54%|█████▍    | 176/326 [00:09<00:07, 20.16it/s]GNN+MLP[test/dna_rna]:  55%|█████▍    | 179/326 [00:10<00:06, 21.91it/s]GNN+MLP[test/dna_rna]:  56%|█████▌    | 182/326 [00:10<00:08, 17.83it/s]GNN+MLP[test/dna_rna]:  57%|█████▋    | 185/326 [00:10<00:07, 19.43it/s]GNN+MLP[test/dna_rna]:  58%|█████▊    | 188/326 [00:10<00:07, 17.50it/s]GNN+MLP[test/dna_rna]:  58%|█████▊    | 190/326 [00:10<00:07, 17.56it/s]GNN+MLP[test/dna_rna]:  59%|█████▉    | 193/326 [00:10<00:07, 18.92it/s]GNN+MLP[test/dna_rna]:  60%|██████    | 196/326 [00:11<00:07, 17.21it/s]GNN+MLP[test/dna_rna]:  61%|██████    | 199/326 [00:11<00:07, 17.12it/s]GNN+MLP[test/dna_rna]:  62%|██████▏   | 201/326 [00:11<00:07, 16.11it/s]GNN+MLP[test/dna_rna]:  63%|██████▎   | 204/326 [00:11<00:07, 16.99it/s]GNN+MLP[test/dna_rna]:  63%|██████▎   | 207/326 [00:11<00:06, 18.35it/s]GNN+MLP[test/dna_rna]:  64%|██████▍   | 210/326 [00:11<00:06, 19.22it/s]GNN+MLP[test/dna_rna]:  65%|██████▌   | 213/326 [00:11<00:05, 19.64it/s]GNN+MLP[test/dna_rna]:  66%|██████▌   | 215/326 [00:12<00:05, 18.51it/s]GNN+MLP[test/dna_rna]:  67%|██████▋   | 218/326 [00:12<00:05, 20.55it/s]GNN+MLP[test/dna_rna]:  68%|██████▊   | 221/326 [00:12<00:05, 20.57it/s]GNN+MLP[test/dna_rna]:  69%|██████▊   | 224/326 [00:12<00:04, 21.34it/s]GNN+MLP[test/dna_rna]:  70%|██████▉   | 227/326 [00:12<00:06, 16.37it/s]GNN+MLP[test/dna_rna]:  71%|███████   | 230/326 [00:12<00:05, 16.74it/s]GNN+MLP[test/dna_rna]:  71%|███████▏  | 233/326 [00:13<00:05, 15.63it/s]GNN+MLP[test/dna_rna]:  72%|███████▏  | 235/326 [00:13<00:06, 14.57it/s]GNN+MLP[test/dna_rna]:  73%|███████▎  | 238/326 [00:13<00:05, 17.27it/s]GNN+MLP[test/dna_rna]:  74%|███████▎  | 240/326 [00:13<00:05, 16.00it/s]GNN+MLP[test/dna_rna]:  74%|███████▍  | 242/326 [00:13<00:05, 15.69it/s]GNN+MLP[test/dna_rna]:  75%|███████▌  | 246/326 [00:13<00:04, 19.44it/s]GNN+MLP[test/dna_rna]:  76%|███████▋  | 249/326 [00:14<00:04, 15.57it/s]GNN+MLP[test/dna_rna]:  77%|███████▋  | 252/326 [00:14<00:04, 14.89it/s]GNN+MLP[test/dna_rna]:  78%|███████▊  | 255/326 [00:14<00:04, 17.03it/s]GNN+MLP[test/dna_rna]:  79%|███████▉  | 258/326 [00:14<00:03, 17.25it/s]GNN+MLP[test/dna_rna]:  80%|████████  | 261/326 [00:14<00:03, 19.41it/s]GNN+MLP[test/dna_rna]:  81%|████████  | 264/326 [00:14<00:03, 18.74it/s]GNN+MLP[test/dna_rna]:  82%|████████▏ | 267/326 [00:15<00:03, 17.80it/s]GNN+MLP[test/dna_rna]:  83%|████████▎ | 270/326 [00:15<00:02, 18.82it/s]GNN+MLP[test/dna_rna]:  84%|████████▍ | 274/326 [00:15<00:02, 22.17it/s]GNN+MLP[test/dna_rna]:  85%|████████▍ | 277/326 [00:15<00:02, 20.47it/s]GNN+MLP[test/dna_rna]:  86%|████████▌ | 280/326 [00:15<00:02, 16.93it/s]GNN+MLP[test/dna_rna]:  87%|████████▋ | 282/326 [00:16<00:03, 14.13it/s]GNN+MLP[test/dna_rna]:  87%|████████▋ | 284/326 [00:16<00:03, 12.29it/s]GNN+MLP[test/dna_rna]:  88%|████████▊ | 286/326 [00:16<00:03, 12.75it/s]GNN+MLP[test/dna_rna]:  89%|████████▊ | 289/326 [00:16<00:02, 15.96it/s]GNN+MLP[test/dna_rna]:  89%|████████▉ | 291/326 [00:16<00:02, 16.25it/s]GNN+MLP[test/dna_rna]:  90%|████████▉ | 293/326 [00:16<00:02, 14.87it/s]GNN+MLP[test/dna_rna]:  90%|█████████ | 295/326 [00:16<00:01, 15.67it/s]GNN+MLP[test/dna_rna]:  91%|█████████ | 297/326 [00:17<00:01, 16.51it/s]GNN+MLP[test/dna_rna]:  92%|█████████▏| 300/326 [00:17<00:01, 18.68it/s]GNN+MLP[test/dna_rna]:  93%|█████████▎| 302/326 [00:17<00:01, 17.26it/s]GNN+MLP[test/dna_rna]:  93%|█████████▎| 304/326 [00:17<00:01, 13.37it/s]GNN+MLP[test/dna_rna]:  94%|█████████▍| 306/326 [00:17<00:01, 13.82it/s]GNN+MLP[test/dna_rna]:  95%|█████████▍| 309/326 [00:17<00:01, 15.57it/s]GNN+MLP[test/dna_rna]:  95%|█████████▌| 311/326 [00:18<00:01, 13.40it/s]GNN+MLP[test/dna_rna]:  96%|█████████▋| 314/326 [00:18<00:00, 16.59it/s]GNN+MLP[test/dna_rna]:  98%|█████████▊| 318/326 [00:18<00:00, 20.48it/s]GNN+MLP[test/dna_rna]:  98%|█████████▊| 321/326 [00:18<00:00, 22.03it/s]GNN+MLP[test/dna_rna]:  99%|█████████▉| 324/326 [00:18<00:00, 23.79it/s]GNN+MLP[test/dna_rna]: 100%|██████████| 326/326 [00:18<00:00, 17.50it/s]
loaded 326 test graphs (0 failed to load)
reproduced:                y_true=(87873, 2) y_prob=(87873, 2)

--- STEP 1: does the reconstructed ordering match? ---
PASS y_true identical (87,873 residues) -> ordering proven

--- STEP 2: is runs/<setup>/mlp/model.pt the gnn_mlp embedder? ---
max|diff|=1.192e-07  mean|diff|=4.483e-09  corr=1.000000
PASS -> the mlp checkpoint IS the discarded flat_mlp. gnn_mlp is deployable.
```
