# JBBind

Per-residue protein binding-site prediction from a structure. Give it a PDB ID or a
structure file, pick a chain, and it scores every solvent-accessible residue for whether
it binds a protein or a nucleic acid — painted on the structure in Mol*.

```
python -m jbbind.cli serve             # the web app — everything is in the browser
python -m jbbind.cli predict 1ycr --chain A
python -m jbbind.cli batch targets.csv --out results/
python predict_bindingsites.py 1ycr_A  # one chain: scores, figures, HTML report
```

---

## Install

### From source

JBBind runs on the CPU only. Install the CPU build of torch first; plain `pip install
torch` pulls several GB of CUDA libraries that are never used.

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export PATH="$PATH:/path/to/voronota/expansion_js"   # voronota-js must be on PATH
```

The ESM-2 650M weights (2.6 GB) download to `~/.cache/torch` on first use; the 15 models
(20 checkpoint files) ship in `models/`.

### Container

```bash
# Apptainer
make sif
apptainer run --cleanenv --bind /scratch/jbbind:/data jbbind.sif          # web app on :8000
apptainer exec --cleanenv --bind /scratch/jbbind:/data jbbind.sif \
    jbbind predict 1ycr --chain A

# Docker
make docker && docker run -p 8000:8000 -v jbbind-data:/data jbbind:cpu
```

The images bundle voronota, the ESM-2 weights and every checkpoint. If `make sif` fails at
"creating squashfs", use `make sandbox` instead and run `jbbind-sandbox/`.

---

## Launch

```bash
make serve                     # http://127.0.0.1:8000
make serve HOST=0.0.0.0        # expose on the LAN (open the port in firewalld)
```

First start takes 30–60 s while ESM-2 loads; `/readyz` reports `degraded` until it is up.

**As a service**, so it is always up:

```bash
cp deploy/jbbind.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jbbind
loginctl enable-linger $USER   # survive logout
```

`systemctl --user status jbbind`, `stop jbbind`, `journalctl --user -u jbbind -f`.
See `deploy/README.md`.

**From another machine**, forward the port — in VS Code, the **Ports** panel → *Forward a
Port* → `8000`; from a terminal, `ssh -N -L 8000:127.0.0.1:8000 <host>`.

---

## Use the web app

Three pages: **Predict**, **Metrics**, **Settings**.

On **Predict**:

1. Type a 4-character PDB ID and press *Load*, or upload a `.pdb` / `.cif` / `.gz`.
2. Pick the chain and press *Predict*.
3. The structure appears in Mol*, coloured by score: **blue** for protein binding,
   **red** for nucleic-acid (DNA/RNA) binding.

The left rail controls the view live: task, which label colours the structure, threshold,
continuous-vs-threshold colouring, molecular surface. The
right panel has a sequence track, a residue table showing **every label of the selected
task side by side**, and a Download tab. Hovering the structure or the track names the
residue; clicking one flies the camera to it. The theme selector in the top bar switches
between light, dark and the system setting.

**Settings** sets the defaults the Predict page and the CLI both read: architecture, task,
long-chain handling, RCSB source, threshold, colouring.

---

## Tasks and models

Five tasks, each a set of independent per-residue binary labels:

| task | labels |
|---|---|
| `protein_nucleic` | Protein · Nucleic acid |
| `protein` | Protein |
| `homo_hetero` | Homo · Hetero |
| `nucleic` | Nucleic acid |
| `dna_rna` | DNA · RNA |

Three architectures per task, `gnn_mlp` by default:

| arch | inputs | test PR-AUC (macro, `protein`) |
|---|---|---|
| `gnn_mlp` | Voronoi contact graph + ESM-2 embedding | 0.576 |
| `joint` | same, trained end to end | 0.579 |
| `mlp` | ESM-2 sequence embedding only | 0.549 |

Scores are **uncalibrated** — rank residues by score rather than reading a score as a
probability. Every reported metric uses a fixed threshold of 0.50.

Chains up to 5000 residues are predicted. ESM-2 takes 1022 tokens at a time, so longer
chains are embedded with overlapping windows (`tile`, the default); *Settings → Long
chains* switches to `truncate`, which stops at 1022 and matches how the training
embeddings were made.

---

## Command line

```bash
python -m jbbind.cli info                          # device, checkpoints, cache
python -m jbbind.cli predict 1ycr --chain A
python -m jbbind.cli predict 1ycr --all-chains --out results/
python -m jbbind.cli batch targets.csv --out results/ --workers 8
python -m jbbind.cli serve --host 0.0.0.0 --port 8000
```

`predict` takes a PDB ID or a path to a local structure; `--setup` restricts to one task,
`--arch` overrides the architecture, `--out` writes CSV, `--json` emits the full result and
`--top N` sets how many rows print.

`batch` takes `pdb_id[,chain]` per line (blank chain means every protein chain) or paths to
local files, and writes:

```
results/chains/<pdb>_<chain>.csv     per-chain scores
results/predictions.parquet          tidy: pdb_id, chain, residue, setup, label, score
results/manifest.jsonl               resume state
results/failures.csv                 what failed and why
```

Re-running skips completed chains.

### `predict_bindingsites.py`

One chain in, a folder out — no server.

```bash
python predict_bindingsites.py 1ycr_A                    # PDB ID + chain
python predict_bindingsites.py 3hdd_A --setup dna_rna    # a different task
python predict_bindingsites.py 6lu7 --all-chains         # every protein chain
python predict_bindingsites.py model.pdb --chain A       # a local structure
python predict_bindingsites.py --list targets.txt        # pdb_id[,chain] per line
```

```
predictions/3hdd_A/
    report_3hdd_A.html                     interactive Mol* report  <- opens
    predictions_3hdd_A.csv                 every residue, every label
    annotated_3hdd_A_dna_rna_DNA.pdb       score in the B-factor column
    3hdd_A_dna_rna_DNA.png                 the figure
    3hdd_A_dna_rna_DNA.pml                 PyMOL session script
    3hdd_A_dna_rna_DNA.cxc                 ChimeraX session script
predictions/_assets/                       Mol*, shared by every report under this root
```

The report opens in your browser when the run produced exactly one. `--standalone` inlines
Mol* into a single ~5 MB file that survives being emailed. `--setup all` runs all five
tasks in one pass; `--threshold` (default 0.50) sets which residues count as hits;
`--no-report`, `--no-open`, `--port` and `--serve` / `--no-serve` override the rest.

On a remote host with no display the script serves the output directory on
`http://127.0.0.1:8010` and opens that instead of a `file://` URL, blocking until Ctrl+C.

---

## API

```
GET  /healthz  /readyz  /api/v1/meta  /api/v1/models  /api/v1/metrics
GET  /api/v1/structures/by-pdb-id/{id}     POST /api/v1/structures        (upload)
POST /api/v1/predict            -> 202 {job_id}
GET  /api/v1/jobs/{id}          /api/v1/jobs/{id}/events   (SSE progress)
GET  /api/v1/artifacts/{id}/{receptor.pdb|predictions.csv|predictions.pdb|pymol.txt}
GET  /api/v1/artifacts/{id}/{report.html|figure.png|session.pml|session.cxc}
GET  /api/v1/settings           PUT /api/v1/settings
```

POST returns a `job_id`; poll `/jobs/{id}` or subscribe to `/jobs/{id}/events`. Errors are
RFC 9457 `application/problem+json` with a `code`. Interactive docs at `/api/docs`.

Downloads: `predictions.csv` (every residue, every label, every task), `predictions.pdb`
(score in the B-factor column, **`-1.00` means not predicted**, never `0.00`),
`receptor.pdb` (the atoms the model saw, renumbered to SEQRES indices), `report.html`,
`figure.png`, and PyMOL / ChimeraX session scripts.

---

## Configuration

| environment variable | default | |
|---|---|---|
| `JBBIND_MODELS` | `models/` | checkpoint directory |
| `JBBIND_CACHE` | `/data/cache` | structures, embeddings, predictions |
| `JBBIND_MAX_RESIDUES` | `5000` | longest chain accepted |
| `JBBIND_MAX_UPLOAD` | `30 MB` | largest upload accepted |
| `JBBIND_JOB_WORKERS` | `2` | concurrent prediction jobs |

Everything else lives on the Settings page and persists to `$JBBIND_CACHE/settings.json`.
Resident memory is ~3.3 GB per worker; one worker is right, a
second doubles the ESM-2 copy for no throughput gain.

## Development

`make test` runs the suite, `make info` prints device and checkpoint state, `make help`
lists every target. `docs/VERIFICATION.md` records the parity and provenance checks behind
the checkpoints.

Voronota, ESM-2 and Mol* are bundled — see `NOTICE` for licences and citations.
