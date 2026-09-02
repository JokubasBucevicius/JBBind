# JBBind

Per-residue protein binding-site prediction, served from a structure.

Give it a PDB ID or a structure file, pick a chain, and it scores every solvent-accessible
residue for whether it binds a protein or a nucleic acid — painted on the 3D structure,
with a metrics dashboard for the models behind it and a settings page for swapping them.

```
python predict_bindingsites.py 1ycr_A  # scores, figures, interactive report
jbbind predict 1ycr --chain A          # one chain, to the terminal
jbbind batch targets.csv --out out/    # thousands of chains, resumable
jbbind serve                           # the web app
```

---

## What it predicts

Five tasks, each a set of independent per-residue binary labels. "Non-binding" is the
implicit all-zero case, never an explicit class.

| task | labels | notes |
|---|---|---|
| `protein_nucleic` | Protein · Nucleic acid | broadest; weakest metrics |
| `protein` | Protein | protein–protein interface |
| `homo_hetero` | Homo · Hetero | self-association vs a different protein |
| `nucleic` | Nucleic acid | best PR-AUC of the five |
| `dna_rna` | DNA · RNA | DNA is predicted markedly better than RNA |

Four architectures per task, all trained by the research pipeline this app serves:

| arch | inputs | test PR-AUC (macro, `protein`) |
|---|---|---|
| **`gnn_mlp`** (default) | Voronoi contact graph + ESM-2 embedding | 0.576 |
| `joint` | same, trained end to end with an auxiliary sequence loss | 0.579 |
| `mlp` | ESM-2 sequence embedding only | 0.549 |
| `gnn` | structure only — kept as a control | 0.360 |

The ordering `joint ≈ gnn_mlp > mlp ≫ gnn` holds across all five tasks: the sequence
embedding carries most of the signal and structure adds a real but smaller increment.

### Read the scores as a ranking, not a probability

Training used focal loss with per-label `pos_weight`, which deliberately pushes the model
to over-predict positives. On `protein`, precision at the 0.50 operating point is 0.39
against recall 0.82. The raw sigmoid is **not calibrated** — a residue at 0.7 is not "70%
likely to bind". Rank residues by score and take the top of the list.

Every reported number uses a fixed threshold of 0.50, which is the operating point
`benchmarks.py` evaluates at. No per-label tuning was done.

---

## Quick start

### Container (recommended)

```bash
# Apptainer — builds and runs with no docker daemon
make sif
apptainer run --cleanenv --bind /scratch/jbbind:/data jbbind.sif       # web app on :8000
apptainer exec --cleanenv --bind /scratch/jbbind:/data jbbind.sif \
    jbbind predict 1ycr --chain A

# If `make sif` dies at "creating squashfs" with exit status 139, that is mksquashfs
# segfaulting on the host, not a problem with the definition file. Build a sandbox
# instead — it runs identically, just as a directory rather than a single file:
make sandbox
apptainer run --cleanenv --bind /scratch/jbbind:/data jbbind-sandbox/

# Docker
make docker && docker run -p 8000:8000 -v jbbind-data:/data jbbind:cpu
```

The image bakes in voronota, the ESM-2 650M weights (2.6 GB) and all 20 checkpoints, so it
needs no network except to fetch structures from RCSB.

GPU: add `--nv` (Apptainer) or build `make docker-cuda` and run with `--gpus all`. Only
worth it for batch runs — ESM-2 is the bottleneck and everything else is milliseconds.

### From source

```bash
pip install -r requirements.txt
export PATH="$PATH:/path/to/voronota/expansion_js"     # needs voronota-js on PATH
make serve                                             # http://127.0.0.1:8000
```

**Viewing it over SSH / from VS Code Remote:** the app binds to localhost, so forward the
port. In VS Code, open the **Ports** panel → *Forward a Port* → `8000`, then open the
forwarded URL. From a plain terminal: `ssh -L 8000:127.0.0.1:8000 <host>`.

First start takes 30–60 s while ESM-2 loads; `/readyz` reports `degraded` until it is up.

---

## `predict_bindingsites.py` — one chain in, scores and pictures out

The standalone front end, in the shape the published tools use. No server, no browser,
nothing to configure: name a chain and it writes a folder.

```bash
export PATH="$PATH:/path/to/voronota/expansion_js"
python predict_bindingsites.py 1ycr_A                    # PDB ID + chain
python predict_bindingsites.py 3hdd_A --setup dna_rna    # a different task
python predict_bindingsites.py 6lu7 --all-chains         # every protein chain
python predict_bindingsites.py model.pdb --chain A       # a local structure
python predict_bindingsites.py --list targets.txt        # pdb_id[,chain] per line
```

```
predictions/3hdd_A/
    report_3hdd_A.html                     interactive Mol* report  <- opens
    predictions_3hdd_A.csv                 every residue, every requested label
    annotated_3hdd_A_dna_rna_DNA.pdb       score in the B-factor column
    3hdd_A_dna_rna_DNA.png                 the figure
    3hdd_A_dna_rna_DNA.pml                 PyMOL session script
    3hdd_A_dna_rna_DNA.cxc                 ChimeraX session script
predictions/_assets/                       Mol*, copied once, shared by every report
```

The report is the point of the run: the structure in Mol*, coloured by score, with the
label, the threshold, the colouring mode and the surface all live, a sequence track and a
sorted residue table beside it. Hovering the structure names the residue; clicking one —
in the viewer, the track or the table — flies the camera to it. It opens in your browser
when the run produced exactly one report (`--open` / `--no-open` overrides; `--no-report`
skips it).

Mol* is 4.8 MB, so it is copied once into `<out>/_assets/` and shared by every report
under that root rather than inlined a hundred times. `--standalone` inlines it instead,
giving one ~5 MB file that survives being emailed.

**Opening it from a remote host.** A `file://` URL is useless there: under VS Code Remote
`$BROWSER` is a helper that runs `code --openExternal`, which opens the URL on *your*
machine, where `/home/you/predictions/…` does not exist. So when there is no local display
the script serves the output directory on `http://127.0.0.1:8010` instead and opens that —
VS Code forwards the port by itself, and `ssh -L 8010:127.0.0.1:8010 <host>` reaches it
from a plain terminal. It blocks while serving, because the page fetches Mol* from
`_assets/` on load and a reload needs the server still up; Ctrl+C stops it. `--port`
changes the port (`0` picks a free one) and `--serve` / `--no-serve` overrides the
detection.

`--setup` defaults to `protein` and takes any of the five tasks, or `all` to run every
one of them in a single pass — the tessellation and the ESM forward are shared, so all
five cost about as much as one. `--threshold` (default 0.50) sets which residues are
counted as hits, highlighted in the figure and selected in the viewer scripts. Only
`gnn_mlp` is served.

The figure carries four panels: two 3D views of the CA trace coloured by score, the
highest-scoring residues with their SASA, the score along the chain, and the ranked score
curve. The seven-stop ramp and the out-of-ramp grey are the same ones the web viewer uses,
interpolated in OKLab by the same rule, so a residue looks identical in both.

Three things the figure is deliberately careful about, all of which are easy to get wrong:

- **Unscored residues are drawn, in grey, outside the ramp.** Buried residues and anything
  past ESM-2's 1022-token limit have no prediction. Omitting them from the 3D trace would
  read as a break in the chain; colouring them pale blue would read as a confident
  negative. They also get a tick rug under the sequence track.
- **The B-factor sentinel is `-1.00`, never `0.00`**, and both viewer scripts exclude it
  from the colour ramp explicitly. `spectrum b` over the raw range would otherwise stretch
  the ramp down to −1 and push every real score into the top half of the scale.
- **The colour axis is pinned to 0–1**, not to the chain's own min and max, so two chains
  can be put side by side. The cost is that a chain whose scores all sit in a narrow band
  looks uniformly mid-blue — which is honest, given the calibration warning above, but
  means the ranked-score panel is the one to read for such chains.

A worked example: `3hdd_A` is the engrailed homeodomain on DNA. The `dna_rna` model puts
23 residues above 0.5 with a sharp cliff in the ranked curve — the N-terminal arm (R5, T6,
F8) and helix 3 (I47, W48, Q50, N51, R53, A54, K55, K57, K58), which is the recognition
helix — and scores RNA at most 0.202 on the same chain.

---

## The 3D viewer

Mol* renders the structure, wrapped by `jbbind/static/viewer.js`. Nothing else in the
front end touches the plugin: `viewer.js` takes residue indices and CSS colours and hands
back hover and click events, so the score ramp has one definition and the engine stays
replaceable.

Scores reach the geometry through a registered Mol* colour theme (`jbbind-score`) rather
than through the B-factor column, so the seven-stop ramp, its OKLab interpolation and the
out-of-ramp grey for unscored residues are byte-identical to the sequence track, the
legend and `predict_bindingsites.py`. Residues are addressed by `auth_seq_id`, which is
the SEQRES index because `receptor.pdb` is already renumbered to it.

The theme carries no parameters, so Mol* cannot tell that new scores are the same theme
under a different meaning; changing the task, label or threshold therefore rebuilds the
representations instead of recolouring them in place. That is a few tens of ms for one
chain, and the threshold slider is debounced so a drag does not queue a rebuild per pixel.
The molecular surface is the one expensive rebuild, which is why it is off by default.

Mol*'s own panels are all disabled — the page already has a left rail, a legend and a
sequence track — and reset-camera is the only viewport button kept. Not every button has
a `viewportShow*` option: illumination and XR are reachable only through Mol*'s `config`
array and both default to visible, so check the viewport after a Mol* upgrade.

`predict_bindingsites.py` inlines this same `viewer.js` into its HTML report, with the
`export` keywords stripped, because browsers refuse ES modules on `file://` URLs. That
keeps one Mol* wrapper rather than two that drift apart, and it is why `viewer.js` must
not grow an `import` — a test asserts it has none.

---

## Everything from the browser

`jbbind serve` is a complete front end: nothing needs the command line except starting
the server. The Predict page fetches or takes an upload, picks a chain, runs the model
and shows the result, and its Download tab serves every artifact
`predict_bindingsites.py` writes — the standalone HTML report, the four-panel figure, the
CSV, the B-factor PDB, the receptor, and PyMOL and ChimeraX sessions. The report and the
figure come out byte-identical either way, because both front ends call the same
`jbbind.core` modules rather than each holding a copy.

That refactor is what made it possible: `colour.py`, `figure.py`, `report.py` and
`viewers.py` moved out of the script and into the package, and the script became the thin
wrapper its docstring always claimed it was. `figure.py` is the only module that imports
matplotlib, and the web app imports it inside the figure endpoint so a server that never
draws one never pays for it.

---

## How it works

```
PDB ID or upload
      │
      ├─ fetch          RCSB asymmetric unit, cached 30 days
      ├─ normalize      gemmi: one model, one chain, standard residues,
      │                 RENUMBERED TO SEQRES INDICES        ← the critical step
      ├─ voronota       Voronoi tessellation → per-atom features + contacts
      ├─ aggregate      atoms → residues, keep the solvent-accessible ones
      ├─ ESM-2 650M     layer 33, on the canonical sequence
      └─ model          per-residue sigmoid scores, mapped back to author numbering
```

### Why renumbering is the critical step

The training data numbered residues by their **1-based position in the SEQRES sequence**,
not by author numbering, and the training loader indexed embeddings with
`embedding_indices = ID_resSeq - 1` (`train_multilabel.py:329`). Verified on 250 random
training chains: `one_letter(ID_resName) == s1_sequence[ID_resSeq - 1]` holds for 94.4%.

Author numbering from RCSB is *not* that index — it has gaps, insertion codes, negative
values and expression-tag offsets. Feeding it through unchanged would silently pair each
residue with the wrong embedding on every gapped chain, and gapped chains are the majority.
So JBBind renumbers before voronota ever sees the structure, prefers mmCIF `label_seq_id`,
falls back to sequence alignment, and asserts `one_letter(resname) == sequence[i-1]` for
every residue at runtime.

This also fixes a latent bug for free: `groupby("ID_resSeq")` in the training code merges
residues that share a number but differ by insertion code (common in antibodies). SEQRES
indices are unique, so it cannot happen here.

### Other inherited behaviours, kept deliberately

- **Per-graph min-max normalisation.** Every feature is rescaled against its own chain's
  extremes, so the same residue gets different inputs depending on what else is in the
  chain. Predict on the deposited chain, not a fragment.
- **One-directional edges.** The training code canonicalises each contact to
  `(min, max)` resSeq and never adds the reverse edge. A test asserts `edge_index` stays
  asymmetric, so nobody "fixes" it.
- **ESM-2's 1022-token limit.** `extract.py` truncated there when the training embeddings
  were made, so residues past 1022 have no embedding and are not predicted. `tile` mode
  covers the whole chain but is explicitly out of distribution.
- **Buried residues are dropped**, matching training. The UI renders them in a grey that
  sits outside the score ramp so "not predicted" can never read as "low score".

---

## Verification

The core risk in lifting models out of a research pipeline is silently changing what they
see. Every claim below is a command you can re-run.

| check | result | command |
|---|---|---|
| **Tier 1** — tensors bit-identical to the original training code | **pass**, `torch.equal` on 3 golden chains | `make test` |
| **Tier 2** — model outputs match the original classes, 5 setups × 4 archs | **20/20 pass**, atol 1e-6 | `make test` |
| **Tier 3a** — forked voronota script reproduces the training features | **40/40 chains byte-exact** (only `bsite_area` differs, by design) | `make verify-tool` |
| **Tier 3b** — does serving from RCSB change the answer vs the training files? | **median Pearson r 1.0000, median max\|Δ\| 0.0000** over 39 chains (p90 max\|Δ\| 0.071) | `make verify-provenance` |
| **Tier 3c** — SEQRES renumbering matches the training numbering | **190/190 = 100%** of sound chains | `make verify-renumbering` |
| **P0.5** — the `gnn_mlp` embedder recovery (below) | max abs diff **1.19e-07** vs the recorded `predictions.npz` | `scripts/spike_gnn_mlp_embedder.py` |

172 tests run in ~20 s and need no access to the research tree — the fixtures are
committed. They also run **inside** the built container (`make sif-test`), which is where a
torch/pandas version drift would show up first: the container reproduces the host's
predictions exactly (1YCR chain A → M62 0.689, T63 0.669, F55 0.662).

The parity assertions run single-threaded with deterministic algorithms. GATv2Conv's
message passing is a scatter-add, and multi-threaded CPU reductions accumulate in
nondeterministic order — which made the 1e-6 tolerance occasionally lucky. Removing the
nondeterminism is the fix; loosening the tolerance would not be.

### The `gnn_mlp` checkpoints were incomplete, and are recovered

`GNNMLPArch.prepare()` trains a prerequisite MLP whose 64-d penultimate activations become
64 of the GNN's 69 input features — then `save_checkpoint` saves only the GNN. The
embedder was discarded, so the five `gnn_mlp` checkpoints could not be loaded at all.

They are recoverable because `main.py:60-62` seeds `torch`/`numpy`/`random` before anything
else, and nothing between seeding and `trainer.train_mlp` consumes the torch global RNG —
so the `--arch mlp` job and the `--arch gnn_mlp` job train the *same* MLP.
`scripts/spike_gnn_mlp_embedder.py` proves it: it reconstructs the `dna_rna` test split
(matching the recorded 1630 qualifying / 1140-164-326 split exactly), confirms `y_true` is
bit-identical to `predictions.npz`, then reproduces `y_prob` to **1.19e-07** using
`runs/dna_rna/mlp/model.pt` as the embedder. `export_models.py` copies it in as
`embedder.pt` and records the provenance in `MANIFEST.json`.

**Worth fixing upstream:** add `flat_mlp.state_dict()` to `architectures.py:71` so future
runs are self-contained.

### Two data-quality findings worth acting on

1. **~5.6% of training chains were mis-aligned.** In a 250-chain sample, that fraction had
   `ID_resSeq` that did not index its own sequence (e.g. `5il2_B` at 4% residue identity,
   `1gkm_A` at 33%), so they trained against effectively random embeddings. This is
   pre-existing, not introduced here, but it caps what the reported metrics could have
   been. Re-deriving numbering for those chains and re-embedding is the obvious follow-up.
2. **PPI3D subunit labels are not RCSB chain labels.** PPI3D renames chains — its
   `7qhp_B` is RCSB chain `T`, its `5ikl_F` is RCSB chain `B`. If you look up a training
   chain ID in JBBind, the chain letter may differ. Both verification scripts resolve
   chains by sequence rather than by label because of this.

---

## Batch mode

```bash
jbbind batch targets.csv --out results/ --workers 8
```

`targets.csv` is `pdb_id[,chain]` per line (blank chain means every protein chain), or
paths to local files. Output:

```
results/chains/<pdb>_<chain>.csv     per-chain scores
results/predictions.parquet          everything, tidy: pdb_id, chain, residue, setup, label, score
results/manifest.jsonl               resume state
results/failures.csv                 what failed and why
```

Re-running skips completed chains. Fetches are per *entry*, not per chain; ESM embeddings
are deduplicated by sequence hash and shared across tasks, architectures and entries, so
a homomer costs one forward pass, not one per chain.

---

## Performance

Measured on this build host (52-core Xeon, `OMP_NUM_THREADS` unset), CPU only, cold cache:

| chain | residues | total | voronota | ESM-2 | model (5 tasks) |
|---|---|---|---|---|---|
| 1YCR A | 109 | 2.6 s | 1.6 s | 0.6 s | 0.39 s |
| 6LU7 A | 306 | 6.0 s | 4.7 s | 0.8 s | 0.41 s |
| 1GFL A | 238 | 7.4 s | 3.7 s | 3.4 s | 0.15 s |

Warm cache: **~0.06 s**. Resident memory is **3.3 GB** per worker on CPU (0.7 GB torch +
2.5 GB ESM-2); on GPU the weights move to VRAM and RSS drops to ~1.3 GB. Run one worker
and scale with the job queue — a second worker doubles the ESM copy for no throughput gain.

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

Prediction is job-based: a chain takes seconds to tens of seconds, which is long enough
that a synchronous endpoint would be at the mercy of every proxy timeout. Errors are
RFC 9457 `application/problem+json` with a machine-readable `code`.

Interactive docs at `/api/docs`.

### Downloads

- `predictions.csv` — every residue, every label, every task
- `predictions.pdb` — the structure with the score in the B-factor column;
  **`-1.00` means not predicted**, never `0.00`
- `receptor.pdb` — the exact atoms the model saw, renumbered to SEQRES indices
- a **Copy PyMOL selection** button — `select bs, chain A and resi 12+15+18+…`

---

## Layout

```
predict_bindingsites.py   the standalone one-chain front end (figures, viewer scripts)
jbbind/
  core/nn/          model classes (copied verbatim), registry, label setups
  core/structure/   fetch, normalize, modified residues  ← the renumbering lives here
  core/features/    voronota subprocess, atom→residue aggregation, PyG graph
  core/esm/         ESM-2 embedder + sequence-hash cache
  core/             pipeline, batch, jobs, cache, artifacts
  core/colour.py    the score ramp every rendering shares
  core/figure.py    the four-panel PNG   ← the only thing that imports matplotlib
  core/report.py    the standalone interactive HTML report
  core/viewers.py   PyMOL and ChimeraX session scripts
  static/           three-page SPA; viewer.js wraps Mol*, vendored locally
  report_template.html  the HTML report's template
tools/              the forked voronota script + the pristine original
models/             20 checkpoints, MANIFEST.json, METRICS.json
scripts/            model export, fixture generation, the four verification scripts
tests/              parity, normalization, API, CLI — no research repo needed
```

### Code lifted from the research repo

The model classes and feature code are **copied verbatim**, not imported. `train_multilabel`
builds a module-level `device` at import (forcing CUDA init), prints at import, sys.path-hacks
a training-only module, imports matplotlib, and exposes `NUM_LABELS`/`LABEL_NAMES` globals
that `setups.LabelSetup.activate()` *mutates* — unusable when serving five label setups from
one process. And the source repo is not version-controlled, so it cannot be pinned.

Every copied block carries its source path, line range, source SHA256 and copy date, and
every intentional change is marked `# DEVIATION:`. `tests/test_parity_tensors.py` imports
the original module and asserts bit-identical tensors, which is what makes the copy
verifiable rather than a fork that quietly drifts.

---

## Known limitations

- **Provenance shift — measured, and smaller than expected.** Training features came from
  PPI3D interface-coordinate files; JBBind serves the RCSB asymmetric unit. Combined with
  per-graph min-max normalisation, that could have shifted the inputs meaningfully. Over
  39 chains it does not: median Pearson r between the two provenances is 1.0000 and the
  median max\|Δscore\| is 0.0000 — for most chains the coordinates are simply identical.
  There is a tail, though: the 10th percentile of r is 0.92 and the 90th percentile of
  max\|Δ\| is 0.071, so a minority of chains do move. `make verify-provenance` reproduces
  the distribution; `rcsb_assembly` in Settings switches to biological assembly 1.
- **`protein_nucleic` is weak** (macro PR-AUC 0.344 for `gnn_mlp`) — nucleic-acid residues
  are ~0.3% of that task's training data. Prefer `protein` and `nucleic` separately.
- **Scores are uncalibrated** (see above). Per-setup isotonic calibration on the validation
  split is the obvious improvement.
- **Structures without SEQRES** (AlphaFold models, minimised structures) fall back to the
  observed sequence, which changes the ESM alignment relative to training. The app warns
  loudly and never does this silently.
- **The web UI has not been rendered in a browser** — this build host has neither a
  browser nor node. The API, CLI, batch and container paths are all exercised end-to-end
  by tests and by hand, and the front-end passes static checks (delimiter balance, every
  DOM id it references), but nobody has looked at it. That is the one thing to try first.

## Attribution

Voronota, ESM-2 and Mol* are bundled — see `NOTICE` for licences and citations.
