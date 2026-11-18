/* Predict page: viewer wiring, sequence track, residue table, downloads. */

import { api, cssVar, el, fmt, labelKind, rampSteps, scoreColor, state } from "/static/app.js";
import * as viewer from "/static/viewer.js";

let sortKey = "p0";
let sortDir = -1;
let hoverResi = null;

const $ = (id) => document.getElementById(id);

const view = {
  get setup() { return $("setup").value; },
  get labels() { return state.meta?.setups?.[$("setup").value]?.label_names || []; },
  get label() { return Number(document.querySelector('input[name="label"]:checked')?.value ?? 0); },
  /** "protein" (blue ramp) or "nucleic" (red ramp) for the label on display. */
  get kind() { return labelKind($("setup").value, this.label); },
  get threshold() { return Number($("threshold").value); },
  get colorMode() { return $("colormode").value; },
  /** "raw", "refined" or "change"; always "raw" while the result has no refinement. */
  get source() { return state.result?.refinement ? $("score-source").value : "raw"; },
};

/** The refine fields as the API names them, from the Predict panel's controls. */
function refineSettings() {
  return {
    refine_method: $("refine-method").value,
    refine_strength: Number($("refine-strength").value),
    gumbel_tau: Number($("gumbel-tau").value),
    gumbel_samples: Math.round(Number($("gumbel-samples").value)),
  };
}

/* ------------------------------------------------------------------ scores */

/** The score thresholds and hits are judged on: raw, or refined when that is shown. */
function decisionScore(r, setup, label) {
  if (!r) return null;
  return view.source !== "raw" && r.pr ? r.pr[setup][label] : r.p[setup][label];
}

/** What the colour ramps show: the decision score, or refined − raw in "change" mode. */
function displayValue(r, setup, label) {
  if (!r) return null;
  if (view.source === "change") return r.pr ? r.pr[setup][label] - r.p[setup][label] : null;
  return decisionScore(r, setup, label);
}

/** Colour for a displayValue. Change mode maps Δ ∈ [−0.5, 0.5] onto the diverging ramp. */
function displayColor(v, kind) {
  if (view.source === "change") {
    return scoreColor(v === null ? null : 0.5 + v, { steps: rampSteps("diverging") });
  }
  return scoreColor(v, { mode: view.colorMode, threshold: view.threshold,
                         steps: rampSteps(kind) });
}

/* --------------------------------------------------------------- lifecycle */

export function init() {
  $("load-pdb").addEventListener("click", loadPdbId);
  $("pdbid").addEventListener("keydown", (e) => { if (e.key === "Enter") loadPdbId(); });
  $("upload").addEventListener("change", loadUpload);
  $("run").addEventListener("click", runPrediction);
  $("chain").addEventListener("change", () => { $("run").disabled = false; });

  $("setup").addEventListener("change", () => { renderLabels(); repaint(); });
  $("threshold").addEventListener("input", () => {
    $("thr-value").textContent = view.threshold.toFixed(2);
    $("ramp-tick").style.left = `${view.threshold * 100}%`;
    repaint();
  });
  $("colormode").addEventListener("change", repaint);
  $("show-surface").addEventListener("change", repaint);

  $("refine-method").addEventListener("change", () => {
    // Choosing a method is asking to see its effect.
    if ($("refine-method").value !== "none" && $("score-source").value === "raw") {
      $("score-source").value = "refined";
    }
    syncRefineControls();
    scheduleRefine();
  });
  for (const [id, digits] of [["refine-strength", 2], ["gumbel-tau", 2]]) {
    $(id).addEventListener("input", () => {
      $(`${id}-value`).textContent = Number($(id).value).toFixed(digits);
      scheduleRefine();
    });
  }
  $("gumbel-samples").addEventListener("change", scheduleRefine);
  $("score-source").addEventListener("change", repaint);

  $("subtabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-sub]");
    if (!b) return;
    for (const x of $("subtabs").children) x.setAttribute("aria-selected", String(x === b));
    for (const p of document.querySelectorAll(".subpanel")) {
      p.classList.toggle("active", p.dataset.sub === b.dataset.sub);
    }
    if (b.dataset.sub === "sequence") drawTrack();
  });

  $("restable").querySelector("thead").addEventListener("click", (e) => {
    const th = e.target.closest("th[data-sort]");
    if (!th) return;
    const key = th.dataset.sort;
    if (key === sortKey) sortDir = -sortDir;
    else { sortKey = key; sortDir = key.startsWith("p") ? -1 : 1; }
    renderTable();
  });

  const track = $("seqtrack");
  track.addEventListener("mousemove", onTrackMove);
  track.addEventListener("mouseleave", () => { hoverResi = null; drawTrack(); });
  track.addEventListener("click", () => { if (hoverResi) focusResidue(hoverResi); });

  window.addEventListener("resize", () => { drawTrack(); viewer.resize(); });
  // Fired by the theme picker and by an OS theme change (see app.js).
  window.addEventListener("themechange", () => {
    viewer.setBackground(cssVar("--surface-0"));
    repaint();
  });
}

export function onShow() {
  if (!$("setup").options.length) {
    renderSetups();
    applySavedSettings();
  }
  viewer.resize();
  drawTrack();
}

/** Adopt the server-side defaults the Settings page writes. */
function applySavedSettings() {
  const s = state.settings;
  if (!s) return;
  $("threshold").value = String(s.threshold);
  $("thr-value").textContent = Number(s.threshold).toFixed(2);
  $("ramp-tick").style.left = `${s.threshold * 100}%`;
  $("colormode").value = s.color_mode;
  $("show-surface").checked = !!s.show_surface;
  $("refine-method").value = s.refine_method || "none";
  $("refine-strength").value = String(s.refine_strength ?? 0.5);
  $("refine-strength-value").textContent = Number(s.refine_strength ?? 0.5).toFixed(2);
  $("gumbel-tau").value = String(s.gumbel_tau ?? 0.5);
  $("gumbel-tau-value").textContent = Number(s.gumbel_tau ?? 0.5).toFixed(2);
  $("gumbel-samples").value = String(s.gumbel_samples ?? 64);
  if ($("refine-method").value !== "none") $("score-source").value = "refined";
  syncRefineControls();
  updateThresholdHint();
}

/* -------------------------------------------------------------- refinement */

/** Show the parameters the chosen method takes, and say what it does. */
function syncRefineControls() {
  const method = $("refine-method").value;
  $("refine-params").hidden = method === "none";
  $("gumbel-params").hidden = method !== "gumbel";
  const arch = state.result?.arch || state.settings?.arch;
  const noContacts = state.result && !state.result.has_contacts;
  $("refine-hint").textContent =
    method === "none"
      ? "Optionally let each residue borrow evidence from the residues it touches " +
        "(Voronoi contacts). Binding sites are patches, not single residues."
      : noContacts
        ? "This prediction has no contact graph (the sequence-only MLP skips the " +
          "tessellation). Press Predict again to compute it."
        : (method === "bayes"
            ? "Adds each contact's log-odds, relative to the chain's baseline, weighted " +
              "by contact area."
            : "Samples which residue anchors each patch, weighted by contact area and " +
              "score; the spread across samples is shown as ± sd.") +
          (arch === "mlp" ? "" : " The GNN models already pass messages over these " +
                                 "contacts, so expect smaller changes than for the MLP.") +
          " Post-processing: the reported PR-AUC is for raw scores.";
}

let refineTimer = null;
let refineSeq = 0;

/** Re-refine the current job's scores. Milliseconds on the server; the model is not
 *  re-run. Debounced for slider drags, and a stale reply is dropped. */
function scheduleRefine() {
  clearTimeout(refineTimer);
  if (!state.jobId || !state.result) { repaint(); return; }
  if (!state.result.has_contacts && $("refine-method").value !== "none") {
    syncRefineControls();
    return;
  }
  refineTimer = setTimeout(async () => {
    const seq = ++refineSeq;
    try {
      const result = await api.post(`/api/v1/jobs/${state.jobId}/refine`, refineSettings());
      if (seq !== refineSeq) return;
      state.result = result;
      syncRefineControls();
      repaint();
    } catch (err) {
      showStatus(`${err.code ? err.code + ": " : ""}${err.message}`,
                 { error: true, hideAfter: 8000 });
    }
  }, 150);
}

/* ------------------------------------------------------------------ inputs */

function renderSetups() {
  const sel = $("setup");
  sel.replaceChildren();
  const setups = state.meta?.setups || {};
  for (const [name, s] of Object.entries(setups)) {
    sel.append(el("option", { value: name }, `${name} — ${s.label_names.join(" / ")}`));
  }
  sel.value = state.settings?.setup || "protein_nucleic";
  renderLabels();
}

/** PR-AUC for a setup/arch/label, from MANIFEST metrics, used as an honesty cue. */
function prAuc(setup, label) {
  const arch = state.settings?.arch || "gnn_mlp";
  const m = state.models?.find((x) => x.setup === setup && x.arch === arch)?.metrics || {};
  return m[`PR AUC (${label})`] ?? m["PR AUC (macro)"] ?? null;
}

function renderLabels() {
  const setup = view.setup;
  const labels = state.meta?.setups?.[setup]?.label_names || [];
  $("setup-hint").textContent = state.meta?.setups?.[setup]?.hint || "";
  renderScoreColumns(labels);

  const box = $("labels");
  box.replaceChildren();
  labels.forEach((label, i) => {
    const auc = prAuc(setup, label);
    box.append(el("label", {},
      el("input", { type: "radio", name: "label", value: String(i), checked: i === 0,
                    onchange: repaint }),
      // A CSS variable, not a resolved colour, so the swatch follows a theme switch.
      el("span", { class: "swatch", style: {
        background: `var(--score-${labelKind(setup, i) === "nucleic" ? "na-" : ""}4)` } }),
      el("span", {}, label),
      el("span", { class: "metric", title: "test-set PR-AUC for this label (raw model)",
                   "data-auc": auc === null ? "" : fmt.n(auc, 2) }, metricText(auc === null
                     ? "" : fmt.n(auc, 2))),
    ));
  });
  updateThresholdHint();
  renderLegend();
}

/** The PR-AUC chip; it describes the raw model, which matters once refined scores show. */
function metricText(auc) {
  if (!auc) return "";
  return `PR-AUC ${auc}${view.source === "raw" ? "" : " (raw model)"}`;
}

/** One sortable score column per label of the selected task. The radio above
 *  picks which of them drives the colouring, the threshold and the downloads;
 *  the table shows them all so a residue can be read across its labels. */
function renderScoreColumns(labels) {
  const tr = $("restable").querySelector("thead tr");
  for (const th of tr.querySelectorAll("th[data-label]")) th.remove();
  labels.forEach((name, i) => {
    tr.append(el("th", { "data-sort": `p${i}`, "data-label": String(i) }, name));
  });
  // Refined − raw for the active label; shown only when there is a refinement.
  tr.append(el("th", { "data-sort": "delta", "data-label": "delta", hidden: true }, "Δ"));
  if (sortKey !== "delta" &&
      (!/^p\d+$/.test(sortKey) || Number(sortKey.slice(1)) >= labels.length)) {
    sortKey = "p0";
    sortDir = -1;
  }
}

function updateThresholdHint() {
  const res = state.result;
  if (!res) {
    $("thr-hint").textContent =
      "0.50 is the operating point every reported metric uses.";
    return;
  }
  const n = res.residues.filter(
    (r) => decisionScore(r, view.setup, view.label) >= view.threshold).length;
  const total = res.residues.length;
  $("thr-hint").textContent =
    `${n} of ${total} predicted residues at or above ${view.threshold.toFixed(2)} ` +
    `(${fmt.pct(n / Math.max(1, total), 0)}).`;
}

async function loadPdbId() {
  const id = $("pdbid").value.trim();
  if (!id) return;
  await withStatus(`fetching ${id.toUpperCase()} from RCSB…`, async () => {
    const info = await api.get(`/api/v1/structures/by-pdb-id/${encodeURIComponent(id)}`);
    setStructure(info);
  });
}

async function loadUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  await withStatus(`reading ${file.name}…`, async () => {
    const info = await api.upload("/api/v1/structures", file);
    setStructure(info);
  });
}

function setStructure(info) {
  state.structure = info;
  const sel = $("chain");
  sel.replaceChildren();
  for (const c of info.chains) {
    sel.append(el("option", { value: c.chain_id },
      `${c.chain_id} — ${c.n_observed} of ${c.n_seqres} residues` +
      (c.numbering_source === "observed" ? " (no SEQRES)" : "")));
  }
  $("chain-field").hidden = false;
  $("chain-hint").textContent =
    `${info.chains.length} protein chain${info.chains.length === 1 ? "" : "s"} in ${info.source}.`;
  $("run").disabled = false;
  $("viewer-title").textContent = info.source;
  renderWarnings(info.warnings || []);
}

/* -------------------------------------------------------------- prediction */

async function runPrediction() {
  const chain = $("chain").value;
  $("run").disabled = true;
  try {
    await withStatus("queued…", async (setText) => {
      const job = await api.post("/api/v1/predict", {
        structure_id: state.structure.structure_id,
        chain_id: chain,
        settings: refineSettings(),
      });
      state.jobId = job.job_id;

      const result = await new Promise((resolve, reject) => {
        const src = new EventSource(`/api/v1/jobs/${job.job_id}/events`);
        src.onmessage = async (ev) => {
          const data = JSON.parse(ev.data);
          if (data.message) setText(data.message);
          if (!data.final) return;
          src.close();
          // Everything after close() must settle the promise: a throw here would
          // otherwise leave it pending and the spinner turning for good.
          try {
            const j = await api.get(`/api/v1/jobs/${job.job_id}`);
            if (j.status === "done") resolve(j.result);
            else reject(Object.assign(new Error(j.error?.message || "prediction failed"),
                                      { code: j.error?.code }));
          } catch (err) {
            reject(err);
          }
        };
        src.onerror = () => { src.close(); reject(new Error("lost connection to the server")); };
      });

      state.result = result;
      await onResult(result, setText);
    });
  } finally {
    $("run").disabled = false;
  }
}

async function onResult(result, setText = () => {}) {
  renderWarnings(result.warnings);
  syncRefineControls();
  $("viewer-title").textContent =
    `${result.source} · chain ${result.chain_id} · ${result.arch}`;
  renderTable();
  drawTrack();
  renderDownloads();
  updateThresholdHint();
  $("seq-meta").textContent =
    `${result.n_predicted} predicted · ${result.unpredicted.length} not predicted · ` +
    `${result.sequence.length} in sequence · ${Object.entries(result.timings_ms)
      .map(([k, v]) => `${k} ${v}ms`).join("  ")}`;
  // Last: the cheap panels should be on screen before the structure parses. The
  // prediction is done by now; say so, rather than leave the last model stage up.
  setText("prediction done — drawing the structure…");
  await loadViewer(result);
}

function renderWarnings(warnings) {
  const box = $("warnings");
  box.replaceChildren();
  for (const w of warnings || []) {
    box.append(el("div", { class: "banner" },
      el("strong", {}, humanizeCode(w.code)), w.detail || ""));
  }
}

const CODE_TITLES = {
  no_seqres: "No SEQRES record — alignment is approximate",
  esm_truncated: "Chain longer than ESM-2's 1022-residue limit",
  esm_range_dropped: "Some residues fall outside the embedding",
  multiple_models: "Multi-model structure",
  nonstandard_residues_dropped: "Non-standard residues removed",
  sequence_mismatch_dropped: "Residues dropped on sequence mismatch",
  partial_alignment: "Partial sequence alignment",
};
const humanizeCode = (c) => CODE_TITLES[c] || c;

/* ---------------------------------------------------------------- 3D view */

async function loadViewer(result) {
  await viewer.mount($("viewer"), { background: cssVar("--surface-0") });
  viewer.onHover(onViewerHover);
  viewer.onClick(focusResidue);
  // The receptor is ~40 KB of PDB text and is deliberately not in the job JSON,
  // so it comes from its own endpoint. Reading it off the result — which never
  // carried it — is what left this viewer empty.
  const pdb = await api.text(`/api/v1/artifacts/${state.jobId}/receptor.pdb`);
  await viewer.loadStructure(pdb);
  await paintViewer();
  viewer.resetCamera();
}

let _byResi = null;
function byResi() {
  if (_byResi && _byResi.__for === state.result) return _byResi;
  const map = { __for: state.result };
  for (const r of state.result?.residues || []) map[r.i] = r;
  _byResi = map;
  return map;
}

/** Displayed value for the task and label (see displayValue); null where there is none. */
function scoreLookup() {
  const map = byResi();
  const setup = view.setup, label = view.label;
  return (resi) => displayValue(map[resi], setup, label);
}

let paintTimer = null;

/** Coalesce the repaints a slider drag produces into one rebuild. */
function schedulePaint() {
  clearTimeout(paintTimer);
  paintTimer = setTimeout(paintViewer, 120);
}

function paintViewer() {
  clearTimeout(paintTimer);
  if (!viewer.isReady() || !state.result) return Promise.resolve();
  const score = scoreLookup();
  const kind = view.kind;
  return viewer.paint({
    colorOf: (resi) => displayColor(score(resi), kind),
    showSurface: $("show-surface").checked,
    unpredicted: cssVar("--unpredicted"),
  }).catch((err) => console.error("could not repaint the viewer", err));
}

function repaint() {
  for (const m of document.querySelectorAll("#labels .metric")) {
    m.textContent = metricText(m.dataset.auc);
  }
  updateThresholdHint();
  renderLegend();
  _byResi = null;
  drawTrack();
  renderTable();
  // The download URLs embed setup, label and threshold, so they go stale here too.
  if (state.jobId) renderDownloads();
  schedulePaint();
}

/** The legend ramp follows the displayed label: blue for protein, red for nucleic acid. */
function renderLegend() {
  const change = view.source === "change";
  $("ramp").classList.toggle("nucleic", !change && view.kind === "nucleic");
  $("ramp").classList.toggle("diverging", change);
  $("ramp-tick").hidden = change;
  $("ramp-lo").textContent = change ? "−0.5" : "0";
  $("ramp-hi").textContent = change ? "+0.5" : "1";
  const name = view.labels[view.label];
  $("ramp-label").textContent = !name ? ""
    : change ? `${name}: refined − raw`
    : `${name} ${view.source === "refined" ? "refined score" : "score"}`;
}

function focusResidue(resi) {
  hoverResi = resi;
  describeResidue(resi);
  drawTrack();
  viewer.focusResidue(resi, byResi()[resi]?.nb || []);
}

/** Mol* hover: keep the track cursor and the hover line in step with the 3D view. */
function onViewerHover(resi) {
  if (resi === hoverResi) return;
  hoverResi = resi;
  if (resi !== null) describeResidue(resi);
  drawTrack();
}

/* --------------------------------------------------------- sequence track */

function drawTrack() {
  const canvas = $("seqtrack");
  const result = state.result;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth || 360;
  const height = 86;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (!result) return;

  const n = result.sequence.length;
  const map = byResi();
  const kind = view.kind;
  const pad = 8;
  const cellW = (width - pad * 2) / n;
  const rows = 3;                        // colour band, threshold marks, ruler
  const bandTop = 12, bandH = 34;

  // One cell per SEQRES position. Canvas, not DOM: a 1000-residue chain would be
  // 1000 nodes and janky on every repaint.
  for (let i = 0; i < n; i++) {
    const r = map[i + 1];
    ctx.fillStyle = displayColor(displayValue(r, view.setup, view.label), kind);
    const x = pad + i * cellW;
    ctx.fillRect(x, bandTop, Math.max(cellW, 0.8), bandH);
    if (!r) {
      // Hatch unobserved/unpredicted positions so "no data" is visually distinct
      // from "low score" even for a viewer who cannot separate the greys.
      ctx.strokeStyle = cssVar("--surface-0");
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      ctx.moveTo(x, bandTop + bandH);
      ctx.lineTo(x + Math.max(cellW, 2), bandTop);
      ctx.stroke();
    }
  }

  ctx.strokeStyle = cssVar("--border");
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, bandTop, width - pad * 2, bandH);

  // Ruler
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "top";
  const tickEvery = niceStep(n, Math.floor((width - 16) / 52));
  for (let i = tickEvery; i <= n; i += tickEvery) {
    const x = pad + (i - 0.5) * cellW;
    ctx.fillRect(x, bandTop + bandH, 1, 4);
    ctx.fillText(String(i), Math.min(x + 3, width - 26), bandTop + bandH + 5);
  }

  if (hoverResi) {
    // The hovered residue's contacts, as ticks above the band: its patch at a glance.
    ctx.fillStyle = cssVar("--text-secondary");
    for (const j of map[hoverResi]?.nb || []) {
      ctx.fillRect(pad + (j - 1) * cellW, bandTop - 6, Math.max(cellW, 1.5), 3);
    }
    const x = pad + (hoverResi - 1) * cellW;
    ctx.strokeStyle = cssVar("--text-primary");
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x - 0.5, bandTop - 3, Math.max(cellW, 2) + 1, bandH + 6);
  }
}

function niceStep(n, targetTicks) {
  const raw = Math.max(1, n / Math.max(1, targetTicks));
  const mag = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

function onTrackMove(e) {
  const result = state.result;
  if (!result) return;
  const canvas = $("seqtrack");
  const rect = canvas.getBoundingClientRect();
  const pad = 8;
  const n = result.sequence.length;
  const cellW = (rect.width - pad * 2) / n;
  const i = Math.floor((e.clientX - rect.left - pad) / cellW) + 1;
  if (i < 1 || i > n) { hoverResi = null; drawTrack(); return; }
  hoverResi = i;
  describeResidue(i);
  drawTrack();
}

/** Fill the hover line for a SEQRES index, from the track or from the viewer. */
function describeResidue(i) {
  const result = state.result;
  if (!result) return;
  const r = byResi()[i];
  const aa = result.sequence[i - 1] ?? "?";
  const setup = view.setup;
  const scores = view.labels.map((name, k) => {
    let t = `${name} ${fmt.n(r?.p[setup][k], 3)}`;
    if (r?.pr) t += ` → ${fmt.n(r.pr[setup][k], 3)}`;
    if (r?.psd) t += ` ± ${fmt.n(r.psd[setup][k], 3)}`;
    return t;
  }).join(" · ");
  // Contacts with the score the colouring uses, strongest contact first.
  const map = byResi();
  const contacts = (r?.nb || []).slice(0, 8).map((j) =>
    `${result.sequence[j - 1] ?? "?"}${j} ${fmt.n(decisionScore(map[j], setup, view.label), 2)}`);
  $("seq-hover").textContent = r
    ? `${aa}${i} (auth ${r.chain}/${r.auth}${r.icode}) · ${scores}` +
      ` · SASA ${r.sas === null ? "—" : fmt.n(r.sas, 1)} Å²` +
      (contacts.length ? ` · contacts: ${contacts.join(", ")}` +
                         (r.nb.length > 8 ? ` +${r.nb.length - 8}` : "") : "")
    : `${aa}${i} · not predicted (unobserved in the structure, or buried)`;
}

/* ------------------------------------------------------------------ table */

function renderTable() {
  const tbody = $("restable").querySelector("tbody");
  tbody.replaceChildren();
  const result = state.result;
  if (!result) return;

  const setup = view.setup, label = view.label, labels = view.labels;
  const steps = labels.map((_, k) => rampSteps(labelKind(setup, k)));
  const refined = !!result.refinement;
  const delta = (r) => (r.pr ? r.pr[setup][label] - r.p[setup][label] : 0);
  const rows = [...result.residues].sort((a, b) => {
    const get = (r) => (/^p\d+$/.test(sortKey) ? decisionScore(r, setup, Number(sortKey.slice(1)))
      : sortKey === "delta" ? delta(r)
      : sortKey === "aa" ? r.aa
      : sortKey === "sas" ? (r.sas ?? -1)
      : sortKey === "auth" ? r.auth : r.i);
    const x = get(a), y = get(b);
    return (x > y ? 1 : x < y ? -1 : 0) * sortDir;
  });

  for (const th of $("restable").querySelectorAll("th[data-sort]")) {
    const base = th.textContent.replace(/[ ▲▼]+$/, "");
    th.textContent = th.dataset.sort === sortKey ? `${base} ${sortDir < 0 ? "▼" : "▲"}` : base;
    th.classList.toggle("active-label", th.dataset.label === String(label));
    if (th.dataset.sort === "delta") th.hidden = !refined;
  }

  // Render a bounded slice: a 1000-residue chain with a full DOM table is slow to
  // build and slower to re-sort, and nobody scrolls past the first few hundred.
  const LIMIT = 400;
  const frag = document.createDocumentFragment();
  for (const r of rows.slice(0, LIMIT)) {
    const hit = decisionScore(r, setup, label) >= view.threshold;
    frag.append(el("tr", {
      class: hit ? "hit" : "",
      onmouseenter: () => { hoverResi = r.i; drawTrack(); },
      onclick: () => focusResidue(r.i),
      style: { cursor: "pointer" },
    },
      el("td", {}, r.i),
      el("td", {}, `${r.auth}${r.icode}`),
      el("td", {}, r.aa),
      el("td", {}, r.sas === null ? "—" : fmt.n(r.sas, 1)),
      ...labels.map((_, k) => {
        const p = decisionScore(r, setup, k);
        return el("td", { class: k === label ? "active-label" : "" },
          el("div", { class: "bar-cell" },
            el("div", { class: "bar", style: { width: `${Math.max(2, p * 46)}px`,
                                               background: scoreColor(p, { steps: steps[k] }) } }),
            el("span", {}, fmt.n(p, 3))));
      }),
      refined ? el("td", { class: "mono" },
        `${delta(r) > 0 ? "+" : ""}${fmt.n(delta(r), 3)}`) : null,
    ));
  }
  tbody.append(frag);
  if (rows.length > LIMIT) {
    tbody.append(el("tr", {}, el("td", { colspan: String(4 + labels.length + (refined ? 1 : 0)),
                                          class: "muted",
      style: { textAlign: "center", padding: "10px" } },
      `showing the top ${LIMIT} of ${rows.length} — download the CSV for all`)));
  }
}

/* -------------------------------------------------------------- downloads */

function renderDownloads() {
  const box = $("downloads");
  const job = state.jobId;
  // Everything the command-line script writes, for the current task and label, and for
  // the scores on display: "change" downloads the refined scores it is drawn from.
  const source = view.source === "raw" ? "raw" : "refined";
  const q = `setup=${view.setup}&label=${view.label}&threshold=${view.threshold}` +
            `&source=${source}`;
  box.replaceChildren(
    link(`/api/v1/artifacts/${job}/report.html?threshold=${view.threshold}&source=${source}`,
         "report.html",
         "This page as one standalone file — Mol* included, opens anywhere, no server."),
    link(`/api/v1/artifacts/${job}/figure.png?${q}`, "figure.png",
         "The four-panel figure: two 3D views, top residues, score along the chain."),
    link(`/api/v1/artifacts/${job}/predictions.csv`, "predictions.csv",
         "Every residue, every label, every task" +
         (state.result?.refinement ? " — raw and refined columns." : ".")),
    link(`/api/v1/artifacts/${job}/predictions.pdb?setup=${view.setup}&label=${view.label}` +
         `&source=${source}`,
         "predictions.pdb",
         "Structure with the current score in the B-factor column (−1.00 = not predicted)."),
    link(`/api/v1/artifacts/${job}/receptor.pdb`, "receptor.pdb",
         "The exact atoms the model saw, renumbered to SEQRES indices."),
    link(`/api/v1/artifacts/${job}/session.pml?${q}`, "session.pml",
         "PyMOL session: loads predictions.pdb, colours by score, shows the hits."),
    link(`/api/v1/artifacts/${job}/session.cxc?${q}`, "session.cxc",
         "The same session for ChimeraX."),
    el("button", { onclick: copyPymol }, "Copy PyMOL selection",
       el("span", { class: "sub" }, "select … chain X and resi 12+15+18 …")),
  );
}

function link(href, title, sub) {
  return el("a", { href, download: "" }, title, el("span", { class: "sub" }, sub));
}

async function copyPymol() {
  const url = `/api/v1/artifacts/${state.jobId}/pymol.txt` +
    `?setup=${view.setup}&label=${view.label}&threshold=${view.threshold}` +
    `&source=${view.source === "raw" ? "raw" : "refined"}`;
  const text = await (await fetch(url)).text();
  try {
    await navigator.clipboard.writeText(text);
    flash("copied to clipboard");
  } catch {
    window.prompt("Copy the selection:", text);
  }
}

/* ----------------------------------------------------------------- status */

let statusTimer = null;

/** The status line above the viewer. `null` hides it; the spinner shows only while
 *  `busy`. A pending auto-hide is cancelled by whatever status comes next, so an old
 *  error's timer can never blank a newer message. */
function showStatus(message, { busy = false, error = false, hideAfter = 0 } = {}) {
  clearTimeout(statusTimer);
  const bar = $("status");
  bar.hidden = message === null;
  bar.querySelector(".spinner").hidden = !busy;
  bar.style.color = error ? cssVar("--danger") : "";
  $("status-text").textContent = message ?? "";
  if (hideAfter) statusTimer = setTimeout(() => showStatus(null), hideAfter);
}

async function withStatus(initial, fn) {
  showStatus(initial, { busy: true });
  try {
    await fn((t) => showStatus(t, { busy: true }));
    showStatus(null);
  } catch (err) {
    showStatus(`${err.code ? err.code + ": " : ""}${err.message}`,
               { error: true, hideAfter: 8000 });
    console.error(err);
  }
}

function flash(message) {
  showStatus(message, { hideAfter: 1600 });
}
