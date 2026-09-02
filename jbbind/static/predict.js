/* Predict page: viewer wiring, sequence track, residue table, downloads. */

import { api, cssVar, el, fmt, rampSteps, scoreColor, state } from "/static/app.js";
import * as viewer from "/static/viewer.js";

let sortKey = "p";
let sortDir = -1;
let hoverResi = null;

const $ = (id) => document.getElementById(id);

const view = {
  get setup() { return $("setup").value; },
  get label() { return Number(document.querySelector('input[name="label"]:checked')?.value ?? 0); },
  get threshold() { return Number($("threshold").value); },
  get colorMode() { return $("colormode").value; },
};

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
  $("show-sidechains").addEventListener("change", repaint);

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
    else { sortKey = key; sortDir = key === "p" ? -1 : 1; }
    renderTable();
  });

  const track = $("seqtrack");
  track.addEventListener("mousemove", onTrackMove);
  track.addEventListener("mouseleave", () => { hoverResi = null; drawTrack(); });
  track.addEventListener("click", () => { if (hoverResi) focusResidue(hoverResi); });

  window.addEventListener("resize", () => { drawTrack(); viewer.resize(); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
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
  $("show-sidechains").checked = !!s.show_sidechains;
  updateThresholdHint();
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

  const box = $("labels");
  box.replaceChildren();
  labels.forEach((label, i) => {
    const auc = prAuc(setup, label);
    box.append(el("label", {},
      el("input", { type: "radio", name: "label", value: String(i), checked: i === 0,
                    onchange: repaint }),
      el("span", {}, label),
      el("span", { class: "metric", title: "test-set PR-AUC for this label" },
         auc === null ? "" : `PR-AUC ${fmt.n(auc, 2)}`),
    ));
  });
  updateThresholdHint();
}

function updateThresholdHint() {
  const res = state.result;
  if (!res) {
    $("thr-hint").textContent =
      "0.50 is the operating point every reported metric uses.";
    return;
  }
  const n = res.residues.filter((r) => r.p[view.setup][view.label] >= view.threshold).length;
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
      });
      state.jobId = job.job_id;

      const result = await new Promise((resolve, reject) => {
        const src = new EventSource(`/api/v1/jobs/${job.job_id}/events`);
        src.onmessage = async (ev) => {
          const data = JSON.parse(ev.data);
          if (data.message) setText(data.message);
          if (!data.final) return;
          src.close();
          const j = await api.get(`/api/v1/jobs/${job.job_id}`);
          if (j.status === "done") resolve(j.result);
          else reject(Object.assign(new Error(j.error?.message || "prediction failed"),
                                    { code: j.error?.code }));
        };
        src.onerror = () => { src.close(); reject(new Error("lost connection to the server")); };
      });

      state.result = result;
      await onResult(result);
    });
  } finally {
    $("run").disabled = false;
  }
}

async function onResult(result) {
  renderWarnings(result.warnings);
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
  // Last: the cheap panels should be on screen before the structure parses.
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
  await viewer.loadStructure(result.receptor_pdb || "");
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

/** Score lookup for the displayed task and label; null where there is none. */
function scoreLookup() {
  const map = byResi();
  const setup = view.setup, label = view.label;
  return (resi) => {
    const r = map[resi];
    return r ? r.p[setup][label] : null;
  };
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
  const opts = { mode: view.colorMode, threshold: view.threshold, steps: rampSteps() };
  return viewer.paint({
    colorOf: (resi) => scoreColor(score(resi), opts),
    // Guard the null: an unscored residue must not become a hit at threshold 0.
    isHit: (resi) => { const p = score(resi); return p !== null && p >= view.threshold; },
    showSurface: $("show-surface").checked,
    showSticks: $("show-sidechains").checked,
    unpredicted: cssVar("--unpredicted"),
  }).catch((err) => console.error("could not repaint the viewer", err));
}

function repaint() {
  updateThresholdHint();
  _byResi = null;
  drawTrack();
  renderTable();
  schedulePaint();
}

function focusResidue(resi) {
  hoverResi = resi;
  describeResidue(resi);
  drawTrack();
  viewer.focusResidue(resi);
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
  const steps = rampSteps();
  const opts = { mode: view.colorMode, threshold: view.threshold, steps };
  const pad = 8;
  const cellW = (width - pad * 2) / n;
  const rows = 3;                        // colour band, threshold marks, ruler
  const bandTop = 12, bandH = 34;

  // One cell per SEQRES position. Canvas, not DOM: a 1000-residue chain would be
  // 1000 nodes and janky on every repaint.
  for (let i = 0; i < n; i++) {
    const r = map[i + 1];
    const p = r ? r.p[view.setup][view.label] : null;
    ctx.fillStyle = scoreColor(p, opts);
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
  $("seq-hover").textContent = r
    ? `${aa}${i} (auth ${r.chain}/${r.auth}${r.icode}) · score ${fmt.n(r.p[view.setup][view.label], 3)}` +
      ` · SASA ${r.sas === null ? "—" : fmt.n(r.sas, 1)} Å²`
    : `${aa}${i} · not predicted (unobserved in the structure, buried, or past the ESM limit)`;
}

/* ------------------------------------------------------------------ table */

function renderTable() {
  const tbody = $("restable").querySelector("tbody");
  tbody.replaceChildren();
  const result = state.result;
  if (!result) return;

  const setup = view.setup, label = view.label;
  const rows = [...result.residues].sort((a, b) => {
    const get = (r) => (sortKey === "p" ? r.p[setup][label]
      : sortKey === "aa" ? r.aa
      : sortKey === "sas" ? (r.sas ?? -1)
      : sortKey === "auth" ? r.auth : r.i);
    const x = get(a), y = get(b);
    return (x > y ? 1 : x < y ? -1 : 0) * sortDir;
  });

  for (const th of $("restable").querySelectorAll("th[data-sort]")) {
    const base = th.textContent.replace(/[ ▲▼]+$/, "");
    th.textContent = th.dataset.sort === sortKey ? `${base} ${sortDir < 0 ? "▼" : "▲"}` : base;
  }

  // Render a bounded slice: a 1000-residue chain with a full DOM table is slow to
  // build and slower to re-sort, and nobody scrolls past the first few hundred.
  const LIMIT = 400;
  const frag = document.createDocumentFragment();
  for (const r of rows.slice(0, LIMIT)) {
    const p = r.p[setup][label];
    const hit = p >= view.threshold;
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
      el("td", {},
        el("div", { class: "bar-cell" },
          el("div", { class: "bar", style: { width: `${Math.max(2, p * 46)}px`,
                                             background: scoreColor(p, { threshold: view.threshold }) } }),
          el("span", {}, fmt.n(p, 3)))),
    ));
  }
  tbody.append(frag);
  if (rows.length > LIMIT) {
    tbody.append(el("tr", {}, el("td", { colspan: "5", class: "muted",
      style: { textAlign: "center", padding: "10px" } },
      `showing the top ${LIMIT} of ${rows.length} — download the CSV for all`)));
  }
}

/* -------------------------------------------------------------- downloads */

function renderDownloads() {
  const box = $("downloads");
  const job = state.jobId;
  box.replaceChildren(
    link(`/api/v1/artifacts/${job}/predictions.csv`, "predictions.csv",
         "Every residue, every label, every task."),
    link(`/api/v1/artifacts/${job}/predictions.pdb?setup=${view.setup}&label=${view.label}`,
         "predictions.pdb",
         "Structure with the current score in the B-factor column (−1.00 = not predicted)."),
    link(`/api/v1/artifacts/${job}/receptor.pdb`, "receptor.pdb",
         "The exact atoms the model saw, renumbered to SEQRES indices."),
    el("button", { onclick: copyPymol }, "Copy PyMOL selection",
       el("span", { class: "sub" }, "select … chain X and resi 12+15+18 …")),
  );
}

function link(href, title, sub) {
  return el("a", { href, download: "" }, title, el("span", { class: "sub" }, sub));
}

async function copyPymol() {
  const url = `/api/v1/artifacts/${state.jobId}/pymol.txt` +
    `?setup=${view.setup}&label=${view.label}&threshold=${view.threshold}`;
  const text = await (await fetch(url)).text();
  try {
    await navigator.clipboard.writeText(text);
    flash("copied to clipboard");
  } catch {
    window.prompt("Copy the selection:", text);
  }
}

/* ----------------------------------------------------------------- status */

async function withStatus(initial, fn) {
  const bar = $("status"), text = $("status-text");
  bar.hidden = false;
  bar.classList.remove("banner", "error");
  text.textContent = initial;
  const setText = (t) => { text.textContent = t; };
  try {
    await fn(setText);
    bar.hidden = true;
  } catch (err) {
    bar.querySelector(".spinner").style.visibility = "hidden";
    text.textContent = `${err.code ? err.code + ": " : ""}${err.message}`;
    bar.style.color = cssVar("--danger");
    setTimeout(() => {
      bar.hidden = true;
      bar.style.color = "";
      bar.querySelector(".spinner").style.visibility = "";
    }, 8000);
    console.error(err);
  }
}

function flash(message) {
  const bar = $("status"), text = $("status-text");
  bar.hidden = false;
  bar.querySelector(".spinner").style.visibility = "hidden";
  text.textContent = message;
  setTimeout(() => {
    bar.hidden = true;
    bar.querySelector(".spinner").style.visibility = "";
  }, 1600);
}
