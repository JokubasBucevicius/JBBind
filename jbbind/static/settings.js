/* Settings page — server-persisted defaults shared by the UI and the CLI. */

import { api, archColor, el, fmt, state } from "/static/app.js";

const ARCH_ORDER = ["gnn_mlp", "joint", "mlp", "gnn"];
let caches = [];
let dirty = {};

const $ = (id) => document.getElementById(id);

export function init() { /* rendered on first show */ }

export async function onShow() {
  await refresh();
}

async function refresh() {
  try {
    const data = await api.get("/api/v1/settings");
    state.settings = data.settings;
    caches = data.caches;
  } catch (err) {
    $("settings-body").replaceChildren(
      el("div", { class: "banner error" }, el("strong", {}, "Could not load settings"),
         err.message));
    return;
  }
  dirty = {};
  render();
}

function render() {
  const s = state.settings;
  const models = state.models || [];
  const setups = state.meta?.setups || {};

  const archAvailable = (a) => models.some((m) => m.arch === a);
  const archMetric = (a) => {
    const vals = models.filter((m) => m.arch === a)
      .map((m) => m.metrics?.["PR AUC (macro)"]).filter((v) => v !== undefined);
    return vals.length ? vals.reduce((x, y) => x + y, 0) / vals.length : null;
  };

  $("settings-body").replaceChildren(
    el("div", { class: "card" },
      el("h3", {}, "Model"),
      el("p", { class: "sub" },
        "Applies to new predictions and to `jbbind batch`. Drop a checkpoint into " +
        "models/<setup>/<arch>/ and it appears here."),

      row("Architecture",
        "Which trained network runs. GNN+MLP and Joint are consistently the strongest; " +
        "the structure-only GNN is far behind and is kept as a control.",
        el("div", { class: "radio-list" },
          ...ARCH_ORDER.map((a) => {
            const avail = archAvailable(a);
            const auc = archMetric(a);
            return el("label", { style: { opacity: avail ? 1 : 0.45 } },
              el("input", { type: "radio", name: "set-arch", value: a,
                            checked: s.arch === a, disabled: !avail,
                            onchange: () => { dirty.arch = a; markDirty(); } }),
              el("span", { class: "swatch", style: { background: archColor(a) } }),
              el("span", {}, archLabel(a)),
              el("span", { class: "metric" },
                 avail ? (auc === null ? "" : `mean PR-AUC ${fmt.n(auc, 3)}`) : "not installed"));
          }))),

      row("Default task", "Which label setup the Predict page opens with.",
        select(Object.keys(setups).map((k) => [k,
          `${k} — ${setups[k].label_names.join(" / ")}`]), s.setup,
          (v) => { dirty.setup = v; markDirty(); })),
    ),

    el("div", { class: "card" },
      el("h3", {}, "Inference"),
      row("Device", "CUDA is used automatically when a GPU is visible.",
        select([["auto", `auto (currently ${state.meta?.device || "?"})`],
                ["cpu", "cpu"], ["cuda", "cuda"]], s.device,
          (v) => { dirty.device = v; markDirty(); })),

      row("Long chains",
        "ESM-2 was run with a 1022-token limit when the training embeddings were " +
        "generated. `truncate` reproduces that exactly; `tile` covers the whole chain " +
        "with overlapping windows but is out of distribution.",
        select([["truncate", "truncate at 1022 (training parity)"],
                ["tile", "tile with overlapping windows (experimental)"]],
          s.esm_long_seq_mode, (v) => { dirty.esm_long_seq_mode = v; markDirty(); })),

      row("RCSB source",
        "The asymmetric unit keeps chain IDs as deposited. Assembly 1 is closer to how " +
        "the training structures were prepared, but renames chains.",
        select([["", "asymmetric unit"], ["1", "biological assembly 1"]],
          s.rcsb_assembly === null ? "" : String(s.rcsb_assembly),
          (v) => { dirty.rcsb_assembly = v === "" ? null : Number(v); markDirty(); })),
    ),

    el("div", { class: "card" },
      el("h3", {}, "Decision threshold"),
      el("p", { class: "sub" },
        "Every reported metric uses 0.50. Because training used focal loss with class " +
        "re-weighting, the raw sigmoid over-predicts positives — raising the threshold " +
        "trades recall for precision, but the scores remain uncalibrated either way."),
      row("Default threshold", "Used for colouring, the residue table and PyMOL selections.",
        el("div", {},
          el("input", { type: "range", min: "0", max: "1", step: "0.01",
                        value: String(s.threshold),
                        oninput: (e) => {
                          $("thr-readout").textContent = Number(e.target.value).toFixed(2);
                          dirty.threshold = Number(e.target.value); markDirty();
                        } }),
          el("div", { class: "mono", id: "thr-readout",
                      style: { marginTop: "4px" } }, s.threshold.toFixed(2)))),
    ),

    el("div", { class: "card" },
      el("h3", {}, "Display"),
      row("Colouring", "How the score maps onto the structure.",
        select([["continuous", "continuous score ramp"],
                ["threshold", "above / below threshold"]], s.color_mode,
          (v) => { dirty.color_mode = v; markDirty(); })),
      row("Molecular surface",
        "Off by default — surface generation is the slow part of rendering.",
        checkbox(s.show_surface, (v) => { dirty.show_surface = v; markDirty(); })),
      row("Side-chain sticks", "Show sticks for residues above the threshold.",
        checkbox(s.show_sidechains, (v) => { dirty.show_sidechains = v; markDirty(); })),
    ),

    el("div", { class: "card" },
      el("h3", {}, "Cache"),
      el("p", { class: "sub" },
        "Everything is content-addressed, so clearing only costs recomputation. " +
        "The ESM layer is the expensive one — it is keyed by sequence hash and shared " +
        "across tasks, architectures and entries."),
      el("div", { class: "chart-scroll" },
        el("table", { class: "data" },
          el("thead", {}, el("tr", {}, el("th", {}, "Layer"), el("th", {}, "Entries"),
            el("th", {}, "Size"), el("th", {}, "Cap"), el("th", {}, ""))),
          el("tbody", {}, ...caches.map((c) => el("tr", {},
            el("td", {}, c.namespace),
            el("td", {}, fmt.int(c.entries)),
            el("td", {}, bytes(c.bytes)),
            el("td", {}, c.max_bytes ? bytes(c.max_bytes) : "—"),
            el("td", {}, el("button", { class: "secondary",
              onclick: () => clearCache(c.namespace) }, "Clear"))))))),
    ),

    el("div", { class: "save-bar" },
      el("button", { class: "primary", id: "save", style: { width: "auto" },
                     disabled: true, onclick: save }, "Save settings"),
      el("button", { class: "secondary", onclick: refresh }, "Discard"),
      el("span", { class: "msg", id: "save-msg" }, "")),
  );
}

const archLabel = (a) =>
  ({ gnn_mlp: "GNN+MLP — structure + sequence", joint: "Joint — end-to-end",
     mlp: "MLP — sequence only", gnn: "GNN — structure only" })[a] || a;

function row(title, hint, control) {
  return el("div", { class: "setting-row" },
    el("div", { class: "k" }, title, hint && el("div", { class: "hint" }, hint)),
    el("div", {}, control));
}

function select(options, value, onchange) {
  return el("select", { onchange: (e) => onchange(e.target.value) },
    ...options.map(([v, label]) =>
      el("option", { value: v, selected: String(value) === String(v) }, label)));
}

function checkbox(value, onchange) {
  return el("label", { style: { display: "flex", gap: "8px", alignItems: "center" } },
    el("input", { type: "checkbox", checked: value, style: { width: "auto" },
                  onchange: (e) => onchange(e.target.checked) }),
    el("span", { class: "muted", style: { fontSize: "12.5px" } }, "enabled"));
}

function bytes(n) {
  if (!n) return "0";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

function markDirty() {
  const btn = $("save");
  if (btn) btn.disabled = Object.keys(dirty).length === 0;
  const msg = $("save-msg");
  if (msg) msg.textContent = "unsaved changes";
}

async function save() {
  const msg = $("save-msg");
  try {
    const data = await api.put("/api/v1/settings", dirty);
    state.settings = data.settings;
    dirty = {};
    $("save").disabled = true;
    msg.textContent = "saved";
    setTimeout(() => { if (msg.textContent === "saved") msg.textContent = ""; }, 2500);
  } catch (err) {
    msg.textContent = `${err.code}: ${err.message}`;
  }
}

async function clearCache(namespace) {
  await fetch(`/api/v1/cache/clear?namespace=${encodeURIComponent(namespace)}`,
              { method: "POST" });
  await refresh();
}
