/* JBBind — shared state, API client, colour scale, page router. */

export const api = {
  async get(url) {
    const r = await fetch(url);
    if (!r.ok) throw await toError(r);
    return r.json();
  },
  async put(url, body) {
    const r = await fetch(url, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw await toError(r);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw await toError(r);
    return r.json();
  },
  async upload(url, file) {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(url, { method: "POST", body: fd });
    if (!r.ok) throw await toError(r);
    return r.json();
  },
};

async function toError(response) {
  let payload = {};
  try { payload = await response.json(); } catch { /* non-JSON error body */ }
  const err = new Error(payload.detail || payload.message || response.statusText);
  err.code = payload.code || `HTTP ${response.status}`;
  err.status = response.status;
  return err;
}

/* ------------------------------------------------------------------ colour */

/** Sequential ramp steps, read from CSS so light/dark stay in one place. */
export function rampSteps() {
  const s = getComputedStyle(document.documentElement);
  return [0, 1, 2, 3, 4, 5, 6].map((i) => s.getPropertyValue(`--score-${i}`).trim());
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const hexToRgb = (h) => {
  const v = h.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
};
const rgbToHex = (c) =>
  "#" + c.map((x) => Math.round(Math.max(0, Math.min(255, x))).toString(16).padStart(2, "0")).join("");

/* sRGB <-> linear <-> OKLab, so the ramp is interpolated perceptually: equal
   score steps then read as equal colour steps. */
const s2lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const lin2s = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

function srgbToOklab(hex) {
  const [r, g, b] = hexToRgb(hex).map((v) => s2lin(v / 255));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function oklabToSrgb([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const bb = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;
  return rgbToHex([r, g, bb].map((c) => lin2s(Math.max(0, Math.min(1, c))) * 255));
}

/** Colour for a score in [0,1]; null/undefined -> the out-of-ramp grey. */
export function scoreColor(p, { mode = "continuous", threshold = 0.5, steps = null } = {}) {
  if (p === null || p === undefined || Number.isNaN(p)) return cssVar("--unpredicted");
  const ramp = steps || rampSteps();
  if (mode === "threshold") return p >= threshold ? ramp[ramp.length - 1] : ramp[0];
  const t = Math.max(0, Math.min(1, p)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(t));
  const f = t - i;
  const A = srgbToOklab(ramp[i]);
  const B = srgbToOklab(ramp[i + 1]);
  return oklabToSrgb([0, 1, 2].map((k) => A[k] + (B[k] - A[k]) * f));
}

/** Fixed architecture -> series slot. Never cycled, never rank-dependent. */
export const ARCH_SLOT = { gnn_mlp: 1, joint: 2, mlp: 3, gnn: 4 };
export const archColor = (arch) => cssVar(`--series-${ARCH_SLOT[arch] || 1}`);

export const fmt = {
  n: (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d)),
  int: (x) => (x === null || x === undefined ? "—" : Number(x).toLocaleString()),
  pct: (x, d = 1) => (x === null || x === undefined ? "—" : (100 * x).toFixed(d) + "%"),
};

/* ------------------------------------------------------------------ shared */

export const state = {
  meta: null,
  models: null,
  settings: null,
  structure: null,
  result: null,
  jobId: null,
};

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

/* ------------------------------------------------------------------ router */

const pages = {};
export function registerPage(name, mod) { pages[name] = mod; }

function show(name) {
  for (const p of document.querySelectorAll(".page")) {
    p.classList.toggle("active", p.id === `page-${name}`);
  }
  for (const b of document.querySelectorAll("#nav button")) {
    if (b.dataset.page === name) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  }
  location.hash = name;
  pages[name]?.onShow?.();
}

async function boot() {
  document.getElementById("nav").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-page]");
    if (b) show(b.dataset.page);
  });

  try {
    const [meta, models, settings] = await Promise.all([
      api.get("/api/v1/meta"),
      api.get("/api/v1/models"),
      api.get("/api/v1/settings"),
    ]);
    state.meta = meta;
    state.models = models.models;
    state.settings = settings.settings;
    document.getElementById("device-badge").textContent =
      `${meta.device}${meta.cuda_available && meta.device === "cpu" ? " (GPU available)" : ""}`;
  } catch (err) {
    document.getElementById("device-badge").textContent = "offline";
    console.error("failed to load app metadata", err);
  }

  const [predict, metrics, settingsPage] = await Promise.all([
    import("/static/predict.js"),
    import("/static/metrics.js"),
    import("/static/settings.js"),
  ]);
  registerPage("predict", predict);
  registerPage("metrics", metrics);
  registerPage("settings", settingsPage);
  predict.init();
  metrics.init();
  settingsPage.init();

  const initial = (location.hash || "#predict").slice(1);
  show(pages[initial] ? initial : "predict");
}

boot();
