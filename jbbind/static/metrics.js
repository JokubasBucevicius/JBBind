/* Metrics page — how the trained models actually perform.
 *
 * Data comes from models/METRICS.json, precomputed by scripts/export_models.py from the
 * training runs' metrics.json + predictions.npz (curves downsampled to ~200 points).
 *
 * Colour: the four architectures occupy fixed categorical slots 1-4 (never cycled, never
 * rank-dependent), validated on the adjacent pairlist in both modes by
 * scripts/validate_palette.py. Two light-mode slots sit under 3:1 on the light surface, so
 * the relief rule applies and every chart here ships direct labels AND a table view.
 */

import { api, archColor, cssVar, el, fmt, state } from "/static/app.js";

const ARCH_ORDER = ["gnn_mlp", "joint", "mlp", "gnn"];
const ARCH_LABEL = { gnn_mlp: "GNN+MLP", joint: "Joint", mlp: "MLP", gnn: "GNN" };
let metrics = null;
let loaded = false;

const $ = (id) => document.getElementById(id);

export function init() {
  $("m-setup").addEventListener("change", render);
}

export async function onShow() {
  if (loaded) return;
  loaded = true;
  try {
    metrics = await api.get("/api/v1/metrics");
  } catch (err) {
    $("metrics-body").replaceChildren(
      el("div", { class: "banner error" },
        el("strong", {}, "Could not load metrics"), err.message));
    return;
  }
  const sel = $("m-setup");
  sel.replaceChildren();
  for (const name of Object.keys(metrics.setups)) {
    const labels = state.meta?.setups?.[name]?.label_names?.join(" / ") || "";
    sel.append(el("option", { value: name }, `${name} — ${labels}`));
  }
  sel.value = state.settings?.setup || Object.keys(metrics.setups)[0];
  render();
}

function render() {
  const setup = $("m-setup").value;
  const node = metrics.setups[setup];
  const labels = state.meta?.setups?.[setup]?.label_names || [];
  $("m-setup-desc").textContent = state.meta?.setups?.[setup]?.hint || "";

  // Several of these return null when a setup has no curves or no 3-class table;
  // replaceChildren would stringify a null into the page, so drop them here.
  const cards = [
    caveats(setup),
    datasetCard(setup),
    comparisonCard(setup, node, labels),
    curvesCard(setup, node, labels, "pr"),
    curvesCard(setup, node, labels, "roc"),
    confusionCard(setup, node, labels),
    threeClassCard(setup),
  ].filter(Boolean);
  $("metrics-body").replaceChildren(...cards);
}

/* ------------------------------------------------------------------ cards */

function caveats(setup) {
  const node = metrics.setups[setup];
  const best = ARCH_ORDER
    .map((a) => [a, node.archs[a]?.metrics?.["PR AUC (macro)"]])
    .filter(([, v]) => v !== undefined)
    .sort((x, y) => y[1] - x[1])[0];

  return el("div", { class: "card" },
    el("h3", {}, "How to read these numbers"),
    el("ul", { style: { margin: "8px 0 0", paddingLeft: "18px", fontSize: "12.5px",
                        color: "var(--text-secondary)", lineHeight: "1.7" } },
      el("li", {}, el("strong", {}, "Scores are not calibrated probabilities. "),
        "Training used focal loss with per-label pos_weight, which pushes the model to " +
        "over-predict positives — recall runs far ahead of precision at the 0.50 " +
        "operating point. Treat the output as a ranking score."),
      el("li", {}, el("strong", {}, "Every number is at a fixed threshold of 0.50. "),
        "No per-label threshold tuning was done."),
      el("li", {}, el("strong", {}, "PR-AUC is the metric to trust here. "),
        "These classes are heavily imbalanced, so ROC-AUC flatters every model; the PR " +
        "baseline (the dashed line on each PR panel) is the positive rate."),
      el("li", {}, el("strong", {}, "About 5.6% of training chains were mis-aligned. "),
        "A sample of 250 chains found that fraction where the structure's residue " +
        "numbering did not index its own sequence, so those chains trained against " +
        "effectively random embeddings. This caps what these numbers could have been."),
      best && el("li", {}, `Best architecture for this task: `,
        el("strong", {}, ARCH_LABEL[best[0]]), ` at PR-AUC ${fmt.n(best[1], 3)} (macro).`),
    ));
}

function datasetCard(setup) {
  const d = metrics.dataset?.[setup];
  const node = metrics.setups[setup];
  const anyArch = node.archs[ARCH_ORDER.find((a) => node.archs[a])] || {};

  const stats = [
    d && ["Qualifying chains", fmt.int(d.qualifying), `of ${fmt.int(d.available)} available`],
    d && ["Train / val / test", `${fmt.int(d.train_chains)} / ${fmt.int(d.val_chains)} / ${fmt.int(d.test_chains)}`, "chains, split by chain"],
    ["Test residues", fmt.int(anyArch.n_test_residues), "surface residues scored"],
  ].filter(Boolean);

  const balance = d?.class_balance || {};
  const positives = anyArch.positives_per_label || {};

  return el("div", { class: "card" },
    el("h3", {}, "Dataset"),
    el("p", { class: "sub" },
      "Chains qualify for a task if they have at least one binding residue of that type, " +
      "so each task trains on a different subset."),
    el("div", { class: "stat-row" },
      ...stats.map(([label, value, sub]) =>
        el("div", { class: "stat" },
          el("div", { class: "value" }, value),
          el("div", { class: "label" }, label),
          sub && el("div", { class: "label" }, sub)))),
    Object.keys(positives).length ? el("div", { class: "chart-scroll",
                                                style: { marginTop: "14px" } },
      el("table", { class: "data" },
        el("thead", {}, el("tr", {},
          el("th", {}, "Label"), el("th", {}, "Positive test residues"),
          el("th", {}, "Positive rate"),
          el("th", {}, "Train positives"))),
        el("tbody", {}, ...Object.entries(positives).map(([label, count]) =>
          el("tr", {},
            el("td", {}, label),
            el("td", {}, fmt.int(count)),
            el("td", {}, fmt.pct(count / (anyArch.n_test_residues || 1), 2)),
            el("td", {}, balance[label]
              ? `${fmt.int(balance[label].count)} (${balance[label].pct}%)` : "—")))))) : null);
}

function comparisonCard(setup, node, labels) {
  const metricKeys = [
    ["PR AUC", (l) => `PR AUC (${l})`],
    ["ROC AUC", (l) => `ROC AUC (${l})`],
    ["F1", (l) => `F1 (${l})`],
    ["Precision", (l) => `Precision (${l})`],
    ["Recall", (l) => `Recall (${l})`],
  ];

  const archs = ARCH_ORDER.filter((a) => node.archs[a]);
  const groups = [];
  for (const label of labels) {
    for (const [name, key] of metricKeys.slice(0, 3)) {
      groups.push({
        title: labels.length > 1 ? `${name} — ${label}` : name,
        values: archs.map((a) => ({
          arch: a, value: node.archs[a].metrics?.[key(label)] ?? null,
        })),
      });
    }
  }

  return el("div", { class: "card" },
    el("h3", {}, "Architecture comparison"),
    el("p", { class: "sub" },
      "Higher is better. The structure-only GNN is the control: the gap between it and " +
      "everything else is the contribution of the ESM-2 sequence embedding."),
    legend(archs),
    el("div", { class: "chart-grid" }, ...groups.map(groupedBars)),
    el("div", { class: "chart-scroll", style: { marginTop: "16px" } },
      metricsTable(node, labels, metricKeys, archs)));
}

function curvesCard(setup, node, labels, kind) {
  const archs = ARCH_ORDER.filter((a) => node.archs[a]?.curves);
  if (!archs.length) return null;
  const title = kind === "pr" ? "Precision–recall curves" : "ROC curves";
  const sub = kind === "pr"
    ? "The dashed line is the positive rate — the precision a coin flip would get. " +
      "Distance above it is what the model adds."
    : "The dashed diagonal is chance. With classes this imbalanced, ROC-AUC looks " +
      "generous; prefer the PR panel above.";

  return el("div", { class: "card" },
    el("h3", {}, title),
    el("p", { class: "sub" }, sub),
    legend(archs),
    el("div", { class: "chart-grid" },
      ...labels.map((label) => curvePanel(node, archs, label, kind))));
}

function confusionCard(setup, node, labels) {
  const archs = ARCH_ORDER.filter((a) => node.archs[a]?.confusion);
  if (!archs.length) return null;
  return el("div", { class: "card" },
    el("h3", {}, "Confusion at threshold 0.50"),
    el("p", { class: "sub" },
      "Row-normalised: each row sums to 100%. The bottom row is recall; the wide " +
      "false-positive column on the top row is the over-prediction the focal loss buys."),
    el("div", { class: "chart-grid" },
      ...labels.flatMap((label) => archs.map((a) => confusionPanel(node, a, label)))
               .filter(Boolean)));
}

function threeClassCard(setup) {
  const rows = metrics.three_class?.[setup];
  if (!rows?.length) return null;
  const cols = Object.keys(rows[0]).filter((c) => c !== "");
  return el("div", { class: "card" },
    el("h3", {}, "Binary vs 3-class comparison"),
    el("p", { class: "sub" },
      "The same architectures scored on a shared 3-class task, comparing models trained " +
      "as independent binary labels against models trained on the 3-class formulation."),
    el("div", { class: "chart-scroll" },
      el("table", { class: "data" },
        el("thead", {}, el("tr", {}, el("th", {}, "Model"),
          ...cols.map((c) => el("th", {}, c)))),
        el("tbody", {}, ...rows.map((r) => el("tr", {},
          el("td", {}, r[""] ?? r.model ?? ""),
          ...cols.map((c) => el("td", {}, fmt.n(Number(r[c]), 3)))))))));
}

/* ----------------------------------------------------------------- charts */

function legend(archs) {
  return el("div", { class: "chart-legend" },
    ...archs.map((a) => el("span", { class: "series-key" },
      el("i", { style: { background: archColor(a) } }), ARCH_LABEL[a])));
}

const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}, ...kids) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c) n.append(c);
  return n;
};
const svgText = (x, y, text, attrs = {}) => {
  const n = svgEl("text", { x, y, "font-size": 10.5, fill: cssVar("--text-muted"), ...attrs });
  n.textContent = text;
  return n;
};

function groupedBars({ title, values }) {
  const W = 330, H = 190, padL = 40, padR = 12, padT = 10, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const max = Math.max(0.2, ...values.map((v) => v.value ?? 0)) * 1.15;
  const bw = Math.min(46, (plotW / values.length) - 10);

  const g = [svgEl("line", { x1: padL, y1: padT + plotH, x2: W - padR, y2: padT + plotH,
                             stroke: cssVar("--border-strong"), "stroke-width": 1 })];

  for (let i = 0; i <= 4; i++) {
    const v = (max * i) / 4;
    const y = padT + plotH - (v / max) * plotH;
    if (i > 0) g.push(svgEl("line", { x1: padL, y1: y, x2: W - padR, y2: y,
                                      stroke: cssVar("--surface-3"), "stroke-width": 1 }));
    g.push(svgText(padL - 6, y + 3.5, v.toFixed(1), { "text-anchor": "end" }));
  }

  values.forEach((v, i) => {
    const cx = padL + (plotW / values.length) * (i + 0.5);
    const x = cx - bw / 2;
    const h = v.value === null ? 0 : (v.value / max) * plotH;
    const y = padT + plotH - h;
    if (h > 0) {
      // 4px rounded data-end anchored to the baseline: round the top corners only.
      const r = Math.min(4, h);
      g.push(svgEl("path", {
        d: `M${x},${y + h} L${x},${y + r} Q${x},${y} ${x + r},${y} ` +
           `L${x + bw - r},${y} Q${x + bw},${y} ${x + bw},${y + r} L${x + bw},${y + h} Z`,
        fill: archColor(v.arch),
      }));
    }
    // Direct label on every bar — required by the relief rule, since two of the four
    // series sit below 3:1 against the light surface.
    g.push(svgText(cx, y - 5, v.value === null ? "—" : v.value.toFixed(3),
      { "text-anchor": "middle", fill: cssVar("--text-primary"), "font-size": 10,
        "font-weight": 600 }));
    g.push(svgText(cx, padT + plotH + 14, ARCH_LABEL[v.arch], { "text-anchor": "middle" }));
  });

  return el("figure", { style: { margin: 0 } },
    el("figcaption", { style: { fontSize: "12.5px", fontWeight: "550", marginBottom: "4px" } },
      title),
    svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
                   "aria-label": title }, ...g));
}

function curvePanel(node, archs, label, kind) {
  const W = 330, H = 300, padL = 42, padR = 12, padT = 10, padB = 38;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const X = (v) => padL + v * plotW;
  const Y = (v) => padT + (1 - v) * plotH;

  const g = [];
  for (let i = 0; i <= 4; i++) {
    const t = i / 4;
    g.push(svgEl("line", { x1: padL, y1: Y(t), x2: W - padR, y2: Y(t),
                           stroke: cssVar("--surface-3"), "stroke-width": 1 }));
    g.push(svgText(padL - 6, Y(t) + 3.5, t.toFixed(2), { "text-anchor": "end" }));
    g.push(svgText(X(t), padT + plotH + 14, t.toFixed(2), { "text-anchor": "middle" }));
  }

  const first = node.archs[archs[0]].curves[label];
  if (kind === "pr" && first?.baseline !== undefined) {
    g.push(svgEl("line", { x1: padL, y1: Y(first.baseline), x2: W - padR, y2: Y(first.baseline),
                           stroke: cssVar("--text-muted"), "stroke-width": 1.5,
                           "stroke-dasharray": "5 4" }));
    g.push(svgText(W - padR - 2, Y(first.baseline) - 4,
      `baseline ${fmt.n(first.baseline, 3)}`, { "text-anchor": "end" }));
  } else if (kind === "roc") {
    g.push(svgEl("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1),
                           stroke: cssVar("--text-muted"), "stroke-width": 1.5,
                           "stroke-dasharray": "5 4" }));
  }

  for (const arch of archs) {
    const c = node.archs[arch].curves?.[label];
    if (!c) continue;
    const pts = kind === "pr" ? c.pr : c.roc;
    if (!pts?.length) continue;
    const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(2)},${Y(y).toFixed(2)}`).join("");
    // 2px lines, with a surface-coloured ring underneath so overlapping curves stay
    // separable where they cross.
    g.push(svgEl("path", { d, fill: "none", stroke: cssVar("--surface-0"),
                           "stroke-width": 4, "stroke-linejoin": "round" }));
    g.push(svgEl("path", { d, fill: "none", stroke: archColor(arch), "stroke-width": 2,
                           "stroke-linejoin": "round" }));
  }

  const aucRows = archs.map((a) => {
    const c = node.archs[a].curves?.[label];
    return c ? `${ARCH_LABEL[a]} ${fmt.n(kind === "pr" ? c.pr_auc : c.roc_auc, 3)}` : null;
  }).filter(Boolean);
  aucRows.forEach((row, i) => {
    const [name] = row.split(" ");
    const arch = archs.find((a) => ARCH_LABEL[a] === name);
    g.push(svgEl("rect", { x: padL + 8, y: padT + 8 + i * 15, width: 8, height: 8, rx: 2,
                           fill: archColor(arch) }));
    g.push(svgText(padL + 21, padT + 16 + i * 15, row,
      { fill: cssVar("--text-primary"), "font-size": 10.5 }));
  });

  g.push(svgText(padL + plotW / 2, H - 6, kind === "pr" ? "recall" : "false positive rate",
    { "text-anchor": "middle" }));
  g.push(svgText(-(padT + plotH / 2), 12, kind === "pr" ? "precision" : "true positive rate",
    { "text-anchor": "middle", transform: "rotate(-90)" }));

  return el("figure", { style: { margin: 0 } },
    el("figcaption", { style: { fontSize: "12.5px", fontWeight: "550", marginBottom: "4px" } },
      label),
    svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
                   "aria-label": `${kind} curve for ${label}` }, ...g));
}

function confusionPanel(node, arch, label) {
  const c = node.archs[arch].confusion?.[label];
  if (!c) return null;
  const rows = [
    ["actual negative", c.tn, c.fp],
    ["actual positive", c.fn, c.tp],
  ];
  const ramp = [0, 1, 2, 3, 4, 5, 6].map((i) => cssVar(`--score-${i}`));

  return el("figure", { style: { margin: 0 } },
    el("figcaption", { style: { fontSize: "12.5px", fontWeight: "550", marginBottom: "6px" } },
      el("span", { class: "series-key" },
        el("i", { style: { background: archColor(arch) } }), `${ARCH_LABEL[arch]} — ${label}`)),
    el("table", { class: "data" },
      el("thead", {}, el("tr", {}, el("th", {}, ""),
        el("th", {}, "pred. negative"), el("th", {}, "pred. positive"))),
      el("tbody", {}, ...rows.map(([name, a, b]) => {
        const total = a + b || 1;
        return el("tr", {}, el("td", {}, name),
          ...[a, b].map((v) => {
            const frac = v / total;
            const bg = ramp[Math.min(ramp.length - 1, Math.round(frac * (ramp.length - 1)))];
            return el("td", { style: { background: bg,
                                       color: frac > 0.55 ? cssVar("--surface-0") : cssVar("--text-primary") } },
              `${fmt.pct(frac, 1)}`, el("div", { class: "muted",
                style: { fontSize: "10.5px", color: "inherit", opacity: 0.75 } }, fmt.int(v)));
          }));
      }))));
}

function metricsTable(node, labels, metricKeys, archs) {
  const header = el("tr", {}, el("th", {}, "Architecture"),
    ...labels.flatMap((l) => metricKeys.map(([name]) =>
      el("th", {}, labels.length > 1 ? `${name} (${l})` : name))));

  const bodyRows = archs.map((a) => {
    const m = node.archs[a].metrics || {};
    return el("tr", {},
      el("td", {}, el("span", { class: "series-key" },
        el("i", { style: { background: archColor(a) } }), ARCH_LABEL[a])),
      ...labels.flatMap((l) => metricKeys.map(([, key]) => {
        const v = m[key(l)];
        const best = Math.max(...archs.map((x) => node.archs[x].metrics?.[key(l)] ?? -1));
        return el("td", { class: v === best ? "best" : "" }, fmt.n(v, 3));
      })));
  });

  return el("table", { class: "data" },
    el("thead", {}, header), el("tbody", {}, ...bodyRows));
}
