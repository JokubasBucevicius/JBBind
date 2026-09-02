/* Mol* viewer: structure display, per-residue score colouring, hover and focus.

   Everything Mol*-specific lives here. The rest of the app speaks in residue
   indices and CSS colours and never touches the plugin, so the engine stays
   swappable and predict.js stays about prediction.

   Residue identity is the PDB residue number of `receptor.pdb`, which the server
   renumbers to SEQRES indices — the same `r.i` the sequence track and the table
   use. Mol* calls that auth_seq_id.
*/

const SCORE_THEME = "jbbind-score";

let viewer = null;          // molstar.Viewer
let plugin = null;          // PluginUIContext
let structure = null;       // StateObjectSelector for the Structure node
let owned = [];             // state refs created by the last paint(), ours to delete
let colorByResi = new Map(); // resi -> Mol* Color (0xRRGGBB), rebuilt each paint
let unpredictedColor = 0xb8b7b2;
let paintSeq = 0;           // discards a paint that a newer one has overtaken
let paintQueue = Promise.resolve(); // paints run one at a time, never interleaved
let handlers = { hover: null, click: null };
let mounting = null;        // in-flight mount(), so a second call awaits the first

const S = () => window.molstar.lib.structure;
const T = () => window.molstar.lib.plugin.StateTransforms;
const PluginConfig = () => window.molstar.lib.plugin.PluginConfig;

/** "#rrggbb" -> the integer Mol* means by a Color. */
const toColor = (hex) => parseInt(String(hex).replace("#", ""), 16) || 0;

export const isReady = () => plugin !== null;

/* ---------------------------------------------------------------- lifecycle */

/** Create the plugin inside `container`. Idempotent; safe to call concurrently. */
export function mount(container, options = {}) {
  if (!mounting) mounting = create(container, options);
  return mounting;
}

async function create(container, { background } = {}) {
  viewer = await window.molstar.Viewer.create(container, {
    // No Mol* chrome: the page already has a left rail, a legend and a sequence
    // track, and two sets of controls disagreeing about the same state is worse
    // than none. Reset-camera is the one button kept, because nothing else in
    // the page can undo a spun camera.
    //
    // Every button is named, but not all of them have a viewportShow* option:
    // the illumination and XR buttons are only reachable through `config`, and
    // both default to visible. A Mol* upgrade that adds a button will show it
    // here until it is added below, so check the viewport after one.
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowRemoteState: false,
    layoutShowSequence: false,
    layoutShowLog: false,
    layoutShowLeftPanel: false,
    viewportShowReset: true,
    viewportShowExpand: false,
    viewportShowControls: false,
    viewportShowSettings: false,
    viewportShowSelectionMode: false,
    viewportShowAnimation: false,
    viewportShowToggleFullscreen: false,
    viewportShowTrajectoryControls: false,
    viewportShowScreenshotControls: false,
    volumeStreamingDisabled: true,
    config: [
      [PluginConfig().Viewport.ShowIllumination, false],
      [PluginConfig().Viewport.ShowXR, "never"],
    ],
  });
  plugin = viewer.plugin;
  plugin.representation.structure.themes.colorThemeRegistry.add(scoreThemeProvider());
  if (background) setBackground(background);

  plugin.behaviors.interaction.hover.subscribe((e) => {
    handlers.hover?.(residueOf(e?.current?.loci));
  });
  plugin.behaviors.interaction.click.subscribe((e) => {
    const resi = residueOf(e?.current?.loci);
    if (resi !== null) handlers.click?.(resi);
  });
}

export function onHover(fn) { handlers.hover = fn; }
export function onClick(fn) { handlers.click = fn; }

export function resize() { viewer?.handleResize(); }

export function setBackground(hex) {
  plugin?.canvas3d?.setProps({ renderer: { backgroundColor: toColor(hex) } });
}

/* ------------------------------------------------------------------ loading */

/** Parse a PDB string and keep the structure; representations come from paint(). */
export async function loadStructure(pdbText) {
  if (!plugin) return;
  paintSeq++;               // any paint still queued is for the outgoing structure
  await plugin.clear();
  owned = [];
  colorByResi = new Map();
  const data = await plugin.builders.data.rawData({ data: pdbText || "", label: "receptor" });
  const trajectory = await plugin.builders.structure.parseTrajectory(data, "pdb");
  const model = await plugin.builders.structure.createModel(trajectory);
  structure = await plugin.builders.structure.createStructure(model);
}

/* ----------------------------------------------------------------- painting */

/**
 * Rebuild the representations for the current scores.
 *
 * `colorOf(resi)` returns a CSS hex colour, `isHit(resi)` whether the residue is
 * at or above the threshold. Both are re-read for every residue on every call,
 * so the caller owns the ramp and this stays ignorant of setups and labels.
 *
 * Representations are torn down and rebuilt rather than recoloured in place: the
 * colour theme carries no parameters, so Mol* has nothing to diff and would not
 * notice new scores under the same theme name. Rebuilding a single chain is a
 * few tens of ms; the caller debounces the threshold slider so a drag does not
 * queue one rebuild per pixel.
 *
 * Calls are serialised and stale ones are dropped. Two overlapping rebuilds
 * would race over `owned`, and the loser would go on adding representations
 * under a component the winner had already deleted.
 */
export function paint(spec) {
  const seq = ++paintSeq;
  const run = () => (seq === paintSeq ? rebuild(spec) : undefined);
  paintQueue = paintQueue.then(run, run);
  return paintQueue;
}

async function rebuild({ colorOf, isHit, showSurface, showSticks, unpredicted }) {
  if (!plugin || !structure?.data) return;
  const struct = structure.data;

  unpredictedColor = toColor(unpredicted);
  colorByResi = new Map();
  eachResidue(struct, (resi) => {
    const c = colorOf(resi);
    colorByResi.set(resi, c === null || c === undefined ? unpredictedColor : toColor(c));
  });

  if (owned.length) {
    const remove = plugin.build();
    for (const ref of owned) remove.delete(ref);
    await remove.commit();
    owned = [];
  }

  const polymer = await plugin.builders.structure.tryCreateComponentStatic(structure, "polymer");
  if (!polymer) return;
  owned.push(polymer.ref);

  await plugin.builders.structure.representation.addRepresentation(polymer, {
    type: "cartoon",
    color: SCORE_THEME,
  });

  if (showSurface) {
    await plugin.builders.structure.representation.addRepresentation(polymer, {
      type: "molecular-surface",
      typeParams: { alpha: 0.85 },
      color: SCORE_THEME,
    });
  }

  if (showSticks) {
    const bundle = bundleOf(struct, isHit);
    if (bundle) {
      const update = plugin.build();
      const hits = update
        .to(structure)
        .apply(T().Model.StructureSelectionFromBundle, { bundle, label: "above threshold" });
      await update.commit();
      owned.push(hits.ref);
      await plugin.builders.structure.representation.addRepresentation(hits.selector, {
        type: "ball-and-stick",
        typeParams: { sizeFactor: 0.22, aromaticBonds: false },
        color: SCORE_THEME,
      });
    }
  }
}

/** Frame one residue and highlight it, for a click on the track or the table. */
export function focusResidue(resi) {
  const loci = lociOf(resi);
  if (!loci) return;
  plugin.managers.camera.focusLoci(loci);
  plugin.managers.interactivity.lociHighlights.highlightOnly({ loci });
}

export function resetCamera() {
  plugin?.managers?.camera?.reset();
}

/* -------------------------------------------------------- structure walking */

/** Call `fn(resi, unit, element)` once per atom, with the residue number. */
function eachElement(struct, fn) {
  const { StructureElement, StructureProperties } = S();
  const loc = StructureElement.Location.create(struct);
  for (const unit of struct.units) {
    loc.unit = unit;
    const elements = unit.elements;
    for (let i = 0; i < elements.length; i++) {
      loc.element = elements[i];
      fn(StructureProperties.residue.auth_seq_id(loc), unit, elements[i]);
    }
  }
}

/** Call `fn(resi)` once per distinct residue number. */
function eachResidue(struct, fn) {
  let last = null;
  eachElement(struct, (resi) => {
    if (resi !== last) { last = resi; fn(resi); }
  });
}

/** A sub-Structure of every atom whose residue passes `keep`, or null if none do. */
function subStructureOf(struct, keep) {
  const builder = struct.subsetBuilder(true);
  let n = 0;
  let resi = null;
  let wanted = false;
  let unitId = null;
  eachElement(struct, (r, unit, element) => {
    if (unit.id !== unitId) {
      if (unitId !== null) builder.commitUnit();
      builder.beginUnit(unit.id);
      unitId = unit.id;
      resi = null;
    }
    // keep() is asked once per residue, not once per atom.
    if (r !== resi) { resi = r; wanted = keep(r); }
    if (wanted) { builder.addElement(element); n++; }
  });
  if (unitId !== null) builder.commitUnit();
  return n ? builder.getStructure() : null;
}

/** A Bundle of every atom whose residue passes `keep`, or null if none do. */
function bundleOf(struct, keep) {
  const sub = subStructureOf(struct, keep);
  return sub ? S().StructureElement.Bundle.fromSubStructure(struct, sub) : null;
}

/** A Loci for a single residue number, or null if it is not in the structure. */
function lociOf(resi) {
  const struct = structure?.data;
  if (!struct) return null;
  const sub = subStructureOf(struct, (r) => r === resi);
  return sub ? S().Structure.toSubStructureElementLoci(struct, sub) : null;
}

/** The residue number under a hover/click Loci, or null for empty or non-structure loci. */
function residueOf(loci) {
  const { StructureElement, StructureProperties } = S();
  if (!loci || !StructureElement.Loci.is(loci) || loci.elements.length === 0) return null;
  const location = StructureElement.Loci.getFirstLocation(loci);
  return location ? StructureProperties.residue.auth_seq_id(location) : null;
}

/* ------------------------------------------------------------ colour theme */

/**
 * A colour theme that reads the map paint() just built.
 *
 * Granularity is `group`, so Mol* asks once per residue rather than once per
 * atom, and a residue with no prediction gets the same out-of-ramp grey the
 * sequence track and the legend use.
 */
function scoreThemeProvider() {
  const { StructureElement, Bond, StructureProperties } = S();

  function factory(ctx, props) {
    const location = StructureElement.Location.create(ctx.structure?.root);
    const at = (unit, element) => {
      location.unit = unit;
      location.element = element;
      const resi = StructureProperties.residue.auth_seq_id(location);
      const c = colorByResi.get(resi);
      return c === undefined ? unpredictedColor : c;
    };
    return {
      factory,
      granularity: "group",
      preferSmoothing: true,
      color: (loc) => {
        if (StructureElement.Location.is(loc)) return at(loc.unit, loc.element);
        if (Bond.isLocation(loc)) return at(loc.aUnit, loc.aUnit.elements[loc.aIndex]);
        return unpredictedColor;
      },
      props,
      description: "Predicted binding probability for the displayed task and label.",
    };
  }

  return {
    name: SCORE_THEME,
    label: "JBBind score",
    category: "Miscellaneous",
    factory,
    getParams: () => ({}),
    defaultValues: {},
    isApplicable: (ctx) => !!ctx.structure,
  };
}
