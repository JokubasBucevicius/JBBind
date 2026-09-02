#!/usr/bin/env python3
"""JBBind — per-residue binding-site prediction from a structure.

A single-command front end in the shape the published tools use: give it a PDB chain,
get scores and pictures.

    python predict_bindingsites.py 1ycr_A
    python predict_bindingsites.py 1ycr --chain A --setup nucleic
    python predict_bindingsites.py 6lu7 --all-chains --setup all
    python predict_bindingsites.py my_model.pdb --chain A
    python predict_bindingsites.py --list targets.txt

Each chain gets its own folder under --out (default ``predictions/``):

    predictions/1ycr_A/
        predictions_1ycr_A.csv                 every residue, every requested label
        annotated_1ycr_A_protein_Protein.pdb   score in the B-factor column
        1ycr_A_protein_Protein.png             the figure
        1ycr_A_protein_Protein.pml             PyMOL session script
        1ycr_A_protein_Protein.cxc             ChimeraX session script

This is a thin wrapper. Every number it prints comes from ``jbbind.core.pipeline``, the
same code path the web app and the test suite exercise — the point of the script is the
interface, not a second implementation.

Only the ``gnn_mlp`` architecture is served here. It is the default checkpoint set and the
one the benchmarks were run with; ``--arch`` exists but is deliberately undocumented in
the examples above.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Headless by default: this runs on compute nodes with no display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from jbbind.core.artifacts import UNPREDICTED_B, predictions_csv, predictions_pdb
from jbbind.core.nn.setups import SETUPS
from jbbind.settings import Settings, UserSettings, UserSettingsStore

# --------------------------------------------------------------------------- colour
# The seven stops and the out-of-ramp grey are lifted verbatim from static/style.css
# (--score-0..6, --unpredicted) so a figure and the web viewer colour a residue the same.
# The stylesheet's ramp is interpolated in OKLab by static/app.js; _oklab_ramp below
# reproduces that rather than letting matplotlib interpolate in sRGB, which would bend the
# midtones and make equal score steps read as unequal colour steps.
SCORE_STOPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
               "#2a78d6", "#1c5cab", "#0d366b"]
UNPREDICTED_COLOR = "#b8b7b2"


def _srgb_to_oklab(hexcolor: str) -> np.ndarray:
    v = hexcolor.lstrip("#")
    rgb = np.array([int(v[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def _oklab_to_srgb(lab: np.ndarray) -> tuple:
    L, a, b_ = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b_) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b_) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b_) ** 3
    lin = np.array([+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)
    return tuple(np.clip(srgb, 0.0, 1.0))


def _oklab_ramp(stops: list[str], n: int = 256) -> LinearSegmentedColormap:
    labs = [_srgb_to_oklab(s) for s in stops]
    out = []
    for t in np.linspace(0.0, 1.0, n):
        x = t * (len(labs) - 1)
        i = min(len(labs) - 2, int(np.floor(x)))
        f = x - i
        out.append(_oklab_to_srgb(labs[i] + (labs[i + 1] - labs[i]) * f))
    return LinearSegmentedColormap.from_list("jbbind", out, N=n)


CMAP = _oklab_ramp(SCORE_STOPS)


# --------------------------------------------------------------------------- targets

def parse_target(spec: str, chain_arg: str | None) -> tuple[str, str | None]:
    """``1ycr_A`` / ``1ycr`` / ``path/to/file.pdb`` -> (target, chain or None).

    An explicit --chain always wins, so ``--list`` files and --chain can be mixed without
    the file silently overriding the flag.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty target")
    if Path(spec).exists():
        return spec, chain_arg
    m = re.fullmatch(r"([0-9a-zA-Z]{4})[_:.]([A-Za-z0-9]{1,4})", spec)
    if m:
        return m.group(1).lower(), chain_arg or m.group(2)
    if re.fullmatch(r"[0-9a-zA-Z]{4}", spec):
        return spec.lower(), chain_arg
    raise ValueError(
        f"cannot read {spec!r} as a PDB ID, a <pdb>_<chain> pair, or an existing file")


def read_target_list(path: Path, chain_arg: str | None) -> list[tuple[str, str | None]]:
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "," in line:
            pdb, _, chain = line.partition(",")
            out.append(parse_target(pdb.strip(), chain.strip() or chain_arg))
        else:
            out.append(parse_target(line, chain_arg))
    return out


# --------------------------------------------------------------------------- plotting

def ca_coordinates(receptor_pdb: str) -> dict[int, np.ndarray]:
    """seqres index -> CA coordinate, read from the atoms the model actually saw.

    receptor.pdb is voronota's tessellated selection and its resSeq is already the SEQRES
    index (see core/structure/normalize.py), so this needs no remapping.
    """
    coords: dict[int, np.ndarray] = {}
    for line in receptor_pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA":
            continue
        try:
            resi = int(line[22:26])
            coords[resi] = np.array([float(line[30:38]), float(line[38:46]),
                                     float(line[46:54])])
        except ValueError:
            continue
    return coords


def _draw_structure(ax, xyz, order, scores, title, azim):
    """CA trace in 3D, residues coloured by score.

    Unscored residues (buried, or past ESM's 1022-token limit) are drawn in the
    out-of-ramp grey rather than omitted: a hole in the trace would read as a gap in the
    chain, and a pale blue dot would read as a confident negative.
    """
    known = np.array([s is not None for s in scores])
    vals = np.array([0.0 if s is None else s for s in scores], dtype=float)

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#767570", lw=0.8, alpha=0.55,
            zorder=1)
    if (~known).any():
        ax.scatter(*xyz[~known].T, s=14, c=UNPREDICTED_COLOR, depthshade=False,
                   linewidths=0, zorder=2)
    if known.any():
        # Draw ascending so the confident residues land on top of the pale ones.
        idx = np.argsort(vals[known])
        pts = xyz[known][idx]
        ax.scatter(*pts.T, s=30, c=vals[known][idx], cmap=CMAP, vmin=0, vmax=1,
                   depthshade=False, linewidths=0.3, edgecolors="#ffffff", zorder=3)

    ax.set_title(title, fontsize=8, color="#55554f", pad=0)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=azim)
    # Equal aspect: an anisotropic box distorts the fold into something misleading.
    span = (xyz.max(0) - xyz.min(0)).max() / 2.0
    mid = (xyz.max(0) + xyz.min(0)) / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    # zoom= is what actually fills the axes; a cube aspect alone leaves mpl3d's default
    # margin, which shrinks the structure to a third of the panel.
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.5)
    except TypeError:            # matplotlib < 3.6 has no zoom=
        ax.set_box_aspect((1, 1, 1))


def make_figure(result, setup: str, label_index: int, threshold: float,
                out_png: Path, top_n: int = 15) -> None:
    label = result.label_names[setup][label_index]
    residues = result.residues
    scored = {r.seqres_index: r.probs[setup][label_index] for r in residues}

    coords = ca_coordinates(result.receptor_pdb)
    order = sorted(coords)
    xyz = np.array([coords[i] for i in order]) if order else np.zeros((0, 3))
    scores3d = [scored.get(i) for i in order]

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1.0],
                  width_ratios=[1.0, 1.0, 1.05],
                  hspace=0.26, wspace=0.20,
                  left=0.045, right=0.975, top=0.885, bottom=0.085)

    hits = sorted((r for r in residues if r.probs[setup][label_index] >= threshold),
                  key=lambda r: -r.probs[setup][label_index])
    fig.suptitle(f"JBBind · {result.source} chain {result.chain_id} · "
                 f"{setup}:{label}", fontsize=13, fontweight="bold", y=0.968)
    fig.text(0.5, 0.925,
             f"{result.arch} · {len(residues)} residues scored, "
             f"{len(result.unpredicted)} not predicted · "
             f"{len(hits)} at or above {threshold:g}"
             f"   —   scores rank residues; they are not calibrated probabilities",
             ha="center", fontsize=8.5, color="#55554f")

    if len(xyz):
        for col, azim in ((0, -60), (1, 30)):
            ax = fig.add_subplot(gs[0, col], projection="3d")
            _draw_structure(ax, xyz, order, scores3d,
                            f"view {1 if col == 0 else 2}", azim)

    # -- top residues ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.set_axis_off()
    ranked = sorted(residues, key=lambda r: -r.probs[setup][label_index])[:top_n]
    ax.set_title(f"highest-scoring residues", fontsize=9.5, loc="left",
                 color="#1a1a17", pad=8)
    rows = [[f"{r.one_letter}{r.auth_seq_id}{r.auth_icode}".strip(),
             f"{r.probs[setup][label_index]:.3f}",
             "—" if r.sas_area is None else f"{r.sas_area:.0f}"] for r in ranked]
    if rows:
        tbl = ax.table(cellText=rows, colLabels=["residue", "score", "SASA"],
                       cellLoc="center", loc="upper center",
                       colWidths=[0.36, 0.32, 0.32])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.16)
        for (row, _col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#e6e5e0")
            if row == 0:
                cell.set_text_props(fontweight="bold", color="#55554f")
                cell.set_facecolor("#f7f6f3")
            else:
                p = ranked[row - 1].probs[setup][label_index]
                cell.set_facecolor(CMAP(p) if p >= threshold else "#ffffff")
                if p >= threshold and p > 0.62:
                    cell.set_text_props(color="#ffffff")

    # -- score along the chain -------------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    idx = np.array([r.seqres_index for r in residues])
    val = np.array([r.probs[setup][label_index] for r in residues])
    ax.bar(idx, val, width=1.0, color=[CMAP(v) for v in val], linewidth=0)
    ax.axhline(threshold, color="#eb6834", lw=1.0, ls="--", zorder=4)
    ax.text(ax.get_xlim()[1], threshold, f" {threshold:g} ", va="center", ha="left",
            fontsize=7.5, color="#eb6834")
    if result.unpredicted:
        # A rug, so "not predicted" is visible instead of being an invisible gap.
        ax.scatter([u["seqres_index"] for u in result.unpredicted],
                   np.full(len(result.unpredicted), -0.035),
                   marker="|", s=18, color=UNPREDICTED_COLOR, linewidths=0.9,
                   clip_on=False)
    # Neighbouring maxima are often 1-2 residues apart, so alternate the offset rather
    # than letting the labels overprint each other.
    placed: list[int] = []
    levels = 0
    for r in ranked[:8]:
        near = sum(abs(r.seqres_index - q) < 6 for q in placed)
        levels = max(levels, near)
        placed.append(r.seqres_index)
        ax.annotate(f"{r.one_letter}{r.auth_seq_id}".strip(),
                    (r.seqres_index, r.probs[setup][label_index]),
                    textcoords="offset points", xytext=(0, 4 + 9 * near), ha="center",
                    fontsize=7, color="#1c5cab")
    ax.set_xlabel("residue (SEQRES index)", fontsize=9)
    ax.set_ylabel(f"{label} score", fontsize=9)
    # Headroom for however many levels the labels stacked to, without moving the ticks:
    # a peak at 0.93 with three labels above it would otherwise run into the panel title.
    ax.set_ylim(0, 1.0 + 0.10 * levels)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("score along the chain   (grey ticks below the axis: not predicted)",
                 fontsize=9.5, loc="left", color="#1a1a17", pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # -- ranked curve ----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    srt = np.sort(val)[::-1] if len(val) else np.zeros(0)
    ax.plot(np.arange(1, len(srt) + 1), srt, color="#2a78d6", lw=1.6)
    ax.fill_between(np.arange(1, len(srt) + 1), srt, color="#2a78d6", alpha=0.13)
    ax.axhline(threshold, color="#eb6834", lw=1.0, ls="--")
    if len(hits):
        ax.axvline(len(hits), color="#eb6834", lw=1.0, ls=":")
        ax.text(len(hits), 1.0, f" {len(hits)}", fontsize=7.5, color="#eb6834",
                va="top", ha="left")
    ax.set_xlabel("residues, ranked by score", fontsize=9)
    ax.set_ylabel("score", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("ranked scores", fontsize=9.5, loc="left", color="#1a1a17", pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    cax = fig.add_axes([0.20, 0.468, 0.24, 0.015])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                 cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=7.5, length=2)
    cax.set_title(f"{label} score", fontsize=7.5, color="#55554f", pad=3)

    fig.savefig(out_png, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- viewers

def pymol_script(name: str, pdb_file: str, script: str, label: str,
                 threshold: float) -> str:
    """A session, not just a selection string: load, colour by score, show the hits.

    B = -1 is the not-predicted sentinel, so it is excluded from the ramp explicitly and
    left grey. Spectrum over the full B range would otherwise stretch the ramp down to -1
    and push every real score into the top half of the scale.

    The ramp stops are defined with set_color rather than passed to `spectrum` as hex:
    spectrum's palette argument takes colour *names*, so hex literals there are silently
    misread.
    """
    stops = " ".join(f"jb{i}" for i in range(len(SCORE_STOPS)))
    defs = "\n".join(
        "set_color jb%d, [%.4f, %.4f, %.4f]"
        % ((i,) + tuple(int(h.lstrip("#")[j:j + 2], 16) / 255.0 for j in (0, 2, 4)))
        for i, h in enumerate(SCORE_STOPS))
    return f"""\
# JBBind — {name} · {label}
#   pymol {script}
{defs}

load {pdb_file}, {name}
hide everything
show cartoon, {name}
set cartoon_transparency, 0.15

select unpredicted, {name} and b < -0.5
select scored, {name} and not unpredicted
color grey70, unpredicted
spectrum b, {stops}, scored, minimum=0, maximum=100

select bindingsite, scored and b >= {threshold * 100:g}
show sticks, bindingsite and not (name C+N+O)
color orange, bindingsite and elem C
deselect

bg_color white
set ray_opaque_background, 0
orient {name}
# ray 1600, 1200
# png {name}_{slug(label)}.png, dpi=300
"""


def chimerax_script(name: str, pdb_file: str, script: str, label: str,
                    threshold: float) -> str:
    return f"""\
# JBBind — {name} · {label}
# chimerax {script}
open {pdb_file}
hide atoms
show cartoon
color grey(70%)

# B = score x 100; B = -1 marks a residue with no prediction and keeps the grey above.
color byattribute bfactor #1 & @@bfactor>=0 palette 0,#cde2fb:33,#6da7ec:66,#2a78d6:100,#0d366b
select #1 & @@bfactor>={threshold * 100:g}
show sel atoms
color sel orange atoms
~select

set bgColor white
view
# save {name}_{slug(label)}.png width 1600 supersample 3
"""


# --------------------------------------------------------------------------- driver

def build(cfg: Settings, arch: str, threshold: float):
    from jbbind.core.cache import CacheSet
    from jbbind.core.esm.embedder import EsmEmbedder
    from jbbind.core.nn.registry import ModelRegistry
    from jbbind.core.pipeline import Pipeline

    caches = CacheSet(cfg.cache_dir, esm_max_bytes=cfg.esm_cache_bytes,
                      chain_max_bytes=cfg.chain_cache_bytes)
    stored = UserSettingsStore(cfg.cache_dir / "settings.json").get()
    d = asdict(stored)
    d.update(arch=arch, threshold=threshold)
    user = UserSettings(**d)
    registry = ModelRegistry(cfg.models_dir, cfg.device)
    embedder = EsmEmbedder(cfg.device, cache=caches.esm,
                           long_seq_mode=user.esm_long_seq_mode)
    return Pipeline(cfg, registry, embedder, caches), user


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def run_chain(pipeline, user, raw, sid, source, chain_id, name, setups, out_root,
              threshold, no_figures, verbose) -> dict:
    t0 = time.perf_counter()
    result = pipeline.predict(
        raw=raw, structure_id=sid, source=source, chain_id=chain_id, user=user,
        setups=setups,
        progress=(lambda s, m: print(f"    [{s}] {m}", file=sys.stderr))
        if verbose else None)
    elapsed = time.perf_counter() - t0

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / f"predictions_{name}.csv"]
    written[0].write_text(predictions_csv(result))

    for setup in setups:
        for i, label in enumerate(result.label_names[setup]):
            tag = f"{slug(setup)}_{slug(label)}"
            pdb_path = out_dir / f"annotated_{name}_{tag}.pdb"
            pdb_path.write_text(predictions_pdb(result, setup, i))
            (out_dir / f"{name}_{tag}.pml").write_text(
                pymol_script(name, pdb_path.name, f"{name}_{tag}.pml", label, threshold))
            (out_dir / f"{name}_{tag}.cxc").write_text(
                chimerax_script(name, pdb_path.name, f"{name}_{tag}.cxc", label,
                                threshold))
            written += [pdb_path, out_dir / f"{name}_{tag}.pml",
                        out_dir / f"{name}_{tag}.cxc"]
            if not no_figures:
                png = out_dir / f"{name}_{tag}.png"
                make_figure(result, setup, i, threshold, png)
                written.append(png)

    print(f"  {name}: {result.n_predicted} residues scored, "
          f"{len(result.unpredicted)} not predicted, {elapsed:.1f}s -> {out_dir}/")
    for setup in setups:
        for i, label in enumerate(result.label_names[setup]):
            vals = [r.probs[setup][i] for r in result.residues]
            n_hit = sum(v >= threshold for v in vals)
            top = max(vals) if vals else float("nan")
            print(f"      {f'{setup}:{label}':<28} {n_hit:>4} at or above "
                  f"{threshold:g}   max {top:.3f}")
    for w in result.warnings:
        print(f"      warning [{w['code']}] {w['detail']}", file=sys.stderr)
    return {"name": name, "files": written, "result": result}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="predict_bindingsites.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="*",
                   help="PDB ID, <pdb>_<chain>, or a path to a PDB/mmCIF file")
    p.add_argument("--list", dest="list_file",
                   help="file of targets, one `pdb_id[,chain]` per line")
    p.add_argument("--chain", help="chain to predict (default: the first protein chain)")
    p.add_argument("--all-chains", action="store_true",
                   help="predict every protein chain in the entry")
    p.add_argument("--setup", default="protein",
                   help="label setup: " + ", ".join(SETUPS) + ", or 'all' "
                        "(default: protein)")
    p.add_argument("--arch", default="gnn_mlp", help=argparse.SUPPRESS)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="decision threshold for the highlighted sites (default 0.5)")
    p.add_argument("--out", default="predictions", help="output directory")
    p.add_argument("--assembly", type=int, default=None,
                   help="fetch this biological assembly instead of the asymmetric unit")
    p.add_argument("--no-figures", action="store_true", help="skip the PNGs")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.setup == "all":
        setups = list(SETUPS)
    elif args.setup in SETUPS:
        setups = [args.setup]
    else:
        p.error(f"unknown setup {args.setup!r}; choose from {', '.join(SETUPS)}, all")

    targets: list[tuple[str, str | None]] = []
    try:
        for t in args.targets:
            targets.append(parse_target(t, args.chain))
        if args.list_file:
            targets += read_target_list(Path(args.list_file), args.chain)
    except ValueError as exc:
        p.error(str(exc))
    if not targets:
        p.error("give at least one target, or --list a file of them")

    cfg = Settings()
    pipeline, user = build(cfg, args.arch, args.threshold)
    if args.assembly is not None:
        user.rcsb_assembly = args.assembly
    out_root = Path(args.out)

    print(f"JBBind · {user.arch} · setups: {', '.join(setups)} · device {cfg.device}")
    print(f"{len(targets)} target(s) -> {out_root}/")

    failed = 0
    for spec, chain in targets:
        try:
            if Path(spec).exists():
                raw, sid, _ = pipeline.load_structure(data=Path(spec).read_bytes())
                source = f"file {Path(spec).name}"
                stem = Path(spec).stem
            else:
                raw, sid, source = pipeline.load_structure(
                    pdb_id=spec, assembly=user.rcsb_assembly)
                stem = spec.lower()

            chains, _ = pipeline.describe_structure(raw)
            if not chains:
                raise RuntimeError(f"{spec}: no protein chain to predict on")
            if chain:
                wanted = [chain]
            elif args.all_chains:
                wanted = [c.chain_id for c in chains]
            else:
                wanted = [chains[0].chain_id]

            for chain_id in wanted:
                run_chain(pipeline, user, raw, sid, source, chain_id,
                          f"{stem}_{chain_id}", setups, out_root, args.threshold,
                          args.no_figures, args.verbose)
        except Exception as exc:
            failed += 1
            code = getattr(exc, "code", None)
            label = f"[{code}] " if code else ""
            print(f"  {spec}{'_' + chain if chain else ''}: FAILED {label}"
                  f"{getattr(exc, 'message', exc)}", file=sys.stderr)
            hint = HINTS.get(code)
            if hint:
                print(f"      {hint}", file=sys.stderr)
            if code is None and args.verbose:
                raise

    if failed:
        print(f"\n{failed} target(s) failed", file=sys.stderr)
    return 1 if failed else 0


HINTS = {
    "VoronotaMissing":
        "Put voronota-js on PATH: export PATH=\"$PATH:/path/to/voronota/expansion_js\"",
    "PdbNotFound": "Check the ID at https://www.rcsb.org, or pass a local file.",
    "ChainNotFound":
        "Run the same target with --all-chains to see which chains exist. PDB entries and "
        "derived datasets often disagree about the chain letter.",
    "NoPolymerChains": "This entry has no protein chain long enough to predict on.",
    "SequenceMappingFailed":
        "The observed residues could not be aligned to SEQRES, so the ESM alignment would "
        "be a guess.",
    "NoSurfaceResidues": "No solvent-accessible residue in this chain.",
    "TooManyResidues": "Raise JBBIND_MAX_RESIDUES to run this chain anyway.",
    "RcsbUnavailable": "RCSB could not be reached; pass a local file instead.",
}


if __name__ == "__main__":
    raise SystemExit(main())
