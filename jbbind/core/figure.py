"""The four-panel score figure.

Two 3D views of the CA trace coloured by score, the highest-scoring residues
with their SASA, the score along the chain, and the ranked score curve.

Three things it is deliberately careful about:

* **Unscored residues are drawn, in grey, outside the ramp.** Buried residues
  and anything past ESM-2's 1022-token limit have no prediction. Omitting them
  from the 3D trace would read as a break in the chain; colouring them pale blue
  would read as a confident negative.
* **The colour axis is pinned to 0-1**, not to the chain's own min and max, so
  two chains can be put side by side.
* **The ramp is the shared one** in ``colour.py``, so a residue is the same
  colour here, in the HTML report and in the web viewer.

matplotlib is imported at module scope with the Agg backend forced, so importing
this module is safe on a headless compute node -- but nothing else in the
package imports it unless a figure is actually asked for.
"""

from __future__ import annotations

import io
import threading

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import Normalize                      # noqa: E402
from matplotlib.gridspec import GridSpec                     # noqa: E402

from .artifacts import slug                                  # noqa: E402
from .colour import SCORE_STOPS, UNPREDICTED_COLOR, cmap     # noqa: E402

CMAP = cmap()

# plt.figure/plt.close touch a process-global figure manager, so two threads
# rendering at once can interleave and lose figures. The web app serves figures
# from its thread pool, so every render goes through this.
_RENDER_LOCK = threading.Lock()

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


def figure_png(result, setup: str, label_index: int, threshold: float,
               top_n: int = 15) -> bytes:
    """The figure as PNG bytes, for the web app's download endpoint."""
    buf = io.BytesIO()
    make_figure(result, setup, label_index, threshold, buf, top_n)
    return buf.getvalue()


def make_figure(result, setup: str, label_index: int, threshold: float,
                out_png, top_n: int = 15) -> None:
    """Render to a path or to any file-like object matplotlib can save into."""
    with _RENDER_LOCK:
        _render(result, setup, label_index, threshold, out_png, top_n)


def _render(result, setup, label_index, threshold, out_png, top_n) -> None:
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

    fig.savefig(out_png, dpi=170, format="png")
    plt.close(fig)


