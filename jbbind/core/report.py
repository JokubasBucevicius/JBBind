"""The standalone interactive HTML report.

One file that opens in a browser: the structure in Mol* coloured by score, with
the label, threshold, colouring mode and surface live, a sequence track and a
sorted residue table beside it.

The page inlines ``static/viewer.js`` -- the very file the web app imports --
with its ``export`` keywords stripped, because browsers refuse ES modules on
``file://`` URLs. That keeps one Mol* wrapper rather than two that drift apart,
and it is why ``viewer.js`` must not grow an ``import``.

Continuous colours are resolved here, by ``colour.score_hex``, and inlined as
data, so the page interpolates nothing and the ramp keeps a single definition
across the report, the figure and the viewer scripts.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from .colour import SCORE_STOPS, UNPREDICTED_COLOR, score_hex

PACKAGE = Path(__file__).resolve().parent.parent
REPORT_TEMPLATE = PACKAGE / "report_template.html"
VIEWER_JS = PACKAGE / "static" / "viewer.js"
MOLSTAR_JS = PACKAGE / "static" / "vendor" / "molstar.js"
MOLSTAR_CSS = PACKAGE / "static" / "vendor" / "molstar.css"

_EXPORT = re.compile(r"^export\s+", re.MULTILINE)
_IMPORT = re.compile(r"^import\s+", re.MULTILINE)


def deexport(js: str) -> str:
    """``viewer.js`` as a classic script.

    The report cannot import an ES module: browsers refuse module scripts on
    ``file://`` URLs, which is exactly how this page is opened. Stripping the
    ``export`` keywords lets the report inline the very file the web app
    imports, instead of keeping a second copy of the Mol* wrapper that would
    quietly drift out of step with it.
    """
    if _IMPORT.search(js):
        raise RuntimeError("viewer.js must not import anything: the HTML report "
                           "inlines it as a classic script")
    return _EXPORT.sub("", js)


def report_data(result, setups, threshold: float, name: str) -> dict:
    """Everything the report needs, with the continuous colours precomputed.

    Colours are resolved here rather than in the page so the ramp keeps a single
    definition: the report, the PNG and the viewer scripts all inherit ``CMAP``.
    The page only has to pick between the two endpoint colours for the
    above/below-threshold mode, which needs no interpolation.
    """
    residues = []
    for r in result.residues:
        sas = float(r.sas_area) if r.sas_area is not None else float("nan")
        residues.append({
            "i": r.seqres_index,
            "ch": r.auth_chain,
            "auth": r.auth_seq_id,
            "ic": r.auth_icode or "",
            "aa": r.one_letter,
            "sas": None if np.isnan(sas) else round(sas, 2),
            "p": {s: [round(float(v), 6) for v in r.probs[s]] for s in setups},
            "c": {s: [score_hex(v) for v in r.probs[s]] for s in setups},
        })
    return {
        "name": name,
        "source": result.source,
        "chain": result.chain_id,
        "arch": result.arch,
        "setups": list(setups),
        "labels": {s: list(result.label_names[s]) for s in setups},
        "threshold": threshold,
        "sequence": result.sequence,
        "residues": residues,
        "receptorPdb": result.receptor_pdb,
        "warnings": result.warnings,
        "nPredicted": result.n_predicted,
        "nUnpredicted": len(result.unpredicted),
        "unpredicted": UNPREDICTED_COLOR,
        "rampLo": SCORE_STOPS[0],
        "rampHi": SCORE_STOPS[-1],
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }


def ensure_assets(root: Path) -> Path:
    """Mol* copied once per output root, shared by every report under it.

    Inlining the 4.8 MB bundle in each report would make a hundred-chain run a
    half-gigabyte of identical bytes. ``--standalone`` inlines it for the one
    report you want to send someone.
    """
    assets = root / "_assets"
    assets.mkdir(parents=True, exist_ok=True)
    for src in (MOLSTAR_JS, MOLSTAR_CSS):
        dst = assets / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
    return assets


def report_html(result, setups, threshold: float, name: str,
                assets: str | None = None) -> str:
    """The report as one HTML string.

    ``assets`` is a relative href to a directory holding ``molstar.js`` and
    ``molstar.css``; ``None`` inlines the 4.8 MB bundle instead, which is what a
    download from the web app wants — one file that works wherever it lands.
    """
    template = REPORT_TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(report_data(result, setups, threshold, name),
                         separators=(",", ":"))
    # "</" cannot appear literally inside a <script> block; inside JSON it is
    # only ever part of a string, where the escape is a no-op.
    payload = payload.replace("</", "<\\/")

    if assets is None:
        css = "<style>\n" + MOLSTAR_CSS.read_text(encoding="utf-8") + "\n</style>"
        js = "<script>\n" + MOLSTAR_JS.read_text(encoding="utf-8") + "\n</script>"
    else:
        css = f'<link rel="stylesheet" href="{assets}/molstar.css">'
        js = f'<script src="{assets}/molstar.js"></script>'

    fills = {
        "TITLE": f"JBBind — {name}",
        "HEADING": f"{result.source} · chain {result.chain_id}",
        "RAMP_CSS": ", ".join(SCORE_STOPS),
        "MOLSTAR_CSS": css,
        "MOLSTAR_JS": js,
        "VIEWER_JS": deexport(VIEWER_JS.read_text(encoding="utf-8")),
        "DATA": payload,
    }
    html = template
    for key, value in fills.items():
        token = "{{" + key + "}}"
        if token not in html:
            raise RuntimeError(f"report template has no {token} placeholder")
        html = html.replace(token, value)
    # Checked by name, not by looking for a leftover "{{": minified Mol* is full
    # of them, and inlining puts the whole bundle into this string.
    for key in fills:
        if "{{" + key + "}}" in html:
            raise RuntimeError(f"unsubstituted {{{{{key}}}}} left in the report")
    return html


def write_report(result, setups, threshold: float, name: str, out_dir: Path,
                 standalone: bool) -> Path:
    """Write the report into ``out_dir``, sharing Mol* under ``<root>/_assets``."""
    if standalone:
        assets = None
    else:
        assets = Path(os.path.relpath(ensure_assets(out_dir.parent),
                                      out_dir)).as_posix()
    path = out_dir / f"report_{name}.html"
    path.write_text(report_html(result, setups, threshold, name, assets),
                    encoding="utf-8")
    return path
