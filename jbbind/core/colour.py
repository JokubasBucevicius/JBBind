"""The score colour ramp, shared by every rendering of a prediction.

Seven stops and an out-of-ramp grey, lifted verbatim from ``static/style.css``
(``--score-0..6``, ``--unpredicted``), so the figure, the HTML report, the viewer
scripts and the web viewer all colour a residue identically. The stylesheet's
ramp is interpolated in OKLab by ``static/app.js``; the lookup table below
reproduces that rather than letting matplotlib interpolate in sRGB, which would
bend the midtones and make equal score steps read as unequal colour steps.

The table is the single source of truth. ``score_hex`` reads it directly and
``cmap`` hands the very same rows to matplotlib, so a residue cannot come out
one colour in the PNG and another in the report. That also keeps matplotlib out
of the import path: only the figure needs it, and it is imported when asked for.
"""

from __future__ import annotations

import numpy as np

SCORE_STOPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
               "#2a78d6", "#1c5cab", "#0d366b"]
UNPREDICTED_COLOR = "#b8b7b2"

RAMP_SIZE = 256


def srgb_to_oklab(hexcolor: str) -> np.ndarray:
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


def oklab_to_srgb(lab: np.ndarray) -> tuple:
    L, a, b_ = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b_) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b_) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b_) ** 3
    lin = np.array([+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)
    return tuple(np.clip(srgb, 0.0, 1.0))


def _build_lut(stops: list[str], n: int = RAMP_SIZE) -> np.ndarray:
    labs = [srgb_to_oklab(s) for s in stops]
    out = np.empty((n, 3), dtype=float)
    for k, t in enumerate(np.linspace(0.0, 1.0, n)):
        x = t * (len(labs) - 1)
        i = min(len(labs) - 2, int(np.floor(x)))
        f = x - i
        out[k] = oklab_to_srgb(labs[i] + (labs[i + 1] - labs[i]) * f)
    return out


RAMP_LUT = _build_lut(SCORE_STOPS)


def ramp_index(p: float, n: int = RAMP_SIZE) -> int:
    """Score in [0,1] -> row of RAMP_LUT, by matplotlib's own index arithmetic.

    Colormap.__call__ does ``int(x * N)`` and folds the ``x == 1`` case back to
    ``N - 1``. Reproducing that exactly is what makes score_hex and the figure
    agree on the last bucket instead of differing by one row at the top of the
    ramp.
    """
    x = float(np.clip(p, 0.0, 1.0))
    return min(int(x * n), n - 1)


def score_hex(p: float) -> str:
    """A score's colour as ``#rrggbb``, from the same table the figure uses."""
    r, g, b = RAMP_LUT[ramp_index(p)]
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in (r, g, b)))


def cmap():
    """The ramp as a matplotlib colormap. Imports matplotlib; the figure only."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(RAMP_LUT, name="jbbind")
