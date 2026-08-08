#!/usr/bin/env python3
"""Python twin of the dataviz skill's validate_palette.js (no node on this host).

Ports the same math and the same thresholds: OKLCH lightness band, chroma floor,
Machado-Oliveira-Fernandes (2009) severity-1.0 CVD simulation, OKLab dE x100
separation, and WCAG contrast against the chart surface.

    python scripts/validate_palette.py "#2a78d6,#eb6834" --mode light --pairs all
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}

WS = " \t\n\v\f\r      　" + \
     "".join(chr(c) for c in range(0x2000, 0x200b))


def is_hex(v: str) -> bool:
    return bool(re.fullmatch(r"#?[0-9a-fA-F]{6}", v))


def split_colors(raw: str) -> list[str]:
    return [c for c in (p.strip(WS) for p in (raw or "").split(",")) if c]


def hex2srgb(h: str) -> list[float]:
    h = h.strip(WS).lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h: str) -> list[float]:
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h: str) -> float:
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb: list[float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h: str) -> tuple[float, float]:
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h: str, kind: str) -> list[float]:
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette: list[str], mode: str = "light", surface: str | None = None,
             pairs: str = "adjacent") -> tuple[bool, list[tuple[str, str, str]]]:
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    report: list[tuple[str, str, str]] = []
    ok = True

    offband = [(c, round(oklch(c)[0], 3)) for c in palette
               if not (lo <= oklch(c)[0] <= hi)]
    ok &= not offband
    report.append(("Lightness band", "PASS" if not offband else "FAIL",
                   f"outside L {lo}-{hi}: {offband}" if offband
                   else f"all {len(palette)} inside L {lo}-{hi}"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    report.append(("Chroma floor", "PASS" if not lowc else "FAIL",
                   f"reads gray: {lowc}" if lowc else f"all >= {CHROMA_FLOOR}"))

    n = len(palette)
    pairlist = (list(itertools.combinations(range(n), 2)) if pairs == "all"
                else [(i, i + 1) for i in range(n - 1)])

    worst_cvd, worst_pair = math.inf, None
    for i, j in pairlist:
        d = min(delta_e(palette[i], palette[j], "protan"),
                delta_e(palette[i], palette[j], "deutan"))
        if d < worst_cvd:
            worst_cvd, worst_pair = d, (palette[i], palette[j])
    if pairlist:
        if worst_cvd < CVD_FLOOR:
            ok = False
            status = "FAIL"
        elif worst_cvd < CVD_TARGET:
            status = "WARN"
        else:
            status = "PASS"
        report.append((f"CVD separation ({pairs})", status,
                       f"worst dE {worst_cvd:.1f} on {worst_pair} "
                       f"(target >= {CVD_TARGET}, floor {CVD_FLOOR})"))

        worst_norm, norm_pair = math.inf, None
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j])
            if d < worst_norm:
                worst_norm, norm_pair = d, (palette[i], palette[j])
        nok = worst_norm >= NORMAL_FLOOR
        ok &= nok
        report.append((f"Normal-vision floor ({pairs})", "PASS" if nok else "FAIL",
                       f"worst dE {worst_norm:.1f} on {norm_pair} "
                       f"(floor {NORMAL_FLOOR})"))

    low_contrast = [(c, round(contrast(c, surface), 2)) for c in palette
                    if contrast(c, surface) < CONTRAST_MIN]
    report.append(("Contrast vs surface", "PASS" if not low_contrast else "WARN",
                   f"below {CONTRAST_MIN}:1 (relief rule: direct labels or table): "
                   f"{low_contrast}" if low_contrast
                   else f"all >= {CONTRAST_MIN}:1 vs {surface}"))
    return ok, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    ap.add_argument("--surface", default=None)
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    args = ap.parse_args()

    palette = split_colors(args.palette)
    bad = [c for c in palette if not is_hex(c)]
    if bad or not palette:
        print(f"invalid hex values: {bad or 'empty palette'}")
        return 2
    palette = ["#" + c.lstrip("#") for c in palette]

    ok, report = validate(palette, args.mode, args.surface, args.pairs)
    width = max(len(r[0]) for r in report)
    print(f"\n{args.mode} mode, pairs={args.pairs}, "
          f"surface={args.surface or DEFAULT_SURFACE[args.mode]}")
    for name, status, detail in report:
        print(f"  {status:4}  {name:<{width}}  {detail}")
    print(f"\n  => {'OK' if ok else 'FAILED'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
