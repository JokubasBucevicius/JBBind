"""The shared score ramp.

The figure, the HTML report, the viewer scripts and the web viewer all colour a
residue through this module, so these are the tests that keep them agreeing."""

import re

import numpy as np
import pytest

from jbbind.core import colour

CMAP = colour.cmap()


class TestColourRamp:
    def test_endpoints_match_the_stylesheet(self):
        """The CLI figures and the web viewer must agree on what a score looks like."""
        for hexcolor, t in ((colour.SCORE_STOPS[0], 0.0), (colour.SCORE_STOPS[-1], 1.0)):
            want = np.array([int(hexcolor.lstrip("#")[i:i + 2], 16) / 255.0
                             for i in (0, 2, 4)])
            got = np.array(CMAP(t)[:3])
            assert np.allclose(got, want, atol=2 / 255)

    def test_oklab_roundtrip(self):
        for stop in colour.SCORE_STOPS:
            want = np.array([int(stop.lstrip("#")[i:i + 2], 16) / 255.0
                             for i in (0, 2, 4)])
            got = np.array(colour.oklab_to_srgb(colour.srgb_to_oklab(stop)))
            assert np.allclose(got, want, atol=1e-6)

    def test_monotone_lightness(self):
        """A sequential ramp that brightens anywhere would misread as non-monotone score."""
        lum = [0.2126 * r + 0.7152 * g + 0.0722 * b
               for r, g, b, _ in (CMAP(t) for t in np.linspace(0, 1, 64))]
        assert all(a >= b - 1e-9 for a, b in zip(lum, lum[1:]))


class TestScoreHex:
    def test_endpoints_are_the_ramp_stops(self):
        assert colour.score_hex(0.0).lower() == colour.SCORE_STOPS[0].lower()
        assert colour.score_hex(1.0).lower() == colour.SCORE_STOPS[-1].lower()

    def test_clamps_out_of_range(self):
        assert colour.score_hex(-3.0) == colour.score_hex(0.0)
        assert colour.score_hex(9.9) == colour.score_hex(1.0)

    def test_always_six_digit_hex(self):
        """viewer.js parses these with parseInt(hex, 16); a short form would silently shift."""
        for t in np.linspace(0, 1, 33):
            assert re.fullmatch(r"#[0-9a-f]{6}", colour.score_hex(t))
