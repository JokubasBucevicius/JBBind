"""Tests for the predict_bindingsites.py front end.

Only the pure parts: target parsing, the colour ramp and the viewer scripts. The
prediction path itself is already covered by test_parity_*.py — the script is a wrapper
over it and must not grow logic of its own that needs separate coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict_bindingsites as pb


class TestParseTarget:
    @pytest.mark.parametrize("spec,pdb,chain", [
        ("1ycr_A", "1ycr", "A"),
        ("1YCR_A", "1ycr", "A"),
        ("1ycr:A", "1ycr", "A"),
        ("1ycr.A", "1ycr", "A"),
        ("8cb2_AAA", "8cb2", "AAA"),   # multi-character chain ids exist
        ("1ycr", "1ycr", None),
        ("6LU7", "6lu7", None),
    ])
    def test_forms(self, spec, pdb, chain):
        assert pb.parse_target(spec, None) == (pdb, chain)

    def test_explicit_chain_wins(self):
        """--chain must override the suffix, or a --list run silently ignores the flag."""
        assert pb.parse_target("1ycr_A", "B") == ("1ycr", "B")

    def test_existing_path_is_a_file(self, tmp_path):
        f = tmp_path / "model.pdb"
        f.write_text("ATOM\n")
        assert pb.parse_target(str(f), "A") == (str(f), "A")

    @pytest.mark.parametrize("spec", ["", "  ", "notapdbid", "1yc", "1ycr_"])
    def test_rejects_junk(self, spec):
        with pytest.raises(ValueError):
            pb.parse_target(spec, None)


class TestTargetList:
    def test_forms_and_comments(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("1ycr,B\n3hdd_A\n# comment\n\n6lu7   # trailing\n")
        assert pb.read_target_list(f, None) == [
            ("1ycr", "B"), ("3hdd", "A"), ("6lu7", None)]

    def test_bare_comma_falls_back_to_flag(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("1ycr,\n")
        assert pb.read_target_list(f, "C") == [("1ycr", "C")]


class TestColourRamp:
    def test_endpoints_match_the_stylesheet(self):
        """The CLI figures and the web viewer must agree on what a score looks like."""
        for hexcolor, t in ((pb.SCORE_STOPS[0], 0.0), (pb.SCORE_STOPS[-1], 1.0)):
            want = np.array([int(hexcolor.lstrip("#")[i:i + 2], 16) / 255.0
                             for i in (0, 2, 4)])
            got = np.array(pb.CMAP(t)[:3])
            assert np.allclose(got, want, atol=2 / 255)

    def test_oklab_roundtrip(self):
        for stop in pb.SCORE_STOPS:
            want = np.array([int(stop.lstrip("#")[i:i + 2], 16) / 255.0
                             for i in (0, 2, 4)])
            got = np.array(pb._oklab_to_srgb(pb._srgb_to_oklab(stop)))
            assert np.allclose(got, want, atol=1e-6)

    def test_monotone_lightness(self):
        """A sequential ramp that brightens anywhere would misread as non-monotone score."""
        lum = [0.2126 * r + 0.7152 * g + 0.0722 * b
               for r, g, b, _ in (pb.CMAP(t) for t in np.linspace(0, 1, 64))]
        assert all(a >= b - 1e-9 for a, b in zip(lum, lum[1:]))


class TestSlug:
    @pytest.mark.parametrize("text,want", [
        ("Nucleic acid", "Nucleic_acid"), ("DNA", "DNA"), ("protein_nucleic",
                                                           "protein_nucleic")])
    def test_filename_safe(self, text, want):
        assert pb.slug(text) == want


class TestViewerScripts:
    def test_pymol_excludes_the_sentinel_from_the_ramp(self):
        s = pb.pymol_script("1ycr_A", "annotated.pdb", "1ycr_A.pml", "Protein", 0.5)
        assert "b < -0.5" in s and "color grey70, unpredicted" in s
        # spectrum must be pinned to 0-100, never left to infer from the data range.
        assert "minimum=0, maximum=100" in s
        # palette entries must be defined names, not hex literals
        assert "set_color jb0" in s and "spectrum b, jb0" in s
        assert "#" not in s.split("spectrum b,")[1].split("\n")[0]

    def test_chimerax_uses_the_atom_attribute_selector(self):
        s = pb.chimerax_script("1ycr_A", "annotated.pdb", "1ycr_A.cxc", "Protein", 0.5)
        assert "@@bfactor>=0" in s          # atom attribute, not the residue-level ::
        assert "@@bfactor>=50" in s
        assert "::bfactor" not in s

    def test_threshold_reaches_both_scripts(self):
        assert "b >= 30" in pb.pymol_script("n", "a.pdb", "s.pml", "Protein", 0.3)
        assert "@@bfactor>=30" in pb.chimerax_script("n", "a.pdb", "s.cxc", "Protein", 0.3)


class TestCoordinates:
    def test_reads_ca_only_and_keeps_seqres_numbering(self):
        pdb = (
            "ATOM      1  N   ARG A   5      28.897  13.608  40.938  1.00 93.06           N\n"
            "ATOM      2  CA  ARG A   5      28.659  14.989  40.521  1.00 93.06           C\n"
            "ATOM      3  CA  THR A   6       1.000   2.000   3.000  1.00 90.61           C\n"
            "TER\n")
        coords = pb.ca_coordinates(pdb)
        assert sorted(coords) == [5, 6]
        assert np.allclose(coords[6], [1.0, 2.0, 3.0])
