"""PyMOL and ChimeraX session scripts, and the filename slug they share."""

import pytest

from jbbind.core.artifacts import slug
from jbbind.core.viewers import chimerax_script, pymol_script


class TestSlug:
    @pytest.mark.parametrize("text,want", [
        ("Nucleic acid", "Nucleic_acid"), ("DNA", "DNA"), ("protein_nucleic",
                                                           "protein_nucleic")])
    def test_filename_safe(self, text, want):
        assert slug(text) == want


class TestViewerScripts:
    def test_pymol_excludes_the_sentinel_from_the_ramp(self):
        s = pymol_script("1ycr_A", "annotated.pdb", "1ycr_A.pml", "Protein", 0.5)
        assert "b < -0.5" in s and "color grey70, unpredicted" in s
        # spectrum must be pinned to 0-100, never left to infer from the data range.
        assert "minimum=0, maximum=100" in s
        # palette entries must be defined names, not hex literals
        assert "set_color jb0" in s and "spectrum b, jb0" in s
        assert "#" not in s.split("spectrum b,")[1].split("\n")[0]

    def test_chimerax_uses_the_atom_attribute_selector(self):
        s = chimerax_script("1ycr_A", "annotated.pdb", "1ycr_A.cxc", "Protein", 0.5)
        assert "@@bfactor>=0" in s          # atom attribute, not the residue-level ::
        assert "@@bfactor>=50" in s
        assert "::bfactor" not in s

    def test_threshold_reaches_both_scripts(self):
        assert "b >= 30" in pymol_script("n", "a.pdb", "s.pml", "Protein", 0.3)
        assert "@@bfactor>=30" in chimerax_script("n", "a.pdb", "s.cxc", "Protein", 0.3)
