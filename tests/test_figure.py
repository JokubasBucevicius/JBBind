"""The four-panel figure.

Only the pure part: reading CA coordinates out of receptor.pdb. Rendering itself
is exercised by the CLI and the web app, not asserted pixel by pixel here."""

import numpy as np

from jbbind.core.figure import ca_coordinates


class TestCoordinates:
    def test_reads_ca_only_and_keeps_seqres_numbering(self):
        pdb = (
            "ATOM      1  N   ARG A   5      28.897  13.608  40.938  1.00 93.06           N\n"
            "ATOM      2  CA  ARG A   5      28.659  14.989  40.521  1.00 93.06           C\n"
            "ATOM      3  CA  THR A   6       1.000   2.000   3.000  1.00 90.61           C\n"
            "TER\n")
        coords = ca_coordinates(pdb)
        assert sorted(coords) == [5, 6]
        assert np.allclose(coords[6], [1.0, 2.0, 3.0])
