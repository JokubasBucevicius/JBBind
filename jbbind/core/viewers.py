"""Viewer session scripts: PyMOL and ChimeraX.

Not just a selection string but a session that loads the annotated PDB, colours
it by the score in the B-factor column and shows the residues above the
threshold, so the file opens into something already worth looking at.
"""

from __future__ import annotations

from .artifacts import UNPREDICTED_B, slug
from .colour import SCORE_STOPS

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


