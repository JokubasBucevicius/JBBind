"""Modified-residue handling.

voronota assigns ``residue_type`` from its VoroMQA typing table, and the C++ does
``if(value >= 0) atom_adjuncts[name] = value;`` (set_adjunct_of_atoms_by_type_number.h:97-102)
— so a residue it does not recognise gets **no** ``residue_type`` adjunct at all. Downstream
that residue either vanishes from the ``[tessellated]`` selection or feeds garbage into
``nn.Embedding(20, 32)``. The training data confirms the invariant: ``residue_type`` is
exactly {0..19} and ``ID_resName`` is exactly the 20 standard names across every chain
sampled.

So every incoming residue must end up as one of the 20 standard names. Mapping to the parent
residue is strongly preferred over dropping: removing atoms changes the Voronoi tessellation
and therefore perturbs the *neighbouring* residues' ``sas_area`` and exposure features.
"""

from __future__ import annotations

import gemmi

STANDARD = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

#: Explicit parent mapping for the modified residues that actually show up in the PDB with
#: any frequency. Anything not here falls back to gemmi's chemical-component table.
EXPLICIT_PARENTS = {
    "MSE": "MET",  # selenomethionine — by far the most common
    "SEC": "CYS", "CSO": "CYS", "CME": "CYS", "CSD": "CYS", "OCS": "CYS",
    "CAS": "CYS", "CSX": "CYS", "SNC": "CYS", "CAF": "CYS", "YCM": "CYS",
    "SEP": "SER", "TPO": "THR", "PTR": "TYR", "TYS": "TYR", "TYI": "TYR",
    "MLY": "LYS", "M3L": "LYS", "KCX": "LYS", "LLP": "LYS", "ALY": "LYS",
    "MLZ": "LYS", "KPI": "LYS", "SAH": "LYS",
    "HYP": "PRO", "HY3": "PRO",
    "PCA": "GLN", "MEN": "ASN", "MEQ": "GLN",
    "CGU": "GLU", "PHD": "ASP", "BFD": "ASP",
    "MHO": "MET", "FME": "MET", "SME": "MET",
    "TRQ": "TRP", "TRO": "TRP", "TRF": "TRP", "TRN": "TRP",
    "HIC": "HIS", "MHS": "HIS", "NEP": "HIS", "HIP": "HIS",
    "DAL": "ALA", "AIB": "ALA", "ABA": "ALA", "ORN": "ALA",
    "MVA": "VAL", "NLE": "LEU", "MLE": "LEU",
    "AGM": "ARG", "ARO": "ARG", "2MR": "ARG",
    "SAC": "SER", "GL3": "GLY", "SAR": "GLY",
    "FTR": "TRP", "PHI": "PHE", "PFF": "PHE", "DAH": "PHE",
}

#: Atom renames needed after a parent substitution, so voronota's per-atom typing and vdW
#: radii apply. MSE's selenium is the case that matters in practice.
ATOM_RENAMES = {
    ("MSE", "SE"): ("SD", "S"),
}


def parent_of(resname: str) -> str | None:
    """Standard parent for a residue name, or None if there is no sensible one."""
    resname = resname.strip().upper()
    if resname in STANDARD:
        return resname
    if resname in EXPLICIT_PARENTS:
        return EXPLICIT_PARENTS[resname]

    info = gemmi.find_tabulated_residue(resname)
    if info is not None and info.is_amino_acid():
        one = info.one_letter_code.upper()
        for three, letter in THREE_TO_ONE.items():
            if letter == one:
                return three
    return None


def normalize_residue(residue: gemmi.Residue) -> str | None:
    """Rewrite a residue in place to its standard parent. Returns the new name, or None.

    None means the residue has no standard parent and the caller must drop it.
    """
    original = residue.name.strip().upper()
    parent = parent_of(original)
    if parent is None:
        return None
    if parent == original:
        return parent

    for atom in residue:
        rename = ATOM_RENAMES.get((original, atom.name.strip().upper()))
        if rename is not None:
            atom.name, element = rename
            atom.element = gemmi.Element(element)
    residue.name = parent
    return parent


def one_letter(resname: str) -> str:
    return THREE_TO_ONE.get(resname.strip().upper(), "X")
