"""Structure normalization: the SEQRES renumbering that the ESM alignment depends on."""

from __future__ import annotations

import gemmi
import pytest

from jbbind.core.structure.modres import one_letter, parent_of
from jbbind.core.structure.normalize import (MIN_CHAIN_LENGTH, NormalizationError,
                                             VORONOTA_CHAIN_ID, list_chains,
                                             prepare_chain, read_structure)

AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


def make_pdb(residues, chain="A", seqres=None, models=1):
    """A minimal PDB: `residues` is [(resSeq, icode, resname)]."""
    lines = []
    if seqres:
        for i in range(0, len(seqres), 13):
            names = " ".join(f"{n:>3}" for n in seqres[i:i + 13])
            lines.append(f"SEQRES {i // 13 + 1:>3} {chain} {len(seqres):>4}  {names}")
    serial = 1
    for m in range(models):
        if models > 1:
            lines.append(f"MODEL     {m + 1:>4}")
        for j, (num, icode, name) in enumerate(residues):
            for atom, elem in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
                x = 3.0 * j + {"N": 0.0, "CA": 1.0, "C": 2.0, "O": 2.5}[atom]
                lines.append(
                    f"ATOM  {serial:>5}  {atom:<3}{name:>4} {chain}{num:>4}{icode:1}   "
                    f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {elem:>2}")
                serial += 1
        if models > 1:
            lines.append("ENDMDL")
    lines.append("END")
    return ("\n".join(lines) + "\n").encode()


def simple(n=12, start=1, name_cycle=AA):
    return [(start + i, " ", name_cycle[i % len(name_cycle)]) for i in range(n)]


def test_dense_numbering_is_preserved():
    residues = simple(12)
    seqres = [r[2] for r in residues]
    st = read_structure(make_pdb(residues, seqres=seqres))
    prepared = prepare_chain(st, "A")
    assert [r.seqres_index for r in prepared.residues] == list(range(1, 13))
    assert prepared.numbering_source == "seqres"
    assert prepared.sequence == "".join(one_letter(n) for n in seqres)


def test_gapped_chain_maps_onto_seqres_not_rank():
    """The case that makes rank-based indexing catastrophic.

    SEQRES has 12 residues; the structure resolves 1-4 and 9-12. Positions must come out
    as 1,2,3,4,9,10,11,12 -- not 1..8.
    """
    seqres = [AA[i] for i in range(12)]
    observed = [(i + 1, " ", seqres[i]) for i in list(range(4)) + list(range(8, 12))]
    st = read_structure(make_pdb(observed, seqres=seqres))
    prepared = prepare_chain(st, "A")
    assert [r.seqres_index for r in prepared.residues] == [1, 2, 3, 4, 9, 10, 11, 12]
    for r in prepared.residues:
        assert prepared.sequence[r.seqres_index - 1] == r.one_letter


def test_author_numbering_offset_is_renumbered():
    """Author numbering starting at 501 must still map onto SEQRES positions 1..n."""
    seqres = [AA[i] for i in range(10)]
    observed = [(501 + i, " ", seqres[i]) for i in range(10)]
    st = read_structure(make_pdb(observed, seqres=seqres))
    prepared = prepare_chain(st, "A")
    assert [r.seqres_index for r in prepared.residues] == list(range(1, 11))
    assert [r.auth_seq_id for r in prepared.residues] == list(range(501, 511))


def test_negative_author_numbering():
    seqres = [AA[i] for i in range(10)]
    observed = [(-3 + i, " ", seqres[i]) for i in range(10)]
    st = read_structure(make_pdb(observed, seqres=seqres))
    prepared = prepare_chain(st, "A")
    assert [r.seqres_index for r in prepared.residues] == list(range(1, 11))
    assert prepared.residues[0].auth_seq_id == -3


def test_insertion_codes_get_distinct_indices():
    """Antibody-style 100/100A/100B must not collapse.

    train_multilabel's ``groupby("ID_resSeq")`` would merge them; SEQRES renumbering
    prevents that from ever reaching voronota.
    """
    seqres = [AA[i] for i in range(6)]
    observed = [(100, " ", seqres[0]), (100, "A", seqres[1]), (100, "B", seqres[2]),
                (101, " ", seqres[3]), (102, " ", seqres[4]), (103, " ", seqres[5])]
    st = read_structure(make_pdb(observed, seqres=seqres))
    with pytest.raises(NormalizationError) as exc:
        prepare_chain(st, "A")           # 6 residues < MIN_CHAIN_LENGTH
    assert exc.value.code == "ChainTooShort"

    seqres = [AA[i % 20] for i in range(12)]
    observed = ([(100, " ", seqres[0]), (100, "A", seqres[1]), (100, "B", seqres[2])] +
                [(101 + i, " ", seqres[3 + i]) for i in range(9)])
    st = read_structure(make_pdb(observed, seqres=seqres))
    prepared = prepare_chain(st, "A")
    idx = [r.seqres_index for r in prepared.residues]
    assert len(idx) == len(set(idx)), "insertion-code residues collapsed onto one index"
    assert [r.auth_icode for r in prepared.residues[:3]] == ["", "A", "B"]


def test_pdb_written_for_voronota_is_renumbered_and_single_chain():
    seqres = [AA[i] for i in range(12)]
    observed = [(500 + i, " ", seqres[i]) for i in range(12)]
    st = read_structure(make_pdb(observed, chain="XYZ" [0], seqres=seqres))
    prepared = prepare_chain(st, "X")
    atom_lines = [l for l in prepared.pdb_text.splitlines() if l.startswith("ATOM")]
    assert atom_lines, "no atoms written"
    chains = {l[21] for l in atom_lines}
    assert chains == {VORONOTA_CHAIN_ID}
    resseqs = sorted({int(l[22:26]) for l in atom_lines})
    assert resseqs == list(range(1, 13))
    assert all(l[26] == " " for l in atom_lines), "insertion codes must be cleared"


def test_mse_is_mapped_to_met():
    assert parent_of("MSE") == "MET"
    seqres = ["MET"] + [AA[i] for i in range(1, 12)]
    observed = [(1, " ", "MSE")] + [(i + 1, " ", seqres[i]) for i in range(1, 12)]
    st = read_structure(make_pdb(observed, seqres=seqres))
    prepared = prepare_chain(st, "A")
    assert prepared.residues[0].resname == "MET"
    assert prepared.residues[0].one_letter == "M"
    # The selenium must be renamed too, or voronota's atom typing and radii are wrong.
    first = [l for l in prepared.pdb_text.splitlines() if l.startswith("ATOM")][:4]
    assert not any(" SE " in l for l in first)


def test_unknown_residue_is_dropped_with_a_warning():
    seqres = [AA[i] for i in range(12)]
    observed = [(i + 1, " ", seqres[i]) for i in range(12)]
    observed[5] = (6, " ", "UNK")
    st = read_structure(make_pdb(observed, seqres=seqres))
    prepared = prepare_chain(st, "A")
    assert 6 not in {r.seqres_index for r in prepared.residues}
    assert any(w["code"] in ("nonstandard_residues_dropped", "sequence_mismatch_dropped")
               for w in prepared.warnings)


def test_multi_model_keeps_only_the_first():
    """Required: the voronota script imports with -as-assembly, so extra models would be
    merged into superimposed copies and wreck the tessellation."""
    seqres = [AA[i] for i in range(12)]
    observed = [(i + 1, " ", seqres[i]) for i in range(12)]
    st = read_structure(make_pdb(observed, seqres=seqres, models=5))
    prepared = prepare_chain(st, "A")
    assert len(prepared.residues) == 12
    assert any(w["code"] == "multiple_models" for w in prepared.warnings)
    atom_lines = [l for l in prepared.pdb_text.splitlines() if l.startswith("ATOM")]
    assert len(atom_lines) == 12 * 4


def test_no_seqres_falls_back_loudly():
    observed = simple(12)
    st = read_structure(make_pdb(observed, seqres=None))
    prepared = prepare_chain(st, "A")
    assert prepared.numbering_source == "observed"
    assert any(w["code"] == "no_seqres" for w in prepared.warnings), \
        "the degraded numbering path must never be silent"


def test_short_chain_is_refused():
    st = read_structure(make_pdb(simple(MIN_CHAIN_LENGTH - 1)))
    with pytest.raises(NormalizationError) as exc:
        prepare_chain(st, "A")
    assert exc.value.code in ("ChainTooShort", "NoPolymerChains")


def test_missing_chain_reports_what_is_available():
    st = read_structure(make_pdb(simple(12), chain="A"))
    with pytest.raises(NormalizationError) as exc:
        prepare_chain(st, "Q")
    assert exc.value.code == "ChainNotFound"
    assert "A" in exc.value.extra.get("available", [])


def test_list_chains_reports_both_lengths():
    seqres = [AA[i] for i in range(20)]
    observed = [(i + 1, " ", seqres[i]) for i in range(12)]
    st = read_structure(make_pdb(observed, seqres=seqres))
    chains, _ = list_chains(st)
    assert len(chains) == 1
    assert chains[0].n_observed == 12
    assert chains[0].n_seqres == 20
