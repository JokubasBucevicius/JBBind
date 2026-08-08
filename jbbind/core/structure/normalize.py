"""Turn an arbitrary PDB/mmCIF into a clean, SEQRES-numbered, single-chain PDB.

This module carries the correctness burden of the whole application. Everything downstream
assumes ``ID_resSeq`` is a 1-based index into the chain's canonical (SEQRES) sequence,
because that is what the training data had and what ``train_multilabel.py:329``'s
``embedding_indices = ID_resSeq - 1`` relies on. Verified on 250 random training chains:
``one_letter(ID_resName) == s1_sequence[ID_resSeq - 1]`` holds for 94.4% of them.

Author numbering from RCSB is *not* that index — it has gaps, insertion codes, negative
values and expression-tag offsets. Feeding it through unchanged would silently pair each
residue with the wrong ESM embedding on every gapped chain, and gapped chains are the
majority.

The design principle: all the messy handling happens here, in Python, where it is testable.
``voronota-js`` only ever sees a single chain, single model, standard residues, renumbered
1..L, no insertion codes.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Optional

import gemmi

from .modres import STANDARD, normalize_residue, one_letter

#: Minimum polymer length worth predicting on. GATv2 over a handful of nodes is meaningless.
MIN_CHAIN_LENGTH = 8
#: PDB format has 4 columns for resSeq.
MAX_SEQRES_INDEX = 9999
#: Below this, the observed->canonical alignment is not trustworthy.
MIN_ALIGNMENT_FRACTION = 0.8


class NormalizationError(Exception):
    """Raised when a structure or chain cannot be prepared for inference."""

    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


@dataclass
class ResidueRecord:
    """One residue of the prepared chain, keyed by its SEQRES index."""

    seqres_index: int          # 1-based index into the canonical sequence
    auth_chain: str
    auth_seq_id: int
    auth_icode: str
    resname: str               # after modified-residue substitution
    resname_original: str
    one_letter: str


@dataclass
class ChainInfo:
    """A candidate chain, as offered to the user before prediction."""

    chain_id: str              # auth chain id, what the user sees
    entity_id: str
    n_observed: int
    n_seqres: int
    numbering_source: str      # "seqres" | "observed"
    sequence: str


@dataclass
class PreparedChain:
    """A chain ready for voronota + ESM."""

    chain_id: str
    pdb_text: str              # single chain, renamed to A, renumbered to SEQRES indices
    sequence: str              # canonical sequence; the ESM input
    numbering_source: str
    residues: list[ResidueRecord]
    warnings: list[dict] = field(default_factory=list)

    @property
    def residue_map(self) -> dict[int, ResidueRecord]:
        return {r.seqres_index: r for r in self.residues}


def read_structure(data: bytes, name: str = "input") -> gemmi.Structure:
    """Parse PDB or mmCIF (optionally gzipped) from bytes."""
    import gzip

    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    text = data.decode("utf-8", errors="replace")
    try:
        st = gemmi.read_structure_string(text)
    except Exception as exc:
        raise NormalizationError("ParseError", f"could not parse structure: {exc}") from exc
    if len(st) == 0:
        raise NormalizationError("ParseError", "structure contains no models")
    st.name = name
    return st


def _clean(st: gemmi.Structure) -> list[dict]:
    """Reduce to a single model of standard protein residues. Returns warnings."""
    warnings: list[dict] = []

    # Keep model 0 only. This is mandatory, not cosmetic: the voronota script imports with
    # '-as-assembly' (describe-receptor-chain, mirroring the original's line 244), which
    # merges all MODEL blocks into one object. A 20-model NMR ensemble would become 20
    # superimposed copies and a meaningless tessellation.
    if len(st) > 1:
        warnings.append({
            "code": "multiple_models",
            "detail": f"structure has {len(st)} models (likely NMR); using model 1 only.",
        })
        while len(st) > 1:
            del st[len(st) - 1]

    st.setup_entities()
    st.remove_alternative_conformations()  # training data has ID_altLoc == '.' throughout
    st.remove_hydrogens()
    st.remove_waters()
    st.remove_ligands_and_waters()
    return warnings


def _canonical_sequence(st: gemmi.Structure, chain: gemmi.Chain,
                        polymer: gemmi.ResidueSpan) -> tuple[str, str]:
    """(sequence, numbering_source). Prefers SEQRES; falls back to observed residues."""
    entity = st.get_entity_of(polymer)
    if entity is not None and entity.full_sequence:
        seq = gemmi.one_letter_code(entity.full_sequence)
        if seq and len(seq) >= len(polymer):
            return seq.upper(), "seqres"
    return polymer.make_one_letter_sequence().upper(), "observed"


def _map_to_seqres(polymer: gemmi.ResidueSpan, sequence: str,
                   numbering_source: str) -> tuple[dict[int, int], list[dict]]:
    """Map each observed residue's position in the polymer to its 1-based SEQRES index.

    Two strategies, in order of trustworthiness:

    1. mmCIF ``label_seq_id``, which is *by definition* the 1-based index into
       ``_entity_poly_seq``. When present and self-consistent this needs no alignment.
    2. Sequence alignment of the observed one-letter string against the canonical one,
       via ``difflib.SequenceMatcher``. For two near-identical strings its matching blocks
       are exactly the gap structure; a full Needleman-Wunsch would add risk, not accuracy.

    Either way the invariant ``one_letter(resname) == sequence[idx - 1]`` is checked for
    every residue, and violators are dropped. That invariant is the app's contract with the
    checkpoints, so it is enforced at runtime, not just in tests.
    """
    warnings: list[dict] = []
    observed = "".join(one_letter(r.name) for r in polymer)

    mapping: dict[int, int] = {}
    if numbering_source == "seqres":
        label_ok = 0
        for pos, res in enumerate(polymer):
            lab = res.label_seq
            if lab is not None and 1 <= lab <= len(sequence) \
                    and sequence[lab - 1] == observed[pos]:
                mapping[pos] = lab
                label_ok += 1
        if label_ok < 0.95 * len(polymer):
            mapping = {}  # not trustworthy; fall through to alignment
        else:
            return mapping, warnings

    if numbering_source == "observed":
        # The canonical sequence IS the observed sequence, so the mapping is the identity.
        return {pos: pos + 1 for pos in range(len(polymer))}, warnings

    matcher = difflib.SequenceMatcher(None, observed, sequence, autojunk=False)
    matched = 0
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k + 1
            matched += 1
    frac = matched / max(1, len(observed))
    if frac < MIN_ALIGNMENT_FRACTION:
        raise NormalizationError(
            "SequenceMappingFailed",
            f"only {frac:.0%} of observed residues could be aligned to the canonical "
            f"sequence; refusing to guess the ESM alignment.",
            matched_fraction=round(frac, 4))
    if frac < 0.99:
        warnings.append({
            "code": "partial_alignment",
            "detail": f"{matched}/{len(observed)} observed residues aligned to the "
                      f"canonical sequence ({frac:.1%}).",
            "matched_fraction": round(frac, 4),
        })
    return mapping, warnings


def list_chains(st: gemmi.Structure) -> tuple[list[ChainInfo], list[dict]]:
    """Protein chains worth predicting on."""
    warnings = _clean(st)
    out: list[ChainInfo] = []
    for chain in st[0]:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        kind = polymer.check_polymer_type()
        if kind not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            continue
        if len(polymer) < MIN_CHAIN_LENGTH:
            continue
        sequence, source = _canonical_sequence(st, chain, polymer)
        entity = st.get_entity_of(polymer)
        out.append(ChainInfo(
            chain_id=chain.name,
            entity_id=entity.name if entity else "",
            n_observed=len(polymer),
            n_seqres=len(sequence),
            numbering_source=source,
            sequence=sequence,
        ))
    if not out:
        raise NormalizationError(
            "NoPolymerChains",
            f"no protein chain of at least {MIN_CHAIN_LENGTH} residues was found.")
    return out, warnings


def prepare_chain(st: gemmi.Structure, chain_id: str) -> PreparedChain:
    """Produce the single-chain, SEQRES-numbered PDB that voronota will consume."""
    warnings = _clean(st)

    chain = None
    for c in st[0]:
        if c.name == chain_id:
            chain = c
            break
    if chain is None:
        available = [c.name for c in st[0]]
        raise NormalizationError("ChainNotFound",
                                 f"chain {chain_id!r} not found; available: {available}",
                                 available=available)

    polymer = chain.get_polymer()
    if len(polymer) < MIN_CHAIN_LENGTH:
        raise NormalizationError(
            "ChainTooShort",
            f"chain {chain_id} has {len(polymer)} residues; the minimum is "
            f"{MIN_CHAIN_LENGTH}.")

    sequence, numbering_source = _canonical_sequence(st, chain, polymer)
    if numbering_source == "observed":
        warnings.append({
            "code": "no_seqres",
            "detail": "this structure has no SEQRES/_entity_poly record, so the canonical "
                      "sequence was taken from the observed residues. Any unmodelled "
                      "residue shifts the ESM alignment relative to how the models were "
                      "trained; treat the prediction as approximate.",
        })
    if len(sequence) > MAX_SEQRES_INDEX:
        raise NormalizationError(
            "TooManyResidues",
            f"sequence length {len(sequence)} exceeds the {MAX_SEQRES_INDEX}-residue "
            f"limit imposed by the PDB format's residue-number field.")

    # Substitute modified residues before anything reads a residue name.
    dropped_nonstandard: list[str] = []
    for res in polymer:
        if normalize_residue(res) is None:
            dropped_nonstandard.append(res.name)

    mapping, map_warnings = _map_to_seqres(polymer, sequence, numbering_source)
    warnings.extend(map_warnings)

    records: list[ResidueRecord] = []
    kept: list[gemmi.Residue] = []
    invariant_failures = 0
    for pos, res in enumerate(polymer):
        if res.name not in STANDARD:
            continue
        idx = mapping.get(pos)
        if idx is None:
            continue
        letter = one_letter(res.name)
        # The contract with the checkpoints. Enforced here, at runtime.
        if sequence[idx - 1] != letter:
            invariant_failures += 1
            continue
        records.append(ResidueRecord(
            seqres_index=idx,
            auth_chain=chain.name,
            auth_seq_id=res.seqid.num,
            auth_icode=(res.seqid.icode or " ").strip(),
            resname=res.name,
            resname_original=res.name,
            one_letter=letter,
        ))
        kept.append(res)

    if dropped_nonstandard:
        uniq = sorted(set(dropped_nonstandard))
        warnings.append({
            "code": "nonstandard_residues_dropped",
            "detail": f"{len(dropped_nonstandard)} residue(s) with no standard parent were "
                      f"removed: {', '.join(uniq[:10])}"
                      f"{'…' if len(uniq) > 10 else ''}. Removing atoms perturbs "
                      f"neighbouring residues' surface features slightly.",
            "names": uniq,
        })
    if invariant_failures:
        warnings.append({
            "code": "sequence_mismatch_dropped",
            "detail": f"{invariant_failures} residue(s) did not match the canonical "
                      f"sequence at their mapped position and were dropped rather than "
                      f"paired with the wrong embedding.",
        })
    if not records:
        raise NormalizationError(
            "ChainTooShort",
            f"chain {chain_id} has no residue that could be mapped onto its sequence.")

    pdb_text = _write_single_chain_pdb(kept, records)
    return PreparedChain(chain_id=chain_id, pdb_text=pdb_text, sequence=sequence,
                         numbering_source=numbering_source, residues=records,
                         warnings=warnings)


#: The chain id handed to voronota. Renaming sidesteps multi-character auth ids (PPI3D has
#: chains like `AAA`) which do not fit the PDB format's single-character column.
VORONOTA_CHAIN_ID = "A"


def _write_single_chain_pdb(residues: list[gemmi.Residue],
                            records: list[ResidueRecord]) -> str:
    """Emit a minimal PDB: one chain, one model, resSeq = SEQRES index, no insertion codes."""
    st = gemmi.Structure()
    st.name = "chain"
    model = gemmi.Model("1")
    chain = gemmi.Chain(VORONOTA_CHAIN_ID)

    for res, rec in zip(residues, records):
        new = gemmi.Residue()
        new.name = rec.resname
        new.seqid = gemmi.SeqId(rec.seqres_index, " ")
        new.het_flag = "A"
        for atom in res:
            a = gemmi.Atom()
            a.name = atom.name
            a.element = atom.element
            a.pos = atom.pos
            a.occ = 1.00
            a.b_iso = 0.00
            a.altloc = "\0"
            new.add_atom(a)
        chain.add_residue(new)

    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st.make_pdb_string()
