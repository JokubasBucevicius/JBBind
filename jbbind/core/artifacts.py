"""Downloadable artifacts: CSV, B-factor PDB, PyMOL selection."""

from __future__ import annotations

import csv
import io
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import PredictionResult

#: B-factor sentinel for residues with no prediction (buried, or past ESM truncation).
#: Chosen negative so it can never be mistaken for a probability, and so PyMOL/Chimera
#: colour ramps put it clearly outside the data range.
UNPREDICTED_B = -1.00


def slug(text: str) -> str:
    """A label or setup name as a filename fragment."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def predictions_csv(result: "PredictionResult", setup: str | None = None) -> str:
    setups = [setup] if setup else list(result.label_names)
    buf = io.StringIO()
    writer = csv.writer(buf)

    header = ["seqres_index", "auth_chain", "auth_seq_id", "auth_icode",
              "resname", "one_letter", "sas_area"]
    for s in setups:
        for label in result.label_names[s]:
            header.append(f"{s}:{label}")
    writer.writerow(header)

    for r in result.residues:
        row = [r.seqres_index, r.auth_chain, r.auth_seq_id, r.auth_icode,
               r.resname, r.one_letter,
               "" if r.sas_area is None else f"{r.sas_area:.4f}"]
        for s in setups:
            row.extend(f"{p:.5f}" for p in r.probs[s])
        writer.writerow(row)
    return buf.getvalue()


def predictions_pdb(result: "PredictionResult", setup: str, label_index: int = 0) -> str:
    """The receptor structure with the chosen probability written into the B-factor column.

    ``receptor.pdb`` comes from voronota's ``[tessellated]`` selection, which is exactly the
    atom set behind graph_nodes.csv, and its resSeq is already the SEQRES index — so the
    join below is exact rather than a best-effort match.
    """
    labels = result.label_names[setup]
    label = labels[label_index]
    by_resi = {r.seqres_index: r.probs[setup][label_index] for r in result.residues}

    out: list[str] = [
        f"REMARK 100 JBBind per-residue binding score",
        f"REMARK 100 source     {result.source}",
        f"REMARK 100 chain      {result.chain_id}",
        f"REMARK 100 model      {result.arch}",
        f"REMARK 100 setup      {setup}",
        f"REMARK 100 label      {label}",
        f"REMARK 100 B-factor   = score x 100 (0-100)",
        f"REMARK 100 B-factor   = {UNPREDICTED_B:.2f} means no prediction "
        f"(buried residue, or beyond the ESM-2 1022-residue limit)",
        f"REMARK 100 NOTE       scores are uncalibrated sigmoid outputs, not probabilities",
    ]

    for line in result.receptor_pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            if line.startswith("END"):
                continue
            out.append(line)
            continue
        try:
            resi = int(line[22:26])
        except ValueError:
            out.append(line)
            continue
        prob = by_resi.get(resi)
        b = UNPREDICTED_B if prob is None else prob * 100.0
        out.append(f"{line[:60]}{b:6.2f}{line[66:]}".rstrip())
    out.append("END")
    return "\n".join(out) + "\n"


def pymol_selection(result: "PredictionResult", setup: str, label_index: int = 0,
                    threshold: float = 0.5, name: str = "bindingsite") -> str:
    labels = result.label_names[setup]
    hits = sorted(r.auth_seq_id for r in result.residues
                  if r.probs[setup][label_index] >= threshold)
    if not hits:
        return f"# no residue scores >= {threshold:g} for {setup}:{labels[label_index]}"
    chain = result.residues[0].auth_chain
    return (f"# {setup}:{labels[label_index]} >= {threshold:g}  ({len(hits)} residues)\n"
            f"select {name}, chain {chain} and resi {'+'.join(map(str, hits))}")
