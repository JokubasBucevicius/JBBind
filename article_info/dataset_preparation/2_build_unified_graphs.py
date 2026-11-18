#!/usr/bin/env python3
"""
Unified pipeline for the all_class dataset.

Drives everything from ppi3d_filtered.csv: classifies each row
(homo / hetero / peptide / DNA / RNA / Hybrid), downloads one PDB per
complex, generates per-chain graphs via extract-and-describe-receptor-protein
(--chain-id "A,B"), extracts interfaces with voronota, and merges per-atom
binding flags across all complexes that share the same subunit_1 (chain A)
or subunit_2 (chain B; only PP rows contribute to chain B).

Output flag columns (unchanged from the previous pipeline):
  binds_any, binds_protein, binds_homo_protein, binds_hetero_protein,
  binds_nucleic, binds_dna, binds_rna

Steps:
    --step download   # 1. Download per-complex PDBs
    --step generate   # 2. Build base graphs (one per chain) per complex
    --step extract    # 3. Voronota interface extraction per chain
    --step build      # 4. Aggregate flags across complexes, write finals
    --step stats      # Stats over the final output
    --step all        # download -> generate -> extract -> build
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ── Configuration ────────────────────────────────────────────────────────────

ALL_CLASS_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH       = os.path.join(ALL_CLASS_DIR, "ppi3d_test_data_filtered.csv")
STRUCTURES_DIR = os.path.join(ALL_CLASS_DIR, "structures")
OUTPUT_DIR     = os.path.join(ALL_CLASS_DIR, "graphs", "unified")
PER_COMPLEX    = os.path.join(OUTPUT_DIR, "_per_complex")
BSITE_DIR      = os.path.join(OUTPUT_DIR, "_bsite")
LOG_DIR        = os.path.join(OUTPUT_DIR, "_logs")
GRAPH_TOOL     = os.path.join(ALL_CLASS_DIR, "graphs", "tools",
                              "extract-and-describe-receptor-protein")

N_DOWNLOAD_WORKERS  = 4
N_GRAPH_GEN_WORKERS = 36
N_EXTRACT_WORKERS   = 36
N_BUILD_WORKERS     = 36

FLAG_COLS = ['binds_any', 'binds_protein', 'binds_homo_protein',
             'binds_hetero_protein', 'binds_nucleic', 'binds_dna', 'binds_rna']
LEGACY_COLS = ['bsite', 'ssDNA_bind', 'dsDNA_bind', 'RNA_bind',
               'homo_bind', 'hetero_bind']


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_bsite_id(bsite_id):
    """Parse 'c<A>r<20>R<SER>A<O>' -> ('A', '20', 'SER', 'O') or None."""
    m = re.match(r'c<(\w+)>r<(-?\d+\w*)>R<(\w+)>A<(\w+)>', bsite_id)
    if m:
        return (m.group(1), m.group(2), m.group(3), m.group(4))
    return None


def read_bsite_atoms(bsite_path):
    """Read bsite_areas.tsv -> {atom_key: max_area} for area > 0."""
    atoms = {}
    if not os.path.exists(bsite_path):
        return atoms
    with open(bsite_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    area = float(parts[1])
                except ValueError:
                    continue
                if area > 0:
                    key = parse_bsite_id(parts[0])
                    if key:
                        atoms[key] = max(atoms.get(key, 0.0), area)
    return atoms


def make_atom_key(row):
    return (str(row['ID_chainID']), str(row['ID_resSeq']),
            str(row['ID_resName']), str(row['ID_name']))


def detect_protein_chains(pdb_path):
    """Return list of protein chain IDs (in PDB order) using voronota."""
    try:
        with open(pdb_path, 'rb') as f:
            pdb_content = f.read()
        result = subprocess.run(
            ["voronota-js-pdb-utensil-print-sequence-from-structure",
             "--selection", "[-protein]"],
            input=pdb_content, capture_output=True, timeout=30,
        )
        chains = []
        for line in result.stdout.decode().split('\n'):
            if line.startswith('>'):
                token = line.lstrip('>').split()[0]
                if token and token not in chains:
                    chains.append(token)
        if chains:
            return chains
    except Exception:
        pass
    # Fallback: parse ATOM records (col 22)
    chains = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM  ') and len(line) >= 22:
                    c = line[21].strip()
                    if c and c not in chains:
                        chains.append(c)
    except Exception:
        pass
    return chains


def _merge_atom_dicts(base, new):
    for key, area in new.items():
        base[key] = max(base.get(key, 0.0), area)


def base_name(subunit_1, subunit_2):
    return f"{subunit_1}-{subunit_2}"


def complex_pdb_path(subunit_1, subunit_2):
    return os.path.join(STRUCTURES_DIR, f"{base_name(subunit_1, subunit_2)}.pdb")


# ── Classification ───────────────────────────────────────────────────────────

def classify(row):
    """Return (set_of_flags, kind) where kind ∈ {'pp','peptide','na',''}."""
    if int(row.get('protein_protein_interaction', 0) or 0) == 1:
        if int(row.get('homo', 0) or 0) == 1:
            return ({'homo_protein'}, 'pp')
        return ({'hetero_protein'}, 'pp')
    if int(row.get('protein_peptide_interaction', 0) or 0) == 1:
        return ({'hetero_protein'}, 'peptide')
    if int(row.get('protein_nucleic_interaction', 0) or 0) == 1:
        na_type = str(row.get('na_type', '') or '').strip()
        if na_type == 'DNA':    return ({'dna'}, 'na')
        if na_type == 'RNA':    return ({'rna'}, 'na')
        if na_type == 'Hybrid': return ({'dna', 'rna'}, 'na')
        title = str(row.get('subunit_2_title', '') or '').upper()
        flags = set()
        if 'DNA' in title: flags.add('dna')
        if 'RNA' in title: flags.add('rna')
        return (flags, 'na')
    return (set(), '')


# ── Chain resolution ─────────────────────────────────────────────────────────

def expected_chain(subunit_id):
    """'1a09_B' -> 'B' (last token after '_')."""
    if not isinstance(subunit_id, str) or '_' not in subunit_id:
        return None
    return subunit_id.split('_')[-1]


def resolve_chains(pdb_path, subunit_1, subunit_2, kind, log_writer=None):
    """Resolve actual PDB chain letters for the chain A side and (PP only)
    chain B side. Returns dict {'A': actual_letter_or_None, 'B': ... or None}.

    For non-PP rows the 'B' slot is always None (peptide/NA: chain A only).
    """
    chains = detect_protein_chains(pdb_path)
    exp_a = expected_chain(subunit_1)
    exp_b = expected_chain(subunit_2) if kind == 'pp' else None

    actual_a = exp_a if (exp_a in chains) else (chains[0] if chains else None)
    actual_b = None
    if kind == 'pp':
        if exp_b in chains and exp_b != actual_a:
            actual_b = exp_b
        else:
            for c in chains:
                if c != actual_a:
                    actual_b = c
                    break

    if log_writer is not None:
        if kind == 'pp':
            if actual_a != exp_a or actual_b != exp_b:
                log_writer.write(
                    f"{base_name(subunit_1, subunit_2)}\t"
                    f"expected={exp_a},{exp_b}\tactual={actual_a},{actual_b}\t"
                    f"all_protein_chains={','.join(chains)}\n"
                )
        else:
            if actual_a != exp_a:
                log_writer.write(
                    f"{base_name(subunit_1, subunit_2)}\t"
                    f"expected={exp_a}\tactual={actual_a}\t"
                    f"all_protein_chains={','.join(chains)}\n"
                )

    return {'A': actual_a, 'B': actual_b}


# ── Step 1: Download ─────────────────────────────────────────────────────────

def _download_one(args):
    url, out_path = args
    if os.path.exists(out_path):
        return ('exists', out_path)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(resp.content)
        return ('downloaded', out_path)
    except Exception as e:
        return (f'error: {e}', out_path)


def download_complexes(df):
    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    tasks = []
    for _, row in df.iterrows():
        url = row.get('download_url')
        if not isinstance(url, str) or not url:
            continue
        out_path = complex_pdb_path(row['subunit_1'], row['subunit_2'])
        tasks.append((url, out_path))

    print(f"Download tasks: {len(tasks)} (workers={N_DOWNLOAD_WORKERS})")
    counts = defaultdict(int)
    errors = []
    with ThreadPoolExecutor(max_workers=N_DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(_download_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            status, path = fut.result()
            key = 'error' if status.startswith('error') else status
            counts[key] += 1
            if key == 'error':
                errors.append((path, status))
    print("\nDownload results:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if errors:
        print("\nFirst 10 errors:")
        for p, e in errors[:10]:
            print(f"  {p}: {e}")


# ── Step 2: Generate base graphs ─────────────────────────────────────────────

def _clear_chain_dirs(per_complex_dir, base):
    """Remove any stale {base}_chain_* dirs before regenerating."""
    if not os.path.isdir(per_complex_dir):
        return
    prefix = f"{base}_chain_"
    for entry in os.listdir(per_complex_dir):
        if entry.startswith(prefix):
            full = os.path.join(per_complex_dir, entry)
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)


def _rename_chains_to_positional(per_complex_dir, base, actual_a, actual_b):
    """Rename {base}_chain_<actual_a> -> _chain_A and _chain_<actual_b> -> _chain_B.

    Handles the swap case (actual_a == 'B' and actual_b == 'A') by routing
    through temporary names so neither rename clobbers the other's source.

    Returns (ok_a, ok_b) booleans.
    """
    tmp_a = os.path.join(per_complex_dir, f"{base}_TMP_A")
    tmp_b = os.path.join(per_complex_dir, f"{base}_TMP_B")
    dst_a = os.path.join(per_complex_dir, f"{base}_chain_A")
    dst_b = os.path.join(per_complex_dir, f"{base}_chain_B")
    src_a = os.path.join(per_complex_dir, f"{base}_chain_{actual_a}") if actual_a else None
    src_b = os.path.join(per_complex_dir, f"{base}_chain_{actual_b}") if actual_b else None

    # Step 1: stage sources to unique temp names (so dst clobber can't race)
    ok_a = False
    ok_b = False
    if src_a and os.path.isdir(src_a):
        if os.path.isdir(tmp_a):
            shutil.rmtree(tmp_a, ignore_errors=True)
        os.rename(src_a, tmp_a)
        ok_a = True
    if src_b and os.path.isdir(src_b):
        if os.path.isdir(tmp_b):
            shutil.rmtree(tmp_b, ignore_errors=True)
        os.rename(src_b, tmp_b)
        ok_b = True

    # Step 2: move temp names to final positional slots
    if ok_a:
        if os.path.isdir(dst_a):
            shutil.rmtree(dst_a, ignore_errors=True)
        os.rename(tmp_a, dst_a)
    if ok_b:
        if os.path.isdir(dst_b):
            shutil.rmtree(dst_b, ignore_errors=True)
        os.rename(tmp_b, dst_b)

    return ok_a, ok_b


def _generate_one(args):
    pdb_path, base, kind, subunit_1, subunit_2, per_complex_dir, tool, log_path = args

    final_a = os.path.join(per_complex_dir, f"{base}_chain_A", "graph_nodes.csv")
    final_b = os.path.join(per_complex_dir, f"{base}_chain_B", "graph_nodes.csv")
    needs_b = (kind == 'pp')
    if os.path.exists(final_a) and (not needs_b or os.path.exists(final_b)):
        return (base, 'exists')

    if not os.path.exists(pdb_path):
        return (base, 'pdb_missing')

    # Resolve chain letters
    log_writer = open(log_path, 'a')
    try:
        chain_map = resolve_chains(pdb_path, subunit_1, subunit_2, kind, log_writer)
    finally:
        log_writer.close()

    actual_a = chain_map['A']
    actual_b = chain_map['B']
    if actual_a is None:
        return (base, 'no_protein_chain')

    # Clear any stale outputs from previous runs so the rename step starts clean.
    _clear_chain_dirs(per_complex_dir, base)

    # The tool treats --chain-id as the "receptor" side; everything not in that
    # set is the partner. Passing both chains at once leaves no partner, so we
    # invoke once per chain.
    runs = [actual_a]
    if needs_b and actual_b is not None:
        runs.append(actual_b)

    last_stderr = b''
    for actual in runs:
        try:
            result = subprocess.run(
                [tool,
                 "--input-complex", pdb_path,
                 "--output-dir", per_complex_dir,
                 "--chain-id", actual,
                 "--no-faspr"],
                capture_output=True, timeout=300,
            )
            last_stderr = result.stderr
        except subprocess.TimeoutExpired:
            return (base, 'timeout')
        except Exception as e:
            return (base, f'error: {e}')

    # Rename actual-letter dirs to positional A/B (handles the swap case)
    ok_a, ok_b_actual = _rename_chains_to_positional(
        per_complex_dir, base, actual_a,
        actual_b if (needs_b and actual_b is not None) else None,
    )
    ok_b = True if not (needs_b and actual_b is not None) else ok_b_actual

    if not ok_a:
        return (base, f'no_output_A: {last_stderr.decode()[:200]}')
    if needs_b and actual_b is not None and not ok_b:
        return (base, f'no_output_B: {last_stderr.decode()[:200]}')

    return (base, 'generated')


def generate_base_graphs(df):
    if not os.path.exists(GRAPH_TOOL):
        print(f"ERROR: graph tool not found: {GRAPH_TOOL}")
        sys.exit(1)
    os.makedirs(PER_COMPLEX, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "chain_resolution.log")
    open(log_path, 'w').close()  # truncate per run

    seen_bases = set()
    tasks = []
    for _, row in df.iterrows():
        flags, kind = classify(row)
        if not kind:
            continue
        s1 = row['subunit_1']
        s2 = row['subunit_2']
        base = base_name(s1, s2)
        if base in seen_bases:
            continue
        seen_bases.add(base)
        tasks.append((complex_pdb_path(s1, s2), base, kind, s1, s2,
                      PER_COMPLEX, GRAPH_TOOL, log_path))

    print(f"Generate tasks: {len(tasks)} (workers={N_GRAPH_GEN_WORKERS})")
    counts = defaultdict(int)
    errors = []
    with ThreadPoolExecutor(max_workers=N_GRAPH_GEN_WORKERS) as pool:
        futures = [pool.submit(_generate_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            base, status = fut.result()
            key = status.split(':')[0] if ':' in status else status
            counts[key] += 1
            if key not in ('generated', 'exists'):
                errors.append((base, status))

    print("\nGeneration results:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if errors:
        print("\nFirst 10 errors:")
        for b, e in errors[:10]:
            print(f"  {b}: {e}")
    print(f"\nChain-resolution warnings logged to: {log_path}")


# ── Step 3: Extract interfaces ──────────────────────────────────────────────

def _extract_one(args):
    pdb_path, bsite_path, actual_chain = args
    if os.path.exists(bsite_path) and os.path.getsize(bsite_path) > 0:
        return (bsite_path, 'exists')
    if not os.path.exists(pdb_path):
        return (bsite_path, 'pdb_missing')
    if actual_chain is None:
        return (bsite_path, 'no_chain')
    os.makedirs(os.path.dirname(bsite_path), exist_ok=True)
    try:
        result = subprocess.run(
            ["voronota-js-fast-iface-contacts",
             "--input", pdb_path,
             "--as-assembly",
             "--subselect-contacts",
             f"[-a1 [-chain {actual_chain}] -a2! [-chain {actual_chain}]]",
             "--output-bsite-file", bsite_path],
            capture_output=True, timeout=120,
        )
        if os.path.exists(bsite_path) and os.path.getsize(bsite_path) > 0:
            return (bsite_path, 'extracted')
        return (bsite_path, f'no_output: {result.stderr.decode()[:200]}')
    except subprocess.TimeoutExpired:
        return (bsite_path, 'timeout')
    except Exception as e:
        return (bsite_path, f'error: {e}')


def extract_all_interfaces(df):
    try:
        subprocess.run(["voronota-js-fast-iface-contacts", "--help"],
                       capture_output=True, timeout=10)
    except FileNotFoundError:
        print("ERROR: voronota-js-fast-iface-contacts not on PATH")
        sys.exit(1)

    os.makedirs(BSITE_DIR, exist_ok=True)

    tasks = []
    for _, row in df.iterrows():
        flags, kind = classify(row)
        if not kind:
            continue
        s1, s2 = row['subunit_1'], row['subunit_2']
        pdb = complex_pdb_path(s1, s2)
        if not os.path.exists(pdb):
            continue
        chain_map = resolve_chains(pdb, s1, s2, kind)
        base = base_name(s1, s2)
        if chain_map['A'] is not None:
            tasks.append((pdb,
                          os.path.join(BSITE_DIR, f"{base}_chain_A.tsv"),
                          chain_map['A']))
        if kind == 'pp' and chain_map['B'] is not None:
            tasks.append((pdb,
                          os.path.join(BSITE_DIR, f"{base}_chain_B.tsv"),
                          chain_map['B']))

    print(f"Extract tasks: {len(tasks)} (workers={N_EXTRACT_WORKERS})")
    counts = defaultdict(int)
    errors = []
    with ThreadPoolExecutor(max_workers=N_EXTRACT_WORKERS) as pool:
        futures = [pool.submit(_extract_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            path, status = fut.result()
            key = status.split(':')[0] if ':' in status else status
            counts[key] += 1
            if key not in ('extracted', 'exists'):
                errors.append((path, status))
    print("\nExtraction results:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if errors:
        print("\nFirst 10 errors:")
        for p, e in errors[:10]:
            print(f"  {p}: {e}")


# ── Step 4: Build aggregated graphs ─────────────────────────────────────────

def _aggregate_contributions(df):
    """Walk all rows, read per-complex bsite files, and build:
        agg_chainA[s1][flag] -> {atom_key: max_area}
        agg_chainB[s2][flag] -> {atom_key: max_area}    (PP rows only)
        per_chain_repr[s1]['A'] -> path of representative chain_A graph dir
        per_chain_repr[s2]['B'] -> path of representative chain_B graph dir
    """
    agg_a = defaultdict(lambda: defaultdict(dict))
    agg_b = defaultdict(lambda: defaultdict(dict))
    repr_a = {}
    repr_b = {}

    for _, row in df.iterrows():
        flags, kind = classify(row)
        if not kind or not flags:
            continue
        s1, s2 = row['subunit_1'], row['subunit_2']
        base = base_name(s1, s2)

        bsite_a = os.path.join(BSITE_DIR, f"{base}_chain_A.tsv")
        graph_a = os.path.join(PER_COMPLEX, f"{base}_chain_A")
        atoms_a = read_bsite_atoms(bsite_a)
        if atoms_a:
            for flag in flags:
                _merge_atom_dicts(agg_a[s1][flag], atoms_a)
        if s1 not in repr_a and os.path.isdir(graph_a):
            repr_a[s1] = graph_a

        if kind == 'pp':
            bsite_b = os.path.join(BSITE_DIR, f"{base}_chain_B.tsv")
            graph_b = os.path.join(PER_COMPLEX, f"{base}_chain_B")
            atoms_b = read_bsite_atoms(bsite_b)
            if atoms_b:
                for flag in flags:
                    _merge_atom_dicts(agg_b[s2][flag], atoms_b)
            if s2 not in repr_b and os.path.isdir(graph_b):
                repr_b[s2] = graph_b

    return agg_a, agg_b, repr_a, repr_b


def _write_aggregated_graph(args):
    protein_id, side, src_dir, flag_to_atoms, out_root = args
    nodes_path = os.path.join(src_dir, "graph_nodes.csv")
    links_path = os.path.join(src_dir, "graph_links.csv")
    if not os.path.exists(nodes_path):
        return (protein_id, side, 'no_nodes_csv', 0)
    try:
        df = pd.read_csv(nodes_path)
    except Exception as e:
        return (protein_id, side, f'read_error: {e}', 0)

    atom_keys = df.apply(make_atom_key, axis=1)

    # Aggregated max-area per atom (across all flags)
    all_areas = {}
    for flag in ('homo_protein', 'hetero_protein', 'dna', 'rna'):
        _merge_atom_dicts(all_areas, flag_to_atoms.get(flag, {}))
    if all_areas:
        df['bsite_area'] = atom_keys.map(all_areas).fillna(0.0)

    # Source flags
    df['binds_homo_protein']   = atom_keys.isin(set(flag_to_atoms.get('homo_protein', {}).keys())).astype(int)
    df['binds_hetero_protein'] = atom_keys.isin(set(flag_to_atoms.get('hetero_protein', {}).keys())).astype(int)
    df['binds_dna']            = atom_keys.isin(set(flag_to_atoms.get('dna', {}).keys())).astype(int)
    df['binds_rna']            = atom_keys.isin(set(flag_to_atoms.get('rna', {}).keys())).astype(int)
    df['binds_protein'] = ((df['binds_homo_protein'] == 1) |
                           (df['binds_hetero_protein'] == 1)).astype(int)
    df['binds_nucleic'] = ((df['binds_dna'] == 1) |
                           (df['binds_rna'] == 1)).astype(int)
    df['binds_any']     = ((df['binds_protein'] == 1) |
                           (df['binds_nucleic'] == 1)).astype(int)

    for col in LEGACY_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    out_dir = os.path.join(out_root, protein_id)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "graph_nodes.csv"), index=False)
    if os.path.exists(links_path):
        shutil.copy2(links_path, os.path.join(out_dir, "graph_links.csv"))

    return (protein_id, side, 'ok', len(df))


def build_unified_graphs(df):
    print("Aggregating per-complex contributions...")
    agg_a, agg_b, repr_a, repr_b = _aggregate_contributions(df)
    print(f"  chain A proteins (subunit_1): {len(agg_a)}  reps: {len(repr_a)}")
    print(f"  chain B proteins (subunit_2): {len(agg_b)}  reps: {len(repr_b)}")

    # Merge A-side and B-side per protein_id; same protein appearing as both
    # subunit_1 and subunit_2 collapses into one output directory.
    merged = defaultdict(lambda: defaultdict(dict))
    repr_dir = {}
    for pid, flag_to_atoms in agg_a.items():
        for flag, atoms in flag_to_atoms.items():
            _merge_atom_dicts(merged[pid][flag], atoms)
        if pid in repr_a:
            repr_dir.setdefault(pid, repr_a[pid])
    for pid, flag_to_atoms in agg_b.items():
        for flag, atoms in flag_to_atoms.items():
            _merge_atom_dicts(merged[pid][flag], atoms)
        if pid in repr_b:
            repr_dir.setdefault(pid, repr_b[pid])

    tasks = []
    for pid, flag_to_atoms in merged.items():
        if pid not in repr_dir:
            continue
        tasks.append((pid, '-', repr_dir[pid], dict(flag_to_atoms), OUTPUT_DIR))

    print(f"\nBuild tasks: {len(tasks)} (workers={N_BUILD_WORKERS})")
    counts = defaultdict(int)
    total_atoms = 0
    errors = []
    with Pool(processes=N_BUILD_WORKERS) as pool:
        for protein, side, status, n in tqdm(
                pool.imap_unordered(_write_aggregated_graph, tasks),
                total=len(tasks), desc="Building"):
            counts[status] += 1
            if status == 'ok':
                total_atoms += n
            else:
                errors.append((protein, side, status))

    print("\nBuild results:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"  total atoms written: {total_atoms:,}")
    if errors:
        print("\nFirst 10 errors:")
        for p, side, e in errors[:10]:
            print(f"  {p}: {e}")

    # Clean up intermediates: only the final aggregated graphs are needed.
    if counts.get('ok', 0) > 0:
        for path in (PER_COMPLEX, BSITE_DIR):
            if os.path.isdir(path):
                print(f"Removing intermediate directory: {path}")
                shutil.rmtree(path)


# ── Step 5: Stats ────────────────────────────────────────────────────────────

def print_statistics():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Output directory not found: {OUTPUT_DIR}")
        return

    dirs = [d for d in os.listdir(OUTPUT_DIR)
            if os.path.isdir(os.path.join(OUTPUT_DIR, d))
            and not d.startswith('_')]
    print(f"Aggregated graph directories: {len(dirs)}")

    total_atoms = 0
    flag_totals = {c: 0 for c in FLAG_COLS}
    proteins_with_flag = {c: 0 for c in FLAG_COLS}
    combo_counts = defaultdict(int)

    print("\nScanning aggregated graphs...")
    for d in tqdm(dirs, desc="Scanning"):
        nodes_csv = os.path.join(OUTPUT_DIR, d, "graph_nodes.csv")
        if not os.path.exists(nodes_csv):
            continue
        try:
            df = pd.read_csv(nodes_csv)
        except Exception:
            continue
        total_atoms += len(df)
        for c in FLAG_COLS:
            if c in df.columns:
                n = int(df[c].sum())
                flag_totals[c] += n
                if n > 0:
                    proteins_with_flag[c] += 1
        if all(c in df.columns for c in
               ['binds_homo_protein', 'binds_hetero_protein',
                'binds_dna', 'binds_rna']):
            mask = (df['binds_homo_protein'] | df['binds_hetero_protein']
                    | df['binds_dna'] | df['binds_rna']) == 1
            for _, row in df[mask].iterrows():
                combo = []
                if row['binds_homo_protein']   == 1: combo.append('homo')
                if row['binds_hetero_protein'] == 1: combo.append('hetero')
                if row['binds_dna']            == 1: combo.append('dna')
                if row['binds_rna']            == 1: combo.append('rna')
                combo_counts['+'.join(combo)] += 1

    print(f"\n{'='*60}\nAGGREGATED GRAPH STATISTICS\n{'='*60}")
    print(f"Total proteins: {len(dirs)}")
    print(f"Total atoms:    {total_atoms:,}\n")
    print("Atom-level flag counts:")
    for c in FLAG_COLS:
        print(f"  {c:>22}: {flag_totals[c]:>10,}  in {proteins_with_flag[c]:>5} proteins")
    if combo_counts:
        print("\nBinding type combinations (atom-level):")
        for k, v in sorted(combo_counts.items(), key=lambda x: -x[1]):
            print(f"  {k:>30}: {v:>10,}")


# ── Memory-saving streaming mode ─────────────────────────────────────────────

FLAG_TO_COL = {'homo_protein':   'binds_homo_protein',
               'hetero_protein': 'binds_hetero_protein',
               'dna':             'binds_dna',
               'rna':             'binds_rna'}


def _set_derived_flags(df):
    df['binds_protein'] = ((df['binds_homo_protein'] == 1) |
                           (df['binds_hetero_protein'] == 1)).astype(int)
    df['binds_nucleic'] = ((df['binds_dna'] == 1) |
                           (df['binds_rna'] == 1)).astype(int)
    df['binds_any']     = ((df['binds_protein'] == 1) |
                           (df['binds_nucleic'] == 1)).astype(int)


def _apply_flags_streaming(protein_id, src_dir, bsite_path, flags):
    """Update unified/{protein_id}/graph_nodes.csv with this row's bsite atoms.

    On first encounter the per-complex graph is copied as the protein's
    canonical graph (with flag columns initialized to 0). On subsequent
    encounters flags are OR'd in and bsite_area is taken as max.
    """
    src_nodes = os.path.join(src_dir, "graph_nodes.csv")
    src_links = os.path.join(src_dir, "graph_links.csv")
    if not os.path.exists(src_nodes):
        return
    out_dir   = os.path.join(OUTPUT_DIR, protein_id)
    out_nodes = os.path.join(out_dir, "graph_nodes.csv")
    out_links = os.path.join(out_dir, "graph_links.csv")

    if not os.path.exists(out_nodes):
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(src_nodes)
        for col in LEGACY_COLS:
            if col in df.columns:
                df = df.drop(columns=[col])
        for col in FLAG_COLS:
            df[col] = 0
        df['bsite_area'] = 0.0
        if os.path.exists(src_links) and not os.path.exists(out_links):
            shutil.copy2(src_links, out_links)
    else:
        df = pd.read_csv(out_nodes)

    atoms = read_bsite_atoms(bsite_path)
    if atoms:
        atom_keys = df.apply(make_atom_key, axis=1)
        in_bsite = atom_keys.isin(set(atoms.keys())).astype(int).values
        for flag in flags:
            col = FLAG_TO_COL[flag]
            df[col] = np.maximum(df[col].astype(int).values, in_bsite)
        new_areas = atom_keys.map(atoms).fillna(0.0).values
        df['bsite_area'] = np.maximum(df['bsite_area'].values, new_areas)

    _set_derived_flags(df)
    df.to_csv(out_nodes, index=False)


def run_memory_mode(df):
    """Single-threaded streaming pipeline.

    For each row: download -> generate -> extract -> contribute to aggregate,
    then delete the row's PDB, _bsite files, and _per_complex graph dirs.
    """
    if not os.path.exists(GRAPH_TOOL):
        print(f"ERROR: graph tool not found: {GRAPH_TOOL}")
        sys.exit(1)
    try:
        subprocess.run(["voronota-js-fast-iface-contacts", "--help"],
                       capture_output=True, timeout=10)
    except FileNotFoundError:
        print("ERROR: voronota-js-fast-iface-contacts not on PATH")
        sys.exit(1)

    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    os.makedirs(PER_COMPLEX, exist_ok=True)
    os.makedirs(BSITE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "chain_resolution.log")
    open(log_path, 'w').close()

    counts = defaultdict(int)

    for _, row in tqdm(list(df.iterrows()), desc="Streaming"):
        flags, kind = classify(row)
        if not kind or not flags:
            counts['skipped_no_class'] += 1
            continue
        s1, s2 = row['subunit_1'], row['subunit_2']
        url = row.get('download_url')
        base = base_name(s1, s2)
        pdb_path = complex_pdb_path(s1, s2)

        # 1. Download
        if not os.path.exists(pdb_path):
            if not isinstance(url, str) or not url:
                counts['no_url'] += 1
                continue
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                with open(pdb_path, 'wb') as f:
                    f.write(resp.content)
            except Exception as e:
                counts['download_error'] += 1
                continue

        # 2. Resolve chains + generate per-complex graph
        with open(log_path, 'a') as lw:
            chain_map = resolve_chains(pdb_path, s1, s2, kind, lw)
        actual_a, actual_b = chain_map['A'], chain_map['B']
        if actual_a is None:
            counts['no_chain'] += 1
            try: os.remove(pdb_path)
            except OSError: pass
            continue
        _clear_chain_dirs(PER_COMPLEX, base)
        gen_chains = [actual_a]
        if kind == 'pp' and actual_b is not None:
            gen_chains.append(actual_b)
        try:
            for ch in gen_chains:
                subprocess.run(
                    [GRAPH_TOOL,
                     "--input-complex", pdb_path,
                     "--output-dir", PER_COMPLEX,
                     "--chain-id", ch,
                     "--no-faspr"],
                    capture_output=True, timeout=300, check=False,
                )
        except Exception:
            counts['generate_error'] += 1
            try: os.remove(pdb_path)
            except OSError: pass
            continue

        per_a_dir = os.path.join(PER_COMPLEX, f"{base}_chain_A")
        per_b_dir = os.path.join(PER_COMPLEX, f"{base}_chain_B")
        ok_a, ok_b_actual = _rename_chains_to_positional(
            PER_COMPLEX, base, actual_a,
            actual_b if (kind == 'pp' and actual_b is not None) else None,
        )
        ok_b = ok_b_actual if (kind == 'pp' and actual_b is not None) else False

        # 3. Extract interfaces (per chain)
        bsite_a = os.path.join(BSITE_DIR, f"{base}_chain_A.tsv")
        bsite_b = os.path.join(BSITE_DIR, f"{base}_chain_B.tsv")
        if ok_a:
            try:
                subprocess.run(
                    ["voronota-js-fast-iface-contacts",
                     "--input", pdb_path, "--as-assembly",
                     "--subselect-contacts",
                     f"[-a1 [-chain {actual_a}] -a2! [-chain {actual_a}]]",
                     "--output-bsite-file", bsite_a],
                    capture_output=True, timeout=120, check=False,
                )
            except Exception:
                pass
        if ok_b:
            try:
                subprocess.run(
                    ["voronota-js-fast-iface-contacts",
                     "--input", pdb_path, "--as-assembly",
                     "--subselect-contacts",
                     f"[-a1 [-chain {actual_b}] -a2! [-chain {actual_b}]]",
                     "--output-bsite-file", bsite_b],
                    capture_output=True, timeout=120, check=False,
                )
            except Exception:
                pass

        # 4. Update aggregated graphs
        if ok_a and os.path.isdir(per_a_dir):
            _apply_flags_streaming(s1, per_a_dir, bsite_a, flags)
        if ok_b and os.path.isdir(per_b_dir):
            _apply_flags_streaming(s2, per_b_dir, bsite_b, flags)
        counts['ok'] += 1

        # Cleanup row intermediates
        for p in (pdb_path, bsite_a, bsite_b):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
        for d in (per_a_dir, per_b_dir):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    # Cleanup empty intermediate dirs
    for d in (PER_COMPLEX, BSITE_DIR):
        try: os.rmdir(d)
        except OSError: pass

    print("\nStreaming results:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


# ── Main ─────────────────────────────────────────────────────────────────────

def _protein_sample(df, ids_str):
    """Take every row where subunit_1 or subunit_2 matches any of the given IDs."""
    ids = {x.strip() for x in ids_str.split(',') if x.strip()}
    sub = df[df['subunit_1'].isin(ids) | df['subunit_2'].isin(ids)].copy()
    print(f"Protein sample for {sorted(ids)}:")
    for pid in sorted(ids):
        as_s1 = int((sub['subunit_1'] == pid).sum())
        as_s2 = int((sub['subunit_2'] == pid).sum())
        print(f"  {pid}: {as_s1} rows as subunit_1, {as_s2} rows as subunit_2")
    return sub


def _stratified_sample(df, n_per_class):
    """Take the first n rows of each interaction class for smoke testing."""
    def class_label(row):
        flags, kind = classify(row)
        if kind == 'pp':
            return 'homo' if 'homo_protein' in flags else 'hetero'
        if kind == 'peptide':
            return 'peptide'
        if kind == 'na':
            if flags == {'dna', 'rna'}: return 'hybrid'
            if flags == {'dna'}:        return 'dna'
            if flags == {'rna'}:        return 'rna'
        return None

    labels = df.apply(class_label, axis=1)
    parts = []
    print("Stratified sample:")
    for cls in ('homo', 'hetero', 'peptide', 'dna', 'rna', 'hybrid'):
        sub = df[labels == cls].head(n_per_class)
        print(f"  {cls:>8}: {len(sub)} rows")
        if len(sub):
            parts.append(sub)
    return pd.concat(parts) if parts else df.head(0).copy()


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline driven by ppi3d_filtered.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--step', required=True,
                        choices=['download', 'generate', 'extract',
                                 'build', 'stats', 'all'])
    parser.add_argument('--limit', type=int, default=None,
                        help="Process only the first N rows (smoke testing)")
    parser.add_argument('--sample', type=int, default=None,
                        help="Stratified smoke test: take the first N rows of "
                             "each interaction class (homo, hetero, peptide, "
                             "dna, rna, hybrid). Mutually exclusive with --limit.")
    parser.add_argument('--sample-protein', type=str, default=None,
                        help="Take every row containing the given protein ID(s) "
                             "as subunit_1 or subunit_2 (comma-separated for "
                             "multiple). Useful for testing cross-row aggregation.")
    parser.add_argument('--memory', action='store_true',
                        help="Memory-saving mode: single-threaded streaming "
                             "pipeline that runs steps 1->2->3->4 per complex "
                             "and deletes the PDB, _bsite, and _per_complex "
                             "artifacts after each row. Requires --step all.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.step == 'stats':
        print_statistics()
        return

    if args.memory and args.step != 'all':
        print("ERROR: --memory requires --step all")
        sys.exit(2)
    selectors = sum(x is not None for x in
                    (args.limit, args.sample, args.sample_protein))
    if selectors > 1:
        print("ERROR: --limit, --sample, --sample-protein are mutually exclusive")
        sys.exit(2)

    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    elif args.sample is not None:
        df = _stratified_sample(df, args.sample)
    elif args.sample_protein is not None:
        df = _protein_sample(df, args.sample_protein)
    print(f"  {len(df)} rows")

    if args.memory:
        print("\n" + "="*60 + "\nMEMORY-SAVING STREAMING PIPELINE\n" + "="*60)
        run_memory_mode(df)
        print("\nDone!")
        return

    if args.step in ('download', 'all'):
        print("\n" + "="*60 + "\nSTEP 1: DOWNLOAD COMPLEXES\n" + "="*60)
        download_complexes(df)
    if args.step in ('generate', 'all'):
        print("\n" + "="*60 + "\nSTEP 2: GENERATE BASE GRAPHS\n" + "="*60)
        generate_base_graphs(df)
    if args.step in ('extract', 'all'):
        print("\n" + "="*60 + "\nSTEP 3: EXTRACT INTERFACES\n" + "="*60)
        extract_all_interfaces(df)
    if args.step in ('build', 'all'):
        print("\n" + "="*60 + "\nSTEP 4: BUILD UNIFIED GRAPHS\n" + "="*60)
        build_unified_graphs(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
