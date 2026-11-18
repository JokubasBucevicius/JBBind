import pandas as pd

DATA_DIR = "/home/jokubasb/protein_protein/all_class"
PPI3D_PATH = f"{DATA_DIR}/ppi3d_test_data_full.csv"
SABDAB_PATH = f"{DATA_DIR}/sabdab_summary_all.tsv"
OUT_PATH = f"{DATA_DIR}/ppi3d_test_data_filtered.csv"

# Stems catch both "ribosome"/"ribosomal", "nucleosome"/"nucleosomal", and "rRNA"
RIBOSOME_NUCLEOSOME_TERMS = ["ribosom", "nucleosom", "rrna"]
TEXT_COLS = ["pdb_annotation", "subunit_1_title", "subunit_2_title"]


def classify_chain_nt_type(chain_seq):
    """Classify one x-delimited chain as DNA/RNA/Hybrid/Ambiguous by nucleotide composition."""
    s = chain_seq.upper()
    has_t = "T" in s
    has_u = "U" in s
    if has_t and has_u:
        return "Hybrid"
    if has_t:
        return "DNA"
    if has_u:
        return "RNA"
    return "Ambiguous"  # only A/G/C — cannot determine from sequence alone


def classify_nucleic_acid(s2_sequence, subunit_2_title):
    """
    Determine nucleic acid type from sequence chain composition and title fallback.

    Chains in s2_sequence are delimited by lowercase 'x'.
    T nucleotides indicate DNA; U nucleotides indicate RNA.

    Returns (na_type, na_strand_count, na_chain_types_str).
    """
    if pd.isna(s2_sequence) or str(s2_sequence).strip() == "":
        return "None", 0, ""

    chains = [c for c in str(s2_sequence).split("x") if c.strip()]
    if not chains:
        return "None", 0, ""

    chain_types = [classify_chain_nt_type(c) for c in chains]
    unique = set(chain_types)

    has_dna = "DNA" in unique
    has_rna = "RNA" in unique
    has_hybrid_chain = "Hybrid" in unique

    if has_hybrid_chain or (has_dna and has_rna):
        na_type = "Hybrid"
    elif has_dna:
        na_type = "DNA"
    elif has_rna:
        na_type = "RNA"
    else:
        # All chains are ambiguous (only A/G/C) — fall back to title keywords
        title_up = str(subunit_2_title).upper()
        title_has_dna = "DNA" in title_up
        title_has_rna = "RNA" in title_up
        if title_has_dna and title_has_rna:
            na_type = "Hybrid"
        elif title_has_dna:
            na_type = "DNA"
        elif title_has_rna:
            na_type = "RNA"
        else:
            na_type = "Ambiguous"

    return na_type, len(chains), ",".join(chain_types)


print("Loading data...")
ppi = pd.read_csv(PPI3D_PATH)
sabdab = pd.read_csv(SABDAB_PATH, sep="\t")

print(f"  ppi3d_full rows: {len(ppi):,}")
print(f"  SAbDab rows:     {len(sabdab):,}")

# --- Filter 1: remove antibody complexes present in SAbDab ---
sabdab_pdbs = set(sabdab["pdb"].str.lower().dropna())
mask_antibody = ppi["pdb_id"].str.lower().isin(sabdab_pdbs)
n_antibody = mask_antibody.sum()
ppi = ppi[~mask_antibody].copy()
print(f"\n[1] Removed {n_antibody:,} antibody rows (SAbDab match) -> {len(ppi):,} remain")

# --- Filter 2: remove complexes with mismatched taxonomy IDs ---
s1 = pd.to_numeric(ppi["s1_taxonomy_id"], errors="coerce")
s2 = pd.to_numeric(ppi["s2_taxonomy_id"], errors="coerce")

both_valid = s1.notna() & s2.notna() & (s1 != 0) & (s2 != 0)
mask_taxon_mismatch = both_valid & (s1 != s2)
n_taxon = mask_taxon_mismatch.sum()
ppi = ppi[~mask_taxon_mismatch].copy()
print(f"[2] Removed {n_taxon:,} taxonomy-mismatch rows  -> {len(ppi):,} remain")

# --- Filter 3: remove ribosomes, ribosomal proteins, nucleosomes, and rRNA ---
combined_text = ppi[TEXT_COLS].fillna("").agg(" ".join, axis=1).str.lower()
pattern = "|".join(RIBOSOME_NUCLEOSOME_TERMS)
mask_ribo_nucl = combined_text.str.contains(pattern, regex=True)
n_ribo_nucl = mask_ribo_nucl.sum()
ppi = ppi[~mask_ribo_nucl].copy()
print(f"[3] Removed {n_ribo_nucl:,} ribosome/ribosomal protein/nucleosome/rRNA rows -> {len(ppi):,} remain")

# --- Add nucleic acid classification columns ---
print("\nClassifying nucleic acid interactions...")
nucleic_mask = ppi["protein_nucleic_interaction"] == 1
nucleic_rows = ppi[nucleic_mask]
n_nucleic = nucleic_mask.sum()
print(f"  Nucleic interaction rows: {n_nucleic:,}")

results = nucleic_rows.apply(
    lambda row: classify_nucleic_acid(row["s2_sequence"], row["subunit_2_title"]),
    axis=1,
)
na_types, na_counts, na_chain_types_list = zip(*results) if n_nucleic > 0 else ([], [], [])

ppi["na_type"] = "None"
ppi["na_strand_count"] = 0
ppi["na_chain_types"] = ""

if n_nucleic > 0:
    ppi.loc[nucleic_mask, "na_type"] = list(na_types)
    ppi.loc[nucleic_mask, "na_strand_count"] = list(na_counts)
    ppi.loc[nucleic_mask, "na_chain_types"] = list(na_chain_types_list)

na_dist = ppi.loc[nucleic_mask, "na_type"].value_counts()
print("\nNucleic acid type distribution:")
for t, c in na_dist.items():
    print(f"  {t}: {c:,}")

# --- Save ---
ppi.to_csv(OUT_PATH, index=False)
print(f"\nSaved filtered dataset to: {OUT_PATH}")
print(f"Final row count: {len(ppi):,}")
