import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

DATA_PATH = "/home/jokubas/Magistras/duomenys/all_class/ppi3d_filtered.csv"
OUT_PATH  = "/home/jokubas/Magistras/duomenys/all_class/eda_ppi3d.png"

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows")

HAS_NA_COLS = "na_type" in df.columns and "na_strand_count" in df.columns

# ── Classify interaction type ─────────────────────────────────────────────────
def classify(row):
    if row["protein_protein_interaction"] == 1:
        return "PP Homo" if row["homo"] == 1 else "PP Hetero"
    if row["protein_peptide_interaction"] == 1:
        return "Protein-Peptide"
    if row["protein_nucleic_interaction"] == 1:
        if HAS_NA_COLS:
            na_type = str(row["na_type"]).strip()
            if na_type == "Hybrid":
                return "Protein-Hybrid"
            if na_type == "DNA":
                return "Protein-DNA"
            if na_type == "RNA":
                return "Protein-RNA"
        # Fallback: title keywords when na_type is Ambiguous/None or column missing
        title = str(row["subunit_2_title"]).upper()
        if "RNA" in title and "DNA" in title:
            return "Protein-Hybrid"
        if "RNA" in title:
            return "Protein-RNA"
        if "DNA" in title:
            return "Protein-DNA"
        return "Protein-Nucleic"
    return "Other"

df["interaction_class"] = df.apply(classify, axis=1)
print("\nClass distribution:")
print(df["interaction_class"].value_counts())

CLASS_ORDER  = ["PP Homo", "PP Hetero", "Protein-Peptide",
                "Protein-DNA", "Protein-RNA", "Protein-Hybrid"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868",
                "#C44E52", "#8172B2", "#937860"]

# ── Interface residue percentage ──────────────────────────────────────────────
# Each interface residue buries ~65 Å² on average (Janin et al.).
# Dividing the interface area by this constant gives an estimated count of
# S1 residues at the interface; dividing by total S1 residues gives %.
AVG_BURIED_AREA_PER_RES = 65  # Å²
df["pct_interface_res_s1"] = (df["area"] / AVG_BURIED_AREA_PER_RES) / df["s1_number_of_residues"] * 100

CLUSTER_COL = "s1_sequence_cluster_95"

# ── Overlap Analysis 1: subunits in both homo & hetero PP interactions ────────
pp_df = df[df["protein_protein_interaction"] == 1].dropna(subset=[CLUSTER_COL]).copy()

pp_agg = pp_df.groupby(CLUSTER_COL)["homo"].agg(
    has_homo=lambda x: (x == 1).any(),
    has_hetero=lambda x: (x == 0).any(),
)
pp_homo_only   = int((pp_agg["has_homo"]   & ~pp_agg["has_hetero"]).sum())
pp_hetero_only = int((~pp_agg["has_homo"]  &  pp_agg["has_hetero"]).sum())
pp_both        = int((pp_agg["has_homo"]   &  pp_agg["has_hetero"]).sum())
print(f"\nPP clusters — homo-only: {pp_homo_only:,} | both: {pp_both:,} | hetero-only: {pp_hetero_only:,}")

# ── Overlap Analysis 2: subunits binding both DNA and RNA ─────────────────────
na_df = df[df["protein_nucleic_interaction"] == 1].dropna(subset=[CLUSTER_COL]).copy()

if HAS_NA_COLS:
    na_df["binds_dna"] = na_df["na_type"].isin(["DNA", "Hybrid"])
    na_df["binds_rna"] = na_df["na_type"].isin(["RNA", "Hybrid"])
else:
    title_up = na_df["subunit_2_title"].str.upper().fillna("")
    na_df["binds_dna"] = title_up.str.contains("DNA")
    na_df["binds_rna"] = title_up.str.contains("RNA")

na_agg = na_df.groupby(CLUSTER_COL).agg(
    has_dna=("binds_dna", "any"),
    has_rna=("binds_rna", "any"),
)
na_dna_only = int((na_agg["has_dna"]  & ~na_agg["has_rna"]).sum())
na_rna_only = int((~na_agg["has_dna"] &  na_agg["has_rna"]).sum())
na_both     = int((na_agg["has_dna"]  &  na_agg["has_rna"]).sum())
print(f"NA clusters  — DNA-only: {na_dna_only:,} | both: {na_both:,} | RNA-only: {na_rna_only:,}")

# ── Overlap Analysis 3: subunits binding both proteins and nucleic acids ──────
pp_clusters = set(pp_df[CLUSTER_COL].unique())
na_clusters = set(na_df[CLUSTER_COL].unique())
pp_na_both  = int(len(pp_clusters & na_clusters))
pp_na_pp_only = int(len(pp_clusters - na_clusters))
pp_na_na_only = int(len(na_clusters - pp_clusters))
print(f"PP+NA clusters — PP-only: {pp_na_pp_only:,} | both: {pp_na_both:,} | NA-only: {pp_na_na_only:,}")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
               height_ratios=[1.1, 1.0, 0.85])

ax_class   = fig.add_subplot(gs[0, :])
ax_hexbin  = fig.add_subplot(gs[1, 0])
ax_hist    = fig.add_subplot(gs[1, 1])
ax_overlap = fig.add_subplot(gs[2, :])

# ── Plot 1: Class distribution ────────────────────────────────────────────────
counts = df["interaction_class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
bars = ax_class.bar(counts.index, counts.values, color=CLASS_COLORS,
                    edgecolor="white", linewidth=0.8, width=0.55)

for bar, val in zip(bars, counts.values):
    if val == 0:
        continue
    pct = val / len(df) * 100
    ax_class.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + counts.max() * 0.01,
        f"{val:,}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax_class.set_title("Interaction Class Distribution", fontsize=14, fontweight="bold", pad=12)
ax_class.set_ylabel("Number of complexes", fontsize=11)
ax_class.set_xlabel("Interaction class", fontsize=11)
ax_class.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_class.set_ylim(0, counts.max() * 1.22)
ax_class.spines[["top", "right"]].set_visible(False)
ax_class.tick_params(axis="x", labelsize=10)

# ── Plot 2: % interface residues vs subunit-1 size (hexbin) ──────────────────
plot_df = df[df["s1_number_of_residues"].between(1, 2000)].copy()
plot_df = plot_df[plot_df["pct_interface_res_s1"].between(0, 100)].copy()

hb = ax_hexbin.hexbin(
    plot_df["s1_number_of_residues"],
    plot_df["pct_interface_res_s1"],
    gridsize=50, cmap="YlOrRd", mincnt=1, bins="log", linewidths=0.2,
)
fig.colorbar(hb, ax=ax_hexbin, label="log10(count)")
ax_hexbin.set_title("Interface Residues (% of S1)\nvs Subunit-1 Size",
                    fontsize=12, fontweight="bold")
ax_hexbin.set_xlabel("Subunit-1 size (residues)", fontsize=11)
ax_hexbin.set_ylabel("Est. % of S1 residues at interface", fontsize=11)
ax_hexbin.spines[["top", "right"]].set_visible(False)
ax_hexbin.set_xlim(0, 2000)
ax_hexbin.set_ylim(0, 100)

# ── Plot 3: Subunit-1 size distribution ──────────────────────────────────────
s1_sizes = df["s1_number_of_residues"].dropna()
s1_clip  = s1_sizes.clip(upper=2000)

ax_hist.hist(s1_clip, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.9)
p25, p50, p75 = s1_sizes.quantile([0.25, 0.5, 0.75])
for val, lbl, ls in [(p25, f"Q1={p25:.0f}", "--"),
                      (p50, f"Median={p50:.0f}", "-"),
                      (p75, f"Q3={p75:.0f}", "--")]:
    ax_hist.axvline(min(val, 2000), color="#C44E52", linestyle=ls, linewidth=1.5, label=lbl)

ax_hist.legend(fontsize=9)
ax_hist.set_title("Subunit-1 Size Distribution\n(clipped at 2,000 residues)",
                  fontsize=12, fontweight="bold")
ax_hist.set_xlabel("Number of residues", fontsize=11)
ax_hist.set_ylabel("Count", fontsize=11)
ax_hist.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_hist.spines[["top", "right"]].set_visible(False)
mean_s1 = s1_sizes.mean()
ax_hist.text(
    0.97, 0.93,
    f"n = {len(s1_sizes):,}\nmean = {mean_s1:.0f}\nmedian = {p50:.0f}",
    transform=ax_hist.transAxes, ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
)

# ── Plot 4: Overlap summary — three groups of 3 bars ─────────────────────────
pp_vals  = [pp_homo_only,    pp_both,      pp_hetero_only]
mid_vals = [pp_na_pp_only,   pp_na_both,   pp_na_na_only]
na_vals  = [na_dna_only,     na_both,      na_rna_only]

pp_lbls  = ["Homo\nonly",  "Homo &\nHetero",    "Hetero\nonly"]
mid_lbls = ["PP\nonly",    "PP &\nNucleic",      "Nucleic\nonly"]
na_lbls  = ["DNA\nonly",   "DNA &\nRNA",         "RNA\nonly"]

pp_cols  = ["#4C72B0", "#7B9ED9", "#DD8452"]
mid_cols = ["#4C72B0", "#2a9d8f", "#8172B2"]
na_cols  = ["#C44E52", "#A55FAE", "#8172B2"]

gap = 1.4
x_pp  = np.array([0.0, 0.8, 1.6])
x_mid = x_pp  + 1.6 + gap
x_na  = x_mid + 1.6 + gap

all_x    = np.concatenate([x_pp, x_mid, x_na])
all_vals = pp_vals + mid_vals + na_vals
all_cols = pp_cols + mid_cols + na_cols
all_lbls = pp_lbls + mid_lbls + na_lbls

bars_ov = ax_overlap.bar(all_x, all_vals, color=all_cols,
                          edgecolor="white", linewidth=0.8, width=0.65)

for bar, val in zip(bars_ov, all_vals):
    ax_overlap.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(all_vals) * 0.015,
        f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax_overlap.set_xticks(all_x)
ax_overlap.set_xticklabels(all_lbls, fontsize=10)
ax_overlap.set_ylabel("Number of sequence clusters (95% id)", fontsize=10)
ax_overlap.set_title("Category Overlap: Proteins Appearing in Multiple Interaction Types",
                     fontsize=13, fontweight="bold", pad=10)
ax_overlap.set_ylim(0, max(all_vals) * 1.18)
ax_overlap.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_overlap.spines[["top", "right"]].set_visible(False)

# Group labels below the tick labels
y_grp = -max(all_vals) * 0.13
for mid_x, label in [(x_pp.mean(),  "Protein–Protein\nInteractions"),
                      (x_mid.mean(), "PP & Nucleic\nOverlap"),
                      (x_na.mean(),  "Nucleic Acid\nInteractions")]:
    ax_overlap.text(mid_x, y_grp, label, ha="center", va="top",
                    fontsize=11, fontweight="bold", color="#333333")

# Vertical separators between groups
for sep_x in [(x_pp[-1] + x_mid[0]) / 2, (x_mid[-1] + x_na[0]) / 2]:
    ax_overlap.axvline(sep_x, color="#cccccc", linewidth=1.2, linestyle="--")

# ── Save ──────────────────────────────────────────────────────────────────────
fig.suptitle("PPI3D Filtered Dataset — Exploratory Data Analysis",
             fontsize=15, fontweight="bold", y=1.005)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved figure to: {OUT_PATH}")
