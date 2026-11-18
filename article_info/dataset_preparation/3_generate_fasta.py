import pandas as pd
from pathlib import Path

# Load CSV file
csv_file =  'all_class/ppi3d_test_data_filtered.csv'

# Read CSV into DataFrame
df = pd.read_csv(csv_file)
original_count = len(df)
print(f"Loaded {original_count} rows")

# Collect unique subunit_1 and subunit_2 sequences (deduplicated by chain ID)
seen = {}
for _, row in df.iterrows():
    if row['subunit_1'] not in seen:
        seen[row['subunit_1']] = row['s1_sequence'].upper()
    if row['subunit_2'] not in seen:
        seen[row['subunit_2']] = row['s2_sequence'].upper()

print(f"Unique subunit_1 + subunit_2 chains: {len(seen)}")

# Output FASTA file
fasta_file = 'ppi3d_test_filtered_subunits.fasta'

with open(fasta_file, 'w') as fasta_out:
    for subunit, sequence in seen.items():
        fasta_out.write(f'>{subunit}\n')
        fasta_out.write(f'{sequence}\n')

print(f"Written to: {fasta_file}")
print(f"Total sequences: {len(seen)}")