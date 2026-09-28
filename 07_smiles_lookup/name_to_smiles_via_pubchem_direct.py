#!/usr/bin/env python3
"""
Drug name -> SMILES via a direct PubChem name lookup
-------------------------------------------------------
Batch script: reads a metadata CSV's drug_name column, filters out
combination-drug rows, queries PubChem's name-lookup REST endpoint
directly for each unique single drug name, and writes a new CSV with a
'smiles' column added.

This is the simpler, more direct counterpart to
name_to_smiles_via_node_normalizer.py: it queries PubChem's name endpoint
literally, with no synonym/name-resolution step, so it will miss drugs
whose name in the metadata doesn't exactly match how PubChem indexes them
(abbreviations, alternate salt forms, etc.). This is the version that has
actually been run against the full dataset.

Handles the fact that PubChem's JSON response doesn't always use the same
field name for the SMILES property by checking several possible field
names.

Usage:
    python name_to_smiles_via_pubchem_direct.py --base-dir /path/to/AutoPK
"""

import argparse
import os
import time
from typing import Optional

import pandas as pd
import requests

DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")


def get_smiles_robust(drug_name: str) -> Optional[str]:
    """
    A more robust SMILES-fetching function that tries several possible
    field names in PubChem's response.
    """
    drug_name = str(drug_name).strip()

    # PubChem REST API
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    endpoint = f"{base_url}/compound/name/{drug_name}/property/IsomericSMILES/JSON"

    try:
        response = requests.get(endpoint, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # get the properties dict
            properties = data["PropertyTable"]["Properties"][0]

            # try all possible SMILES field names
            possible_fields = [
                "IsomericSMILES",      # standard field name
                "SMILES",               # alternate field name
                "CanonicalSMILES",     # another format
                "smiles",               # lowercase
                "isomeric_smiles",     # underscore format
            ]

            for field in possible_fields:
                if field in properties:
                    return properties[field]

            # if none matched, print the available fields
            print(f"    Available fields: {list(properties.keys())}")
            return None
        else:
            return None

    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK root dir (default: $AUTOPK_BASE_DIR or '.')")
    ap.add_argument("--input-file", default=None,
                     help="Input metadata CSV (default: <base-dir>/pk_metadata_figure_level.csv)")
    ap.add_argument("--output-file", default=None,
                     help="Output CSV with the smiles column added "
                          "(default: <base-dir>/pk_metadata_with_smiles_fixed.csv)")
    ap.add_argument("--column-name", default="drug_name",
                     help="Column in the input CSV holding the drug name (default: drug_name)")
    ap.add_argument("--delay", type=float, default=0.3,
                     help="Seconds to sleep between PubChem requests (default: 0.3)")
    ap.add_argument("--yes", action="store_true",
                     help="Skip the interactive y/n confirmation prompt before starting")
    args = ap.parse_args()

    base_dir = args.base_dir
    input_file = args.input_file or os.path.join(base_dir, "pk_metadata_figure_level.csv")
    output_file = args.output_file or os.path.join(base_dir, "pk_metadata_with_smiles_fixed.csv")
    column_name = args.column_name
    delay = args.delay

    print("=" * 70)
    print("Drug Name to SMILES (direct PubChem lookup)")
    print("=" * 70)
    print()
    print("Automatically detects whichever SMILES field name PubChem returns.")
    print()

    # read CSV
    print(f"Reading: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"Read {len(df)} rows")
    except Exception as e:
        print(f"Failed: {e}")
        return

    # check the column exists
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found")
        return

    # get unique drugs, filtering out combination drugs
    unique_drugs = df[column_name].dropna().unique()
    single_drugs = []

    for drug in unique_drugs:
        drug_str = str(drug).strip()
        if ',' not in drug_str and drug_str.lower() not in ['unknown', '']:
            single_drugs.append(drug_str)

    print(f"Unique drugs: {len(unique_drugs)}")
    print(f"Single drugs: {len(single_drugs)}")
    print(f"Estimated time: ~{len(single_drugs) * delay:.0f}s")
    print()

    if not args.yes:
        response = input("Continue? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled")
            return

    print()
    print("Starting processing...")
    print("=" * 70)

    # build the mapping
    drug_to_smiles = {}

    for i, drug_name in enumerate(single_drugs, 1):
        print(f"\n[{i}/{len(single_drugs)}] {drug_name}")

        smiles = get_smiles_robust(drug_name)

        if smiles:
            drug_to_smiles[drug_name] = smiles
            print(f"  Found: {smiles[:60]}...")
        else:
            drug_to_smiles[drug_name] = None
            print("  Not found")

        if i < len(single_drugs):
            time.sleep(delay)

    # apply to the DataFrame
    print()
    print("Generating output file...")
    df['smiles'] = df[column_name].map(drug_to_smiles)

    # save
    try:
        df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
    except Exception as e:
        print(f"Save failed: {e}")
        return

    # stats
    found = df['smiles'].notna().sum()
    total = len(df)
    unique_found = df[df['smiles'].notna()][column_name].nunique()

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"SMILES found: {found}/{total} rows ({found/total*100:.1f}%)")
    print(f"Drugs succeeded: {unique_found}/{len(single_drugs)}")
    print()

    # show successes
    if unique_found > 0:
        print("Succeeded:")
        success = df[df['smiles'].notna()][[column_name, 'smiles']].drop_duplicates(column_name)
        for idx, row in list(success.iterrows())[:15]:
            smiles_preview = row['smiles'][:50] + '...' if len(row['smiles']) > 50 else row['smiles']
            print(f"  - {row[column_name]}")
            print(f"    {smiles_preview}")

    # show failures
    failed_count = len(single_drugs) - unique_found
    if failed_count > 0:
        print()
        print(f"Not found ({failed_count}):")
        failed = df[df['smiles'].isna()][column_name].unique()
        for drug in [d for d in failed if pd.notna(d) and ',' not in str(d)][:10]:
            print(f"  - {drug}")
        if failed_count > 10:
            print(f"  ... and {failed_count - 10} more")

    print()
    print(f"Output: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
