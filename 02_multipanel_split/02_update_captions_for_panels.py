#!/usr/bin/env python3
"""
STEP 2: Update captions.csv (or captions.xlsx)
Based on split_metadata.json, add captions for every panel produced by
01_detect_and_split_panels.py.

Panel labels arriving here are already normalized to a single uppercase
letter (or a PANEL<N> placeholder flagged NEEDS_REVIEW) by STEP1 - this
script does not need to do any further label cleanup.

Usage:
    python3 02_update_captions_for_panels.py <pmid> --base /path/to/extracted_images
    (--base defaults to the AUTOPK_IMAGES_DIR environment variable if set,
     otherwise the current directory)
"""

import os
import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import re

DEFAULT_IMAGES_DIR = os.environ.get("AUTOPK_IMAGES_DIR", ".")


class CaptionsUpdater:
    def __init__(self, pmid_folder_path):
        self.pmid_folder = Path(pmid_folder_path)
        self.pmid = self.pmid_folder.name

        if not self.pmid_folder.exists():
            print(f"ERROR: path does not exist: {pmid_folder_path}")
            sys.exit(1)

        self.metadata_file = self.pmid_folder / 'split_metadata.json'

        if not self.metadata_file.exists():
            print(f"ERROR: split_metadata.json not found")
            print(f"   Run first: python3 01_detect_and_split_panels.py {self.pmid}")
            sys.exit(1)

        # auto-detect captions file format (xlsx first, then csv)
        self.captions_file = None
        self.file_format = None

        xlsx_file = self.pmid_folder / 'captions.xlsx'
        csv_file = self.pmid_folder / 'captions.csv'

        if xlsx_file.exists():
            self.captions_file = xlsx_file
            self.file_format = 'xlsx'
        elif csv_file.exists():
            self.captions_file = csv_file
            self.file_format = 'csv'
        else:
            print(f"ERROR: neither captions.csv nor captions.xlsx found")
            sys.exit(1)

        print(f"Detected format: {self.file_format.upper()}")

    def read_captions(self):
        if self.file_format == 'xlsx':
            return pd.read_excel(self.captions_file)
        else:
            return pd.read_csv(self.captions_file)

    def save_captions(self, df):
        if self.file_format == 'xlsx':
            df.to_excel(self.captions_file, index=False)
        else:
            df.to_csv(self.captions_file, index=False)

    def run(self):
        print("\n" + "=" * 80)
        print(f"STEP 2: Update captions file - PMID {self.pmid}")
        print("=" * 80)

        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        df = self.read_captions()

        print(f"\nCurrent captions file has {len(df)} row(s)")

        new_rows = []
        n_needs_review = 0

        for base_name, image_info in metadata['processed_images'].items():
            if image_info['status'] != 'multipanel_split':
                continue

            original_file = image_info['original_file']
            original_caption = None

            matching_rows = df[df['image_filename'].str.contains(base_name, na=False)]
            if not matching_rows.empty:
                original_caption = matching_rows.iloc[0]['caption']

            if original_caption is None:
                matching_rows = df[df['image_filename'] == original_file]
                if not matching_rows.empty:
                    original_caption = matching_rows.iloc[0]['caption']

            if original_caption is None:
                print(f"WARNING: could not find original caption for {original_file}, using empty value")
                original_caption = ""

            for label, panel_info in image_info['panels'].items():
                output_file = panel_info['output_file']
                panel_title = panel_info.get('title', '')
                extraction_method = panel_info.get('extraction_method', 'unknown')
                confidence = panel_info.get('confidence', 0)
                label_status = panel_info.get('label_status', 'OK')

                if label_status == 'NEEDS_REVIEW':
                    n_needs_review += 1

                if panel_title:
                    fig_match = re.search(r'Figure\s+(\d+|[A-Z])', original_caption)
                    fig_num = fig_match.group(1) if fig_match else ""

                    if fig_num:
                        new_caption = f"Figure {fig_num} Panel {label}: {panel_title}. {original_caption}"
                    else:
                        new_caption = f"Panel {label}: {panel_title}. {original_caption}"
                else:
                    new_caption = original_caption

                new_rows.append({
                    'image_filename': output_file,
                    'caption': new_caption
                })

                flag = " [NEEDS_REVIEW: messy label from model]" if label_status == 'NEEDS_REVIEW' else ""
                print(f"  {output_file}{flag}")
                print(f"     extraction method: {extraction_method} (confidence: {confidence:.2f})")
                print(f"     title: {panel_title[:60]}...")

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df_updated = pd.concat([df, new_df], ignore_index=True)

            self.save_captions(df_updated)

            print("\n" + "=" * 80)
            print(f"STEP 2 complete")
            print(f"Captions file updated ({self.file_format.upper()})")
            print(f"   Before: {len(df)} row(s)")
            print(f"   After:  {len(df_updated)} row(s)")
            print(f"   Added:  {len(new_rows)} row(s)")
            if n_needs_review:
                print(f"   WARNING: {n_needs_review} row(s) came from a panel with a "
                      f"messy label (NEEDS_REVIEW) - check split_metadata.json")
            print("=" * 80)
            print(f"\nSaved: {self.captions_file.name}\n")
        else:
            print("\nNo multipanel figures found that needed updating")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("pmid", help="PMID subfolder to process, e.g. 18573931")
    ap.add_argument("--base", default=DEFAULT_IMAGES_DIR,
                     help="Path to the extracted_images root folder "
                          "(default: $AUTOPK_IMAGES_DIR env var, or current directory)")
    args = ap.parse_args()

    pmid_folder = Path(args.base) / args.pmid

    updater = CaptionsUpdater(pmid_folder)
    updater.run()
