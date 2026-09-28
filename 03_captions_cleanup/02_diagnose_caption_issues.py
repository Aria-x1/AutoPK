#!/usr/bin/env python3
"""
Diagnose caption issues across all PMID folders in extracted_images/.

Scans every PMID folder's captions.csv (or captions.xlsx) and flags three
kinds of problems, all needing a human look:

  1. orphan_caption   - caption record exists, but the image file it points
                         to no longer exists on disk (e.g. deleted after
                         caption extraction ran)
  2. missing_caption  - image file exists on disk, but there is no caption
                         record for it at all
  3. no_caption_found - caption record exists, but its content is literally
                         "No caption found" (Claude could not match a
                         caption to this image during extraction)

Output: CAPTION_ISSUES.csv in the base directory, with columns:
    pmid, image_filename, issue_type, detail

Usage:
    python3 02_diagnose_caption_issues.py --base /path/to/extracted_images
"""

import argparse
import pandas as pd
from pathlib import Path


def load_captions(pmid_folder: Path):
    """Return (dataframe, format_label) or (None, None) if no captions file."""
    csv_path = pmid_folder / "captions.csv"
    xlsx_path = pmid_folder / "captions.xlsx"

    if csv_path.exists():
        try:
            return pd.read_csv(csv_path), "csv"
        except Exception as e:
            print(f"  [ERROR] failed to read {csv_path}: {e}")
            return None, None
    elif xlsx_path.exists():
        try:
            return pd.read_excel(xlsx_path), "xlsx"
        except Exception as e:
            print(f"  [ERROR] failed to read {xlsx_path}: {e}")
            return None, None
    else:
        return None, None


def get_images_on_disk(pmid_folder: Path):
    """Images can live directly in the pmid folder or in a figures/ subfolder."""
    images = set()

    figures_subfolder = pmid_folder / "figures"
    if figures_subfolder.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            images.update(f.name for f in figures_subfolder.glob(ext))

    if not images:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            images.update(f.name for f in pmid_folder.glob(ext))

    return images


def diagnose(base_path: str):
    base = Path(base_path)

    if not base.exists():
        print(f"[ERROR] path does not exist: {base_path}")
        return

    pmid_folders = sorted([d for d in base.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"Found {len(pmid_folders)} PMID folders under {base}\n")

    issues = []
    stats = {
        "total_pmids": len(pmid_folders),
        "no_captions_file": 0,
        "orphan_caption": 0,
        "missing_caption": 0,
        "no_caption_found": 0,
    }

    for i, pmid_folder in enumerate(pmid_folders, 1):
        pmid = pmid_folder.name

        df, fmt = load_captions(pmid_folder)
        if df is None:
            print(f"[{i}/{len(pmid_folders)}] {pmid}: no captions file found")
            stats["no_captions_file"] += 1
            continue

        if "image_filename" not in df.columns or "caption" not in df.columns:
            print(f"[{i}/{len(pmid_folders)}] {pmid}: captions file missing expected columns")
            continue

        images_on_disk = get_images_on_disk(pmid_folder)
        caption_records = df["image_filename"].astype(str).tolist()

        # 1. orphan_caption: record exists, image file does not
        orphans = [f for f in caption_records if f not in images_on_disk]
        for f in orphans:
            issues.append({
                "pmid": pmid,
                "image_filename": f,
                "issue_type": "orphan_caption",
                "detail": "caption record exists but image file not found on disk",
            })
        stats["orphan_caption"] += len(orphans)

        # 2. missing_caption: image exists, no record
        missing = [f for f in images_on_disk if f not in caption_records]
        for f in missing:
            issues.append({
                "pmid": pmid,
                "image_filename": f,
                "issue_type": "missing_caption",
                "detail": "image file exists but no caption record found",
            })
        stats["missing_caption"] += len(missing)

        # 3. no_caption_found: record exists but content says so
        no_cap_mask = df["caption"].astype(str).str.contains(
            "No caption found", case=False, na=False
        )
        for f in df.loc[no_cap_mask, "image_filename"].astype(str).tolist():
            issues.append({
                "pmid": pmid,
                "image_filename": f,
                "issue_type": "no_caption_found",
                "detail": "caption extraction returned 'No caption found'",
            })
        stats["no_caption_found"] += int(no_cap_mask.sum())

        n_problems = len(orphans) + len(missing) + int(no_cap_mask.sum())
        status = "OK" if n_problems == 0 else f"{n_problems} issue(s)"
        print(f"[{i}/{len(pmid_folders)}] {pmid} ({fmt}): {status}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total PMID folders scanned:      {stats['total_pmids']}")
    print(f"Folders with no captions file:   {stats['no_captions_file']}")
    print(f"orphan_caption issues:           {stats['orphan_caption']}")
    print(f"missing_caption issues:          {stats['missing_caption']}")
    print(f"no_caption_found issues:         {stats['no_caption_found']}")
    print(f"Total issues:                    {len(issues)}")

    if issues:
        out_df = pd.DataFrame(issues)
        out_path = base / "CAPTION_ISSUES.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\nIssue list saved to: {out_path}")
    else:
        print("\nNo issues found.")

    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to extracted_images folder")
    args = ap.parse_args()
    diagnose(args.base)
