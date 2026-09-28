#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AutoPK - LLM fallback review script
================================================================================
For the rows in curve_manifest.csv that rule-based matching couldn't resolve
(UNMATCHED / NEEDS_REVIEW), group them by source figure (all unresolved
curves from the same figure are asked about together), and send the caption
+ metadata fields + each curve's original column header to Claude Haiku, so
it can judge the Drug / Analyte Name / Administration Value+Unit / Species /
Matrix for each curve.

Two exclusion lists (curated by hand):
  SKIP_PMIDS       - confirmed problematic, not worth chasing -- rows for
                     these PMIDs are skipped entirely, never sent to the LLM
  REEXTRACT_PMIDS  - needs re-digitizing / re-exporting -- also not sent to
                     the LLM this round (the underlying data is suspect;
                     wait until it's re-extracted, otherwise we'd be
                     processing bad data)

Output:
  curve_manifest_llm_reviewed.csv  - original manifest plus a set of llm_*
                                      columns
  llm_review_raw_responses.jsonl   - raw request/response for every call, so
                                      you can spot-check or reproduce results

Install dependencies:
  pip install anthropic --break-system-packages
Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 02_llm_review_unmatched_curves.py --base-dir /path/to/AutoPK
================================================================================
"""

import os
import re
import json
import time
import argparse
import pandas as pd

try:
    from anthropic import Anthropic
except ImportError:
    raise SystemExit("Install the dependency first: pip install anthropic --break-system-packages")


# ==============================================================================
# Configuration - paths are resolved from --base-dir / AUTOPK_BASE_DIR, or
# can be overridden individually
# ==============================================================================

DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")

MODEL = "claude-haiku-4-5-20251001"
BASE_MAX_TOKENS = 512
PER_CURVE_TOKENS = 180   # roughly how many output tokens one curve needs (fields + notes)
MAX_TOKENS_CAP = 8192

# ------------------------------------------------------------------------------
# Hand-curated exclusion lists
# ------------------------------------------------------------------------------
SKIP_PMIDS = {
    "18197124", "19632955", "21283017", "21896913", "29746648", "30643165",
    "32065631", "33547874", "35599363", "35815666", "37242634",
}
REEXTRACT_PMIDS = {
    "24615728", "25390686", "26605514", "27497648", "33753329", "34450632",
}


# ==============================================================================
# Helpers
# ==============================================================================

def needs_review(row):
    return (row["match_status"] in ("UNMATCHED", "NEEDS_REVIEW")
            or row["panel_analyte_status"] == "NEEDS_REVIEW")


def extract_orig_col(add_props):
    """Pull the original column header out of Additional Properties, e.g.
    'orig_col=600 mg; panel_extra=...' -> '600 mg'"""
    m = re.search(r"orig_col=([^;]*)", str(add_props))
    return m.group(1).strip() if m else ""


def build_prompt(caption, meta_row, curves):
    """
    curves: list of dict {row_index, orig_col, curve_label, source_panel}
    Returns the user message text to send to the model.
    """
    lines = []
    lines.append("You are helping me clean up a pharmacokinetic (PK) dataset. "
                  "Below is information about one figure and the curves it contains.")
    lines.append("Please determine the dosing information that each curve corresponds to.")
    lines.append("")
    lines.append(f"Figure caption: {caption}")
    lines.append(f"Metadata drug_name (may be a combination of multiple drugs): {meta_row.get('drug_name','')}")
    lines.append(f"Metadata dose (may be a combination of multiple doses): {meta_row.get('dose','')}")
    lines.append(f"Metadata species: {meta_row.get('species','')}")
    lines.append(f"Metadata matrix (may be a combination of multiple matrices): {meta_row.get('matrix','')}")
    lines.append(f"Metadata route: {meta_row.get('route_of_administration','')}")
    lines.append("")
    lines.append("The curves in this figure that need your judgment (original digitization "
                 "column header / panel marker):")
    for c in curves:
        panel_val = c["source_panel"]
        has_panel = isinstance(panel_val, str) and panel_val.strip() and panel_val.lower() != "nan"
        panel_info = f" [panel={panel_val}]" if has_panel else ""
        lines.append(f"  - id={c['row_index']}: original column header = \"{c['orig_col']}\"{panel_info}")
    lines.append("")
    lines.append("For each curve, please determine:")
    lines.append("  - drug: the actual compound that was administered (if it's a prodrug, "
                 "give the prodrug name, e.g. fosamprenavir rather than amprenavir)")
    lines.append("  - analyte: the compound actually measured in the biological sample")
    lines.append("  - dose: the dose + unit for this curve (e.g. '300 mg'), or an empty "
                 "string if not applicable")
    lines.append("  - species: the species for this curve; use the metadata value if there is "
                 "no curve-specific information")
    lines.append("  - matrix: the matrix for this curve (e.g. plasma / serum / PBMC / vaginal "
                 "tissue); if the column header itself specifies the matrix (e.g. names a "
                 "tissue), use that instead of copying the generic metadata value")
    lines.append("  - confidence: a number between 0 and 1 indicating how confident you are "
                 "in this entire row's judgment")
    lines.append("  - notes: note any uncertainty here, or if the column header itself gives "
                 "no clue to its actual meaning (e.g. it's a shape/color description, a "
                 "statistic name, or an abbreviation you can't resolve), and lower the "
                 "confidence accordingly")
    lines.append("")
    lines.append("Return ONLY a JSON array, with no other text and no markdown code fences. Format:")
    lines.append('[{"id": <row_index>, "drug": "...", "analyte": "...", "dose": "...", '
                 '"species": "...", "matrix": "...", "confidence": 0.9, "notes": "..."}, ...]')
    return "\n".join(lines)


def call_llm(client, prompt, n_curves=1, retries=3):
    max_tokens = min(MAX_TOKENS_CAP, BASE_MAX_TOKENS + PER_CURVE_TOKENS * n_curves)
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
            return json.loads(text), text
        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON parse failed (attempt {attempt+1}, max_tokens={max_tokens}): {e}")
            time.sleep(1)
        except Exception as e:
            print(f"    [WARN] API call failed (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None, ""


# ==============================================================================
# Main
# ==============================================================================

def main(manifest_path, meta_path, out_path, log_path, limit=None, dry_run=False):
    mf = pd.read_csv(manifest_path)
    meta = pd.read_csv(meta_path)
    meta["_key"] = meta["csv_filename"].astype(str).str.strip()

    mf["source_pmid"] = mf["source_pmid"].astype(str)

    # ---------- apply exclusion lists ----------
    before = len(mf)
    excluded_skip = mf[mf["source_pmid"].isin(SKIP_PMIDS)]
    excluded_reextract = mf[mf["source_pmid"].isin(REEXTRACT_PMIDS)]
    print(f"Total rows: {before}")
    print(f"Rows in SKIP_PMIDS (excluded outright, not sent to LLM, not in final output): {len(excluded_skip)}")
    print(f"Rows in REEXTRACT_PMIDS (flagged for re-extraction, not sent to LLM): {len(excluded_reextract)}")

    mf["llm_note"] = ""
    mf.loc[mf["source_pmid"].isin(REEXTRACT_PMIDS), "llm_note"] = "NEEDS_REEXTRACTION"

    processable = mf[~mf["source_pmid"].isin(SKIP_PMIDS | REEXTRACT_PMIDS)].copy()

    # ---------- rows that need review ----------
    review_mask = processable.apply(needs_review, axis=1)
    review_rows = processable[review_mask].copy()
    print(f"Rows remaining for review after exclusions: {len(review_rows)}")

    review_rows["_orig_col"] = review_rows["Additional Properties"].apply(extract_orig_col)

    # ---------- group by source figure ----------
    groups = review_rows.groupby("source_panel_csv")
    print(f"Number of figures requiring an LLM call after grouping: {groups.ngroups}")

    if limit:
        print(f"(--limit {limit}: only running the first {limit} groups for testing)")

    if dry_run:
        print("\n[DRY RUN] Not calling the API for real, just printing how many prompts would be sent.")
        print(f"Expected number of calls: {min(groups.ngroups, limit or groups.ngroups)}")
        return

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    updates = {}   # row_index -> dict of updated fields
    n_done = 0

    with open(log_path, "w") as logf:
        for source_csv, grp in groups:
            if limit and n_done >= limit:
                break
            n_done += 1

            fname = os.path.basename(source_csv)
            mrow = meta[meta["_key"] == fname]
            mrow = mrow.iloc[0].to_dict() if not mrow.empty else {}
            caption = mrow.get("caption", "")

            curves = [{"row_index": idx, "orig_col": r["_orig_col"],
                       "curve_label": r["curve_label"], "source_panel": r["source_panel"]}
                      for idx, r in grp.iterrows()]

            prompt = build_prompt(caption, mrow, curves)
            print(f"  [{n_done}/{groups.ngroups}] {fname} ({len(curves)} curves) ...")

            result, raw_text = call_llm(client, prompt, n_curves=len(curves))
            logf.write(json.dumps({"source_csv": source_csv, "prompt": prompt,
                                    "response": raw_text}, ensure_ascii=False) + "\n")

            if result is None and len(curves) > 1:
                # Whole group failed (possibly still too large/complex) -> split
                # into two halves and retry each separately, rather than
                # leaving the entire group with no result at all
                print(f"    [RETRY] Group failed, splitting in half and retrying (n={len(curves)})")
                mid = len(curves) // 2
                for half in (curves[:mid], curves[mid:]):
                    if not half:
                        continue
                    half_prompt = build_prompt(caption, mrow, half)
                    half_result, half_raw = call_llm(client, half_prompt, n_curves=len(half))
                    logf.write(json.dumps({"source_csv": source_csv, "prompt": half_prompt,
                                            "response": half_raw, "is_retry_half": True},
                                           ensure_ascii=False) + "\n")
                    if half_result:
                        result = (result or []) + half_result

            if result is None:
                print(f"    [ERROR] Failed completely (including the split retry), skipping this group")
                continue

            for item in result:
                try:
                    idx = int(item["id"])
                except Exception:
                    continue
                updates[idx] = item

    # ---------- write results back ----------
    for idx, item in updates.items():
        mf.at[idx, "Drug"] = item.get("drug", mf.at[idx, "Drug"])
        mf.at[idx, "Analyte Name"] = item.get("analyte", mf.at[idx, "Analyte Name"])
        if item.get("dose"):
            mf.at[idx, "Administration Value + Unit"] = item["dose"]
        if item.get("species"):
            mf.at[idx, "Species"] = item["species"]
        if item.get("matrix"):
            mf.at[idx, "Matrix"] = item["matrix"]
        mf.at[idx, "match_status"] = "LLM_REVIEWED"
        mf.at[idx, "llm_note"] = f"confidence={item.get('confidence','')}; {item.get('notes','')}"

    mf.to_csv(out_path, index=False)
    print(f"\nDone. Updated {len(updates)} rows.")
    print(f"Output -> {out_path}")
    print(f"Raw request/response log -> {log_path}")

    low_conf = mf[mf["llm_note"].str.contains(r"confidence=0\.[0-4]", na=False, regex=True)]
    if len(low_conf):
        print(f"\nRows with low confidence (<0.5): {len(low_conf)}, recommend spot-checking these by hand")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK root dir (default: $AUTOPK_BASE_DIR or '.')")
    ap.add_argument("--manifest", default=None,
                     help="Input curve manifest CSV (default: <base-dir>/extracted_curves/curve_manifest.csv)")
    ap.add_argument("--meta", default=None,
                     help="Figure-level PK metadata CSV (default: <base-dir>/extract_metadata_trial2/step_2_calculated_parameters/pk_metadata_figure_level.csv)")
    ap.add_argument("--out", default=None,
                     help="Output CSV (default: <base-dir>/extracted_curves/curve_manifest_llm_reviewed.csv)")
    ap.add_argument("--log", default=None,
                     help="Raw request/response log (default: <base-dir>/extracted_curves/llm_review_raw_responses.jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N groups, for small-scale testing")
    ap.add_argument("--dry-run", action="store_true", help="Don't call the API, just see how many groups would run")
    args = ap.parse_args()

    base_dir = args.base_dir
    manifest_path = args.manifest or os.path.join(base_dir, "extracted_curves", "curve_manifest.csv")
    meta_path = args.meta or os.path.join(
        base_dir, "extract_metadata_trial2", "step_2_calculated_parameters", "pk_metadata_figure_level.csv")
    out_path = args.out or os.path.join(base_dir, "extracted_curves", "curve_manifest_llm_reviewed.csv")
    log_path = args.log or os.path.join(base_dir, "extracted_curves", "llm_review_raw_responses.jsonl")

    main(manifest_path, meta_path, out_path, log_path, limit=args.limit, dry_run=args.dry_run)
