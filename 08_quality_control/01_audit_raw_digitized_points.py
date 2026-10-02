#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 2 scan - AutoPK / PDR must-have field audit on RAW digitized point CSVs.

Checks every raw CSV pointed to by csv_location in calculated_pk_curves,
quantifying two must-have fields that Tier-1 extraction can't answer:
    #9  Analyte Value        (per-point concentration values)
    #13 Time since last dose (the time axis)
Also opportunistically recovers:
    #2  Unit / #3 Administration Value  (dose salvaged from column headers)
    #10 LLOQ / BLQ markers

Usage:
    python3 01_audit_raw_digitized_points.py --base-dir /path/to/AutoPK
Optional:
    python3 01_audit_raw_digitized_points.py --base /Users/xxy/Documents/MML --input calculated_pk_curves_v2.csv
"""

import argparse
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------- configuration
DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")
DEFAULT_BASE = os.environ.get("AUTOPK_MML_DIR", DEFAULT_BASE_DIR)  # base dir that csv_location is relative to
DEFAULT_INPUT = "calculated_pk_curves_v2.csv"      # input table (v1 or v2 both work)
OUT_DETAIL = "tier2_per_csv.csv"                   # per-file detail
OUT_SUMMARY = "tier2_summary.txt"                  # summary report

TIME_PAT = re.compile(r"\b(time|hour|hr\b|\bh\b|day|week|min|elapsed)", re.I)
TIME_UNIT_PAT = re.compile(r"\((?:in\s*)?(hours?|hrs?|h|days?|d|minutes?|min|weeks?|wk)\)|,\s*(hours?|hrs?|h|days?|d|min)\b", re.I)
CONC_UNIT_PAT = re.compile(r"\(\s*((?:ng|µg|ug|mg|g|pmol|fmol|nmol|µmol|umol)\s*/\s*"
                           r"(?:ml|l|dl|g|kg|million cells|10\^6 cells)|nM|µM|uM|mM)\s*\)", re.I)
DOSE_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*(mg/kg|mg|µg/kg|µg|mcg|ug|g)\b", re.I)
BLQ_PAT = re.compile(r"\b(blq|bql|lloq|loq|below limit|nd|not detect)", re.I)
NA_TOKENS = {"n/a", "na", "nan", "", "-", "--", "none", "null"}


def is_number(x):
    try:
        float(str(x).strip())
        return True
    except (ValueError, TypeError):
        return False


def detect_layout(df):
    """Determine whether this is wide (first column is time) or long
    (one column is time, another is concentration)."""
    cols = [str(c) for c in df.columns]
    if len(cols) < 2:
        return "unknown", None, []
    # long: first column is a non-numeric label + there's a clear time
    # column that isn't the first one
    time_idx = [i for i, c in enumerate(cols) if TIME_PAT.search(c)]
    if time_idx and time_idx[0] > 0:
        first_col_numeric = df.iloc[:, 0].map(is_number).mean() > 0.8
        if not first_col_numeric:
            ti = time_idx[0]
            value_cols = [c for i, c in enumerate(cols) if i != ti and i != 0]
            return "long", cols[ti], value_cols
    # wide: first column is time
    if TIME_PAT.search(cols[0]) or df.iloc[:, 0].map(is_number).mean() > 0.8:
        return "wide", cols[0], cols[1:]
    return "unknown", None, cols[1:]


def scan_csv(path):
    r = {
        "file_exists": False, "read_error": "", "layout": "", "n_rows": 0,
        "time_col": "", "time_unit": "", "has_time": False,
        "n_value_cols": 0, "value_cells_total": 0, "value_cells_numeric": 0,
        "value_fill_rate": None, "has_analyte_value": False,
        "conc_unit": "", "has_conc_unit": False,
        "dose_in_header": "", "dose_unit_in_header": "", "has_dose": False,
        "blq_marker": "", "has_blq": False,
    }
    if not os.path.exists(path):
        return r
    r["file_exists"] = True
    try:
        df = pd.read_csv(path, keep_default_na=False, na_values=[])
    except Exception as e:
        r["read_error"] = str(e)[:120]
        return r
    if df.empty or len(df.columns) < 2:
        r["read_error"] = "empty or single-column"
        return r

    r["n_rows"] = len(df)
    layout, time_col, value_cols = detect_layout(df)
    r["layout"] = layout
    r["time_col"] = str(time_col) if time_col else ""
    r["n_value_cols"] = len(value_cols)

    # --- #13 Time since last dose ---
    if time_col is not None:
        tc = str(time_col)
        r["has_time"] = bool(TIME_PAT.search(tc)) or df[time_col].map(is_number).mean() > 0.8
        m = TIME_UNIT_PAT.search(tc)
        if m:
            r["time_unit"] = (m.group(1) or m.group(2) or "").lower()
        elif re.search(r"\bday\b", tc, re.I):
            r["time_unit"] = "day"
        elif re.search(r"week", tc, re.I):
            r["time_unit"] = "week"

    # --- #9 Analyte Value ---
    total = numeric = 0
    blq_hits = []
    for c in value_cols:
        col = df[c]
        for v in col:
            s = str(v).strip()
            total += 1
            if is_number(s):
                numeric += 1
            elif s.lower() not in NA_TOKENS and BLQ_PAT.search(s):
                blq_hits.append(s)
    r["value_cells_total"] = total
    r["value_cells_numeric"] = numeric
    r["value_fill_rate"] = round(numeric / total, 4) if total else None
    r["has_analyte_value"] = numeric > 0

    # --- concentration unit ---
    for c in value_cols:
        m = CONC_UNIT_PAT.search(str(c))
        if m:
            r["conc_unit"] = m.group(1)
            break
    r["has_conc_unit"] = bool(r["conc_unit"])

    # --- #2/#3 dose (from column headers) ---
    doses, units = [], []
    for c in value_cols:
        m = DOSE_PAT.search(str(c))
        if m:
            doses.append(m.group(1))
            units.append(m.group(2).lower())
    if doses:
        r["dose_in_header"] = "|".join(doses)
        r["dose_unit_in_header"] = "|".join(sorted(set(units)))
        r["has_dose"] = True

    # --- #10 LLOQ / BLQ ---
    hdr_blq = [str(c) for c in df.columns if BLQ_PAT.search(str(c))]
    if blq_hits or hdr_blq:
        r["blq_marker"] = "|".join(sorted(set(blq_hits + hdr_blq))[:3])
        r["has_blq"] = True

    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK root dir (default: $AUTOPK_BASE_DIR or '.'); "
                          "used to fill in --base when not given")
    ap.add_argument("--base", default=None,
                     help="Base directory that csv_location is relative to "
                          "(default: --base-dir, or $AUTOPK_MML_DIR if set)")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path to the calculated_pk_curves CSV")
    args = ap.parse_args()

    base = args.base or args.base_dir

    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] Input table not found: {args.input}\n    Use --input to specify the path")

    df = pd.read_csv(args.input)
    figs = df.groupby("csv_filename").first().reset_index()
    print(f"Input: {args.input}")
    print(f"Base dir: {base}")
    print(f"Figures to scan: {len(figs)}\n")

    rows = []
    for _, f in figs.iterrows():
        rel = str(f["csv_location"])
        path = rel if os.path.isabs(rel) else os.path.join(base, rel)
        res = scan_csv(path)
        res.update({
            "csv_filename": f["csv_filename"],
            "pmid": f["pmid"],
            "resolved_path": path,
        })
        rows.append(res)

    out = pd.DataFrame(rows)
    cols = ["csv_filename", "pmid", "file_exists", "layout", "n_rows",
            "has_time", "time_col", "time_unit",
            "has_analyte_value", "n_value_cols", "value_cells_total",
            "value_cells_numeric", "value_fill_rate",
            "has_conc_unit", "conc_unit",
            "has_dose", "dose_in_header", "dose_unit_in_header",
            "has_blq", "blq_marker", "read_error", "resolved_path"]
    out = out[cols]
    out.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    # ---------------- summary ----------------
    n = len(out)
    found = out[out["file_exists"]]
    ok = found[found["read_error"] == ""]
    L = []
    L.append("=" * 68)
    L.append("Tier 2 scan - raw digitized CSV audit")
    L.append("=" * 68)
    L.append(f"Total figures          : {n}")
    L.append(f"Files found            : {out['file_exists'].sum()}/{n}")
    L.append(f"Successfully read      : {len(ok)}/{n}")
    if out["file_exists"].sum() < n:
        L.append("")
        L.append("[!] Example missing files:")
        for p in out[~out["file_exists"]]["resolved_path"].head(5):
            L.append(f"    {p}")
    if len(ok):
        L.append("")
        L.append("--- Layout distribution ---")
        for k, v in ok["layout"].value_counts().items():
            L.append(f"    {k:10s}: {v}")
        L.append("")
        L.append("--- PDR must-have coverage (based on %d successfully read files) ---" % len(ok))
        t = ok["has_time"].sum()
        a = ok["has_analyte_value"].sum()
        L.append(f"    #13 Time since last dose : {t}/{len(ok)} ({t/len(ok):.0%})")
        L.append(f"    #9  Analyte Value        : {a}/{len(ok)} ({a/len(ok):.0%})")
        L.append("")
        L.append("--- Supplementary fields ---")
        for lab, c in [("Time unit recognizable", "time_unit"), ("Conc. unit recognizable", "conc_unit"),
                       ("Header contains dose", "dose_in_header"), ("Has BLQ/LLOQ marker", "blq_marker")]:
            k = (ok[c].astype(str).str.len() > 0).sum()
            L.append(f"    {lab:24s}: {k}/{len(ok)} ({k/len(ok):.0%})")
        L.append("")
        L.append("--- Time unit distribution ---")
        for k, v in ok[ok["time_unit"] != ""]["time_unit"].value_counts().items():
            L.append(f"    {k:8s}: {v}")
        L.append("")
        L.append("--- Concentration unit distribution ---")
        for k, v in ok[ok["conc_unit"] != ""]["conc_unit"].value_counts().head(10).items():
            L.append(f"    {k:20s}: {v}")
        tot_cells = int(ok["value_cells_total"].sum())
        num_cells = int(ok["value_cells_numeric"].sum())
        L.append("")
        L.append("--- Total data points ---")
        L.append(f"    Total value cells : {tot_cells}")
        L.append(f"    Numeric           : {num_cells} ({num_cells/tot_cells:.1%})" if tot_cells else "")
    L.append("")
    L.append(f"Per-file detail written to: {OUT_DETAIL}")

    report = "\n".join(L)
    print(report)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"Summary written to: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
