#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AutoPK - Multi-curve figure splitter
================================================================================
Splits the digitized CSVs in extracted_images/ into "one sub-CSV per curve",
writes them to extracted_curves/, and produces a curve_manifest.csv index.

Handles two kinds of figures:
  [Branch 1] Subpanel figures   filenames ending in _A/_B/_C/... appearing as a group
                                 -> the original whole-figure CSV is superseded (not digitized)
                                 -> panel-to-analyte assignment comes from the caption's
                                    "(X) drugname (abbr)" pattern
                                 -> within a panel, curves are further split by column header
  [Branch 2] Plain single figures   no panel suffix
                                 -> curves are split directly by column header, matching
                                    headers against the metadata's dose / matrix / species
                                    value lists

Conventions:
  * n/a points are kept as-is (downstream NCA drops them itself); n_points_valid
    is recorded separately in the manifest
  * concentration columns are named  concentration (<original column header>)
  * metadata fields that are "unknown" -> left blank in the output, never written as "unknown"
  * Drug (what was administered) and Analyte (what was measured) are kept separate;
    prodrugs are resolved via PRODRUG_MAP
  * anything uncertain is always flagged NEEDS_REVIEW, never guessed

Join key: metadata's csv_filename column  <->  the CSV filename on disk
================================================================================
"""

import os
import re
import glob
import argparse
import pandas as pd


# ==============================================================================
# 1. Configuration - paths are resolved from --base-dir / AUTOPK_BASE_DIR, or
#    can be overridden individually
# ==============================================================================

DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")

# Relative path prefix written into the manifest (kept relative so the
# manifest is portable across machines); override with --images-rel or
# AUTOPK_IMAGES_REL if your images live under a different relative path.
DEFAULT_IMAGES_ROOT_REL = os.environ.get("AUTOPK_IMAGES_REL", "AutoPK/extracted_images")

# Minimum number of valid points a curve needs to be output at all
MIN_VALID_POINTS = 3

# Common column names that "look like a curve but are actually metadata" --
# sample sizes/statistics, not concentration data
NON_CURVE_COLNAME_PATTERNS = re.compile(
    r"^(n|sample size|number of (subjects|patients)|no\.? of (subjects|patients)|"
    r"subjects?|patients?|count)s?$", re.I)


# ------------------------------------------------------------------------------
# Prodrug -> active-metabolite map   >>> extend and verify against your corpus <<<
#
# key   = the thing administered (prodrug / salt form), lowercase
# value = the thing actually measured in plasma (analyte)
#
# Note: this table only handles the most common one-to-one cases. Cases like
# TFV -> TFV-DP (the intracellular diphosphate active metabolite, measured in
# PBMCs) where one parent maps to multiple analytes depending on matrix don't
# belong in this table -- handle those via the matrix field instead.
# ------------------------------------------------------------------------------
PRODRUG_MAP = {
    "fosamprenavir": "amprenavir",
    "fapv": "amprenavir",
    "tenofovir disoproxil fumarate": "tenofovir",
    "tenofovir disoproxil": "tenofovir",
    "tenofovir df": "tenofovir",
    "tdf": "tenofovir",
    "tenofovir alafenamide": "tenofovir",
    "taf": "tenofovir",
    "valacyclovir": "acyclovir",
    "valganciclovir": "ganciclovir",
}

# Abbreviation -> full name (used to resolve abbreviations found in column headers)
ABBREV_MAP = {
    "apv": "amprenavir",
    "fapv": "fosamprenavir",
    "tnv": "tenofovir",
    "tfv": "tenofovir",
    "tfv-dp": "tenofovir diphosphate",
    "tfvdp": "tenofovir diphosphate",
    "ftc": "emtricitabine",
    "rtv": "ritonavir",
    "atv": "atazanavir",
    "drv": "darunavir",
    "sqv": "saquinavir",
    "ral": "raltegravir",
    "efv": "efavirenz",
    "3tc": "lamivudine",
    "azt": "zidovudine",
}


# ==============================================================================
# 2. Small helpers
# ==============================================================================

def norm(s):
    """Normalize: collapse whitespace + lowercase"""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def norm_hard(s):
    """Aggressive normalization: keep only alphanumerics. Used to match
    '600 mg' against '600mg'."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def safe_name(s):
    """Make a string filename-safe while keeping it readable: any run of
    non-alphanumeric characters -> a single underscore"""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", str(s))).strip("_")


def slug(s):
    """Compact label (used in curve_label)"""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s))


def blank_if_unknown(v):
    """metadata's "unknown" -> blank"""
    return "" if norm(v) in ("unknown", "nan", "none", "") else str(v).strip()


def split_field_values(v):
    """
    'amprenavir 1400 mg, ritonavir 100 mg and 200 mg'
      -> ['amprenavir 1400 mg', 'ritonavir 100 mg', '200 mg']
    '75 mg, 150 mg, 300 mg, 600 mg'
      -> ['75 mg','150 mg','300 mg','600 mg']
    """
    if not isinstance(v, str) or norm(v) in ("", "unknown", "nan", "none"):
        return []
    v = re.sub(r"\s+and\s+", ", ", v)
    return [x.strip() for x in v.split(",") if x.strip()]


def unit_from_col(colname):
    """Pull the concentration unit out of a column header's parentheses:
    prefer one containing '/' (e.g. ng/mL), otherwise blank"""
    for u in re.findall(r"\(([^)]+)\)", str(colname)):
        if "/" in u:
            return u.strip()
    return ""


def time_unit_from_col(colname):
    """'Time (hours)' -> 'hours'"""
    m = re.search(r"\(([^)]+)\)", str(colname))
    return m.group(1).strip() if m else ""


def dose_from_col(colname):
    """
    Extract a dose from a column header. Prefers a 'RTV 200 mg' style
    boosting dose with a drug-name prefix, otherwise falls back to any
    '<number> <unit>'.
    Returns (boosting_drug, dose_string)
    """
    m = re.search(r"([A-Za-z]{2,})\s*[/\s]*(\d+(?:\.\d+)?\s*(?:mg|g|µg|ug|mcg)(?:/kg)?)",
                  str(colname), re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m2 = re.search(r"(\d+(?:\.\d+)?\s*(?:mg|g|µg|ug|mcg)(?:/kg)?)", str(colname), re.I)
    return "", (m2.group(1).strip() if m2 else "")


def resolve_drug_analyte(analyte_candidate, caption="", meta_drug="", col_header=""):
    """
    Given a string that "looks like a drug name", split it into
    (drug administered, analyte measured).

    Logic:
      1. Expand any abbreviation to its full name first
      2. If it's itself a prodrug -> Drug = it, Analyte = active metabolite
      3. Otherwise check whether the column header / caption / metadata
         drug_name mentions its prodrug form
         - column header evidence is the strongest: 'fAPV w/RTV 200 mg'
           contains fAPV -> fosamprenavir
         - caption evidence: 'dosing with tenofovir DF' -> TDF
      4. If none of the above -> Drug = Analyte = itself
    Returns (drug, analyte, status)
    """
    a = norm(analyte_candidate)
    a = ABBREV_MAP.get(a, a)
    if not a:
        return "", "", "NEEDS_REVIEW"

    # case 2: it's itself a prodrug
    if a in PRODRUG_MAP:
        return a, PRODRUG_MAP[a], "MATCHED"

    # case 3a: an abbreviation token in the column header -> expand -> check
    # whether it's a's prodrug
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", str(col_header)):
        expanded = ABBREV_MAP.get(norm(tok), norm(tok))
        if expanded in PRODRUG_MAP and PRODRUG_MAP[expanded] == a:
            return expanded, a, "MATCHED"

    # case 3b: the prodrug's full name appears in the caption / metadata drug_name
    hay = norm(caption) + " " + norm(meta_drug)
    for pro, act in PRODRUG_MAP.items():
        if act == a and pro in hay:
            return pro, a, "MATCHED"

    # case 4
    return a, a, "MATCHED"


# ==============================================================================
# 3. Caption parsing: panel letter -> analyte
# ==============================================================================

# Drug names are at most 3 words; use a bounded repeat rather than an
# unbounded lazy match, otherwise the regex will greedily swallow unrelated
# text earlier in the sentence (e.g. treating an entire clause as "the drug name")
_NAME_TOKEN = r"[A-Za-z][A-Za-z0-9\-]*"
_NAME_GROUP = rf"(?:{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})"

# Strict mode: requires "(A) drugname (abbr)" -- the abbreviation parentheses
# must be present, which gives high accuracy (case-insensitive on both letters)
# The abbreviation must be at least 2 characters, otherwise the next panel's
# "(b)" would be mistaken for the previous drug's abbreviation
CAP_STRICT = re.compile(rf"\(([A-Za-z])\)\s*({_NAME_GROUP})\s*\(([A-Za-z0-9\-]{{2,}})\)")
# Loose mode: "(A) drugname" -- no abbreviation parentheses, easy to
# mis-capture, only used when strict mode finds nothing
CAP_LOOSE = re.compile(rf"\(([A-Za-z])\)\s*({_NAME_GROUP})(?=\s*[,;.]|\s+and\b|$)")

# Reverse mode: some papers write "drugname (abbr) (A)" or "drugname (a)",
# with the letter last
CAP_REVERSE_STRICT = re.compile(rf"({_NAME_GROUP})\s*\(([A-Za-z0-9\-]{{2,}})\)\s*\(([A-Za-z])\)")
CAP_REVERSE_LOOSE = re.compile(rf"({_NAME_GROUP})\s*\(([A-Za-z])\)(?=\s*[,;.]|\s+and\b|$)")


_NAME_STOPWORDS = {"for", "and", "of", "in", "the", "with", "versus", "vs", "following",
                    "over", "during", "after", "before", "a", "an", "at"}


def _trim_name_noise(name):
    """
    The bounded-repeat capture occasionally picks up a leading connector word
    (e.g. 'for plasma tenofovir'). Keep only the last 1-2 words, and strip any
    leftover leading connector words (for/and/of etc.).
    """
    toks = name.split()
    toks = toks[-2:] if len(toks) > 2 else toks
    while len(toks) > 1 and toks[0].lower() in _NAME_STOPWORDS:
        toks = toks[1:]
    return " ".join(toks)


def caption_panel_map(caption):
    """
    Returns ({LETTER(uppercase): {'name':..., 'abbr':...}}, mode)
    mode is one of {'strict','loose','none'}
    Tries both forward ("(A) drugname (abbr)") and reverse
    ("drugname (abbr) (A)") ordering, trusting strict mode (abbreviation
    parentheses present) first; any letters still missing after both
    forward and reverse strict passes fall back to loose mode.
    """
    out = {}
    if not isinstance(caption, str):
        return out, "none"

    for m in CAP_STRICT.finditer(caption):
        letter, name, abbr = m.group(1).upper(), _trim_name_noise(m.group(2).strip()), m.group(3).strip()
        if 1 <= len(name.split()) <= 3:
            out.setdefault(letter, {"name": name, "abbr": abbr})
    for m in CAP_REVERSE_STRICT.finditer(caption):
        name, abbr, letter = _trim_name_noise(m.group(1).strip()), m.group(2).strip(), m.group(3).upper()
        if 1 <= len(name.split()) <= 3:
            out.setdefault(letter, {"name": name, "abbr": abbr})
    if out:
        return out, "strict"

    for m in CAP_LOOSE.finditer(caption):
        letter, name = m.group(1).upper(), _trim_name_noise(m.group(2).strip())
        if 1 <= len(name.split()) <= 3:
            out.setdefault(letter, {"name": name, "abbr": ""})
    for m in CAP_REVERSE_LOOSE.finditer(caption):
        name, letter = _trim_name_noise(m.group(1).strip()), m.group(2).upper()
        if 1 <= len(name.split()) <= 3:
            out.setdefault(letter, {"name": name, "abbr": ""})
    return (out, "loose") if out else ({}, "none")


def parse_per_analyte_dose(dose_str):
    """
    'amprenavir 1400 mg, ritonavir 100 mg and 200 mg, tenofovir 300 mg'
      -> {'amprenavir': ['1400 mg'], 'ritonavir': ['100 mg','200 mg'], 'tenofovir': ['300 mg']}
    If the dose field has no drug names at all (e.g. '75 mg, 150 mg'), returns {}
    """
    res = {}
    if not isinstance(dose_str, str):
        return res
    for part in re.split(r",\s*(?=[A-Za-z])", dose_str):
        mm = re.match(r"\s*([A-Za-z][A-Za-z\-\s]*?)\s+(\d.*)", part.strip())
        if mm:
            drug = norm(mm.group(1))
            doses = re.findall(r"\d+(?:\.\d+)?\s*(?:mg|g|µg|ug|mcg)(?:/kg)?", mm.group(2), re.I)
            if doses:
                res.setdefault(drug, []).extend(doses)
    return res


# ==============================================================================
# 4. Column header -> metadata field matching (used by Branch 2)
# ==============================================================================

MATCH_FIELDS = ["dose", "matrix", "species"]


def find_matching_drug_in_text(text, candidates):
    """
    When metadata's drug_name is a multi-value list (e.g. a combination-drug
    study), check the column header text for which of these candidate drug
    names (or their abbreviations) actually appears there.
    Returns the matching candidate's original text, or "" if none match.
    """
    text_n = norm_hard(text)
    for cand in candidates:
        cand_n = norm_hard(cand)
        if cand_n and cand_n in text_n:
            return cand
    for cand in candidates:
        cand_norm = norm(cand)
        for abbr, full in ABBREV_MAP.items():
            if full == cand_norm and norm_hard(abbr) in text_n:
                return cand
    return ""


def match_col_to_field(colname, meta_row):
    """
    Match a column header like '600 mg' against metadata's dose/matrix/species
    value lists.
    Returns (field, value, status)
    """
    label = norm(colname)
    label_h = norm_hard(colname)
    # also try with parenthesized content stripped (handles 'Blood Plasma (ng/mL)')
    stripped = norm(re.sub(r"\([^)]*\)", "", str(colname)))
    stripped_h = norm_hard(re.sub(r"\([^)]*\)", "", str(colname)))

    for field in MATCH_FIELDS:
        for v in split_field_values(meta_row.get(field, "")):
            vn, vh = norm(v), norm_hard(v)
            if label in (vn,) or label_h == vh or stripped == vn or stripped_h == vh:
                return field, v, "MATCHED"
            # substring fallback: column header 'fAPV w/RTV 200 mg' contains '200 mg'
            if vh and vh in label_h:
                return field, v, "MATCHED_SUBSTRING"

    return None, None, "UNMATCHED"


# ==============================================================================
# 5. Manifest columns
# ==============================================================================

MANIFEST_COLUMNS = [
    # --- provenance ---
    "sub_csv_path", "source_pmid", "source_figure_base", "source_panel",
    "source_panel_csv", "source_panel_png", "source_parent_png",
    "is_subpanel", "curve_index", "curve_label",
    "split_dimension", "match_status", "panel_analyte_status", "caption_parse_mode",
    # --- Minimum Required Information ---
    "Drug", "Drug SMILES",
    "Administration Value + Unit", "Frequency",
    "Infusion", "Infusion Duration", "Administration Route",
    "Analyte Name", "Analyte Unit", "LLOQ Value",
    "Species", "Matrix", "Time Unit",
    # --- reported (left blank, filled in later by NCA) ---
    "Half life (hours)", "CMAX", "AUC", "TMAX",
    # --- counts ---
    "n_points_valid", "n_points_total",
    "Additional Properties",
]


# ==============================================================================
# 6. Core: split one CSV
# ==============================================================================

def split_one_csv(csv_path, meta_row, pmid, out_root, images_root_rel,
                  panel_letter=None, figure_base=None,
                  panel_analyte=None, panel_abbr=None,
                  caption_mode="", panel_status="", panel_extra_label=""):
    """
    Split one CSV (may be a panel, or a plain single figure).
    Returns a list of manifest rows.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [ERROR] Could not read {csv_path}: {e}")
        return []

    if df.shape[1] < 2:
        print(f"  [SKIP] Not enough columns: {os.path.basename(csv_path)}")
        return []

    # ---------- structural check: the first column must look like a time axis
    # (mostly numeric) ----------
    # Some raw digitized tables are actually long-format, e.g. with a first
    # column of category labels ("Observed Concentrations"/"Fitted Curve"),
    # not numeric time values. That table structure doesn't match the
    # "time + multiple curves" assumption, and forcing a split would produce
    # garbage columns, so skip the whole file.
    first_col_numeric_ratio = pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().mean()
    if first_col_numeric_ratio < 0.5:
        print(f"  [SKIP] First column doesn't look like a time axis "
              f"(numeric ratio {first_col_numeric_ratio:.0%}), likely a long-format/"
              f"non-standard table, skipping entire file: {os.path.basename(csv_path)}")
        return []

    fname = os.path.basename(csv_path)
    base_stem = os.path.splitext(fname)[0]
    time_col = df.columns[0]
    tunit = time_unit_from_col(time_col)
    caption = str(meta_row.get("caption", ""))
    meta_drug = str(meta_row.get("drug_name", ""))
    per_dose = parse_per_analyte_dose(meta_row.get("dose", ""))

    out_dir = os.path.join(out_root, str(pmid))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for ci, col in enumerate(df.columns[1:], start=1):

        # ---------- exclude sample-size/statistic pseudo-curve columns ----------
        col_bare = re.sub(r"\([^)]*\)", "", str(col)).strip()
        if NON_CURVE_COLNAME_PATTERNS.match(col_bare):
            print(f"  [SKIP] Looks like a sample-size/statistic column, not a curve: {fname} :: {col}")
            continue

        # ---------- valid point-count check ----------
        n_valid = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
        if n_valid < MIN_VALID_POINTS:
            print(f"  [SKIP] Not enough valid points ({n_valid}): {fname} :: {col}")
            continue

        boost_drug, col_dose = dose_from_col(col)
        cunit = unit_from_col(col)

        # ---------- determine Drug / Analyte / dose / split dimension ----------
        if panel_letter:
            # ===== Branch 1: subpanel =====
            # analyte comes from the caption's (X) mapping -- the abbreviation
            # is cleaner than the full name (which occasionally carries
            # connector-word noise), so prefer the abbreviation if present
            analyte_candidate = panel_abbr or panel_analyte or ""
            drug, analyte, da_status = resolve_drug_analyte(
                analyte_candidate, caption=caption, meta_drug=meta_drug, col_header=col)

            # this analyte's own administered dose: look it up in the
            # per-analyte dose map
            dl = per_dose.get(norm(analyte), []) or per_dose.get(norm(drug), [])
            if len(dl) == 1:
                admin_dose = dl[0]
            elif len(dl) > 1:
                # this drug has multiple doses (e.g. RTV 100/200 mg) -- it is
                # itself the split dimension, use the dose extracted from the
                # column header
                admin_dose = col_dose if col_dose in [norm_hard(x) and x for x in dl] or col_dose in dl else ""
                if not admin_dose and col_dose:
                    # loosely: does the normalized column-header dose match
                    # any of the list entries?
                    for x in dl:
                        if norm_hard(x) == norm_hard(col_dose):
                            admin_dose = x
                            break
            else:
                admin_dose = ""

            split_dim = "panel_analyte + curve_dose"
            m_status = da_status
            label_bits = [panel_abbr or slug(analyte)]
            if col_dose:
                label_bits.append((slug(boost_drug) if boost_drug else "") + slug(col_dose))
            if panel_extra_label:
                label_bits.append(slug(panel_extra_label))
            curve_label = "_".join([b for b in label_bits if b]) or f"c{ci}"

        else:
            # ===== Branch 2: plain single figure =====
            n_curve_cols = df.shape[1] - 1   # curve columns other than the time column

            if n_curve_cols == 1:
                # only one curve -- there's no "split dimension" to speak of,
                # nothing needs matching, just copy metadata's single value
                # directly (dose/matrix/species should already be single-valued)
                field, value, m_status = None, None, "SINGLE_CURVE_NO_SPLIT_NEEDED"
                split_dim = "none (single curve)"

                dn = split_field_values(meta_drug)
                analyte_guess = dn[0] if len(dn) == 1 else meta_drug
                drug, analyte, _ = resolve_drug_analyte(
                    analyte_guess, caption=caption, meta_drug=meta_drug, col_header=col)

                dv = split_field_values(meta_row.get("dose", ""))
                admin_dose = dv[0] if len(dv) == 1 else blank_if_unknown(meta_row.get("dose", ""))
                matrix_override = ""
                species_override = ""

            else:
                # a genuine multi-curve figure -- match against column headers
                field, value, m_status = match_col_to_field(col, meta_row)

                dn = split_field_values(meta_drug)
                if len(dn) == 1:
                    analyte_guess = dn[0]
                elif len(dn) > 1:
                    # metadata combines multiple drugs (a combination study)
                    # -- check the column header text for which candidate
                    # drug name actually appears there, rather than giving up
                    analyte_guess = find_matching_drug_in_text(col, dn)
                else:
                    analyte_guess = ""
                drug, analyte, _ = resolve_drug_analyte(
                    analyte_guess, caption=caption, meta_drug=meta_drug, col_header=col)

                admin_dose = ""
                matrix_override = ""
                species_override = ""
                if field == "dose":
                    admin_dose = value
                    split_dim = "dose"
                elif field == "matrix":
                    matrix_override = value
                    split_dim = "matrix"
                elif field == "species":
                    species_override = value
                    split_dim = "species"
                else:
                    split_dim = "UNKNOWN"

                # if dose wasn't matched but metadata is single-valued -> copy
                # it directly (e.g. all curves share the same dose, and the
                # real split dimension is matrix/species)
                if not admin_dose:
                    dv = split_field_values(meta_row.get("dose", ""))
                    if len(dv) == 1:
                        admin_dose = dv[0]

            curve_label = slug(col) or f"c{ci}"

        # ---------- Matrix / Species (Branch 2 may override from the column header) ----------
        matrix_val = blank_if_unknown(meta_row.get("matrix", ""))
        species_val = blank_if_unknown(meta_row.get("species", ""))
        if not panel_letter:
            if 'matrix_override' in dir() and matrix_override:
                matrix_val = matrix_override
            if 'species_override' in dir() and species_override:
                species_val = species_override
        # metadata is multi-valued and wasn't selected by the column header ->
        # leave it blank (don't guess)
        if len(split_field_values(meta_row.get("matrix", ""))) > 1 and \
           (panel_letter or not locals().get("matrix_override")):
            pass  # keep metadata's original value; multi-value cases need manual review in the manifest

        # ---------- write the sub-CSV ----------
        sub_name = f"{safe_name(base_stem)}__c{ci}_{curve_label}.csv"
        sub_path = os.path.join(out_dir, sub_name)
        sub = df[[time_col, col]].copy()
        sub.columns = [time_col, f"concentration ({col})"]
        sub.to_csv(sub_path, index=False)   # n/a values kept as-is

        # ---------- manifest row ----------
        rows.append({
            "sub_csv_path": os.path.relpath(sub_path, out_root),
            "source_pmid": str(pmid),
            "source_figure_base": figure_base or base_stem,
            "source_panel": panel_letter or "",
            "source_panel_csv": f"{images_root_rel}/{pmid}/{fname}",
            "source_panel_png": f"{images_root_rel}/{pmid}/{base_stem}.png",
            "source_parent_png": (f"{images_root_rel}/{pmid}/{figure_base}.png"
                                  if panel_letter else ""),
            "is_subpanel": bool(panel_letter),
            "curve_index": f"c{ci}",
            "curve_label": curve_label,
            "split_dimension": split_dim,
            "match_status": m_status,
            "panel_analyte_status": panel_status,
            "caption_parse_mode": caption_mode,

            "Drug": drug,
            "Drug SMILES": blank_if_unknown(meta_row.get("component_smiles", "")),
            "Administration Value + Unit": admin_dose,
            "Frequency": "",
            "Infusion": "",
            "Infusion Duration": "",
            "Administration Route": blank_if_unknown(meta_row.get("route_of_administration", "")),
            "Analyte Name": analyte,
            "Analyte Unit": cunit,
            "LLOQ Value": "",
            "Species": species_val,
            "Matrix": matrix_val,
            "Time Unit": tunit,

            "Half life (hours)": "", "CMAX": "", "AUC": "", "TMAX": "",
            "n_points_valid": n_valid,
            "n_points_total": len(df),
            "Additional Properties": f"orig_col={col}" + (f"; panel_extra={panel_extra_label}" if panel_extra_label else ""),
        })

    return rows


# ==============================================================================
# 7. Main
# ==============================================================================

PANEL_SUFFIX = re.compile(r"^(?P<base>.+?)_(?P<letter>[A-Z])$")   # kept for legacy reference, no longer the primary logic

# recognizes the hash-based naming core: <pmid>_p<page>_<index>_<hex hash>
HASH_CORE = re.compile(r"^(\d+_p\d+_\d+_[0-9a-fA-F]{6,})")


def split_core_suffix(stem):
    """
    Splits a filename stem into (core, suffix).
    Two conventions:
      1. hash-style:  <pmid>_p<page>_<idx>_<hash>[_<anything>]
                    -> core runs up to the hash, everything after (no matter
                       how long or what it contains) is the suffix
      2. other styles (e.g. Screenshot timestamps): the core itself contains
                    no underscore, so split at the first underscore -- before
                    it is the core, after it (whether "A" "a" "(a)" "a)"
                    "None" "" or "(a) Oral study") is treated as the whole
                    suffix
    No underscore -> (stem, None), meaning there's no suffix and it's not a
    panel candidate
    """
    m = HASH_CORE.match(stem)
    if m:
        core = m.group(1)
        rest = stem[len(core):]
        if rest.startswith("_"):
            return core, rest[1:]     # drop the immediate underscore; suffix may be an empty string
        return stem, None             # no extra suffix, the hash filename itself is the whole thing

    if "_" in stem:
        core, _, rest = stem.partition("_")
        return core, rest

    return stem, None


def extract_panel_key(suffix):
    """
    Pulls a clean "panel letter" key (uppercased) out of an arbitrary suffix
    form, so it can be aligned against the caption's (X) markers.
    'A' -> 'A' | 'a' -> 'A' | '(a)' -> 'A' | 'a)' -> 'A'
    '(a) Oral study' -> ('A', 'Oral study')   <- extra descriptive text is
                                                  carried along too
    'None' / '' / '.' etc. where no letter can be extracted -> (None, original suffix)
    """
    if suffix is None:
        return None, ""
    m = re.match(r"^\(?([A-Za-z])\)?\s*(.*)$", suffix.strip())
    if m and m.group(1).isalpha():
        return m.group(1).upper(), m.group(2).strip()
    return None, suffix.strip()


def detect_panels(csv_paths):
    """
    Returns (panel_groups, singles).
    panel_groups = {core: {panel_key_or_rawsuffix: (path, extra_label)}}
    Only a core with >=2 distinct suffixes among its files counts as a panel group.
    """
    by_core = {}
    for p in csv_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        core, suffix = split_core_suffix(stem)
        by_core.setdefault(core, []).append((p, suffix))

    panel_groups, singles = {}, []
    for core, items in by_core.items():
        if len(items) < 2:
            singles.append(items[0][0])
            continue

        # >=2 files share the same core -> this is a panel group (including
        # the one with an empty/None suffix, the "original whole figure")
        group = {}
        for path, suffix in items:
            if suffix is None:
                group["__ORIGINAL__"] = (path, "")
                continue
            if suffix == "":
                # has an underscore but nothing after it (e.g. '..._hash_.csv')
                # -- this is NOT "the original with no suffix", it's a real,
                # distinct second file of unknown content, so it can't share
                # the __ORIGINAL__ key (that would overwrite it and silently
                # drop an entire figure). Give it its own key and flag for review.
                group["RAW::(empty_suffix)"] = (path, "")
                continue
            key, extra = extract_panel_key(suffix)
            dict_key = key if key else f"RAW::{suffix}"
            group[dict_key] = (path, extra)
        panel_groups[core] = group

    return panel_groups, singles


TARGET_FIGURE_TYPE = "concentration-time curve"

# Miscellaneous files that are obviously not "digitized figure data" -- skip
# these silently, without even logging them
NON_DATA_FILENAME_PATTERNS = re.compile(r"captions?\.csv$|readme|\.DS_Store", re.I)


def main(meta_path, images_root, out_root, images_root_rel):
    meta_all = pd.read_csv(meta_path)
    meta_all["_key"] = meta_all["csv_filename"].astype(str).str.strip()

    # ---------- keep only concentration-time curves ----------
    meta_ct = meta_all[meta_all["figure_type"] == TARGET_FIGURE_TYPE].copy()
    ct_keys = set(meta_ct["_key"])
    print(f"Rows in metadata with figure_type == '{TARGET_FIGURE_TYPE}': {len(meta_ct)}")
    print(f"Unique PMIDs involved (in theory extracted_curves/ should have this many folders): {meta_ct['pmid'].nunique()}")
    print()

    all_rows, superseded, unmatched_meta, wrong_type_skipped = [], [], [], []

    pmid_dirs = sorted([d for d in glob.glob(os.path.join(images_root, "*"))
                        if os.path.isdir(d)])
    print(f"Scanning {len(pmid_dirs)} PMID folders...\n")

    processed_base_figures = set()   # (pmid, base_stem) dedup, used to compare "how many figures were actually processed"

    empty_pmid_dirs = []

    for pmid_dir in pmid_dirs:
        pmid = os.path.basename(pmid_dir)
        all_csvs = sorted(glob.glob(os.path.join(pmid_dir, "*.csv")))
        # filter out non-data files like captions.csv
        csvs = [c for c in all_csvs if not NON_DATA_FILENAME_PATTERNS.search(os.path.basename(c))]
        if not csvs:
            if pmid in set(meta_ct["pmid"].astype(str)):
                empty_pmid_dirs.append((pmid, len(all_csvs)))
            continue

        panel_groups, singles = detect_panels(csvs)

        # ---------- Branch 1: subpanels ----------
        for core, group in panel_groups.items():
            panel_items = {k: v for k, v in group.items() if k != "__ORIGINAL__"}
            original = group.get("__ORIGINAL__")

            # whether the whole group belongs to the target figure_type (check any member)
            all_paths = [v[0] for v in group.values()]
            any_in_ct = any(os.path.basename(p) in ct_keys for p in all_paths)
            if not any_in_ct:
                continue   # entire group isn't the target type -> normal filtering, not logged

            if original is not None:
                superseded.append(f"{pmid}/{os.path.basename(original[0])}")

            # take the caption from any panel member's metadata row (the whole figure shares one)
            any_path = list(panel_items.values())[0][0] if panel_items else (original[0] if original else None)
            mrow_any = meta_all[meta_all["_key"] == os.path.basename(any_path)] if any_path else pd.DataFrame()
            caption = str(mrow_any.iloc[0]["caption"]) if not mrow_any.empty else ""
            pmap, cap_mode = caption_panel_map(caption)

            raw_keys = [k for k in panel_items if k.startswith("RAW::")]
            letter_keys = [k for k in panel_items if not k.startswith("RAW::")]
            missing = set(letter_keys) - set(pmap.keys())
            group_status = "NEEDS_REVIEW" if (missing or raw_keys or cap_mode != "strict") else "MATCHED"

            for key, (path, extra_label) in panel_items.items():
                fname = os.path.basename(path)
                if fname not in ct_keys:
                    wrong_type_skipped.append(f"{pmid}/{fname}")
                    continue
                mrow = meta_all[meta_all["_key"] == fname]
                if mrow.empty:
                    unmatched_meta.append(f"{pmid}/{fname}")
                    continue
                mrow = mrow.iloc[0].to_dict()

                if key.startswith("RAW::"):
                    # a suffix with no extractable letter (None / . / Main /
                    # Inset etc.) -- can't be aligned via caption, honestly
                    # flag NEEDS_REVIEW and leave panel_analyte blank for
                    # manual/LLM judgment
                    info, letter_for_label = {}, key.replace("RAW::", "")
                    this_status = "NEEDS_REVIEW"
                else:
                    info = pmap.get(key, {})
                    letter_for_label = key
                    this_status = group_status if info else "NEEDS_REVIEW"

                processed_base_figures.add((pmid, core))
                all_rows += split_one_csv(
                    path, mrow, pmid, out_root, images_root_rel,
                    panel_letter=letter_for_label, figure_base=core,
                    panel_analyte=info.get("name", ""),
                    panel_abbr=info.get("abbr", ""),
                    caption_mode=cap_mode,
                    panel_status=this_status,
                    panel_extra_label=extra_label,
                )

        # ---------- Branch 2: plain single figures ----------
        for path in singles:
            fname = os.path.basename(path)
            if fname not in ct_keys:
                wrong_type_skipped.append(f"{pmid}/{fname}")
                continue
            mrow = meta_all[meta_all["_key"] == fname]
            if mrow.empty:
                unmatched_meta.append(f"{pmid}/{fname}")
                continue
            base_stem = os.path.splitext(fname)[0]
            processed_base_figures.add((pmid, base_stem))
            all_rows += split_one_csv(path, mrow.iloc[0].to_dict(), pmid, out_root, images_root_rel)

    # ---------- output ----------
    os.makedirs(out_root, exist_ok=True)
    mf = pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS)
    mf_path = os.path.join(out_root, "curve_manifest.csv")
    mf.to_csv(mf_path, index=False)

    ct_stems = meta_ct["_key"].str.replace(".csv", "", regex=False)
    cores = ct_stems.apply(lambda s: split_core_suffix(s)[0])
    core_counts = cores.value_counts()
    expected_figures = len(core_counts)  # each core counts as one distinct figure (regardless of how many panels it splits into)
    expected_pmids = meta_ct["pmid"].nunique()
    actual_pmids = mf["source_pmid"].nunique() if len(mf) else 0
    actual_figures = len(processed_base_figures)

    print("\n" + "=" * 70)
    print(f"Sub-curve CSVs generated: {len(mf)}")
    print(f"Superseded whole figures (original figure of a panel group): {len(superseded)}  examples: {superseded[:5]}")
    print(f"Skipped for wrong figure_type (normal filtering, not data loss): {len(wrong_type_skipped)}")
    found_pmid_names = set(os.path.basename(d) for d in pmid_dirs)
    expected_pmid_names = set(meta_ct["pmid"].astype(str))
    missing_dirs_entirely = sorted(expected_pmid_names - found_pmid_names)
    print(f"PMIDs with a CT curve in metadata but no folder at all on disk: {len(missing_dirs_entirely)}  {missing_dirs_entirely[:20]}")
    print(f"PMID folders that exist but contain no data CSV at all (possibly never digitized): {len(empty_pmid_dirs)}  {empty_pmid_dirs[:20]}")
    print(f"No metadata match (a genuine gap, worth investigating): {len(unmatched_meta)}  examples: {unmatched_meta[:10]}")
    print()
    print(f"--- Coverage check ---")
    print(f"Expected distinct figure count: {expected_figures}  |  actually processed: {actual_figures}")
    print(f"Expected PMID folder count: {expected_pmids}  |  PMIDs appearing in manifest: {actual_pmids}")
    if actual_pmids < expected_pmids:
        missing_pmids = set(meta_ct["pmid"].astype(str)) - set(mf["source_pmid"].astype(str))
        print(f"Missing PMIDs (have a CT curve in metadata but don't appear in the manifest): {sorted(missing_pmids)[:20]}")
    print("=" * 70)

    if len(mf):
        print(f"\nmatch_status distribution:\n{mf['match_status'].value_counts().to_string()}")
        print(f"\npanel_analyte_status distribution:\n{mf['panel_analyte_status'].value_counts().to_string()}")
        need = mf[(mf["match_status"] == "UNMATCHED") |
                  (mf["panel_analyte_status"] == "NEEDS_REVIEW")]
        print(f"\nRows needing manual/LLM review: {len(need)}")
    print(f"\nmanifest -> {mf_path}")
    print("=" * 70)
    return mf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK root dir (default: $AUTOPK_BASE_DIR or '.')")
    ap.add_argument("--meta", default=None,
                     help="Figure-level PK metadata CSV (default: <base-dir>/extract_metadata_trial2/step_2_calculated_parameters/pk_metadata_figure_level.csv)")
    ap.add_argument("--images", default=None,
                     help="Directory of extracted digitized-CSV images (default: <base-dir>/extracted_images)")
    ap.add_argument("--out", default=None,
                     help="Output directory for split curves + manifest (default: <base-dir>/extracted_curves)")
    ap.add_argument("--images-rel", default=DEFAULT_IMAGES_ROOT_REL,
                     help="Relative path prefix written into the manifest for image/CSV "
                          "provenance columns (default: $AUTOPK_IMAGES_REL or "
                          "'AutoPK/extracted_images')")
    args = ap.parse_args()

    base_dir = args.base_dir
    meta_path = args.meta or os.path.join(
        base_dir, "extract_metadata_trial2", "step_2_calculated_parameters", "pk_metadata_figure_level.csv")
    images_root = args.images or os.path.join(base_dir, "extracted_images")
    out_root = args.out or os.path.join(base_dir, "extracted_curves")

    main(meta_path, images_root, out_root, args.images_rel)
