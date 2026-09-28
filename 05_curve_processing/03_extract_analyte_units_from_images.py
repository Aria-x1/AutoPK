"""
extract_analyte_units.py

Re-extracts the "Analyte Unit" for every curve in pk_curve_level_master_index.csv
by reading the y-axis label directly off the source figure image, using
GPT-4o-mini vision (detail="high").

Design:
  - Grouped by unique figure (source_pmid + source_figure_base), NOT by row.
    All curves that come from the same figure image share one axis -> one call
    per figure, result applied to every curve row from that figure.
  - Ignores the old "Missing Unit comments - not in captions" column entirely
    (per instruction) -- this is a from-scratch re-extraction, not a fill-in.
  - Does not touch that old column in the output; only overwrites/creates:
        Analyte Unit                 (overwritten for ALL rows, not just missing)
        Analyte Unit Confidence      (high / medium / low, as reported by model)
        Analyte Unit Raw Axis Text   (verbatim-ish reading of the axis label)
        Analyte Unit Notes           (anything odd: blurry, multiple candidate
                                       units, panel-specific units, etc.)

Usage:
    export OPENAI_API_KEY=sk-...
    python3 extract_analyte_units.py --base-dir /path/to/AutoPK

Requires:
    pip install openai pandas --break-system-packages
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config -- paths are resolved from --base-dir / AUTOPK_BASE_DIR, or can be
# overridden individually
# ---------------------------------------------------------------------------
DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")

MODEL = "gpt-4o-mini"
IMAGE_DETAIL = "high"

IMAGE_EXTENSIONS = [".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp", ".webp"]

# gpt-4o-mini high-detail images run ~30-40k billed tokens each, and the
# default org TPM cap is 200k/min -- so pacing matters a lot more than the
# call count does. ~13s between calls keeps us under ~5 images/min (~180k
# tokens/min), safely below a 200k TPM cap.
SLEEP_BETWEEN_CALLS_SEC = 13
MAX_RETRIES = 5
RETRY_BACKOFF_BASE_SEC = 20  # first retry waits ~20s, then 40s, 80s, ...

# If True: load OUTPUT_CSV (if it exists) instead of MASTER_CSV, and only
# (re)call the model for figures that don't already have a clean success in
# it. Figures that already succeeded are left untouched.
RESUME = True

# Set this to True for ONE run whenever the prompt/extraction logic changes
# in a way that could affect previously-"successful" results (like this
# time, after discovering the model was fabricating units for axes with no
# printed unit text). Forces every figure to be re-called regardless of
# RESUME. Set back to False afterward for normal incremental runs.
FORCE_FULL_RERUN = True

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are reading a pharmacokinetic concentration-vs-time figure \
extracted from a scientific publication. Your ONLY job is to identify the unit \
of the y-axis (the concentration axis), by reading text that is ACTUALLY PRINTED \
in the image.

CRITICAL RULE: You may only report a unit if you can point to literal characters \
printed on the image spelling out that unit (e.g. "ng/mL" written next to or \
below the axis, in an axis title, or in the legend). Many PK figures have a \
y-axis with ONLY numeric tick labels (e.g. "10, 100, 1000") and NO unit text \
anywhere in the image -- the unit lives in the figure caption or methods text, \
which you cannot see. In that case you MUST report unit_text_visible_in_image: \
false and analyte_unit: null. Do NOT fill in a "typical" or "plausible" unit \
for this kind of drug/assay from general knowledge -- that is a fabrication, \
even if it happens to be correct. If you are not looking at literal printed \
characters, you do not know the unit.

Respond with ONLY a raw JSON object (no markdown fences, no commentary) with \
exactly these keys:
{
  "unit_text_visible_in_image": <true if literal unit characters are printed somewhere in the image, false otherwise>,
  "analyte_unit": "<the unit exactly as printed, e.g. 'ng/mL', 'mg/L', 'ng/g', 'nM'; MUST be null if unit_text_visible_in_image is false>",
  "raw_axis_text": "<the exact text you are reading it from, e.g. 'Concentration (ng/mL)'; null if unit_text_visible_in_image is false>",
  "confidence": "<high | medium | low>",
  "notes": "<empty string, or a short note -- e.g. 'no unit text anywhere in image, only numeric tick marks', blurriness, multiple panels with different units, ambiguous characters, etc.>"
}
"""

USER_PROMPT = (
    "Read the y-axis (concentration axis) label on this pharmacokinetic curve "
    "figure and report its unit as specified."
)


def normalize_unit(raw):
    """Canonicalize spelling/casing variants of the same unit, e.g.
    'ng/ml' -> 'ng/mL', 'mcg/mL' -> 'µg/mL', 'mg/l' -> 'mg/L',
    'fmol/million cells' -> 'fmol/10^6 cells'. Idempotent -- safe to
    re-run on already-normalized values.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return raw
    s = str(raw).strip()
    if not s:
        return raw

    # unify micrograms spellings -> µg (mcg, ug, alternate mu character)
    s = re.sub(r"(?i)\bmcg\b", "µg", s)
    s = re.sub(r"(?i)\bug\b", "µg", s)
    s = s.replace("μg", "µg")  # normalize the Greek "μ" (U+03BC) to micro sign "µ" (U+00B5)
    s = s.replace("μ", "µ")

    # unify "million cells" -> "10^6 cells"
    s = re.sub(r"(?i)million\s*cells", "10^6 cells", s)

    if "/" in s:
        num, denom = s.split("/", 1)
        num, denom = num.strip(), denom.strip()

        # denominator casing (volume/mass units)
        if re.fullmatch(r"(?i)ml", denom):
            denom = "mL"
        elif re.fullmatch(r"(?i)l", denom):
            denom = "L"
        elif re.fullmatch(r"(?i)g", denom):
            denom = "g"

        # numerator casing (common mass/molar prefixes)
        num_map = {"ng": "ng", "pg": "pg", "mg": "mg", "fmol": "fmol",
                   "pmol": "pmol", "nmol": "nmol", "µg": "µg", "µmol": "µmol"}
        low = num.lower()
        if low in num_map:
            num = num_map[low]

        s = f"{num}/{denom}"
    else:
        # standalone molar units, e.g. nM, µM, pM
        if re.fullmatch(r"(?i)nm", s):
            s = "nM"
        elif re.fullmatch(r"(?i)(u|µ)m", s):
            s = "µM"
        elif re.fullmatch(r"(?i)pm", s):
            s = "pM"

    return s


def find_image_for_figure(images_dir: Path, source_pmid, source_figure_base: str) -> "Path | None":
    """Locate the actual image file for a given figure entry.

    source_figure_base looks like:
        AutoPK/extracted_images/15673751/15673751_p03_01_ac0bfd2a4a19f45b.csv
    but the real file on disk is the same basename with an image extension,
    sitting in extracted_images/<pmid>/.
    """
    pmid_folder = images_dir / str(source_pmid)
    if not pmid_folder.is_dir():
        return None

    stem = Path(source_figure_base).stem  # e.g. "15673751_p03_01_ac0bfd2a4a19f45b"

    # 1) exact stem + known image extension
    for ext in IMAGE_EXTENSIONS:
        candidate = pmid_folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # 2) case-insensitive / fuzzy match within the folder
    for f in pmid_folder.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.stem == stem:
            return f

    # 3) last resort: only one image file in the folder -> use it
    images_in_folder = [f for f in pmid_folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    if len(images_in_folder) == 1:
        return images_in_folder[0]

    return None


def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_model(client: OpenAI, image_path: Path) -> dict:
    b64 = encode_image_b64(image_path)
    ext = image_path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext == "jpg" else ext

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{mime};base64,{b64}",
                                    "detail": IMAGE_DETAIL,
                                },
                            },
                        ],
                    },
                ],
                temperature=0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                return {
                    "unit_text_visible_in_image": False,
                    "analyte_unit": None,
                    "raw_axis_text": raw,
                    "confidence": "low",
                    "notes": "JSON_PARSE_FAILED",
                }

            # belt-and-suspenders: if the model says no unit text is visible
            # but still filled in analyte_unit anyway, don't trust that unit
            visible = result.get("unit_text_visible_in_image")
            if visible is False and result.get("analyte_unit"):
                result["notes"] = (
                    (result.get("notes") or "")
                    + " [OVERRIDDEN: model reported unit_text_visible_in_image=false "
                    "but still returned a unit -- discarded]"
                ).strip()
                result["analyte_unit"] = None
                result["confidence"] = "low"
            return result
        except Exception as e:
            last_error = e
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                print(f"    rate limited, retry {attempt}/{MAX_RETRIES} after {wait}s...")
                time.sleep(wait)
                continue
            break

    return {
        "unit_text_visible_in_image": False,
        "analyte_unit": None,
        "raw_axis_text": None,
        "confidence": "low",
        "notes": f"API_ERROR: {last_error}",
    }


def already_succeeded(df: pd.DataFrame, pmid, fig_base) -> bool:
    """A figure counts as already done if we got a clean (non-error) result
    for it -- this now includes legitimate 'no unit text visible in image'
    results (null unit is a valid answer, not a failure). Only API/parse
    failures get retried on resume."""
    mask = (df["source_pmid"] == pmid) & (df["source_figure_base"] == fig_base)
    sub = df.loc[mask]
    if sub.empty:
        return False
    notes = sub["Analyte Unit Notes"].iloc[0]
    confidence = sub["Analyte Unit Confidence"].iloc[0]
    if pd.isna(notes):
        notes = ""
    if pd.isna(confidence):
        return False  # never got any result at all (e.g. missing image)
    failed = ("API_ERROR" in str(notes)) or ("JSON_PARSE_FAILED" in str(notes))
    return not failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK root dir (default: $AUTOPK_BASE_DIR or '.')")
    ap.add_argument("--master-csv", default=None,
                     help="Input master index CSV (default: <base-dir>/pk_curve_level_master_index.csv)")
    ap.add_argument("--images-dir", default=None,
                     help="Directory of extracted figure images (default: <base-dir>/extracted_images)")
    ap.add_argument("--output-csv", default=None,
                     help="Output CSV (default: <base-dir>/pk_curve_level_master_index_units_updated.csv)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    master_csv = Path(args.master_csv) if args.master_csv else base_dir / "pk_curve_level_master_index.csv"
    images_dir = Path(args.images_dir) if args.images_dir else base_dir / "extracted_images"
    output_csv = Path(args.output_csv) if args.output_csv else base_dir / "pk_curve_level_master_index_units_updated.csv"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running.")
    client = OpenAI(api_key=api_key)

    if RESUME and output_csv.exists():
        print(f"Resuming from existing output: {output_csv}")
        df = pd.read_csv(output_csv)
    else:
        df = pd.read_csv(master_csv)

    # ensure new columns exist
    for col in [
        "Analyte Unit",
        "Analyte Unit Confidence",
        "Analyte Unit Raw Axis Text",
        "Analyte Unit Notes",
        "Analyte Unit Text Visible In Image",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    unique_figures = df[["source_pmid", "source_figure_base"]].drop_duplicates()
    print(f"Found {len(unique_figures)} unique figures across {len(df)} curve rows.")

    if FORCE_FULL_RERUN:
        print("FORCE_FULL_RERUN is True -- re-calling the model for ALL figures, "
              "ignoring any previously 'successful' results.")
        to_process = [
            (row.source_pmid, row.source_figure_base)
            for row in unique_figures.itertuples(index=False)
        ]
    else:
        to_process = [
            (row.source_pmid, row.source_figure_base)
            for row in unique_figures.itertuples(index=False)
            if not (RESUME and already_succeeded(df, row.source_pmid, row.source_figure_base))
        ]
    n_skipped = len(unique_figures) - len(to_process)
    if n_skipped:
        print(f"Skipping {n_skipped} figures that already have a clean result.")
    print(f"Will call the model for {len(to_process)} figures.")

    results_by_figure = {}
    missing_images = []

    for i, (pmid, fig_base) in enumerate(to_process, start=1):
        img_path = find_image_for_figure(images_dir, pmid, fig_base)

        if img_path is None:
            print(f"[{i}/{len(to_process)}] MISSING IMAGE for pmid={pmid} fig={fig_base}")
            missing_images.append((pmid, fig_base))
            continue

        print(f"[{i}/{len(to_process)}] pmid={pmid} -> {img_path.name}")
        result = call_vision_model(client, img_path)

        print(f"    -> unit={result.get('analyte_unit')!r} "
              f"confidence={result.get('confidence')!r} "
              f"raw={result.get('raw_axis_text')!r}")

        results_by_figure[(pmid, fig_base)] = result

        # write out incrementally after every figure so a later crash/rate
        # limit doesn't lose progress already made in this run
        mask = (df["source_pmid"] == pmid) & (df["source_figure_base"] == fig_base)
        df.loc[mask, "Analyte Unit"] = result.get("analyte_unit")
        df.loc[mask, "Analyte Unit Confidence"] = result.get("confidence")
        df.loc[mask, "Analyte Unit Raw Axis Text"] = result.get("raw_axis_text")
        df.loc[mask, "Analyte Unit Notes"] = result.get("notes")
        df.loc[mask, "Analyte Unit Text Visible In Image"] = result.get("unit_text_visible_in_image")
        df.to_csv(output_csv, index=False)

        if i < len(to_process):
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    # already written incrementally during the loop above; final save here
    # covers the case where nothing needed (re)processing at all
    if "Analyte Unit (pre-normalization)" not in df.columns:
        df["Analyte Unit (pre-normalization)"] = pd.NA
    needs_norm_col = df["Analyte Unit (pre-normalization)"].isna() & df["Analyte Unit"].notna()
    df.loc[needs_norm_col, "Analyte Unit (pre-normalization)"] = df.loc[needs_norm_col, "Analyte Unit"]
    df["Analyte Unit"] = df["Analyte Unit"].apply(normalize_unit)

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    print("\nAnalyte Unit value counts after normalization:")
    print(df["Analyte Unit"].value_counts())

    if missing_images:
        print(f"\n{len(missing_images)} figures had no locatable image file:")
        for pmid, fig_base in missing_images:
            print(f"  pmid={pmid}  {fig_base}")

    low_conf = df[df["Analyte Unit Confidence"] == "low"][
        ["source_pmid", "source_figure_base", "Analyte Unit", "Analyte Unit Notes"]
    ].drop_duplicates()
    if len(low_conf):
        print(f"\n{len(low_conf)} figures flagged LOW confidence -- review these:")
        print(low_conf.to_string(index=False))

    no_unit_visible = df[df["Analyte Unit Text Visible In Image"] == False][
        ["source_pmid", "source_figure_base", "curve_label"]
    ].drop_duplicates()
    if len(no_unit_visible):
        print(f"\n{len(no_unit_visible)} figures have NO unit text anywhere in the image "
              f"(unit must come from the caption/methods text instead -- not something "
              f"vision-on-the-cropped-figure can ever solve):")
        print(no_unit_visible.to_string(index=False))


if __name__ == "__main__":
    main()
