"""
QC Script: Figure Data Extraction Quality Control
Uses OpenAI GPT-4o Vision to QC the extracted data against its source figure.

Usage:
    python 02_vision_qc_against_figures.py --root /path/to/AutoPK/extracted_images
                         --api_key YOUR_KEY (or set the OPENAI_API_KEY env var)
                         --output qc_results.csv
                         --workers 4 (number of parallel threads, default 3)
"""

import os
import sys
import csv
import json
import base64
import argparse
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# -- dependency check ------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai first: pip install openai")
    sys.exit(1)

# -- configuration -----------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MODEL = "gpt-4o"
MAX_CSV_CHARS = 3000   # max characters of a CSV to send to the model (avoid exceeding token limits)
MAX_RETRIES = 3
RETRY_DELAY = 5         # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# -- Prompt -------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert reviewer of pharmacokinetic (PK) data extracted
from scientific figures. Your job is to compare extracted CSV data against the original
figure and flag serious structural or labeling errors only.

Use your scientific judgment. Be lenient — only flag issues that completely
invalidate the extracted data's usability.
Calibration guideline: in a healthy extraction pipeline, over 85% of figures should PASS.
If you are flagging more than 1 in 7 figures, you are likely being too strict.
Default to PASS when uncertain.

Respond ONLY with a valid JSON object. No extra text."""

USER_PROMPT_TEMPLATE = """Please QC this pharmacokinetic figure and its extracted data.

The figure image is attached. The extracted CSV data is below:
```
{csv_content}
```

FLAG only these serious problems:
- An entire data series or group is completely missing from the CSV
- Data from two separate sub-figures merged into one CSV with incorrect column headers
- Values in completely wrong order of magnitude (e.g., 10000x off)
- Completely wrong chart structure (e.g., figure has 3 groups, CSV only has 1)

Do NOT flag these:
- Individual missing data points
- Minor numerical imprecision or slight misreads
- Missing a few timepoints out of many
- Small differences between extracted values and visual estimates
- Predicted vs observed value inconsistencies

Respond with exactly this JSON:
{{
  "status": "PASS" or "FLAG",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "chart_type": "line/bar/scatter/boxplot/other",
  "major_issues": ["specific issue 1", ...],
  "brief_reason": "one sentence summary"
}}"""


# -- core functions -----------------------------------------------------------
def encode_image(image_path: Path) -> tuple[str, str]:
    """Convert an image to base64, returning (base64_str, media_type)."""
    suffix = image_path.suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), media_type


def read_csv_content(csv_path: Path, max_chars: int = MAX_CSV_CHARS) -> str:
    """Read a CSV's content, truncating if it's too long."""
    try:
        content = csv_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[Error reading CSV: {e}]"
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n... [truncated, total {len(content)} chars]"
    return content


def call_openai_qc(client: OpenAI, image_path: Path, csv_path: Path) -> dict:
    """Call GPT-4o to QC one figure+CSV pair, returning a result dict."""
    csv_content = read_csv_content(csv_path)
    img_b64, media_type = encode_image(image_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{img_b64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": USER_PROMPT_TEMPLATE.format(csv_content=csv_content)
                }
            ]
        }
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)
            return {
                "status": result.get("status", "ERROR"),
                "confidence": result.get("confidence", ""),
                "chart_type": result.get("chart_type", ""),
                "major_issues": "; ".join(result.get("major_issues", [])),
                "brief_reason": result.get("brief_reason", ""),
                "error": ""
            }
        except json.JSONDecodeError as e:
            return {
                "status": "ERROR",
                "confidence": "",
                "chart_type": "",
                "major_issues": "",
                "brief_reason": f"JSON parse error: {e}",
                "error": str(e)
            }
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                log.warning(f"Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                return {
                    "status": "ERROR",
                    "confidence": "",
                    "chart_type": "",
                    "major_issues": "",
                    "brief_reason": f"API error: {e}",
                    "error": str(e)
                }

    return {"status": "ERROR", "confidence": "", "chart_type": "",
            "major_issues": "", "brief_reason": "Max retries exceeded", "error": ""}


def find_pairs(root: Path) -> list[dict]:
    """
    Supports two directory structures:
    1. root/subfolder/image + csv  (PMID-subfolder structure)
    2. root/image + csv            (files directly flat in root)
    """
    import re

    def normalize(stem: str) -> str:
        s = re.sub(r'[\s\(\)]+', '_', stem)
        s = re.sub(r'_+', '_', s)
        return s.strip('_').lower()

    pairs = []

    def collect_pairs_from_dir(folder: Path, paper_id: str):
        images = {normalize(f.stem): f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS}
        csvs = {normalize(f.stem): f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() == ".csv"
                and "copy" not in f.stem.lower()}
        for norm_stem, img_path in images.items():
            if norm_stem in csvs:
                pairs.append({
                    "paper_id": paper_id,
                    "figure_name": img_path.stem,
                    "image_path": img_path,
                    "csv_path": csvs[norm_stem],
                })

    subdirs = [f for f in sorted(root.iterdir()) if f.is_dir()]
    if subdirs:
        for subfolder in subdirs:
            collect_pairs_from_dir(subfolder, paper_id=subfolder.name)
    else:
        collect_pairs_from_dir(root, paper_id=root.name)

    return pairs


def process_one(pair: dict, client: OpenAI) -> dict:
    """Process a single figure+CSV pair, returning one result row."""
    log.info(f"Processing: {pair['paper_id']} / {pair['figure_name']}")
    qc = call_openai_qc(client, pair["image_path"], pair["csv_path"])
    return {
        "paper_id": pair["paper_id"],
        "figure_name": pair["figure_name"],
        "image_path": str(pair["image_path"]),
        "csv_path": str(pair["csv_path"]),
        **qc,
        "processed_at": datetime.now().isoformat()
    }


# -- main -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QC figure data extraction with GPT-4o Vision")
    parser.add_argument("--root", required=True, help="Path to extracted_images folder")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", default="qc_results.csv", help="Output CSV path")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers (default: 3, keep low to avoid rate limits)")
    parser.add_argument("--resume", action="store_true", help="Skip already processed figures (reads existing output CSV)")
    parser.add_argument("--only_papers", nargs="+", default=None,
                        help="Only process these paper IDs, e.g. --only_papers 9517952 14693529")
    parser.add_argument("--skip_papers", nargs="+", default=None,
                        help="Skip these paper IDs entirely, e.g. --skip_papers 11557462")
    parser.add_argument("--start_from", default=None,
                        help="Start processing from this paper_id (alphabetical order, inclusive)")
    parser.add_argument("--max_figures", type=int, default=None,
                        help="Stop after processing this many figures (useful for testing)")
    # -- progress control --
    parser.add_argument("--only_paper", default=None,
                        help="Only process one specific paper (its subfolder name, e.g. 9517952)")
    parser.add_argument("--start_from_paper", default=None,
                        help="Start processing from a given paper (ascending alpha/numeric order, skips everything before it)")
    parser.add_argument("--stop_after_paper", default=None,
                        help="Stop after processing a given paper (inclusive)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N figures (for testing, e.g. --limit 5)")
    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key found. Use --api_key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: Root path not found: {root}")
        sys.exit(1)

    # find all figure+CSV pairs
    log.info(f"Scanning {root} for figure+CSV pairs...")
    pairs = find_pairs(root)
    log.info(f"Found {len(pairs)} figure+CSV pairs")

    # -- progress-control filters ------------------------------------------
    # 1. only process a single paper
    if args.only_paper:
        pairs = [p for p in pairs if p["paper_id"] == args.only_paper]
        log.info(f"--only_paper {args.only_paper}: {len(pairs)} pairs")

    # 2. start from a given paper (paper_id sorted as a string, since
    # subfolder names are plain digit strings)
    if args.start_from_paper:
        pairs = [p for p in pairs if p["paper_id"] >= args.start_from_paper]
        log.info(f"--start_from_paper {args.start_from_paper}: {len(pairs)} pairs remaining")

    # 3. stop at a given paper (inclusive)
    if args.stop_after_paper:
        pairs = [p for p in pairs if p["paper_id"] <= args.stop_after_paper]
        log.info(f"--stop_after_paper {args.stop_after_paper}: {len(pairs)} pairs in range")

    # filter: only process the specified papers
    if args.only_papers:
        only_set = set(args.only_papers)
        pairs = [p for p in pairs if p["paper_id"] in only_set]
        log.info(f"--only_papers filter: {len(pairs)} pairs remaining")

    # filter: skip the specified papers
    if args.skip_papers:
        skip_set = set(args.skip_papers)
        pairs = [p for p in pairs if p["paper_id"] not in skip_set]
        log.info(f"--skip_papers filter: {len(pairs)} pairs remaining")

    # filter: start from a given paper_id (in folder-name alpha/numeric order)
    if args.start_from:
        # pairs are already sorted by paper_id (find_pairs uses sorted())
        idx = next((i for i, p in enumerate(pairs) if p["paper_id"] >= args.start_from), 0)
        pairs = pairs[idx:]
        log.info(f"--start_from {args.start_from}: starting at index {idx}, {len(pairs)} pairs remaining")

    # resume mode: skip already-processed pairs
    already_done = set()
    if args.resume and Path(args.output).exists():
        with open(args.output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_done.add((row["paper_id"], row["figure_name"]))
        log.info(f"Resume mode: {len(already_done)} already done, skipping them")
        pairs = [p for p in pairs if (p["paper_id"], p["figure_name"]) not in already_done]
        log.info(f"Remaining to process: {len(pairs)}")

    # 4. cap the count (for testing)
    if args.limit:
        pairs = pairs[:args.limit]
        log.info(f"--limit {args.limit}: will process {len(pairs)} pairs only")

    if not pairs:
        log.info("Nothing to process.")
        return

    # cap the max number to process (for testing)
    if args.max_figures:
        pairs = pairs[:args.max_figures]
        log.info(f"--max_figures: capped at {len(pairs)} figures")

    # rough cost estimate
    estimated_cost = len(pairs) * 0.003  # rough estimate, gpt-4o ~$0.003/figure with image
    log.info(f"Estimated API cost: ~${estimated_cost:.2f} USD (rough estimate)")

    # output CSV columns
    fieldnames = [
        "paper_id", "figure_name", "status", "confidence", "chart_type",
        "major_issues", "brief_reason", "error",
        "image_path", "csv_path", "processed_at"
    ]

    # open the output file (append in resume mode, otherwise write)
    write_mode = "a" if args.resume and Path(args.output).exists() else "w"
    out_file = open(args.output, write_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_mode == "w":
        writer.writeheader()

    # stats
    stats = {"PASS": 0, "FLAG": 0, "ERROR": 0}

    # parallel processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, pair, client): pair for pair in pairs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                writer.writerow(result)
                out_file.flush()  # write immediately so interruptions don't lose progress
                stats[result["status"]] = stats.get(result["status"], 0) + 1
                log.info(f"[{i}/{len(pairs)}] {result['paper_id']}/{result['figure_name']} -> {result['status']}")
            except Exception as e:
                log.error(f"Unexpected error: {e}")

    out_file.close()

    # final stats
    log.info("=" * 50)
    log.info(f"Done! Results saved to: {args.output}")
    log.info(f"PASS: {stats.get('PASS', 0)} | FLAG: {stats.get('FLAG', 0)} | ERROR: {stats.get('ERROR', 0)}")
    log.info(f"FLAG rate: {stats.get('FLAG', 0) / max(len(pairs), 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
