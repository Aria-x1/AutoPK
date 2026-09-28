#!/usr/bin/env python3
"""
Paper-level two-stage PK parameter extraction pipeline.

Unlike the figure-level pipeline, this one does not use captions.csv or
tie records to a specific figure. It reads each PDF's full text + tables
directly and extracts one record per distinct experimental condition
(dose x timepoint x population x route, etc.) found anywhere in the paper.

Two-stage approach:
  Stage 1 (Claude Haiku, cheap) - scan the full text + tables and locate
           WHICH sections (tables, results paragraphs) contain PK data.
           Location only, no extraction yet.
  Stage 2 (Claude Sonnet, accurate) - given those located sections plus the
           full text and tables, extract every PK record with a full
           parameter schema (t_half_h, AUC_value, Cmax_value, Tmax_h,
           Cmin_value, Ctrough_value, CL_F_value, CLrenal_value, Vd_value,
           bioavailability_pct, protein_binding_pct, renal_excretion_pct,
           dose/route/population/species/etc.)

Output: <output-dir>/pk_extracted_multi.json - one flat array of records
across all processed papers (distinguish papers via the "pmid" field).
Also writes a checkpoint file so a run can be resumed after interruption.

Requires ANTHROPIC_API_KEY (via environment).

Usage:
    python3 paper_level_scan_extract.py --pdf-dir /path/to/pdfs --output-dir ./pk_extraction_output
    (--pdf-dir defaults to <base-dir>/LLM-POC-Open-Articles, where
     --base-dir defaults to the AUTOPK_BASE_DIR environment variable if set,
     otherwise the current directory)
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import anthropic
from datetime import datetime
import pymupdf  # PyMuPDF for PDF processing

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")


class Config:
    # Paths - populated from CLI args in main(), do not hardcode here
    PDF_DIR = None
    OUTPUT_DIR = None
    OUTPUT_FILE = "pk_extracted_multi.json"
    CHECKPOINT_FILE = "pk_extraction_checkpoint.json"
    LOG_FILE = "pk_extraction_log.txt"

    # API settings
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

    # Models
    MODEL_SCANNER = "claude-haiku-4-5-20251001"    # cheap scanner (fastest & cheapest)
    MODEL_EXTRACTOR = "claude-sonnet-4-6"          # accurate extractor (best quality)

    # Costs (per token)
    COST_HAIKU_INPUT = 0.80 / 1_000_000
    COST_HAIKU_OUTPUT = 4.00 / 1_000_000
    COST_SONNET_INPUT = 3.00 / 1_000_000
    COST_SONNET_OUTPUT = 15.00 / 1_000_000

    # Processing
    SAVE_EVERY_N = 10
    MAX_RETRIES = 3
    SLEEP_BETWEEN_CALLS = 1


# ============================================================================
# Utilities
# ============================================================================

class Logger:
    def __init__(self, log_file: str, output_dir: str = None):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.log_file = os.path.join(output_dir, log_file)
        else:
            self.log_file = log_file

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(message, str):
            message = message.encode('ascii', 'ignore').decode('ascii')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
        except Exception:
            with open(self.log_file, 'a', encoding='ascii', errors='ignore') as f:
                f.write(log_msg + '\n')


class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.stage1_cost = 0.0
        self.stage2_cost = 0.0
        self.api_calls = 0

    def add_cost(self, input_tokens: int, output_tokens: int, model: str, stage: int):
        if "haiku" in model.lower():
            cost = (input_tokens * Config.COST_HAIKU_INPUT +
                   output_tokens * Config.COST_HAIKU_OUTPUT)
        else:
            cost = (input_tokens * Config.COST_SONNET_INPUT +
                   output_tokens * Config.COST_SONNET_OUTPUT)

        self.total_cost += cost
        if stage == 1:
            self.stage1_cost += cost
        else:
            self.stage2_cost += cost
        self.api_calls += 1

    def report(self) -> str:
        return (f"Total Cost: ${self.total_cost:.2f} | "
                f"Stage1: ${self.stage1_cost:.2f} | "
                f"Stage2: ${self.stage2_cost:.2f} | "
                f"API Calls: {self.api_calls}")


# ============================================================================
# PDF Processing
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Extract text and tables from PDF"""
    try:
        doc = pymupdf.open(pdf_path)

        full_text = ""
        tables_text = []
        num_pages = len(doc)

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

            try:
                tables = page.find_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        table_data = table.extract()
                        if table_data:
                            tables_text.append({
                                'page': page_num + 1,
                                'table_idx': table_idx,
                                'data': table_data
                            })
            except Exception:
                pass

        doc.close()

        return {
            'full_text': full_text,
            'tables': tables_text,
            'num_pages': num_pages
        }

    except Exception as e:
        error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        return {
            'error': error_msg,
            'full_text': '',
            'tables': [],
            'num_pages': 0
        }


def format_table_for_llm(table_data: List[List[str]]) -> str:
    """Format table data as text for LLM"""
    if not table_data:
        return ""

    lines = []
    for row in table_data:
        lines.append(" | ".join(str(cell) if cell else "" for cell in row))

    return "\n".join(lines)


# ============================================================================
# Stage 1: Scanning (Claude Haiku)
# ============================================================================

STAGE1_PROMPT = """You are a pharmacokinetics (PK) data extraction assistant.

Your task: Scan this research paper and identify ALL sections that contain PK parameter data.

Look for:
1. **Tables** with PK parameters (t1/2, AUC, Cmax, Tmax, clearance, volume of distribution, etc.)
2. **Results sections** mentioning PK values
3. **Methods sections** with dosing information

For each identified section, extract:
- Section type: "table", "results_paragraph", "methods_paragraph"
- Location: page number, table number, or section heading
- Brief description: what PK data is present

Output as JSON array:
[
  {
    "section_type": "table",
    "location": "Table 2, Page 5",
    "description": "PK parameters for different doses at steady state",
    "contains_pk_data": true
  },
  ...
]

If NO PK data found, return: []

Be thorough - we don't want to miss any PK data!"""


def scan_paper(client: anthropic.Anthropic, pdf_data: Dict[str, Any],
               pmid: str, logger: Logger, cost_tracker: CostTracker) -> List[Dict]:
    """Stage 1: Scan paper for PK data sections"""

    input_parts = [f"PMID: {pmid}\n\n"]

    full_text = pdf_data['full_text']
    full_text = full_text.encode('ascii', 'ignore').decode('ascii')

    if len(full_text) > 100000:
        full_text = full_text[:100000] + "\n\n[... text truncated ...]"
    input_parts.append(full_text)

    if pdf_data['tables']:
        input_parts.append("\n\n=== EXTRACTED TABLES ===\n")
        for t in pdf_data['tables']:
            input_parts.append(f"\n--- Table on Page {t['page']} ---\n")
            table_text = format_table_for_llm(t['data'])
            table_text = table_text.encode('ascii', 'ignore').decode('ascii')
            input_parts.append(table_text)

    user_message = "".join(input_parts)

    try:
        response = client.messages.create(
            model=Config.MODEL_SCANNER,
            max_tokens=4000,
            temperature=0,
            messages=[
                {"role": "user", "content": STAGE1_PROMPT},
                {"role": "assistant", "content": "I'll scan the paper for PK data sections."},
                {"role": "user", "content": user_message}
            ]
        )

        cost_tracker.add_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            Config.MODEL_SCANNER,
            stage=1
        )

        result_text = response.content[0].text.strip()

        if result_text.startswith('['):
            sections = json.loads(result_text)
        else:
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                sections = json.loads(json_match.group(0))
            else:
                sections = []

        logger.log(f"  Stage 1: Found {len(sections)} PK sections")
        return sections

    except Exception as e:
        error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        logger.log(f"  Stage 1 ERROR: {error_msg}")
        return []


# ============================================================================
# Stage 2: Extraction (Claude Sonnet)
# ============================================================================

STAGE2_PROMPT = """You are a pharmacokinetics (PK) data extraction expert.

Extract ALL PK parameter data from the provided sections into structured JSON records.

**CRITICAL RULES:**
1. Create a SEPARATE record for each unique combination of experimental conditions:
   - Different dose -> separate record
   - Different time point -> separate record
   - Different route -> separate record
   - Different population -> separate record
   - Different feeding state -> separate record

2. Extract ALL available PK parameters for each record

3. Preserve original units (don't convert mg/m2 to mg, etc.)

4. Use null for missing values

**Output Format:**
[
  {
    "pmid": "string",
    "drug": "string",
    "dose_value": number or null,
    "dose_unit": "string" or null,
    "route": "oral|IV|subcutaneous|..." or null,
    "dosing_day": number or null,
    "feeding_state": "fasted|fed|not_specified" or null,
    "study_phase": "single_dose|steady_state|multiple_dose" or null,
    "species": "human|mouse|rat|macaque|..." or null,
    "population": "string description" or null,
    "n": number or null,
    "sex": "male|female|mixed" or null,
    "age_range": "string" or null,
    "weight_kg": number or null,

    "t_half_h": number or null,
    "AUC_value": number or null,
    "AUC_unit": "string" or null,
    "Cmax_value": number or null,
    "Cmax_unit": "string" or null,
    "Tmax_h": number or null,
    "Cmin_value": number or null,
    "Cmin_unit": "string" or null,
    "Ctrough_value": number or null,
    "Ctrough_unit": "string" or null,
    "CL_F_value": number or null,
    "CL_F_unit": "string" or null,
    "CLrenal_value": number or null,
    "CLrenal_unit": "string" or null,
    "Vd_value": number or null,
    "Vd_unit": "string" or null,
    "bioavailability_pct": number or null,
    "protein_binding_pct": number or null,
    "renal_excretion_pct": number or null,

    "concomitant_drugs": "string" or null,
    "food_effect": "string" or null,
    "measurement_type": "mean|median|geometric_mean" or null,
    "source_table": "string" or null,
    "source_section": "string" or null,
    "extraction_confidence": "high|medium|low"
  }
]

**Example:**
If Table 2 shows PK for 3 doses (100mg, 200mg, 400mg), create 3 separate records.
If steady-state data at Day 1, Day 7, Day 14, create 3 records per dose.

Return ONLY the JSON array, no other text."""


def extract_pk_data(client: anthropic.Anthropic, pdf_data: Dict[str, Any],
                   sections: List[Dict], pmid: str, logger: Logger,
                   cost_tracker: CostTracker) -> List[Dict]:
    """Stage 2: Extract PK parameters from identified sections"""

    if not sections:
        logger.log(f"  Stage 2: No sections to extract, skipping")
        return []

    input_parts = [f"PMID: {pmid}\n\n"]

    for section in sections:
        input_parts.append(f"\n=== {section['location']} ===\n")
        input_parts.append(f"Description: {section['description']}\n\n")

    if pdf_data['tables']:
        input_parts.append("\n=== TABLES ===\n")
        for t in pdf_data['tables']:
            input_parts.append(f"\n--- Page {t['page']} ---\n")
            table_text = format_table_for_llm(t['data'])
            table_text = table_text.encode('ascii', 'ignore').decode('ascii')
            input_parts.append(table_text)

    full_text = pdf_data['full_text']
    full_text = full_text.encode('ascii', 'ignore').decode('ascii')

    if len(full_text) > 80000:
        full_text = full_text[:80000] + "\n[truncated]"
    input_parts.append(f"\n\n=== FULL TEXT FOR CONTEXT ===\n{full_text}")

    user_message = "".join(input_parts)

    try:
        response = client.messages.create(
            model=Config.MODEL_EXTRACTOR,
            max_tokens=8000,
            temperature=0,
            messages=[
                {"role": "user", "content": STAGE2_PROMPT},
                {"role": "assistant", "content": "I'll extract all PK data into structured records."},
                {"role": "user", "content": user_message}
            ]
        )

        cost_tracker.add_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            Config.MODEL_EXTRACTOR,
            stage=2
        )

        result_text = response.content[0].text.strip()

        if result_text.startswith('['):
            records = json.loads(result_text)
        else:
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                records = json.loads(json_match.group(0))
            else:
                records = []

        logger.log(f"  Stage 2: Extracted {len(records)} PK records")
        return records

    except Exception as e:
        error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        logger.log(f"  Stage 2 ERROR: {error_msg}")
        return []


# ============================================================================
# Main Pipeline
# ============================================================================

def load_checkpoint() -> Dict[str, Any]:
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, Config.CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {
        'processed_pmids': [],
        'all_records': [],
        'last_saved': None
    }


def save_checkpoint(checkpoint: Dict[str, Any]):
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    checkpoint['last_saved'] = datetime.now().isoformat()
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, Config.CHECKPOINT_FILE)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def save_final_output(records: List[Dict]):
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def process_paper(client: anthropic.Anthropic, pdf_path: str, pmid: str,
                 logger: Logger, cost_tracker: CostTracker) -> List[Dict]:
    """Process a single paper through both stages"""

    logger.log(f"\nProcessing PMID: {pmid}")

    logger.log(f"  Extracting PDF...")
    pdf_data = extract_text_from_pdf(pdf_path)

    if pdf_data.get('error'):
        logger.log(f"  PDF extraction failed: {pdf_data['error']}")
        return []

    logger.log(f"  PDF: {pdf_data['num_pages']} pages, {len(pdf_data['tables'])} tables")

    time.sleep(Config.SLEEP_BETWEEN_CALLS)
    sections = scan_paper(client, pdf_data, pmid, logger, cost_tracker)

    if not sections:
        logger.log(f"  No PK data found, skipping Stage 2")
        return []

    time.sleep(Config.SLEEP_BETWEEN_CALLS)
    records = extract_pk_data(client, pdf_data, sections, pmid, logger, cost_tracker)

    logger.log(f"  Completed: {len(records)} records | {cost_tracker.report()}")

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK project root (default: $AUTOPK_BASE_DIR env var, "
                          "or current directory). Used only to derive defaults below.")
    ap.add_argument("--pdf-dir", default=None,
                     help="Folder of source PDFs (default: <base-dir>/LLM-POC-Open-Articles)")
    ap.add_argument("--output-dir", default=None,
                     help="Where to write pk_extracted_multi.json and checkpoint files "
                          "(default: <base-dir>/pk_extraction_output)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    Config.PDF_DIR = args.pdf_dir if args.pdf_dir else str(base_dir / "LLM-POC-Open-Articles")
    Config.OUTPUT_DIR = args.output_dir if args.output_dir else str(base_dir / "pk_extraction_output")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logger = Logger(Config.LOG_FILE, Config.OUTPUT_DIR)
    cost_tracker = CostTracker()

    if not Config.ANTHROPIC_API_KEY:
        logger.log("ERROR: ANTHROPIC_API_KEY environment variable not set")
        return

    client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    logger.log("=" * 80)
    logger.log("Paper-Level Two-Stage PK Extraction Pipeline - Starting")
    logger.log("=" * 80)

    checkpoint = load_checkpoint()
    logger.log(f"Checkpoint loaded: {len(checkpoint['processed_pmids'])} papers already processed")

    pdf_dir = Path(Config.PDF_DIR)
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))

    logger.log(f"Found {len(all_pdfs)} PDF files")

    to_process = [p for p in all_pdfs if p.stem not in checkpoint['processed_pmids']]

    logger.log(f"To process: {len(to_process)} papers")
    logger.log(f"Estimated cost: ${len(to_process) * 0.11:.2f} (rough estimate)")

    if not to_process:
        logger.log("All papers already processed!")
        return

    print("\nPress Enter to continue, Ctrl+C to cancel...")
    input()

    all_records = checkpoint['all_records']
    processed_pmids = checkpoint['processed_pmids']

    for idx, pdf_path in enumerate(to_process, 1):
        pmid = pdf_path.stem

        logger.log(f"\n{'='*60}")
        logger.log(f"Paper {idx}/{len(to_process)} | Total processed: {len(processed_pmids) + idx}")

        try:
            records = process_paper(client, str(pdf_path), pmid, logger, cost_tracker)

            all_records.extend(records)
            processed_pmids.append(pmid)

            if idx % Config.SAVE_EVERY_N == 0:
                checkpoint['all_records'] = all_records
                checkpoint['processed_pmids'] = processed_pmids
                save_checkpoint(checkpoint)
                save_final_output(all_records)
                logger.log(f"  Checkpoint saved ({len(all_records)} total records)")

        except KeyboardInterrupt:
            logger.log("\nInterrupted by user")
            checkpoint['all_records'] = all_records
            checkpoint['processed_pmids'] = processed_pmids
            save_checkpoint(checkpoint)
            save_final_output(all_records)
            logger.log("Progress saved!")
            break

        except Exception as e:
            error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            logger.log(f"  ERROR processing {pmid}: {error_msg}")
            continue

    checkpoint['all_records'] = all_records
    checkpoint['processed_pmids'] = processed_pmids
    save_checkpoint(checkpoint)
    save_final_output(all_records)

    logger.log("\n" + "=" * 80)
    logger.log("EXTRACTION COMPLETE!")
    logger.log("=" * 80)
    logger.log(f"Papers processed: {len(processed_pmids)}")
    logger.log(f"Total records extracted: {len(all_records)}")
    if processed_pmids:
        logger.log(f"Records per paper: {len(all_records)/len(processed_pmids):.1f}")
    logger.log(f"\n{cost_tracker.report()}")
    logger.log(f"\nAll outputs saved to: {Config.OUTPUT_DIR}/")
    logger.log(f"  - {Config.OUTPUT_FILE}")
    logger.log(f"  - {Config.CHECKPOINT_FILE}")
    logger.log(f"  - {Config.LOG_FILE}")
    logger.log("=" * 80)


if __name__ == "__main__":
    main()
