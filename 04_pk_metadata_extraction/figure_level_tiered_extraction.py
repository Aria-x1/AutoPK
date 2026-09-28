#!/usr/bin/env python3
"""
Figure-level tiered PK metadata + parameter extraction pipeline.

For each figure in each paper, this pipeline extracts:
  - metadata (drug, dose, route, species, matrix, study design, figure type)
  - PK parameters (half-life, clearance, AUC, Cmax, Tmax, etc.)

Approach: escalating tiers, cheapest first.
  Tier 1A - extract metadata from the figure's caption alone (GPT-4o-mini)
  Tier 1B - if incomplete, supplement missing fields from the full PDF text
            (GPT-4o-mini)
  Tier 2  - extract PK parameter values from the full PDF text (GPT-4o-mini)
  Tier 3  - if quality score is low, re-extract PK parameters by sending the
            PDF directly to Claude Sonnet (used only when needed - this is
            the expensive fallback)

Output: one JSON file per PMID at <output-dir>/<pmid>_extracted.json,
containing a list of per-figure records with tier provenance and a quality
score attached to each.

Requires OPENAI_API_KEY and ANTHROPIC_API_KEY (via environment or a .env file).

Usage:
    python3 figure_level_tiered_extraction.py \
        --base-dir /path/to/AutoPK

    (--base-dir defaults to the AUTOPK_BASE_DIR environment variable if set,
     otherwise the current directory. Expects the standard AutoPK layout
     under it: LLM-POC-Open-Articles/, extracted_images/, and creates
     extracted_metadata/ for output - override any of these individually
     with --papers-dir / --extracted-dir / --output-dir.)
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import json
import openai
from anthropic import Anthropic
import base64
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")


# ===== Tier 1A: extract from caption =====

TIER1A_PROMPT = """
Extract basic metadata from this figure caption.

CAPTION:
{caption}

Return JSON:
{{
  "drug_info": {{
    "drug_name": "full drug name",
    "dose": "dose with unit (e.g., 300 mg)",
    "route": "oral/IV/SC/IM",
    "concomitant_drugs": ["other drugs with doses"]
  }},

  "study_subject": {{
    "species": "human/mouse/rat/dog",
    "population": "adult/pediatric/elderly",
    "health_status": "healthy/HIV-infected/disease",
    "sample_size": integer or null
  }},

  "bioanalysis": {{
    "matrix": "plasma/serum/blood/urine",
    "analyte": "parent drug/metabolite"
  }},

  "study_design": {{
    "design_type": "single_dose/steady_state/multiple_dose",
    "data_type": "mean/median/individual"
  }},

  "figure_classification": {{
    "figure_type": "one of the types below",
    "has_pk_curve": true/false
  }}
}}

FIGURE TYPES (choose the most appropriate):
- "concentration-time curve" - plasma/serum concentration over time
- "PK/PD relationship" - relationship between PK and pharmacodynamic effects
- "population distribution histogram" - distribution of PK parameters
- "model validation plot" - observed vs predicted concentrations
- "goodness of fit plot" - model fitting diagnostics
- "clearance vs age/weight relationship" - clearance vs demographics
- "antiviral response over time" - viral load or CD4 over time
- "bone mineral density comparison" - BMD measurements
- "survival curve" - Kaplan-Meier survival
- "disease-free survival curve" - DFS analysis
- "study design schema" - flowchart/diagram
- "other" - none of the above

CLASSIFICATION KEYWORDS:
- "concentration", "plasma", "profile" -> concentration-time curve
- "observed vs predicted", "validation" -> model validation plot
- "goodness of fit", "residuals" -> goodness of fit plot
- "distribution", "histogram" -> population distribution histogram
- "survival", "Kaplan-Meier" -> survival curve
- "clearance vs", "relationship with age/weight" -> clearance vs age/weight relationship

Extract ONLY from caption. Use null for missing info.
Set has_pk_curve = true only for concentration-time curves.
Return ONLY valid JSON.
"""


# ===== Tier 1B: supplement from PDF =====

TIER1B_SUPPLEMENT_PROMPT = """
The caption did not provide complete metadata. Extract MISSING information from the paper.

CAPTION:
{caption}

CURRENT METADATA:
{current_metadata}

PAPER TEXT:
{paper_text}

Fill in MISSING fields (null or empty):
{{
  "drug_info": {{
    "drug_name": "fill if null",
    "dose": "fill if null",
    "route": "fill if null",
    "concomitant_drugs": ["fill if empty"]
  }},

  "study_subject": {{
    "species": "fill if null",
    "population": "fill if null",
    "health_status": "fill if null",
    "sample_size": "fill if null"
  }},

  "bioanalysis": {{
    "matrix": "fill if null",
    "analyte": "fill if null"
  }},

  "study_design": {{
    "design_type": "fill if null",
    "data_type": "fill if null"
  }},

  "figure_classification": {{
    "figure_type": "fill if null",
    "has_pk_curve": "fill if null"
  }}
}}

Keep existing non-null values unchanged.
Return ONLY valid JSON.
"""


# ===== Tier 2: PK parameter extraction =====

TIER2_PROMPT = """
Extract ALL pharmacokinetic parameters from this paper.

CONDITION: {condition_description}

PAPER TEXT:
{paper_text}

Search Abstract, Results, Tables, Discussion. Return JSON:
{{
  "pk_parameters": {{
    "half_life": {{
      "terminal": {{"value": float, "unit": "hours", "source": "Table X"}},
      "distribution": {{"value": float, "unit": "hours", "source": "..."}}
    }},
    "clearance": {{
      "total": {{"value": float, "unit": "L/h", "source": "..."}},
      "renal": {{"value": float, "unit": "mL/min/1.73m2", "source": "..."}}
    }},
    "volume_distribution": {{"value": float, "unit": "L", "source": "..."}},
    "auc": {{"value": float, "unit": "mg*h/L", "source": "..."}},
    "cmax": {{"value": float, "unit": "mg/L", "source": "..."}},
    "tmax": {{"value": float, "unit": "hours", "source": "..."}},
    "cmin": {{"value": float, "unit": "mg/L", "source": "..."}},
    "bioavailability": {{"value": float, "unit": "percent", "source": "..."}},
    "protein_binding": {{"percent_bound": float, "source": "..."}},
    "renal_excretion": {{"percent_of_dose": float, "source": "..."}}
  }},
  "extraction_quality": {{
    "confidence": "high/medium/low",
    "parameters_found": [...],
    "parameters_not_found": [...]
  }}
}}

CRITICAL: Renal clearance is VERY IMPORTANT.
Return ONLY valid JSON.
"""


# ===== Extractor class =====

class PharmacokineticsExtractor:
    """Tiered PK metadata + parameter extractor."""

    def __init__(self):
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')

        if not openai_key or not anthropic_key:
            raise ValueError(
                "API keys not found! Set OPENAI_API_KEY and ANTHROPIC_API_KEY "
                "as environment variables or in a .env file"
            )

        self.gpt = openai.OpenAI(api_key=openai_key)
        self.claude = Anthropic(api_key=anthropic_key)

        self.stats = {
            'tier1a_caption': 0,
            'tier1b_supplement': 0,
            'tier2_gpt': 0,
            'tier3_claude': 0,
            'total_cost': 0
        }

        print(f"API keys loaded")
        print(f"   OpenAI:    {openai_key[:15]}...")
        print(f"   Anthropic: {anthropic_key[:15]}...")

    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            reader = PdfReader(pdf_path)
            text_parts = []

            for page in reader.pages:
                text_parts.append(page.extract_text())

            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"    WARNING: PDF error: {e}")
            return ""

    def check_metadata_completeness(self, metadata: dict) -> dict:
        """Check metadata completeness"""

        missing = []

        critical_fields = {
            'drug_info.drug_name': metadata.get('drug_info', {}).get('drug_name'),
            'drug_info.dose': metadata.get('drug_info', {}).get('dose'),
            'drug_info.route': metadata.get('drug_info', {}).get('route'),
            'bioanalysis.matrix': metadata.get('bioanalysis', {}).get('matrix'),
            'study_subject.species': metadata.get('study_subject', {}).get('species'),
            'figure_classification.figure_type': metadata.get('figure_classification', {}).get('figure_type'),
        }

        for field, value in critical_fields.items():
            if not value or value == 'null':
                missing.append(field)

        concomitant = metadata.get('drug_info', {}).get('concomitant_drugs', [])
        if not isinstance(concomitant, list):
            missing.append('drug_info.concomitant_drugs')

        return {
            'is_complete': len(missing) == 0,
            'missing_fields': missing,
            'completeness_score': int((6 - len(missing)) / 6 * 100)
        }

    def tier1a_extract_from_caption(self, caption: str) -> dict:
        """Tier 1A: extract from caption"""

        try:
            response = self.gpt.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a pharmacokinetics expert."},
                    {"role": "user", "content": TIER1A_PROMPT.format(caption=caption)}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=800
            )

            self.stats['tier1a_caption'] += 1
            self.stats['total_cost'] += 0.0005

            result = json.loads(response.choices[0].message.content)
            result['_tier1a_source'] = 'caption'
            return result

        except Exception as e:
            print(f"    ERROR: Tier 1A failed: {e}")
            return {}

    def tier1b_supplement_from_pdf(self, caption: str, current_metadata: dict,
                                    pdf_text: str) -> dict:
        """Tier 1B: supplement from PDF"""

        if len(pdf_text) > 40000:
            pdf_text = pdf_text[:40000]

        try:
            response = self.gpt.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a pharmacokinetics expert."},
                    {"role": "user", "content": TIER1B_SUPPLEMENT_PROMPT.format(
                        caption=caption,
                        current_metadata=json.dumps(current_metadata, indent=2),
                        paper_text=pdf_text
                    )}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1000
            )

            self.stats['tier1b_supplement'] += 1
            self.stats['total_cost'] += 0.001

            result = json.loads(response.choices[0].message.content)
            result['_tier1b_supplemented'] = True
            return result

        except Exception as e:
            print(f"    ERROR: Tier 1B failed: {e}")
            return current_metadata

    def tier1_extract_metadata(self, caption: str, pdf_text: str) -> dict:
        """Tier 1: full metadata extraction"""

        metadata = self.tier1a_extract_from_caption(caption)

        if not metadata:
            return {}

        completeness = self.check_metadata_completeness(metadata)

        if not completeness['is_complete']:
            print(f"    Tier 1B: caption {completeness['completeness_score']}% complete, supplementing...")
            print(f"    Missing: {', '.join(completeness['missing_fields'][:3])}")

            supplemented = self.tier1b_supplement_from_pdf(caption, metadata, pdf_text)
            metadata = self._merge_metadata(metadata, supplemented)

            completeness_after = self.check_metadata_completeness(metadata)
            print(f"    After: {completeness_after['completeness_score']}% complete")

        metadata['_tier1_completeness'] = completeness

        return metadata

    def _merge_metadata(self, original: dict, supplement: dict) -> dict:
        """Merge metadata"""

        merged = original.copy()

        for key in ['drug_info', 'study_subject', 'bioanalysis', 'study_design', 'figure_classification']:
            if key in supplement and isinstance(supplement[key], dict):
                if key not in merged:
                    merged[key] = {}

                for subkey, value in supplement[key].items():
                    if value not in [None, '', 'null', []]:
                        merged[key][subkey] = value

        if '_tier1b_supplemented' in supplement:
            merged['_tier1b_supplemented'] = True

        return merged

    def tier2_extract_pk_parameters(self, caption: str, metadata: dict,
                                     pdf_text: str) -> dict:
        """Tier 2: PK parameter extraction"""

        drug_name = metadata.get('drug_info', {}).get('drug_name', 'unknown')
        dose = metadata.get('drug_info', {}).get('dose', '')
        concomitant = metadata.get('drug_info', {}).get('concomitant_drugs', [])

        condition_desc = f"{drug_name} {dose}"
        if concomitant:
            condition_desc += f" with {', '.join(concomitant)}"

        if len(pdf_text) > 120000:
            pdf_text = pdf_text[:60000] + "\n\n[...truncated...]\n\n" + pdf_text[-60000:]

        try:
            response = self.gpt.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a pharmacokinetics expert."},
                    {"role": "user", "content": TIER2_PROMPT.format(
                        condition_description=condition_desc,
                        paper_text=pdf_text
                    )}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=2000
            )

            self.stats['tier2_gpt'] += 1
            self.stats['total_cost'] += 0.002

            result = json.loads(response.choices[0].message.content)
            result['_tier2_model'] = 'gpt-4o-mini'
            return result

        except Exception as e:
            print(f"    ERROR: Tier 2 failed: {e}")
            return {
                'pk_parameters': {},
                'extraction_quality': {'confidence': 'failed', 'error': str(e)}
            }

    def tier3_reextract_with_claude(self, caption: str, metadata: dict,
                                     pdf_path: Path, issues: list) -> dict:
        """Tier 3: re-extract with Claude"""

        try:
            with open(pdf_path, 'rb') as f:
                pdf_data = base64.b64encode(f.read()).decode()

            drug_name = metadata.get('drug_info', {}).get('drug_name', 'unknown')
            dose = metadata.get('drug_info', {}).get('dose', '')

            prompt = f"""
Previous extraction had issues: {json.dumps(issues)}

CAPTION: {caption}
CONDITION: {drug_name} {dose}

Extract ALL PK parameters with MAXIMUM precision.
Focus on: renal clearance, half-life, AUC, Cmax, Cmin.

Return JSON with same structure.
"""

            message = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_data
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            self.stats['tier3_claude'] += 1
            self.stats['total_cost'] += 0.035

            response_text = message.content[0].text
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]

            result = json.loads(response_text)
            result['_tier3_model'] = 'claude-sonnet-4'
            result['_reprocessed'] = True
            return result

        except Exception as e:
            print(f"    ERROR: Tier 3 failed: {e}")
            return {
                'pk_parameters': {},
                'extraction_quality': {'confidence': 'failed', 'error': str(e)}
            }

    def assess_quality(self, tier1_result: dict, tier2_result: dict) -> dict:
        """Assess quality"""

        issues = []
        score = 100

        tier1_completeness = tier1_result.get('_tier1_completeness', {})
        if tier1_completeness.get('completeness_score', 0) < 80:
            issues.append(f"Incomplete metadata ({tier1_completeness.get('completeness_score')}%)")
            score -= 15

        if not tier1_result.get('drug_info', {}).get('drug_name'):
            issues.append("Missing drug name")
            score -= 30

        pk_params = (tier2_result.get('pk_parameters', {}) or
                     tier2_result.get('pharmacokinetic_parameters', {}))

        half_life = pk_params.get('half_life', {})
        has_half_life = False

        if isinstance(half_life, dict):
            has_half_life = bool(half_life.get('terminal', {}).get('value'))
        elif isinstance(half_life, str):
            has_half_life = 'not reported' not in half_life.lower()

        if not has_half_life:
            issues.append("Missing half_life")
            score -= 25

        clearance = pk_params.get('clearance', {})
        has_renal_cl = False

        if isinstance(clearance, dict):
            renal_cl = clearance.get('renal', {})
            if isinstance(renal_cl, dict):
                has_renal_cl = bool(renal_cl.get('value'))
            elif isinstance(renal_cl, str):
                has_renal_cl = 'not reported' not in renal_cl.lower()

        if not has_renal_cl:
            issues.append("Missing renal clearance (CRITICAL)")
            score -= 25

        has_auc = (pk_params.get('auc', {}).get('value') or
                   pk_params.get('AUC24h', {}).get('value') or
                   pk_params.get('AUC12h', {}).get('value'))

        if not has_auc:
            issues.append("Missing AUC")
            score -= 20

        confidence = tier2_result.get('extraction_quality', {}).get('confidence', 'low')
        if confidence == 'low':
            issues.append("Low confidence")
            score -= 20

        needs_reprocessing = score < 60

        return {
            'score': max(0, score),
            'issues': issues,
            'needs_reprocessing': needs_reprocessing,
            'confidence': 'high' if score >= 80 else 'medium' if score >= 60 else 'low'
        }

    def extract_single_row(self, row: dict, pmid: str, pdf_path: Path) -> dict:
        """Extract a single figure's row"""

        image_filename = row['image_filename']
        caption = row['caption']

        print(f"  {image_filename}")

        print(f"    Reading PDF...")
        pdf_text = self.extract_pdf_text(pdf_path)

        if not pdf_text:
            return {
                'metadata': {
                    'pmid': pmid,
                    'image_filename': image_filename,
                    'caption': caption,
                    'error': 'PDF reading failed'
                }
            }

        print(f"    Tier 1: extracting metadata...")
        tier1_result = self.tier1_extract_metadata(caption, pdf_text)

        print(f"    Tier 2: extracting PK parameters...")
        tier2_result = self.tier2_extract_pk_parameters(caption, tier1_result, pdf_text)

        quality = self.assess_quality(tier1_result, tier2_result)
        print(f"    Quality: {quality['score']}/100 ({quality['confidence']})")

        if quality['issues']:
            print(f"    Issues: {', '.join(quality['issues'][:2])}")

        if quality['needs_reprocessing']:
            print(f"    Reprocessing with Claude...")
            tier2_result = self.tier3_reextract_with_claude(
                caption, tier1_result, pdf_path, quality['issues']
            )
            quality = self.assess_quality(tier1_result, tier2_result)
            print(f"    New quality: {quality['score']}/100")

        result = {
            'metadata': {
                'pmid': pmid,
                'image_filename': image_filename,
                'caption': caption
            },
            **tier1_result,
            **tier2_result,
            'quality_assessment': quality
        }

        return result


# ===== Main pipeline =====

def process_pmid(pmid: str, extractor: PharmacokineticsExtractor,
                  papers_dir: Path, extracted_dir: Path, output_dir: Path):
    """Process a single PMID"""

    print(f"\n{'='*70}")
    print(f"Processing PMID: {pmid}")
    print(f"{'='*70}")

    pmid_dir = extracted_dir / pmid

    captions_file = None
    for ext in ['.csv', '.xlsx', '.xls', '.txt', '.json']:
        test_file = pmid_dir / f"captions{ext}"
        if test_file.exists():
            captions_file = test_file
            break

    if not captions_file:
        print(f"  No captions file")
        return None

    try:
        if captions_file.suffix in ['.csv']:
            captions_df = pd.read_csv(captions_file)
        elif captions_file.suffix in ['.xlsx', '.xls']:
            captions_df = pd.read_excel(captions_file)
        else:
            print(f"  Unsupported format: {captions_file.suffix}")
            return None
    except Exception as e:
        print(f"  Read error: {e}")
        return None

    print(f"  Found {len(captions_df)} figures")

    pdf_path = papers_dir / f"{pmid}.pdf"
    if not pdf_path.exists():
        print(f"  PDF not found")
        return None

    results = []

    for idx, row in captions_df.iterrows():
        try:
            result = extractor.extract_single_row(row.to_dict(), pmid, pdf_path)
            results.append(result)
            print(f"    Done")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'metadata': {'pmid': pmid, 'error': str(e)}})

    output_file = output_dir / f"{pmid}_extracted.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {output_file}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK project root (default: $AUTOPK_BASE_DIR env var, "
                          "or current directory). Expects LLM-POC-Open-Articles/ "
                          "and extracted_images/ under it unless overridden below.")
    ap.add_argument("--papers-dir", default=None,
                     help="Override: folder of source PDFs (default: <base-dir>/LLM-POC-Open-Articles)")
    ap.add_argument("--extracted-dir", default=None,
                     help="Override: folder of per-PMID extracted_images/captions (default: <base-dir>/extracted_images)")
    ap.add_argument("--output-dir", default=None,
                     help="Override: where to write <pmid>_extracted.json files (default: <base-dir>/extracted_metadata)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    papers_dir = Path(args.papers_dir) if args.papers_dir else base_dir / "LLM-POC-Open-Articles"
    extracted_dir = Path(args.extracted_dir) if args.extracted_dir else base_dir / "extracted_images"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "extracted_metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Figure-Level Tiered PK Extraction Pipeline - BATCH MODE")
    print("="*70)

    try:
        extractor = PharmacokineticsExtractor()
    except ValueError as e:
        print(f"\n{e}")
        return

    pmid_folders = sorted([d.name for d in extracted_dir.iterdir()
                          if d.is_dir() and not d.name.startswith('.')])

    print(f"\nFound {len(pmid_folders)} PMIDs to process")
    print(f"Output directory: {output_dir}\n")

    success_count = 0
    fail_count = 0
    total_figures = 0

    for i, pmid in enumerate(pmid_folders, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(pmid_folders)}] Processing PMID: {pmid}")
        print(f"{'='*70}")

        try:
            results = process_pmid(pmid, extractor, papers_dir, extracted_dir, output_dir)

            if results:
                success_count += 1
                total_figures += len(results)
                print(f"{pmid}: extracted {len(results)} figures")
            else:
                fail_count += 1
                print(f"{pmid}: no results")

        except Exception as e:
            fail_count += 1
            print(f"{pmid}: error - {e}")
            import traceback
            traceback.print_exc()

        if i % 10 == 0:
            progress = i / len(pmid_folders) * 100
            print(f"\n{'-'*70}")
            print(f"Progress: {i}/{len(pmid_folders)} ({progress:.1f}%)")
            print(f"Success: {success_count} | Failed: {fail_count}")
            print(f"Total figures extracted: {total_figures}")
            print(f"Current cost: ${extractor.stats['total_cost']:.2f}")
            print(f"{'-'*70}")

    print(f"\n{'='*70}")
    print("BATCH EXTRACTION COMPLETED")
    print(f"{'='*70}")
    print(f"Total PMIDs processed: {len(pmid_folders)}")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"Total figures extracted: {total_figures}")
    print(f"\nModel usage:")
    print(f"  Tier 1A (caption):      {extractor.stats['tier1a_caption']}")
    print(f"  Tier 1B (supplement):   {extractor.stats['tier1b_supplement']}")
    print(f"  Tier 2 (PK params):     {extractor.stats['tier2_gpt']}")
    print(f"  Tier 3 (Claude):        {extractor.stats['tier3_claude']}")
    print(f"\nTotal cost: ${extractor.stats['total_cost']:.2f}")
    print(f"Average cost per figure: ${extractor.stats['total_cost']/max(total_figures,1):.4f}")
    print(f"{'='*70}\n")

    summary = {
        'total_pmids': len(pmid_folders),
        'success': success_count,
        'failed': fail_count,
        'total_figures': total_figures,
        'model_usage': extractor.stats,
        'timestamp': str(pd.Timestamp.now())
    }

    summary_file = output_dir / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}\n")


if __name__ == "__main__":
    main()
