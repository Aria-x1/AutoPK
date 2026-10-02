# AutoPK

A pipeline for extracting pharmacokinetic (PK) data from scientific literature using LLMs.

Given a set of PDF papers, AutoPK extracts figures and tables, identifies concentration time curves, digitizes them, resolves drug/dose/species/matrix information, and calculates standard PK parameters (Cmax, Tmax, AUC, half life).

## Pipeline stages

Each folder is a stage in the pipeline, meant to be run roughly in order.

- **01_extraction** - Extracts images, tables, and captions from source PDFs.
- **02_multipanel_split** - Detects multi panel figures and splits them into individual panels, using Claude Vision to read panel labels.
- **03_captions_cleanup** - Diagnoses and fixes caption matching issues (orphaned captions, missing captions, unmatched panels).
- **04_pk_metadata_extraction** - Extracts PK metadata (drug, dose, species, matrix, route, etc.) at the figure and paper level using LLMs. Contains two active approaches: a tiered per-figure extraction and a two-stage paper level scan and extract.
- **05_curve_processing** - Splits digitized concentration time CSVs into one sub-CSV per curve, resolves each curve's drug/analyte/dose via caption parsing and metadata matching, and uses an LLM to review curves that rule-based matching can't resolve. Also re-extracts y-axis units directly from figure images.
- **06_nca_calculation** - Calculates non-compartmental analysis (NCA) parameters (Cmax, Tmax, AUC, half life) from the processed curves. Contains two independent approaches (manifest-based and raw-CSV-based); see each script's docstring for how they differ.
- **07_smiles_lookup** - Resolves drug names to SMILES chemical structure strings. Contains two independent approaches (via SRI name resolution / node normalizer, and via direct PubChem name lookup); see each script's docstring for how they differ.
- **08_quality_control** - Audits raw digitized points for must-have fields, runs a Vision-based QC pass comparing extracted CSVs against their source figures, and validates final calculated PK parameters against literature-reported values.

## Requirements

- Python 3.9+
- `pip install -r requirements.txt` (pandas, numpy, requests, anthropic, openai, pillow, pytesseract, scipy)
- An `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` environment variable, depending on which scripts you run

## Configuration

Scripts read paths from command line arguments, which default to environment variables when set:

```bash
export AUTOPK_BASE_DIR=/path/to/your/AutoPK/data
```

Run any script with `--help` to see its specific arguments.

## Status

Actively maintained and continuously updated as the pipeline evolves.
