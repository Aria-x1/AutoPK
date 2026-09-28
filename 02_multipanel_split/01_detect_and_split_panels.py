#!/usr/bin/env python3
"""
STEP 1: Split multipanel figures
  1. Use Claude Vision to detect multipanel structure and panel bounding boxes
  2. Crop each panel out with PIL
  3. Extract each panel's title text with OCR
  4. Fall back to Vision when OCR confidence is low
  5. Write split_metadata.json (does NOT touch captions.csv)

Panel label convention:
  - Every panel label is normalized to a single uppercase letter (A-Z).
  - If Claude's response can't be reduced to a single letter, the panel is
    marked panel_label_status = "NEEDS_REVIEW" and gets a placeholder label
    (PANEL1, PANEL2, ...) instead of guessing. This is what previously
    produced messy filenames like "_a)", "_(a) Oral study", etc.

Output filename convention:
  {original_stem}_panel{LETTER}{ext}
  e.g. 18573931_p04_01_ac0bfd2a_panelA.png
  (the "panel" token avoids collisions with the hash suffix in the stem)

Usage:
    python3 01_detect_and_split_panels.py <pmid> --base /path/to/extracted_images
    (--base defaults to the AUTOPK_IMAGES_DIR environment variable if set,
     otherwise the current directory)
"""

import os
import sys
import re
import argparse
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import pytesseract
from anthropic import Anthropic
import base64

DEFAULT_IMAGES_DIR = os.environ.get("AUTOPK_IMAGES_DIR", ".")


def normalize_panel_label(raw_label, fallback_index):
    """
    Normalize whatever Claude returns for a panel label into a single
    uppercase letter A-Z.

    Returns (label, status):
      status == "OK"            -> raw_label was already a clean single letter
      status == "NEEDS_REVIEW"  -> raw_label was messy; using a placeholder
                                    (PANEL1, PANEL2, ...) instead of guessing
    """
    if isinstance(raw_label, str):
        s = raw_label.strip()
        if re.fullmatch(r"[A-Za-z]", s):
            return s.upper(), "OK"

    return f"PANEL{fallback_index}", "NEEDS_REVIEW"


class MultiPanelSplitter:
    def __init__(self, pmid_folder_path):
        self.pmid_folder = Path(pmid_folder_path)
        self.pmid = self.pmid_folder.name

        if not self.pmid_folder.exists():
            print(f"ERROR: path does not exist: {pmid_folder_path}")
            sys.exit(1)

        self.client = Anthropic()
        self.metadata = {
            'pmid': self.pmid,
            'timestamp': datetime.now().isoformat(),
            'processed_images': {}
        }

    def get_image_files(self):
        """Get all image files in the folder, excluding ones already split
        (files ending in _panel<LETTER>)."""
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            images.extend(self.pmid_folder.glob(ext))

        images = [
            img for img in images
            if not re.search(r"_panel[A-Za-z]\.(png|jpg|jpeg)$", img.name, re.IGNORECASE)
        ]

        return sorted(images)

    def image_to_base64(self, image_path):
        with open(image_path, 'rb') as img_file:
            return base64.standard_b64encode(img_file.read()).decode('utf-8')

    def detect_multipanel(self, image_path):
        """Use Claude Vision to detect multipanel structure."""
        print(f"\n  Detecting: {image_path.name}...")

        base64_image = self.image_to_base64(image_path)

        prompt = """Analyze this image and determine whether it is a multipanel figure.

Respond in this JSON format:
{
  "is_multipanel": true/false,
  "num_panels": <number>,
  "panels": [
    {
      "label": "A",
      "bbox": [x1, y1, x2, y2],
      "title": "panel title text if any",
      "position": "brief description of panel position"
    },
    ...
  ],
  "notes": "any other observations"
}

If this is a single panel or panels cannot be identified, set is_multipanel to false.
Each "label" MUST be a single letter (A, B, C, ...) with no extra characters,
punctuation, or words attached.
bbox uses normalized coordinates (0 to 1), where [0,0] is top-left and [1,1] is bottom-right.
"""

        try:
            message = self.client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png" if image_path.suffix.lower() == '.png' else "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            response_text = message.content[0].text

            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                print(f"    WARNING: could not parse response")
                return {"is_multipanel": False}

        except Exception as e:
            print(f"    ERROR: vision detection failed: {e}")
            return {"is_multipanel": False}

    def split_image(self, image_path, panels_info):
        """Split image according to bbox, normalizing panel labels."""
        print(f"  Splitting image...")

        img = Image.open(image_path)
        width, height = img.size

        split_results = []

        for i, panel in enumerate(panels_info.get('panels', []), start=1):
            raw_label = panel.get('label', '')
            label, label_status = normalize_panel_label(raw_label, i)
            if label_status == "NEEDS_REVIEW":
                print(f"    WARNING: messy panel label {raw_label!r} from model, "
                      f"using placeholder {label} (flagged for review)")

            bbox = panel.get('bbox', [0, 0, 1, 1])

            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            panel_img = img.crop((x1, y1, x2, y2))

            output_filename = f"{image_path.stem}_panel{label}{image_path.suffix}"
            output_path = self.pmid_folder / output_filename
            panel_img.save(output_path)

            print(f"    Saved: {output_filename}")

            split_results.append({
                'label': label,
                'label_status': label_status,
                'raw_label': str(raw_label),
                'output_file': output_filename,
                'output_path': str(output_path),
                'bbox': bbox
            })

        return split_results

    def extract_text_with_ocr(self, image_path):
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)

            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['confidence'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return text.strip(), avg_confidence / 100

        except Exception as e:
            print(f"    WARNING: OCR failed: {e}")
            return "", 0

    def extract_panel_title_with_vision(self, image_path):
        print(f"      Re-extracting title with Vision...")

        base64_image = self.image_to_base64(image_path)

        prompt = """Extract the panel's title text from this image.
Return only the title text, nothing else.
If there is no clear title, return "NO_TITLE"."""

        try:
            message = self.client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png" if image_path.suffix.lower() == '.png' else "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            return message.content[0].text.strip()

        except Exception as e:
            print(f"      ERROR: vision title extraction failed: {e}")
            return ""

    def process_image(self, image_path):
        base_name = image_path.stem

        print(f"\nProcessing: {image_path.name}")

        panels_info = self.detect_multipanel(image_path)

        if not panels_info.get('is_multipanel', False):
            print(f"  Single panel or not recognized as multipanel, skipping")
            self.metadata['processed_images'][base_name] = {
                'status': 'single_panel',
                'action': 'skipped'
            }
            return

        split_results = self.split_image(image_path, panels_info)

        panel_titles = {}

        for split_result in split_results:
            label = split_result['label']
            output_path = Path(split_result['output_path'])

            print(f"\n  Extracting title for Panel {label}...")

            ocr_text, ocr_confidence = self.extract_text_with_ocr(output_path)

            print(f"    OCR result: {ocr_text[:50]}... (confidence: {ocr_confidence:.2f})")

            if ocr_confidence < 0.7:
                vision_text = self.extract_panel_title_with_vision(output_path)
                if vision_text and vision_text != "NO_TITLE":
                    panel_titles[label] = {
                        'title': vision_text,
                        'extraction_method': 'vision',
                        'confidence': 0.95
                    }
                    print(f"    Vision extracted: {vision_text}")
                else:
                    panel_titles[label] = {
                        'title': ocr_text,
                        'extraction_method': 'ocr',
                        'confidence': ocr_confidence
                    }
            else:
                panel_titles[label] = {
                    'title': ocr_text,
                    'extraction_method': 'ocr',
                    'confidence': ocr_confidence
                }

        self.metadata['processed_images'][base_name] = {
            'status': 'multipanel_split',
            'original_file': image_path.name,
            'num_panels': len(split_results),
            'panels': {
                sr['label']: {
                    'output_file': sr['output_file'],
                    'label_status': sr['label_status'],
                    'raw_label': sr['raw_label'],
                    **panel_titles.get(sr['label'], {'title': '', 'extraction_method': 'none', 'confidence': 0})
                }
                for sr in split_results
            }
        }

    def run(self):
        print("\n" + "=" * 80)
        print(f"STEP 1: Split multipanel figures - PMID {self.pmid}")
        print("=" * 80)

        images = self.get_image_files()

        if not images:
            print(f"No images found in folder")
            return

        print(f"Found {len(images)} image file(s)\n")

        for i, img_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}]", end=" ")
            self.process_image(img_path)

        metadata_file = self.pmid_folder / 'split_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        n_needs_review = sum(
            1 for img in self.metadata['processed_images'].values()
            if img.get('status') == 'multipanel_split'
            for p in img.get('panels', {}).values()
            if p.get('label_status') == 'NEEDS_REVIEW'
        )

        print("\n" + "=" * 80)
        print(f"STEP 1 complete")
        print(f"Metadata saved to: {metadata_file}")
        print(f"Processed {len(self.metadata['processed_images'])} image(s)")
        if n_needs_review:
            print(f"WARNING: {n_needs_review} panel(s) flagged NEEDS_REVIEW "
                  f"(model did not return a clean single-letter label)")
        print("=" * 80)
        print("\nNext: review split_metadata.json to confirm the split is correct,")
        print("then run: python3 02_update_captions_for_panels.py <pmid>\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("pmid", help="PMID subfolder to process, e.g. 18573931")
    ap.add_argument("--base", default=DEFAULT_IMAGES_DIR,
                     help="Path to the extracted_images root folder "
                          "(default: $AUTOPK_IMAGES_DIR env var, or current directory)")
    args = ap.parse_args()

    pmid_folder = Path(args.base) / args.pmid

    splitter = MultiPanelSplitter(pmid_folder)
    splitter.run()
