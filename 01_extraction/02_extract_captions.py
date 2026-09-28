#!/usr/bin/env python3
"""
Extract figure captions from all papers and match them to images using Claude API.

Usage:
    python extract_captions_all_papers.py --api-key YOUR_CLAUDE_API_KEY
    python extract_captions_all_papers.py --api-key YOUR_CLAUDE_API_KEY --delete-manifest
    
    Or set environment variable:
    export ANTHROPIC_API_KEY=your_api_key
    python extract_captions_all_papers.py --delete-manifest
"""

import os
import sys
import base64
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

try:
    import fitz  # PyMuPDF for PDF extraction
except ImportError:
    fitz = None

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic library not found. Install with: pip install anthropic")
    sys.exit(1)


class BatchCaptionExtractor:
    def __init__(self, api_key: str = None, delete_manifest: bool = False):
        """Initialize the batch caption extractor."""
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(
                "API key not found. Please provide --api-key or set ANTHROPIC_API_KEY environment variable"
            )
        
        self.client = Anthropic(api_key=api_key)
        self.base_path = Path.home() / "Documents/MML/AutoPK"
        self.papers_dir = self.base_path / "LLM-POC-Open-Articles"
        self.images_base_dir = self.base_path / "extracted_images"
        self.delete_manifest = delete_manifest
        
        self.log(f"📁 Base path: {self.base_path}")
        self.log(f"📄 Papers dir: {self.papers_dir}")
        self.log(f"🖼️  Images base dir: {self.images_base_dir}")
        if self.delete_manifest:
            self.log(f"🗑️  Will delete manifest.csv files")
        
        self._validate_paths()
        self.stats = {
            'total_papers': 0,
            'successful_papers': 0,
            'failed_papers': 0,
            'skipped_papers': 0,
            'total_images': 0,
            'images_with_caption': 0,
            'images_without_caption': 0,
        }
    
    def _validate_paths(self):
        """Validate that required paths exist."""
        if not self.papers_dir.exists():
            raise FileNotFoundError(f"Papers directory not found: {self.papers_dir}")
        self.log(f"✅ Papers directory found")
        
        if not self.images_base_dir.exists():
            raise FileNotFoundError(f"Images base directory not found: {self.images_base_dir}")
        self.log(f"✅ Images base directory found")
    
    def log(self, msg: str, end: str = "\n"):
        """Print formatted log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}", end=end)
    
    def get_paper_folders(self) -> List[Path]:
        """Get all paper folders in extracted_images."""
        folders = [f for f in self.images_base_dir.iterdir() if f.is_dir()]
        folders.sort()
        self.log(f"🎯 Found {len(folders)} paper folders")
        for folder in folders:
            pmid = folder.name
            pdf_exists = (self.papers_dir / f"{pmid}.pdf").exists()
            status = "✅" if pdf_exists else "⚠️"
            self.log(f"  {status} {pmid}")
        return folders
    
    def get_image_files(self, images_dir: Path) -> List[Path]:
        """Get all image files from a directory."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        images = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        images.sort()
        return images
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        if fitz is None:
            return self._extract_with_claude(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num, page in enumerate(doc):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            self.log(f"    ⚠️  Error extracting with PyMuPDF: {e}")
            return self._extract_with_claude(pdf_path)
    
    def _extract_with_claude(self, pdf_path: Path) -> str:
        """Fallback: Extract PDF using Claude's document understanding."""
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.standard_b64encode(f.read()).decode('utf-8')
        
        message = self.client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=4096,
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
                        {
                            "type": "text",
                            "text": "Extract all figure captions from this PDF. Format: 'Figure X: [caption text]'. Include all text exactly as written."
                        }
                    ]
                }
            ]
        )
        
        return message.content[0].text
    
    def extract_figure_captions(self, pdf_text: str) -> Dict[str, str]:
        """Extract all figure captions from PDF text."""
        message = self.client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract ALL figure captions from this PDF text. 
Return as JSON dictionary with figure names as keys and full caption text as values.
Format: {{"Figure 1": "full caption text", "Figure 2": "full caption text", ...}}

Important: Return ONLY valid JSON, no other text.

PDF Text:
{pdf_text}"""
                }
            ]
        )
        
        try:
            response_text = message.content[0].text
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                captions = json.loads(json_str)
                return captions
        except json.JSONDecodeError:
            pass
        
        return {}
    
    def image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 string."""
        with open(image_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')
    
    def get_image_media_type(self, image_path: Path) -> str:
        """Get media type for image."""
        suffix = image_path.suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(suffix, 'image/png')
    
    def identify_figure_caption(self, image_path: Path, available_captions: Dict[str, str]) -> str:
        """Use Claude to identify which caption matches the image."""
        image_base64 = self.image_to_base64(image_path)
        media_type = self.get_image_media_type(image_path)
        
        # Check if image is a scientific figure
        check_message = self.client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Is this a scientific figure/chart/graph/table (not a logo or decorative image)? Answer yes or no only."
                        }
                    ]
                }
            ]
        )
        
        is_scientific = "yes" in check_message.content[0].text.lower()
        
        if not is_scientific:
            return "No caption found"
        
        # Match to caption
        captions_list = "\n".join([f"- {k}: {v}" for k, v in available_captions.items()])
        
        match_message = self.client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""This is a scientific figure from a research paper. Match it to the most appropriate caption from the list below.

Available captions:
{captions_list}

Return ONLY the exact caption text that matches this figure. If no caption matches, return "No caption found".
Do not add any explanation, just return the caption text exactly as it appears in the list."""
                        }
                    ]
                }
            ]
        )
        
        return match_message.content[0].text.strip()
    
    def process_paper(self, paper_folder: Path) -> bool:
        """Process a single paper folder."""
        pmid = paper_folder.name
        pdf_path = self.papers_dir / f"{pmid}.pdf"
        output_csv = paper_folder / "captions.csv"
        
        # Skip if already processed
        if output_csv.exists():
            self.log(f"⏭️  PMID {pmid}: Already processed (captions.csv exists)")
            self.stats['skipped_papers'] += 1
            return True
        
        self.log(f"\n{'='*60}")
        self.log(f"📖 Processing PMID: {pmid}")
        self.log(f"{'='*60}")
        
        # Check if PDF exists
        if not pdf_path.exists():
            self.log(f"❌ PDF not found: {pdf_path}")
            self.stats['failed_papers'] += 1
            return False
        
        try:
            # Extract PDF text
            self.log(f"  📖 Extracting text from PDF...")
            pdf_text = self.extract_pdf_text(pdf_path)
            self.log(f"  ✅ Extracted {len(pdf_text)} characters")
            
            # Extract figure captions
            self.log(f"  🔍 Extracting figure captions...")
            pdf_captions = self.extract_figure_captions(pdf_text)
            
            if not pdf_captions:
                self.log(f"  ⚠️  No captions found in PDF")
                pdf_captions = {}
            else:
                self.log(f"  ✅ Extracted {len(pdf_captions)} captions")
            
            # Get image files
            images = self.get_image_files(paper_folder)
            
            if not images:
                self.log(f"  ⚠️  No images found in folder")
                self.stats['successful_papers'] += 1
                return True
            
            self.log(f"  🎯 Found {len(images)} images")
            
            # Process each image
            results = []
            for i, image_path in enumerate(images, 1):
                self.log(f"    [{i}/{len(images)}] {image_path.name}...", end=" ")
                caption = self.identify_figure_caption(image_path, pdf_captions)
                results.append((image_path.name, caption))
                
                if caption != "No caption found":
                    self.log(f"✅")
                    self.stats['images_with_caption'] += 1
                else:
                    self.log(f"⏭️")
                    self.stats['images_without_caption'] += 1
                
                self.stats['total_images'] += 1
            
            # Save to CSV
            captions_csv = paper_folder / "captions.csv"
            with open(captions_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image_filename', 'caption'])
                for filename, caption in results:
                    writer.writerow([filename, caption])
            
            self.log(f"  💾 Saved results to {captions_csv.name}")
            
            # Delete manifest.csv if requested
            if self.delete_manifest:
                manifest_path = paper_folder / "manifest.csv"
                if manifest_path.exists():
                    manifest_path.unlink()
                    self.log(f"  🗑️  Deleted manifest.csv")
            
            self.stats['successful_papers'] += 1
            return True
        
        except Exception as e:
            self.log(f"  ❌ Error processing {pmid}: {e}")
            import traceback
            traceback.print_exc()
            self.stats['failed_papers'] += 1
            return False
    
    def run(self):
        """Run the full extraction pipeline for all papers."""
        self.log("🚀 Starting batch caption extraction pipeline...\n")
        
        try:
            # Get all paper folders
            all_paper_folders = self.get_paper_folders()
            
            # Filter to only folders without captions.csv (not yet completed)
            pending_folders = []
            skipped_count = 0
            for folder in all_paper_folders:
                captions_file = folder / "captions.csv"
                if captions_file.exists():
                    skipped_count += 1
                else:
                    pending_folders.append(folder)
            
            self.log(f"\n⏭️  Skipping {skipped_count} already completed papers")
            self.log(f"📋 Processing {len(pending_folders)} remaining papers\n")
            
            self.stats['total_papers'] = len(pending_folders)
            
            if not pending_folders:
                self.log("✅ All papers already processed!")
                return
            
            # Process each pending paper
            for paper_folder in pending_folders:
                self.process_paper(paper_folder)
            
            # Print final summary
            self.print_summary()
        
        except Exception as e:
            self.log(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def print_summary(self):
        """Print final statistics."""
        self.log(f"\n{'='*60}")
        self.log(f"📊 FINAL SUMMARY")
        self.log(f"{'='*60}")
        self.log(f"Total papers: {self.stats['total_papers']}")
        self.log(f"  ✅ Successful: {self.stats['successful_papers']}")
        self.log(f"  ⏭️  Skipped (already done): {self.stats['skipped_papers']}")
        self.log(f"  ❌ Failed: {self.stats['failed_papers']}")
        self.log(f"\nTotal images processed: {self.stats['total_images']}")
        self.log(f"  ✅ With caption: {self.stats['images_with_caption']}")
        self.log(f"  ⏭️  Without caption: {self.stats['images_without_caption']}")
        
        if self.stats['images_with_caption'] > 0:
            percentage = (self.stats['images_with_caption'] / self.stats['total_images']) * 100
            self.log(f"  📈 Success rate: {percentage:.1f}%")
        
        self.log(f"\n🎉 Batch processing completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract figure captions from all papers using Claude API"
    )
    parser.add_argument(
        "--api-key",
        help="Claude API key (or set ANTHROPIC_API_KEY env variable)"
    )
    parser.add_argument(
        "--delete-manifest",
        action="store_true",
        help="Delete manifest.csv files after processing"
    )
    args = parser.parse_args()
    
    # Confirm deletion if requested
    if args.delete_manifest:
        response = input(
            "⚠️  You requested to delete all manifest.csv files. Continue? (yes/no): "
        )
        if response.lower() != "yes":
            print("Cancelled.")
            sys.exit(0)
    
    extractor = BatchCaptionExtractor(
        api_key=args.api_key,
        delete_manifest=args.delete_manifest
    )
    extractor.run()


if __name__ == "__main__":
    main()