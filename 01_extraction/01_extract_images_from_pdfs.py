import os
import hashlib
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF
import pandas as pd


def extract_images_from_pdf(pdf_path, out_root):
    """
    Extracts all raster images from a single PDF and saves them into a subfolder.
    Returns a DataFrame manifest for this file.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_root) / pdf_path.stem  # Creates folder named after PDF
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Error opening {pdf_path.name}: {e}")
        return None

    records = []

    for page_index, page in enumerate(doc, start=1):
        for img_idx, img in enumerate(page.get_images(full=True), start=1):
            try:
                xref = img[0]
                info = doc.extract_image(xref)
                img_bytes = info["image"]
                ext = info.get("ext", "bin")
                width = info.get("width")
                height = info.get("height")
                colorspace = info.get("colorspace")
                bpc = info.get("bpc")

                # Create unique filename using hash
                digest = hashlib.sha256(img_bytes).hexdigest()[:16]
                filename = f"{pdf_path.stem}_p{page_index:02d}_{img_idx:02d}_{digest}.{ext}"
                filepath = out_dir / filename
                
                # Save the image
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                # Record metadata
                records.append({
                    "pdf_file": pdf_path.name,
                    "page": page_index,
                    "image_index_on_page": img_idx,
                    "file": str(filepath),
                    "ext": ext,
                    "width_px": width,
                    "height_px": height,
                    "colorspace": colorspace,
                    "bits_per_component": bpc,
                    "sha256_16": digest
                })
            except Exception as e:
                print(f"⚠️  Error extracting image {img_idx} from page {page_index} of {pdf_path.name}: {e}")
                continue

    doc.close()

    # Save manifest for this PDF
    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "manifest.csv", index=False)
        print(f"✅ {pdf_path.name}: Extracted {len(records)} images")
        return df
    else:
        print(f"⚠️  {pdf_path.name}: No images found")
        return None


def batch_extract(input_folder, output_folder, master_manifest="master_manifest.csv"):
    """
    Loops through all PDFs in input_folder and runs extraction for each.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_records = []
    pdf_files = sorted(input_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {input_folder}")
        return

    print(f"🔍 Found {len(pdf_files)} PDFs in {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    print("=" * 60)
    
    for pdf in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        try:
            df = extract_images_from_pdf(pdf, output_folder)
            if df is not None:
                all_records.append(df)
        except Exception as e:
            print(f"❌ Fatal error processing {pdf.name}: {e}")
            continue

    print("=" * 60)
    
    # Combine all manifests into one master CSV
    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        master_path = output_folder / master_manifest
        combined.to_csv(master_path, index=False)
        print(f"✅ Master manifest saved: {master_path}")
        print(f"📊 Total images extracted: {len(combined)}")
        print(f"📁 Total PDFs processed: {len(all_records)}")
    else:
        print("⚠️  No images extracted from any PDFs.")


if __name__ == "__main__":
    # Updated paths
    input_folder = "/Users/xxy/Documents/MML/MEDiC/LLM-POC-Open-Articles"
    output_folder = "/Users/xxy/Documents/MML/MEDiC/extracted_images"
    
    batch_extract(input_folder, output_folder)