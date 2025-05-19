#!/usr/bin/env python3
import sys, csv, cv2
from pathlib import Path
from pyzbar.pyzbar import decode
from difflib import get_close_matches

def extract_barcodes(image_path: Path):
    """Return a list of decoded barcode strings from the image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    barcodes = decode(img)
    return [b.data.decode('utf-8') for b in barcodes]

def load_ocr_texts(csv_path: Path):
    """Load all the 'text' entries from raw_ocr.csv."""
    texts = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
    return texts

def compare(barcodes, texts, threshold=0.8):
    """
    For each scanned barcode, check:
     - Exact match in OCR texts
     - If not, a fuzzy match above threshold
    """
    for code in barcodes:
        print(f"\n🔎 Scanned barcode: {code}")
        if code in texts:
            print("✅ Exact match found in OCR output.")
        else:
            # try fuzzy matching
            close = get_close_matches(code, texts, n=1, cutoff=threshold)
            if close:
                print(f"⚠️ No exact match, but close match: {close[0]}")
            else:
                print("❌ No match found in OCR output.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_barcode.py <image.jpg> <raw_ocr.csv>")
        sys.exit(1)
    img_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])
    if not img_path.exists() or not csv_path.exists():
        print("❌ One of the files does not exist.", file=sys.stderr)
        sys.exit(1)

    barcodes = extract_barcodes(img_path)
    if not barcodes:
        print("⚠️ No barcodes detected in the image.")
        sys.exit(0)

    texts = load_ocr_texts(csv_path)
    compare(barcodes, texts)

if __name__ == "__main__":
    main()
