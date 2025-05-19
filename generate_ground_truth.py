#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
from universal_receipt_ocr import extract_blocks, call_llm_extract

# ——— CONFIG ———
RECEIPT_DIR    = Path("large-receipt-image-dataset-SRD")        # folder with your 200 receipt images
GROUND_TRUTH   = Path("ground_truth.csv")
# —————————————————

if not RECEIPT_DIR.is_dir():
    print(f"❌ Folder not found: {RECEIPT_DIR}")
    sys.exit(1)

all_data = {}
all_fields = set()

# 1) Process each receipt
for img_path in sorted(RECEIPT_DIR.iterdir()):
    if img_path.suffix.lower() not in {".png",".jpg",".jpeg"}:
        continue
    print(f"⏳ Processing {img_path.name}…")
    blocks = extract_blocks(img_path)
    structured = call_llm_extract(blocks)
    all_data[img_path.name] = structured
    all_fields.update(structured.keys())

if not all_data:
    print("❌ No receipts processed.")
    sys.exit(1)

# Sort columns for consistency
fields = sorted(all_fields)

# 2) Write ground_truth.csv
with GROUND_TRUTH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Header row: filename + all fields
    writer.writerow(["filename"] + fields)
    # One row per image
    for fname, data in all_data.items():
        row = [fname] + [data.get(field, "") for field in fields]
        writer.writerow(row)

print(f"✅ ground-truth written to {GROUND_TRUTH}")
