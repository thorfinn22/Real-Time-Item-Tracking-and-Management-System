#!/usr/bin/env python3
import os
import sys
import csv
import json
from pathlib import Path
from difflib import SequenceMatcher
from universal_receipt_ocr import extract_blocks, call_llm_extract  # your existing functions

# CONFIG ———
IMG_DIR     = Path("large-receipt-image-dataset-SRD")            # folder containing your 200 .png/.jpg files
GROUND_TRUTH = Path("ground_truth.csv")   # CSV: filename + all structured columns
OUTPUT_SUMMARY = Path("batch_summary.csv")
# —————————

# 1) Load ground-truth into a dict: filename → { field: value, … }
gt = {}
with GROUND_TRUTH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fn = row["filename"]
        gt[fn] = {k:v for k,v in row.items() if k != "filename"}

# 2) Prepare accumulators
fields = list(next(iter(gt.values())).keys())
field_totals = {fld: 0.0 for fld in fields}
field_counts = {fld: 0 for fld in fields}
exact_matches = 0
total_files = 0

# 3) Loop through each image in IMG_DIR
for img_path in sorted(IMG_DIR.iterdir()):
    if img_path.name not in gt:
        print(f"⚠️  Skipping {img_path.name}: no ground truth row.")
        continue

    total_files += 1
    # OCR + GPT structuring
    blocks = extract_blocks(img_path)
    try:
        pred = call_llm_extract(blocks)
    except Exception as e:
        print(f"❌ Error on {img_path.name}: {e}")
        continue

    truth = gt[img_path.name]
    file_exact = True

    # Compare each field
    for fld in fields:
        tval = truth.get(fld,"") or ""
        pval = pred.get(fld,"") or ""
        sim  = SequenceMatcher(None, tval, pval).ratio()
        field_totals[fld] += sim
        field_counts[fld] += 1
        if sim < 1.0:
            file_exact = False

    if file_exact:
        exact_matches += 1

# 4) Compute averages
with OUTPUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["field","avg_similarity"])
    for fld in fields:
        avg = field_totals[fld] / field_counts[fld] if field_counts[fld] else 0
        writer.writerow([fld, f"{avg:.3f}"])
    # overall exact-rate
    writer.writerow([])
    writer.writerow(["total_files", total_files])
    writer.writerow(["exact_match_files", exact_matches])
    writer.writerow(["exact_match_rate", f"{exact_matches/total_files:.3f}"])

print(f"✅ Done! Summary written to {OUTPUT_SUMMARY}")
print(f"Exact-match rate: {exact_matches}/{total_files} = {exact_matches/total_files:.2%}")
