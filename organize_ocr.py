#!/usr/bin/env python3
import os
import sys
import json
import csv
import openai

from pathlib import Path
from json import JSONDecodeError

# ——— CONFIG ———
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("❌ Please set OPENAI_API_KEY", file=sys.stderr)
    sys.exit(1)

# Quick API‐reachability check
try:
    openai.models.list()
    print("✅ OpenAI API detected and reachable.")
except Exception as e:
    print("❌ OpenAI API not reachable:", e, file=sys.stderr)
    sys.exit(1)

OUTPUT_CSV = "structured_output.csv"
# —————————————————

def load_raw_blocks(csv_path: Path):
    """Load the raw OCR CSV into a list of dicts."""
    blocks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            blocks.append({
                "text": row["text"],
                "conf": float(row["confidence"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
            })
    return blocks

def structure_with_llm(blocks):
    """
    Ask GPT-4 to extract and organize all relevant fields
    from the raw OCR blocks, returning a flat JSON object.
    """
    prompt = f"""
You are given a list of OCR text blocks from a receipt.  
Each block is a JSON object with keys: text, conf, x, y.

From these blocks, extract every meaningful field (e.g. Date, Total, Item, Price, Tax, Merchant, etc.)
and organize them into a single JSON object mapping field names to their values.
Return **only** valid JSON—no extra commentary.

OCR blocks:
{json.dumps(blocks, indent=2)}
"""
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You output JSON only."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except JSONDecodeError:
        print("❌ Invalid JSON from GPT:", content, file=sys.stderr)
        sys.exit(1)

def write_structured_csv(data: dict, original_csv: str):
    """
    Write out a one-row CSV whose columns are the keys in `data`,
    prefixed by 'source_file'.
    """
    headers = ["source_file"] + list(data.keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        row = {"source_file": original_csv}
        row.update(data)
        writer.writerow(row)
    print(f"✅ Wrote structured CSV to {OUTPUT_CSV} with columns: {headers}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python organize_ocr.py raw_ocr.csv", file=sys.stderr)
        sys.exit(1)

    raw_csv = Path(sys.argv[1])
    if not raw_csv.exists():
        print(f"❌ File not found: {raw_csv}", file=sys.stderr)
        sys.exit(1)

    # 1) Load the raw OCR blocks
    blocks = load_raw_blocks(raw_csv)

    # 2) Ask GPT-4 to organize them
    structured = structure_with_llm(blocks)

    # 3) Write out the structured CSV
    write_structured_csv(structured, raw_csv.name)

if __name__ == "__main__":
    main()
