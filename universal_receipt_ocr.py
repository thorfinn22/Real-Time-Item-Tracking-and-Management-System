#!/usr/bin/env python3
import os
import sys
import json
import csv
import cv2
import easyocr
import openai

from pathlib import Path
from json import JSONDecodeError

# ——— CONFIG ———
openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    # v0.28 uses the Model resource directly
    openai.Model.list()
    print("✅ OpenAI API detected and reachable.")
except Exception as e:
    print("❌ OpenAI API not detected or unreachable.")
    print("   Error:", e)
    sys.exit(1)

OUTPUT_CSV = "output.csv"
# —————————————————

def extract_blocks(image_path: Path):
    """Load image and return EasyOCR blocks: dicts of text, conf, x, y."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(rgb, detail=1)
    blocks = []
    for bbox, text, conf in results:
        x = float((bbox[0][0] + bbox[2][0]) / 2)
        y = float((bbox[0][1] + bbox[2][1]) / 2)
        blocks.append({"text": text, "conf": float(conf), "x": x, "y": y})
    return blocks

def call_llm_extract(blocks):
    """
    Ask GPT-4 to return ALL label:value pairs it finds in the receipt,
    as a single flat JSON object with no extra commentary.
    """
    prompt = f"""
You are given a list of OCR text blocks from a receipt.
Each block is a JSON object with keys: text, conf, x, y.

Extract **every** field name (label) and its corresponding value from the receipt.
Return **only** a single JSON object mapping labels to values—no commentary.

OCR blocks:
{json.dumps(blocks, indent=2)}
"""
    # Use the v0.28 ChatCompletion interface
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system",  "content": "Output valid JSON only."},
            {"role": "user",    "content": prompt}
        ],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except JSONDecodeError:
        # Dump raw content for debugging
        print("❌ LLM returned invalid JSON:", content, file=sys.stderr)
        raise

def write_csv(data: dict, image_name: str):
    """
    Write one-row CSV where columns are the keys of `data` (in that order),
    prefixed by 'filename'.
    """
    headers = ["filename"] + list(data.keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        row = {"filename": image_name, **data}
        writer.writerow(row)
    print(f"✅ Wrote {OUTPUT_CSV} with columns: {headers}")

def main(image_file: str):
    path = Path(image_file)
    if not path.exists():
        print(f"❌ File not found: {path}", file=sys.stderr)
        sys.exit(1)

    # 1) OCR
    blocks = extract_blocks(path)

    # 2) Structure via GPT
    structured = call_llm_extract(blocks)

    # 3) CSV output
    write_csv(structured, path.name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python universal_receipt_ocr.py <path/to/receipt.jpg>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
