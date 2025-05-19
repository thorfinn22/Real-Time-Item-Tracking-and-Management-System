#!/usr/bin/env python3
import sys
import csv
import cv2
import easyocr
import numpy as np               # ← Ensure numpy is imported
from pathlib import Path

def extract_blocks(image_path: Path):
    """Load image and return EasyOCR blocks: list of (bbox, text, conf)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(["en"], gpu=False)
    return reader.readtext(rgb, detail=1)

def dump_raw_csv(blocks, out_csv="raw_ocr.csv"):
    """Write raw OCR blocks to CSV for inspection."""
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text","confidence","x","y"])
        writer.writeheader()
        for bbox, text, conf in blocks:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x_center = sum(xs)/4
            y_center = sum(ys)/4
            writer.writerow({
                "text": text,
                "confidence": round(conf,3),
                "x": round(x_center,1),
                "y": round(y_center,1)
            })
    print(f"✅ Raw OCR data written to {out_csv}")

def annotate_image(blocks, image_path: Path, out_image="annotated.jpg"):
    """Draw bounding boxes + text onto the image and save for visual check."""
    img = cv2.imread(str(image_path))
    for bbox, text, _ in blocks:
        # Draw the polygon
        pts = np.array(bbox, np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,0), 2)
        # Compute a clean Python-int position for the text
        x0, y0 = bbox[0]
        org = (int(x0), int(y0) - 5)
        cv2.putText(
            img,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1,
            cv2.LINE_AA
        )
    cv2.imwrite(out_image, img)
    print(f"✅ Annotated image saved as {out_image}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_easyocr.py <path/to/image.jpg>", file=sys.stderr)
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"❌ File not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    # 1) extract raw blocks
    blocks = extract_blocks(img_path)

    # 2) dump them to CSV
    dump_raw_csv(blocks, out_csv="raw_ocr.csv")

    # 3) create annotated image
    annotate_image(blocks, img_path, out_image="annotated.jpg")

if __name__ == "__main__":
    main()
