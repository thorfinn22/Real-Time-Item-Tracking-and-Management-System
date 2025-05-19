# test_ocr.py
import sys
from pathlib import Path

import easyocr
import cv2

# 1) Decide which image to read:
#    - If you passed a filename on the command line, use that.
#    - Otherwise, default to 'download.jpeg' next to this script.
if len(sys.argv) > 1:
    img_path = Path(sys.argv[1])
else:
    img_path = Path(__file__).parent / "download.jpeg"

# 2) Check it exists
if not img_path.exists():
    raise FileNotFoundError(f"❌ Image not found: {img_path.resolve()}")

# 3) Load via OpenCV
img = cv2.imread(str(img_path))
if img is None:
    raise RuntimeError(f"❌ OpenCV failed to load: {img_path.resolve()}")

# 4) Convert BGR→RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 5) Run EasyOCR
reader = easyocr.Reader(["en"], gpu=False)
results = reader.readtext(img)  # default detail=1

# 6) Print in an organized way
for bbox, text, conf in results:
    print(f"{text!r}    (conf={conf:.2f})")
