
# Real-Time Receipt OCR & Verification Pipeline

A complete end-to-end system that:

1. **Extracts** every text block from receipt images using EasyOCR  
2. **Structures** those blocks into labeled fields (`Order ID`, `Date`, `Total`, etc.) via GPT-4  
3. **Serves** a live `/scan` page that uses your phone’s camera + QuaggaJS to scan a barcode  
4. **Compares** the scanned code against each field in the structured CSV, showing per-field similarity  
5. **Batch-tests** the pipeline over a folder of receipts and reports per-field and overall accuracy

---

## 🔧 Requirements

- Windows/macOS/Linux  
- Python 3.10+  
- Git, ngrok (for HTTPS tunnel)  
- A valid OpenAI API key (GPT-4)

---

## 📦 Installation

1. **Clone** or copy this repo into `C:\Users\arnob\Documents\EasyOCR-master` (or your preferred path).
2. **Create** & **activate** a venv:
   ```powershell
   cd C:\Users\arnob\Documents\EasyOCR-master
   python -m venv venv
   .\venv\Scripts\Activate.ps1      # PowerShell
   # source venv/bin/activate       # macOS/Linux
````

3. **Install** dependencies:

   ```powershell
   pip install easyocr openai==0.28.0 torch torchvision torchaudio \
               fastapi uvicorn quagga-python pyzbar opencv-python-headless \
               numpy python-multipart python-bidi scipy
   ```
4. **Set** your OpenAI key in the session:

   ```powershell
   $Env:OPENAI_API_KEY="sk-…your key here…"
   ```

---

## 🚀 Quickstart

### 1. OCR Extraction

```powershell
python check_easyocr.py ./receipts/sample1.png
```

* **Outputs**:

  * `raw_ocr.csv` — every detected text block (text, confidence, x/y)
  * `annotated.jpg` — receipt image with bounding boxes & labels

### 2. GPT-4 Structuring

```powershell
python universal_receipt_ocr.py ./receipts/sample1.png
```

* **Outputs**:

  * `output.csv` — a single row of named fields and values, e.g.:

    |    filename | Order ID |       Date | Total | … |
    | ----------: | -------: | ---------: | ----: | - |
    | sample1.png |    12345 | 2025-05-19 | 88.50 | … |

### 3. Live Barcode Verification

1. **Start** the FastAPI app:

   ```powershell
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
2. **Expose** via ngrok (in a second terminal):

   ```powershell
   cd C:\Users\arnob\Tools\ngrok
   .\ngrok.exe http 8000
   ```
3. **On your phone**, open the HTTPS URL that ngrok prints (e.g. `https://abcd1234.ngrok-free.app/scan`).
4. **Grant camera access** and point at the barcode on your printed receipt.
5. **See** a neon-green table comparing each field in `output.csv` against the scanned code with a similarity score (1.00 = exact).

### 4. Batch Accuracy Testing

1. **Generate** your ground truth for all receipts:

   ```powershell
   python generate_ground_truth.py
   ```

   * Reads every image in `receipts/`, runs OCR+GPT, writes `ground_truth.csv` (one row per file).
2. **Run** the batch tester:

   ```powershell
   python batch_test.py
   ```

   * Produces `batch_summary.csv` with per-field average similarity and overall exact-match rate.
3. **Inspect** `batch_summary.csv` to see which fields need improvement and your pipeline’s accuracy.

---

## 📁 Project Structure

```
EasyOCR-master/
├─ receipts/                    # your folder of receipt images
├─ venv/                        # Python virtual-env
├─ check_easyocr.py            # EasyOCR → raw_ocr.csv + annotated.jpg
├─ universal_receipt_ocr.py    # GPT-4 structuring → output.csv
├─ app.py                       # FastAPI + QuaggaJS live scan & compare
├─ generate_ground_truth.py     # Build ground_truth.csv over all receipts
├─ batch_test.py                # Compute batch accuracy summary
├─ requirements.txt             # pinned dependencies
└─ README.md                    # this file
```

---

## 🤝 License

This project is MIT-licensed. Feel free to adapt and extend!
