#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-first PaddleOCR evaluator for order sheets.

- Scans a folder of images (default: ./data)
- Runs PaddleOCR on each image
- ALWAYS calls Qwen/Qwen3-4B (Hugging Face InferenceClient, provider="nebius")
  to extract header KV pairs from the OCR text
- NO products table; NO total_units; NO total_layers
- Writes predictions to CSV
- Compares predictions to annotations.jsonl (ground truth, with products removed)
- Saves raw OCR entries per image to JSON
- Saves raw LLM JSON per image to JSON

Install:
  pip install paddleocr==2.7.0.3 paddlepaddle==2.6.1
  pip install huggingface_hub

Set token (choose one):
  PowerShell:   $env:HF_TOKEN="hf_xxx"
  CMD:          set HF_TOKEN=hf_xxx
  bash/zsh:     export HF_TOKEN=hf_xxx

Run:
  python batch_eval_llm_only.py \
    --data_dir ./data \
    --annotations ./data/annotations.jsonl \
    --out_csv ./output/predictions.csv \
    --metrics_json ./output/metrics.json \
    --raw_json ./output/raw_ocr.json \
    --raw_llm_json ./output/llm_preds.json \
    --hf_token hf_xxx  # or omit to read HF_TOKEN from env
"""

import os, re, sys, json, csv, argparse, statistics
from pathlib import Path

# -------------------------------
# Config
# -------------------------------
HF_PROVIDER = "nebius"
DEFAULT_HF_MODEL = "Qwen/Qwen3-4B"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Canonical output schema (ONLY these keys)
PALLET_OUT_KEYS = [
    "route", "pallet_number", "delivery_date", "load", "dock",
    "shipment_id", "destination", "asn_number", "salesman",
    "total_cases", "printed_date", "page_number"
]

# -------------------------------
# Imports (PaddleOCR)
# -------------------------------
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None
    print("[FATAL] PaddleOCR import failed. Install paddleocr+paddlepaddle.", file=sys.stderr)

# -------------------------------
# Normalization helpers
# -------------------------------

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u00A0", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[:：\\-–—]+$", "", s)  # drop trailing colons/dashes
    return s

def norm_num(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"[^0-9.\-]", "", s)
    if re.fullmatch(r"\d+", s):
        s = str(int(s))
    return s

def _pad2(n: str) -> str:
    try:
        return f"{int(n):02d}"
    except Exception:
        return n

def norm_date(s: str) -> str:
    """Return MM/DD/YYYY if a date is found, else a normalized text fallback."""
    if not s:
        return ""
    s = s.strip()
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s)
    if m:
        mm = _pad2(m.group(1))
        dd = _pad2(m.group(2))
        yyyy = m.group(3)
        return f"{mm}/{dd}/{yyyy}"
    return norm_text(s)

def norm_datetime(s: str) -> str:
    """Return MM/DD/YYYY HH:MM. Accepts '12/05/202411:30' or '12/5/2024 11:30'."""
    if not s:
        return ""
    s = s.strip()
    # common forms:
    # 1) 12/05/202411:30 (no space)
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\s*(\d{1,2}):(\d{2})\b", s)
    if m:
        mm = _pad2(m.group(1))
        dd = _pad2(m.group(2))
        yyyy = m.group(3)
        hh = _pad2(m.group(4))
        minu = _pad2(m.group(5))
        return f"{mm}/{dd}/{yyyy} {hh}:{minu}"
    # 2) date only
    d = norm_date(s)
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", d):
        return f"{d} 00:00"
    return norm_text(s)

def normalize_dock(s: str) -> str:
    """Canonicalize dock like 'D05' (zero-padded)."""
    if not s:
        return ""
    s = str(s).strip().upper()
    m = re.search(r"(\d{1,2})", s)
    if m:
        return f"D{int(m.group(1)):02d}"
    # if purely 'D' or malformed, just uppercase trimmed
    if s.startswith("D"):
        return "D"
    return s

def normalize_destination(addr: str) -> str:
    """
    Fix common OCR punctuation errors:
    - '. ' between street and city -> ', '
    - '.ST' / '. SD' etc. before state -> ', ST'
    - ensure a single space after commas
    - collapse multiple spaces
    """
    if not addr:
        return ""
    s = str(addr).strip()

    # Turn period before Capitalized token into comma if likely a city boundary.
    s = re.sub(r"\.\s+(?=[A-Z])", ", ", s)

    # Period glued to 2-letter state code -> comma+space
    s = re.sub(r"\.(?= ?[A-Z]{2}\b)", ", ", s)

    # Missing space after comma
    s = re.sub(r",\s*", ", ", s)

    # Collapse spaces
    s = re.sub(r"\s{2,}", " ", s)

    return s.strip()

def _sanitize_field_value(canon_key: str, value: str) -> str:
    """Strip label leftovers like 'ID' / 'Number' from values (keep original case)."""
    if not value:
        return ""
    if canon_key == "shipment_id":
        v = re.sub(r"\bID\b\.?:?$", "", value, flags=re.I).strip()
        v = re.sub(r"^\bShipment\b\.?\s*\bID\b\.?:?", "", v, flags=re.I).strip()
        v = re.sub(r"^\bID\b\.?:?", "", v, flags=re.I).strip()
        return v
    if canon_key == "asn_number":
        v = re.sub(r"\bNUMBER\b\.?:?$", "", value, flags=re.I).strip()
        v = re.sub(r"^\bASN\b\.?\s*\bNUMBER\b\.?:?", "", v, flags=re.I).strip()
        v = re.sub(r"^\bNUMBER\b\.?:?", "", v, flags=re.I).strip()
        return v
    return value

def enforce_schema_and_normalize(d: dict) -> dict:
    """Ensure only our keys exist; fill missing; normalize + sanitize for OUTPUT."""
    out = {}
    for k in PALLET_OUT_KEYS:
        v = "" if d is None else d.get(k, "")
        v = "" if v is None else str(v)

        if k == "delivery_date":
            v = norm_date(v)
        elif k == "printed_date":
            v = norm_datetime(v)
        elif k in {"total_cases", "page_number", "pallet_number"}:
            v = norm_num(v) or v
        elif k == "dock":
            v = normalize_dock(v)
        elif k == "destination":
            v = normalize_destination(v)

        if k in {"shipment_id", "asn_number"}:
            v = _sanitize_field_value(k, v)

        out[k] = v
    return out

# -------------------------------
# OCR utilities
# -------------------------------

def to_rect(box):
    """Supports [x1,y1,x2,y2] or [[x,y]*4]."""
    if not box:
        return (0,0,0,0,0,0,0,0)
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int,float)) for v in box):
        x1,y1,x2,y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    else:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x1,y1,x2,y2 = min(xs), min(ys), max(xs), max(ys)
    cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
    w,h = (x2-x1), (y2-y1)
    return x1,y1,x2,y2,cx,cy,w,h

def _parse_line_to_box_text(line):
    """Fallback parser for classic PaddleOCR tuple/dict line formats."""
    box, text = None, None
    if isinstance(line, dict):
        text = line.get("text") or line.get("transcription") or line.get("label")
        if "points" in line and isinstance(line["points"], (list, tuple)):
            pts = line["points"]
            if len(pts) >= 4 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in pts[:4]):
                box = [[float(pts[i][0]), float(pts[i][1])] for i in range(4)]
        elif "bbox" in line and isinstance(line["bbox"], (list, tuple)) and len(line["bbox"]) >= 4:
            x1,y1,x2,y2 = line["bbox"][:4]
            box = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    if (box is None or text is None) and isinstance(line, (list, tuple)):
        try:
            maybe_box, maybe_pair = line[0], line[1]
            if isinstance(maybe_pair, (list, tuple)) and len(maybe_pair) >= 1 and isinstance(maybe_pair[0], str):
                text = maybe_pair[0]
            if isinstance(maybe_box, (list, tuple)):
                if len(maybe_box) >= 4 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in maybe_box[:4]):
                    box = [[float(maybe_box[i][0]), float(maybe_box[i][1])] for i in range(4)]
                elif len(maybe_box) >= 8 and all(isinstance(v, (int,float)) for v in maybe_box[:8]):
                    mb = list(maybe_box)
                    box = [[mb[0],mb[1]],[mb[2],mb[3]],[mb[4],mb[5]],[mb[6],mb[7]]]
        except Exception:
            pass
        if (box is None or text is None):
            for el in line:
                if text is None:
                    if isinstance(el, str): text = el
                    elif isinstance(el, (list, tuple)) and len(el) >= 1 and isinstance(el[0], str):
                        text = el[0]
                if box is None and isinstance(el, (list, tuple)):
                    if len(el) >= 4 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in el[:4]):
                        box = [[float(el[i][0]), float(el[i][1])] for i in range(4)]
                    elif len(el) >= 8 and all(isinstance(v, (int,float)) for v in el[:8]):
                        el = list(el)
                        box = [[el[0],el[1]],[el[2],el[3]],[el[4],el[5]],[el[6],el[7]]]
    if box is None: box = [[0,0],[0,0],[0,0],[0,0]]
    if text is None: text = ""
    return box, text

def _entries_from_predict_page(page):
    """Convert a PaddleOCR.predict() page to entries list."""
    def _as_list(x):
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:
            pass
        return x

    def _first_nonempty_key(d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                v = _as_list(d[k])
                try:
                    if hasattr(v, "__len__") and len(v) == 0:
                        continue
                except Exception:
                    pass
                return v
        return []

    entries, heights = [], []

    page_dict = None
    if hasattr(page, "to_dict"):
        try:
            page_dict = page.to_dict()
        except Exception:
            page_dict = None
    elif isinstance(page, dict):
        page_dict = page

    if page_dict is not None:
        rec_texts = _first_nonempty_key(page_dict, ["rec_texts", "texts"])
        rec_boxes = _first_nonempty_key(page_dict, ["rec_boxes", "boxes", "rec_polys", "dt_polys", "det_polys"])
        for txt, box in zip(rec_texts, rec_boxes):
            x1,y1,x2,y2,cx,cy,w,h = to_rect(box)
            entries.append({
                "text": str(txt),
                "norm": norm_text(txt),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "w": w, "h": h,
            })
            heights.append(h)
        return entries, heights

    if isinstance(page, (list, tuple)):
        for line in page:
            box, txt = _parse_line_to_box_text(line)
            x1,y1,x2,y2,cx,cy,w,h = to_rect(box)
            entries.append({
                "text": str(txt),
                "norm": norm_text(txt),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "w": w, "h": h,
            })
            heights.append(h)

    return entries, heights

def run_ocr(ocr, img_path):
    """Return (entries, median_h) using PaddleOCR.predict()."""
    raw = ocr.predict(str(img_path))
    entries, heights = [], []
    pages = raw if isinstance(raw, (list, tuple)) else [raw]
    for page in pages:
        e, hs = _entries_from_predict_page(page)
        entries.extend(e); heights.extend(hs)
    median_h = statistics.median(heights) if heights else 20
    return entries, median_h

# -------------------------------
# LLM (HF InferenceClient / Nebius)
# -------------------------------

def parse_json_obj_from_text(txt: str):
    """Extract first JSON object from free-form text (handles <think> blocks)."""
    if not txt:
        return None
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S|re.I)
    try:
        return json.loads(txt)
    except Exception:
        pass
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        frag = txt[start:end+1]
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None

def build_llm_messages(example_gt: dict, ocr_entries: list):
    """
    Few-shot prompt:
    - Gives one GT example to force exact schema
    - Provides flattened OCR text
    - Asks for ONLY JSON with our keys (no products)
    """
    schema = {
        "route": "", "pallet_number": "", "delivery_date": "", "load": "",
        "dock": "", "shipment_id": "", "destination": "", "asn_number": "", "salesman": "",
        "total_cases": "", "printed_date": "", "page_number": ""
    }
    # Compact OCR text (order by y then x)
    ents_sorted = sorted(ocr_entries, key=lambda e: (e["cy"], e["cx"]))
    page_text = "\n".join(e["text"] for e in ents_sorted if e["text"].strip())

    sys_msg = {
        "role": "system",
        "content": (
            "You extract fields from noisy OCR text. "
            "Return ONLY a single JSON object with these EXACT keys:\n"
            + ", ".join(schema.keys()) + ".\n"
            "Rules:\n"
            "- Do not include any other keys.\n"
            "- If unknown, use an empty string.\n"
            "- Normalize dates like m/d/yyyy.\n"
            "- For shipment_id and asn_number, remove any stray 'ID' or 'Number' words.\n"
            "- total_cases must be numeric text only.\n"
            "- NO products; NO totals for units or layers."
        )
    }

    ex_msg = {
        "role": "user",
        "content": (
            "EXAMPLE OUTPUT FORMAT (learn schema only):\n"
            + json.dumps({k: example_gt.get(k,"") for k in schema.keys()}, ensure_ascii=False)
        )
    }

    usr_msg = {
        "role": "user",
        "content": (
            "OCR_TEXT_BEGIN\n" + page_text + "\nOCR_TEXT_END\n"
            "Output ONLY the JSON object with the specified keys."
        )
    }
    return [sys_msg, ex_msg, usr_msg]

def call_llm_qwen(messages, model_name: str, hf_token: str):
    """Call Qwen via huggingface_hub.InferenceClient(provider='nebius')."""
    try:
        from huggingface_hub import InferenceClient
    except Exception as e:
        print(f"[FATAL] huggingface_hub missing: {e}", file=sys.stderr)
        sys.exit(2)

    if not hf_token:
        print("[FATAL] No HF token. Pass --hf_token or set HF_TOKEN env.", file=sys.stderr)
        sys.exit(2)

    client = InferenceClient(provider=HF_PROVIDER, api_key=hf_token)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        msg = completion.choices[0].message
        content = getattr(msg, "content", None) if msg else None
        return parse_json_obj_from_text(content), content
    except Exception as e:
        print(f"[FATAL] LLM call failed: {e}", file=sys.stderr)
        sys.exit(2)

# -------------------------------
# Accuracy (LLM ONLY, header fields only)
# -------------------------------

EVAL_FIELD_ORDER = PALLET_OUT_KEYS

def normalize_for_compare(key, val):
    if key in {"total_cases","page_number","pallet_number"}:
        return norm_num(val)
    if key == "delivery_date":
        return norm_date(val)
    if key == "printed_date":
        return norm_datetime(val)
    if key == "dock":
        return normalize_dock(val)
    if key == "destination":
        # punctuation-insensitive compare for addresses
        s = normalize_destination(val)
        s = s.lower().strip()
        s = re.sub(r"[,\.\s]+", " ", s)  # collapse commas/periods/spaces
        return s
    return norm_text(val)

def compare_header_fields(pred: dict, gt: dict):
    scores = {}
    for k in EVAL_FIELD_ORDER:
        pv = normalize_for_compare(k, pred.get(k,""))
        gv = normalize_for_compare(k, gt.get(k,""))
        scores[k] = 1.0 if pv == gv and gv != "" else 0.0
    return sum(scores.values())/len(scores), scores

# -------------------------------
# I/O helpers
# -------------------------------

def load_annotations_jsonl(path):
    """Load GT; drop products/total_units/total_layers; keep only PALLET_OUT_KEYS."""
    gt = {}
    if not os.path.isfile(path):
        return gt
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            img = rec.get("image")
            suffix = rec.get("suffix", "")
            try:
                payload = json.loads(suffix) if isinstance(suffix, str) else suffix
            except Exception:
                continue
            # Reduce to our header-only schema
            pared = {k: payload.get(k, "") for k in PALLET_OUT_KEYS}
            gt[img] = pared
    return gt

def write_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["image"] + PALLET_OUT_KEYS + ["field_accuracy", "overall_accuracy"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in cols})

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data", help="Folder of images")
    ap.add_argument("--annotations", default="./data/annotations.jsonl", help="Ground-truth annotations (.jsonl)")
    ap.add_argument("--out_csv", default="./output/predictions.csv", help="Output CSV path")
    ap.add_argument("--metrics_json", default="./output/metrics.json", help="Output metrics JSON path")
    ap.add_argument("--raw_json", default="./output/raw_ocr.json", help="Path to write raw OCR entries")
    ap.add_argument("--raw_llm_json", default="./output/llm_preds.json", help="Path to write raw LLM preds (parsed + raw text)")
    ap.add_argument("--hf_token", default=None, help="Hugging Face token (else env HF_TOKEN)")
    ap.add_argument("--llm_model", default=DEFAULT_HF_MODEL, help="HF model id (default: Qwen/Qwen3-4B)")
    ap.add_argument("--lang", default="en", help="PaddleOCR language (default: en)")
    args = ap.parse_args()

    # Load GT
    gt = load_annotations_jsonl(args.annotations)
    if gt:
        print(f"[info] Loaded {len(gt)} annotations from {args.annotations}")
    else:
        print("[warn] No annotations file found; metrics won't be computed.", file=sys.stderr)

    # Make example for prompt (if GT exists, pick first; else all-empty schema)
    if gt:
        first_img = next(iter(gt.keys()))
        example_gt = gt[first_img]
    else:
        example_gt = {k: "" for k in PALLET_OUT_KEYS}

    # OCR required (we always LLM, but LLM consumes OCR text)
    if PaddleOCR is None:
        print("[FATAL] PaddleOCR not available.", file=sys.stderr)
        sys.exit(2)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[FATAL] Data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(2)

    images = [p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    images.sort()
    if not images:
        print(f"[warn] No images found in {data_dir}", file=sys.stderr)

    # One OCR instance
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang
    )

    per_image_rows = []
    field_accs = []
    raw_ocr_dump = []
    raw_llm_dump = []
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    for img_path in images:
        print(f"[proc] {img_path.name}")

        # OCR
        entries, median_h = run_ocr(ocr, img_path)
        raw_ocr_dump.append({"image": img_path.name, "entries": entries})

        # LLM (always)
        msgs = build_llm_messages(example_gt, entries)
        llm_parsed, llm_raw_text = call_llm_qwen(msgs, args.llm_model, hf_token)
        llm_pred = enforce_schema_and_normalize(llm_parsed or {})
        raw_llm_dump.append({"image": img_path.name, "llm_raw": llm_raw_text, "llm_parsed": llm_pred})

        # Build CSV row (LLM-only)
        row = {"image": img_path.name}
        for k in PALLET_OUT_KEYS:
            row[k] = llm_pred.get(k, "")

        # Evaluate (if GT present)
        fa = overall = ""
        if img_path.name in gt:
            g = gt[img_path.name]
            fa_val, _ = compare_header_fields(llm_pred, g)
            fa = round(fa_val * 100.0, 2)
            overall = fa
            field_accs.append(fa_val)

        row["field_accuracy"] = fa
        row["overall_accuracy"] = overall
        per_image_rows.append(row)

    # CSV
    write_csv(per_image_rows, args.out_csv)
    print(f"[done] Wrote CSV -> {args.out_csv}")

    # Metrics
    metrics = {
        "num_images": len(images),
        "num_with_ground_truth": len([r for r in per_image_rows if r.get("overall_accuracy") != ""]),
        "mean_field_accuracy": (sum(field_accs)/len(field_accs) if field_accs else 0.0),
        "mean_overall_accuracy": (sum(field_accs)/len(field_accs) if field_accs else 0.0),
    }
    os.makedirs(os.path.dirname(args.metrics_json), exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] Wrote metrics -> {args.metrics_json}")
    if metrics["num_with_ground_truth"] > 0:
        print(f"[summary] Field Acc / Overall: {metrics['mean_overall_accuracy']*100:.2f}%")

    # Raw OCR JSON
    raw_dir = os.path.dirname(args.raw_json)
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
    with open(args.raw_json, "w", encoding="utf-8") as f:
        json.dump(raw_ocr_dump, f, ensure_ascii=False, indent=2)
    print(f"[done] Wrote raw OCR -> {args.raw_json}")

    # Raw LLM preds JSON
    os.makedirs(os.path.dirname(args.raw_llm_json), exist_ok=True)
    with open(args.raw_llm_json, "w", encoding="utf-8") as f:
        json.dump(raw_llm_dump, f, ensure_ascii=False, indent=2)
    print(f"[done] Wrote raw LLM preds -> {args.raw_llm_json}")


if __name__ == "__main__":
    main()
