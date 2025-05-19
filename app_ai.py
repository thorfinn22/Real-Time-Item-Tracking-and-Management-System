# app_ai.py

import os
import json
import csv
import cv2
import easyocr
import numpy as np
import openai

from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from tempfile import NamedTemporaryFile
from json import JSONDecodeError

# ————— CONFIG —————
openai.api_key = os.getenv("OPENAI_API_KEY")
LABELS = [
    "Product Name","Order ID","Truck ID","Date",
    "Current Warehouse ID","Destination Warehouse ID",
    "Loading Time","Shipping Dock ID","Loading Bay","Stow Position",
    "Estimated Departure Time","Estimated Arrival Time",
    "Priority Class","Loading Priority","Order Reference","Shipping Carrier"
]
# —————————————————

app = FastAPI()
ocr = easyocr.Reader(["en"], gpu=False)

def call_llm_to_structure(raw_blocks):
    """
    Ask GPT-4 to map raw OCR blocks → our LABELS schema,
    and raise an HTTPException if it doesn't return valid JSON.
    """
    prompt = f"""
You are given a list of OCR text blocks from a warehouse receipt.
Each block is a JSON object with "text", "conf", "x", and "y".
Extract EXACTLY the following fields as a single JSON object with these keys:
{json.dumps(LABELS)}

OCR blocks:
{json.dumps(raw_blocks, indent=2)}

**IMPORTANT**: Reply ONLY with valid JSON—no extra commentary.
Example:
{{
  "Product Name": "Acme Widgets",
  "Order ID": "987654112349587825G",
  …
}}
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
        data = json.loads(content)
    except JSONDecodeError:
        # Log the raw response for debugging
        print("⚠️ Invalid JSON from LLM:", content)
        # Return a 502 with the raw text in the error detail
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned invalid JSON. See server log for raw response."
        )
    return data

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
      <head><title>OCR-AI API</title></head>
      <body style="font-family:sans-serif; text-align:center; margin-top:50px;">
        <h1>🎉 OCR-AI API detected!</h1>
        <p>POST your receipt image to <code>/ocr-ai/</code> for a structured CSV download.</p>
      </body>
    </html>
    """

@app.post("/ocr-ai/")
async def ocr_ai(file: UploadFile = File(...)):
    # 1) Read & OCR
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    raw = ocr.readtext(img, detail=1)
    blocks = [
        {
            "text": txt,
            "conf": float(conf),
            "x": float((bbox[0][0] + bbox[2][0]) / 2),
            "y": float((bbox[0][1] + bbox[2][1]) / 2)
        }
        for bbox, txt, conf in raw
    ]

    # 2) Structure via GPT
    try:
        structured = call_llm_to_structure(blocks)
    except HTTPException as e:
        # Return JSON error so client sees a clear message
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    # 3) Build CSV in temp file
    tmp = NamedTemporaryFile(mode="w+", newline="", delete=False, suffix=".csv")
    writer = csv.DictWriter(tmp, fieldnames=["filename"] + LABELS)
    writer.writeheader()
    row = {"filename": file.filename, **structured}
    writer.writerow(row)
    tmp.flush()
    tmp.seek(0)

    # 4) Stream CSV back
    payload = tmp.read().encode("utf-8")
    tmp.close()
    os.unlink(tmp.name)

    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={file.filename}.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
