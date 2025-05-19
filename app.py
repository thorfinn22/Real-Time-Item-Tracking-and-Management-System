#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import csv
from pathlib import Path
from difflib import SequenceMatcher

app = FastAPI()

# ——— Load your structured CSV once at startup ———
CSV_PATH = Path("structured_output.csv")
if not CSV_PATH.exists():
    raise FileNotFoundError("structured_output.csv not found — run your OCR+AI pipeline first")

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    try:
        structured_row = next(reader)
    except StopIteration:
        raise ValueError("structured_output.csv must contain at least one data row")

@app.get("/scan", response_class=HTMLResponse)
async def scan_page():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Barcode vs OCR Comparison</title>
  <script src="https://unpkg.com/quagga/dist/quagga.min.js"></script>
  <style>
    body,html { margin:0; padding:0; background:#000; color:#0f0; font-family:sans-serif; }
    #container { position:relative; width:100vw; height:70vh; }
    #status { position:absolute; top:1em; left:1em; background:rgba(0,0,0,0.6); padding:0.5em; }
    table { width:100%; border-collapse:collapse; margin-top:1em; }
    th,td { border:1px solid #0f0; padding:0.5em; text-align:left; }
    th { background:rgba(0,255,0,0.1); }
  </style>
</head>
<body>
  <div id="container">
    <div id="status">Initializing camera…</div>
  </div>
  <script>
    const status = document.getElementById("status");
    window.addEventListener("load", () => {
      Quagga.init({
        inputStream: {
          name: "Live",
          type: "LiveStream",
          target: document.getElementById('container'),
          constraints: { facingMode: "environment" }
        },
        decoder: {
          readers: ["code_128_reader","ean_reader","upc_reader","upc_e_reader"]
        },
        locate: true
      }, err => {
        if (err) {
          status.textContent = "Camera error: " + err;
          return;
        }
        Quagga.start();
        status.textContent = "Point at the barcode…";
      });
      Quagga.onDetected(data => {
        const code = data.codeResult.code;
        status.textContent = "Scanned: " + code;
        Quagga.stop();
        fetch(`/compare?code=${encodeURIComponent(code)}`)
          .then(res => res.json())
          .then(json => renderComparison(json))
          .catch(err => { status.textContent = "Error: " + err; });
      });
    });

    function renderComparison({ scanned, comparison }) {
      // remove scanner
      document.getElementById('container').remove();
      status.textContent = `Scanned: ${scanned}`;
      // build table
      const tbl = document.createElement('table');
      const header = tbl.insertRow();
      ["Field","CSV Value","Scanned Code","Similarity"].forEach(h => {
        const th = document.createElement('th');
        th.innerText = h;
        header.appendChild(th);
      });
      comparison.forEach(row => {
        const tr = tbl.insertRow();
        [row.column, row.csv_value, scanned, row.similarity].forEach(text => {
          const td = tr.insertCell();
          td.innerText = text;
        });
      });
      document.body.appendChild(tbl);
    }
  </script>
</body>
</html>
"""

@app.get("/compare")
async def compare(code: str = Query(..., description="Scanned barcode value")):
    """
    Returns JSON with scanned code and a comparison array:
      [
        { column, csv_value, similarity },
        …
      ]
    where similarity is a 0.0–1.0 ratio.
    """
    results = []
    for col, val in structured_row.items():
        val_str = val or ""
        sim = SequenceMatcher(None, val_str, code).ratio() if val_str else 0.0
        results.append({
            "column": col,
            "csv_value": val_str,
            "similarity": round(sim, 2)
        })
    return JSONResponse({"scanned": code, "comparison": results})
