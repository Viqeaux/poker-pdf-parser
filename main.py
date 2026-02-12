# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import io
import json
import traceback
import os

import requests
import fitz  # PyMuPDF

# OCR optional
try:
    import pytesseract  # type: ignore
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False

# ----------------------------
# App Setup
# ----------------------------

app = FastAPI()

# 🔥 REQUIRED FOR GPT ACTIONS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUILD_ID = "production-v5"


# ----------------------------
# Request Models
# ----------------------------

class OpenAIFileRef(BaseModel):
    download_link: str

    class Config:
        extra = "allow"  # critical for GPT bridge


class ParseRequest(BaseModel):
    openaiFileIdRefs: List[OpenAIFileRef]
    options: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"  # critical for GPT bridge


# ----------------------------
# File Fetch Helper
# ----------------------------

def fetch_pdf_bytes(download_link: str) -> bytes:
    if not isinstance(download_link, str) or not download_link.strip():
        raise ValueError("download_link must be a non-empty string")

    download_link = download_link.strip()

    # Case 1: Signed URL from GPT Actions (production case)
    if download_link.startswith("http://") or download_link.startswith("https://"):
        r = requests.get(download_link, timeout=60)
        r.raise_for_status()
        return r.content

    # Case 2: Local path (rare tool runtime case)
    if download_link.startswith("/mnt/") and os.path.exists(download_link):
        with open(download_link, "rb") as f:
            return f.read()

    # Case 3: OpenAI file ID
    if download_link.startswith("file-") or download_link.startswith("file_"):
        from openai import OpenAI
        client = OpenAI()
        response = client.files.content(download_link)
        return response.read()

    raise ValueError(f"Unsupported download_link format: {download_link}")


# ----------------------------
# Minimal PDF Validation
# ----------------------------

def count_pages(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


# ----------------------------
# API Endpoint
# ----------------------------

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):

    opts = req.options or {}
    debug = bool(opts.get("debug", False))

    out_results: List[Dict[str, Any]] = []
    out_errors: List[Dict[str, Any]] = []

    for ref in req.openaiFileIdRefs:
        try:
            pdf_bytes = fetch_pdf_bytes(ref.download_link)
            pages_scanned = count_pages(pdf_bytes)

            out_results.append({
                "pages_scanned": pages_scanned
            })

        except Exception as e:
            err = {
                "type": type(e).__name__,
                "message": str(e)
            }

            if debug:
                err["traceback"] = traceback.format_exc()
                err["download_link"] = ref.download_link

            out_errors.append(err)

    payload = {
        "BUILD_ID": BUILD_ID,
        "ocr_available": OCR_AVAILABLE,
        "results": out_results,
        "errors": out_errors,
    }

    return {
        "BUILD_ID": BUILD_ID,
        "ocr_available": OCR_AVAILABLE,
        "errors_count": len(out_errors),
        "hands_detected": 0,
        "hand_history_text": json.dumps(payload, indent=2),
    }
