# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import traceback
import io
import os

import requests
import fitz
from PIL import Image, ImageOps

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUILD_ID = "json-engine-v6-runtime-compatible"


@app.get("/")
async def root():
    return {"status": "ok"}


class ParseRequest(BaseModel):
    file_url: str
    options: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


def fetch_pdf_bytes(file_url: str) -> bytes:

    if not isinstance(file_url, str):
        raise ValueError("file_url must be a string")

    file_url = file_url.strip()

    # Case 1: GPT runtime local mount
    if file_url.startswith("/mnt/") and os.path.exists(file_url):
        with open(file_url, "rb") as f:
            return f.read()

    # Case 2: Signed HTTPS URL
    if file_url.startswith("http"):
        r = requests.get(file_url, timeout=60)
        r.raise_for_status()
        return r.content

    raise ValueError(f"Unsupported file_url format: {file_url}")


def ocr_digits(img: Image.Image):
    if not OCR_AVAILABLE:
        return None

    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    text = pytesseract.image_to_string(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    )

    cleaned = "".join(c for c in text if c.isdigit())
    if not cleaned:
        return None

    try:
        return int(cleaned)
    except:
        return None


def extract_frame_state(image: Image.Image):

    w, h = image.size

    pot_box = (
        int(w * 0.45),
        int(h * 0.32),
        int(w * 0.55),
        int(h * 0.38)
    )

    try:
        pot = ocr_digits(image.crop(pot_box)) or 0
    except:
        pot = 0

    return {"pot": pot}


@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):

    debug = bool(req.options.get("debug", False)) if req.options else False

    results = []
    errors = []

    try:

        pdf_bytes = fetch_pdf_bytes(req.file_url)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        frames = []

        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

            state = extract_frame_state(img)
            frames.append(state)

        doc.close()

        results.append({
            "pages_scanned": len(frames),
            "frames": frames
        })

    except Exception as e:
        err = {
            "type": type(e).__name__,
            "message": str(e)
        }
        if debug:
            err["traceback"] = traceback.format_exc()
        errors.append(err)

    payload = {
        "BUILD_ID": BUILD_ID,
        "ocr_available": OCR_AVAILABLE,
        "results": results,
        "errors": errors
    }

    return {
        "BUILD_ID": BUILD_ID,
        "ocr_available": OCR_AVAILABLE,
        "errors_count": len(errors),
        "hands_detected": 0,
        "hand_history_text": json.dumps(payload, indent=2)
    }
