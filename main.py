# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import traceback
import math
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


# --------------------------------------------------
# App Setup
# --------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUILD_ID = "json-engine-v4.1-mnt-supported"


# --------------------------------------------------
# Health Endpoint (REQUIRED FOR GPT ACTIONS)
# --------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok"}


# --------------------------------------------------
# Request Model (SIMPLIFIED)
# --------------------------------------------------

class ParseRequest(BaseModel):
    file_url: str
    options: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


# --------------------------------------------------
# File Download (PRODUCTION + GPT RUNTIME SAFE)
# --------------------------------------------------

def fetch_pdf_bytes(file_url: str) -> bytes:
    if not isinstance(file_url, str):
        raise ValueError("file_url must be a string")

    file_url = file_url.strip()

    # Case 1: Signed HTTPS URL (production)
    if file_url.startswith("http"):
        r = requests.get(file_url, timeout=60)
        r.raise_for_status()
        return r.content

    # Case 2: GPT runtime local mount (/mnt/data/...)
    if file_url.startswith("/mnt/") and os.path.exists(file_url):
        with open(file_url, "rb") as f:
            return f.read()

    raise ValueError(f"Unsupported file_url format: {file_url}")


# --------------------------------------------------
# OCR Helpers
# --------------------------------------------------

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


# --------------------------------------------------
# Geometry Helpers
# --------------------------------------------------

def polar(cx, cy, r, deg):
    rad = math.radians(deg)
    return int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))


def extract_frame_state(image: Image.Image):

    w, h = image.size
    cx = w // 2
    cy = int(h * 0.44)
    radius = int(w * 0.35)

    seat_angles = [-90, -50, -10, 30, 70, 110, 150, 190, 230]

    seats = {}

    for idx, angle in enumerate(seat_angles):
        sx, sy = polar(cx, cy, radius, angle)

        box_w = int(w * 0.09)
        box_h = int(h * 0.05)

        stack_box = (
            sx - box_w // 2,
            sy - box_h // 2,
            sx + box_w // 2,
            sy + box_h // 2
        )

        try:
            stack = ocr_digits(image.crop(stack_box))
        except:
            stack = None

        if stack and stack > 0:
            seats[idx + 1] = {"stack": stack}

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

    return {
        "seats": seats,
        "pot": pot
    }


# --------------------------------------------------
# Main Endpoint
# --------------------------------------------------

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):

    debug = False
    if req.options:
        debug = bool(req.options.get("debug", False))

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
