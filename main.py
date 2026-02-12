# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import traceback
import io

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

BUILD_ID = "json-engine-v5-stable-https-only"


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok"}


# --------------------------------------------------
# Request Models
# --------------------------------------------------

class OpenAIFileRef(BaseModel):
    download_link: str

    class Config:
        extra = "allow"


class ParseRequest(BaseModel):
    openaiFileIdRefs: List[OpenAIFileRef]
    options: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


# --------------------------------------------------
# File Download (HTTPS ONLY)
# --------------------------------------------------

def fetch_pdf_bytes(download_link: str) -> bytes:
    if not isinstance(download_link, str):
        raise ValueError("download_link must be a string")

    download_link = download_link.strip()

    if not download_link.startswith("http"):
        raise ValueError(f"Invalid download_link (expected signed HTTPS URL): {download_link}")

    r = requests.get(download_link, timeout=60)
    r.raise_for_status()
    return r.content


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
# Frame Extraction (Minimal Engine)
# --------------------------------------------------

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

    return {
        "pot": pot
    }


# --------------------------------------------------
# Main Endpoint
# --------------------------------------------------

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):

    debug = bool(req.options.get("debug", False)) if req.options else False

    results = []
    errors = []

    try:

        for ref in req.openaiFileIdRefs:

            try:
                pdf_bytes = fetch_pdf_bytes(ref.download_link)
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

            except Exception as inner_e:
                err = {
                    "type": type(inner_e).__name__,
                    "message": str(inner_e)
                }
                if debug:
                    err["traceback"] = traceback.format_exc()
                errors.append(err)

    except Exception as outer_e:
        errors.append({
            "type": type(outer_e).__name__,
            "message": str(outer_e),
            "traceback": traceback.format_exc()
        })

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
