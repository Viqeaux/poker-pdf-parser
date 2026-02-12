# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import json
import traceback
import os
import math
import io

import requests
import fitz  # PyMuPDF
from PIL import Image, ImageOps

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUILD_ID = "phase1-seat-detection-v1"


# ----------------------------
# Request Models
# ----------------------------

class OpenAIFileRef(BaseModel):
    download_link: str

    class Config:
        extra = "allow"


class ParseRequest(BaseModel):
    openaiFileIdRefs: List[OpenAIFileRef]
    options: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


# ----------------------------
# File Fetch Helper
# ----------------------------

def fetch_pdf_bytes(download_link: str) -> bytes:
    if not download_link:
        raise ValueError("download_link is empty")

    download_link = download_link.strip()

    # Signed URL
    if download_link.startswith("http://") or download_link.startswith("https://"):
        r = requests.get(download_link, timeout=60)
        r.raise_for_status()
        return r.content

    # Local path (rare case)
    if download_link.startswith("/mnt/") and os.path.exists(download_link):
        with open(download_link, "rb") as f:
            return f.read()

    # OpenAI file ID
    if download_link.startswith("file-") or download_link.startswith("file_"):
        from openai import OpenAI
        client = OpenAI()
        response = client.files.content(download_link)
        return response.read()

    raise ValueError(f"Unsupported download_link format: {download_link}")


# ----------------------------
# Geometry Helpers
# ----------------------------

def polar_to_cartesian(cx, cy, radius, angle_deg):
    angle_rad = math.radians(angle_deg)
    x = cx + radius * math.cos(angle_rad)
    y = cy + radius * math.sin(angle_rad)
    return int(x), int(y)


def ocr_digits_only(img):
    if not OCR_AVAILABLE or pytesseract is None:
        return None

    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    text = pytesseract.image_to_string(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=0123456789,"
    )

    cleaned = "".join(c for c in text if c in "0123456789")
    if not cleaned:
        return None

    try:
        return int(cleaned)
    except:
        return None


def extract_seats_and_stacks(image):
    w, h = image.size

    # Table center tuned for 1280x720 PokerStars layout
    cx = w // 2
    cy = int(h * 0.44)
    radius = int(w * 0.35)

    seat_angles = [-90, -50, -10, 30, 70, 110, 150, 190, 230]

    seats = []

    for idx, angle in enumerate(seat_angles):
        sx, sy = polar_to_cartesian(cx, cy, radius, angle)

        box_w = int(w * 0.09)
        box_h = int(h * 0.05)

        stack_box = (
            sx - box_w // 2,
            sy - box_h // 2,
            sx + box_w // 2,
            sy + box_h // 2
        )

        crop = image.crop(stack_box)
        stack_value = ocr_digits_only(crop)

        if stack_value and stack_value > 0:
            seats.append({
                "seat_index": idx + 1,
                "stack": stack_value,
                "anchor": (sx, sy)
            })

    return seats


# ----------------------------
# Frame Rendering
# ----------------------------

def render_page_to_image(pdf_bytes: bytes, page_index: int = 0) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=150)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


# ----------------------------
# API Endpoint
# ----------------------------

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):

    opts = req.options or {}
    debug = bool(opts.get("debug", False))

    out_results = []
    out_errors = []

    for ref in req.openaiFileIdRefs:
        try:
            pdf_bytes = fetch_pdf_bytes(ref.download_link)

            image = render_page_to_image(pdf_bytes, page_index=0)

            seats = extract_seats_and_stacks(image)

            out_results.append({
                "detected_seats": seats,
                "seat_count": len(seats)
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
