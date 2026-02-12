# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import traceback
import os
import math
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

BUILD_ID = "json-engine-v2.1"


# --------------------------------------------------
# 🔥 HEALTH ENDPOINT (CRITICAL FOR GPT ACTIONS)
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
# File Fetch
# --------------------------------------------------

def fetch_pdf_bytes(download_link: str) -> bytes:
    download_link = download_link.strip()

    if download_link.startswith("http"):
        r = requests.get(download_link, timeout=60)
        r.raise_for_status()
        return r.content

    if download_link.startswith("/mnt/") and os.path.exists(download_link):
        with open(download_link, "rb") as f:
            return f.read()

    if download_link.startswith("file-") or download_link.startswith("file_"):
        from openai import OpenAI
        client = OpenAI()
        response = client.files.content(download_link)
        return response.read()

    raise ValueError("Unsupported download_link format")


# --------------------------------------------------
# OCR Helpers
# --------------------------------------------------

def ocr_digits(img):
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


def extract_frame_state(image):

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

        stack = ocr_digits(image.crop(stack_box))

        if stack and stack > 0:
            seats[idx + 1] = {
                "stack": stack
            }

    pot_box = (
        int(w * 0.45),
        int(h * 0.32),
        int(w * 0.55),
        int(h * 0.38)
    )

    pot = ocr_digits(image.crop(pot_box)) or 0

    return {
        "seats": seats,
        "pot": pot
    }


# --------------------------------------------------
# Blind + Ante Detection (Tolerant Mode)
# --------------------------------------------------

def detect_blinds_and_antes(prev_state, curr_state, tolerance=3):

    if not prev_state:
        return None

    prev_seats = prev_state.get("seats", {})
    curr_seats = curr_state.get("seats", {})

    deltas = []

    for seat, curr in curr_seats.items():
        prev = prev_seats.get(seat)
        if not prev:
            continue

        delta = prev["stack"] - curr["stack"]
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return None

    deltas.sort()
    unique_deltas = sorted(set(deltas))

    if len(unique_deltas) < 3:
        return None

    ante = unique_deltas[0]
    sb = unique_deltas[1]
    bb = unique_deltas[2]

    player_count = len(curr_seats)
    expected_pot = (ante * player_count) + sb + bb
    actual_pot = curr_state.get("pot", 0)

    result = {
        "ante": ante,
        "small_blind": sb,
        "big_blind": bb,
        "players": player_count
    }

    if abs(expected_pot - actual_pot) > tolerance:
        result["pot_warning"] = {
            "expected": expected_pot,
            "actual": actual_pot
        }

    return result


# --------------------------------------------------
# API Endpoint
# --------------------------------------------------

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):

    opts = req.options or {}
    debug = bool(opts.get("debug", False))

    results = []
    errors = []

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

            hands = []
            prev_state = None

            for state in frames:

                # New hand detection via pot reset
                if prev_state and prev_state["pot"] > 0 and state["pot"] == 0:
                    prev_state = None
                    continue

                # Detect blind frame
                if prev_state and prev_state["pot"] == 0 and state["pot"] > 0:
                    blind_info = detect_blinds_and_antes(prev_state, state)
                    if blind_info:
                        hands.append({
                            "blind_structure": blind_info,
                            "frames": []
                        })

                if hands:
                    hands[-1]["frames"].append(state)

                prev_state = state

            results.append({
                "hands_detected": len(hands),
                "hands": hands
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
        "errors": errors,
    }

    return {
        "BUILD_ID": BUILD_ID,
        "ocr_available": OCR_AVAILABLE,
        "errors_count": len(errors),
        "hands_detected": sum(r["hands_detected"] for r in results),
        "hand_history_text": json.dumps(payload, indent=2),
    }
