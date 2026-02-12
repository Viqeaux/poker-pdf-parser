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
import re

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

BUILD_ID = "json-engine-v1"


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


def ocr_rank(img):
    if not OCR_AVAILABLE:
        return None

    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    text = pytesseract.image_to_string(
        gray,
        config="--psm 10 -c tessedit_char_whitelist=AKQJT98765432"
    )

    text = text.strip().upper()
    if text in list("AKQJT98765432"):
        return text
    return None


# --------------------------------------------------
# Geometry
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

    # --- Seats + Stacks ---
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
                "stack": stack,
                "bet": 0
            }

    # --- Pot ---
    pot_box = (
        int(w * 0.45),
        int(h * 0.32),
        int(w * 0.55),
        int(h * 0.38)
    )
    pot = ocr_digits(image.crop(pot_box))

    # --- Board (rank only) ---
    board = []
    board_box = (
        int(w * 0.34),
        int(h * 0.26),
        int(w * 0.66),
        int(h * 0.48)
    )

    bw = board_box[2] - board_box[0]
    slot_w = bw // 5

    for i in range(5):
        slot = (
            board_box[0] + i * slot_w,
            board_box[1],
            board_box[0] + (i + 1) * slot_w,
            board_box[3]
        )
        rank_crop = image.crop(slot)
        rank = ocr_rank(rank_crop)
        if rank:
            board.append(rank)

    return {
        "seats": seats,
        "pot": pot or 0,
        "board": board
    }


# --------------------------------------------------
# Frame Differencing
# --------------------------------------------------

def infer_actions(prev_state, curr_state):

    actions = []

    if not prev_state:
        return actions

    prev_seats = prev_state["seats"]
    curr_seats = curr_state["seats"]

    for seat, curr in curr_seats.items():
        prev = prev_seats.get(seat)

        if not prev:
            continue

        stack_delta = prev["stack"] - curr["stack"]

        if stack_delta > 0:
            actions.append({
                "seat": seat,
                "action": "bet_or_call",
                "amount": stack_delta
            })

    return actions


# --------------------------------------------------
# API
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

            # --- Build Hands ---
            hands = []
            current_hand = {
                "actions": [],
                "frames": []
            }

            prev_state = None

            for state in frames:

                actions = infer_actions(prev_state, state)

                if state["pot"] == 0 and prev_state and prev_state["pot"] > 0:
                    hands.append(current_hand)
                    current_hand = {
                        "actions": [],
                        "frames": []
                    }

                current_hand["frames"].append(state)
                current_hand["actions"].extend(actions)

                prev_state = state

            if current_hand["frames"]:
                hands.append(current_hand)

            results.append({
                "hands_detected": len(hands),
                "hands": hands
            })

        except Exception as e:
            err = {"type": type(e).__name__, "message": str(e)}
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
