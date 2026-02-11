# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import io
import json
import re

import requests
import fitz  # PyMuPDF
from PIL import Image, ImageStat, ImageOps

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

app = FastAPI()


class ParseRequest(BaseModel):
    openaiFileIdRefs: List[Any]
    options: Optional[dict] = None


# ----------------------------
# Rendering
# ----------------------------

def render_page_to_pil(doc: fitz.Document, page_index: int, dpi: int = 150) -> Image.Image:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


# ----------------------------
# Geometry + activity
# ----------------------------

def board_region_box(img: Image.Image) -> List[int]:
    w, h = img.size
    return [
        int(w * 0.34),
        int(h * 0.26),
        int(w * 0.66),
        int(h * 0.48),
    ]


def activity_for_box(img: Image.Image, box: List[int]) -> float:
    crop = img.crop(tuple(box))
    stat = ImageStat.Stat(crop)
    stddev = [float(x) for x in stat.stddev]
    return float(sum(stddev) / max(len(stddev), 1))


def card_slot_boxes_from_board(board_box: List[int]) -> Dict[str, List[int]]:
    left, top, right, bottom = board_box
    bw = right - left

    pad_x = int(bw * 0.04)
    pad_y = int((bottom - top) * 0.12)

    x0 = left + pad_x
    y0 = top + pad_y
    x1 = right - pad_x
    y1 = bottom - pad_y

    inner_w = x1 - x0
    gap = int(inner_w * 0.02)
    slot_w = int((inner_w - gap * 4) / 5)

    boxes: Dict[str, List[int]] = {}
    for i, name in enumerate(["flop1", "flop2", "flop3", "turn", "river"]):
        sx0 = x0 + i * (slot_w + gap)
        boxes[name] = [sx0, y0, sx0 + slot_w, y1]

    return boxes


# ----------------------------
# Street logic
# ----------------------------

def street_from_slots(slots: Dict[str, float]) -> str:
    if slots["river"] >= 50:
        return "RIVER"
    if slots["turn"] >= 50:
        return "TURN"
    if sum(1 for k in ("flop1", "flop2", "flop3") if slots[k] >= 20) == 3:
        return "FLOP"
    return "PREFLOP"


# ----------------------------
# OCR (rank only) + normalization
# ----------------------------

RANK_WHITELIST = "AKQJT98765432"


def rank_crop_from_slot(img: Image.Image, slot_box: List[int]) -> Image.Image:
    left, top, right, bottom = slot_box
    w = right - left
    h = bottom - top
    return img.crop((left, top, left + int(w * 0.35), top + int(h * 0.35)))


def ocr_rank_raw(img: Image.Image) -> str:
    if not OCR_AVAILABLE:
        return "OCR_UNAVAILABLE"

    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    text = pytesseract.image_to_string(
        gray,
        config="--psm 10 -c tessedit_char_whitelist=AKQJT9876543210"
    )
    return (text or "").strip().upper()


def normalize_rank(raw: str) -> str:
    """
    Conservative rank normalization:
    - Accepts raw OCR output and returns one of: A K Q J T 9..2 or "?"
    - Converts "10" (or variants containing 10) into "T"
    - Strips everything else
    """
    if not raw:
        return "?"

    s = raw.strip().upper()

    # Handle common multi-char ten cases first
    # Examples: "10", "1O", "IO", "T0" (rare), extra whitespace/newlines
    s_compact = re.sub(r"\s+", "", s)
    s_compact = s_compact.replace("O", "0")  # treat O as 0 only for the ten check

    if "10" in s_compact:
        return "T"

    # Keep only valid rank characters
    kept = "".join(ch for ch in s_compact if ch in RANK_WHITELIST)

    if not kept:
        return "?"

    # If OCR returned multiple valid chars, take the first
    return kept[0]


# ----------------------------
# API
# ----------------------------

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):
    refs = req.openaiFileIdRefs or []
    opts = req.options or {}

    max_pages = int(opts.get("max_pages_to_scan", 75))
    max_gap_pages = int(opts.get("max_gap_pages", 10))

    # You can turn OCR off from the GPT call if you want
    ocr_enabled = bool(opts.get("ocr_enabled", True))

    results: List[Dict[str, Any]] = []

    for ref in refs:
        if not isinstance(ref, dict) or "download_link" not in ref:
            continue

        pdf_bytes = requests.get(ref["download_link"], timeout=60).content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        try:
            pages = min(doc.page_count, max_pages)
            timeline: List[Dict[str, Any]] = []

            last_street: Optional[str] = None
            for i in range(pages):
                img = render_page_to_pil(doc, i)
                board_box = board_region_box(img)
                slot_boxes = card_slot_boxes_from_board(board_box)

                slot_activity = {name: activity_for_box(img, box) for name, box in slot_boxes.items()}
                street = street_from_slots(slot_activity)

                if street != last_street:
                    timeline.append({"page_index": i, "street": street})
                    last_street = street

            # Hand windows (partial-aware)
            hands: List[Dict[str, Any]] = []
            current: Optional[Dict[str, Any]] = None
            last_page: Optional[int] = None
            hand_index = 0

            for entry in timeline:
                p = int(entry["page_index"])
                s = str(entry["street"])

                if s == "FLOP" and current is None:
                    hand_index += 1
                    current = {
                        "hand_index": hand_index,
                        "start_page_index": p,
                        "end_page_index": None,
                        "partial": True,
                    }

                if current and last_page is not None and (p - last_page) > max_gap_pages:
                    current["end_page_index"] = last_page
                    current["partial"] = True
                    hands.append(current)
                    current = None

                last_page = p

            if current:
                current["end_page_index"] = last_page
                current["partial"] = True
                hands.append(current)

            # OCR one representative page per hand
            ocr_results: List[Dict[str, Any]] = []
            if ocr_enabled and hands:
                for h in hands:
                    # Prefer end page if present (often later street), else start page
                    candidate_pages = [h.get("end_page_index"), h.get("start_page_index")]
                    page_idx = next((p for p in candidate_pages if isinstance(p, int)), h.get("start_page_index", 0))

                    img = render_page_to_pil(doc, page_idx)
                    board_box = board_region_box(img)
                    slot_boxes = card_slot_boxes_from_board(board_box)

                    slot_activity = {name: activity_for_box(img, box) for name, box in slot_boxes.items()}
                    street = street_from_slots(slot_activity)

                    ranks_raw: Dict[str, Optional[str]] = {}
                    ranks_norm: Dict[str, Optional[str]] = {}

                    for name, box in slot_boxes.items():
                        allowed = (
                            (name.startswith("flop") and street in ("FLOP", "TURN", "RIVER")) or
                            (name == "turn" and street in ("TURN", "RIVER")) or
                            (name == "river" and street == "RIVER")
                        )

                        if not allowed:
                            ranks_raw[name] = None
                            ranks_norm[name] = None
                            continue

                        raw = ocr_rank_raw(rank_crop_from_slot(img, box))
                        ranks_raw[name] = raw
                        ranks_norm[name] = normalize_rank(raw)

                    ocr_results.append(
                        {
                            "hand_index": h["hand_index"],
                            "page_index": page_idx,
                            "street": street,
                            "card_ranks_raw": ranks_raw,
                            "card_ranks": ranks_norm,
                        }
                    )

            results.append(
                {
                    "street_timeline": timeline,
                    "hand_windows": hands,
                    "card_rank_ocr": ocr_results,
                }
            )

        finally:
            doc.close()

    return {
        "BUILD_ID": "ocr-rank-norm-v1",
        "hand_history_text": json.dumps({"results": results}, indent=2),
        "hands_detected": sum(len(r.get("hand_windows", [])) for r in results),
        "ocr_available": OCR_AVAILABLE,
    }
