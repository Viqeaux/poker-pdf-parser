# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import io
import json
import traceback

import requests
import fitz  # PyMuPDF
from PIL import Image, ImageStat, ImageOps

# OCR optional
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False

app = FastAPI()

BUILD_ID = "ocr-rank-norm-v1"


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
    bh = bottom - top

    pad_x = int(bw * 0.04)
    pad_y = int(bh * 0.12)

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
def street_from_slots(slot_activity: Dict[str, float]) -> str:
    # tuned thresholds you’ve been using
    if slot_activity.get("river", 0.0) >= 50.0:
        return "RIVER"
    if slot_activity.get("turn", 0.0) >= 50.0:
        return "TURN"
    flop_count = sum(1 for k in ("flop1", "flop2", "flop3") if slot_activity.get(k, 0.0) >= 20.0)
    if flop_count == 3:
        return "FLOP"
    return "PREFLOP"


# ----------------------------
# OCR (rank only) + normalization
# ----------------------------
def rank_crop_from_slot(img: Image.Image, slot_box: List[int]) -> Image.Image:
    left, top, right, bottom = slot_box
    w = right - left
    h = bottom - top
    # top-left corner where rank usually sits
    return img.crop((left, top, left + int(w * 0.35), top + int(h * 0.35)))


def ocr_rank_raw(rank_crop: Image.Image) -> str:
    if not OCR_AVAILABLE or pytesseract is None:
        return "OCR_UNAVAILABLE"

    gray = ImageOps.autocontrast(ImageOps.grayscale(rank_crop))
    txt = pytesseract.image_to_string(
        gray,
        config="--psm 10 -c tessedit_char_whitelist=AKQJT9876543210"
    )
    txt = (txt or "").strip().upper()
    return txt if txt else "?"


def normalize_rank(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s or s == "?":
        return None

    # keep only plausible characters
    # common OCR junk: spaces, punctuation, stray letters
    s = "".join(ch for ch in s if ch in "AKQJT9876543210")

    if not s:
        return None

    # handle "10" variants
    if s == "10" or s == "1O" or s == "IO":
        return "T"

    # common confusions
    s = s.replace("0", "O")  # if it’s actually an O
    # but ranks don’t include O, so treat O as zero/ten-ish if paired with 1
    if s in ("O",):
        return None

    # if multiple chars survived, pick the best guess
    # prefer face ranks if present, else first digit
    for ch in s:
        if ch in "AKQJT":
            return ch
    for ch in s:
        if ch in "98765432":
            return ch

    # if it’s "1" alone, likely junk (rank isn’t 1)
    return None


# ----------------------------
# API
# ----------------------------
@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):
    refs = req.openaiFileIdRefs or []
    opts = req.options or {}

    debug = bool(opts.get("debug", False))
    max_pages = int(opts.get("max_pages_to_scan", 75))
    max_gap_pages = int(opts.get("max_gap_pages", 10))

    out_results: List[Dict[str, Any]] = []
    out_errors: List[Dict[str, Any]] = []

    for ref in refs:
        if not isinstance(ref, dict) or "download_link" not in ref:
            out_errors.append({"error": "bad_ref", "detail": "Each ref must be an object with download_link."})
            continue

        url = ref["download_link"]

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            pdf_bytes = r.content

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                pages = min(doc.page_count, max_pages)

                # Build street timeline (changes only)
                street_timeline: List[Dict[str, Any]] = []
                last_street: Optional[str] = None

                for i in range(pages):
                    img = render_page_to_pil(doc, i)
                    board_box = board_region_box(img)
                    slot_boxes = card_slot_boxes_from_board(board_box)

                    slot_activity = {name: activity_for_box(img, box) for name, box in slot_boxes.items()}
                    street = street_from_slots(slot_activity)

                    if street != last_street:
                        street_timeline.append({"page_index": i, "street": street})
                        last_street = street

                # Hand windows: start at FLOP; end when we see a PREFLOP later OR a big gap
                hand_windows: List[Dict[str, Any]] = []
                current = None
                last_change_page = None
                hand_index = 0

                for entry in street_timeline:
                    p = entry["page_index"]
                    s = entry["street"]

                    if s == "FLOP" and current is None:
                        hand_index += 1
                        current = {
                            "hand_index": hand_index,
                            "start_page_index": p,
                            "end_page_index": None,
                            "partial": True,
                        }

                    # gap rule based on *change pages*
                    if current and last_change_page is not None and (p - last_change_page) > max_gap_pages:
                        current["end_page_index"] = last_change_page
                        current["partial"] = True
                        hand_windows.append(current)
                        current = None

                    # close on a PREFLOP after we already started
                    if s == "PREFLOP" and current is not None:
                        current["end_page_index"] = last_change_page
                        current["partial"] = False
                        hand_windows.append(current)
                        current = None

                    last_change_page = p

                if current is not None:
                    current["end_page_index"] = last_change_page
                    current["partial"] = True
                    hand_windows.append(current)

                # OCR ranks: one representative page per hand (prefer end_page if it exists)
                card_rank_ocr: List[Dict[str, Any]] = []
                for h in hand_windows:
                    candidate_pages = [h.get("end_page_index"), h.get("start_page_index")]
                    page_idx = next((p for p in candidate_pages if isinstance(p, int)), None)
                    if page_idx is None:
                        continue

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

                        if allowed:
                            raw = ocr_rank_raw(rank_crop_from_slot(img, box))
                            ranks_raw[name] = raw
                            ranks_norm[name] = normalize_rank(raw)
                        else:
                            ranks_raw[name] = None
                            ranks_norm[name] = None

                    card_rank_ocr.append(
                        {
                            "hand_index": h["hand_index"],
                            "page_index": page_idx,
                            "street": street,
                            "card_ranks_raw": ranks_raw,
                            "card_ranks": ranks_norm,
                        }
                    )

                out_results.append(
                    {
                        "pages_scanned": pages,
                        "street_timeline": street_timeline,
                        "hand_windows": hand_windows,
                        "card_rank_ocr": card_rank_ocr,
                    }
                )

            finally:
                doc.close()

        except Exception as e:
            err = {
                "type": type(e).__name__,
                "message": str(e),
            }
            if debug:
                err["traceback"] = traceback.format_exc()
                err["ref"] = {"download_link": url}
            out_errors.append(err)

    response_obj = {
        "BUILD_ID": BUILD_ID,
        "ocr_available": OCR_AVAILABLE,
        "results": out_results,
        "errors": out_errors,
    }

    return {
        "hand_history_text": json.dumps(response_obj, indent=2),
        "hands_detected": sum(len(r.get("hand_windows", [])) for r in out_results),
        "BUILD_ID": BUILD_ID,
        "ocr_available": OCR_AVAILABLE,
        "errors_count": len(out_errors),
    }
