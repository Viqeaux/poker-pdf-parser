# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional, Dict, Tuple
import io
import json
import traceback

import requests
import fitz  # PyMuPDF
from PIL import Image, ImageStat, ImageOps

# OCR optional (Render often won't have tesseract binary installed)
try:
    import pytesseract  # type: ignore

    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False

app = FastAPI()

BUILD_ID = "ocr-rank-norm-v2"


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
    """
    Heuristic community-card board region for typical 16:9 table captures.
    Returns [left, top, right, bottom] in pixels.
    """
    w, h = img.size
    return [int(w * 0.34), int(h * 0.26), int(w * 0.66), int(h * 0.48)]


def activity_for_box(img: Image.Image, box: List[int]) -> float:
    """
    Activity score = avg stddev across RGB channels in the crop.
    """
    crop = img.crop(tuple(box))
    stat = ImageStat.Stat(crop)
    stddev = [float(x) for x in stat.stddev]
    return float(sum(stddev) / max(len(stddev), 1))


def card_slot_boxes_from_board(board_box: List[int]) -> Dict[str, List[int]]:
    """
    Split the board region into 5 fixed slots: flop1, flop2, flop3, turn, river.
    """
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
def coerce_thresholds(opts: dict) -> Dict[str, float]:
    """
    slot_thresholds can be:
      - dict with keys flop/turn/river
      - single number (legacy)
      - missing => defaults
    """
    default = {"flop": 20.0, "turn": 50.0, "river": 50.0}
    raw = opts.get("slot_thresholds", default)

    if isinstance(raw, dict):
        return {
            "flop": float(raw.get("flop", default["flop"])),
            "turn": float(raw.get("turn", default["turn"])),
            "river": float(raw.get("river", default["river"])),
        }

    # legacy numeric threshold
    try:
        v = float(raw)
        return {"flop": v, "turn": max(50.0, v), "river": max(50.0, v)}
    except Exception:
        return default


def street_from_slots(slot_activity: Dict[str, float], thresholds: Dict[str, float]) -> str:
    """
    Determine street based on per-slot activity.
    """
    if float(slot_activity.get("river", 0.0)) >= float(thresholds["river"]):
        return "RIVER"
    if float(slot_activity.get("turn", 0.0)) >= float(thresholds["turn"]):
        return "TURN"

    flop_count = sum(
        1
        for k in ("flop1", "flop2", "flop3")
        if float(slot_activity.get(k, 0.0)) >= float(thresholds["flop"])
    )
    if flop_count == 3:
        return "FLOP"

    return "PREFLOP"


# ----------------------------
# OCR (rank only) + normalization
# ----------------------------
def rank_crop_from_slot(img: Image.Image, slot_box: List[int]) -> Image.Image:
    """
    Crop the top-left corner of a card where rank usually lives.
    """
    left, top, right, bottom = slot_box
    w = right - left
    h = bottom - top
    return img.crop((left, top, left + int(w * 0.35), top + int(h * 0.35)))


def ocr_rank_raw(rank_crop: Image.Image) -> str:
    if not OCR_AVAILABLE or pytesseract is None:
        return "OCR_UNAVAILABLE"

    gray = ImageOps.autocontrast(ImageOps.grayscale(rank_crop))
    txt = pytesseract.image_to_string(
        gray,
        config="--psm 10 -c tessedit_char_whitelist=AKQJT9876543210",
    )
    txt = (txt or "").strip().upper()
    return txt if txt else "?"


def normalize_rank(raw: Optional[str]) -> Optional[str]:
    """
    Normalize OCR outputs to a single rank symbol:
      A K Q J T 9..2
    """
    if raw is None:
        return None

    s = str(raw).strip().upper()
    if not s or s == "?":
        return None

    # Keep only plausible characters
    s = "".join(ch for ch in s if ch in "AKQJT9876543210IO")
    if not s:
        return None

    # 10 variants
    if "10" in s or s in ("IO", "1O", "OI", "O1"):
        return "T"

    # Common confusions:
    # treat I as 1, O as 0 (but rank '1' alone is junk)
    s = s.replace("I", "1").replace("O", "0")

    # Prefer faces if present
    for ch in s:
        if ch in "AKQJT":
            return ch

    # Then digits 9..2
    for ch in s:
        if ch in "98765432":
            return ch

    return None


def allowed_slots_for_street(street: str) -> Tuple[str, ...]:
    if street == "RIVER":
        return ("flop1", "flop2", "flop3", "turn", "river")
    if street == "TURN":
        return ("flop1", "flop2", "flop3", "turn")
    if street == "FLOP":
        return ("flop1", "flop2", "flop3")
    return tuple()


# ----------------------------
# API
# ----------------------------
@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):
    refs = req.openaiFileIdRefs or []
    opts = req.options or {}

    debug = bool(opts.get("debug", False))
    max_pages_to_scan = int(opts.get("max_pages_to_scan", 75))
    max_gap_pages = int(opts.get("max_gap_pages", 10))

    want_timeline = bool(opts.get("timeline", True))
    want_hand_windows = bool(opts.get("include_hand_windows", True))
    want_rank_ocr = bool(opts.get("include_rank_ocr", True))  # keep default True; harmless if OCR unavailable

    thresholds = coerce_thresholds(opts)

    out_results: List[Dict[str, Any]] = []
    out_errors: List[Dict[str, Any]] = []

    for ref in refs:
        # Expect dict with download_link (as you designed)
        if not isinstance(ref, dict) or "download_link" not in ref:
            out_errors.append(
                {
                    "type": "bad_ref",
                    "message": "Each ref must be an object with a download_link string.",
                }
            )
            continue

        url = ref.get("download_link")
        if not isinstance(url, str) or not url.strip():
            out_errors.append({"type": "bad_ref", "message": "download_link must be a non-empty string."})
            continue

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            pdf_bytes = r.content

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                pages_scanned = min(int(doc.page_count), int(max_pages_to_scan))

                street_timeline: List[Dict[str, Any]] = []
                last_street: Optional[str] = None

                # Scan pages and compute street changes
                if want_timeline or want_hand_windows or want_rank_ocr:
                    for i in range(pages_scanned):
                        img = render_page_to_pil(doc, i, dpi=150)
                        board_box = board_region_box(img)
                        slot_boxes = card_slot_boxes_from_board(board_box)
                        slot_activity = {name: activity_for_box(img, box) for name, box in slot_boxes.items()}
                        street = street_from_slots(slot_activity, thresholds)

                        if street != last_street:
                            street_timeline.append({"page_index": i, "street": street})
                            last_street = street

                # Build hand windows: start at FLOP; end on later PREFLOP OR big gap between change pages
                hand_windows: List[Dict[str, Any]] = []
                if want_hand_windows and street_timeline:
                    current: Optional[Dict[str, Any]] = None
                    last_change_page: Optional[int] = None
                    hand_index = 0

                    for entry in street_timeline:
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

                        # gap rule based on *street change pages*
                        if current is not None and last_change_page is not None and (p - last_change_page) > max_gap_pages:
                            current["end_page_index"] = last_change_page
                            current["partial"] = True
                            hand_windows.append(current)
                            current = None

                        # close on PREFLOP after hand started
                        if s == "PREFLOP" and current is not None:
                            current["end_page_index"] = last_change_page
                            current["partial"] = False
                            hand_windows.append(current)
                            current = None

                        last_change_page = p

                    # if still open, it is partial
                    if current is not None:
                        current["end_page_index"] = last_change_page
                        current["partial"] = True
                        hand_windows.append(current)

                # OCR ranks: one representative page per hand (prefer end_page if available)
                card_rank_ocr: List[Dict[str, Any]] = []
                if want_rank_ocr and hand_windows:
                    for h in hand_windows:
                        candidate_pages = [h.get("end_page_index"), h.get("start_page_index")]
                        page_idx = next((p for p in candidate_pages if isinstance(p, int)), None)
                        if page_idx is None:
                            continue
                        if page_idx < 0 or page_idx >= pages_scanned:
                            continue

                        img = render_page_to_pil(doc, page_idx, dpi=150)
                        board_box = board_region_box(img)
                        slot_boxes = card_slot_boxes_from_board(board_box)
                        slot_activity = {name: activity_for_box(img, box) for name, box in slot_boxes.items()}
                        street = street_from_slots(slot_activity, thresholds)

                        allowed = set(allowed_slots_for_street(street))

                        ranks_raw: Dict[str, Optional[str]] = {}
                        ranks_norm: Dict[str, Optional[str]] = {}

                        for name, box in slot_boxes.items():
                            if name in allowed:
                                raw = ocr_rank_raw(rank_crop_from_slot(img, box))
                                ranks_raw[name] = raw
                                ranks_norm[name] = normalize_rank(raw)
                            else:
                                ranks_raw[name] = None
                                ranks_norm[name] = None

                        card_rank_ocr.append(
                            {
                                "hand_index": int(h.get("hand_index", 0)),
                                "page_index": page_idx,
                                "street": street,
                                "card_ranks_raw": ranks_raw,
                                "card_ranks": ranks_norm,
                            }
                        )

                out_results.append(
                    {
                        "pages_scanned": pages_scanned,
                        "thresholds": thresholds,
                        "street_timeline": street_timeline if want_timeline else [],
                        "hand_windows": hand_windows if want_hand_windows else [],
                        "card_rank_ocr": card_rank_ocr if want_rank_ocr else [],
                    }
                )

            finally:
                doc.close()

        except Exception as e:
            err: Dict[str, Any] = {"type": type(e).__name__, "message": str(e)}
            if debug:
                err["traceback"] = traceback.format_exc()
                err["download_link"] = url
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
        "hands_detected": sum(len(r.get("hand_windows", [])) for r in out_results),
        "hand_history_text": json.dumps(payload, indent=2),
    }
