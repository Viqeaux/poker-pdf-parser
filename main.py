# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import io
import json
import base64

import requests
import fitz  # PyMuPDF
from PIL import Image, ImageStat

app = FastAPI()


class ParseRequest(BaseModel):
    openaiFileIdRefs: List[Any]
    options: Optional[dict] = None


def render_page_to_pil(doc: fitz.Document, page_index: int, dpi: int = 150) -> Image.Image:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img.convert("RGB")


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[len(s) // 2]


def board_region_box(img: Image.Image) -> List[int]:
    """
    Heuristic crop for the community-card board area on a typical 16:9 poker table capture.
    Returns [left, top, right, bottom] in pixels.
    """
    w, h = img.size
    left = int(w * 0.34)
    top = int(h * 0.26)
    right = int(w * 0.66)
    bottom = int(h * 0.48)
    return [left, top, right, bottom]


def board_region_activity(img: Image.Image) -> Dict[str, Any]:
    box = board_region_box(img)
    crop = img.crop(tuple(box))
    stat = ImageStat.Stat(crop)
    stddev = [float(x) for x in stat.stddev]
    activity = sum(stddev) / max(len(stddev), 1)
    return {"crop_box_px": box, "activity": float(activity)}


def crop_to_small_base64_jpeg(img: Image.Image, box: List[int], max_width: int = 260, quality: int = 35) -> str:
    """
    If crops are enabled, keep them tiny:
    - crop
    - downscale to max_width
    - JPEG compress
    - base64 encode
    """
    crop = img.crop(tuple(box))

    w, h = crop.size
    if w > max_width:
        new_h = int(h * (max_width / float(w)))
        crop = crop.resize((max_width, max(1, new_h)))

    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def activity_for_box(img: Image.Image, box: List[int]) -> float:
    """
    Compute an "activity" score (avg stddev across RGB) for a given pixel box.
    """
    crop = img.crop(tuple(box))
    stat = ImageStat.Stat(crop)
    stddev = [float(x) for x in stat.stddev]
    return float(sum(stddev) / max(len(stddev), 1))


def card_slot_boxes_from_board(board_box: List[int]) -> Dict[str, List[int]]:
    """
    Split the board region into 5 fixed card slots:
    flop1, flop2, flop3, turn, river.

    This assumes the board region tightly covers the community-card band.
    The exact fractions may need tuning after we see stats.
    """
    left, top, right, bottom = board_box
    bw = right - left
    bh = bottom - top

    # Inner padding so we don't include table felt edges
    pad_x = int(bw * 0.04)
    pad_y = int(bh * 0.12)

    x0 = left + pad_x
    y0 = top + pad_y
    x1 = right - pad_x
    y1 = bottom - pad_y

    inner_w = x1 - x0
    inner_h = y1 - y0

    # 5 slots across. Assume small gaps between cards.
    gap = int(inner_w * 0.02)
    slot_w = int((inner_w - gap * 4) / 5)
    slot_h = int(inner_h)

    boxes: Dict[str, List[int]] = {}
    names = ["flop1", "flop2", "flop3", "turn", "river"]
    for idx, name in enumerate(names):
        sx0 = x0 + idx * (slot_w + gap)
        sx1 = sx0 + slot_w
        boxes[name] = [sx0, y0, sx1, y0 + slot_h]

    return boxes


@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):
    refs = req.openaiFileIdRefs or []
    opts = req.options or {}

    # Defaults chosen to avoid ResponseTooLargeError
    debug: bool = bool(opts.get("debug", True))
    include_crops: bool = bool(opts.get("include_crops", False))  # OFF by default
    max_crops: int = int(opts.get("max_crops", 2))
    crop_max_width: int = int(opts.get("crop_max_width", 260))
    crop_quality: int = int(opts.get("crop_quality", 35))

    # Card-slot stats (no crops): how many pages to sample once first_active_page is found
    # 2 means [first_active_page, first_active_page+1]
    slot_pages: int = int(opts.get("slot_pages", 2))

    results: List[Dict[str, Any]] = []

    for ref in refs:
        if not (isinstance(ref, dict) and "download_link" in ref):
            continue

        url = ref["download_link"]

        r = requests.get(url, timeout=60)
        r.raise_for_status()
        pdf_bytes = r.content

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            total_pages = doc.page_count
            max_pages_to_scan = min(75, total_pages)

            # Baseline-jump detector settings (tunable via options if you want)
            BASELINE_PAGES = int(opts.get("baseline_pages", 8))
            MARGIN = float(opts.get("margin", 20.0))
            RATIO = float(opts.get("ratio", 2.0))

            activities: List[float] = []
            per_page: List[Dict[str, Any]] = []
            crop_box_px: Optional[List[int]] = None
            render_size_px: Optional[List[int]] = None

            # Scan pages
            for i in range(max_pages_to_scan):
                img = render_page_to_pil(doc, i, dpi=150)
                if render_size_px is None:
                    render_size_px = [img.size[0], img.size[1]]

                info = board_region_activity(img)
                if crop_box_px is None:
                    crop_box_px = info["crop_box_px"]

                act = float(info["activity"])
                activities.append(act)

                if debug:
                    per_page.append({"page_index": i, "activity": act})

            # Baseline = median of first BASELINE_PAGES
            baseline_slice = activities[: min(BASELINE_PAGES, len(activities))]
            baseline = median(baseline_slice)

            # First page that “jumps”
            first_active_page: Optional[int] = None
            for i, act in enumerate(activities):
                if act >= baseline + MARGIN or (baseline > 0 and act >= baseline * RATIO):
                    first_active_page = i
                    break

            # Top pages by activity (small, useful)
            indexed = list(enumerate(activities))
            indexed.sort(key=lambda t: t[1], reverse=True)
            top_active_pages = [{"page_index": idx, "activity": float(val)} for idx, val in indexed[:10]]

            # Optional tiny crops (OFF by default)
            sample_board_crops: List[Dict[str, Any]] = []
            if include_crops:
                sample_pages: List[int] = []
                if first_active_page is not None:
                    sample_pages.append(first_active_page)
                    if first_active_page + 1 < total_pages:
                        sample_pages.append(first_active_page + 1)
                if not sample_pages:
                    sample_pages = [0]

                seen = set()
                sample_pages_clean: List[int] = []
                for p in sample_pages:
                    if 0 <= p < total_pages and p not in seen:
                        sample_pages_clean.append(p)
                        seen.add(p)
                    if len(sample_pages_clean) >= max_crops:
                        break

                for p in sample_pages_clean:
                    try:
                        img = render_page_to_pil(doc, p, dpi=150)
                        box = board_region_box(img)
                        tiny_b64 = crop_to_small_base64_jpeg(
                            img, box, max_width=crop_max_width, quality=crop_quality
                        )
                        sample_board_crops.append(
                            {
                                "page_index": p,
                                "crop_box_px": box,
                                "board_crop_base64_jpeg": tiny_b64,
                                "note": "Tiny JPEG crop (downscaled/compressed) to avoid response limits.",
                            }
                        )
                    except Exception as e:
                        sample_board_crops.append({"page_index": p, "error": f"{type(e).__name__}: {str(e)}"})

            # NEW: Card-slot activity stats (no crops)
            card_slot_stats: List[Dict[str, Any]] = []
            if first_active_page is not None and crop_box_px is not None and slot_pages > 0:
                pages_to_sample: List[int] = []
                for k in range(slot_pages):
                    p = first_active_page + k
                    if 0 <= p < total_pages:
                        pages_to_sample.append(p)

                for p in pages_to_sample:
                    try:
                        img = render_page_to_pil(doc, p, dpi=150)
                        board_box = board_region_box(img)  # recompute from img to be safe
                        slot_boxes = card_slot_boxes_from_board(board_box)

                        slots_out: Dict[str, Any] = {}
                        for name, box in slot_boxes.items():
                            slots_out[name] = {
                                "slot_box_px": box,
                                "activity": activity_for_box(img, box),
                            }

                        card_slot_stats.append(
                            {
                                "page_index": p,
                                "board_box_px": board_box,
                                "slots": slots_out,
                            }
                        )
                    except Exception as e:
                        card_slot_stats.append(
                            {
                                "page_index": p,
                                "error": f"{type(e).__name__}: {str(e)}",
                            }
                        )

            result: Dict[str, Any] = {
                "bytes": len(pdf_bytes),
                "pages": total_pages,
                "scanned_pages": max_pages_to_scan,
                "render_dpi": 150,
                "render_size_px_example": render_size_px,
                "board_crop_box_px": crop_box_px,
                "baseline_pages": BASELINE_PAGES,
                "baseline_activity_median_first_n": baseline,
                "baseline_margin": MARGIN,
                "baseline_ratio": RATIO,
                "first_board_region_active_page": first_active_page,
                "top_active_pages": top_active_pages,
                "slot_pages": slot_pages,
                "card_slot_stats": card_slot_stats,
            }

            if debug:
                result["per_page_activity_first_20"] = per_page[:20]

            if include_crops:
                result["sample_board_crops"] = sample_board_crops

            results.append(result)

        finally:
            doc.close()

    debug_payload = {
        "mode": "DEBUG" if debug else "HAND_HISTORY_ONLY",
        "files_received": len(refs),
        "results": results,
        "notes": [
            "Crops/base64 are OFF by default to prevent ResponseTooLargeError.",
            "Card-slot stats are computed from a fixed 5-slot split inside the heuristic board region.",
            "This still does not read ranks/suits yet; it provides activity-only signals per card slot.",
        ],
    }

    return {
        "hand_history_text": json.dumps(debug_payload, indent=2),
        "warnings": [
            "Parser is in DEBUG mode (lightweight) unless options.debug=false.",
            "No OCR yet; board + card-slot activity detection only.",
        ],
        "hands_detected": 0,
    }
