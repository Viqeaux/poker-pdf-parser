# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional, Dict, Union, Tuple
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


def street_state_from_slots(
    slots: Dict[str, Dict[str, Any]],
    thresholds: Union[float, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Convert slot activity into a simple board/street state.

    thresholds can be:
      - float (single threshold for all slots)
      - dict with keys: "flop", "turn", "river"
        where flop threshold applies to flop1/2/3, etc.
    """
    if isinstance(thresholds, dict):
        flop_t = float(thresholds.get("flop", 20.0))
        turn_t = float(thresholds.get("turn", 50.0))
        river_t = float(thresholds.get("river", 50.0))
    else:
        flop_t = float(thresholds)
        turn_t = max(50.0, float(thresholds))
        river_t = max(50.0, float(thresholds))

    def present(name: str) -> bool:
        try:
            act = float(slots[name]["activity"])
        except Exception:
            return False

        if name in ("flop1", "flop2", "flop3"):
            return act >= flop_t
        if name == "turn":
            return act >= turn_t
        if name == "river":
            return act >= river_t
        return False

    flop_present_count = sum(1 for n in ["flop1", "flop2", "flop3"] if present(n))
    flop_present = flop_present_count == 3
    turn_present = present("turn")
    river_present = present("river")

    if river_present:
        street = "RIVER"
    elif turn_present:
        street = "TURN"
    elif flop_present:
        street = "FLOP"
    else:
        street = "PREFLOP"

    return {
        "thresholds": {"flop": flop_t, "turn": turn_t, "river": river_t},
        "flop_present_count": int(flop_present_count),
        "flop_present": bool(flop_present),
        "turn_present": bool(turn_present),
        "river_present": bool(river_present),
        "street": street,
    }


def collapse_street_timeline(page_streets: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    """
    Given (page_index, street) for pages, collapse consecutive duplicates.
    """
    out: List[Dict[str, Any]] = []
    last: Optional[str] = None
    for page_index, street in page_streets:
        if last is None or street != last:
            out.append({"page_index": int(page_index), "street": street})
            last = street
    return out


def infer_hand_windows(street_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Very lightweight heuristic:
    - A hand "starts" when we enter FLOP/TURN/RIVER from PREFLOP
    - A hand "ends" when we transition back to PREFLOP after having been in a later street
    This yields page-index windows; it is intentionally conservative.
    """
    windows: List[Dict[str, Any]] = []
    active_start: Optional[int] = None
    active_seen_postflop_hint = False  # any street != PREFLOP

    for i, item in enumerate(street_timeline):
        page = int(item["page_index"])
        street = str(item["street"])

        if street != "PREFLOP":
            active_seen_postflop_hint = True
            if active_start is None:
                active_start = page

        if street == "PREFLOP" and active_start is not None and active_seen_postflop_hint:
            # end page is previous change-point page (best we can do from change points)
            prev_page = int(street_timeline[i - 1]["page_index"]) if i > 0 else page
            windows.append(
                {
                    "hand_index": len(windows) + 1,
                    "start_page_index": active_start,
                    "end_page_index": prev_page,
                    "end_reason": "street_reset_to_preflop",
                }
            )
            active_start = None
            active_seen_postflop_hint = False

    # If we end still in an active hand, leave it open-ended
    if active_start is not None and active_seen_postflop_hint:
        windows.append(
            {
                "hand_index": len(windows) + 1,
                "start_page_index": active_start,
                "end_page_index": None,
                "end_reason": "end_of_scan",
            }
        )

    return windows


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

    # Card-slot stats pages (detail view around first_active_page)
    slot_pages: int = int(opts.get("slot_pages", 2))

    # Timeline mode: compute street for every scanned page and collapse changes
    timeline: bool = bool(opts.get("timeline", True))
    include_hand_windows: bool = bool(opts.get("include_hand_windows", True))

    # Thresholds: allow either float (legacy) or dict {"flop":20,"turn":50,"river":50}
    slot_thresholds = opts.get("slot_thresholds", {"flop": 20.0, "turn": 50.0, "river": 50.0})

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
            max_pages_to_scan = min(int(opts.get("max_pages_to_scan", 75)), total_pages)

            # Baseline-jump detector settings
            BASELINE_PAGES = int(opts.get("baseline_pages", 8))
            MARGIN = float(opts.get("margin", 20.0))
            RATIO = float(opts.get("ratio", 2.0))

            activities: List[float] = []
            per_page: List[Dict[str, Any]] = []
            crop_box_px: Optional[List[int]] = None
            render_size_px: Optional[List[int]] = None

            # Timeline helpers
            page_streets: List[Tuple[int, str]] = []

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

                if timeline:
                    # compute street state from slot activity for this page
                    board_box = board_region_box(img)
                    slot_boxes = card_slot_boxes_from_board(board_box)
                    slots_out: Dict[str, Any] = {}
                    for name, box in slot_boxes.items():
                        slots_out[name] = {"activity": activity_for_box(img, box)}
                    street_state = street_state_from_slots(slots_out, slot_thresholds)
                    page_streets.append((i, street_state["street"]))

            # Baseline = median of first BASELINE_PAGES
            baseline_slice = activities[: min(BASELINE_PAGES, len(activities))]
            baseline = median(baseline_slice)

            # First page that “jumps”
            first_active_page: Optional[int] = None
            for i, act in enumerate(activities):
                if act >= baseline + MARGIN or (baseline > 0 and act >= baseline * RATIO):
                    first_active_page = i
                    break

            # Top pages by activity
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

            # Card-slot activity stats + street_state (detail view around first_active_page)
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
                        board_box = board_region_box(img)
                        slot_boxes = card_slot_boxes_from_board(board_box)

                        slots_out: Dict[str, Any] = {}
                        for name, box in slot_boxes.items():
                            slots_out[name] = {
                                "slot_box_px": box,
                                "activity": activity_for_box(img, box),
                            }

                        street_state = street_state_from_slots(slots_out, slot_thresholds)

                        card_slot_stats.append(
                            {
                                "page_index": p,
                                "board_box_px": board_box,
                                "slots": slots_out,
                                "street_state": street_state,
                            }
                        )
                    except Exception as e:
                        card_slot_stats.append({"page_index": p, "error": f"{type(e).__name__}: {str(e)}"})

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
                "slot_thresholds": street_state_from_slots(
                    {"flop1": {"activity": 0}, "flop2": {"activity": 0}, "flop3": {"activity": 0}, "turn": {"activity": 0}, "river": {"activity": 0}},
                    slot_thresholds
                )["thresholds"],
                "card_slot_stats": card_slot_stats,
            }

            if debug:
                result["per_page_activity_first_20"] = per_page[:20]

            if include_crops:
                result["sample_board_crops"] = sample_board_crops

            if timeline:
                street_timeline = collapse_street_timeline(page_streets)
                result["street_timeline"] = street_timeline
                if include_hand_windows:
                    result["hand_windows"] = infer_hand_windows(street_timeline)

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
            "street_state uses slot-specific thresholds (flop/turn/river) to reduce false positives.",
            "street_timeline collapses per-page street into change points to keep output small.",
            "hand_windows are heuristic page ranges inferred from street_timeline resets.",
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
