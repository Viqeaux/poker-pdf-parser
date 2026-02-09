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


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def board_region_box(img: Image.Image) -> List[int]:
    """
    Heuristic crop for the community-card board area on a typical 16:9 poker table capture.
    Uses normalized coordinates, returns [left, top, right, bottom] in pixels.
    """
    w, h = img.size
    left = int(w * 0.34)
    top = int(h * 0.26)
    right = int(w * 0.66)
    bottom = int(h * 0.48)
    return [left, top, right, bottom]


def board_region_stats(img: Image.Image) -> Dict[str, Any]:
    box = board_region_box(img)
    crop = img.crop(tuple(box))
    stat = ImageStat.Stat(crop)

    mean = [float(x) for x in stat.mean]
    stddev = [float(x) for x in stat.stddev]
    activity = sum(stddev) / max(len(stddev), 1)

    return {
        "crop_box_px": box,
        "mean_rgb": mean,
        "stddev_rgb": stddev,
        "activity": float(activity),
        "crop": crop,
    }


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[len(s) // 2]


@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):
    refs = req.openaiFileIdRefs or []
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

            # Baseline-jump detector settings
            BASELINE_PAGES = 8
            MARGIN = 20.0
            RATIO = 2.0

            per_page: List[Dict[str, Any]] = []
            activities: List[float] = []

            crop_box_px: Optional[List[int]] = None
            render_size_px: Optional[List[int]] = None

            # Scan pages and collect activity scores
            for i in range(max_pages_to_scan):
                img = render_page_to_pil(doc, i, dpi=150)
                if render_size_px is None:
                    render_size_px = [img.size[0], img.size[1]]

                stats = board_region_stats(img)
                if crop_box_px is None:
                    crop_box_px = stats["crop_box_px"]

                act = float(stats["activity"])
                activities.append(act)

                per_page.append(
                    {
                        "page_index": i,
                        "activity": act,
                    }
                )

            # Compute baseline = median of first BASELINE_PAGES
            baseline_slice = activities[: min(BASELINE_PAGES, len(activities))]
            baseline = median(baseline_slice)

            # Find first page that jumps above baseline significantly
            first_active_page: Optional[int] = None
            for i, act in enumerate(activities):
                if act >= baseline + MARGIN or (baseline > 0 and act >= baseline * RATIO):
                    first_active_page = i
                    break

            # Top activity pages
            indexed = list(enumerate(activities))
            indexed.sort(key=lambda t: t[1], reverse=True)
            top_active_pages = [
                {"page_index": idx, "activity": float(val)} for idx, val in indexed[:10]
            ]

            # Sample crops (page 0, first_active_page, first_active_page+1)
            sample_pages: List[int] = [0]
            if first_active_page is not None:
                sample_pages.append(first_active_page)
                sample_pages.append(first_active_page + 1)

            # Deduplicate, clamp to range, preserve order
            seen = set()
            sample_pages_clean: List[int] = []
            for p in sample_pages:
                if 0 <= p < total_pages and p not in seen:
                    sample_pages_clean.append(p)
                    seen.add(p)

            sample_board_crops: List[Dict[str, Any]] = []
            for p in sample_pages_clean:
                try:
                    img = render_page_to_pil(doc, p, dpi=150)
                    box = board_region_box(img)
                    crop = img.crop(tuple(box))
                    sample_board_crops.append(
                        {
                            "page_index": p,
                            "crop_box_px": box,
                            "board_crop_base64_png": pil_to_base64_png(crop),
                        }
                    )
                except Exception as e:
                    sample_board_crops.append(
                        {
                            "page_index": p,
                            "error": f"{type(e).__name__}: {str(e)}",
                        }
                    )

            results.append(
                {
                    "bytes": len(pdf_bytes),
                    "pages": total_pages,
                    "scanned_pages": max_pages_to_scan,
                    "render_dpi": 150,
                    "render_size_px_example": render_size_px,
                    # Baseline detector params + computed baseline
                    "baseline_pages": BASELINE_PAGES,
                    "baseline_activity_median_first_n": baseline,
                    "baseline_margin": MARGIN,
                    "baseline_ratio": RATIO,
                    # Output
                    "first_board_region_active_page": first_active_page,
                    "board_crop_box_px": crop_box_px,
                    "top_active_pages": top_active_pages,
                    "per_page_activity_first_20": per_page[:20],
                    "sample_board_crops": sample_board_crops,
                }
            )
        finally:
            doc.close()

    debug_payload = {
        "mode": "DEBUG",
        "files_received": len(refs),
        "results": results,
        "notes": [
            "This does not read card ranks/suits.",
            "It detects a jump in visual activity relative to an early-page baseline in a heuristic board region.",
            "It also returns sample crops for validation.",
        ],
    }

    return {
        "hand_history_text": json.dumps(debug_payload, indent=2),
        "warnings": [
            "Parser is in DEBUG mode.",
            "This does not read card ranks/suits. It only detects visual activity in a heuristic board region.",
        ],
        "hands_detected": 0,
    }
