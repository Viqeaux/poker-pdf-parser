# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional
import requests
import fitz  # PyMuPDF
from PIL import Image, ImageStat
import io

app = FastAPI()

class ParseRequest(BaseModel):
    openaiFileIdRefs: List[Any]
    options: Optional[dict] = None

def render_page_to_pil(doc: fitz.Document, page_index: int, dpi: int = 150) -> Image.Image:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img.convert("RGB")

def board_region_stats(img: Image.Image) -> dict:
    """
    Heuristic crop for community-card board area on a typical 16:9 poker table capture.
    Your pages render at 960x540 (at 150 dpi in our setup), so these coords are tuned for that.
    If needed later, we can adjust the crop box.
    """
    w, h = img.size
    # Crop roughly center area where flop/turn/river cards usually appear
    left = int(w * 0.34)
    top = int(h * 0.26)
    right = int(w * 0.66)
    bottom = int(h * 0.48)

    crop = img.crop((left, top, right, bottom))
    stat = ImageStat.Stat(crop)

    # Mean and stddev across RGB channels
    mean = [float(x) for x in stat.mean]
    stddev = [float(x) for x in stat.stddev]

    # A single "activity" score: average stddev across channels
    activity = sum(stddev) / max(len(stddev), 1)

    return {
        "crop_box_px": [left, top, right, bottom],
        "mean_rgb": mean,
        "stddev_rgb": stddev,
        "activity": activity,
    }

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):
    refs = req.openaiFileIdRefs or []

    results = []

    for ref in refs:
        if isinstance(ref, dict) and "download_link" in ref:
            url = ref["download_link"]
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            pdf_bytes = r.content

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            max_pages_to_scan = min(75, doc.page_count)  # scan up to 75 pages for now
            per_page = []

            # We will find the first page where the board-region "activity" crosses a threshold.
            # This is NOT OCR. It is a pixel-variance signal that something visually changed in that region.
            THRESH_ACTIVITY = 10.0

            first_active_page = None

            for i in range(max_pages_to_scan):
                img = render_page_to_pil(doc, i, dpi=150)
                stats = board_region_stats(img)
                per_page.append({
                    "page_index": i,
                    "activity": stats["activity"],
                })
                if first_active_page is None and stats["activity"] >= THRESH_ACTIVITY:
                    first_active_page = i

            doc.close()

            results.append({
                "bytes": len(pdf_bytes),
                "pages": doc.page_count,
                "scanned_pages": max_pages_to_scan,
                "threshold_activity": THRESH_ACTIVITY,
                "first_board_region_active_page": first_active_page,
                "per_page_activity_first_20": per_page[:20],
            })

    return {
        "hand_history_text": (
            "DEBUG MODE\n"
            f"Files received: {len(refs)}\n"
            f"Board-region activity scan results: {results}"
        ),
        "warnings": [
            "Parser is in DEBUG mode.",
            "This does not read card ranks/suits. It only detects visual activity in a heuristic board region."
        ],
        "hands_detected": 0,
    }
