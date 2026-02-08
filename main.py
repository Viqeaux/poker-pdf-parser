# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional
import requests
import fitz  # PyMuPDF

app = FastAPI()

class ParseRequest(BaseModel):
    openaiFileIdRefs: List[Any]
    options: Optional[dict] = None

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

            info = {
                "bytes": len(pdf_bytes),
                "pages": doc.page_count,
                "rendered_pages": []
            }

            # Render first two pages (or fewer if short PDF)
            pages_to_render = min(2, doc.page_count)
            for i in range(pages_to_render):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=150)
                info["rendered_pages"].append({
                    "page_index": i,
                    "image_width_px": pix.width,
                    "image_height_px": pix.height
                })

            doc.close()
            results.append(info)

    return {
        "hand_history_text": (
            "DEBUG MODE\n"
            f"Files received: {len(refs)}\n"
            f"PDF render info: {results}"
        ),
        "warnings": [
            "Parser is in DEBUG mode.",
            "Rendered first two pages to images (no OCR, no poker parsing)."
        ],
        "hands_detected": 0,
    }
