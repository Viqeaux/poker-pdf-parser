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

    pdf_infos = []

    for ref in refs:
        if isinstance(ref, dict) and "download_link" in ref:
            url = ref["download_link"]
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            pdf_bytes = r.content

            # Open PDF in memory and count pages
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = doc.page_count

            # Get size of first/last page (helps confirm consistent capture)
            first = doc.load_page(0).rect
            last = doc.load_page(page_count - 1).rect
            doc.close()

            pdf_infos.append({
                "bytes": len(pdf_bytes),
                "pages": page_count,
                "first_page_size_pts": [float(first.width), float(first.height)],
                "last_page_size_pts": [float(last.width), float(last.height)],
            })

    return {
        "hand_history_text": (
            "DEBUG MODE\n"
            f"Files received: {len(refs)}\n"
            f"PDF infos: {pdf_infos}"
        ),
        "warnings": [
            "Parser is in DEBUG mode.",
            "Only downloading PDFs and counting pages (no poker parsing yet)."
        ],
        "hands_detected": 0,
    }
