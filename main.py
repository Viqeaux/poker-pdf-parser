# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional
import requests

app = FastAPI()

class ParseRequest(BaseModel):
    openaiFileIdRefs: List[Any]
    options: Optional[dict] = None

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(req: ParseRequest):
    refs = req.openaiFileIdRefs or []

    downloaded_bytes = []

    for ref in refs:
        if isinstance(ref, dict) and "download_link" in ref:
            url = ref["download_link"]
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            downloaded_bytes.append(len(r.content))

    return {
        "hand_history_text": (
            "DEBUG MODE\n"
            f"Files received: {len(refs)}\n"
            f"Files downloaded: {len(downloaded_bytes)}\n"
            f"Byte sizes: {downloaded_bytes}"
        ),
        "warnings": [
            "Parser is in DEBUG mode.",
            "PDF bytes successfully downloaded, no parsing yet."
        ],
        "hands_detected": 0,
    }
