# main.py
from fastapi import FastAPI, Request
from typing import Any, List

app = FastAPI()

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(request: Request):
    body: Any = await request.json()

    # Safely extract file refs no matter how ChatGPT sends them
    raw_refs = body.get("openaiFileIdRefs", [])
    normalized: List[str] = []

    for item in raw_refs:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(
                item.get("id")
                or item.get("file_id")
                or item.get("name")
                or str(item)
            )
        else:
            normalized.append(str(item))

    fake_text = (
        "PokerStars Hand #0000000000:  No Limit Hold'em ($0.02/$0.05 USD) - 2026/01/01 00:00:00 ET\n"
        "Table 'Test Table' 6-max Seat #1 is the button\n"
        "Seat 1: Player1 ($5.00 in chips)\n"
        "Seat 2: Player2 ($5.00 in chips)\n"
        "Seat 3: Player3 ($5.00 in chips)\n"
        "Player2: posts small blind $0.02\n"
        "Player3: posts big blind $0.05\n"
        "*** HOLE CARDS ***\n"
        "Player1: folds\n"
        "Player2: calls $0.03\n"
        "Player3: checks\n"
        "*** FLOP *** [As Kh Tc]\n"
        "Player2: bets $0.10\n"
        "Player3: folds\n"
        "Uncalled bet ($0.10) returned to Player2\n"
        "Player2 collected $0.12 from pot\n"
        "*** SUMMARY ***\n"
    )

    return {
        "hand_history_text": fake_text,
        "warnings": [
            "This is a fake response from the placeholder server.",
            f"Received file refs: {normalized}",
        ],
        "hands_detected": 1,
    }
