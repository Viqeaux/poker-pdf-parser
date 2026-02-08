# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class ParseRequest(BaseModel):
    openaiFileIdRefs: List[str]
    options: Optional[dict] = None

class ParseResponse(BaseModel):
    hand_history_text: str
    warnings: List[str]
    hands_detected: int

@app.post("/parse_poker_pdf", response_model=ParseResponse)
async def parse_poker_pdf(req: ParseRequest):
    # This is a placeholder implementation that proves the endpoint works.
    # It returns a fake PokerStars-style hand history so you can finish wiring.
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
    return ParseResponse(
        hand_history_text=fake_text,
        warnings=["This is a fake response from the placeholder server."],
        hands_detected=1,
    )
