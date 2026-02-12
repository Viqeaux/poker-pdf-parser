# main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import traceback
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUILD_ID = "diagnostic-v1"


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok"}


# --------------------------------------------------
# Diagnostic Endpoint
# --------------------------------------------------

@app.post("/parse_poker_pdf")
async def parse_poker_pdf(request: Request):

    try:
        raw_body = await request.json()
    except Exception:
        raw_body = {"error": "could not parse json"}

    try:
        headers = dict(request.headers)
    except Exception:
        headers = {"error": "could not read headers"}

    return {
        "BUILD_ID": BUILD_ID,
        "received_body": raw_body,
        "received_headers": headers
    }
