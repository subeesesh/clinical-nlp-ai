"""
main.py
-------
FastAPI entry-point for the Clinical NLP Diagnosis Assistant.

Endpoints
---------
GET  /         → health check
GET  /health   → model status
POST /diagnose → analyze a clinical note
"""

from __future__ import annotations

import sys
import os

# Ensure sibling modules (classifier, entity_extractor, preprocess) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from classifier import load_model, predict
from entity_extractor import extract_entities


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical NLP Diagnosis Assistant",
    description=(
        "AI-powered REST API that analyzes clinical notes, predicts diseases, "
        "and extracts medical entities such as symptoms, medications, and conditions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow any origin so the standalone HTML frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model – loaded once at startup ────────────────────────────────────────────

print("[STARTUP] Loading disease classifier …")
_model = load_model()
print("[STARTUP] Classifier ready ✓")


# ── Schemas ───────────────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        example="Patient reports frequent urination, fatigue, and high blood sugar levels.",
    )


class DiagnoseResponse(BaseModel):
    diagnosis: str
    confidence: float
    symptoms: list[str]
    medications: list[str]
    conditions: list[str]
    all_probabilities: dict[str, float]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def root():
    return {
        "message": "Clinical NLP Diagnosis Assistant API is running.",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Status"])
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/diagnose", response_model=DiagnoseResponse, tags=["Diagnosis"])
def diagnose(request: DiagnoseRequest):
    """
    Analyze a clinical note and return:
    - **diagnosis**         – predicted disease name
    - **confidence**        – model's confidence (0–1)
    - **symptoms**          – detected symptom keywords
    - **medications**       – detected medication keywords
    - **conditions**        – detected condition keywords
    - **all_probabilities** – probability for every disease class
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Clinical note text cannot be empty.")

    # --- Classification ---
    result = predict(_model, text)

    # --- Entity Extraction ---
    entities = extract_entities(text)

    return DiagnoseResponse(
        diagnosis          = result["diagnosis"],
        confidence         = result["confidence"],
        symptoms           = entities["symptoms"],
        medications        = entities["medications"],
        conditions         = entities["conditions"],
        all_probabilities  = result["all_probabilities"],
    )


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
