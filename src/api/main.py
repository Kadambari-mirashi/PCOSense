"""
FastAPI application — exposes the multi-agent PCOS assessment pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

# Project root on path for `src.*` imports
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

load_dotenv(_ROOT + "/.env")

from src.agents import PCOSOrchestrator
from src.api.schemas import PatientAssessmentRequest, patient_dict_from_request
from src.database import SupabaseClient

log = logging.getLogger(__name__)

_orchestrator: PCOSOrchestrator | None = None


def get_orchestrator() -> PCOSOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        db = SupabaseClient()
        _orchestrator = PCOSOrchestrator(db=db if db.is_configured() else None)
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    yield
    global _orchestrator
    _orchestrator = None


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://127.0.0.1:3838",
        "http://localhost:3838",
        "http://127.0.0.1:7860",
        "http://localhost:7860",
    ]


app = FastAPI(
    title="PCOSense API",
    description="REST API for the PCOSense multi-agent screening pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "service": "pcosense-api"}


@app.post("/api/v1/assess")
def assess_patient(body: PatientAssessmentRequest) -> dict:
    """
    Run Data Validator → Evidence Retriever → Risk Assessor.

    Returns ``validation``, ``evidence``, ``assessment``, and ``metadata``.
    If Supabase is configured, results are persisted automatically.
    """
    patient = patient_dict_from_request(body)
    if not patient:
        raise HTTPException(
            status_code=400,
            detail="No patient fields provided. Send at least one clinical feature.",
        )

    try:
        result = get_orchestrator().run(patient)
    except FileNotFoundError as exc:
        log.exception("Model or artifact missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return jsonable_encoder(result)


@app.get("/api/v1/feature-info")
def feature_info() -> dict:
    """Expose model feature metadata for clients (optional)."""
    from src.ml_model import PCOSPredictor

    try:
        return PCOSPredictor.get_instance().feature_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
