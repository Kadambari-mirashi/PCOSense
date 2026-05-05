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
from src.quality_control import QualityController

log = logging.getLogger(__name__)

_orchestrator: PCOSOrchestrator | None = None
_qc: QualityController | None = None


def get_orchestrator() -> PCOSOrchestrator:
    """Return singleton orchestrator; safe to call after startup."""
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized — call during startup")
    return _orchestrator


def get_qc() -> QualityController:
    """Return singleton QC controller; safe to call after startup."""
    if _qc is None:
        raise RuntimeError("QualityController not initialized — call during startup")
    return _qc


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _qc
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Initialize heavy resources once at startup, not on first request
    db = SupabaseClient()
    _orchestrator = PCOSOrchestrator(db=db if db.is_configured() else None)
    _qc = QualityController()

    # Preload XGBoost predictor + SHAP explainer so the first request is fast.
    try:
        from src.ml_model import PCOSPredictor
        PCOSPredictor.get_instance()
    except Exception as exc:  # noqa: BLE001
        log.warning("Predictor preload failed: %s", exc)

    log.info("PCOSense pipeline initialized")
    yield
    _orchestrator = None
    _qc = None


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Default to localhost only; set CORS_ORIGINS env var for production
    return ["http://127.0.0.1:3838", "http://localhost:3838"]


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

    Returns ``validation``, ``evidence``, ``assessment``, ``quality_control``, and ``metadata``.
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
        
        # Generate quality control metrics
        qc_metrics = get_qc().create_metrics_report(
            patient_data=patient,
            prediction_result=result.get("assessment", {}),
            rag_results=result.get("evidence", {}),
        )
        
        result["quality_control"] = qc_metrics.to_dict()
        
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


@app.get("/api/v1/quality-summary")
def quality_summary() -> dict:
    """
    Get system-wide quality control performance metrics.
    
    Shows aggregate statistics about assessment quality across all predictions,
    useful for stakeholder dashboards and compliance reporting.
    """
    return get_qc().get_performance_summary()


# ── Mount Shiny frontend so everything runs as a single service ──────────
from src.app.app import app as shiny_app  # noqa: E402

app.mount("/", shiny_app)
