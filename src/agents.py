"""
agents.py — Multi-Agent PCOS Assessment System
================================================
Provides three specialised AI agents and an orchestrator:

  Agent 1: DataValidatorAgent         — validate & sanitise patient input
  Agent 2: ClinicalEvidenceRetriever  — RAG search + PubMed for evidence
  Agent 3: RiskAssessorAgent          — XGBoost + SHAP + NHANES context

  PCOSOrchestrator                    — chain all agents sequentially
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `src.*` imports resolve
# whether invoked as `python src/agents.py` or `python -m src.agents`
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_fetcher import (
    compute_percentile,
    fetch_nhanes_baseline,
    fetch_pubmed_papers,
)
from src.ml_model import PCOSPredictor, predict_pcos
from src.ollama_client import OllamaClient
from src.rag_system import RAGSystem

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 1 — Data Validator
# ═══════════════════════════════════════════════════════════════════════════
class DataValidatorAgent:
    """
    Validates patient input data for physiological plausibility and
    completeness before downstream processing.

    Tools used: Ollama Llama 3.2 (consistency reasoning)
    """

    SYSTEM_PROMPT = (
        "You are a medical data quality validator for a PCOS screening system. "
        "Given a patient data dictionary, check for:\n"
        "1. Required fields present and non-null\n"
        "2. Values within physiologically plausible ranges\n"
        "3. Internal consistency (e.g., BMI vs height/weight)\n"
        "4. Flag any suspicious or outlier values\n\n"
        "Respond ONLY with valid JSON in this schema:\n"
        '{"status": "valid"|"invalid", "flags": [{"field": "...", "issue": "...", '
        '"severity": "warning"|"error"}], "confidence_score": 0.0-1.0, '
        '"notes": "brief summary"}'
    )

    VALID_RANGES: dict[str, tuple[float, float]] = {
        " Age (yrs)": (12, 60),
        "Weight (Kg)": (25, 200),
        "Height(Cm) ": (120, 210),
        "BMI": (12, 65),
        "Pulse rate(bpm) ": (40, 150),
        "RR (breaths/min)": (8, 40),
        "Hb(g/dl)": (5, 20),
        "Cycle length(days)": (15, 90),
        "FSH(mIU/mL)": (0.1, 80),
        "LH(mIU/mL)": (0.1, 80),
        "TSH (mIU/L)": (0.01, 20),
        "PRL(ng/mL)": (0.1, 100),
        "Vit D3 (ng/mL)": (3, 150),
        "RBS(mg/dl)": (40, 400),
        "BP _Systolic (mmHg)": (70, 220),
        "BP _Diastolic (mmHg)": (40, 140),
        "Endometrium (mm)": (1, 25),
        "Follicle No. (L)": (0, 30),
        "Follicle No. (R)": (0, 30),
        "Avg. F size (L) (mm)": (1, 30),
        "Avg. F size (R) (mm)": (1, 30),
    }

    KEY_FIELDS = [
        " Age (yrs)", "BMI", "Cycle(R/I)", "Follicle No. (L)",
        "Follicle No. (R)", "LH(mIU/mL)", "FSH(mIU/mL)",
    ]

    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama

    def run(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """Validate *patient_data* and return structured results."""
        start = time.time()
        flags: list[dict[str, str]] = []
        validated = dict(patient_data)

        # ── programmatic range checks ───────────────────────────────────
        for field, (lo, hi) in self.VALID_RANGES.items():
            val = patient_data.get(field)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                flags.append({"field": field, "issue": f"Non-numeric value: {val}", "severity": "error"})
                continue
            if val < lo or val > hi:
                flags.append({
                    "field": field,
                    "issue": f"Value {val} outside expected range [{lo}–{hi}]",
                    "severity": "warning" if (lo * 0.5 <= val <= hi * 1.5) else "error",
                })

        # ── missing key fields ──────────────────────────────────────────
        missing = [f for f in self.KEY_FIELDS if patient_data.get(f) is None]
        for f in missing:
            flags.append({"field": f, "issue": "Required field missing", "severity": "warning"})

        # ── BMI consistency check ───────────────────────────────────────
        weight = patient_data.get("Weight (Kg)")
        height = patient_data.get("Height(Cm) ")
        bmi = patient_data.get("BMI")
        if weight and height and bmi:
            try:
                expected_bmi = float(weight) / ((float(height) / 100) ** 2)
                if abs(expected_bmi - float(bmi)) > 3:
                    flags.append({
                        "field": "BMI",
                        "issue": f"BMI={bmi} inconsistent with weight/height (expected ~{expected_bmi:.1f})",
                        "severity": "warning",
                    })
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # ── binary field checks ─────────────────────────────────────────
        binary_fields = [
            "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
            "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)",
            "Reg.Exercise(Y/N)", "Pregnant(Y/N)",
        ]
        for bf in binary_fields:
            val = patient_data.get(bf)
            if val is not None and val not in (0, 1, 0.0, 1.0):
                flags.append({"field": bf, "issue": f"Expected 0 or 1, got {val}", "severity": "error"})

        # ── LLM consistency analysis ────────────────────────────────────
        llm_analysis: dict[str, Any] = {}
        if self.ollama.is_available():
            try:
                subset = {k: v for k, v in patient_data.items() if v is not None}
                prompt = (
                    f"Validate this patient data for a PCOS screening:\n"
                    f"{json.dumps(subset, indent=2, default=str)}\n\n"
                    "Check for physiological plausibility and internal consistency."
                )
                llm_analysis = self.ollama.generate_json(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                )
            except Exception as exc:
                log.warning("LLM validation skipped: %s", exc)

        # ── determine overall status ────────────────────────────────────
        errors = [f for f in flags if f["severity"] == "error"]
        status = "invalid" if errors else "valid"
        confidence = max(0.0, 1.0 - len(flags) * 0.05)

        elapsed = time.time() - start
        return {
            "agent": "DataValidator",
            "status": status,
            "validated_data": validated,
            "flags": flags,
            "confidence_score": round(confidence, 2),
            "llm_analysis": llm_analysis,
            "elapsed_sec": round(elapsed, 2),
        }


def _ensure_str_list(val: Any) -> list[str]:
    """Normalise an LLM field that should be a list of strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x is not None and str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val]
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Agent 2 — Clinical Evidence Retriever
# ═══════════════════════════════════════════════════════════════════════════
class ClinicalEvidenceRetriever:
    """
    Retrieves and synthesises clinical evidence relevant to a patient
    profile from the local Chroma knowledge base and PubMed.

    Tools used:
      - Chroma vector DB (local paper search)
      - PubMed API (latest research)
      - Ollama embeddings (vectorise queries)
      - Ollama Llama 3.2 (synthesise evidence)
    """

    SYSTEM_PROMPT = (
        "You are a clinical researcher specialising in PCOS. Given a patient "
        "profile and retrieved medical literature, synthesise the evidence into "
        "a structured clinical summary. Address:\n"
        "1. Which Rotterdam criteria might be met\n"
        "2. Relevant hormone findings vs population norms\n"
        "3. Key diagnostic indicators from the literature\n"
        "4. Any red flags requiring attention\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"clinical_summary": "paragraph summarising the clinical picture", '
        '"diagnostic_criteria_met": ["criterion 1 description", "criterion 2 description"], '
        '"key_findings": ["finding 1", "finding 2"], '
        '"red_flags": ["flag 1 or empty list if none"]}'
    )

    def __init__(
        self,
        ollama: OllamaClient,
        rag: RAGSystem,
    ) -> None:
        self.ollama = ollama
        self.rag = rag

    _MAX_QUERY_TERMS = 4

    def _build_query(self, validated_data: dict[str, Any]) -> str:
        """Construct a concise search query from the most notable patient characteristics."""
        parts: list[str] = []

        bmi = validated_data.get("BMI")
        if bmi and float(bmi) > 25:
            parts.append("elevated BMI obesity")

        lh = validated_data.get("LH(mIU/mL)")
        fsh = validated_data.get("FSH(mIU/mL)")
        if lh and fsh:
            try:
                ratio = float(lh) / float(fsh) if float(fsh) > 0 else 0
                if ratio > 2:
                    parts.append("elevated LH/FSH ratio")
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        if validated_data.get("hair growth(Y/N)") == 1:
            parts.append("hirsutism androgen excess")
        if validated_data.get("Skin darkening (Y/N)") == 1:
            parts.append("acanthosis nigricans insulin resistance")
        if validated_data.get("Pimples(Y/N)") == 1:
            parts.append("acne")
        if validated_data.get("Weight gain(Y/N)") == 1:
            parts.append("weight gain metabolic")
        if validated_data.get("Cycle(R/I)") and int(validated_data["Cycle(R/I)"]) != 1:
            parts.append("irregular menstrual cycle oligomenorrhea")

        follicle_l = validated_data.get("Follicle No. (L)", 0) or 0
        follicle_r = validated_data.get("Follicle No. (R)", 0) or 0
        if (float(follicle_l) + float(follicle_r)) > 20:
            parts.append("polycystic ovarian morphology")

        # Keep top terms to avoid PubMed returning 0 results on overly long queries
        top_parts = parts[: self._MAX_QUERY_TERMS]
        return "PCOS " + " ".join(top_parts)

    def run(self, validated_data: dict[str, Any]) -> dict[str, Any]:
        """Retrieve and synthesise clinical evidence for the patient."""
        start = time.time()

        query = self._build_query(validated_data)
        log.info("Evidence query: %s", query)

        # ── Tool 1: Chroma local papers ─────────────────────────────────
        local_papers: list[dict] = []
        try:
            local_papers = self.rag.retrieve_papers(query, n_results=3)
        except Exception as exc:
            log.warning("Chroma retrieval failed: %s", exc)

        # ── Tool 2: PubMed API ──────────────────────────────────────────
        pubmed_papers: list[dict] = []
        try:
            pubmed_papers = fetch_pubmed_papers(query, max_papers=3)
        except Exception as exc:
            log.warning("PubMed fetch failed: %s", exc)

        # ── Tool 3 & 4: Ollama synthesis ────────────────────────────────
        all_evidence = ""
        for p in local_papers:
            title = p.get("metadata", {}).get("title", "Local paper")
            all_evidence += f"\n[Local] {title}:\n{p.get('document', '')[:500]}\n"

        for p in pubmed_papers:
            all_evidence += f"\n[PubMed: {p.get('pmid', '')}] {p.get('title', '')}:\n"
            all_evidence += f"{p.get('abstract', 'No abstract')[:500]}\n"

        clinical_analysis: dict[str, Any] = {}
        if self.ollama.is_available() and all_evidence.strip():
            try:
                prompt = (
                    f"PATIENT PROFILE:\n{json.dumps({k: v for k, v in validated_data.items() if v is not None}, indent=2, default=str)}\n\n"
                    f"RETRIEVED EVIDENCE:\n{all_evidence}\n\n"
                    "Synthesise the evidence into a clinical summary for this patient."
                )
                clinical_analysis = self.ollama.generate_json(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                )
            except Exception as exc:
                log.warning("LLM synthesis failed: %s", exc)

        elapsed = time.time() - start
        return {
            "agent": "ClinicalEvidenceRetriever",
            "query_used": query,
            "retrieved_papers": [
                {
                    "source": "chroma",
                    "title": p.get("metadata", {}).get("title", ""),
                    "year": p.get("metadata", {}).get("year", ""),
                    "distance": p.get("distance", 0),
                    "excerpt": p.get("document", "")[:300],
                }
                for p in local_papers
            ],
            "pubmed_papers": [
                {
                    "source": "pubmed",
                    "pmid": p.get("pmid", ""),
                    "title": p.get("title", ""),
                    "pubdate": p.get("pubdate", ""),
                }
                for p in pubmed_papers
            ],
            "clinical_summary": str(clinical_analysis.get("clinical_summary", "")),
            "diagnostic_criteria": _ensure_str_list(clinical_analysis.get("diagnostic_criteria_met")),
            "key_findings": _ensure_str_list(clinical_analysis.get("key_findings")),
            "red_flags": _ensure_str_list(clinical_analysis.get("red_flags")),
            "elapsed_sec": round(elapsed, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Agent 3 — Risk Assessor
# ═══════════════════════════════════════════════════════════════════════════
class RiskAssessorAgent:
    """
    Predicts PCOS risk using the XGBoost model and contextualises results
    with SHAP explanations, NHANES population data, and LLM reasoning.

    Tools used:
      - XGBoost model (predict risk 0–1)
      - SHAP (explain feature contributions)
      - NHANES API (population percentiles)
      - Ollama Llama 3.2 (synthesise recommendation)
    """

    SYSTEM_PROMPT = (
        "You are a clinical decision-support AI for PCOS screening. Given a "
        "patient's ML risk prediction, SHAP feature explanations, population "
        "context, and clinical evidence, generate a clinical recommendation.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"recommendation": "...", "confidence_assessment": "...", '
        '"follow_up_tests": [...], "lifestyle_suggestions": [...]}'
    )

    def __init__(
        self,
        ollama: OllamaClient,
        predictor: PCOSPredictor | None = None,
    ) -> None:
        self.ollama = ollama
        self.predictor = predictor

    def _ensure_predictor(self) -> PCOSPredictor:
        if self.predictor is None:
            self.predictor = PCOSPredictor.get_instance()
        return self.predictor

    def run(
        self,
        validated_data: dict[str, Any],
        clinical_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Predict risk, explain, contextualise, and recommend."""
        start = time.time()
        predictor = self._ensure_predictor()

        # ── Tool 1 & 2: XGBoost prediction + SHAP ──────────────────────
        prediction = predictor.predict(validated_data, top_n=5)

        top_factors = [
            {
                "feature": rf.feature,
                "shap_value": rf.shap_value,
                "raw_value": rf.raw_value,
                "direction": rf.direction,
                "magnitude": rf.magnitude,
            }
            for rf in prediction.top_risk_factors
        ]

        # ── Tool 3: NHANES population context ──────────────────────────
        population_context: list[dict[str, Any]] = []
        context_features = [
            "LH(mIU/mL)", "FSH(mIU/mL)", "TSH (mIU/L)", "BMI",
            "Hb(g/dl)", "RBS(mg/dl)", "Vit D3 (ng/mL)", "PRL(ng/mL)",
        ]
        for feat in context_features:
            val = validated_data.get(feat)
            if val is not None:
                try:
                    pct = compute_percentile(feat, float(val))
                    if pct:
                        population_context.append(pct)
                except (TypeError, ValueError):
                    pass

        # ── Tool 4: Ollama recommendation synthesis ─────────────────────
        llm_recommendation: dict[str, Any] = {}
        if self.ollama.is_available():
            try:
                context_summary = "\n".join(
                    f"  {c['hormone']}: {c['value']} {c['unit']} "
                    f"({c['percentile']}th percentile — {c['interpretation']})"
                    for c in population_context
                )
                prompt = (
                    f"RISK PREDICTION:\n"
                    f"  Risk score: {prediction.risk_score}\n"
                    f"  Risk label: {prediction.risk_label}\n"
                    f"  Model AUROC: {prediction.model_auroc}\n\n"
                    f"TOP CONTRIBUTING FACTORS (SHAP):\n"
                    + "\n".join(
                        f"  {f['feature']}: SHAP={f['shap_value']:+.4f} "
                        f"(value={f['raw_value']}, {f['direction']} risk, {f['magnitude']} impact)"
                        for f in top_factors
                    )
                    + f"\n\nPOPULATION CONTEXT (NHANES):\n{context_summary}\n\n"
                    f"CLINICAL EVIDENCE SUMMARY:\n{clinical_evidence.get('clinical_summary', 'N/A')}\n\n"
                    "Generate a clinical recommendation for this patient."
                )
                llm_recommendation = self.ollama.generate_json(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                )
            except Exception as exc:
                log.warning("LLM recommendation failed: %s", exc)

        elapsed = time.time() - start
        return {
            "agent": "RiskAssessor",
            "risk_score": prediction.risk_score,
            "risk_label": prediction.risk_label,
            "predicted_class": prediction.predicted_class,
            "confidence": prediction.confidence,
            "model_auroc": prediction.model_auroc,
            "threshold_used": prediction.threshold_used,
            "explanation_text": prediction.explanation_text,
            "top_factors": top_factors,
            "population_context": population_context,
            "recommendation": llm_recommendation.get("recommendation", ""),
            "confidence_assessment": llm_recommendation.get("confidence_assessment", ""),
            "follow_up_tests": llm_recommendation.get("follow_up_tests", []),
            "lifestyle_suggestions": llm_recommendation.get("lifestyle_suggestions", []),
            "elapsed_sec": round(elapsed, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator — Chains Agent 1 → 2 → 3
# ═══════════════════════════════════════════════════════════════════════════
class PCOSOrchestrator:
    """
    Coordinate the three agents in sequence:
      1. DataValidatorAgent       → validate input
      2. ClinicalEvidenceRetriever → retrieve evidence
      3. RiskAssessorAgent         → predict + recommend

    Optionally persists results to Supabase.
    """

    def __init__(
        self,
        ollama: OllamaClient | None = None,
        rag: RAGSystem | None = None,
        predictor: PCOSPredictor | None = None,
        db: Any | None = None,
    ) -> None:
        self.ollama = ollama or OllamaClient()
        self.rag = rag or RAGSystem(ollama=self.ollama)
        self.predictor = predictor

        self.validator = DataValidatorAgent(self.ollama)
        self.evidence_retriever = ClinicalEvidenceRetriever(self.ollama, self.rag)
        self.risk_assessor = RiskAssessorAgent(self.ollama, self.predictor)

        self.db = db

    def run(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """
        Run the full multi-agent pipeline.

        Returns a dict with keys:
          ``validation``, ``evidence``, ``assessment``, ``metadata``.
        """
        pipeline_start = time.time()

        # ── Agent 1: Validate ───────────────────────────────────────────
        log.info("Agent 1: Validating patient data …")
        validation = self.validator.run(patient_data)

        if validation["status"] == "invalid":
            errors = [f for f in validation["flags"] if f["severity"] == "error"]
            return {
                "validation": validation,
                "evidence": {},
                "assessment": {},
                "metadata": {
                    "status": "rejected",
                    "reason": f"{len(errors)} validation error(s)",
                    "elapsed_sec": round(time.time() - pipeline_start, 2),
                },
            }

        validated = validation["validated_data"]

        # ── Agent 2: Retrieve evidence ──────────────────────────────────
        log.info("Agent 2: Retrieving clinical evidence …")
        evidence = self.evidence_retriever.run(validated)

        # ── Agent 3: Assess risk ────────────────────────────────────────
        log.info("Agent 3: Assessing PCOS risk …")
        assessment = self.risk_assessor.run(validated, evidence)

        pipeline_elapsed = time.time() - pipeline_start

        # ── Persist to Supabase (if configured) ─────────────────────────
        patient_id = None
        if self.db is not None:
            try:
                from src.database import SupabaseClient

                if isinstance(self.db, SupabaseClient) and self.db.is_configured():
                    patient_id = self.db.store_patient(patient_data)
                    self.db.store_prediction(
                        patient_id=patient_id,
                        risk_score=assessment["risk_score"],
                        risk_label=assessment["risk_label"],
                        confidence=assessment.get("confidence"),
                        top_factors=assessment.get("top_factors"),
                        clinical_summary=evidence.get("clinical_summary", ""),
                        recommendation=assessment.get("recommendation", ""),
                        agent_outputs={
                            "validation": validation,
                            "evidence": evidence,
                            "assessment": assessment,
                        },
                    )
                    self.db.audit_log(patient_id, "pipeline_completed")
            except Exception as exc:
                log.warning("Supabase persistence failed: %s", exc)

        return {
            "validation": validation,
            "evidence": evidence,
            "assessment": assessment,
            "metadata": {
                "status": "completed",
                "patient_id": patient_id,
                "elapsed_sec": round(pipeline_elapsed, 2),
                "agents_used": [
                    "DataValidator",
                    "ClinicalEvidenceRetriever",
                    "RiskAssessor",
                ],
            },
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sample_patient = {
        " Age (yrs)": 28,
        "Weight (Kg)": 75,
        "Height(Cm) ": 162,
        "BMI": 28.6,
        "Blood Group": 3,
        "Pulse rate(bpm) ": 78,
        "RR (breaths/min)": 18,
        "Hb(g/dl)": 12.5,
        "Cycle(R/I)": 2,
        "Cycle length(days)": 35,
        "Marraige Status (Yrs)": 3,
        "Pregnant(Y/N)": 0,
        "No. of aborptions": 0,
        "  I   beta-HCG(mIU/mL)": 1.5,
        "FSH(mIU/mL)": 4.5,
        "LH(mIU/mL)": 11.2,
        "FSH/LH": 0.4,
        "Hip(inch)": 40,
        "Waist(inch)": 34,
        "Waist:Hip Ratio": 0.85,
        "TSH (mIU/L)": 3.2,
        "PRL(ng/mL)": 14.0,
        "Vit D3 (ng/mL)": 18.0,
        "PRG(ng/mL)": 0.8,
        "RBS(mg/dl)": 105,
        "Weight gain(Y/N)": 1,
        "hair growth(Y/N)": 1,
        "Skin darkening (Y/N)": 1,
        "Hair loss(Y/N)": 0,
        "Pimples(Y/N)": 1,
        "Fast food (Y/N)": 1,
        "Reg.Exercise(Y/N)": 0,
        "BP _Systolic (mmHg)": 120,
        "BP _Diastolic (mmHg)": 80,
        "Follicle No. (L)": 12,
        "Follicle No. (R)": 14,
        "Avg. F size (L) (mm)": 16,
        "Avg. F size (R) (mm)": 17,
        "Endometrium (mm)": 8.5,
        "LH_FSH_ratio": 2.49,
        "follicle_total": 26,
        "follicle_asymmetry": 2,
    }

    orch = PCOSOrchestrator()
    result = orch.run(sample_patient)

    print("\n" + "=" * 60)
    print("PCOSENSE — Multi-Agent Assessment Result")
    print("=" * 60)

    v = result["validation"]
    print(f"\n[Agent 1] Validation: {v['status']} (confidence {v['confidence_score']})")
    if v["flags"]:
        for f in v["flags"][:5]:
            print(f"  ⚠ {f['field']}: {f['issue']}")

    e = result["evidence"]
    if e:
        print(f"\n[Agent 2] Evidence Retrieved:")
        print(f"  Local papers : {len(e.get('retrieved_papers', []))}")
        print(f"  PubMed papers: {len(e.get('pubmed_papers', []))}")
        if e.get("clinical_summary"):
            print(f"  Summary: {e['clinical_summary'][:200]}…")

    a = result["assessment"]
    if a:
        print(f"\n[Agent 3] Risk Assessment:")
        print(f"  Risk score : {a.get('risk_score', 'N/A')}")
        print(f"  Risk label : {a.get('risk_label', 'N/A')}")
        print(f"  Confidence : {a.get('confidence', 'N/A')}")
        if a.get("top_factors"):
            print("  Top factors:")
            for tf in a["top_factors"][:3]:
                print(f"    {tf['feature']:30s} SHAP={tf['shap_value']:+.4f}")
        if a.get("recommendation"):
            print(f"  Recommendation: {a['recommendation'][:200]}…")

    m = result["metadata"]
    print(f"\nPipeline: {m['status']} in {m['elapsed_sec']:.1f}s")
    print("=" * 60)
