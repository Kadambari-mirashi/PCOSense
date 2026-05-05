"""
quality_control.py — QC & Validation System for PCOSense
=========================================================

Provides structured quality metrics, validation scores, and evidence of AI
system performance. Demonstrates:
  - Input validation success rates
  - Model confidence & prediction quality
  - Agent reliability metrics
  - Output sanity checks
  - Evidence scoring for clinical results

Used by: agents.py + app.py to track and display QC metrics to stakeholders.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class QCStatus(str, Enum):
    """Quality check result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationFlag:
    """A single validation check result."""
    check_name: str
    status: QCStatus
    description: str
    severity: str = "info"  # "error", "warning", "info"


@dataclass
class QCMetrics:
    """Aggregated QC results for a single assessment."""
    timestamp: str
    input_validation_score: float  # 0–1: success of data validation
    model_confidence: float  # 0–1: XGBoost probability strength
    prediction_plausibility_score: float  # 0–1: output sanity checks
    rag_evidence_score: float  # 0–1: quality of retrieved clinical evidence
    overall_quality_score: float  # 0–1: weighted average
    validation_flags: list[ValidationFlag] = field(default_factory=list)
    model_auroc: float = 0.9528  # Known model AUROC from training
    prediction_confidence_label: str = "medium"  # "high" | "medium" | "low"
    key_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = asdict(self)
        result["validation_flags"] = [
            {
                "check_name": f.check_name,
                "status": f.status.value,
                "description": f.description,
                "severity": f.severity,
            }
            for f in self.validation_flags
        ]
        return result


def _get(d: dict, *keys: str):
    """Return the first present value among `keys` in `d`, else None.

    Used to bridge clean API keys (e.g. ``age``) and the legacy pipeline
    keys (e.g. `` Age (yrs)``) that come from the raw dataset.
    """
    for k in keys:
        if k in d:
            return d[k]
    return None


class QualityController:
    """
    Main QC orchestrator. Evaluates and scores each stage of the assessment
    pipeline, collecting evidence of system reliability.
    """

    def __init__(self) -> None:
        self.metrics: list[QCMetrics] = []

    def validate_input_data(
        self,
        patient_data: dict[str, Any],
        validation_flags: list[dict[str, str]] | None = None,
    ) -> tuple[float, list[ValidationFlag]]:
        """
        Score input validation quality.

        Returns:
          - input_validation_score: 0–1
          - validation_flags: list of raised flags
        """
        flags: list[ValidationFlag] = []

        # Required fields — penalise score only if these are missing.
        required_pairs = [
            ("age", " Age (yrs)"),
            ("bmi", "BMI"),
            ("cycle_ri", "Cycle(R/I)"),
        ]
        # Optional labs / ultrasound — informational only, not a penalty.
        optional_pairs = [
            ("lh", "LH(mIU/mL)"),
            ("fsh", "FSH(mIU/mL)"),
            ("follicle_l", "Follicle No. (L)"),
            ("follicle_r", "Follicle No. (R)"),
        ]

        missing_required = [
            api_key
            for api_key, legacy_key in required_pairs
            if _get(patient_data, api_key, legacy_key) is None
        ]
        missing_optional = [
            api_key
            for api_key, legacy_key in optional_pairs
            if _get(patient_data, api_key, legacy_key) is None
        ]

        if missing_required:
            flags.append(ValidationFlag(
                "required_fields",
                QCStatus.WARNING,
                f"Missing required fields: {', '.join(missing_required)}",
                "warning",
            ))
        if missing_optional:
            flags.append(ValidationFlag(
                "optional_fields",
                QCStatus.INFO,
                f"Optional fields not provided "
                f"(population baseline used): {', '.join(missing_optional)}",
                "info",
            ))

        present_required = len(required_pairs) - len(missing_required)
        input_validation_score = present_required / len(required_pairs)

        # Check for outliers/implausible values — accept both key formats
        plausibility_checks = [
            (("age", " Age (yrs)"), (12, 60)),
            (("bmi", "BMI"), (12, 65)),
            (("lh", "LH(mIU/mL)"), (0.1, 80)),
            (("fsh", "FSH(mIU/mL)"), (0.1, 80)),
        ]
        plausibility_ranges: dict = {}  # kept for backward compat reference only

        outlier_count = 0
        for (api_key, legacy_key), (min_val, max_val) in plausibility_checks:
            raw = _get(patient_data, api_key, legacy_key)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if not (min_val <= val <= max_val):
                outlier_count += 1
                flags.append(ValidationFlag(
                    f"plausibility_{api_key}",
                    QCStatus.WARNING,
                    f"{api_key} = {val} is outside typical range [{min_val}, {max_val}]",
                    "warning",
                ))

        # Apply outlier penalty
        if outlier_count > 0:
            input_validation_score *= (1 - 0.1 * min(outlier_count, 3))

        return max(0.0, input_validation_score), flags

    def validate_model_output(
        self,
        risk_score: float,
        risk_label: str,
        confidence: float,
        model_auroc: float = 0.9528,
    ) -> tuple[float, list[ValidationFlag], str]:
        """
        Score model output sanity and consistency.

        Returns:
          - prediction_plausibility_score: 0–1
          - validation_flags: list of checks
          - confidence_label: "high" | "medium" | "low"
        """
        flags: list[ValidationFlag] = []
        score = 1.0

        # Check risk_score is in [0, 1]
        if not (0.0 <= risk_score <= 1.0):
            flags.append(ValidationFlag(
                "risk_score_range",
                QCStatus.FAIL,
                f"Risk score {risk_score} outside [0, 1]",
                "error",
            ))
            score -= 0.3

        # Check risk_label matches risk_score
        risk_thresholds = {"Low": (0.0, 0.35), "Medium": (0.35, 0.70), "High": (0.70, 1.0)}
        label_min, label_max = risk_thresholds.get(risk_label, (0, 1))
        if not (label_min <= risk_score <= label_max):
            flags.append(ValidationFlag(
                "risk_label_consistency",
                QCStatus.WARNING,
                f"Risk label '{risk_label}' misaligned with score {risk_score:.2%}",
                "warning",
            ))
            score -= 0.15

        # Check confidence is plausible
        if not (0.5 <= confidence <= 1.0):
            flags.append(ValidationFlag(
                "confidence_range",
                QCStatus.WARNING,
                f"Confidence {confidence} should be in [0.5, 1.0]",
                "warning",
            ))
            score -= 0.1

        # Confidence label based on distance from 50%
        if confidence >= 0.85:
            confidence_label = "high"
        elif confidence >= 0.65:
            confidence_label = "medium"
        else:
            confidence_label = "low"

        # Add passing checks
        if 0.0 <= risk_score <= 1.0:
            flags.append(ValidationFlag(
                "risk_score_valid",
                QCStatus.PASS,
                f"Risk score {risk_score:.1%} is valid",
                "info",
            ))

        return max(0.0, score), flags, confidence_label

    def validate_rag_evidence(
        self,
        retrieved_papers: int = 0,
        evidence_chunks: int = 0,
        citations_count: int = 0,
    ) -> tuple[float, list[ValidationFlag]]:
        """
        Score quality of RAG/evidence retrieval.

        Returns:
          - rag_evidence_score: 0–1
          - validation_flags
        """
        flags: list[ValidationFlag] = []
        score = 0.0

        if retrieved_papers > 0:
            score += 0.3
            flags.append(ValidationFlag(
                "papers_retrieved",
                QCStatus.PASS,
                f"Retrieved {retrieved_papers} relevant papers",
                "info",
            ))
        else:
            flags.append(ValidationFlag(
                "papers_retrieved",
                QCStatus.WARNING,
                "No papers retrieved from RAG system",
                "warning",
            ))

        if evidence_chunks > 0:
            score += 0.35
            flags.append(ValidationFlag(
                "evidence_chunks",
                QCStatus.PASS,
                f"Generated {evidence_chunks} evidence summaries",
                "info",
            ))

        if citations_count > 0:
            score += 0.35
            flags.append(ValidationFlag(
                "citations",
                QCStatus.PASS,
                f"{citations_count} citations included",
                "info",
            ))

        return min(1.0, score), flags

    def compute_overall_quality_score(
        self,
        input_score: float,
        model_score: float,
        plausibility_score: float,
        rag_score: float,
    ) -> float:
        """
        Compute weighted average of all QC components.

        Weights:
          - Input validation: 20%
          - Model output: 50%
          - Output plausibility: 20%
          - RAG evidence: 10%
        """
        weights = {
            "input": (input_score, 0.20),
            "model": (model_score, 0.50),
            "plausibility": (plausibility_score, 0.20),
            "rag": (rag_score, 0.10),
        }

        weighted_sum = sum(score * weight for score, weight in weights.values())
        return min(1.0, max(0.0, weighted_sum))

    def create_metrics_report(
        self,
        patient_data: dict[str, Any],
        prediction_result: dict[str, Any],
        rag_results: dict[str, Any] | None = None,
        validation_flags_from_agent: list[dict[str, str]] | None = None,
    ) -> QCMetrics:
        """
        Comprehensive QC report for a single assessment.

        Returns:
          QCMetrics object with all scores and flags.
        """
        timestamp = datetime.utcnow().isoformat()

        # Stage 1: Input validation
        input_score, input_flags = self.validate_input_data(
            patient_data,
            validation_flags_from_agent,
        )

        # Stage 2: Model output validation
        pred_score, pred_flags, conf_label = self.validate_model_output(
            risk_score=prediction_result.get("risk_score", 0.5),
            risk_label=prediction_result.get("risk_label", "Medium"),
            confidence=prediction_result.get("confidence", 0.5),
            model_auroc=prediction_result.get("model_auroc", 0.9528),
        )

        # Stage 3: RAG/Evidence validation
        rag_results = rag_results or {}
        # Agent 2 emits two paper lists (`retrieved_papers` from Chroma and
        # `pubmed_papers` from PubMed) plus `clinical_summary` text. Fall back
        # to the legacy `papers` / `evidence_chunks` / `citation_count` keys
        # if some other caller passes them.
        n_chroma = len(rag_results.get("retrieved_papers", []))
        n_pubmed = len(rag_results.get("pubmed_papers", []))
        n_papers = (
            n_chroma + n_pubmed
            if (n_chroma + n_pubmed) > 0
            else len(rag_results.get("papers", []))
        )
        # Did the LLM produce a synthesis? Treat any non-empty summary
        # (or explicit evidence_chunks list) as a single high-quality chunk.
        clinical_summary = (rag_results.get("clinical_summary") or "").strip()
        n_chunks = (
            1 if clinical_summary
            else len(rag_results.get("evidence_chunks", []))
        )
        # Citation count: prefer explicit, else derive from PubMed papers.
        citations = (
            rag_results.get("citation_count")
            or n_pubmed
            or 0
        )
        rag_score, rag_flags = self.validate_rag_evidence(
            retrieved_papers=n_papers,
            evidence_chunks=n_chunks,
            citations_count=citations,
        )

        # Compute overall (plausibility uses its own independent score)
        overall_score = self.compute_overall_quality_score(
            input_score,
            pred_score,
            pred_score * input_score,  # joint measure: model confidence × input quality
            rag_score,
        )

        # Collect all flags
        all_flags = input_flags + pred_flags + rag_flags

        key_metrics = {
            "patient_age": _get(patient_data, "age", " Age (yrs)"),
            "patient_bmi": _get(patient_data, "bmi", "BMI"),
            "risk_score_percentage": f"{prediction_result.get('risk_score', 0) * 100:.1f}%",
            "model_auroc": prediction_result.get("model_auroc", 0.9528),
            "papers_used": n_papers,
        }

        metrics = QCMetrics(
            timestamp=timestamp,
            input_validation_score=input_score,
            model_confidence=prediction_result.get("confidence", 0.5),
            prediction_plausibility_score=pred_score,
            rag_evidence_score=rag_score,
            overall_quality_score=overall_score,
            validation_flags=all_flags,
            model_auroc=prediction_result.get("model_auroc", 0.9528),
            prediction_confidence_label=conf_label,
            key_metrics=key_metrics,
        )

        self.metrics.append(metrics)
        return metrics

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get aggregate performance metrics across all assessments.

        Useful for dashboards showing system reliability.
        """
        if not self.metrics:
            return {
                "total_assessments": 0,
                "avg_quality_score": 0.0,
                "avg_model_confidence": 0.0,
                "avg_rag_score": 0.0,
                "error_rate": 0.0,
                "warning_rate": 0.0,
            }

        total = len(self.metrics)
        avg_quality = sum(m.overall_quality_score for m in self.metrics) / total
        avg_confidence = sum(m.model_confidence for m in self.metrics) / total
        avg_rag = sum(m.rag_evidence_score for m in self.metrics) / total

        # Count issues
        total_flags = sum(len(m.validation_flags) for m in self.metrics)
        error_flags = sum(
            sum(1 for f in m.validation_flags if f.severity == "error")
            for m in self.metrics
        )
        warning_flags = sum(
            sum(1 for f in m.validation_flags if f.severity == "warning")
            for m in self.metrics
        )

        return {
            "total_assessments": total,
            "avg_quality_score": round(avg_quality, 3),
            "avg_model_confidence": round(avg_confidence, 3),
            "avg_rag_score": round(avg_rag, 3),
            "error_count": error_flags,
            "warning_count": warning_flags,
            "avg_flags_per_assessment": round(total_flags / total, 2),
        }
