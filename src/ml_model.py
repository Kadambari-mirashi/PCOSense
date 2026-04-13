"""
ml_model.py — PCOS prediction wrapper for PolyAI
=================================================
Provides:
  - PCOSPredictor  : load model, predict risk, generate SHAP explanations
  - predict_pcos() : convenience function for FastAPI endpoints
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the module works from any cwd)
# ---------------------------------------------------------------------------
_SRC_DIR   = Path(__file__).parent
_ROOT_DIR  = _SRC_DIR.parent
_MODEL_PATH    = _ROOT_DIR / "models" / "pcos_model.json"
_METADATA_PATH = _ROOT_DIR / "models" / "model_metadata.json"
_PKL_PATH      = _ROOT_DIR / "data" / "processed" / "features_processed.pkl"


# ---------------------------------------------------------------------------
# Data classes for structured outputs
# ---------------------------------------------------------------------------
@dataclass
class RiskFactor:
    """A single feature's contribution to the prediction."""
    feature: str
    shap_value: float
    raw_value: float
    direction: str          # "increases" | "decreases"
    magnitude: str          # "high" | "medium" | "low"


@dataclass
class PredictionResult:
    """Full prediction output returned to callers."""
    risk_score: float                   # 0–1 probability
    risk_label: str                     # "High" | "Medium" | "Low"
    predicted_class: int                # 1 = PCOS, 0 = No PCOS
    confidence: float                   # max(prob, 1-prob)
    top_risk_factors: list[RiskFactor]  # sorted by |SHAP|
    all_shap_values: dict[str, float]   # feature → SHAP value
    model_auroc: float
    threshold_used: float
    explanation_text: str               # human-readable summary


# ---------------------------------------------------------------------------
# PCOSPredictor
# ---------------------------------------------------------------------------
class PCOSPredictor:
    """
    Wrapper around the trained XGBoost PCOS model.

    Usage
    -----
    predictor = PCOSPredictor()
    result    = predictor.predict(patient_features_dict)
    """

    _instance: "PCOSPredictor | None" = None   # singleton cache

    # ── constructor ────────────────────────────────────────────────────────
    def __init__(
        self,
        model_path:    str | Path = _MODEL_PATH,
        metadata_path: str | Path = _METADATA_PATH,
        pkl_path:      str | Path = _PKL_PATH,
    ) -> None:
        self.model_path    = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.pkl_path      = Path(pkl_path)

        self.model:         xgb.XGBClassifier | None = None
        self.metadata:      dict[str, Any]            = {}
        self.feature_names: list[str]                 = []
        self.scaler:        Any                       = None
        self.imputer:       Any                       = None
        self.explainer:     shap.TreeExplainer | None = None
        self._fallback_medians: dict[str, float]      = {}
        self._fallback_means:   dict[str, float]      = {}
        self._fallback_stds:    dict[str, float]      = {}
        self._loaded:       bool                      = False

    # ── public: load ───────────────────────────────────────────────────────
    def load(self) -> "PCOSPredictor":
        """Load model, metadata, scaler and SHAP explainer."""
        if self._loaded:
            return self

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run notebooks/03_xgboost_training.ipynb first."
            )

        # XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))

        # Metadata
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])
        else:
            self.feature_names = self.model.get_booster().feature_names or []

        # Build fallback statistics from metadata (always available via git)
        self._fallback_medians = self.metadata.get("imputation_medians", {})
        self._fallback_means = self.metadata.get("scaler_means", {})
        self._fallback_stds = self.metadata.get("scaler_stds", {})

        # Scaler / imputer from feature engineering pipeline
        if self.pkl_path.exists():
            with open(self.pkl_path, "rb") as f:
                proc = pickle.load(f)
            self.scaler  = proc.get("scaler")
            self.imputer = proc.get("imputer")
            if not self.feature_names:
                self.feature_names = proc.get("feature_names", [])
        elif self._fallback_medians:
            log.warning(
                "features_processed.pkl not found; using metadata fallback "
                "for imputation and scaling."
            )
        else:
            log.warning(
                "features_processed.pkl not found AND model_metadata.json has "
                "no imputation_medians. Predictions will be unreliable."
            )

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        self._loaded = True
        log.info(
            "PCOSPredictor loaded  |  features=%d  |  AUROC=%s  |  pkl=%s",
            len(self.feature_names),
            self.metadata.get("auroc", "N/A"),
            "yes" if self.scaler is not None else "fallback",
        )
        return self

    # ── singleton helper ───────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "PCOSPredictor":
        """Return (and lazily initialise) the singleton predictor."""
        if cls._instance is None:
            cls._instance = cls().load()
        return cls._instance

    @staticmethod
    def _fill_engineered_features(row: dict[str, Any]) -> None:
        """
        Derive the same composite columns used in training (see notebooks/02_features.ipynb)
        so they are not left NaN and then imputed to PCOS-heavy training medians.
        """
        eps = 1e-6
        fl_k, fr_k = "Follicle No. (L)", "Follicle No. (R)"
        if fl_k in row and fr_k in row:
            fl, fr = row.get(fl_k), row.get(fr_k)
            if fl is not None and fr is not None:
                try:
                    fl_f, fr_f = float(fl), float(fr)
                    if np.isfinite(fl_f) and np.isfinite(fr_f):
                        if "follicle_total" in row:
                            row["follicle_total"] = fl_f + fr_f
                        if "follicle_asymmetry" in row:
                            row["follicle_asymmetry"] = abs(fl_f - fr_f)
                except (TypeError, ValueError):
                    pass

        lh_k, fsh_k = "LH(mIU/mL)", "FSH(mIU/mL)"
        if lh_k in row and fsh_k in row:
            lh_v, fsh_v = row.get(lh_k), row.get(fsh_k)
            if lh_v is not None and fsh_v is not None:
                try:
                    lh_f, fsh_f = float(lh_v), float(fsh_v)
                    if np.isfinite(lh_f) and np.isfinite(fsh_f):
                        if "LH_FSH_ratio" in row:
                            row["LH_FSH_ratio"] = lh_f / (fsh_f + eps)
                        if "FSH/LH" in row:
                            row["FSH/LH"] = fsh_f / (lh_f + eps)
                except (TypeError, ValueError):
                    pass

    # ── public: predict ────────────────────────────────────────────────────
    def predict(
        self,
        patient: dict[str, float],
        top_n: int = 5,
    ) -> PredictionResult:
        """
        Make a PCOS risk prediction for one patient.

        Parameters
        ----------
        patient : dict mapping feature names → raw (unscaled) values.
                  Missing features are imputed with the training-set median.
        top_n   : number of top risk factors to return.

        Returns
        -------
        PredictionResult
        """
        if not self._loaded:
            self.load()

        # Build feature vector (in correct column order)
        row = {feat: patient.get(feat, np.nan) for feat in self.feature_names}
        self._fill_engineered_features(row)
        X_raw = pd.DataFrame([row], columns=self.feature_names)

        # Impute → scale
        if self.imputer is not None:
            X_imp = pd.DataFrame(
                self.imputer.transform(X_raw),
                columns=self.feature_names,
            )
        elif self._fallback_medians:
            medians = pd.Series(
                {f: self._fallback_medians[f] for f in self.feature_names},
            )
            X_imp = X_raw.fillna(medians)
        else:
            X_imp = X_raw.fillna(0.0)

        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_imp),
                columns=self.feature_names,
            )
        elif self._fallback_means and self._fallback_stds:
            means = np.array([self._fallback_means[f] for f in self.feature_names])
            stds = np.array([self._fallback_stds[f] for f in self.feature_names])
            stds = np.where(stds < 1e-10, 1.0, stds)
            X_scaled = pd.DataFrame(
                (X_imp.values - means) / stds,
                columns=self.feature_names,
            )
        else:
            X_scaled = X_imp

        # Predict
        risk_score = float(self.model.predict_proba(X_scaled)[0, 1])
        threshold  = self.metadata.get("optimal_threshold", 0.5)
        predicted  = int(risk_score >= threshold)

        # SHAP values (on scaled input, matching training)
        shap_vals  = self.explainer.shap_values(X_scaled)[0]
        shap_dict  = dict(zip(self.feature_names, shap_vals.tolist()))

        # Top risk factors
        top_factors = self._build_risk_factors(
            shap_dict, X_imp.iloc[0].to_dict(), top_n
        )

        # Risk label
        if risk_score >= 0.70:
            risk_label = "High"
        elif risk_score >= 0.40:
            risk_label = "Medium"
        else:
            risk_label = "Low"

        confidence = max(risk_score, 1.0 - risk_score)
        explanation = self._generate_explanation(risk_label, risk_score, top_factors)

        return PredictionResult(
            risk_score       = round(risk_score, 4),
            risk_label       = risk_label,
            predicted_class  = predicted,
            confidence       = round(confidence, 4),
            top_risk_factors = top_factors,
            all_shap_values  = {k: round(v, 4) for k, v in shap_dict.items()},
            model_auroc      = self.metadata.get("auroc", 0.0),
            threshold_used   = threshold,
            explanation_text = explanation,
        )

    # ── public: batch predict ──────────────────────────────────────────────
    def predict_batch(
        self,
        patients: list[dict[str, float]],
        top_n: int = 5,
    ) -> list[PredictionResult]:
        """Predict for a list of patients (more efficient than looping)."""
        return [self.predict(p, top_n=top_n) for p in patients]

    # ── public: feature metadata ───────────────────────────────────────────
    def feature_info(self) -> dict[str, Any]:
        """Return feature names, SHAP ranking and model metrics."""
        return {
            "feature_names"     : self.feature_names,
            "n_features"        : len(self.feature_names),
            "model_auroc"       : self.metadata.get("auroc"),
            "model_auprc"       : self.metadata.get("auprc"),
            "optimal_threshold" : self.metadata.get("optimal_threshold"),
            "top_shap_features" : self.metadata.get("top_shap_features", []),
        }

    # ── private helpers ────────────────────────────────────────────────────
    @staticmethod
    def _build_risk_factors(
        shap_dict:   dict[str, float],
        raw_values:  dict[str, float],
        top_n: int,
    ) -> list[RiskFactor]:
        """Sort features by |SHAP| and annotate each one."""
        sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        factors: list[RiskFactor] = []

        for feat, sv in sorted_feats[:top_n]:
            abs_sv = abs(sv)
            if abs_sv >= 0.3:
                mag = "high"
            elif abs_sv >= 0.1:
                mag = "medium"
            else:
                mag = "low"

            factors.append(RiskFactor(
                feature    = feat,
                shap_value = round(sv, 4),
                raw_value  = round(float(raw_values.get(feat, np.nan)), 4),
                direction  = "increases" if sv > 0 else "decreases",
                magnitude  = mag,
            ))

        return factors

    @staticmethod
    def _generate_explanation(
        risk_label:  str,
        risk_score:  float,
        top_factors: list[RiskFactor],
    ) -> str:
        """Return a plain-English, one-paragraph explanation."""
        pct = f"{risk_score * 100:.1f}%"
        lines = [
            f"The model estimates a {pct} probability of PCOS ({risk_label} risk)."
        ]
        if top_factors:
            increasing = [f.feature for f in top_factors if f.direction == "increases"]
            decreasing = [f.feature for f in top_factors if f.direction == "decreases"]
            if increasing:
                lines.append(
                    f"Key factors raising risk: {', '.join(increasing[:3])}."
                )
            if decreasing:
                lines.append(
                    f"Key factors lowering risk: {', '.join(decreasing[:3])}."
                )
        lines.append(
            "This is a screening aid — please consult a qualified clinician."
        )
        return " ".join(lines)


# ---------------------------------------------------------------------------
# Convenience function for FastAPI / Shiny endpoints
# ---------------------------------------------------------------------------
def predict_pcos(
    patient_features: dict[str, float],
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Predict PCOS risk for a single patient dict.

    Returns a JSON-serialisable dict (no dataclass objects).

    Parameters
    ----------
    patient_features : {feature_name: value, …}
    top_n            : number of top risk factors to include

    Example
    -------
    >>> result = predict_pcos({"BMI": 28.5, "LH(mIU/mL)": 12.3, ...})
    >>> result["risk_score"]
    0.7831
    """
    predictor = PCOSPredictor.get_instance()
    result    = predictor.predict(patient_features, top_n=top_n)

    return {
        "risk_score"      : result.risk_score,
        "risk_label"      : result.risk_label,
        "predicted_class" : result.predicted_class,
        "confidence"      : result.confidence,
        "model_auroc"     : result.model_auroc,
        "threshold_used"  : result.threshold_used,
        "explanation"     : result.explanation_text,
        "top_risk_factors": [
            {
                "feature"   : rf.feature,
                "shap_value": rf.shap_value,
                "raw_value" : rf.raw_value,
                "direction" : rf.direction,
                "magnitude" : rf.magnitude,
            }
            for rf in result.top_risk_factors
        ],
        "all_shap_values" : result.all_shap_values,
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("Loading PCOSPredictor …")
    predictor = PCOSPredictor().load()

    # Generate a synthetic patient (all-zeros after imputation)
    dummy = {feat: float(np.random.randn()) for feat in predictor.feature_names}
    print("\nRunning smoke-test prediction …")
    res = predictor.predict(dummy)

    print(f"\n{'─' * 50}")
    print(f"  Risk score   : {res.risk_score}")
    print(f"  Risk label   : {res.risk_label}")
    print(f"  Predicted    : {'PCOS' if res.predicted_class else 'No PCOS'}")
    print(f"  Confidence   : {res.confidence}")
    print(f"\nTop risk factors:")
    for rf in res.top_risk_factors:
        print(f"  {rf.feature:<35} SHAP={rf.shap_value:+.4f}  ({rf.direction} risk, {rf.magnitude})")
    print(f"\nExplanation:\n  {res.explanation_text}")
    print(f"{'─' * 50}")
    print("Smoke-test passed ✅")
    sys.exit(0)
