"""
PCOSense Shiny frontend - patient form and multi-agent assessment results.
Requires the FastAPI server (see README). Teal-themed UI.

Long API calls use shiny.extended_task so the Shiny server event loop is not blocked.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from shiny import App, reactive, render, ui
from shiny.reactive import extended_task

_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / ".env")

_default_port = os.getenv("PORT", "8000")
API_BASE = os.getenv("PCOSENSE_API_URL", f"http://127.0.0.1:{_default_port}").rstrip("/")
ASSESS_URL = f"{API_BASE}/api/v1/assess"
HEALTH_URL = f"{API_BASE}/api/v1/health"

UI_BUILD_STAMP = "2026-04-12 • header-layout-v15"

TEAL_CSS = """
:root {
  --pcos-teal: #0d9488;
  --pcos-teal-dark: #0f766e;
  --pcos-teal-darker: #115e59;
  --pcos-teal-muted: #5eead4;
  --pcos-teal-bg: #f0fdfa;
}
body { background: var(--pcos-teal-bg) !important; }
.pcos-header-main {
  background: linear-gradient(135deg, var(--pcos-teal-darker) 0%, var(--pcos-teal) 55%, var(--pcos-teal-muted) 160%);
  color: #fff;
  padding: 2rem 1.75rem 1.6rem;
  border-radius: 0.65rem;
  margin-bottom: 1.35rem;
  box-shadow: 0 4px 14px rgba(17, 94, 89, 0.22);
  text-align: center;
  max-width: 920px;
  margin-left: auto;
  margin-right: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.pcos-header-top {
  width: 100%;
  max-width: 38rem;
  margin: 0 0 1.15rem 0;
  padding: 0 0.5rem 1.1rem 0.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.22);
}
.pcos-header-main h1.pcos-header-title {
  margin: 0 0 0.4rem 0;
  font-weight: 700;
  letter-spacing: -0.03em;
  font-size: clamp(2.1rem, 5.2vw, 2.95rem);
  line-height: 1.1;
  color: #fff;
}
.pcos-header-main .pcos-header-line2 {
  margin: 0 0 0.35rem 0;
  font-size: clamp(1.15rem, 3vw, 1.45rem);
  font-weight: 600;
  line-height: 1.32;
  opacity: 0.96;
}
.pcos-header-main .pcos-header-line3 {
  margin: 0;
  font-size: 1rem;
  font-weight: 500;
  line-height: 1.38;
  opacity: 0.93;
}
.pcos-header-mid {
  width: 100%;
  max-width: 42rem;
  margin: 0 0 1rem 0;
  padding: 0.75rem 1.1rem;
  border-radius: 0.5rem;
  background: rgba(255, 255, 255, 0.11);
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-sizing: border-box;
}
.pcos-header-main .pcos-header-flow {
  margin: 0;
  font-size: 0.8rem;
  font-weight: 400;
  line-height: 1.55;
  opacity: 0.92;
  letter-spacing: 0.01em;
}
.pcos-header-tip {
  width: 100%;
  max-width: 42rem;
  margin: 0 0 0.25rem 0;
  padding: 0 0.35rem;
  box-sizing: border-box;
}
.pcos-header-main .pcos-header-tip .pcos-tagline {
  margin: 0;
  opacity: 0.9;
  font-size: 0.8rem;
  line-height: 1.58;
  letter-spacing: 0.015em;
}
.pcos-header-stamp {
  margin: 0.85rem 0 0 0;
  opacity: 0.55;
  font-size: 0.7rem;
  letter-spacing: 0.04em;
}
.pcos-results-placeholder {
  border: 2px dashed #99f6e4;
  border-radius: 0.65rem;
  padding: 2rem 1.35rem;
  text-align: center;
  background: #fff;
  color: #475569;
  max-width: 920px;
  margin: 0 auto 1rem auto;
  line-height: 1.55;
  font-size: 0.95rem;
  box-shadow: 0 1px 4px rgba(15, 118, 110, 0.06);
}
.pcos-main-inner {
  max-width: 920px;
  margin: 0 auto;
  width: 100%;
  padding: 0 0.5rem 2rem;
}
.pcos-card {
  border: 1px solid #99f6e4;
  border-radius: 0.5rem;
  padding: 1.1rem 1.25rem;
  margin-bottom: 1rem;
  background: #fff;
  box-shadow: 0 1px 3px rgba(15, 118, 110, 0.08);
}
.pcos-card h3 {
  color: var(--pcos-teal-dark);
  font-size: 1.05rem;
  margin: 0 0 0.65rem 0;
  font-weight: 600;
}
.pcos-muted { color: #5c6c6b; font-size: 0.9rem; }
.pcos-help {
  color: #5a6b6a;
  font-size: 0.78rem;
  line-height: 1.45;
  margin: 0.15rem 0 0.6rem 0;
}
.pcos-help-tight { margin-top: -0.15rem; }
.btn-primary, .btn-primary:focus {
  background-color: var(--pcos-teal) !important;
  border-color: var(--pcos-teal-dark) !important;
}
.btn-primary:hover {
  background-color: var(--pcos-teal-dark) !important;
  border-color: var(--pcos-teal-darker) !important;
}
.form-label { font-weight: 500; font-size: 0.88rem; }
.pcos-sidebar .shiny-input-container { margin-bottom: 0.4rem; }
.pcos-sidebar h4 {
  color: var(--pcos-teal-dark);
  font-size: 0.98rem;
  margin: 1rem 0 0.35rem 0;
  font-weight: 600;
  border-bottom: 1px solid #99f6e4;
  padding-bottom: 0.25rem;
}
.pcos-sidebar h4:first-of-type { margin-top: 0.25rem; }
.pcos-h4-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin: 1rem 0 0.4rem 0;
  padding-bottom: 0.25rem;
  border-bottom: 1px solid #99f6e4;
  font-size: 0.98rem;
  font-weight: 600;
  color: var(--pcos-teal-dark);
}
.pcos-h4-row:first-of-type { margin-top: 0.25rem; }
.pcos-h4-text { flex: 1; }
.pcos-info-i {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.2rem;
  height: 1.2rem;
  padding: 0 0.25rem;
  border-radius: 50%;
  background: var(--pcos-teal);
  color: #fff !important;
  font-size: 0.62rem;
  font-weight: 700;
  font-style: italic;
  font-family: Georgia, serif;
  cursor: help;
  line-height: 1;
  flex-shrink: 0;
}
.pcos-intro-bullets { padding-left: 1.1rem; margin: 0.35rem 0 0.5rem 0; font-size: 0.8rem; color: #4a5a59; line-height: 1.45; }
.pcos-unit-sm { font-size: 0.72em; font-weight: 400; color: #64748b; }
.pcos-hormone-block .shiny-input-container > label { font-size: 0.88rem; }
.pcos-hormone-block .pcos-unit-sm { font-size: 0.68rem; }
.pcos-risk-pill {
  display: inline-block;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.9rem;
}
.pcos-risk-high { background: #fee2e2; color: #991b1b; }
.pcos-risk-medium { background: #fef3c7; color: #92400e; }
.pcos-risk-low { background: #d1fae5; color: #065f46; }
.pcos-bar {
  height: 10px;
  border-radius: 999px;
  background: #ccfbf1;
  overflow: hidden;
  margin-top: 0.35rem;
}
.pcos-bar > span {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--pcos-teal-dark), var(--pcos-teal-muted));
  border-radius: 999px;
}
.pcos-api-banner { max-width: 920px; margin: 0 auto 0.75rem; }

/* Results dashboard */
.pcos-results-root {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
}
.pcos-results-hero {
  border-radius: 0.65rem;
  padding: 1.45rem 1.5rem;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 1.35rem;
  align-items: center;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
}
.pcos-results-hero--low {
  background: linear-gradient(135deg, #14532d 0%, #16a34a 55%, #4ade80 130%);
  color: #fff;
}
.pcos-results-hero--medium {
  background: linear-gradient(135deg, #a16207 0%, #eab308 50%, #fde047 130%);
  color: #1c1917;
}
.pcos-results-hero--high {
  background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 45%, #dc2626 100%);
  color: #fff;
}
.pcos-results-score-badge {
  width: 5.25rem;
  height: 5.25rem;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  line-height: 1.1;
  background: rgba(255, 255, 255, 0.18);
  border: 2px solid rgba(255, 255, 255, 0.35);
  flex-shrink: 0;
}
.pcos-results-hero--medium .pcos-results-score-badge {
  background: rgba(255, 255, 255, 0.45);
  border-color: rgba(120, 53, 15, 0.35);
  color: #422006;
}
.pcos-results-badge-pct { font-size: 1.15rem; }
.pcos-results-badge-sub { font-size: 0.62rem; font-weight: 600; opacity: 0.92; text-transform: uppercase; letter-spacing: 0.03em; }
.pcos-results-hero-title {
  font-size: 1.22rem;
  font-weight: 700;
  margin: 0 0 0.55rem 0;
  line-height: 1.28;
}
.pcos-results-hero--medium .pcos-results-hero-title { color: #422006; }
.pcos-results-hero-text {
  margin-top: 0.1rem;
}
.pcos-results-hero--medium .pcos-results-hero-lead { color: #44403c; }
.pcos-results-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
}
@media (max-width: 540px) {
  .pcos-results-metrics { grid-template-columns: 1fr; }
  .pcos-results-hero { grid-template-columns: 1fr; justify-items: center; text-align: center; }
}
.pcos-results-metric {
  border-radius: 0.55rem;
  padding: 0.85rem 1rem;
  border: 1px solid #e2e8f0;
  background: #fff;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
}
.pcos-results-metric-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #64748b;
  margin-bottom: 0.25rem;
  font-weight: 600;
}
.pcos-results-metric-value {
  font-size: 1.65rem;
  font-weight: 700;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.pcos-results-metric--prob-low { color: #15803d; }
.pcos-results-metric--prob-med { color: #a16207; }
.pcos-results-metric--prob-high { color: #b91c1c; }
.pcos-results-metric--conf { color: #15803d; }
.pcos-results-panel-dark {
  background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
  color: #e2e8f0;
  border-radius: 0.65rem;
  padding: 1rem 1.15rem 1.15rem;
  border: 1px solid #334155;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.2);
}
.pcos-results-hero-lead {
  margin: 0 0 0.65rem 0;
  opacity: 0.96;
  font-size: 0.9rem;
  line-height: 1.58;
}
.pcos-results-hero-text .pcos-results-hero-lead:last-child { margin-bottom: 0; }
.pcos-results-evidence-stack {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.pcos-results-data-quality-line {
  margin: 0;
  font-size: 0.88rem;
  line-height: 1.45;
}
.pcos-results-section-label {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
  margin: 0 0 0.65rem 0;
  font-weight: 700;
}
.pcos-results-htable { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.pcos-results-htable th {
  text-align: left;
  padding: 0.35rem 0.4rem 0.5rem;
  border-bottom: 1px solid #475569;
  color: #94a3b8;
  font-weight: 600;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.pcos-results-htable td {
  padding: 0.55rem 0.4rem;
  border-top: 1px solid #334155;
  vertical-align: middle;
}
.pcos-results-feat { color: #f1f5f9; font-weight: 500; }
.pcos-results-effect {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  white-space: nowrap;
}
.pcos-results-effect--up { color: #fca5a5; }
.pcos-results-effect--down { color: #86efac; }
.pcos-results-strength-cell { width: 28%; min-width: 5rem; }
.pcos-results-strength-bar {
  height: 8px;
  border-radius: 999px;
  background: #334155;
  overflow: hidden;
}
.pcos-results-strength-bar > span {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #64748b, #f1f5f9);
  min-width: 4px;
}
.pcos-results-strength-bar--up > span {
  background: linear-gradient(90deg, #991b1b, #f87171);
}
.pcos-results-strength-bar--down > span {
  background: linear-gradient(90deg, #14532d, #4ade80);
}
.pcos-results-pill {
  display: inline-block;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.pcos-results-pill--up {
  background: rgba(248, 113, 113, 0.2);
  color: #fecaca;
  border: 1px solid rgba(248, 113, 113, 0.45);
}
.pcos-results-pill--down {
  background: rgba(74, 222, 128, 0.15);
  color: #bbf7d0;
  border: 1px solid rgba(74, 222, 128, 0.35);
}
.pcos-results-muted { color: #94a3b8; }
.pcos-results-papers h4, .pcos-results-papers .pcos-results-section-label { color: #cbd5e1; }
.pcos-results-papers ul { margin: 0; padding-left: 0; list-style: none; }
.pcos-results-papers li {
  padding: 0.55rem 0;
  border-bottom: 1px solid #334155;
  font-size: 0.84rem;
}
.pcos-results-papers li:last-child { border-bottom: none; }
.pcos-results-card-light {
  border: 1px solid #99f6e4;
  border-radius: 0.55rem;
  padding: 1rem 1.15rem;
  background: #fff;
  box-shadow: 0 2px 8px rgba(15, 118, 110, 0.08);
}
.pcos-results-card-light h3 {
  color: var(--pcos-teal-dark);
  font-size: 1rem;
  margin: 0 0 0.5rem 0;
  font-weight: 600;
}
.pcos-results-flags-wrap {
  font-size: 0.84rem;
  line-height: 1.45;
}
.pcos-results-flagline {
  display: flex;
  gap: 0.45rem;
  align-items: flex-start;
  margin-bottom: 0.45rem;
  color: #fde68a;
}
.pcos-results-bullet {
  flex-shrink: 0;
  margin-top: 0.15rem;
  font-size: 0.75rem;
}
.pcos-results-bullet--error { color: #fca5a5; }
.pcos-results-bullet--warning { color: #fb923c; }
.pcos-results-footnote {
  font-size: 0.78rem;
  color: #64748b;
  margin: 0.25rem 0 0 0;
}

/* Cycle: two large cards, centered as a pair */
.pcos-cycle-outer {
  width: 100%;
  display: flex;
  justify-content: center;
  margin: 0.25rem 0 0.35rem 0;
}
.pcos-cycle-wrap {
  width: 100%;
  max-width: 400px;
}
.pcos-cycle-wrap .shiny-input-radiogroup > .shiny-options-group {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.55rem;
}
.pcos-cycle-wrap .radio { margin: 0 !important; }
.pcos-cycle-wrap .radio label {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  border-radius: 0.65rem;
  padding: 0.8rem 0.65rem;
  cursor: pointer;
  min-height: 5.2rem;
  transition: border-color 0.18s, background 0.18s, box-shadow 0.18s;
}
.pcos-cycle-wrap .radio label:has(input:focus-visible) {
  outline: 2px solid var(--pcos-teal);
  outline-offset: 2px;
}
.pcos-cycle-wrap .radio:nth-child(1) label {
  border: 2px solid #86efac;
  background: #f0fdf4;
}
.pcos-cycle-wrap .radio:nth-child(2) label {
  border: 2px solid #fda4af;
  background: #fff1f2;
}
.pcos-cycle-wrap .radio:nth-child(1) label:hover { background: #ecfdf3; }
.pcos-cycle-wrap .radio:nth-child(2) label:hover { background: #ffe4e6; }
.pcos-cycle-wrap .radio:nth-child(1):has(input:checked) label {
  border-color: #15803d;
  background: #dcfce7;
  box-shadow: 0 0 0 2px rgba(21, 128, 61, 0.35);
}
.pcos-cycle-wrap .radio:nth-child(2):has(input:checked) label {
  border-color: #be123c;
  background: #ffe4e6;
  box-shadow: 0 0 0 2px rgba(190, 18, 60, 0.35);
}
.pcos-cycle-wrap .radio:has(input:checked) .pcos-cycle-title { font-weight: 700; }
/* Hide native radio circle; whole card is the control (still keyboard-accessible). */
.pcos-cycle-wrap .radio label input[type="radio"] {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
.pcos-cycle-card-inner {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  width: 100%;
  align-items: flex-start;
}
.pcos-cycle-line1 {
  display: flex;
  align-items: center;
  width: 100%;
}
.pcos-cycle-title { font-weight: 600; font-size: 0.92rem; line-height: 1.25; }
.pcos-cycle-sub {
  font-size: 0.72rem;
  color: #64748b;
  line-height: 1.4;
  font-weight: 400;
  display: block;
  width: 100%;
}
.pcos-bmi-heading-row {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  margin-bottom: 0.2rem;
}

/* Symptom severity: native dropdown; left accent + tint follow selected value */
.pcos-select-symptom .shiny-input-container {
  margin-bottom: 0.65rem;
}
.pcos-select-symptom .control-label {
  font-weight: 600;
  font-size: 0.88rem;
  color: var(--pcos-teal-dark);
  margin-bottom: 0.3rem;
}
.pcos-select-symptom select.form-select {
  border-radius: 0.5rem;
  padding: 0.5rem 2.25rem 0.5rem 0.65rem;
  border: 1px solid #cbd5e1;
  border-left: 4px solid #94a3b8;
  background-color: #fff;
  font-size: 0.84rem;
  line-height: 1.35;
}
.pcos-select-symptom select.form-select:focus {
  border-color: #99f6e4;
  box-shadow: 0 0 0 0.2rem rgba(13, 148, 136, 0.2);
}
.pcos-select-symptom select.form-select:has(option[value="0"]:checked) {
  border-left-color: #22c55e;
  background-color: rgba(34, 197, 94, 0.09);
}
.pcos-select-symptom select.form-select:has(option[value="1"]:checked) {
  border-left-color: #ca8a04;
  background-color: rgba(234, 179, 8, 0.12);
}
.pcos-select-symptom select.form-select:has(option[value="2"]:checked) {
  border-left-color: #ea580c;
  background-color: rgba(249, 115, 22, 0.1);
}
.pcos-select-symptom select.form-select:has(option[value="3"]:checked) {
  border-left-color: #dc2626;
  background-color: rgba(220, 38, 38, 0.09);
}

/* BMI gauge */
.pcos-bmi-box {
  border: 1px solid #99f6e4;
  border-radius: 0.5rem;
  padding: 0.65rem 0.75rem;
  background: #fff;
  margin: 0.35rem 0 0.75rem 0;
}
.pcos-bmi-value { font-size: 1.35rem; font-weight: 700; color: var(--pcos-teal-darker); }
.pcos-bmi-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.65rem;
  color: #64748b;
  margin-top: 0.2rem;
}
.pcos-bmi-track-wrap {
  position: relative;
  margin: 0.45rem 0 0.15rem;
  height: 26px;
}
.pcos-bmi-gradient {
  height: 22px;
  border-radius: 11px;
  background: linear-gradient(90deg,
    #7dd3fc 0%,
    #7dd3fc 14%,
    #4ade80 14%,
    #4ade80 40%,
    #facc15 40%,
    #facc15 60%,
    #f87171 60%,
    #f87171 100%
  );
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.08);
}
.pcos-bmi-marker {
  position: absolute;
  top: -2px;
  width: 4px;
  height: 26px;
  background: #1e293b;
  border-radius: 2px;
  transform: translateX(-50%);
  box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.pcos-bmi-zones {
  display: flex;
  font-size: 0.62rem;
  color: #475569;
  justify-content: space-between;
  margin-top: 0.15rem;
}
"""

# BMI visual scale: map BMI 15-40 to 0-100% for marker
_BMI_SCALE_LO = 15.0
_BMI_SCALE_HI = 40.0
_LB_PER_KG = 2.2046226218


def _height_total_inches(feet: float, inches: float) -> float:
    """Feet + inches → total inches (imperial)."""
    return max(0.0, float(feet)) * 12.0 + max(0.0, float(inches))


def _compute_bmi_imperial(weight_lb: float, height_inches: float) -> float | None:
    """BMI from pounds and inches: (lb / in²) × 703 (standard US formula)."""
    if weight_lb <= 0 or height_inches <= 0:
        return None
    return round((weight_lb / (height_inches * height_inches)) * 703.0, 2)


def _imperial_to_metric_cm_kg(weight_lb: float, height_inches: float) -> tuple[float, float]:
    """For API / model keys that expect kg and cm."""
    height_cm = height_inches * 2.54
    weight_kg = weight_lb / _LB_PER_KG
    return round(height_cm, 2), round(weight_kg, 2)


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal range"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def _bmi_marker_left_pct(bmi: float) -> float:
    t = (bmi - _BMI_SCALE_LO) / (_BMI_SCALE_HI - _BMI_SCALE_LO)
    return max(0.0, min(100.0, t * 100.0))


def _section_heading(title: str, *tip_blocks: Any) -> ui.Tag:
    """Section title with hover/focus tooltip (ⓘ)."""
    tip_content: Any
    if len(tip_blocks) == 1:
        tip_content = tip_blocks[0]
    else:
        tip_content = ui.div(*tip_blocks, class_="text-start small")
    return ui.div(
        ui.span(title, class_="pcos-h4-text"),
        ui.tooltip(
            ui.tags.span(
                "i",
                class_="pcos-info-i",
                **{"aria-label": f"More about: {title}"},
            ),
            tip_content,
            placement="left",
        ),
        class_="pcos-h4-row",
    )


def _metrics_from_form(input: Any) -> tuple[float | None, float | None, float | None]:
    """Return (bmi, height_cm, weight_kg) from ft/in + lb inputs."""
    hi = _height_total_inches(input.height_ft(), input.height_in())
    lbs = float(input.weight_lbs())
    bmi = _compute_bmi_imperial(lbs, hi)
    if bmi is None:
        return None, None, None
    h_cm, w_kg = _imperial_to_metric_cm_kg(lbs, hi)
    return bmi, h_cm, w_kg


def _parse_opt_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def _build_payload(input: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    age = float(input.age())
    if age > 0:
        payload["age"] = age

    bmi, h_cm, w_kg = _metrics_from_form(input)
    if bmi is not None:
        payload["bmi"] = bmi
        payload["Weight (Kg)"] = w_kg
        payload["Height(Cm) "] = h_cm

    cyc = int(input.cycle_pattern())
    payload["cycle_ri"] = cyc
    if cyc == 1:
        payload["cycle_length_days"] = float(input.cycle_length_days())

    if not input.lh_not_tested():
        payload["lh"] = float(input.lh())
    if not input.fsh_not_tested():
        payload["fsh"] = float(input.fsh())
    if not input.tsh_not_tested():
        payload["tsh"] = float(input.tsh())

    hair = int(input.hair_level())
    payload["hair_growth"] = 1 if hair >= 2 else 0

    skin = int(input.skin_level())
    payload["skin_darkening"] = 1 if skin >= 1 else 0

    acne = int(input.acne_level())
    payload["pimples"] = 1 if acne >= 2 else 0

    wg = int(input.weight_change_level())
    payload["weight_gain"] = 1 if wg >= 1 else 0

    fl = int(input.follicle_l())
    fr = int(input.follicle_r())
    # Only send follicle counts when the user entered real ultrasound data.
    # The form defaults to 0, but 0 is clinically extreme (no one has zero
    # follicles) — the model treats it as a very strong anti-PCOS signal.
    # Omitting them lets the imputer fill training-set medians (~6/ovary),
    # which is a neutral "unknown imaging" baseline.
    if fl > 0 or fr > 0:
        payload["follicle_l"] = fl
        payload["follicle_r"] = fr

    return payload


def _sanitize_display(text: str) -> str:
    return str(text).replace("—", "-").replace("–", "-")


_HERO_DOT_PH = "\uE000"  # placeholder so "No." does not start a false sentence break


def _hero_explanation_blocks(text: str) -> list[str]:
    """Split model explanation into short paragraphs; keep abbreviations like 'No.' intact."""
    t = _sanitize_display(text).strip()
    if not t:
        return []
    t = re.sub(r"\bNo\.\s", _HERO_DOT_PH, t)
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [re.sub(_HERO_DOT_PH, "No. ", p).strip() for p in parts if p.strip()]


def _risk_tier(risk: float | None) -> str:
    if risk is None:
        return "low"
    r = float(risk)
    if r < 0.34:
        return "low"
    if r < 0.67:
        return "medium"
    return "high"


def _headline_from_risk_label(label: str) -> str:
    lab = (label or "").strip().lower()
    if "high" in lab:
        return "High risk of PCOS detected"
    if "medium" in lab or "moderate" in lab:
        return "Moderate PCOS risk on this screen"
    if "low" in lab:
        return "Lower PCOS risk on this screen"
    return (label or "Assessment result").strip() or "Assessment result"


def _metric_prob_class(tier: str) -> str:
    return {
        "low": "pcos-results-metric-value pcos-results-metric--prob-low",
        "medium": "pcos-results-metric-value pcos-results-metric--prob-med",
        "high": "pcos-results-metric-value pcos-results-metric--prob-high",
    }.get(tier, "pcos-results-metric-value pcos-results-metric--prob-med")


def _format_flags(
    flags: list[dict],
    *,
    dark: bool = True,
    empty_note: str | None = None,
) -> ui.Tag:
    if empty_note is None:
        empty_note = "No validation flags."
    if not flags:
        return ui.p(ui.em(empty_note), class_="pcos-results-muted small" if dark else "pcos-muted small")
    if not dark:
        return ui.tags.ul(
            *[
                ui.tags.li(
                    ui.tags.strong(f"{f.get('field', '?')}: "),
                    _sanitize_display(str(f.get("issue", ""))),
                    f" ({f.get('severity', '')})",
                )
                for f in flags[:12]
            ],
            class_="small mb-0",
        )
    lines: list[Any] = []
    for f in flags[:12]:
        sev = (f.get("severity") or "warning").lower()
        bullet_class = "pcos-results-bullet--warning" if sev != "error" else "pcos-results-bullet--error"
        lines.append(
            ui.div(
                ui.span("●", class_=f"pcos-results-bullet {bullet_class}"),
                ui.span(
                    ui.tags.strong(f"{f.get('field', '?')}: "),
                    _sanitize_display(str(f.get("issue", ""))),
                ),
                class_="pcos-results-flagline",
            )
        )
    return ui.div(*lines, class_="pcos-results-flags-wrap")


def _factors_table(factors: list[dict]) -> ui.Tag:
    if not factors:
        return ui.p(ui.em("No factor breakdown available."), class_="pcos-results-muted small")
    slice_ = factors[:8]
    max_abs = max(abs(float(f.get("shap_value") or 0)) for f in slice_) or 1e-9
    rows = []
    for tf in slice_:
        sv = float(tf.get("shap_value") or 0)
        abs_sv = abs(sv)
        width_pct = min(100.0, (abs_sv / max_abs) * 100.0)
        inc = sv > 0
        pill_class = "pcos-results-pill pcos-results-pill--up" if inc else "pcos-results-pill pcos-results-pill--down"
        pill_text = "Raises risk" if inc else "Lowers risk"
        bar_class = "pcos-results-strength-bar pcos-results-strength-bar--up" if inc else "pcos-results-strength-bar pcos-results-strength-bar--down"
        eff_class = "pcos-results-effect pcos-results-effect--up" if sv > 0 else "pcos-results-effect pcos-results-effect--down"
        rows.append(
            ui.tags.tr(
                ui.tags.td(_sanitize_display(str(tf.get("feature", ""))), class_="pcos-results-feat"),
                ui.tags.td(f"{sv:+.2f}", class_=eff_class),
                ui.tags.td(
                    ui.div(ui.span(style=f"width:{width_pct:.1f}%;"), class_=bar_class),
                    class_="pcos-results-strength-cell",
                ),
                ui.tags.td(ui.span(pill_text, class_=pill_class)),
            )
        )
    return ui.tags.table(
        ui.tags.thead(
            ui.tags.tr(
                ui.tags.th("Feature"),
                ui.tags.th("Effect"),
                ui.tags.th("Strength"),
                ui.tags.th("Direction"),
            )
        ),
        ui.tags.tbody(*rows),
        class_="pcos-results-htable",
    )


def _papers_list(title: str, papers: list[dict], kind: str) -> ui.Tag:
    if not papers:
        return ui.div()
    lis = []
    for p in papers[:5]:
        if kind == "chroma":
            t = _sanitize_display(str(p.get("title") or "Local paper"))
            y = p.get("year") or ""
            sub = f"{y} - similarity {p.get('distance', '')}"
        else:
            t = _sanitize_display(str(p.get("title") or "PubMed"))
            sub = f"PMID {p.get('pmid', '')} - {p.get('pubdate', '')}"
        lis.append(
            ui.tags.li(
                ui.tags.strong(t),
                ui.br(),
                ui.span(sub, class_="pcos-results-muted"),
            )
        )
    return ui.div(
        ui.p(title, class_="pcos-results-section-label"),
        ui.tags.ul(*lis),
    )


def _cycle_radio() -> ui.Tag:
    return ui.div(
        ui.div(
            ui.input_radio_buttons(
                "cycle_pattern",
                None,
                choices={
                    "1": ui.div(
                        ui.div(
                            ui.span("Regular", class_="pcos-cycle-title"),
                            class_="pcos-cycle-line1",
                        ),
                        ui.span("About every 21-35 days between starts.", class_="pcos-cycle-sub"),
                        class_="pcos-cycle-card-inner",
                    ),
                    "2": ui.div(
                        ui.div(
                            ui.span("Irregular", class_="pcos-cycle-title"),
                            class_="pcos-cycle-line1",
                        ),
                        ui.span("Timing varies a lot, very far apart, or you often skip months.", class_="pcos-cycle-sub"),
                        class_="pcos-cycle-card-inner",
                    ),
                },
                selected="1",
            ),
            class_="pcos-cycle-wrap",
        ),
        class_="pcos-cycle-outer",
    )


def _symptom_select(id_: str, label: str, choices: dict[str, str]) -> ui.Tag:
    """Single-select dropdown; values stay \"0\"-\"3\" for the model. Native select for CSS severity tint."""
    return ui.div(
        ui.input_select(id_, label, choices, selected="0"),
        class_="pcos-select-symptom",
    )


def _sidebar_inputs() -> ui.Tag:
    hair_choices = {
        "0": "None - No extra hair beyond your usual",
        "1": "Mild - A few hairs, easy to overlook",
        "2": "Moderate - Noticeable on lip, chin, or belly",
        "3": "Severe - Heavy on face, chest, or back",
    }
    skin_choices = {
        "0": "None - No velvety patches",
        "1": "Slight - Neck or folds, subtle",
        "2": "Visible - Clear velvety darkening",
        "3": "Marked - Obvious dark, rough areas",
    }
    acne_choices = {
        "0": "Clear - Little or no acne",
        "1": "Mild - Few small spots",
        "2": "Moderate - Several red or inflamed",
        "3": "Severe - Many large or cystic lesions",
    }
    weight_choices = {
        "0": "Stable - No real change lately",
        "1": "Slight - Small gain you noticed",
        "2": "Moderate - About 5-15 lb without a clear cause",
        "3": "Significant - Over ~15 lb or keeps rising",
    }

    lbl_lh = ui.tags.span("LH", ui.tags.span(" (mIU/mL)", class_="pcos-unit-sm"))
    lbl_fsh = ui.tags.span("FSH", ui.tags.span(" (mIU/mL)", class_="pcos-unit-sm"))
    lbl_tsh = ui.tags.span("TSH", ui.tags.span(" (mIU/L)", class_="pcos-unit-sm"))

    return ui.div(
        _section_heading(
            "About you",
            ui.p(
                "We use your age plus height and weight to estimate BMI with the usual formula "
                "for feet/inches and pounds: (weight in lb ÷ height in inches²) × 703. "
                "We then convert to metric for the research model."
            ),
            ui.p(
                "Underweight is below 18.5, healthy range about 18.5-24.9, overweight 25-29.9, "
                "and 30+ is the obese range on the chart - same bands doctors use for BMI."
            ),
        ),
        ui.p(
            "Tell us a little about yourself (age, body measurements, period cycles, "
            "hormone blood tests, changes, etc.). We’ll take care of the rest, turning your "
            "inputs into meaningful insights!",
            class_="pcos-help",
        ),
        ui.p("Our tool estimates your likelihood of PCOS and provides:", class_="pcos-help mb-0"),
        ui.tags.ul(
            ui.tags.li("A personalized risk score"),
            ui.tags.li("The top factors affecting your risk"),
            ui.tags.li("Evidence-based insights"),
            ui.tags.li("Clear recommendations and guidance"),
            class_="pcos-intro-bullets",
        ),
        ui.input_numeric("age", "Age (years)", value=28, min=12, max=90),
        ui.p("Your age in full years.", class_="pcos-help pcos-help-tight"),
        ui.input_numeric("height_ft", "Height - feet (ft)", value=5, min=0, max=8, step=1),
        ui.input_numeric("height_in", "Height - inches (in)", value=5, min=0, max=95, step=1),
        ui.p("(without shoes)", class_="pcos-help pcos-help-tight"),
        ui.input_numeric("weight_lbs", "Weight (lb)", value=154, min=50, max=500, step=1),
        ui.p("(typical morning weight with light clothing.)", class_="pcos-help pcos-help-tight"),
        ui.output_ui("bmi_panel"),

        _section_heading(
            "Periods & cycle",
            ui.p(
                "The study uses regular (1) vs irregular (2). Regular: fairly predictable spacing. "
                "Irregular: large gaps, unpredictable timing, or skipped months."
            ),
            ui.p(
                "If you pick Regular, we’ll ask for typical days between period starts (first day of "
                "bleeding to the next first day)."
            ),
        ),
        _cycle_radio(),
        ui.p("Choose the option that fits the past year best.", class_="pcos-help"),
        ui.panel_conditional(
            "input.cycle_pattern === '1'",
            ui.input_numeric(
                "cycle_length_days",
                "Days between period starts",
                value=28,
                min=15,
                max=120,
                step=1,
            ),
            ui.p(
                "Only shown when cycles are regular.",
                class_="pcos-help pcos-help-tight",
            ),
        ),

        _section_heading(
            "Hormone blood tests",
            ui.p(
                "These numbers usually come from a blood draw ordered by a clinician. If you haven’t "
                "been tested, check “not tested” and the tool will statistically fill in a plausible value."
            ),
        ),
        ui.p("Use your lab report if you have it.", class_="pcos-help"),
        ui.div(
            ui.input_checkbox("lh_not_tested", "LH: not tested yet", value=True),
            ui.panel_conditional(
                "!input.lh_not_tested",
                ui.input_numeric("lh", lbl_lh, value=8.0, min=0, max=200, step=0.1),
            ),
            ui.p(
                "(Many labs see roughly 2-15 mIU/mL in the follicular phase, but your lab’s reference range is what matters.)",
                class_="pcos-help pcos-help-tight",
            ),
            ui.input_checkbox("fsh_not_tested", "FSH: not tested yet", value=True),
            ui.panel_conditional(
                "!input.fsh_not_tested",
                ui.input_numeric("fsh", lbl_fsh, value=5.0, min=0, max=200, step=0.1),
            ),
            ui.p(
                "(Often around 3-10 mIU/mL in many cycles. A high LH:FSH ratio (about 2:1 or more) can be one "
                "PCOS clue; the model uses LH and FSH together.)",
                class_="pcos-help pcos-help-tight",
            ),
            ui.input_checkbox("tsh_not_tested", "TSH: not tested yet", value=True),
            ui.panel_conditional(
                "!input.tsh_not_tested",
                ui.input_numeric("tsh", lbl_tsh, value=2.0, min=0, max=50, step=0.1),
            ),
            ui.p(
                "(Thyroid screening; many labs use about 0.4-4.0 mIU/L as a broad normal window. Thyroid issues "
                "can mimic some PCOS symptoms.)",
                class_="pcos-help pcos-help-tight",
            ),
            class_="pcos-hormone-block",
        ),

        _section_heading(
            "Hair, skin, acne, weight change",
            ui.p(
                "Extra hair: similar in spirit to the Ferriman-Gallwey scale; moderate or severe options "
                "count as “yes” for the screening model."
            ),
            ui.p(
                "Dark velvety patches: often on neck, underarms, groin, or knuckles. Any option beyond "
                "“none” counts as “yes.”"
            ),
            ui.p(
                "Acne: similar to a simple dermatology “how severe overall” scale; moderate or severe counts as “yes.”"
            ),
            ui.p(
                "Weight change: focus on gain you didn’t plan over the last 6-12 months. Any noticeable "
                "unintended gain counts as “yes.”"
            ),
        ),
        _symptom_select("hair_level", "Extra hair growth (face or body)", hair_choices),
        _symptom_select("skin_level", "Dark, velvety skin patches", skin_choices),
        _symptom_select("acne_level", "Acne / pimples", acne_choices),
        _symptom_select("weight_change_level", "Weight change (last 6-12 months)", weight_choices),
        ui.p(
            "Choose one level per question from the menus. The bar color matches severity: mild (green) "
            "through more concerning (red).",
            class_="pcos-help",
        ),

        ui.accordion(
            ui.accordion_panel(
                ui.span(
                    "Ultrasound results (optional) ",
                    ui.tooltip(
                        ui.tags.span("i", class_="pcos-info-i", **{"aria-label": "About ultrasound fields"}),
                        ui.p(
                            "If you had a pelvic ultrasound, enter follicle counts per ovary. "
                            "About 12 or more small follicles in one ovary can be part of the PCOS picture. "
                            "Leave both at 0 if you have no report; the model will use average population "
                            "values instead of assuming your imaging was clear."
                        ),
                        placement="left",
                    ),
                ),
                ui.input_numeric("follicle_l", "Follicles - left ovary", value=0, min=0, max=50, step=1),
                ui.input_numeric("follicle_r", "Follicles - right ovary", value=0, min=0, max=50, step=1),
                value="us_panel",
            ),
            id="us_accordion",
            multiple=False,
            open=False,
            class_="mt-1",
        ),

        ui.input_action_button(
            "submit",
            "Run assessment",
            class_="btn-primary w-100 mt-3",
        ),
        class_="pcos-sidebar",
    )


app_ui = ui.page_sidebar(
    ui.sidebar(
        _sidebar_inputs(),
        title="Your information",
        width=460,
        open="desktop",
        bg="#ecfdf5",
        class_="border-end",
    ),
    ui.tags.head(ui.tags.style(TEAL_CSS)),
    ui.output_ui("api_status_banner"),
    ui.div(
        ui.div(
            ui.div(
                ui.h1("PCOSense", class_="pcos-header-title"),
                ui.p(
                    "Multi-agent screening for polycystic ovary syndrome\u00a0(PCOS)",
                    class_="pcos-header-line2",
                ),
                ui.p("Your personal PCOS screening companion", class_="pcos-header-line3"),
                class_="pcos-header-top",
            ),
            ui.div(
                ui.p(
                    "Validation, medical evidence lookup, and risk modeling run together in one guided flow.",
                    class_="pcos-header-flow",
                ),
                class_="pcos-header-mid",
            ),
            ui.div(
                ui.p(
                    "Use the sidebar when you're ready. Share what you know; anything you skip is filled in thoughtfully "
                    "so you still get a useful snapshot.",
                    class_="pcos-tagline",
                ),
                class_="pcos-header-tip",
            ),
            ui.p(UI_BUILD_STAMP, class_="pcos-header-stamp"),
            class_="pcos-header-main",
        ),
        ui.div(
            ui.output_ui("results_panel"),
            class_="pcos-main-inner",
        ),
    ),
    title=None,
    window_title="PCOSense",
    fillable=True,
)


def server(input: Any, output: Any, session: Any) -> None:
    form_error = reactive.Value(None)

    @reactive.poll(lambda: int(time.time() // 12), interval_secs=3)
    def api_health_ok() -> bool:
        try:
            with httpx.Client(timeout=4.0) as client:
                client.get(HEALTH_URL).raise_for_status()
            return True
        except Exception:
            return False

    @render.ui
    def bmi_panel() -> ui.Tag:
        hi = _height_total_inches(input.height_ft(), input.height_in())
        lbs = float(input.weight_lbs())
        bmi = _compute_bmi_imperial(lbs, hi)
        if bmi is None:
            return ui.div(
                ui.p("Enter feet, inches, and weight in pounds to see your BMI.", class_="pcos-muted small mb-0"),
                class_="pcos-bmi-box",
            )
        left = _bmi_marker_left_pct(bmi)
        cat = _bmi_category(bmi)
        return ui.div(
            ui.div(
                ui.span("Your estimated BMI", class_="small text-muted"),
                ui.tooltip(
                    ui.tags.span(
                        "i",
                        class_="pcos-info-i",
                        **{"aria-label": "About this BMI chart"},
                    ),
                    ui.div(
                        ui.p("Bar colors: blue - underweight, green - healthy, yellow - overweight, red - higher BMI."),
                        ui.p("Marker maps your BMI onto a 15-40 window along the bar."),
                        ui.p("Formula: (lb ÷ in²) × 703."),
                        class_="text-start small",
                    ),
                    placement="top",
                ),
                class_="pcos-bmi-heading-row",
            ),
            ui.div(f"{bmi}", class_="pcos-bmi-value"),
            ui.p(cat, class_="small mb-1", style="color:#0f766e;font-weight:600;"),
            ui.div(
                ui.div(class_="pcos-bmi-gradient"),
                ui.div(class_="pcos-bmi-marker", style=f"left:{left}%;"),
                class_="pcos-bmi-track-wrap",
            ),
            ui.div(
                ui.span("Under"),
                ui.span("Normal"),
                ui.span("Over"),
                ui.span("Higher"),
                class_="pcos-bmi-zones",
            ),
            class_="pcos-bmi-box",
        )

    @extended_task
    async def run_assess(payload: dict[str, Any]) -> dict[str, Any]:
        timeout = httpx.Timeout(600.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(ASSESS_URL, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text
                try:
                    body = exc.response.json()
                    d = body.get("detail", detail)
                    if isinstance(d, list):
                        detail = "; ".join(str(x) for x in d)
                    else:
                        detail = str(d)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
                raise RuntimeError(f"API {exc.response.status_code}: {detail}") from exc
            try:
                return r.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError("API returned non-JSON response") from exc

    @reactive.effect
    @reactive.event(input.submit)
    def _start_assess() -> None:
        form_error.set(None)
        bmi, _, _ = _metrics_from_form(input)
        if bmi is None:
            form_error.set(
                "Please enter valid height (feet and inches) and weight in pounds so we can estimate BMI."
            )
            return
        payload = _build_payload(input)
        if not payload:
            form_error.set("Something went wrong building your answers - try again.")
            return
        run_assess.invoke(payload)

    @render.ui
    def api_status_banner() -> ui.Tag:
        ok = api_health_ok()
        if ok:
            return ui.div(
                ui.span("API connected", class_="badge bg-success me-2"),
                ui.span(ASSESS_URL, class_="pcos-muted small"),
                class_="pcos-api-banner pcos-muted small",
            )
        return ui.div(
            ui.span("API offline", class_="badge bg-danger me-2"),
            ui.span(
                f"Start backend: uvicorn src.api.main:app --host 127.0.0.1 --port 8000 - {API_BASE}",
                class_="small",
            ),
            class_="pcos-api-banner alert alert-warning py-2 mb-0",
        )

    @render.ui
    def results_panel() -> ui.Tag:
        fe = form_error()
        if fe:
            return ui.div(ui.h3("Check your inputs"), ui.p(fe), class_="pcos-card")

        st = run_assess.status()
        if st == "initial":
            return ui.div(
                ui.p(
                    "Your summary will land in this space when you're ready. After you press ",
                    ui.tags.span(
                        '"Run assessment"',
                        style="font-weight:600;color:#0f766e;",
                    ),
                    " in the sidebar, most visits finish in about a minute while we line up your answers with "
                    "recent research.",
                    class_="pcos-results-placeholder mb-0",
                ),
                class_="pcos-results-root",
            )

        if st == "running":
            return ui.div(
                ui.h3("Assessment in progress"),
                ui.p(
                    "Running the screening pipeline - this often takes about a minute. You can keep this tab open.",
                    class_="pcos-muted",
                ),
                ui.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm text-secondary me-2"),
                    ui.span("Working…"),
                ),
                class_="pcos-card",
            )

        if st == "error":
            err = run_assess.error()
            return ui.div(
                ui.h3("Something went wrong"),
                ui.pre(str(err), class_="bg-light p-2 rounded small"),
                ui.p(
                    "If the API stopped, check the terminal running uvicorn for a traceback.",
                    class_="pcos-muted small",
                ),
                class_="pcos-card",
            )

        if st == "cancelled":
            return ui.div(ui.p("Cancelled."), class_="pcos-card")

        data = run_assess.value()
        meta = data.get("metadata") or {}
        if meta.get("status") == "rejected":
            v = data.get("validation") or {}
            return ui.div(
                ui.div(
                    ui.h3("Assessment not run"),
                    ui.p(
                        "Validation failed - fix the issues below or adjust inputs.",
                        class_="pcos-muted",
                    ),
                    ui.p(ui.tags.strong("Status: "), v.get("status", "")),
                    _format_flags(v.get("flags") or [], dark=False),
                    class_="pcos-card",
                ),
            )

        v = data.get("validation") or {}
        e = data.get("evidence") or {}
        a = data.get("assessment") or {}

        risk = a.get("risk_score")
        label = a.get("risk_label") or ""
        pct = round(float(risk) * 100, 1) if risk is not None else None

        top_factors = a.get("top_factors") or []
        rec = a.get("recommendation") or ""
        summary = e.get("clinical_summary") or ""
        criteria = e.get("diagnostic_criteria") or []

        tier = _risk_tier(float(risk) if risk is not None else None)
        headline = _headline_from_risk_label(str(label))
        explanation_raw = str(a.get("explanation_text") or "")
        explanation_blocks = _hero_explanation_blocks(explanation_raw)
        conf_raw = v.get("confidence_score")
        try:
            conf_val = float(conf_raw) if conf_raw is not None else None
        except (TypeError, ValueError):
            conf_val = None

        hero_inner: list[Any] = [
            ui.div(
                (
                    ui.span(f"{pct}%", class_="pcos-results-badge-pct")
                    if pct is not None
                    else ui.span("-", class_="pcos-results-badge-pct")
                ),
                ui.span("score", class_="pcos-results-badge-sub"),
                class_="pcos-results-score-badge",
            ),
            ui.div(
                ui.p(headline, class_="pcos-results-hero-title"),
                (
                    ui.div(
                        *[
                            ui.p(block, class_="pcos-results-hero-lead")
                            for block in explanation_blocks
                        ],
                        class_="pcos-results-hero-text",
                    )
                    if explanation_blocks
                    else ui.div()
                ),
            ),
        ]

        body_children: list[Any] = [
            ui.div(
                hero_inner[0],
                hero_inner[1],
                class_=f"pcos-results-hero pcos-results-hero--{tier}",
            ),
            ui.div(
                ui.div(
                    ui.p("Probability score", class_="pcos-results-metric-label"),
                    ui.div(
                        f"{float(risk):.4f}" if risk is not None else "-",
                        class_=_metric_prob_class(tier),
                    ),
                    class_="pcos-results-metric",
                ),
                ui.div(
                    ui.p("Validation confidence", class_="pcos-results-metric-label"),
                    ui.div(
                        f"{conf_val:.2f}" if conf_val is not None else "-",
                        class_="pcos-results-metric-value pcos-results-metric--conf",
                    ),
                    class_="pcos-results-metric",
                ),
                class_="pcos-results-metrics",
            ),
            ui.p(
                "The percentage is from an XGBoost model trained on a clinical dataset. "
                "Features you do not provide are filled with that dataset’s typical values before scoring, "
                "so the number reflects the model—not a diagnosis.",
                class_="pcos-results-muted small mt-2 mb-0",
            ),
            ui.div(
                ui.p("Top factors influencing this result", class_="pcos-results-section-label"),
                _factors_table(top_factors),
                class_="pcos-results-panel-dark",
            ),
            ui.div(
                ui.p("Supporting clinical evidence", class_="pcos-results-section-label"),
                ui.div(
                    _papers_list("Knowledge base (Chroma)", e.get("retrieved_papers") or [], "chroma"),
                    _papers_list("PubMed", e.get("pubmed_papers") or [], "pubmed"),
                    class_="pcos-results-evidence-stack",
                ),
                class_="pcos-results-panel-dark",
            ),
            ui.div(
                ui.p("Clinical summary", class_="pcos-results-section-label"),
                (
                    ui.p(_sanitize_display(str(summary)), class_="small", style="color:#cbd5e1;line-height:1.45;")
                    if summary
                    else ui.p(ui.em("No summary returned (LLM may be offline)."), class_="pcos-results-muted small")
                ),
                (
                    ui.div(
                        ui.p("Diagnostic criteria (model output)", class_="pcos-results-section-label"),
                        ui.tags.ul(
                            *[ui.tags.li(_sanitize_display(str(x))) for x in criteria[:8]],
                            style="color:#cbd5e1;font-size:0.84rem;",
                        ),
                    )
                    if criteria
                    else ui.div()
                ),
                class_="pcos-results-panel-dark",
            ),
            ui.div(
                ui.p("Data quality", class_="pcos-results-section-label"),
                ui.p(
                    ui.span("● ", style="color:#4ade80;font-size:0.85rem;"),
                    ui.tags.strong("Validation status: ", style="color:#e2e8f0;"),
                    ui.span(
                        f"{v.get('status', '')} - confidence {v.get('confidence_score', '')}",
                        style="color:#fde68a;",
                    ),
                    class_="pcos-results-data-quality-line",
                ),
                class_="pcos-results-panel-dark",
            ),
            ui.div(
                ui.p("Warnings", class_="pcos-results-section-label"),
                _format_flags(v.get("flags") or [], empty_note="No active warnings."),
                class_="pcos-results-panel-dark",
            ),
            ui.div(
                ui.h3("Recommendation", class_="h6 text-uppercase", style="font-size:0.72rem;letter-spacing:0.06em;color:#64748b;"),
                (
                    ui.p(_sanitize_display(str(rec)), style="line-height:1.5;margin:0;")
                    if rec
                    else ui.p(ui.em("No recommendation text (LLM may be offline)."), class_="pcos-muted mb-0")
                ),
                class_="pcos-results-card-light",
            ),
            ui.p(
                _sanitize_display(
                    f"Pipeline {meta.get('status', '')} in {meta.get('elapsed_sec', '')}s - "
                    "Not a substitute for clinical judgement."
                ),
                class_="pcos-results-footnote",
            ),
        ]

        return ui.div(*body_children, class_="pcos-results-root")


app = App(app_ui, server)
