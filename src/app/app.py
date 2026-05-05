"""
PCOSense Shiny frontend - patient form and multi-agent assessment results.
Requires the FastAPI server (see README).

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

_UI_BUILD_STAMP = "2026-04-12 • header-layout-v15"  # internal only, never rendered

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --primary:      #7C5CBF;
  --primary-lt:   #A589D8;
  --primary-dk:   #5E3EA1;
  --primary-bg:   #F7F3FF;
  --rose:         #E8A4B0;
  --rose-dk:      #C97C84;
  --sage:         #6BAE9A;
  --sage-dk:      #4D9180;
  --surface:      #FFFFFF;
  --bg:           #FAFAF8;
  --border:       #EDE8F8;
  --text:         #1A1A2E;
  --text-muted:   #64748B;
  --text-light:   #94A3B8;
  --shadow-sm:    0 1px 3px rgba(124,92,191,0.08);
  --shadow-md:    0 4px 14px rgba(124,92,191,0.12);
  --radius:       0.65rem;
}

*, *::before, *::after { box-sizing: border-box; }
body {
  background: var(--bg) !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  color: var(--text);
}

/* ── Header ──────────────────────────────────────────────────── */
.pcos-header-main {
  background: var(--surface);
  border-left: 4px solid var(--primary);
  border-radius: var(--radius);
  padding: 1.6rem 1.75rem;
  margin-bottom: 1.25rem;
  box-shadow: var(--shadow-md);
  max-width: 920px;
  margin-left: auto;
  margin-right: auto;
}
.pcos-header-main h1.pcos-header-title {
  margin: 0 0 0.3rem 0;
  font-weight: 800;
  letter-spacing: -0.03em;
  font-size: clamp(1.75rem, 4vw, 2.4rem);
  line-height: 1.1;
  color: var(--primary-dk);
}
.pcos-header-main .pcos-header-sub {
  color: var(--text-muted);
  font-size: 0.95rem;
  font-weight: 400;
  margin: 0;
  line-height: 1.5;
}

/* ── Main container ──────────────────────────────────────────── */
.pcos-main-inner {
  max-width: 920px;
  margin: 0 auto;
  width: 100%;
  padding: 0 0.5rem 2rem;
}

/* ── Cards ───────────────────────────────────────────────────── */
.pcos-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.3rem;
  margin-bottom: 1rem;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.pcos-card h3 {
  color: var(--primary-dk);
  font-size: 1rem;
  margin: 0 0 0.6rem 0;
  font-weight: 600;
}
.pcos-card--light-panel {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.1rem 1.2rem;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.pcos-card--light-panel h3 {
  color: var(--primary-dk);
  font-size: 0.95rem;
  margin: 0 0 0.5rem 0;
  font-weight: 600;
}

/* ── Sidebar inputs ──────────────────────────────────────────── */
.pcos-muted { color: var(--text-muted); font-size: 0.9rem; }
.pcos-help {
  color: var(--text-muted);
  font-size: 0.78rem;
  line-height: 1.45;
  margin: 0.1rem 0 0.55rem 0;
}
.pcos-help-tight { margin-top: -0.1rem; }

.btn-primary, .btn-primary:focus {
  background-color: var(--primary) !important;
  border-color: var(--primary-dk) !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
}
.btn-primary:hover {
  background-color: var(--primary-dk) !important;
  border-color: #4a2f8a !important;
}

.form-label { font-weight: 500; font-size: 0.88rem; }
.pcos-sidebar .shiny-input-container { margin-bottom: 0.4rem; }

.pcos-h4-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin: 1.1rem 0 0.45rem 0;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--primary-dk);
}
.pcos-h4-row:first-of-type { margin-top: 0.2rem; }
.pcos-h4-text { flex: 1; }

.pcos-info-i {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.15rem;
  height: 1.15rem;
  padding: 0 0.2rem;
  border-radius: 50%;
  background: var(--primary-lt);
  color: #fff !important;
  font-size: 0.6rem;
  font-weight: 700;
  font-style: italic;
  font-family: Georgia, serif;
  cursor: help;
  line-height: 1;
  flex-shrink: 0;
}

.pcos-unit-sm { font-size: 0.72em; font-weight: 400; color: var(--text-light); }
.pcos-hormone-block .shiny-input-container > label { font-size: 0.88rem; }

/* ── API status banner ───────────────────────────────────────── */
.pcos-api-banner { max-width: 920px; margin: 0 auto 0.75rem; }
.pcos-api-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 0.4rem;
  vertical-align: middle;
}
.pcos-api-dot--ok  { background: #16a34a; }
.pcos-api-dot--err { background: #dc2626; }

/* ── Placeholder ─────────────────────────────────────────────── */
.pcos-results-placeholder {
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 2.5rem 1.5rem;
  text-align: center;
  background: var(--surface);
  color: var(--text-muted);
  max-width: 920px;
  margin: 0 auto 1rem auto;
  line-height: 1.6;
  font-size: 0.95rem;
  box-shadow: var(--shadow-sm);
}
.pcos-placeholder-icon {
  font-size: 2.5rem;
  margin-bottom: 0.75rem;
  opacity: 0.5;
}

/* ── Progress steps (running state) ──────────────────────────── */
.pcos-steps {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
  flex-direction: column;
  margin-top: 1rem;
}
.pcos-step {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  font-size: 0.88rem;
  color: var(--text-muted);
  padding: 0.5rem 0.75rem;
  border-radius: 0.45rem;
  background: var(--primary-bg);
  width: 100%;
}
.pcos-step--active {
  color: var(--primary-dk);
  font-weight: 600;
  background: var(--primary-bg);
  border: 1px solid var(--primary-lt);
}
.pcos-step--done { color: var(--sage-dk); background: #f0fdf8; }
.pcos-step-num {
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 50%;
  background: var(--primary-lt);
  color: #fff;
  font-size: 0.7rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.pcos-step--done .pcos-step-num { background: var(--sage); }

/* ── Results hero ─────────────────────────────────────────────── */
.pcos-results-root {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
}
.pcos-results-hero {
  border-radius: var(--radius);
  padding: 1.5rem 1.6rem;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 1.4rem;
  align-items: center;
  box-shadow: var(--shadow-md);
}
.pcos-results-hero--low {
  background: linear-gradient(135deg, #1a5c38 0%, #22c55e 60%, #86efac 130%);
  color: #fff;
}
.pcos-results-hero--medium {
  background: linear-gradient(135deg, #92400e 0%, #d97706 55%, #fcd34d 130%);
  color: #fff;
}
.pcos-results-hero--high {
  background: linear-gradient(135deg, #6b1a1a 0%, #dc2626 50%, #f87171 120%);
  color: #fff;
}
.pcos-results-score-badge {
  width: 5.5rem;
  height: 5.5rem;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  line-height: 1.1;
  background: rgba(255,255,255,0.2);
  border: 2.5px solid rgba(255,255,255,0.4);
  flex-shrink: 0;
}
.pcos-results-badge-pct { font-size: 1.2rem; }
.pcos-results-badge-sub { font-size: 0.6rem; font-weight: 600; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.04em; }
.pcos-results-hero-title {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  line-height: 1.25;
}
.pcos-results-hero-lead {
  margin: 0 0 0.6rem 0;
  opacity: 0.94;
  font-size: 0.88rem;
  line-height: 1.6;
}
.pcos-results-hero-text .pcos-results-hero-lead:last-child { margin-bottom: 0; }

@media (max-width: 540px) {
  .pcos-results-hero { grid-template-columns: 1fr; justify-items: center; text-align: center; }
}

/* ── Metric strip ─────────────────────────────────────────────── */
.pcos-results-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
}
@media (max-width: 480px) { .pcos-results-metrics { grid-template-columns: 1fr; } }
.pcos-results-metric {
  border-radius: 0.55rem;
  padding: 0.9rem 1rem;
  border: 1px solid var(--border);
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.pcos-results-metric-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-light);
  margin-bottom: 0.2rem;
  font-weight: 600;
}
.pcos-results-metric-value {
  font-size: 1.7rem;
  font-weight: 700;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.pcos-results-metric--prob-low  { color: #15803d; }
.pcos-results-metric--prob-med  { color: #b45309; }
.pcos-results-metric--prob-high { color: #b91c1c; }

/* ── Factors table (light) ────────────────────────────────────── */
.pcos-factors-panel {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.15rem 1.15rem;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.pcos-section-label {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-light);
  margin: 0 0 0.7rem 0;
  font-weight: 700;
}
.pcos-results-htable { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.pcos-results-htable th {
  text-align: left;
  padding: 0.3rem 0.4rem 0.45rem;
  border-bottom: 2px solid var(--border);
  color: var(--text-light);
  font-weight: 600;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.pcos-results-htable td {
  padding: 0.55rem 0.4rem;
  border-top: 1px solid var(--border);
  vertical-align: middle;
}
.pcos-results-feat { color: var(--text); font-weight: 500; }
.pcos-results-effect { font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap; }
.pcos-results-effect--up   { color: #dc2626; }
.pcos-results-effect--down { color: #16a34a; }
.pcos-results-strength-cell { width: 26%; min-width: 5rem; }
.pcos-results-strength-bar {
  height: 8px;
  border-radius: 999px;
  background: var(--border);
  overflow: hidden;
}
.pcos-results-strength-bar > span {
  display: block; height: 100%; border-radius: 999px; min-width: 4px;
  background: linear-gradient(90deg, #94a3b8, #64748b);
}
.pcos-results-strength-bar--up > span   { background: linear-gradient(90deg, #fca5a5, #dc2626); }
.pcos-results-strength-bar--down > span { background: linear-gradient(90deg, #86efac, #16a34a); }
.pcos-results-pill {
  display: inline-block;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}
.pcos-results-pill--up   { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.pcos-results-pill--down { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.pcos-results-muted { color: var(--text-light); }

/* ── Evidence / clinical panels (light) ──────────────────────── */
.pcos-evidence-panel {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.15rem 1.15rem;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.pcos-results-evidence-stack { display: flex; flex-direction: column; gap: 0.85rem; }
.pcos-evidence-papers-list { margin: 0; padding-left: 0; list-style: none; }
.pcos-evidence-papers-list li {
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.83rem;
  color: var(--text);
}
.pcos-evidence-papers-list li:last-child { border-bottom: none; }
.pcos-evidence-papers-list .paper-sub { color: var(--text-light); font-size: 0.75rem; }

/* ── Recommendation card ──────────────────────────────────────── */
.pcos-rec-card {
  border: 1px solid var(--border);
  border-left: 4px solid var(--primary);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  background: var(--primary-bg);
  box-shadow: var(--shadow-sm);
}
.pcos-rec-card h3 {
  color: var(--primary-dk);
  font-size: 0.95rem;
  margin: 0 0 0.5rem 0;
  font-weight: 600;
}

/* ── Next-steps cards ─────────────────────────────────────────── */
.pcos-nextsteps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem;
}
.pcos-nextstep-card {
  border: 1px solid var(--border);
  border-radius: 0.55rem;
  padding: 0.9rem 1rem;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.pcos-nextstep-card-icon { font-size: 1.35rem; margin-bottom: 0.35rem; }
.pcos-nextstep-card-title { font-weight: 600; font-size: 0.88rem; color: var(--text); margin-bottom: 0.2rem; }
.pcos-nextstep-card-body  { font-size: 0.78rem; color: var(--text-muted); line-height: 1.45; }

/* ── Warnings / flags (light) ────────────────────────────────── */
.pcos-results-flags-wrap { font-size: 0.84rem; line-height: 1.45; }
.pcos-results-flagline {
  display: flex;
  gap: 0.45rem;
  align-items: flex-start;
  margin-bottom: 0.4rem;
  color: var(--text);
}
.pcos-results-bullet { flex-shrink: 0; margin-top: 0.15rem; font-size: 0.75rem; }
.pcos-results-bullet--error   { color: #dc2626; }
.pcos-results-bullet--warning { color: #d97706; }
.pcos-results-bullet--info    { color: #6b7280; }

/* ── Footnote ─────────────────────────────────────────────────── */
.pcos-results-footnote {
  font-size: 0.76rem;
  color: var(--text-light);
  margin: 0.1rem 0 0 0;
  text-align: center;
}

/* ── Quality Control (compact bar, moved to top of results) ───── */
.pcos-qc-bar {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  padding: 0.65rem 1rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  flex-wrap: wrap;
}
.pcos-qc-bar-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 700;
  color: var(--primary);
  flex-shrink: 0;
}
.pcos-qc-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  background: var(--primary-bg);
  color: var(--primary-dk);
  border: 1px solid var(--border);
}
.pcos-qc-chip--high { background: #f0fdf8; color: var(--sage-dk); border-color: #a7f3d0; }
.pcos-qc-chip--med  { background: #fffbeb; color: #b45309; border-color: #fde68a; }
.pcos-qc-chip--low  { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.pcos-qc-bar-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.pcos-qc-bar-dot--high { background: var(--sage); }
.pcos-qc-bar-dot--med  { background: #eab308; }
.pcos-qc-bar-dot--low  { background: #dc2626; }

/* ── Full QC section (expanded, bottom of results) ────────────── */
.pcos-qc-section {
  border-radius: var(--radius);
  padding: 1.15rem 1.3rem;
  background: #faf8ff;
  border: 1px solid var(--border);
}
.pcos-qc-title {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 700;
  color: var(--primary);
  margin: 0 0 0.85rem 0;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.pcos-qc-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 1rem;
  height: 1rem;
  background: var(--primary);
  color: white;
  border-radius: 50%;
  font-size: 0.6rem;
  font-weight: 700;
}
.pcos-qc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.65rem;
  margin-bottom: 0.75rem;
}
.pcos-qc-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.75rem 0.85rem;
  box-shadow: var(--shadow-sm);
}
.pcos-qc-card-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-light);
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.pcos-qc-card-value {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--primary);
  font-variant-numeric: tabular-nums;
}
.pcos-qc-card-sub { font-size: 0.63rem; color: var(--text-light); margin-top: 0.15rem; }
.pcos-qc-flags { margin-top: 0.75rem; font-size: 0.81rem; }
.pcos-qc-flag-item {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
  padding: 0.45rem 0.6rem;
  background: var(--surface);
  border-left: 3px solid var(--border);
  border-radius: 0.25rem;
}
.pcos-qc-flag-item--pass    { border-left-color: #16a34a; background: #f0fdf4; }
.pcos-qc-flag-item--warning { border-left-color: #eab308; background: #fffbeb; }
.pcos-qc-flag-item--error   { border-left-color: #dc2626; background: #fef2f2; }
.pcos-qc-flag-badge {
  display: inline-block;
  padding: 0.12rem 0.38rem;
  border-radius: 0.25rem;
  font-weight: 600;
  font-size: 0.63rem;
  text-transform: uppercase;
  flex-shrink: 0;
}
.pcos-qc-flag-badge--pass    { background: #dcfce7; color: #166534; }
.pcos-qc-flag-badge--warning { background: #fef08a; color: #854d0e; }
.pcos-qc-flag-badge--error   { background: #fecaca; color: #991b1b; }
.pcos-qc-flag-text { flex: 1; line-height: 1.35; color: var(--text-muted); }

/* ── Cycle radio cards ────────────────────────────────────────── */
.pcos-cycle-outer {
  width: 100%;
  display: flex;
  justify-content: center;
  margin: 0.2rem 0 0.35rem 0;
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
  min-height: 5rem;
  transition: border-color 0.18s, background 0.18s, box-shadow 0.18s;
}
.pcos-cycle-wrap .radio label:has(input:focus-visible) {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}
.pcos-cycle-wrap .radio:nth-child(1) label { border: 2px solid #86efac; background: #f0fdf4; }
.pcos-cycle-wrap .radio:nth-child(2) label { border: 2px solid #fda4af; background: #fff1f2; }
.pcos-cycle-wrap .radio:nth-child(1) label:hover { background: #ecfdf3; }
.pcos-cycle-wrap .radio:nth-child(2) label:hover { background: #ffe4e6; }
.pcos-cycle-wrap .radio:nth-child(1):has(input:checked) label {
  border-color: #15803d; background: #dcfce7;
  box-shadow: 0 0 0 2px rgba(21,128,61,0.3);
}
.pcos-cycle-wrap .radio:nth-child(2):has(input:checked) label {
  border-color: #be123c; background: #ffe4e6;
  box-shadow: 0 0 0 2px rgba(190,18,60,0.3);
}
.pcos-cycle-wrap .radio:has(input:checked) .pcos-cycle-title { font-weight: 700; }
.pcos-cycle-wrap .radio label input[type="radio"] {
  position: absolute; width: 1px; height: 1px; padding: 0;
  margin: -1px; overflow: hidden; clip: rect(0,0,0,0);
  white-space: nowrap; border-width: 0;
}
.pcos-cycle-card-inner { display: flex; flex-direction: column; gap: 0.35rem; width: 100%; align-items: flex-start; }
.pcos-cycle-line1 { display: flex; align-items: center; width: 100%; }
.pcos-cycle-title { font-weight: 600; font-size: 0.9rem; line-height: 1.25; }
.pcos-cycle-sub   { font-size: 0.7rem; color: var(--text-muted); line-height: 1.4; font-weight: 400; display: block; width: 100%; }

/* ── Symptom selects ──────────────────────────────────────────── */
.pcos-select-symptom .shiny-input-container { margin-bottom: 0.6rem; }
.pcos-select-symptom .control-label {
  font-weight: 600; font-size: 0.88rem; color: var(--primary-dk); margin-bottom: 0.25rem;
}
.pcos-select-symptom select.form-select {
  border-radius: 0.5rem;
  padding: 0.5rem 2.25rem 0.5rem 0.65rem;
  border: 1px solid #cbd5e1;
  border-left: 4px solid #94a3b8;
  background-color: var(--surface);
  font-size: 0.84rem;
  line-height: 1.35;
}
.pcos-select-symptom select.form-select:focus {
  border-color: var(--primary-lt);
  box-shadow: 0 0 0 0.2rem rgba(124,92,191,0.2);
}
.pcos-select-symptom select.form-select:has(option[value="0"]:checked) { border-left-color:#22c55e; background-color:rgba(34,197,94,0.07); }
.pcos-select-symptom select.form-select:has(option[value="1"]:checked) { border-left-color:#ca8a04; background-color:rgba(234,179,8,0.1); }
.pcos-select-symptom select.form-select:has(option[value="2"]:checked) { border-left-color:#ea580c; background-color:rgba(249,115,22,0.08); }
.pcos-select-symptom select.form-select:has(option[value="3"]:checked) { border-left-color:#dc2626; background-color:rgba(220,38,38,0.07); }

/* ── BMI gauge ────────────────────────────────────────────────── */
.pcos-bmi-box {
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.65rem 0.75rem;
  background: var(--surface);
  margin: 0.3rem 0 0.7rem 0;
}
.pcos-bmi-heading-row { display: flex; align-items: center; gap: 0.35rem; margin-bottom: 0.2rem; }
.pcos-bmi-value { font-size: 1.35rem; font-weight: 700; color: var(--primary-dk); }
.pcos-bmi-track-wrap { position: relative; margin: 0.4rem 0 0.12rem; height: 26px; }
.pcos-bmi-gradient {
  height: 22px;
  border-radius: 11px;
  background: linear-gradient(90deg,
    #7dd3fc 0%, #7dd3fc 14%,
    #4ade80 14%, #4ade80 40%,
    #facc15 40%, #facc15 60%,
    #f87171 60%, #f87171 100%
  );
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.07);
}
.pcos-bmi-marker {
  position: absolute; top: -2px; width: 4px; height: 26px;
  background: var(--text); border-radius: 2px; transform: translateX(-50%);
  box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.pcos-bmi-zones {
  display: flex; font-size: 0.62rem; color: var(--text-muted);
  justify-content: space-between; margin-top: 0.1rem;
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

    if input.has_lab_results():
        payload["lh"] = float(input.lh())
        payload["fsh"] = float(input.fsh())
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


def _qc_chip_class(score: float) -> str:
    if score >= 0.8:
        return "pcos-qc-chip pcos-qc-chip--high"
    if score >= 0.55:
        return "pcos-qc-chip pcos-qc-chip--med"
    return "pcos-qc-chip pcos-qc-chip--low"


def _qc_dot_class(score: float) -> str:
    if score >= 0.8:
        return "pcos-qc-bar-dot pcos-qc-bar-dot--high"
    if score >= 0.55:
        return "pcos-qc-bar-dot pcos-qc-bar-dot--med"
    return "pcos-qc-bar-dot pcos-qc-bar-dot--low"


def _format_qc_bar(qc_data: dict | None) -> ui.Tag:
    """Compact QC trust bar at the top of results."""
    if not qc_data:
        return ui.div()
    overall = qc_data.get("overall_quality_score", 0)
    model_conf = qc_data.get("model_confidence", 0)
    rag = qc_data.get("rag_evidence_score", 0)
    return ui.div(
        ui.span("Result Quality", class_="pcos-qc-bar-label"),
        ui.span(
            ui.div(class_=_qc_dot_class(overall)),
            f"Overall {overall * 100:.0f}%",
            class_=_qc_chip_class(overall),
        ),
        ui.span(
            ui.div(class_=_qc_dot_class(model_conf)),
            f"Model {model_conf * 100:.0f}%",
            class_=_qc_chip_class(model_conf),
        ),
        ui.span(
            ui.div(class_=_qc_dot_class(rag)),
            f"Evidence {rag * 100:.0f}%",
            class_=_qc_chip_class(rag),
        ),
        class_="pcos-qc-bar",
    )


def _next_steps_cards(tier: str) -> ui.Tag:
    """Contextual 'What to do next' cards based on risk tier."""
    if tier == "high":
        steps = [
            ("Doctor visit", "Schedule an appointment with a gynecologist or endocrinologist to discuss these results."),
            ("Hormone panel", "Ask about AMH, testosterone, and androgen levels to complete your picture."),
            ("Lifestyle factors", "Diet and exercise changes can meaningfully reduce PCOS symptoms — ask your doctor about next steps."),
        ]
    elif tier == "medium":
        steps = [
            ("Track your cycle", "Log your periods for 2-3 months to see if irregularity is consistent."),
            ("GP consultation", "Share this report with your doctor for a full evaluation if symptoms persist."),
            ("Healthy habits", "Balanced nutrition and regular movement support hormonal health regardless of diagnosis."),
        ]
    else:
        steps = [
            ("Keep monitoring", "Continue tracking any symptoms. PCOS can develop at any time."),
            ("Annual check-in", "Discuss reproductive and hormonal health at your next routine exam."),
            ("Stay informed", "Understanding your cycle patterns is always worthwhile."),
        ]
    cards = [
        ui.div(
            ui.p(title, class_="pcos-nextstep-card-title"),
            ui.p(body, class_="pcos-nextstep-card-body"),
            class_="pcos-nextstep-card",
        )
        for title, body in steps
    ]
    return ui.div(
        ui.p("What to do next", class_="pcos-section-label", style="margin-bottom:0.6rem;"),
        ui.div(*cards, class_="pcos-nextsteps"),
    )


def _format_qc_metrics(qc_data: dict | None) -> ui.Tag:
    """
    Render quality control metrics in a professional, patient-friendly format.
    Shows validation scores, confidence, and any raised flags.
    """
    if not qc_data:
        return ui.div()

    overall_score = qc_data.get("overall_quality_score", 0)
    input_score = qc_data.get("input_validation_score", 0)
    model_conf = qc_data.get("model_confidence", 0)
    rag_score = qc_data.get("rag_evidence_score", 0)
    flags = qc_data.get("validation_flags", [])

    # Create metric cards
    metric_cards = [
        ui.div(
            ui.div("Data Quality", class_="pcos-qc-card-label"),
            ui.div(f"{input_score * 100:.0f}%", class_="pcos-qc-card-value"),
            ui.div("Input validation", class_="pcos-qc-card-sub"),
            class_="pcos-qc-card",
        ),
        ui.div(
            ui.div("Prediction", class_="pcos-qc-card-label"),
            ui.div(f"{model_conf * 100:.0f}%", class_="pcos-qc-card-value"),
            ui.div("Model confidence", class_="pcos-qc-card-sub"),
            class_="pcos-qc-card",
        ),
        ui.div(
            ui.div("Evidence", class_="pcos-qc-card-label"),
            ui.div(f"{rag_score * 100:.0f}%", class_="pcos-qc-card-value"),
            ui.div("Clinical backing", class_="pcos-qc-card-sub"),
            class_="pcos-qc-card",
        ),
        ui.div(
            ui.div("Overall Score", class_="pcos-qc-card-label"),
            ui.div(f"{overall_score * 100:.0f}%", class_="pcos-qc-card-value"),
            ui.div("System reliability", class_="pcos-qc-card-sub"),
            class_="pcos-qc-card",
        ),
    ]

    # Format flags
    flag_elements = []
    for flag in flags[:8]:
        status = flag.get("status", "info").lower()
        severity = flag.get("severity", "info").lower()
        
        badge_class = f"pcos-qc-flag-badge pcos-qc-flag-badge--{status}"
        item_class = f"pcos-qc-flag-item pcos-qc-flag-item--{status}"
        
        flag_elements.append(
            ui.div(
                ui.div(
                    flag.get("check_name", "Check"),
                    class_=badge_class,
                ),
                ui.div(
                    flag.get("description", ""),
                    class_="pcos-qc-flag-text",
                ),
                class_=item_class,
            )
        )

    qc_section_children = [
        ui.div(
            ui.span("✓", class_="pcos-qc-icon"),
            "Quality Control Report",
            class_="pcos-qc-title",
        ),
        ui.div(*metric_cards, class_="pcos-qc-grid"),
    ]

    if flag_elements:
        qc_section_children.append(
            ui.div(
                ui.div("Validation Checks:", class_="small" , style="font-weight:600;color:#475569;margin-bottom:0.5rem;"),
                ui.div(*flag_elements, class_="pcos-qc-flags"),
            )
        )

    return ui.div(*qc_section_children, class_="pcos-qc-section")


def _format_flags(
    flags: list[dict],
    *,
    dark: bool = False,
    empty_note: str | None = None,
) -> ui.Tag:
    if empty_note is None:
        empty_note = "No validation flags."
    if not flags:
        return ui.p(ui.em(empty_note), class_="pcos-muted small")
    lines: list[Any] = []
    for f in flags[:12]:
        sev = (f.get("severity") or "warning").lower()
        if sev == "error":
            bullet_class = "pcos-results-bullet--error"
        elif sev == "info":
            bullet_class = "pcos-results-bullet--info"
        else:
            bullet_class = "pcos-results-bullet--warning"
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
                ui.tags.th("Factor"),
                ui.tags.th("Influence"),
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
            t = _sanitize_display(str(p.get("title") or "Research paper"))
            y = p.get("year") or ""
            sub = f"Published {y}" if y else "Clinical research"
        else:
            t = _sanitize_display(str(p.get("title") or "PubMed article"))
            pub = p.get("pubdate", "")
            sub = f"PubMed · {pub}" if pub else "PubMed"
        lis.append(
            ui.tags.li(
                ui.tags.strong(t),
                ui.br(),
                ui.span(sub, class_="paper-sub"),
            )
        )
    return ui.div(
        ui.p(title, class_="pcos-section-label"),
        ui.tags.ul(*lis, class_="pcos-evidence-papers-list"),
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
                "BMI is estimated from your height and weight using the standard formula "
                "(lb ÷ in²) × 703, then converted to metric for the model."
            ),
            ui.p("Healthy range: 18.5–24.9 | Overweight: 25–29.9 | 30+ is higher BMI."),
        ),
        ui.input_numeric("age", "Age (years)", value=28, min=12, max=90),
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
                "If you pick Regular, we'll ask for typical days between period starts (first day of "
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
                "These come from a blood draw ordered by a clinician. "
                "If you haven't been tested, leave this section off — the model "
                "will fill in statistically typical values automatically."
            ),
        ),
        ui.div(
            ui.input_checkbox("has_lab_results", "I have blood test results to enter", value=False),
            ui.panel_conditional(
                "input.has_lab_results",
                ui.div(
                    ui.input_numeric("lh", lbl_lh, value=8.0, min=0, max=200, step=0.1),
                    ui.p("Normal range ~2–15 mIU/mL (follicular phase).", class_="pcos-help pcos-help-tight"),
                    ui.input_numeric("fsh", lbl_fsh, value=5.0, min=0, max=200, step=0.1),
                    ui.p("A high LH:FSH ratio (≥2:1) is one PCOS indicator.", class_="pcos-help pcos-help-tight"),
                    ui.input_numeric("tsh", lbl_tsh, value=2.0, min=0, max=50, step=0.1),
                    ui.p("Normal range ~0.4–4.0 mIU/L. Thyroid issues can mimic PCOS symptoms.", class_="pcos-help pcos-help-tight"),
                    class_="pcos-hormone-block",
                ),
            ),
        ),

        _section_heading(
            "Hair, skin, acne, weight change",
            ui.p(
                "Extra hair: similar in spirit to the Ferriman-Gallwey scale; moderate or severe options "
                'count as "yes" for the screening model.'
            ),
            ui.p(
                'Dark velvety patches: often on neck, underarms, groin, or knuckles. Any option beyond '
                '"none" counts as "yes."'
            ),
            ui.p(
                'Acne: similar to a simple dermatology "how severe overall" scale; moderate or severe counts as "yes."'
            ),
            ui.p(
                "Weight change: focus on gain you didn't plan over the last 6-12 months. Any noticeable "
                'unintended gain counts as "yes."'
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
    ui.tags.head(ui.tags.style(APP_CSS)),
    ui.output_ui("api_status_banner"),
    ui.div(
        ui.div(
            ui.h1("PCOSense", class_="pcos-header-title"),
            ui.p(
                "AI-powered PCOS screening \u2014 fill in what you know, get a personalized risk report backed by clinical evidence.",
                class_="pcos-header-sub",
            ),
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

    @reactive.poll(lambda: int(time.time() // 60), interval_secs=60)
    def api_health_ok() -> bool:
        # In single-service mode the API and UI live in the same process,
        # so we just verify the orchestrator can be obtained.
        try:
            from src.api.main import get_orchestrator
            get_orchestrator()
            return True
        except Exception:
            pass
        # Fallback: try HTTP health check (split-mode deployments)
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
        # Prefer in-process call (works in single-service deployments like
        # Posit Connect, where the API and UI share one uvicorn process).
        try:
            import asyncio
            from fastapi.encoders import jsonable_encoder

            from src.api.main import get_orchestrator, get_qc
            from src.api.schemas import (
                PatientAssessmentRequest,
                patient_dict_from_request,
            )

            def _run_pipeline() -> dict[str, Any]:
                # Same mapping the HTTP endpoint applies: clean form keys
                # (age, bmi, lh, …) → legacy model column names
                # (" Age (yrs)", "BMI", "LH(mIU/mL)", …) the agents expect.
                model_payload = patient_dict_from_request(
                    PatientAssessmentRequest(**payload)
                )
                result = get_orchestrator().run(model_payload)
                qc_metrics = get_qc().create_metrics_report(
                    patient_data=model_payload,
                    prediction_result=result.get("assessment", {}),
                    rag_results=result.get("evidence", {}),
                )
                result["quality_control"] = qc_metrics.to_dict()
                return jsonable_encoder(result)

            return await asyncio.to_thread(_run_pipeline)
        except RuntimeError:
            # Orchestrator not initialised (e.g. running Shiny standalone) —
            # fall through to HTTP path.
            pass

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
                ui.span(class_="pcos-api-dot pcos-api-dot--ok"),
                ui.span("Assessment service online", class_="small text-muted"),
                class_="pcos-api-banner d-flex align-items-center",
            )
        return ui.div(
            ui.span(class_="pcos-api-dot pcos-api-dot--err"),
            ui.span(
                "Assessment service offline — results unavailable. Contact support if this persists.",
                class_="small text-danger",
            ),
            class_="pcos-api-banner d-flex align-items-center",
        )

    @render.ui
    def results_panel() -> ui.Tag:
        fe = form_error()
        if fe:
            return ui.div(ui.h3("Check your inputs"), ui.p(fe), class_="pcos-card")

        st = run_assess.status()
        if st == "initial":
            return ui.div(
                ui.div(ui.p("♡", class_="pcos-placeholder-icon"), class_=""),
                ui.p(
                    "Your personalized PCOS risk report will appear here.",
                    style="font-weight:600;color:#1A1A2E;margin:0 0 0.4rem;font-size:1.05rem;",
                ),
                ui.p(
                    'Fill in the sidebar and press "Run assessment" — most results are ready in about a minute.',
                    style="margin:0;",
                ),
                class_="pcos-results-placeholder",
            )

        if st == "running":
            return ui.div(
                ui.h3("Assessment in progress", style="font-size:1rem;font-weight:700;color:#1A1A2E;margin:0 0 0.35rem;"),
                ui.p("This takes about 60 seconds. Keep this tab open.", class_="pcos-muted small mb-0"),
                ui.div(
                    ui.div(
                        ui.div("1", class_="pcos-step-num"),
                        ui.span("Validating your data"),
                        class_="pcos-step pcos-step--active",
                    ),
                    ui.div(
                        ui.div("2", class_="pcos-step-num"),
                        ui.span("Searching clinical evidence"),
                        class_="pcos-step",
                    ),
                    ui.div(
                        ui.div("3", class_="pcos-step-num"),
                        ui.span("Computing risk score + explanations"),
                        class_="pcos-step",
                    ),
                    class_="pcos-steps",
                ),
                class_="pcos-card",
            )

        if st == "error":
            err = run_assess.error()
            err_str = str(err)
            friendly = err_str
            if "422" in err_str or "validation" in err_str.lower():
                friendly = "Some inputs look out of range. Please check your values and try again."
            elif "503" in err_str or "FileNotFoundError" in err_str:
                friendly = "The assessment model isn't available right now. Please try again shortly."
            elif "500" in err_str:
                friendly = "An internal error occurred during the assessment. Please try again."
            return ui.div(
                ui.h3("Something went wrong", style="font-size:1rem;font-weight:700;margin:0 0 0.4rem;"),
                ui.p(friendly, class_="pcos-muted"),
                class_="pcos-card",
            )

        if st == "cancelled":
            return ui.div(ui.p("Assessment cancelled.", class_="pcos-muted"), class_="pcos-card")

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
        qc = data.get("quality_control") or {}

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

        next_steps = _next_steps_cards(tier)

        body_children: list[Any] = [
            # ── QC trust bar at top ──────────────────────────────────────
            _format_qc_bar(qc) if qc else ui.div(),

            # ── Risk hero ────────────────────────────────────────────────
            ui.div(
                hero_inner[0],
                hero_inner[1],
                class_=f"pcos-results-hero pcos-results-hero--{tier}",
            ),

            # ── Risk % metric ────────────────────────────────────────────
            ui.div(
                ui.div(
                    ui.p("PCOS Risk Score", class_="pcos-results-metric-label"),
                    ui.div(
                        f"{pct}%" if pct is not None else "-",
                        class_=_metric_prob_class(tier),
                    ),
                    ui.p(
                        "XGBoost model — not a clinical diagnosis.",
                        style="font-size:0.68rem;color:#94a3b8;margin:0.2rem 0 0;",
                    ),
                    class_="pcos-results-metric",
                ),
                class_="pcos-results-metrics",
            ),

            # ── What's driving the score ─────────────────────────────────
            ui.div(
                ui.p("What's influencing your score", class_="pcos-section-label"),
                _factors_table(top_factors),
                class_="pcos-factors-panel",
            ),

            # ── Recommendation ───────────────────────────────────────────
            ui.div(
                ui.h3("Recommendation"),
                (
                    ui.p(_sanitize_display(str(rec)), style="line-height:1.55;margin:0;font-size:0.9rem;")
                    if rec
                    else ui.p(ui.em("Recommendation unavailable (AI service may be offline)."), class_="pcos-muted small mb-0")
                ),
                class_="pcos-rec-card",
            ),

            # ── Next steps ───────────────────────────────────────────────
            next_steps,

            # ── Supporting evidence ──────────────────────────────────────
            ui.div(
                ui.p("Supporting clinical evidence", class_="pcos-section-label"),
                ui.div(
                    _papers_list("Research library", e.get("retrieved_papers") or [], "chroma"),
                    _papers_list("Latest PubMed research", e.get("pubmed_papers") or [], "pubmed"),
                    class_="pcos-results-evidence-stack",
                ),
                (
                    ui.div(
                        ui.p("Clinical summary", class_="pcos-section-label", style="margin-top:0.85rem;"),
                        ui.p(_sanitize_display(str(summary)), style="font-size:0.84rem;line-height:1.5;color:#374151;margin:0;")
                        if summary
                        else ui.p(ui.em("Clinical summary unavailable (AI service may be offline)."), class_="pcos-results-muted small"),
                    )
                    if summary or criteria
                    else ui.div()
                ),
                class_="pcos-evidence-panel",
            ),

            # ── Validation warnings ──────────────────────────────────────
            (
                ui.div(
                    ui.p("Data validation notes", class_="pcos-section-label"),
                    _format_flags(v.get("flags") or [], dark=False, empty_note="All checks passed."),
                    class_="pcos-card--light-panel",
                )
                if v.get("flags")
                else ui.div()
            ),

            # ── Full QC report ───────────────────────────────────────────
            _format_qc_metrics(qc) if qc else ui.div(),

            # ── Footnote ─────────────────────────────────────────────────
            ui.p(
                _sanitize_display(
                    f"Assessment completed in {meta.get('elapsed_sec', '?')}s. "
                    "This tool is for informational purposes only and is not a substitute for clinical judgement."
                ),
                class_="pcos-results-footnote",
            ),
        ]

        return ui.div(*body_children, class_="pcos-results-root")


app = App(app_ui, server)
