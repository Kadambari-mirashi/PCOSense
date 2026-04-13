# PCOSense: Multi-Agent System for Polycystic Ovary Syndrome Detection

> AI-powered screening tool for Polycystic Ovary Syndrome (PCOS) using a multi-agent architecture, local LLM, and explainable ML.

---

## Overview

PCOSense is a full-stack clinical decision-support application that helps identify early PCOS risk from patient biomarkers. It combines:

- **XGBoost ML model** trained on 541 labeled patient records — **AUROC 0.9528**
- **SHAP explainability** — shows which biomarkers drive each individual prediction
- **Multi-agent AI system** — 3 specialised agents orchestrated sequentially
- **RAG knowledge base** — 27 clinical papers in ChromaDB for evidence retrieval
- **Ollama + Llama 3.2** (local LLM via Ollama) — zero API cost, runs entirely on your machine
- **PubMed API + NHANES data** — latest research and population-level context
- **Supabase** — patient session storage, predictions, and audit trail

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Python Shiny (Phase 3) |
| Backend | FastAPI + Uvicorn (Phase 3) |
| ML Model | XGBoost + SHAP |
| LLM | Ollama · Llama 3.2 (local) |
| Vector DB | ChromaDB (RAG — 27 clinical papers) |
| Database | Supabase (PostgreSQL) |
| APIs | PubMed E-Utilities · NHANES population data |
| Data | Kaggle PCOS Dataset (541 records, 42 features) |

---

## System Architecture

```mermaid
flowchart LR
    DATA[("PCOS Dataset\n541 patients")]:::data

    subgraph TRAIN["Phase 1 · ML Pipeline ✅"]
        direction LR
        EDA["EDA\nNotebook 01"]:::nb
        FEAT["Feature Eng.\nNotebook 02"]:::nb
        MODEL["XGBoost\nAUROC 0.95"]:::model
        WRAP["ML Wrapper\n+ SHAP"]:::nb
    end

    subgraph AGENTS["Phase 2 · Multi-Agent System ✅"]
        direction TB
        A1["Agent 1\nData Validator"]:::agent
        A2["Agent 2\nEvidence Retriever"]:::agent
        A3["Agent 3\nRisk Assessor"]:::agent
        A1 --> A2 --> A3
    end

    subgraph TOOLS["Tools & Data Sources"]
        direction TB
        LLM["Ollama Llama 3.2\n(local LLM)"]:::llm
        RAG["ChromaDB\n27 papers"]:::rag
        PUB["PubMed API"]:::api
        NHS["NHANES Data"]:::api
        DB["Supabase\nPostgreSQL"]:::db
    end

    UI["Shiny UI\nPhase 3 ✅"]:::ui

    DATA --> EDA --> FEAT --> MODEL --> WRAP
    WRAP --> A1
    A2 --> LLM
    A2 --> RAG
    A2 --> PUB
    A3 --> NHS
    A3 --> DB
    A3 --> UI

    classDef data  fill:#4C78A8,stroke:#2c5282,color:#fff
    classDef nb    fill:#48bb78,stroke:#276749,color:#fff
    classDef model fill:#E45756,stroke:#c53030,color:#fff,font-weight:bold
    classDef agent fill:#f6ad55,stroke:#c05621,color:#1a1a1a
    classDef llm   fill:#9f7aea,stroke:#6b46c1,color:#fff
    classDef rag   fill:#68b0ab,stroke:#4a908a,color:#fff
    classDef api   fill:#667eea,stroke:#434190,color:#fff
    classDef db    fill:#fc8181,stroke:#c53030,color:#fff
    classDef ui    fill:#fc8181,stroke:#c53030,color:#fff
```

> See [`docs/multi_agent_architecture.md`](docs/multi_agent_architecture.md) for detailed Mermaid diagrams of the agent workflow, tool calling map, and RAG pipeline.

---

## Multi-Agent Workflow

```
Patient Data (from user input)
    │
    ▼
Agent 1: Data Validator
    ├─ Tool: Ollama Llama 3.2 (consistency reasoning)
    ├─ Programmatic range & completeness checks
    └─ Output: {status, validated_data, flags, confidence_score}
    │
    ▼
Agent 2: Clinical Evidence Retriever
    ├─ Tool 1: ChromaDB (search 27 local clinical papers)
    ├─ Tool 2: PubMed API (fetch latest PCOS research)
    ├─ Tool 3: Ollama Embeddings (vectorise queries)
    ├─ Tool 4: Ollama Llama 3.2 (synthesise evidence)
    └─ Output: {papers[], clinical_summary, diagnostic_criteria}
    │
    ▼
Agent 3: Risk Assessor
    ├─ Tool 1: XGBoost model (predict risk 0–1)
    ├─ Tool 2: SHAP TreeExplainer (feature contributions)
    ├─ Tool 3: NHANES data (population percentiles)
    ├─ Tool 4: Ollama Llama 3.2 (synthesise recommendation)
    └─ Output: {risk_score, top_factors[], recommendation}
    │
    ▼
Supabase (persist results + audit trail)
```

---

## ML Model Results

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Classifier |
| Dataset | 541 patients · 45 raw features |
| Engineered features | 42 (incl. LH/FSH ratio, follicle composites) |
| Training samples | 432 (80% stratified split) |
| Test samples | 109 (20% stratified split) |
| **AUROC** | **0.9528** |
| Explainability | SHAP TreeExplainer |
| Threshold | Youden's J statistic (0.2212) |
| Inference time | < 50 ms per patient |

**Top predictive features (by SHAP):**
- Morphological: follicle count L/R, follicle total
- Symptoms: hair growth, weight gain, skin darkening, pimples
- Hormonal: LH/FSH ratio, TSH
- Cycle: regularity, length
- Metabolic: BMI, RBS

---

## Project Structure

```
PCOSense/
├── notebooks/
│   ├── 01_eda.ipynb                  ✅ EDA — distributions, correlations
│   ├── 02_features.ipynb             ✅ Feature engineering — 42 features
│   ├── 03_xgboost_training.ipynb     ✅ Training — AUROC 0.9528
│   └── 04_rag_setup.ipynb            ✅ ChromaDB knowledge base (27 papers)
├── src/
│   ├── __init__.py
│   ├── ml_model.py                   ✅ XGBoost prediction + SHAP explainer
│   ├── ollama_client.py              ✅ Ollama LLM wrapper (generate + embed)
│   ├── rag_system.py                 ✅ Chroma retrieval + Ollama synthesis
│   ├── agents.py                     ✅ 3 agents + orchestrator
│   ├── data_fetcher.py               ✅ PubMed API + NHANES baselines
│   ├── database.py                   ✅ Supabase client (patients, predictions, audit)
│   ├── api/                          ✅ FastAPI (Phase 3)
│   │   ├── main.py                   ✅ /api/v1/assess, health, feature-info
│   │   └── schemas.py                ✅ Form JSON → model feature keys
│   └── app/                          ✅ Shiny UI (Phase 3)
│       └── app.py                    ✅ Teal-themed form + results
├── models/
│   ├── pcos_model.json               ✅ Trained XGBoost model
│   └── model_metadata.json           ✅ AUROC, features, SHAP rankings
├── data/
│   ├── raw/                          ✅ PCOS_data_without_infertility.xlsx
│   └── processed/
│       ├── features_processed.pkl    ✅ Train/test splits + scaler + imputer
│       └── eda_meta.json             ✅ EDA summary metadata
├── knowledge_base/
│   └── chroma_db/                    ✅ ChromaDB vector database
├── docs/
│   └── multi_agent_architecture.md   ✅ Mermaid agent workflow diagrams
├── .env.example                      ✅ Environment variable template
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.12
- [Ollama](https://ollama.ai/download) installed and running
- [Supabase](https://supabase.com) project (free tier)
- [Kaggle API token](https://www.kaggle.com/settings) (for initial data download only)

### 1 — Clone & set up environment

```bash
git clone <repo-url>
cd PCOSense

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2 — Install Ollama & pull models

```bash
# Install Ollama from https://ollama.ai/download, then:
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3 — Configure environment variables

```bash
cp .env.example .env
# Edit .env with your Supabase URL and key
```

### 4 — Set up Supabase tables

Run the following SQL in [Supabase SQL Editor](https://supabase.com/dashboard):

```sql
CREATE TABLE IF NOT EXISTS patients (
    id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at    TIMESTAMPTZ DEFAULT now(),
    age           REAL,
    bmi           REAL,
    blood_group   TEXT,
    cycle_regular BOOLEAN,
    symptoms      JSONB DEFAULT '{}',
    hormones      JSONB DEFAULT '{}',
    raw_input     JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    patient_id      UUID REFERENCES patients(id),
    created_at      TIMESTAMPTZ DEFAULT now(),
    risk_score      REAL NOT NULL,
    risk_label      TEXT NOT NULL,
    confidence      REAL,
    top_factors     JSONB DEFAULT '[]',
    clinical_summary TEXT,
    recommendation  TEXT,
    model_version   TEXT DEFAULT 'xgboost-v1',
    agent_outputs   JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now(),
    patient_id  UUID REFERENCES patients(id),
    event       TEXT NOT NULL,
    details     JSONB DEFAULT '{}'
);
```

### 5 — Configure Kaggle & train the model (if needed)

```bash
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Run notebooks 01–03 in order (the trained model is committed to the repo — this step is only needed if you want to retrain).

### 6 — Build the knowledge base

```bash
jupyter notebook notebooks/04_rag_setup.ipynb
# Run all cells → creates knowledge_base/chroma_db/
```

### 7 — Verify everything works

```bash
# Test ML model
python src/ml_model.py

# Test Ollama client
python src/ollama_client.py

# Test data fetchers (PubMed + NHANES)
python src/data_fetcher.py

# Test full multi-agent pipeline
python src/agents.py
```

### 8 — Run the API (FastAPI)

From the project root (with your virtual environment activated):

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Assess (example): `POST /api/v1/assess` with JSON body using keys such as `age`, `bmi`, `cycle_ri` (1=regular, 2=irregular), `lh`, `fsh`, `tsh`, `hair_growth`, `skin_darkening`, `pimples`, `weight_gain`, `follicle_l`, `follicle_r`, `cycle_length_days`. You can also send native model column names (e.g. `" Age (yrs)"`) as extra keys.

If `SUPABASE_URL` and `SUPABASE_KEY` are set in `.env`, completed assessments are written to Supabase automatically (same behaviour as `PCOSOrchestrator(db=SupabaseClient())`).

### 9 — Run the Shiny frontend

In a **second** terminal (API must be running unless you change `PCOSENSE_API_URL`):

```bash
shiny run src/app/app.py --reload --host 127.0.0.1 --port 3838
```

Then open [http://127.0.0.1:3838](http://127.0.0.1:3838). Set `PCOSENSE_API_URL` in `.env` if the API is not on port 8000.

---

## Development Phases

- [x] **Phase 1** — XGBoost ML model (AUROC 0.9528) + SHAP explainability
- [x] **Phase 2** — Multi-agent system + RAG (ChromaDB) + Ollama + PubMed/NHANES + Supabase
- [x] **Phase 3** — Python Shiny frontend + FastAPI backend + local deployment (API + UI)

---

## Important Notes

- `kaggle.json` is **never committed** — only needed locally to download the raw data
- `.env` is **never committed** — contains Supabase credentials
- `models/pcos_model.json` is committed — the app loads this at runtime, no retraining needed
- All LLM inference runs locally via Ollama — **zero API costs**
- PubMed and NHANES APIs are public and free (no authentication required)
- The knowledge base contains 27 curated clinical papers on PCOS diagnosis
- The app is designed to run fully offline after initial setup (except PubMed queries)
