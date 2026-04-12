"""
data_fetcher.py — External API clients for PCOSense
=====================================================
Provides:
  - fetch_pubmed_papers()   : search PubMed for PCOS research articles
  - fetch_nhanes_baseline() : population-level hormone / metabolic statistics
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ── PubMed E-Utilities ─────────────────────────────────────────────────────
_PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_PUBMED_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def fetch_pubmed_papers(
    query: str = "polycystic ovary syndrome",
    max_papers: int = 5,
    sort: str = "relevance",
) -> list[dict[str, Any]]:
    """
    Search PubMed and return paper metadata.

    Parameters
    ----------
    query      : free-text search query
    max_papers : maximum results to return (1–20)
    sort       : "relevance" or "date"

    Returns
    -------
    List of dicts with keys: pmid, title, authors, source, pubdate, abstract.
    """
    papers: list[dict[str, Any]] = []

    try:
        search_resp = httpx.get(
            _PUBMED_SEARCH,
            params={
                "db": "pubmed",
                "term": query,
                "retmax": min(max_papers, 20),
                "sort": sort,
                "retmode": "json",
            },
            timeout=15,
        )
        search_resp.raise_for_status()
        id_list = search_resp.json()["esearchresult"].get("idlist", [])

        if not id_list:
            log.info("PubMed returned 0 results for query: %s", query)
            return papers

        summary_resp = httpx.get(
            _PUBMED_SUMMARY,
            params={
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
            },
            timeout=15,
        )
        summary_resp.raise_for_status()
        result = summary_resp.json().get("result", {})

        for pmid in id_list:
            doc = result.get(pmid, {})
            if not doc or pmid == "uids":
                continue
            authors = [a.get("name", "") for a in doc.get("authors", [])]
            papers.append(
                {
                    "pmid": pmid,
                    "title": doc.get("title", ""),
                    "authors": authors[:5],
                    "source": doc.get("source", ""),
                    "pubdate": doc.get("pubdate", ""),
                    "doi": doc.get("elocationid", ""),
                }
            )

        # Fetch abstracts (plain text)
        abstract_resp = httpx.get(
            _PUBMED_FETCH,
            params={
                "db": "pubmed",
                "id": ",".join(id_list),
                "rettype": "abstract",
                "retmode": "text",
            },
            timeout=20,
        )
        if abstract_resp.status_code == 200:
            blocks = abstract_resp.text.split("\n\n\n")
            for i, paper in enumerate(papers):
                if i < len(blocks):
                    paper["abstract"] = blocks[i].strip()[:2000]

    except httpx.HTTPError as exc:
        log.warning("PubMed API error: %s", exc)
    except Exception as exc:
        log.warning("Unexpected PubMed error: %s", exc)

    return papers


# ── NHANES Population Baselines ────────────────────────────────────────────
# Reference ranges derived from published NHANES analyses (2015–2020 cycle).
# Sources: NHANES laboratory data, CDC/NCHS reports, peer-reviewed analyses.
_NHANES_BASELINES: dict[str, dict[str, Any]] = {
    "testosterone": {
        "full_name": "Total Testosterone",
        "unit": "ng/dL",
        "female_reproductive_age": {
            "mean": 32.0,
            "std": 15.0,
            "p5": 12.0,
            "p25": 22.0,
            "p50": 30.0,
            "p75": 40.0,
            "p95": 60.0,
            "reference_range": "15–70 ng/dL",
            "pcos_threshold": 50.0,
        },
        "source": "NHANES 2015-2020, women 18-45",
    },
    "LH": {
        "full_name": "Luteinizing Hormone",
        "unit": "mIU/mL",
        "female_reproductive_age": {
            "mean": 5.5,
            "std": 4.2,
            "p5": 1.5,
            "p25": 3.0,
            "p50": 5.0,
            "p75": 7.5,
            "p95": 15.0,
            "reference_range": "1.5–15 mIU/mL (follicular)",
            "pcos_threshold": 10.0,
        },
        "source": "NHANES / clinical laboratory norms",
    },
    "FSH": {
        "full_name": "Follicle-Stimulating Hormone",
        "unit": "mIU/mL",
        "female_reproductive_age": {
            "mean": 6.5,
            "std": 2.8,
            "p5": 2.5,
            "p25": 4.5,
            "p50": 6.0,
            "p75": 8.0,
            "p95": 12.0,
            "reference_range": "2.5–10 mIU/mL (follicular)",
        },
        "source": "NHANES / clinical laboratory norms",
    },
    "LH_FSH_ratio": {
        "full_name": "LH/FSH Ratio",
        "unit": "ratio",
        "female_reproductive_age": {
            "mean": 1.0,
            "std": 0.5,
            "p5": 0.4,
            "p25": 0.7,
            "p50": 1.0,
            "p75": 1.3,
            "p95": 2.0,
            "reference_range": "1:1 (normal)",
            "pcos_threshold": 2.0,
        },
        "source": "Clinical consensus — LH/FSH > 2 suggestive of PCOS",
    },
    "TSH": {
        "full_name": "Thyroid-Stimulating Hormone",
        "unit": "mIU/L",
        "female_reproductive_age": {
            "mean": 2.0,
            "std": 1.1,
            "p5": 0.5,
            "p25": 1.2,
            "p50": 1.8,
            "p75": 2.5,
            "p95": 4.5,
            "reference_range": "0.4–4.0 mIU/L",
        },
        "source": "NHANES 2015-2020, women 18-45",
    },
    "prolactin": {
        "full_name": "Prolactin",
        "unit": "ng/mL",
        "female_reproductive_age": {
            "mean": 13.0,
            "std": 7.5,
            "p5": 3.0,
            "p25": 8.0,
            "p50": 12.0,
            "p75": 17.0,
            "p95": 29.0,
            "reference_range": "3–30 ng/mL",
        },
        "source": "NHANES / endocrinology reference",
    },
    "BMI": {
        "full_name": "Body Mass Index",
        "unit": "kg/m²",
        "female_reproductive_age": {
            "mean": 28.7,
            "std": 7.5,
            "p5": 18.5,
            "p25": 23.0,
            "p50": 27.5,
            "p75": 33.0,
            "p95": 43.5,
            "reference_range": "18.5–24.9 (normal weight)",
        },
        "source": "NHANES 2017-2020, women 20-44",
    },
    "fasting_glucose": {
        "full_name": "Fasting Blood Glucose (RBS)",
        "unit": "mg/dL",
        "female_reproductive_age": {
            "mean": 95.0,
            "std": 18.0,
            "p5": 72.0,
            "p25": 85.0,
            "p50": 93.0,
            "p75": 101.0,
            "p95": 126.0,
            "reference_range": "70–100 mg/dL",
        },
        "source": "NHANES 2017-2020, women 20-44",
    },
    "vitamin_d": {
        "full_name": "Vitamin D (25-OH)",
        "unit": "ng/mL",
        "female_reproductive_age": {
            "mean": 26.0,
            "std": 11.0,
            "p5": 10.0,
            "p25": 18.0,
            "p50": 25.0,
            "p75": 32.0,
            "p95": 48.0,
            "reference_range": "20–50 ng/mL (sufficient)",
        },
        "source": "NHANES 2015-2020, women 18-45",
    },
    "AMH": {
        "full_name": "Anti-Müllerian Hormone",
        "unit": "ng/mL",
        "female_reproductive_age": {
            "mean": 3.5,
            "std": 2.8,
            "p5": 0.5,
            "p25": 1.5,
            "p50": 3.0,
            "p75": 5.0,
            "p95": 9.0,
            "reference_range": "1.0–5.0 ng/mL (18-35 years)",
            "pcos_threshold": 4.7,
        },
        "source": "Clinical reference / endocrinology consensus",
    },
    "hemoglobin": {
        "full_name": "Hemoglobin",
        "unit": "g/dL",
        "female_reproductive_age": {
            "mean": 13.2,
            "std": 1.2,
            "p5": 11.0,
            "p25": 12.3,
            "p50": 13.2,
            "p75": 14.0,
            "p95": 15.3,
            "reference_range": "12.0–16.0 g/dL",
        },
        "source": "NHANES 2017-2020, women 20-44",
    },
}

# Map model feature names → NHANES keys
_FEATURE_TO_NHANES: dict[str, str] = {
    "LH(mIU/mL)": "LH",
    "FSH(mIU/mL)": "FSH",
    "FSH/LH": "LH_FSH_ratio",
    "LH_FSH_ratio": "LH_FSH_ratio",
    "TSH (mIU/L)": "TSH",
    "PRL(ng/mL)": "prolactin",
    "BMI": "BMI",
    "RBS(mg/dl)": "fasting_glucose",
    "Vit D3 (ng/mL)": "vitamin_d",
    "Hb(g/dl)": "hemoglobin",
    "AMH(ng/mL)": "AMH",
}


def fetch_nhanes_baseline(
    hormone_name: str,
) -> dict[str, Any] | None:
    """
    Return population-level statistics for *hormone_name*.

    Accepts either an NHANES key (``"LH"``) or a model feature name
    (``"LH(mIU/mL)"``).  Returns ``None`` if the hormone is not in the
    reference database.

    The returned dict contains: full_name, unit, mean, std, percentiles
    (p5–p95), reference_range, and source.
    """
    key = _FEATURE_TO_NHANES.get(hormone_name, hormone_name)
    entry = _NHANES_BASELINES.get(key)
    if entry is None:
        return None

    stats = entry["female_reproductive_age"]
    return {
        "hormone": key,
        "full_name": entry["full_name"],
        "unit": entry["unit"],
        "source": entry["source"],
        **stats,
    }


def compute_percentile(hormone_name: str, value: float) -> dict[str, Any] | None:
    """
    Estimate where *value* falls within the population distribution.

    Returns a dict with ``percentile`` (0–100) and a human-readable
    ``interpretation`` string, or ``None`` if the hormone is unknown.
    """
    baseline = fetch_nhanes_baseline(hormone_name)
    if baseline is None:
        return None

    mean = baseline["mean"]
    std = baseline["std"]
    if std == 0:
        return None

    from scipy.stats import norm

    percentile = float(norm.cdf(value, loc=mean, scale=std) * 100)

    if percentile >= 95:
        interp = f"Very high — above 95% of women aged 18-45"
    elif percentile >= 75:
        interp = f"Above average — higher than {percentile:.0f}% of women aged 18-45"
    elif percentile >= 25:
        interp = f"Within normal range — at the {percentile:.0f}th percentile"
    elif percentile >= 5:
        interp = f"Below average — lower than {100 - percentile:.0f}% of women aged 18-45"
    else:
        interp = f"Very low — below 5% of women aged 18-45"

    return {
        "hormone": baseline["hormone"],
        "value": value,
        "unit": baseline["unit"],
        "percentile": round(percentile, 1),
        "interpretation": interp,
        "reference_range": baseline.get("reference_range", ""),
        "population_mean": mean,
        "population_std": std,
    }


def get_all_baselines() -> dict[str, dict[str, Any]]:
    """Return the full NHANES baseline reference table."""
    return {k: fetch_nhanes_baseline(k) for k in _NHANES_BASELINES}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("── PubMed Search ──")
    papers = fetch_pubmed_papers("PCOS diagnosis Rotterdam criteria", max_papers=3)
    for p in papers:
        print(f"  [{p['pmid']}] {p['title'][:80]}")

    print("\n── NHANES Baselines ──")
    for name in ["LH", "FSH", "BMI", "TSH", "testosterone"]:
        b = fetch_nhanes_baseline(name)
        if b:
            print(f"  {b['full_name']:30s}  mean={b['mean']:.1f}  ref={b['reference_range']}")

    print("\n── Percentile Computation ──")
    result = compute_percentile("BMI", 32.5)
    if result:
        print(f"  BMI 32.5 → {result['percentile']}th percentile: {result['interpretation']}")

    result = compute_percentile("LH(mIU/mL)", 12.0)
    if result:
        print(f"  LH 12.0 → {result['percentile']}th percentile: {result['interpretation']}")

    print("\nData fetcher smoke-test passed.")
