"""
rag_system.py — Retrieval-Augmented Generation for PCOSense
============================================================
Provides:
  - RAGSystem : search the Chroma clinical-papers collection and use
                Ollama to synthesise evidence into clinical context
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chromadb

from src.ollama_client import OllamaClient

log = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
_CHROMA_DIR = _ROOT_DIR / "knowledge_base" / "chroma_db"
_COLLECTION_NAME = "clinical_papers"

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a clinical researcher specialising in Polycystic Ovary Syndrome (PCOS). "
    "Given a set of medical research excerpts and a patient query, synthesise the "
    "evidence into a concise, clinically relevant summary. Cite specific papers when "
    "possible. Focus on diagnostic criteria, hormone thresholds, and clinical "
    "significance. Be precise and evidence-based."
)


class RAGSystem:
    """
    Retrieval-Augmented Generation backed by ChromaDB + Ollama.

    The Chroma database is expected to be pre-built by
    ``notebooks/04_rag_setup.ipynb``.

    Parameters
    ----------
    chroma_dir      : path to the persistent Chroma storage
    collection_name : name of the clinical-papers collection
    ollama          : an ``OllamaClient`` instance (created if ``None``)
    """

    def __init__(
        self,
        chroma_dir: str | Path | None = None,
        collection_name: str = _COLLECTION_NAME,
        ollama: OllamaClient | None = None,
    ) -> None:
        self.chroma_dir = Path(chroma_dir) if chroma_dir else _CHROMA_DIR
        self.collection_name = collection_name
        self.ollama = ollama or OllamaClient()
        self._collection: chromadb.Collection | None = None

    # ── lazy collection handle ──────────────────────────────────────────────

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            if not self.chroma_dir.exists():
                raise FileNotFoundError(
                    f"Chroma database not found at {self.chroma_dir}.\n"
                    "Run notebooks/04_rag_setup.ipynb first to build the knowledge base."
                )
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = client.get_collection(self.collection_name)
            log.info(
                "Loaded Chroma collection '%s' (%d documents)",
                self.collection_name,
                self._collection.count(),
            )
        return self._collection

    # ── retrieval ───────────────────────────────────────────────────────────

    def retrieve_papers(
        self,
        query_text: str,
        n_results: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Semantic-search the Chroma collection for *query_text*.

        Returns up to *n_results* papers, each as a dict with:
        ``document``, ``metadata``, ``distance``.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

        papers: list[dict[str, Any]] = []
        if not results["documents"]:
            return papers

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            papers.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": round(dist, 4),
                }
            )
        return papers

    # ── synthesis ───────────────────────────────────────────────────────────

    def synthesize_evidence(
        self,
        query: str,
        papers: list[dict[str, Any]] | None = None,
        n_results: int = 3,
    ) -> dict[str, Any]:
        """
        Retrieve papers and synthesise them into a clinical summary.

        If *papers* is ``None`` the method retrieves them automatically.

        Returns
        -------
        dict with keys: ``query``, ``papers``, ``clinical_summary``.
        """
        if papers is None:
            papers = self.retrieve_papers(query, n_results=n_results)

        if not papers:
            return {
                "query": query,
                "papers": [],
                "clinical_summary": "No relevant clinical papers found in the knowledge base.",
            }

        evidence_block = "\n\n".join(
            f"--- Paper: {p['metadata'].get('title', 'Unknown')} "
            f"({p['metadata'].get('year', 'n.d.')}) ---\n{p['document']}"
            for p in papers
        )

        prompt = (
            f"QUERY: {query}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_block}\n\n"
            "Based on the evidence above, provide a concise clinical summary "
            "addressing the query. Reference specific papers where possible."
        )

        summary = self.ollama.generate(
            prompt=prompt,
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            temperature=0.2,
        )

        return {
            "query": query,
            "papers": [
                {
                    "title": p["metadata"].get("title", "Unknown"),
                    "year": p["metadata"].get("year", ""),
                    "source": p["metadata"].get("source", ""),
                    "relevance_distance": p["distance"],
                    "excerpt": p["document"][:300] + "…" if len(p["document"]) > 300 else p["document"],
                }
                for p in papers
            ],
            "clinical_summary": summary.strip(),
        }

    # ── convenience ─────────────────────────────────────────────────────────

    def paper_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rag = RAGSystem()

    print(f"Chroma collection: {rag.collection_name}")
    print(f"Documents loaded : {rag.paper_count()}")

    test_queries = [
        "elevated testosterone and LH in women",
        "Rotterdam criteria for PCOS diagnosis",
        "insulin resistance metabolic syndrome",
    ]

    for q in test_queries:
        print(f"\n── Query: {q}")
        papers = rag.retrieve_papers(q, n_results=2)
        for p in papers:
            title = p["metadata"].get("title", "?")
            print(f"   [{p['distance']:.4f}] {title}")
