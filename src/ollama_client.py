"""
ollama_client.py — Local Ollama LLM wrapper for PCOSense
=========================================================
Provides:
  - OllamaClient : generate text, structured JSON, and embeddings
                    via a local Ollama instance (zero API cost)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaClient:
    """
    Thin wrapper around the Ollama HTTP API.

    Parameters
    ----------
    host        : Ollama server URL (default from OLLAMA_HOST env var)
    model       : Chat / generation model name
    embed_model : Embedding model name
    timeout     : Request timeout in seconds
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", _DEFAULT_HOST)).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    # ── connection helpers ──────────────────────────────────────────────────

    def _check_connection(self) -> None:
        """Raise ``ConnectionError`` if Ollama is unreachable."""
        try:
            r = self._client.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.host}.\n"
                "  1. Install Ollama  → https://ollama.ai/download\n"
                "  2. Start the app   → `ollama serve`\n"
                f"  3. Pull the model  → `ollama pull {self.model}`\n"
                f"  (underlying error: {exc})"
            ) from exc

    def is_available(self) -> bool:
        """Return *True* if the Ollama server is reachable."""
        try:
            self._check_connection()
            return True
        except ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """Return names of locally-available models."""
        r = self._client.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    # ── text generation ─────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
    ) -> str:
        """
        Generate free-form text from *prompt*.

        Returns the model's response string.
        """
        self._check_connection()
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        r = self._client.post(f"{self.host}/api/generate", json=payload)
        r.raise_for_status()
        return r.json()["response"]

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Generate a response and parse it as JSON.

        Uses Ollama's ``format: "json"`` mode for reliable structured output.
        Falls back to extracting the first JSON object from the response text.
        """
        self._check_connection()
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        r = self._client.post(f"{self.host}/api/generate", json=payload)
        r.raise_for_status()
        text = r.json()["response"]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            log.error("Ollama returned non-JSON: %s", text[:200])
            return {"raw_response": text}

    # ── embeddings ──────────────────────────────────────────────────────────

    def embed(self, text: str | list[str]) -> list[list[float]]:
        """
        Return embedding vectors for *text* (single string or list).

        Uses the configured embed model (default ``nomic-embed-text``).
        """
        self._check_connection()
        if isinstance(text, str):
            text = [text]

        payload = {"model": self.embed_model, "input": text}
        r = self._client.post(f"{self.host}/api/embed", json=payload)
        r.raise_for_status()
        return r.json()["embeddings"]

    # ── context manager ─────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = OllamaClient()

    if not client.is_available():
        print("Ollama is not running. Please start it first.")
        raise SystemExit(1)

    print(f"Ollama is running at {client.host}")
    print(f"Available models: {client.list_models()}")

    resp = client.generate("Say 'hello' in one word.", temperature=0.0)
    print(f"\nGenerate test: {resp.strip()}")

    emb = client.embed("PCOS diagnosis criteria")
    print(f"Embedding test: dim={len(emb[0])}, first 5 values={emb[0][:5]}")

    print("\nOllama client smoke-test passed.")
