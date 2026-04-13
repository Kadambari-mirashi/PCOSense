"""
ollama_client.py — LLM wrapper for PCOSense
=============================================
Provides:
  - OllamaClient : generate text, structured JSON, and embeddings.

Provider selection (checked once at init):
  1. If OPENAI_API_KEY is set  → OpenAI  (gpt-4o-mini, cloud-ready)
  2. Else if local Ollama runs → Ollama  (local, zero API cost)
  3. Else                      → is_available() returns False
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
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OllamaClient:
    """
    Unified LLM wrapper. Uses OpenAI when OPENAI_API_KEY is set,
    otherwise falls back to a local Ollama instance.

    The public interface (is_available, generate, generate_json, embed)
    is identical regardless of backend, so callers need no changes.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        # OpenAI backend (preferred for deployment)
        self._openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._openai_model = os.getenv("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
        self._openai_client: Any = None

        if self._openai_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self._openai_key)
                log.info("LLM backend: OpenAI (%s)", self._openai_model)
            except Exception as exc:
                log.warning("OpenAI SDK init failed, falling back to Ollama: %s", exc)
                self._openai_client = None

        # Ollama backend (local fallback)
        self.host = (host or os.getenv("OLLAMA_HOST", _DEFAULT_HOST)).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
        self.timeout = timeout
        self._http = httpx.Client(timeout=self.timeout)

        if not self._openai_client:
            log.info("LLM backend: Ollama (%s @ %s)", self.model, self.host)

    @property
    def _use_openai(self) -> bool:
        return self._openai_client is not None

    # ── connection helpers ──────────────────────────────────────────────────

    def _check_ollama(self) -> None:
        try:
            r = self._http.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.host}. "
                f"(underlying error: {exc})"
            ) from exc

    def is_available(self) -> bool:
        """Return True if any LLM backend is reachable."""
        if self._use_openai:
            return True
        try:
            self._check_ollama()
            return True
        except ConnectionError:
            return False

    def list_models(self) -> list[str]:
        if self._use_openai:
            return [self._openai_model]
        r = self._http.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    # ── text generation ─────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
    ) -> str:
        if self._use_openai:
            return self._openai_generate(prompt, system_prompt, temperature)
        return self._ollama_generate(prompt, system_prompt, temperature)

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        if self._use_openai:
            return self._openai_generate_json(prompt, system_prompt, temperature)
        return self._ollama_generate_json(prompt, system_prompt, temperature)

    # ── OpenAI implementations ──────────────────────────────────────────────

    def _openai_generate(
        self, prompt: str, system_prompt: str, temperature: float,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self._openai_client.chat.completions.create(
            model=self._openai_model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def _openai_generate_json(
        self, prompt: str, system_prompt: str, temperature: float,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self._openai_client.chat.completions.create(
            model=self._openai_model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.error("OpenAI returned non-JSON: %s", text[:300])
            return {"raw_response": text}

    # ── Ollama implementations ──────────────────────────────────────────────

    def _ollama_generate(
        self, prompt: str, system_prompt: str, temperature: float,
    ) -> str:
        self._check_ollama()
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        r = self._http.post(f"{self.host}/api/generate", json=payload)
        r.raise_for_status()
        return r.json()["response"]

    def _ollama_generate_json(
        self, prompt: str, system_prompt: str, temperature: float,
    ) -> dict[str, Any]:
        self._check_ollama()
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        r = self._http.post(f"{self.host}/api/generate", json=payload)
        r.raise_for_status()
        text = r.json()["response"]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
            try:
                cleaned = text[start:end].replace("\n", " ").replace("\r", " ").replace("\t", " ")
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        log.error("Ollama returned non-JSON: %s", text[:300])
        return {"raw_response": text}

    # ── embeddings ──────────────────────────────────────────────────────────

    def embed(self, text: str | list[str]) -> list[list[float]]:
        if self._use_openai:
            return self._openai_embed(text)
        return self._ollama_embed(text)

    def _openai_embed(self, text: str | list[str]) -> list[list[float]]:
        if isinstance(text, str):
            text = [text]
        resp = self._openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return [item.embedding for item in resp.data]

    def _ollama_embed(self, text: str | list[str]) -> list[list[float]]:
        self._check_ollama()
        if isinstance(text, str):
            text = [text]
        payload = {"model": self.embed_model, "input": text}
        r = self._http.post(f"{self.host}/api/embed", json=payload)
        r.raise_for_status()
        return r.json()["embeddings"]

    # ── context manager ─────────────────────────────────────────────────────

    def close(self) -> None:
        self._http.close()

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
        print("No LLM backend available (no OpenAI key, Ollama not running).")
        raise SystemExit(1)

    backend = "OpenAI" if client._use_openai else "Ollama"
    print(f"Backend: {backend}")
    print(f"Available models: {client.list_models()}")

    resp = client.generate("Say 'hello' in one word.", temperature=0.0)
    print(f"\nGenerate test: {resp.strip()}")

    jresp = client.generate_json(
        'Return JSON: {"greeting": "hello"}',
        system_prompt="Respond only with valid JSON.",
    )
    print(f"JSON test: {jresp}")

    print("\nSmoke-test passed.")
