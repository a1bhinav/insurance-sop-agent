"""Thin, dependency-light LLM providers over httpx.

* AnthropicProvider        – Anthropic Messages API (structured outputs via output_config.format)
* OpenAICompatibleProvider – OpenAI chat completions (also Azure/OpenRouter/Ollama/vLLM via LLM_BASE_URL)
* MockProvider             – no network; tells the harness to use its rule-based extractor and templates

Only `complete()` is used by the rest of the system, so adding a provider is a
few lines. The API token is read from the environment or passed per request by
the UI (header X-LLM-API-Key) — see api/app.py.

Structured output strategy: when a JSON schema is supplied we ask the API to
enforce it; if the endpoint rejects that (older model, proxy, third-party
server) we retry with a looser mode and finally with plain text, and the caller
parses leniently. Assistant-message prefill is deliberately not used — it is
incompatible with JSON output on current Claude models.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from ..config import DEFAULT_MODELS, Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str
    api_key: Optional[str]
    model: str
    extractor_model: str
    base_url: Optional[str] = None
    timeout: float = 60.0

    @classmethod
    def from_settings(cls, s: Settings, api_key: Optional[str] = None, provider: Optional[str] = None,
                      model: Optional[str] = None) -> "LLMConfig":
        prov = (provider or s.llm_provider or "anthropic").lower()
        key = api_key or s.llm_api_key
        mdl = model or (s.llm_model if prov == s.llm_provider else None) or DEFAULT_MODELS.get(prov, DEFAULT_MODELS["anthropic"])
        ext = (s.llm_extractor_model if prov == s.llm_provider and s.llm_extractor_model and not model else mdl)
        if prov != "mock" and not key:
            log.warning("No API key for provider %s - falling back to mock mode", prov)
            prov, mdl, ext = "mock", "mock", "mock"
        return cls(provider=prov, api_key=key, model=mdl, extractor_model=ext, base_url=s.llm_base_url, timeout=s.llm_timeout_seconds)

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def public(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "extractor_model": self.extractor_model,
                "has_key": bool(self.api_key) or self.is_mock}


class LLMProvider:
    name = "base"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout)
        self._schema_unsupported: set[str] = set()   # models that rejected schema enforcement

    def complete(self, system: str, messages: list[dict[str, str]], *, model: Optional[str] = None,
                 max_tokens: int = 700, temperature: float = 0.3, json_mode: bool = False,
                 json_schema: Optional[dict[str, Any]] = None) -> str:
        raise NotImplementedError

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass

    def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    VERSION = "2023-06-01"

    def complete(self, system, messages, *, model=None, max_tokens=700, temperature=0.3, json_mode=False, json_schema=None):
        url = (self.cfg.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        mdl = model or self.cfg.model
        headers = {"x-api-key": self.cfg.api_key or "", "anthropic-version": self.VERSION, "content-type": "application/json"}
        # `temperature` is deprecated on current Claude models (the API rejects it with a 400), so it is not sent.
        body: dict[str, Any] = {
            "model": mdl,
            "max_tokens": max_tokens,
            "system": system,
            "messages": _coalesce(messages),
        }
        use_schema = bool(json_schema) and mdl not in self._schema_unsupported
        if use_schema:
            body["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        resp = self._post(url, headers, body)
        if resp.status_code == 400 and use_schema:
            # Older model or proxy without structured outputs: fall back to plain JSON-in-text.
            log.info("Anthropic structured outputs rejected for %s; retrying without schema (%s)", mdl, resp.text[:160])
            self._schema_unsupported.add(mdl)
            body.pop("output_config", None)
            resp = self._post(url, headers, body)
        if resp.status_code >= 400:
            raise LLMError(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        usage = data.get("usage") or {}
        log.debug("anthropic %s stop=%s in=%s out=%s thinking=%s", mdl, data.get("stop_reason"), usage.get("input_tokens"),
                  usage.get("output_tokens"), (usage.get("output_tokens_details") or {}).get("thinking_tokens"))
        if data.get("stop_reason") == "refusal":
            raise LLMError("Anthropic API returned a refusal stop reason")
        if data.get("stop_reason") == "max_tokens":
            log.warning("anthropic reply hit max_tokens=%s (output_tokens=%s); consider raising the limit", max_tokens, usage.get("output_tokens"))
        return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def complete(self, system, messages, *, model=None, max_tokens=700, temperature=0.3, json_mode=False, json_schema=None):
        url = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        mdl = model or self.cfg.model
        headers = {"Authorization": f"Bearer {self.cfg.api_key or ''}", "content-type": "application/json"}
        body: dict[str, Any] = {
            "model": mdl,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}] + _coalesce(messages),
        }
        attempts: list[Optional[dict[str, Any]]] = []
        if json_schema and mdl not in self._schema_unsupported:
            attempts.append({"type": "json_schema", "json_schema": {"name": "turn", "strict": True, "schema": json_schema}})
        if json_mode or json_schema:
            attempts.append({"type": "json_object"})
        attempts.append(None)
        resp = None
        for fmt in attempts:
            if fmt is None:
                body.pop("response_format", None)
            else:
                body["response_format"] = fmt
            resp = self._post(url, headers, body)
            if resp.status_code == 400 and "temperature" in resp.text and "temperature" in body:
                # newer OpenAI models reject non-default temperature; retry once without it
                body.pop("temperature", None)
                resp = self._post(url, headers, body)
            if resp.status_code == 400 and fmt is not None:
                if fmt["type"] == "json_schema":
                    self._schema_unsupported.add(mdl)
                log.info("response_format %s rejected by %s; retrying looser (%s)", fmt["type"], mdl, resp.text[:160])
                continue
            break
        assert resp is not None
        if resp.status_code >= 400:
            raise LLMError(f"OpenAI-compatible API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {json.dumps(data)[:300]}") from exc


class MockProvider(LLMProvider):
    """Never called for text; its presence tells the harness to use rule-based components."""
    name = "mock"

    def complete(self, system, messages, *, model=None, max_tokens=700, temperature=0.3, json_mode=False, json_schema=None):
        raise LLMError("mock provider has no completion endpoint")


def _coalesce(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Anthropic requires strictly alternating roles starting with 'user';
    merge consecutive same-role messages and drop leading assistant turns."""
    out: list[dict[str, str]] = []
    for m in messages:
        role = "assistant" if m.get("role") == "assistant" else "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if not out and role == "assistant":
            out.append({"role": "user", "content": "(conversation start)"})
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    if not out:
        out.append({"role": "user", "content": "(no message)"})
    return out


def build_provider(cfg: LLMConfig) -> LLMProvider:
    if cfg.provider == "anthropic":
        return AnthropicProvider(cfg)
    if cfg.provider in ("openai", "openai-compatible", "azure", "openrouter", "ollama"):
        return OpenAICompatibleProvider(cfg)
    return MockProvider(cfg)
