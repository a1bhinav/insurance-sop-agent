"""Provider HTTP contracts, exercised against an in-process httpx mock transport."""
from __future__ import annotations

import json

import httpx
import pytest

from sop_agent.llm.provider import AnthropicProvider, LLMConfig, LLMError, OpenAICompatibleProvider


def _mock(provider, handler):
    provider._client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


def test_anthropic_request_shape_and_structured_output():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"type": "message", "stop_reason": "end_turn",
                                         "content": [{"type": "text", "text": "{\"a\": 1}"}]})

    p = AnthropicProvider(LLMConfig(provider="anthropic", api_key="sk-test", model="claude-sonnet-5-5", extractor_model="claude-haiku-5-5"))
    _mock(p, handler)
    out = p.complete("SYS", [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "yo"}],
                     model="claude-haiku-5-5", json_mode=True, json_schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False})
    assert out == '{"a": 1}'
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-test" and seen["headers"]["anthropic-version"] == "2023-06-01"
    body = seen["body"]
    assert body["model"] == "claude-haiku-5-5" and body["system"] == "SYS"
    assert body["messages"][0]["role"] == "user" and body["messages"][-1]["role"] == "user"   # alternation fixed up
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in body                      # deprecated on current Claude models (400 if sent)
    assert not any(m["role"] == "assistant" and m["content"] == "{" for m in body["messages"])   # no prefill


def test_anthropic_falls_back_when_structured_output_rejected_then_remembers():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append("output_config" in body)
        if "output_config" in body:
            return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "output_config: Extra inputs are not permitted"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "{\"ok\": true}"}], "stop_reason": "end_turn"})

    p = AnthropicProvider(LLMConfig(provider="anthropic", api_key="k", model="old-model", extractor_model="old-model"))
    _mock(p, handler)
    assert p.complete("s", [{"role": "user", "content": "u"}], json_schema={"type": "object"}) == '{"ok": true}'
    assert calls == [True, False]
    p.complete("s", [{"role": "user", "content": "u"}], json_schema={"type": "object"})
    assert calls == [True, False, False]          # schema not retried for a model that rejected it


def test_anthropic_errors_are_raised_as_llm_error():
    p = AnthropicProvider(LLMConfig(provider="anthropic", api_key="bad", model="m", extractor_model="m"))
    _mock(p, lambda req: httpx.Response(401, json={"error": {"message": "invalid x-api-key"}}))
    with pytest.raises(LLMError) as exc:
        p.complete("s", [{"role": "user", "content": "u"}])
    assert "401" in str(exc.value)


def test_openai_compatible_cascade_and_base_url():
    formats = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        fmt = (body.get("response_format") or {}).get("type")
        formats.append(fmt)
        assert request.headers["authorization"] == "Bearer k"
        assert str(request.url) == "http://localhost:11434/v1/chat/completions"
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "unsupported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"x\": 2}"}}]})

    p = OpenAICompatibleProvider(LLMConfig(provider="openai", api_key="k", model="llama", extractor_model="llama", base_url="http://localhost:11434/v1"))
    _mock(p, handler)
    assert p.complete("s", [{"role": "user", "content": "u"}], json_mode=True, json_schema={"type": "object"}) == '{"x": 2}'
    assert formats == ["json_schema", "json_object"]
    p.complete("s", [{"role": "user", "content": "u"}])
    assert formats[-1] is None
