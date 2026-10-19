"""The LLM plumbing (with a fake provider — no network) and the HTTP API."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from sop_agent.llm.extractor import LLMExtractor
from sop_agent.llm.json_utils import parse_json_object
from sop_agent.llm.provider import LLMConfig, LLMError, LLMProvider, _coalesce
from sop_agent.llm.responder import LLMResponder
from sop_agent.sop.controller import SOPController
from sop_agent.sop.state import SessionState


class FakeProvider(LLMProvider):
    """Returns canned output and records the prompts it was given."""

    name = "fake"

    def __init__(self, replies):
        self.cfg = LLMConfig(provider="anthropic", api_key="x", model="fake", extractor_model="fake")
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages, *, model=None, max_tokens=700, temperature=0.3, json_mode=False, json_schema=None):
        self.calls.append({"system": system, "messages": messages, "json_mode": json_mode, "json_schema": json_schema})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_parse_json_object_is_forgiving():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": [1, 2,], }') == {"a": [1, 2]}
    assert parse_json_object("no json here") is None


def test_coalesce_alternates_roles_for_anthropic():
    msgs = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    out = _coalesce(msgs)
    assert out[0]["role"] == "user" and out[-1]["content"] == "a\n\nb"


def test_llm_extractor_parses_model_json_and_cross_checks(settings):
    model_json = json.dumps({
        "pii": {"full_name": "Margaret Chen", "dob": "March 15 1985", "id_last4": "4472"},
        "caller_role": "policyholder",
        "intent": "denial_question",
        "case_hints": {"case_type": "healthcare", "status": "denied", "time_reference": "January"},
        "scope": "in_scope", "emotion": {"label": "neutral", "intensity": 0},
    })
    provider = FakeProvider([model_json])
    ex = LLMExtractor(provider, settings, today="2026-02-10")
    sig = ex.extract("My name is Margaret Chen, policy POL-9921, born March 15 1985, last four 4472, about my denied healthcare claim from January", SessionState())
    assert sig.source == "llm"
    assert sig.pii["dob"] == "1985-03-15"                # normalised
    assert sig.pii["policy_number"] == "POL-9921"       # filled in from the rule cross-check
    assert sig.case_hints["month"] == 1                 # derived from time_reference
    assert provider.calls[0]["json_mode"] is True
    assert provider.calls[0]["json_schema"]["required"]            # structured-output schema was requested
    assert "PENDING_QUESTION" in provider.calls[0]["messages"][0]["content"]


def test_llm_extractor_falls_back_to_rules_on_failure(settings):
    provider = FakeProvider([LLMError("boom")])
    ex = LLMExtractor(provider, settings, today="2026-02-10")
    sig = ex.extract("My name is Margaret Chen", SessionState())
    assert sig.source == "fallback" and sig.pii["full_name"] == "Margaret Chen"


def test_llm_responder_prompt_is_phase_constrained_and_falls_back(settings, harness):
    provider = FakeProvider(["Thanks Margaret — I still need two more details.", LLMError("down")])
    responder = LLMResponder(provider, settings)
    ctrl = SOPController(harness.store, harness.controller().extractor, responder, harness.outbox, harness.consent, settings)
    state = SessionState()
    ctrl.opening_message(state)
    res = ctrl.handle_turn(state, "Hi, I'm Margaret Chen, calling about my denied claim")
    system = provider.calls[-1]["system"]
    assert "STRICT" in system and "DIRECTIVE" in system and "MUST NOT" in system
    assert "cl-2048" not in system.lower() and "pathology" not in system.lower()   # no claim facts pre-verification
    assert res.reply.startswith("Thanks Margaret")
    # second turn: provider fails -> template fallback keeps the conversation going
    res2 = ctrl.handle_turn(state, "DOB 1985-03-15, last four 4472")
    assert state.verification.verified and res2.reply


def test_llm_email_composition_uses_model_json_or_template(settings, harness):
    provider = FakeProvider([json.dumps({"subject": "Your call summary", "body": "Hello Margaret,\nCL-2048 ..."}), "not json"])
    responder = LLMResponder(provider, settings)
    state = SessionState()
    ctx = {"company": "X", "agent_name": "Sam", "caller_first_name": "Margaret", "topics_covered": [], "claims": [], "transcript": []}
    assert responder.compose_email(state, ctx) == ("Your call summary", "Hello Margaret,\nCL-2048 ...")
    subject, body = responder.compose_email(state, ctx)      # bad JSON -> template
    assert "summary" in subject.lower() and "Hello Margaret" in body


# ---------------------------------------------------------------- HTTP API
def test_api_full_demo_flow():
    from sop_agent.api import app as app_module

    client = TestClient(app_module.app)
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["llm"]["provider"] == "mock"
    assert [p["name"] for p in health["phases"]][:4] == ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"]

    created = client.post("/api/sessions", json={"consent_scenario": "default"}).json()
    sid = created["session_id"]
    assert created["state"]["phase"] == "VERIFY_ID" and "verify" in created["greeting"].lower()

    r = client.post(f"/api/sessions/{sid}/messages", json={"message": "Why was my claim denied?"}).json()
    assert r["state"]["phase"] == "VERIFY_ID"
    assert "CL-2048" not in r["reply"]

    r = client.post(f"/api/sessions/{sid}/messages", json={
        "message": "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."}).json()
    assert r["state"]["phase"] == "PROCESS_CASE" and r["state"]["selected_case_id"] == "CL-2048"
    assert r["state"]["verification"]["verified"] is True
    assert r["state"]["verification"]["provided"]["full_name"] == "••••"      # PII never shipped to the browser
    assert "CL-2048" in r["reply"]

    r = client.post(f"/api/sessions/{sid}/messages", json={"message": "that's all"}).json()
    assert r["state"]["phase"] == "POST_PROCESS"
    r = client.post(f"/api/sessions/{sid}/messages", json={"message": "yes"}).json()
    assert r["state"]["phase"] == "CLOSED" and len(r["emails"]) == 1
    assert r["emails"][0]["to"] == "margaret@email.com"

    detail = client.get(f"/api/sessions/{sid}").json()
    assert len(detail["history"]) >= 9 and len(detail["traces"]) == 4
    assert client.get("/api/scenarios").json()["scenarios"]
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions", json={"consent_scenario": "bogus"}).status_code == 400
    assert client.get("/").status_code == 200
