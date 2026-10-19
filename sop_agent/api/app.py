"""FastAPI application: JSON API for the SOP agent plus the static test UI.

Auth token for the model: set LLM_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY)
in the environment, or let the browser send one per request in the
`X-LLM-API-Key` header (enabled by ALLOW_CLIENT_API_KEY, default true).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import settings
from ..harness import Harness, SessionRegistry
from ..scenarios import SCENARIOS
from ..sop.state import PHASE_FREEDOM, Phase

log = logging.getLogger("sop_agent.api")
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"

app = FastAPI(title="Insurance Claims SOP Agent", version=__version__,
              description="SOP-guided conversational agent harness: VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

harness = Harness(settings)
sessions = SessionRegistry()


# ---------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    consent_scenario: str = Field(default="default", description="Key from consent_scenarios.json (default | timeout)")


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


def _controller(api_key: Optional[str], provider: Optional[str], model: Optional[str]):
    if api_key and not settings.allow_client_api_key:
        api_key = None
    return harness.controller(api_key=api_key or None, provider=provider or None, model=model or None)


def _history(state) -> list[dict[str, Any]]:
    return [{"role": t.role, "text": t.text, "phase": t.phase, "at": t.at} for t in state.history]


def _emails(state) -> list[dict[str, Any]]:
    return [e.to_dict() for e in harness.outbox.for_session(state.session_id)]


# ---------------------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------------------
@app.get("/api/health")
def health(x_llm_api_key: Optional[str] = Header(default=None), x_llm_provider: Optional[str] = Header(default=None),
           x_llm_model: Optional[str] = Header(default=None)) -> dict[str, Any]:
    ctrl = _controller(x_llm_api_key, x_llm_provider, x_llm_model)
    cfg = ctrl.llm_config
    return {
        "status": "ok",
        "version": __version__,
        "llm": cfg.public(),
        "client_api_key_allowed": settings.allow_client_api_key,
        "phases": [{"name": p.value, "freedom": PHASE_FREEDOM[p]} for p in Phase],
        "consent_scenarios": harness.consent.scenarios(),
        "thresholds": {
            "min_pii_factors": settings.min_pii_factors,
            "rep_min_pii_factors": settings.rep_min_pii_factors,
            "max_verify_attempts": settings.max_verify_attempts,
            "off_topic_limit": settings.off_topic_limit,
            "refusal_limit": settings.refusal_limit,
            "consent_max_checks": settings.consent_max_checks,
        },
        "demo_date": harness.today(),
        "agent": {"name": settings.agent_name, "company": settings.company_name},
    }


@app.get("/api/scenarios")
def scenarios() -> dict[str, Any]:
    return {"scenarios": [{"key": k, **v} for k, v in SCENARIOS.items()]}


@app.post("/api/sessions")
def create_session(body: CreateSessionRequest, x_llm_api_key: Optional[str] = Header(default=None),
                   x_llm_provider: Optional[str] = Header(default=None), x_llm_model: Optional[str] = Header(default=None)) -> dict[str, Any]:
    if body.consent_scenario not in harness.consent.scenarios():
        raise HTTPException(400, f"unknown consent scenario {body.consent_scenario!r}")
    ctrl = _controller(x_llm_api_key, x_llm_provider, x_llm_model)
    state = sessions.create(consent_scenario=body.consent_scenario)
    greeting = ctrl.opening_message(state)
    return {"session_id": state.session_id, "greeting": greeting, "state": state.to_public_dict(),
            "llm": ctrl.llm_config.public()}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "session not found")
    return {"state": state.to_public_dict(), "history": _history(state), "traces": state.traces, "emails": _emails(state)}


@app.post("/api/sessions/{session_id}/messages")
def post_message(session_id: str, body: MessageRequest, x_llm_api_key: Optional[str] = Header(default=None),
                 x_llm_provider: Optional[str] = Header(default=None), x_llm_model: Optional[str] = Header(default=None)) -> dict[str, Any]:
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "session not found")
    ctrl = _controller(x_llm_api_key, x_llm_provider, x_llm_model)
    try:
        result = ctrl.handle_turn(state, body.message)
    except Exception as exc:  # noqa: BLE001 - never leave the caller without a reply
        log.exception("turn failed")
        raise HTTPException(500, f"turn failed: {exc}") from exc
    return {
        "reply": result.reply,
        "state": state.to_public_dict(),
        "trace": result.trace(),
        "emails": _emails(state),
        "llm": ctrl.llm_config.public(),
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    return {"deleted": sessions.delete(session_id)}


@app.get("/api/sessions/{session_id}/emails")
def session_emails(session_id: str) -> dict[str, Any]:
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "session not found")
    return {"emails": _emails(state)}


# ---------------------------------------------------------------------------------------
# static UI
# ---------------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index() -> Any:
    index_file = UI_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "UI not found; API is live at /docs"})
    return FileResponse(index_file)


if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
