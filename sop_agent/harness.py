"""Wiring: build a controller for a given LLM configuration and manage sessions."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from .config import Settings, settings as default_settings
from .data.store import DataStore
from .data.tools import ConsentService, EmailOutbox
from .llm.provider import LLMConfig, LLMProvider, build_provider
from .llm.responder import build_components
from .sop.controller import SOPController, TurnResult
from .sop.state import SessionState


class Harness:
    """One process-wide instance: shared fixtures, outbox and consent service.
    Controllers are built per LLM configuration (so a browser-supplied API key
    gets its own provider without touching the server default)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.store = DataStore(self.settings.fixtures_dir)
        self.outbox = EmailOutbox(self.settings.outbox_dir)
        self.consent = ConsentService(self.store)
        self._controllers: dict[tuple[str, Optional[str], str], SOPController] = {}
        self._lock = threading.Lock()

    def today(self) -> str:
        return self.settings.demo_date or date.today().isoformat()

    def controller(self, api_key: Optional[str] = None, provider: Optional[str] = None,
                   model: Optional[str] = None) -> SOPController:
        cfg = LLMConfig.from_settings(self.settings, api_key=api_key, provider=provider, model=model)
        key = (cfg.provider, cfg.api_key, cfg.model)
        with self._lock:
            ctrl = self._controllers.get(key)
            if ctrl is None:
                llm: LLMProvider = build_provider(cfg)
                extractor, responder = build_components(llm, self.settings, self.today())
                ctrl = SOPController(self.store, extractor, responder, self.outbox, self.consent, self.settings)
                ctrl.llm_config = cfg  # type: ignore[attr-defined]
                self._controllers[key] = ctrl
            return ctrl

    def default_llm_config(self) -> LLMConfig:
        return LLMConfig.from_settings(self.settings)


@dataclass
class SessionEntry:
    state: SessionState
    last_seen: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


class SessionRegistry:
    def __init__(self, ttl_seconds: int = 6 * 3600):
        self.ttl = ttl_seconds
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    def create(self, consent_scenario: str = "default") -> SessionState:
        state = SessionState(consent_scenario=consent_scenario)
        with self._lock:
            self._sessions[state.session_id] = SessionEntry(state=state)
            self._sweep()
        return state

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry:
                entry.last_seen = time.time()
                return entry.state
            return None

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _sweep(self) -> None:
        now = time.time()
        stale = [sid for sid, e in self._sessions.items() if now - e.last_seen > self.ttl]
        for sid in stale:
            self._sessions.pop(sid, None)


def run_script(turns: list[str], consent_scenario: str = "default", settings: Settings | None = None,
               api_key: Optional[str] = None, provider: Optional[str] = None) -> list[TurnResult]:
    """Convenience for tests and demos: run a scripted caller through a fresh session."""
    harness = Harness(settings)
    ctrl = harness.controller(api_key=api_key, provider=provider)
    state = SessionState(consent_scenario=consent_scenario)
    ctrl.opening_message(state)
    return [ctrl.handle_turn(state, t) for t in turns]
