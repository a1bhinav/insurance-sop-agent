from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests drive the SOP with the deterministic rule-based extractor and templates
# so they run without a model API key. A fixed "today" keeps deadlines meaningful.
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("SOP_DEMO_DATE", "2026-02-10")
os.environ.setdefault("OUTBOX_DIR", str(ROOT / "data" / "test_outbox"))

import pytest  # noqa: E402

from sop_agent.config import Settings  # noqa: E402
from sop_agent.harness import Harness  # noqa: E402
from sop_agent.sop.state import SessionState  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def harness(settings) -> Harness:
    return Harness(settings)


class Convo:
    """Small helper: drive one session turn by turn and keep the replies."""

    def __init__(self, harness: Harness, consent_scenario: str = "default", **controller_kwargs):
        self.harness = harness
        self.ctrl = harness.controller(**controller_kwargs)
        self.state = SessionState(consent_scenario=consent_scenario)
        self.greeting = self.ctrl.opening_message(self.state)
        self.results = []

    def say(self, text: str):
        res = self.ctrl.handle_turn(self.state, text)
        self.results.append(res)
        return res

    @property
    def phase(self) -> str:
        return self.state.phase.value

    @property
    def last_reply(self) -> str:
        return self.results[-1].reply if self.results else self.greeting

    @property
    def last_events(self) -> list[str]:
        return self.results[-1].plan.events + self.results[-1].guard_events


@pytest.fixture()
def convo(harness) -> Convo:
    return Convo(harness)


@pytest.fixture()
def make_convo(harness):
    def _make(**kw) -> Convo:
        return Convo(harness, **kw)
    return _make


# Literals that must never appear before verification (from claims.json)
CLAIM_SECRETS = [
    "CL-2048", "CL-2011", "CL-1899", "CL-2102", "CL-3001",
    "pathology report", "office note", "diagnosis report",
    "1450", "780", "425", "3200", "2026-03-18", "March 18",
    "denied because", "appeal deadline",
]


def assert_no_claim_leak(text: str) -> None:
    low = text.lower()
    for s in CLAIM_SECRETS:
        assert s.lower() not in low, f"claim detail leaked before verification: {s!r} in {text!r}"
