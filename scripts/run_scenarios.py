#!/usr/bin/env python3
"""Run scripted callers through the SOP harness and print annotated transcripts.

    python scripts/run_scenarios.py                 # all scenarios, provider from env (mock if no key)
    python scripts/run_scenarios.py demo frustrated # selected scenarios
    LLM_PROVIDER=mock python scripts/run_scenarios.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sop_agent.config import settings  # noqa: E402
from sop_agent.harness import Harness  # noqa: E402
from sop_agent.sop.state import SessionState  # noqa: E402

from sop_agent.scenarios import SCENARIOS  # noqa: E402


def run(name: str, spec: dict, harness: Harness) -> None:
    ctrl = harness.controller()
    state = SessionState(consent_scenario=spec.get("consent_scenario", "default"))
    print("=" * 100)
    print(f"[{name}] {spec['title']}  (provider={ctrl.llm_config.provider}, model={ctrl.llm_config.model})")
    print("=" * 100)
    print(f"AGENT [{state.phase.value}]: {ctrl.opening_message(state)}\n")
    for turn in spec["turns"]:
        before = state.phase.value
        res = ctrl.handle_turn(state, turn)
        print(f"CALLER: {turn}")
        for ev in res.plan.events + res.guard_events:
            print(f"    · {ev}")
        print(f"AGENT [{before} -> {state.phase.value}]: {res.reply}\n")
    if state.email_sent_id:
        email = next(e for e in harness.outbox.sent if e.message_id == state.email_sent_id)
        print(f"--- EMAIL SENT to {email.to}: {email.subject}\n{email.body}\n")


def main() -> None:
    wanted = [a for a in sys.argv[1:] if a in SCENARIOS] or list(SCENARIOS)
    harness = Harness(settings)
    for name in wanted:
        run(name, SCENARIOS[name], harness)


if __name__ == "__main__":
    main()
