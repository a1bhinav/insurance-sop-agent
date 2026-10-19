"""Session state for the SOP harness: phases, memory, verification bookkeeping
and the Plan object handed from the controller to the responder."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from .signals import LOCATOR_FIELDS, PII_FIELDS, TurnSignals


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    CLOSED = "CLOSED"            # terminal: conversation wrapped up
    HUMAN_HANDOFF = "HUMAN_HANDOFF"  # terminal: escalated to a human representative


# How much freedom the LLM gets in each phase. Shown in the UI and injected into prompts.
PHASE_FREEDOM = {
    Phase.VERIFY_ID: "strict",
    Phase.RESOLVE_INTENT: "guided",
    Phase.PROCESS_CASE: "flexible",
    Phase.POST_PROCESS: "guided",
    Phase.CLOSED: "strict",
    Phase.HUMAN_HANDOFF: "strict",
}

# Questions the agent may leave hanging; the controller interprets yes/no answers against these.
PENDING_CONSENT_WAIT = "consent_wait"
PENDING_CONFIRM_CLAIM = "confirm_claim"
PENDING_CHOOSE_CLAIM = "choose_claim"
PENDING_ANYTHING_ELSE = "anything_else"
PENDING_EMAIL_OFFER = "email_offer"
PENDING_HUMAN_OFFER = "human_offer"
PENDING_REP_PATH = "rep_path_offer"


@dataclass
class Memory:
    """Cross-phase memory. Anything useful the caller says is stored here the
    moment they say it, even if the SOP is not ready to act on it yet."""

    intent_hints: list[str] = field(default_factory=list)
    case_hints: dict[str, Any] = field(default_factory=dict)
    pending_questions: list[str] = field(default_factory=list)   # asked before the phase could answer them
    notes: list[str] = field(default_factory=list)
    email_preference: Optional[str] = None
    hint_acknowledged: bool = False      # did we tell the caller "I've noted that, we'll get to it"?

    def absorb(self, sig: TurnSignals, hold_questions: bool = True) -> list[str]:
        """Store what the caller said.

        `hold_questions` is True while the SOP cannot answer yet (pre-verification):
        questions are parked and answered later, and claim hints *accumulate* because
        the caller may describe their claim across several turns. Afterwards, a fresh
        description replaces the old one (the caller is pointing at a claim now).
        """
        events: list[str] = []
        if sig.intent and sig.intent not in ("other",) and sig.intent not in self.intent_hints:
            self.intent_hints.append(sig.intent)
            events.append(f"intent_hint:{sig.intent}")
        if sig.has_case_hints and not hold_questions:
            self.case_hints = {}
        for k, v in sig.case_hints.items():
            if v and self.case_hints.get(k) != v:
                self.case_hints[k] = v
                events.append(f"case_hint:{k}={v}")
        if hold_questions:
            for q in sig.questions:
                if q not in self.pending_questions:
                    self.pending_questions.append(q)
                    events.append(f"question_held_for_later:{q[:60]!r}")
        for n in sig.notes:
            if n not in self.notes:
                self.notes.append(n)
                events.append("note_saved")
        if sig.email_preference and self.email_preference != sig.email_preference:
            self.email_preference = sig.email_preference
            events.append(f"email_preference:{sig.email_preference}")
        return events

    @property
    def primary_intent(self) -> Optional[str]:
        return self.intent_hints[-1] if self.intent_hints else None

    def clear_case_hints(self) -> None:
        self.case_hints = {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConsentState:
    consent_id: str
    status: str = "pending"
    checks: int = 0
    scenario: str = "default"


@dataclass
class VerificationState:
    verified: bool = False
    caller_role: str = "policyholder"             # policyholder | representative
    candidate_party_id: Optional[str] = None     # record we are verifying against (never spoken)
    provided: dict[str, str] = field(default_factory=dict)   # field -> value the caller gave
    matched: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    attempts: int = 0                             # failed attempts (mismatches / unknown records)
    locked: bool = False                          # too many failures: only a human can continue
    refusals: int = 0
    why_explained: bool = False
    # representative path
    rep_name: Optional[str] = None
    rep_relationship: Optional[str] = None
    rep_on_file: Optional[bool] = None
    consent: Optional[ConsentState] = None
    verified_at: Optional[str] = None

    def matched_count(self) -> int:
        return len([m for m in self.matched if m in PII_FIELDS])

    def remaining_fields(self) -> list[str]:
        return [f for f in PII_FIELDS if f not in self.matched and f not in self.mismatched]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Never ship raw PII values to the browser; the UI only needs which fields arrived.
        d["provided"] = {k: ("•" * 4 if k in PII_FIELDS else v) for k, v in self.provided.items()}
        d["matched_count"] = self.matched_count()
        return d


@dataclass
class TurnRecord:
    role: str   # user | agent
    text: str
    phase: str
    at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Plan:
    """What the responder must accomplish this turn.

    directive: natural-language instructions (the SOP's voice) for the LLM
    facts: the ONLY grounded facts the LLM may use (whitelisted by the controller)
    must_not: hard prohibitions for this turn
    tone: empathy / de-escalation guidance derived from detected emotion
    template: key for the deterministic mock responder
    """

    phase: Phase
    directive: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    must_not: list[str] = field(default_factory=list)
    tone: Optional[str] = None
    template: str = "generic"
    template_args: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    pre_verification: bool = True     # controls the leak guard
    allow_claim_details: bool = False

    def add(self, *lines: str) -> None:
        for line in lines:
            if line and line not in self.directive:
                self.directive.append(line)

    def event(self, text: str) -> None:
        self.events.append(text)


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: Phase = Phase.VERIFY_ID
    verification: VerificationState = field(default_factory=VerificationState)
    memory: Memory = field(default_factory=Memory)
    selected_case_id: Optional[str] = None
    candidate_claims: list[dict[str, Any]] = field(default_factory=list)   # briefs offered in a "which claim?" question
    claims_discussed: list[str] = field(default_factory=list)
    topics_covered: list[str] = field(default_factory=list)
    pending_question: Optional[str] = None
    off_topic_strikes: int = 0
    off_topic_total: int = 0
    human_offered: bool = False
    negative_streak: int = 0
    last_emotion: str = "neutral"
    consent_scenario: str = "default"
    email_sent_id: Optional[str] = None
    handoff_reference: Optional[str] = None
    handoff_reason: Optional[str] = None
    history: list[TurnRecord] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    turn_count: int = 0
    # internal bookkeeping (not exposed to the UI)
    consent_request: Any = None            # live ConsentRequest while a representative waits for approval
    earlier_questions_served: bool = False # pre-verification questions have been answered

    # ---- helpers ------------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.phase in (Phase.CLOSED, Phase.HUMAN_HANDOFF)

    def caller_first_name(self) -> Optional[str]:
        """The name the caller gave us (never the record's name, so we never leak it)."""
        v = self.verification
        if v.caller_role == "representative" and v.rep_name:
            return v.rep_name.split()[0]
        full = v.provided.get("full_name")
        return full.split()[0].title() if full else None

    def transition(self, new_phase: Phase, plan: Optional[Plan] = None, reason: str = "") -> None:
        if new_phase != self.phase:
            msg = f"TRANSITION {self.phase.value} -> {new_phase.value}" + (f" ({reason})" if reason else "")
            if plan is not None:
                plan.event(msg)
            self.phase = new_phase

    def recent_history(self, n: int = 8) -> list[TurnRecord]:
        return self.history[-n:]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "freedom": PHASE_FREEDOM[self.phase],
            "verification": self.verification.to_dict(),
            "memory": self.memory.to_dict(),
            "selected_case_id": self.selected_case_id,
            "claims_discussed": self.claims_discussed,
            "topics_covered": self.topics_covered,
            "pending_question": self.pending_question,
            "off_topic_strikes": self.off_topic_strikes,
            "off_topic_total": self.off_topic_total,
            "human_offered": self.human_offered,
            "negative_streak": self.negative_streak,
            "last_emotion": self.last_emotion,
            "consent_scenario": self.consent_scenario,
            "email_sent_id": self.email_sent_id,
            "handoff_reference": self.handoff_reference,
            "handoff_reason": self.handoff_reason,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "is_terminal": self.is_terminal,
        }


__all__ = [
    "LOCATOR_FIELDS",
    "Memory",
    "PENDING_ANYTHING_ELSE",
    "PENDING_CHOOSE_CLAIM",
    "PENDING_CONFIRM_CLAIM",
    "PENDING_CONSENT_WAIT",
    "PENDING_EMAIL_OFFER",
    "PENDING_HUMAN_OFFER",
    "PENDING_REP_PATH",
    "PHASE_FREEDOM",
    "PII_FIELDS",
    "Phase",
    "Plan",
    "SessionState",
    "TurnRecord",
    "VerificationState",
    "ConsentState",
]
