"""Grounded tools the SOP controller may call.

Each function returns plain data (dicts / dataclasses). The responder LLM only
ever sees what the controller chooses to place in a Plan's `facts`, so these
helpers are the boundary between "what the system knows" and "what may be said".
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from .store import Claim, DataStore, Policyholder
from ..utils.normalize import normalize_name

CASE_TYPE_SYNONYMS = {
    "healthcare": {"healthcare", "health", "medical", "hospital", "doctor", "clinic", "health care"},
    "dental": {"dental", "dentist", "teeth", "tooth"},
    "auto": {"auto", "car", "vehicle", "accident", "collision", "automobile"},
}

STATUS_SYNONYMS = {
    "denied": {"denied", "denial", "rejected", "declined", "turned down"},
    "open": {"open", "pending", "in progress", "processing", "ongoing", "active"},
    "closed": {"closed", "settled", "paid", "completed", "finished", "resolved"},
}


def canonical_case_type(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = normalize_name(text)
    for canon, syns in CASE_TYPE_SYNONYMS.items():
        if t == canon or any(s in t for s in syns):
            return canon
    return None


def canonical_status(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = normalize_name(text)
    for canon, syns in STATUS_SYNONYMS.items():
        if t == canon or any(s in t for s in syns):
            return canon
    return None


# --------------------------------------------------------------------------------------
# Claim resolution
# --------------------------------------------------------------------------------------
def resolve_claims(store: DataStore, party_id: str, hints: dict[str, Any]) -> list[Claim]:
    """Narrow a policyholder's claims using whatever hints the caller gave.

    Hints: case_id, case_type, status, month (1-12), year. Each present hint is a
    hard filter; an explicit case_id wins outright. Returns the matching claims
    (possibly all of them when no hints apply).
    """
    claims = store.claims_for_party(party_id)
    case_id = (hints.get("case_id") or "").strip().upper()
    if case_id:
        exact = [c for c in claims if c.case_id.upper() == case_id]
        if exact:
            return exact
    out = claims
    case_type = canonical_case_type(hints.get("case_type"))
    if case_type:
        out = [c for c in out if c.case_type == case_type]
    status = canonical_status(hints.get("status"))
    if status:
        out = [c for c in out if c.status == status]
    month = hints.get("month")
    if isinstance(month, int) and 1 <= month <= 12:
        out = [c for c in out if c.created_month == month]
    year = hints.get("year")
    if isinstance(year, int) and year > 1900:
        out = [c for c in out if c.created_year == year]
    return out


def hints_are_specific(hints: dict[str, Any]) -> bool:
    return any(hints.get(k) for k in ("case_id", "case_type", "status", "month", "year"))


# --------------------------------------------------------------------------------------
# Claim facts bundle for PROCESS_CASE
# --------------------------------------------------------------------------------------
def _fill(template: str, claim: Claim, settings: dict[str, str]) -> str:
    docs = ", ".join(claim.documents_needed) if claim.documents_needed else "the requested documents"
    text = template.replace("{case_id}", claim.case_id).replace("{documents}", docs)
    for key, val in settings.items():
        text = text.replace("{" + key + "}", val)
    return text


UNAVAILABILITY_WORDS = ("don't have", "do not have", "dont have", "can't get", "cannot get", "can't find", "lost", "unavailable",
                        "no longer", "alternative", "substitute", "instead", "what if", "never got", "doesn't exist", "not available",
                        "won't give", "refuses to", "closed down", "out of business")


def matching_followup_guidance(store: DataStore, claim: Claim, question: str, intent: Optional[str]) -> list[dict[str, str]]:
    """Pick the follow-up guidance entries relevant to the caller's question.

    Rules with `match_any` keywords fire on a keyword hit. Rules without keywords
    (the generic "what if I can't get the document" guidance) fire when the intent
    matches *and* the caller is talking about something being unavailable.
    """
    q = (question or "").lower()
    settings = store.guidelines.followup_settings()
    out = []
    for rule in store.guidelines.followup_guidance():
        if rule.get("requires_documents") and not claim.documents_needed:
            continue
        keyword_hit = any(k in q for k in rule.get("match_any", []))
        intent_hit = intent in set(rule.get("intent_hints", []))
        if rule.get("match_any"):
            if not keyword_hit:
                continue
        elif not (intent_hit and any(w in q for w in UNAVAILABILITY_WORDS)):
            continue
        out.append({"topic": rule.get("topic", ""), "guidance": _fill(rule.get("en", ""), claim, settings)})
    return out


def claim_fact_bundle(store: DataStore, claim: Claim, question: str = "", intent: Optional[str] = None,
                      today: Optional[str] = None) -> dict[str, Any]:
    """Everything the responder is allowed to say about one claim."""
    g = store.guidelines
    facts: dict[str, Any] = {"claim": claim.full(), "today": today or date.today().isoformat()}
    descriptions = (store.claim_schema.get("field_descriptions") or {})
    if descriptions:
        facts["amount_field_meanings"] = {k: v.get("description") for k, v in descriptions.items()}
    if claim.appeal_deadline and facts["today"]:
        facts["appeal_deadline_has_passed"] = facts["today"] > claim.appeal_deadline
    if claim.documents_needed:
        docs = []
        for d in claim.documents_needed:
            docs.append({
                "document": d,
                "requirements": g.document_guidance(d),
                "if_unavailable": g.alternative_guidance(d),
            })
        facts["documents_needed_detail"] = docs
        facts["general_submission_guidance"] = g.default_guidance()
        ct = g.case_type_guidance(claim.case_type)
        if ct:
            facts["case_type_submission_guidance"] = ct
    followups = matching_followup_guidance(store, claim, question, intent)
    if followups:
        facts["followup_guidance"] = followups
    settings = g.followup_settings()
    if settings:
        facts["followup_settings"] = settings
    fallback = g.followup_fallback()
    if fallback:
        facts["followup_fallback_if_no_rule"] = fallback
    return facts


# --------------------------------------------------------------------------------------
# Consent (representative calls)
# --------------------------------------------------------------------------------------
@dataclass
class ConsentRequest:
    consent_id: str
    party_id: str
    scenario: str
    sequence: list[str]
    checks: int = 0
    status: str = "pending"

    def check(self) -> str:
        idx = min(self.checks, len(self.sequence) - 1)
        self.status = self.sequence[idx]
        self.checks += 1
        return self.status


class ConsentService:
    """Simulates sending a consent request to the policyholder's phone/email on
    file and polling for approval. The status sequence comes from
    consent_scenarios.json so a tester can force a timeout."""

    def __init__(self, store: DataStore):
        self.store = store

    def scenarios(self) -> list[str]:
        return list(self.store.consent_scenarios.keys())

    def request(self, party_id: str, scenario: str = "default") -> ConsentRequest:
        spec = self.store.consent_scenarios.get(scenario) or self.store.consent_scenarios.get("default") or {}
        sequence = list(spec.get("status_sequence") or ["pending", "approved"])
        return ConsentRequest(
            consent_id=f"CNS-{uuid.uuid4().hex[:6].upper()}",
            party_id=party_id,
            scenario=scenario,
            sequence=sequence,
        )


# --------------------------------------------------------------------------------------
# Email outbox (mock delivery)
# --------------------------------------------------------------------------------------
@dataclass
class OutboundEmail:
    message_id: str
    to: str
    subject: str
    body: str
    sent_at: str
    session_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "to": self.to,
            "subject": self.subject,
            "body": self.body,
            "sent_at": self.sent_at,
            "session_id": self.session_id,
            "meta": self.meta,
        }


class EmailOutbox:
    """Writes each 'sent' email to disk and keeps it in memory for the UI.
    Swap `deliver()` for a real SMTP/SES call in production."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.sent: list[OutboundEmail] = []

    def deliver(self, to: str, subject: str, body: str, session_id: str, meta: Optional[dict[str, Any]] = None) -> OutboundEmail:
        email = OutboundEmail(
            message_id=f"EML-{uuid.uuid4().hex[:8].upper()}",
            to=to,
            subject=subject,
            body=body,
            sent_at=datetime.now().isoformat(timespec="seconds"),
            session_id=session_id,
            meta=meta or {},
        )
        self.sent.append(email)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{email.sent_at.replace(':', '-')}_{email.message_id}.json"
            path.write_text(json.dumps(email.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass  # disk is best-effort; in-memory copy is what the UI shows
        return email

    def for_session(self, session_id: str) -> list[OutboundEmail]:
        return [e for e in self.sent if e.session_id == session_id]


def humanize_date(iso: Optional[str]) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d")
        return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    except ValueError:
        return iso


def extract_case_ids(text: str) -> list[str]:
    return [m.upper() for m in re.findall(r"\bCL-?\d{3,6}\b", text or "", flags=re.I)]


__all__ = [
    "ConsentRequest",
    "ConsentService",
    "EmailOutbox",
    "OutboundEmail",
    "canonical_case_type",
    "canonical_status",
    "claim_fact_bundle",
    "extract_case_ids",
    "hints_are_specific",
    "humanize_date",
    "matching_followup_guidance",
    "resolve_claims",
]
