"""Output guards: deterministic checks applied to every generated reply.

The responder LLM only receives whitelisted facts, so it *shouldn't* be able to
leak claim details before verification — but prompts are not guarantees. These
guards are the belt to the prompt's braces:

* LeakGuard  – before verification, block any reply that contains claim
  identifiers, claim-specific literals from the database, money amounts or
  denial language, and replace it with a safe SOP-compliant message.
* PIIGuard   – never echo values from the record on file that the caller did not
  supply themselves (e.g. the stored phone number or email), masked or not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data.store import DataStore, Policyholder
from ..utils.normalize import normalize_phone
from .state import SessionState

CLAIM_ID_RE = re.compile(r"\bCL-?\d{3,6}\b", re.I)
MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})+(?:\.\d{2})\b|\b\d+\.\d{2}\s*(?:USD|dollars)\b", re.I)
# Claim-outcome *explanations* (not the caller's own words about their status, which
# the agent may acknowledge: "I've noted you're calling about a denied claim").
DENIAL_LANGUAGE_RE = re.compile(
    r"\b(was denied because|denied due to|denial reason|the reason (?:for the denial|it was denied)|"
    r"appeal deadline|reimbursement amount|allowed (?:max(?:imum)? )?amount|net pay|"
    r"missing (?:the )?(?:pathology|office note|diagnosis))\b",
    re.I,
)


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    redacted: bool = False
    hits: list[str] = field(default_factory=list)


class LeakGuard:
    def __init__(self, store: DataStore):
        self.store = store
        self._literals = [s for s in store.sensitive_claim_strings() if len(s) >= 4]

    def scan(self, text: str) -> list[str]:
        hits: list[str] = []
        low = text.lower()
        for lit in self._literals:
            if lit.lower() in low:
                hits.append(f"literal:{lit[:40]}")
        if CLAIM_ID_RE.search(text):
            hits.append("claim_id")
        if MONEY_RE.search(text):
            hits.append("money_amount")
        if DENIAL_LANGUAGE_RE.search(text):
            hits.append("claim_outcome_language")
        return hits

    def check(self, text: str, pre_verification: bool) -> GuardResult:
        if not pre_verification:
            return GuardResult(text=text)
        hits = self.scan(text)
        if hits:
            return GuardResult(text="", blocked=True, hits=hits)
        return GuardResult(text=text)


class PIIGuard:
    """Redact record values the caller never supplied. Values the caller *did*
    supply may be repeated back (e.g. "Thanks, Margaret")."""

    def check(self, text: str, record: Policyholder | None, provided: dict[str, str]) -> GuardResult:
        if record is None:
            return GuardResult(text=text)
        redacted = False
        hits: list[str] = []
        out = text

        def _redact(value: str, label: str, pattern: str | None = None) -> None:
            nonlocal out, redacted
            if not value:
                return
            rx = re.compile(pattern or re.escape(value), re.I)
            if rx.search(out):
                out = rx.sub("[redacted]", out)
                redacted = True
                hits.append(label)

        if "dob" not in provided:
            _redact(record.dob, "dob")
        if "email" not in provided:
            for e in [record.email] + list(record.email_aliases):
                _redact(e, "email")
        if "phone" not in provided:
            for p in [record.phone] + list(record.phone_aliases):
                digits = normalize_phone(p)
                if len(digits) >= 10:
                    # match the number in any common formatting
                    pat = r"\+?1?\D{0,2}" + r"\D{0,2}".join(digits)
                    _redact(digits, "phone", pat)
        if "id_last4" not in provided and record.id_last4:
            _redact(record.id_last4, "id_last4", r"(?<!\d)" + re.escape(record.id_last4) + r"(?!\d)")
        if "full_name" not in provided:
            _redact(record.name, "name")
        return GuardResult(text=out, redacted=redacted, hits=hits)


class OutputGuards:
    def __init__(self, store: DataStore):
        self.store = store
        self.leak = LeakGuard(store)
        self.pii = PIIGuard()

    def record_for(self, state: SessionState) -> Policyholder | None:
        pid = state.verification.candidate_party_id
        return self.store.get_policyholder(pid) if pid else None

    def apply(self, text: str, state: SessionState, pre_verification: bool) -> GuardResult:
        leak = self.leak.check(text, pre_verification)
        if leak.blocked:
            return leak
        pii = self.pii.check(text, self.record_for(state), state.verification.provided)
        return GuardResult(text=pii.text, redacted=pii.redacted, hits=pii.hits)
