"""TurnSignals: the structured reading of one caller utterance.

The extractor (LLM or rule-based mock) fills this in for *every* turn, no matter
which phase the SOP is in. The controller then decides what to do with it. This
is what lets the agent notice "I'm calling about my denied healthcare claim from
January" during identity verification, store it, and act on it later.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..utils.normalize import (
    month_from_text,
    normalize_email,
    normalize_last4,
    normalize_phone,
    normalize_policy_number,
    parse_dob,
    year_from_text,
)

PII_FIELDS = ("full_name", "dob", "phone", "email", "id_last4")
LOCATOR_FIELDS = ("policy_number",)

INTENTS = (
    "status_inquiry",
    "denial_question",
    "document_submission",
    "next_steps",
    "appeal",
    "payment_question",
    "general_claim_question",
    "coverage_question",
    "update_contact_info",
    "other",
)

SCOPES = ("in_scope", "small_talk", "out_of_scope")
EMOTIONS = ("neutral", "frustrated", "angry", "anxious", "confused", "refusing", "grateful")


def _clean_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        v = str(v)
    if not isinstance(v, str):
        return None
    v = v.strip()
    if not v or v.lower() in {"null", "none", "n/a", "unknown", "not provided"}:
        return None
    return v


def _clean_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


def _clean_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass
class TurnSignals:
    # identity
    pii: dict[str, str] = field(default_factory=dict)          # normalized values keyed by PII_FIELDS / policy_number
    caller_role: Optional[str] = None                          # policyholder | representative
    representative: dict[str, str] = field(default_factory=dict)  # name, relationship, policyholder_name
    # intent & case
    intent: Optional[str] = None
    case_hints: dict[str, Any] = field(default_factory=dict)   # case_id, case_type, status, month, year, time_reference
    questions: list[str] = field(default_factory=list)         # grounded questions to answer (PROCESS_CASE)
    notes: list[str] = field(default_factory=list)             # other useful facts the caller volunteered
    # conversation control
    scope: str = "in_scope"
    off_topic_part: Optional[str] = None                        # the off-topic fragment in a mixed message
    emotion: str = "neutral"
    emotion_intensity: int = 0                                  # 0..3
    requests_human: bool = False
    refuses_verification: bool = False
    asks_why_needed: bool = False
    answer: Optional[str] = None                               # yes | no | unclear (for the pending question)
    wants_to_end: bool = False
    email_preference: Optional[str] = None                      # yes | no (when the caller states it early)
    raw: dict[str, Any] = field(default_factory=dict)          # original extractor output (for traces)
    source: str = "llm"                                        # llm | mock | fallback

    # ---- construction --------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "llm") -> "TurnSignals":
        data = data or {}
        sig = cls(raw=data, source=source)

        pii_in = data.get("pii") or {}
        if isinstance(pii_in, dict):
            name = _clean_str(pii_in.get("full_name"))
            if name and len(name.split()) >= 2:
                sig.pii["full_name"] = name
            dob = parse_dob(_clean_str(pii_in.get("dob")))
            if dob:
                sig.pii["dob"] = dob
            phone = normalize_phone(_clean_str(pii_in.get("phone")))
            if len(phone) >= 7:
                sig.pii["phone"] = phone
            email = normalize_email(_clean_str(pii_in.get("email")))
            if email and "@" in email:
                sig.pii["email"] = email
            last4 = normalize_last4(_clean_str(pii_in.get("id_last4")))
            if len(last4) == 4:
                sig.pii["id_last4"] = last4
            pol = normalize_policy_number(_clean_str(pii_in.get("policy_number")))
            if pol:
                sig.pii["policy_number"] = pol

        role = _clean_str(data.get("caller_role"))
        if role in {"policyholder", "representative"}:
            sig.caller_role = role
        rep_in = data.get("representative") or {}
        if isinstance(rep_in, dict):
            for k in ("name", "relationship", "policyholder_name"):
                v = _clean_str(rep_in.get(k))
                if v:
                    sig.representative[k] = v
            if sig.representative.get("name") and not sig.caller_role:
                sig.caller_role = "representative"

        intent = _clean_str(data.get("intent"))
        sig.intent = intent if intent in INTENTS else ("other" if intent else None)

        hints_in = data.get("case_hints") or {}
        if isinstance(hints_in, dict):
            cid = _clean_str(hints_in.get("case_id"))
            if cid:
                m = re.search(r"CL\W*(\d{3,6})", cid.upper())
                sig.case_hints["case_id"] = f"CL-{m.group(1)}" if m else cid.upper()
            for k in ("case_type", "status", "time_reference"):
                v = _clean_str(hints_in.get(k))
                if v:
                    sig.case_hints[k] = v
            month = _clean_int(hints_in.get("month"))
            if month is None and sig.case_hints.get("time_reference"):
                month = month_from_text(sig.case_hints["time_reference"])
            if month and 1 <= month <= 12:
                sig.case_hints["month"] = month
            year = _clean_int(hints_in.get("year"))
            if year is None and sig.case_hints.get("time_reference"):
                year = year_from_text(sig.case_hints["time_reference"])
            if year and year > 1900:
                sig.case_hints["year"] = year

        qs = data.get("questions") or []
        if isinstance(qs, list):
            sig.questions = [q.strip() for q in qs if isinstance(q, str) and q.strip()][:5]
        notes = data.get("notes") or []
        if isinstance(notes, list):
            sig.notes = [n.strip() for n in notes if isinstance(n, str) and n.strip()][:5]

        scope = _clean_str(data.get("scope"))
        sig.scope = scope if scope in SCOPES else "in_scope"
        sig.off_topic_part = _clean_str(data.get("off_topic_part"))

        emo_in = data.get("emotion")
        if isinstance(emo_in, dict):
            label = _clean_str(emo_in.get("label"))
            sig.emotion = label if label in EMOTIONS else "neutral"
            sig.emotion_intensity = max(0, min(3, _clean_int(emo_in.get("intensity")) or 0))
        elif isinstance(emo_in, str):
            sig.emotion = emo_in if emo_in in EMOTIONS else "neutral"
            sig.emotion_intensity = 1 if sig.emotion != "neutral" else 0

        sig.requests_human = _clean_bool(data.get("requests_human"))
        sig.refuses_verification = _clean_bool(data.get("refuses_verification"))
        sig.asks_why_needed = _clean_bool(data.get("asks_why_needed"))
        ans = _clean_str(data.get("answer"))
        sig.answer = ans.lower() if ans and ans.lower() in {"yes", "no", "unclear"} else None
        sig.wants_to_end = _clean_bool(data.get("wants_to_end"))
        pref = _clean_str(data.get("email_preference"))
        sig.email_preference = pref.lower() if pref and pref.lower() in {"yes", "no"} else None
        return sig

    # ---- helpers ---------------------------------------------------------------------
    @property
    def is_negative(self) -> bool:
        return self.emotion in {"frustrated", "angry", "anxious", "confused", "refusing"} and self.emotion_intensity >= 1

    @property
    def has_case_hints(self) -> bool:
        return any(self.case_hints.get(k) for k in ("case_id", "case_type", "status", "month", "year"))

    @property
    def has_substantive_request(self) -> bool:
        return bool(self.questions) or (self.intent not in (None, "other")) or self.has_case_hints

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d
