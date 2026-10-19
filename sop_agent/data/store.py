"""Fixture-backed data store: policyholders, claims, representatives, consent
scenarios and the document guideline knowledge base.

This is the *only* source of claim facts the agent may speak. Every lookup the
SOP controller performs goes through here, which keeps answers grounded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..utils.normalize import (
    emails_match,
    names_match,
    normalize_name,
    normalize_policy_number,
    phones_match,
)


@dataclass
class Policyholder:
    party_id: str
    name: str
    policy_number: str
    dob: str
    id_type: str
    id_last4: str
    phone: str
    email: str
    name_aliases: list[str] = field(default_factory=list)
    phone_aliases: list[str] = field(default_factory=list)
    email_aliases: list[str] = field(default_factory=list)

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else ""

    @property
    def id_label(self) -> str:
        return "national ID" if self.id_type == "national_id_last4" else "SSN"


@dataclass
class Claim:
    case_id: str
    party_id: str
    case_type: str
    created_at: str
    status: str
    summary: str
    denial_reason: Optional[str] = None
    documents_needed: list[str] = field(default_factory=list)
    appeal_deadline: Optional[str] = None
    expected_reimbursement_amount: Optional[str] = None
    allowed_max_amount: Optional[str] = None
    net_pay: Optional[str] = None
    net_fee: Optional[str] = None

    @property
    def created_month(self) -> int:
        return int(self.created_at[5:7])

    @property
    def created_year(self) -> int:
        return int(self.created_at[:4])

    def brief(self) -> dict[str, Any]:
        """Short description used for disambiguation lists (no financial detail)."""
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "created_at": self.created_at,
            "status": self.status,
            "summary": self.summary,
        }

    def full(self) -> dict[str, Any]:
        data = {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "created_at": self.created_at,
            "status": self.status,
            "summary": self.summary,
            "expected_reimbursement_amount_usd": self.expected_reimbursement_amount,
            "allowed_max_amount_usd": self.allowed_max_amount,
            "net_pay_usd": self.net_pay,
            "net_fee_usd": self.net_fee,
        }
        if self.denial_reason:
            data["denial_reason"] = self.denial_reason
        if self.documents_needed:
            data["documents_needed"] = list(self.documents_needed)
        if self.appeal_deadline:
            data["appeal_deadline"] = self.appeal_deadline
        return data


@dataclass
class Representative:
    rep_name: str
    relationship: str
    buyer_name: str
    buyer_party_id: str


class DocumentGuidelines:
    """Wrapper over required_document_guideline.json with fuzzy document lookup."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw or {}

    def _text(self, node: Any) -> Optional[str]:
        if isinstance(node, dict):
            return node.get("en")
        return node

    def default_guidance(self) -> Optional[str]:
        return self._text(self.raw.get("default_guidance"))

    def case_type_guidance(self, case_type: str) -> Optional[str]:
        return self._text((self.raw.get("case_type_guidance") or {}).get(case_type))

    GENERIC_DOC_WORDS = {"report", "note", "notes", "original", "treating", "provider", "supplemental", "copy", "form", "the", "of", "and"}

    def _best_key(self, section: str, document: str) -> Optional[str]:
        """Fuzzy-match a claim's document name ("pathology report") to a
        guideline key ("original pathology report") by overlap of the
        *distinctive* tokens — generic words like "report" don't count, so
        "diagnosis report" does not borrow the pathology-report guidance."""
        keys = list((self.raw.get(section) or {}).keys())
        doc_tokens = set(normalize_name(document).split()) - self.GENERIC_DOC_WORDS
        if not doc_tokens:
            doc_tokens = set(normalize_name(document).split())
        best, best_score = None, 0.0
        for key in keys:
            if key == "default":
                continue
            key_tokens = set(normalize_name(key).split())
            if not key_tokens:
                continue
            score = len(doc_tokens & key_tokens) / len(doc_tokens)
            if score > best_score:
                best, best_score = key, score
        return best if best_score >= 0.5 else None

    def document_guidance(self, document: str) -> Optional[str]:
        key = self._best_key("document_guidance", document)
        return self._text((self.raw.get("document_guidance") or {}).get(key)) if key else None

    def alternative_guidance(self, document: str) -> Optional[str]:
        section = self.raw.get("document_alternative_guidance") or {}
        key = self._best_key("document_alternative_guidance", document)
        text = self._text(section.get(key)) if key else None
        return text or self._text(section.get("default"))

    def followup_settings(self) -> dict[str, str]:
        out = {}
        for k, v in (self.raw.get("claim_followup_settings") or {}).items():
            t = self._text(v)
            if t:
                out[k] = t
        return out

    def followup_guidance(self) -> list[dict[str, Any]]:
        return list(self.raw.get("claim_followup_guidance") or [])

    def followup_fallback(self) -> Optional[str]:
        return self._text(self.raw.get("claim_followup_fallback"))


class DataStore:
    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = Path(fixtures_dir)
        self.policyholders: list[Policyholder] = []
        self.claims: list[Claim] = []
        self.representatives: list[Representative] = []
        self.consent_scenarios: dict[str, dict[str, Any]] = {}
        self.claim_schema: dict[str, Any] = {}
        self.guidelines = DocumentGuidelines({})
        self.load()

    # ---- loading ------------------------------------------------------------
    def _read(self, name: str) -> Any:
        path = self.fixtures_dir / name
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def load(self) -> None:
        self.policyholders = [
            Policyholder(
                party_id=p["party_id"],
                name=p["name"],
                policy_number=p.get("policy_number", ""),
                dob=p.get("dob", ""),
                id_type=p.get("id_type", "ssn_last4"),
                id_last4=str(p.get("id_last4", "")),
                phone=p.get("phone", ""),
                email=p.get("email", ""),
                name_aliases=list(p.get("name_aliases", [])),
                phone_aliases=list(p.get("phone_aliases", [])),
                email_aliases=list(p.get("email_aliases", [])),
            )
            for p in (self._read("policyholders.json") or [])
        ]
        self.claims = [
            Claim(
                case_id=c["case_id"],
                party_id=c["party_id"],
                case_type=c.get("case_type", ""),
                created_at=c.get("created_at", ""),
                status=c.get("status", ""),
                summary=c.get("summary", ""),
                denial_reason=c.get("denial_reason"),
                documents_needed=list(c.get("documents_needed", [])),
                appeal_deadline=c.get("appeal_deadline"),
                expected_reimbursement_amount=c.get("expected_reimbursement_amount"),
                allowed_max_amount=c.get("allowed_max_amount"),
                net_pay=c.get("net_pay"),
                net_fee=c.get("net_fee"),
            )
            for c in (self._read("claims.json") or [])
        ]
        self.representatives = [
            Representative(
                rep_name=r["rep_name"],
                relationship=r.get("relationship", ""),
                buyer_name=r.get("buyer_name", ""),
                buyer_party_id=r.get("buyer_party_id", ""),
            )
            for r in (self._read("representatives.json") or [])
        ]
        self.consent_scenarios = self._read("consent_scenarios.json") or {
            "default": {"status_sequence": ["pending", "approved"]}
        }
        self.claim_schema = self._read("claim_schema.json") or {}
        self.guidelines = DocumentGuidelines(self._read("required_document_guideline.json") or {})

    # ---- policyholders -------------------------------------------------------
    def get_policyholder(self, party_id: str) -> Optional[Policyholder]:
        return next((p for p in self.policyholders if p.party_id == party_id), None)

    def find_by_policy_number(self, policy_number: str) -> Optional[Policyholder]:
        wanted = normalize_policy_number(policy_number)
        return next((p for p in self.policyholders if normalize_policy_number(p.policy_number) == wanted), None)

    def find_by_name(self, name: str) -> list[Policyholder]:
        return [p for p in self.policyholders if names_match(name, p.name, p.name_aliases)]

    def find_by_phone(self, phone: str) -> list[Policyholder]:
        return [p for p in self.policyholders if phones_match(phone, p.phone, p.phone_aliases)]

    def find_by_email(self, email: str) -> list[Policyholder]:
        return [p for p in self.policyholders if emails_match(email, p.email, p.email_aliases)]

    # ---- claims ----------------------------------------------------------------
    def claims_for_party(self, party_id: str) -> list[Claim]:
        return sorted((c for c in self.claims if c.party_id == party_id), key=lambda c: c.created_at, reverse=True)

    def get_claim(self, case_id: str) -> Optional[Claim]:
        wanted = (case_id or "").strip().upper()
        return next((c for c in self.claims if c.case_id.upper() == wanted), None)

    # ---- representatives ----------------------------------------------------------
    def find_representative(self, rep_name: str, policyholder: Optional[str] = None) -> list[Representative]:
        out = []
        for r in self.representatives:
            if not names_match(rep_name, r.rep_name):
                continue
            if policyholder:
                holder = self.get_policyholder(r.buyer_party_id)
                aliases = holder.name_aliases if holder else []
                if not names_match(policyholder, r.buyer_name, aliases):
                    continue
            out.append(r)
        return out

    # ---- sensitive strings (used by the leak guard) ----------------------------------
    def sensitive_claim_strings(self) -> list[str]:
        """Every claim-specific literal that must never appear before verification."""
        out: set[str] = set()
        for c in self.claims:
            out.add(c.case_id)
            if c.denial_reason:
                out.add(c.denial_reason)
            if c.summary:
                out.add(c.summary)
            if c.appeal_deadline:
                out.add(c.appeal_deadline)
            for d in c.documents_needed:
                out.add(d)
            for amt in (c.expected_reimbursement_amount, c.allowed_max_amount, c.net_pay, c.net_fee):
                if amt and amt != "0.00":
                    out.add(amt)
        return sorted(out, key=len, reverse=True)
