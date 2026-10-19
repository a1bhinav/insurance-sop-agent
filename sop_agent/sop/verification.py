"""Identity gate: locate the record the caller claims to be, compare the
identifying fields they supplied, and decide whether the SOP's threshold
(default: 3 matching PII factors) has been met.

The policy number is treated as a *locator*, not a factor: it tells us which
record to compare against but, on its own, proves nothing about who is calling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data.store import DataStore, Policyholder
from ..utils.normalize import emails_match, names_match, phones_match
from .signals import PII_FIELDS
from .state import VerificationState

FIELD_LABELS = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number on file",
    "email": "email address on file",
    "id_last4": "last four digits of your SSN (or national ID)",
    "policy_number": "policy number",
}


@dataclass
class VerifyOutcome:
    located: bool = False
    ambiguous: bool = False
    record: Optional[Policyholder] = None
    matched: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    newly_mismatched: list[str] = field(default_factory=list)
    newly_matched: list[str] = field(default_factory=list)
    verified: bool = False
    remaining: list[str] = field(default_factory=list)
    unknown_record: bool = False      # caller gave identifying info that matches nobody
    needs_locator: bool = False       # caller gave details (e.g. DOB) but nothing that can find a record yet

    @property
    def matched_count(self) -> int:
        return len(self.matched)


def locate(store: DataStore, provided: dict[str, str]) -> tuple[list[Policyholder], bool]:
    """Return (candidates, attempted). `attempted` is False when nothing in
    `provided` could possibly locate a record (e.g. only a DOB so far)."""
    pol = provided.get("policy_number")
    if pol:
        rec = store.find_by_policy_number(pol)
        return ([rec] if rec else []), True
    name = provided.get("full_name")
    if name:
        by_name = store.find_by_name(name)
        if len(by_name) > 1:
            # narrow with phone / email when several people share a name
            for key, finder in (("phone", store.find_by_phone), ("email", store.find_by_email)):
                if provided.get(key):
                    narrowed = [p for p in by_name if p in finder(provided[key])]
                    if narrowed:
                        return narrowed, True
        return by_name, True
    for key, finder in (("phone", store.find_by_phone), ("email", store.find_by_email)):
        if provided.get(key):
            found = finder(provided[key])
            if found:
                return found, True
    return [], bool(provided.get("phone") or provided.get("email"))


def compare(record: Policyholder, provided: dict[str, str]) -> tuple[list[str], list[str]]:
    matched, mismatched = [], []
    for f in PII_FIELDS:
        val = provided.get(f)
        if not val:
            continue
        ok = False
        if f == "full_name":
            ok = names_match(val, record.name, record.name_aliases)
        elif f == "dob":
            ok = val == record.dob
        elif f == "phone":
            ok = phones_match(val, record.phone, record.phone_aliases)
        elif f == "email":
            ok = emails_match(val, record.email, record.email_aliases)
        elif f == "id_last4":
            ok = val == record.id_last4
        (matched if ok else mismatched).append(f)
    return matched, mismatched


def apply_identity_update(
    store: DataStore,
    vstate: VerificationState,
    new_pii: dict[str, str],
    min_factors: int,
    max_attempts: int,
) -> VerifyOutcome:
    """Merge newly supplied fields, re-locate the record and re-evaluate.

    Matches are recomputed from everything supplied so far, which means a caller
    who corrects a mistyped date of birth is re-evaluated cleanly.
    """
    before_mismatched = set(vstate.mismatched)
    before_matched = set(vstate.matched)
    for k, v in new_pii.items():
        if v:
            vstate.provided[k] = v

    outcome = VerifyOutcome()
    candidates, attempted = locate(store, vstate.provided)
    if len(candidates) > 1:
        outcome.ambiguous = True
        outcome.remaining = [f for f in PII_FIELDS if f not in vstate.provided]
        return outcome
    if not candidates:
        outcome.unknown_record = attempted
        outcome.needs_locator = (not attempted) and bool(vstate.provided)
        # A failed lookup only counts as an attempt when a *locator* was (re)supplied this turn.
        if attempted and any(k in new_pii for k in ("full_name", "policy_number", "phone", "email")):
            vstate.attempts += 1
        vstate.candidate_party_id = None
        vstate.matched, vstate.mismatched = [], []
        outcome.remaining = [f for f in PII_FIELDS if f not in vstate.provided]
        if vstate.attempts >= max_attempts:
            vstate.locked = True
        return outcome

    record = candidates[0]
    outcome.located = True
    outcome.record = record
    vstate.candidate_party_id = record.party_id
    matched, mismatched = compare(record, vstate.provided)
    vstate.matched, vstate.mismatched = matched, mismatched
    outcome.matched, outcome.mismatched = list(matched), list(mismatched)
    # Fields supplied *this turn* that don't match — a re-supplied wrong value counts again.
    outcome.newly_mismatched = [f for f in mismatched if f in new_pii or f not in before_mismatched]
    outcome.newly_matched = [f for f in matched if f not in before_matched]
    if outcome.newly_mismatched:
        vstate.attempts += 1
        if vstate.attempts >= max_attempts:
            vstate.locked = True
    outcome.remaining = [f for f in PII_FIELDS if f not in matched and f not in mismatched]
    outcome.verified = len(matched) >= min_factors and not vstate.locked
    return outcome


MISMATCH_LABELS = {
    "full_name": "name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "id_last4": "last four digits of the ID",
}


def describe_options(remaining: list[str]) -> list[str]:
    """Human labels for the fields the caller can still offer. The ID label stays
    generic ("SSN or national ID") so we never reveal which kind of ID the
    record on file uses before the caller is verified."""
    return [FIELD_LABELS[f] for f in remaining if f in FIELD_LABELS]


def describe_mismatch(fields: list[str]) -> list[str]:
    return [MISMATCH_LABELS[f] for f in fields if f in MISMATCH_LABELS]
