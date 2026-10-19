"""Normalization helpers for identity fields.

Callers say dates, phones and names in many shapes; the identity gate compares
normalized forms so that "March 15th 1985", "03/15/1985" and "1985-03-15" all
verify against the same record.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    cleaned = _PUNCT.sub(" ", name.lower())
    return _WS.sub(" ", cleaned).strip()


def name_tokens(name: Optional[str]) -> list[str]:
    return [t for t in normalize_name(name).split(" ") if t]


def names_match(provided: Optional[str], on_file: Optional[str], aliases: Optional[list[str]] = None) -> bool:
    """True when the provided name identifies the person on file.

    Rules: ignore case/punctuation/order; the provided tokens must all appear in
    the record name (or an alias) and at least two tokens must be present (a lone
    first name is not enough). Middle names on either side are tolerated.
    """
    prov = name_tokens(provided)
    if len(prov) < 2:
        return False
    candidates = [on_file] + list(aliases or [])
    for cand in candidates:
        cand_tokens = set(name_tokens(cand))
        if not cand_tokens:
            continue
        if set(prov).issubset(cand_tokens):
            return True
        # Tolerate an extra middle name supplied by the caller: first + last must match.
        if prov[0] in cand_tokens and prov[-1] in cand_tokens and len(prov) >= 2:
            return True
    return False


def _valid_date(y: int, m: int, d: int) -> Optional[str]:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def _expand_year(y: int) -> int:
    if y >= 100:
        return y
    # two-digit year: assume 19xx for anything above the current 2-digit year
    current = date.today().year % 100
    return 2000 + y if y <= current else 1900 + y


def parse_dob(text: Optional[str]) -> Optional[str]:
    """Parse a date of birth from free text into ISO YYYY-MM-DD, or None."""
    if not text:
        return None
    s = text.strip().lower()
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)  # 15th -> 15

    # ISO: 1985-03-15 or 1985/03/15
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", s)
    if m:
        return _valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Numeric: 03/15/1985 (US) — if the first number cannot be a month, treat as D/M/Y.
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), _expand_year(int(m.group(3)))
        if a > 12 and b <= 12:
            return _valid_date(y, b, a)
        return _valid_date(y, a, b) or _valid_date(y, b, a)

    # Month name first: March 15, 1985 / Mar 15 1985
    m = re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b", s)
    if m and m.group(1) in MONTHS:
        return _valid_date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))

    # Day first: 15 March 1985
    m = re.search(r"\b(\d{1,2})\s+(?:of\s+)?([a-z]{3,9})\.?,?\s+(\d{4})\b", s)
    if m and m.group(2) in MONTHS:
        return _valid_date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))

    # Month name, day, 2-digit year: March 15, 85
    m = re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2}),?\s+'?(\d{2})\b", s)
    if m and m.group(1) in MONTHS:
        return _valid_date(_expand_year(int(m.group(3))), MONTHS[m.group(1)], int(m.group(2)))
    return None


def normalize_phone(text: Optional[str]) -> str:
    """Digits only, trimmed to the national 10 digits when a country code is present."""
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def phones_match(provided: Optional[str], on_file: Optional[str], aliases: Optional[list[str]] = None) -> bool:
    p = normalize_phone(provided)
    if len(p) < 7:
        return False
    for cand in [on_file] + list(aliases or []):
        c = normalize_phone(cand)
        if c and (c == p or c.endswith(p) or p.endswith(c)):
            return True
    return False


def normalize_email(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def emails_match(provided: Optional[str], on_file: Optional[str], aliases: Optional[list[str]] = None) -> bool:
    p = normalize_email(provided)
    if not p or "@" not in p:
        return False
    return any(p == normalize_email(c) for c in [on_file] + list(aliases or []) if c)


def normalize_last4(text: Optional[str]) -> str:
    digits = re.sub(r"\D", "", text or "")
    return digits[-4:] if len(digits) >= 4 else ""


def normalize_policy_number(text: Optional[str]) -> str:
    if not text:
        return ""
    s = text.upper().replace(" ", "")
    m = re.search(r"POL-?(\d+)", s)
    return f"POL-{m.group(1)}" if m else s


def mask_email(email: Optional[str]) -> str:
    if not email or "@" not in email:
        return "the email address on file"
    local, _, domain = email.partition("@")
    shown = local[0] if local else ""
    return f"{shown}{'*' * max(3, len(local) - 1)}@{domain}"


def mask_phone(phone: Optional[str]) -> str:
    d = normalize_phone(phone)
    return f"***-***-{d[-4:]}" if len(d) >= 4 else "the phone number on file"


def month_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    for tok in re.findall(r"[a-z]+", text.lower()):
        if tok in MONTHS:
            return MONTHS[tok]
    return None


def year_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return int(m.group(1)) if m else None
