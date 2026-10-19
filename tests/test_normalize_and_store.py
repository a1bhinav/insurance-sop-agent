from __future__ import annotations

from sop_agent.data.tools import canonical_case_type, canonical_status, resolve_claims
from sop_agent.utils.normalize import (
    emails_match,
    mask_email,
    names_match,
    normalize_policy_number,
    parse_dob,
    phones_match,
)


def test_parse_dob_many_formats():
    for raw in ["1985-03-15", "03/15/1985", "3/15/85", "March 15, 1985", "March 15th 1985", "15 March 1985", "Mar 15 1985", "born on 15/03/1985"]:
        assert parse_dob(raw) == "1985-03-15", raw
    assert parse_dob("sometime in January") is None
    assert parse_dob("4472") is None


def test_names_match_tolerates_order_case_and_aliases():
    assert names_match("margaret chen", "Margaret Chen")
    assert names_match("Chen Margaret", "Margaret Chen")
    assert names_match("Margaret A. Chen", "Margaret Chen")
    assert names_match("Yaven Li", "Ya Wen Li", ["Yaven Li"])
    assert not names_match("Margaret", "Margaret Chen")          # a lone first name is not enough
    assert not names_match("Margaret Lopez", "Margaret Chen")


def test_phone_and_email_matching():
    assert phones_match("(650) 521-2836", "+16505212836")
    assert phones_match("650.521.2836", "+16505212836")
    assert not phones_match("650-555-0000", "+16505212836")
    assert emails_match("Margaret@Email.com ", "margaret@email.com")
    assert emails_match("yawen.li@example.com", "yawen.li@gmail.com", ["yawen.li@example.com"])


def test_policy_number_and_mask():
    assert normalize_policy_number("pol 9921") == "POL-9921"
    assert normalize_policy_number("POL-9921") == "POL-9921"
    assert mask_email("margaret@email.com") == "m*******@email.com"
    assert "@" not in mask_email("")


def test_store_lookups(harness):
    s = harness.store
    assert s.find_by_policy_number("POL-9921").party_id == "P9"
    assert [p.party_id for p in s.find_by_name("Yaven Li")] == ["P13"]
    assert [c.case_id for c in s.claims_for_party("P9")] == ["CL-2102", "CL-2048", "CL-1899", "CL-2011"]
    assert s.find_representative("David Chen", "Margaret Chen")
    assert not s.find_representative("David Chen", "Ava Lopez")


def test_claim_resolution_by_hints(harness):
    s = harness.store
    assert canonical_case_type("medical bill") == "healthcare"
    assert canonical_status("rejected") == "denied"
    assert [c.case_id for c in resolve_claims(s, "P9", {"case_type": "healthcare", "status": "denied", "month": 1})] == ["CL-2048"]
    assert {c.case_id for c in resolve_claims(s, "P9", {"month": 1})} == {"CL-2048", "CL-2011"}
    assert [c.case_id for c in resolve_claims(s, "P9", {"case_id": "cl-1899"})] == ["CL-1899"]
    assert resolve_claims(s, "P9", {"case_type": "dental", "status": "denied"}) == []


def test_document_guideline_fuzzy_lookup(harness):
    g = harness.store.guidelines
    assert "pathology" in (g.document_guidance("pathology report") or "")
    assert "office note" in (g.document_guidance("office note") or "").lower()
    assert g.document_guidance("diagnosis report") is None          # no specific guidance in fixtures
    assert g.alternative_guidance("diagnosis report")                # falls back to the default alternative text
