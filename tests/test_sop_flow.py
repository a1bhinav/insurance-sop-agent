"""End-to-end SOP behaviour with the deterministic (mock) extractor/responder."""
from __future__ import annotations

from conftest import assert_no_claim_leak

DEMO_UTTERANCE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim "
    "from January. DOB is 1985-03-15, SSN last four is 4472."
)


def test_opening_message_asks_for_identity_without_claim_data(convo):
    assert "verify" in convo.greeting.lower()
    assert_no_claim_leak(convo.greeting)
    assert convo.phase == "VERIFY_ID"


def test_demo_case_verifies_remembers_hint_and_resolves_claim(convo):
    res = convo.say(DEMO_UTTERANCE)
    v = convo.state.verification
    assert v.verified and set(v.matched) == {"full_name", "dob", "id_last4"}
    assert v.provided["policy_number"] == "POL-9921"          # locator, not a factor
    assert convo.phase == "PROCESS_CASE"
    assert convo.state.selected_case_id == "CL-2048"
    # memory captured the hint before verification was complete
    assert convo.state.memory.case_hints["case_type"] == "healthcare"
    assert convo.state.memory.case_hints["status"] == "denied"
    assert convo.state.memory.case_hints["month"] == 1
    assert "denial_question" in convo.state.memory.intent_hints
    # the answer is grounded in the fixture
    assert "CL-2048" in res.reply
    assert "pathology report" in res.reply and "office note" in res.reply
    assert any(e.startswith("TRANSITION VERIFY_ID -> RESOLVE_INTENT") for e in res.plan.events)
    assert any(e.startswith("TRANSITION RESOLVE_INTENT -> PROCESS_CASE") for e in res.plan.events)


def test_demo_case_full_workflow_through_email(convo, harness):
    convo.say(DEMO_UTTERANCE)
    r = convo.say("What documents do I need to send and how do I submit them?")
    assert "pathology report" in r.reply and "portal" in r.reply.lower()
    r = convo.say("How long will it take once I submit?")
    assert "less than a week" in r.reply
    r = convo.say("No, that's all.")
    assert convo.phase == "POST_PROCESS"
    assert "m*******@email.com" in r.reply and "margaret@email.com" not in r.reply   # masked address only
    r = convo.say("Yes please.")
    assert convo.phase == "CLOSED"
    emails = harness.outbox.for_session(convo.state.session_id)
    assert len(emails) == 1
    email = emails[0]
    assert email.to == "margaret@email.com"
    assert "CL-2048" in email.body and "pathology report" in email.body and "March 18, 2026" in email.body
    assert "What we discussed" in email.body and "Next steps" in email.body


def test_customer_can_skip_the_email(convo, harness):
    convo.say(DEMO_UTTERANCE)
    convo.say("that's all, thanks")
    assert convo.phase == "POST_PROCESS"
    r = convo.say("no thanks")
    assert convo.phase == "CLOSED"
    assert harness.outbox.for_session(convo.state.session_id) == []
    assert "won't send" in r.reply.lower()


def test_no_claim_details_before_verification_and_question_is_held(convo):
    r = convo.say("Hi, why was my claim denied?")
    assert convo.phase == "VERIFY_ID"
    assert_no_claim_leak(r.reply)
    assert convo.state.memory.pending_questions == ["Hi, why was my claim denied?"] or convo.state.memory.pending_questions
    assert "denial_question" in convo.state.memory.intent_hints
    assert "noted" in r.reply.lower()                      # acknowledges without disclosing

    r = convo.say("Margaret Chen")
    assert convo.phase == "VERIFY_ID" and convo.state.verification.matched_count() == 1
    assert_no_claim_leak(r.reply)
    r = convo.say("March 15th, 1985")
    assert convo.phase == "VERIFY_ID" and convo.state.verification.matched_count() == 2
    assert_no_claim_leak(r.reply)
    r = convo.say("margaret@email.com")
    assert convo.state.verification.verified
    assert convo.phase == "PROCESS_CASE" and convo.state.selected_case_id == "CL-2048"
    assert "pathology report" in r.reply                   # held question answered without being asked again


def test_switching_claims_mid_case(convo):
    convo.say(DEMO_UTTERANCE)
    r = convo.say("What about my auto claim, is it still open?")
    assert convo.state.selected_case_id == "CL-2102"
    assert "CL-2102" in r.reply and "open" in r.reply
    assert convo.state.claims_discussed == ["CL-2048", "CL-2102"]


def test_ambiguous_description_asks_to_disambiguate(convo):
    convo.say("My name is Margaret Chen, DOB 1985-03-15, last four 4472. I'm calling about my healthcare claim from January.")
    assert convo.state.verification.verified
    assert convo.phase == "RESOLVE_INTENT"
    reply = convo.last_reply
    assert "CL-2048" in reply and "CL-2011" in reply       # two January healthcare claims
    assert "pathology" not in reply                        # no claim details while disambiguating
    r = convo.say("The denied one, CL-2048")
    assert convo.phase == "PROCESS_CASE" and convo.state.selected_case_id == "CL-2048"
    assert "denied because" in r.reply


def test_no_hints_lists_claims(convo):
    convo.say("Margaret Chen, 1985-03-15, margaret@email.com")
    assert convo.state.verification.verified and convo.phase == "RESOLVE_INTENT"
    assert "CL-2102" in convo.last_reply and "CL-1899" in convo.last_reply


def test_mismatched_fields_are_reported_without_revealing_values(convo):
    r = convo.say("This is Margaret Chen, DOB 1990-01-01, last four 1111")
    assert not convo.state.verification.verified
    assert set(convo.state.verification.mismatched) == {"dob", "id_last4"}
    assert "doesn't match" in r.reply
    assert "1985" not in r.reply and "4472" not in r.reply
    assert convo.state.verification.attempts == 1
    r = convo.say("Sorry, DOB 1985-03-15 and the last four are 4472")
    assert convo.state.verification.verified                 # corrected values re-evaluated cleanly


def test_attempt_limit_locks_and_offers_human(convo):
    convo.say("This is Margaret Chen, DOB 1990-01-01")
    convo.say("ok try 1991-02-02")
    r = convo.say("1992-03-03 then")
    assert convo.state.verification.locked
    assert "human representative" in r.reply
    assert convo.phase == "VERIFY_ID"
    r = convo.say("yes")
    assert convo.phase == "HUMAN_HANDOFF" and convo.state.handoff_reference
    r = convo.say("DOB 1985-03-15 margaret@email.com 4472")    # terminal: no more verification over this channel
    assert convo.phase == "HUMAN_HANDOFF" and not convo.state.verification.verified


def test_unknown_person_is_not_enumerated(convo):
    r = convo.say("My name is John Nobody, DOB 1970-01-01, phone 555-000-1111")
    assert not convo.state.verification.verified
    assert "not finding" in r.reply.lower() or "doesn't match" in r.reply.lower()
    assert_no_claim_leak(r.reply)


def test_dob_and_last4_without_name_asks_for_locator(convo):
    r = convo.say("DOB 1985-03-15, last four 4472")
    assert convo.phase == "VERIFY_ID"
    assert "full name" in r.reply.lower()
    assert convo.state.verification.attempts == 0


def test_representative_path_requires_consent(make_convo, harness):
    c = make_convo(consent_scenario="default")
    r = c.say("Hi, I'm calling on behalf of my mother, Margaret Chen. My name is David Chen, I'm her son.")
    v = c.state.verification
    assert v.caller_role == "representative" and v.rep_on_file is True
    assert not v.verified and c.phase == "VERIFY_ID"
    r = c.say("Her date of birth is March 15, 1985.")
    assert v.consent is not None and v.consent.status == "pending"
    assert "consent" in r.reply.lower()
    assert_no_claim_leak(r.reply)
    r = c.say("She just approved it on her phone.")
    assert v.verified and v.consent.status == "approved"
    assert c.phase == "RESOLVE_INTENT"
    assert "David" in r.reply
    r = c.say("Is her auto claim still open?")
    assert c.state.selected_case_id == "CL-2102" and "open" in r.reply


def test_representative_consent_timeout_escalates(make_convo):
    c = make_convo(consent_scenario="timeout")
    c.say("I'm calling for my mother Margaret Chen, I'm David Chen her son. Her DOB is 1985-03-15.")
    assert c.state.verification.consent.status == "pending"
    c.say("She hasn't gotten anything yet")
    c.say("still nothing")
    c.say("nope")
    assert c.phase == "HUMAN_HANDOFF"
    assert not c.state.verification.verified
    assert "consent" in (c.state.handoff_reason or "")


def test_unlisted_representative_is_refused(make_convo):
    c = make_convo()
    r = c.say("I'm calling on behalf of my wife Ava Lopez, this is Carlos Lopez.")
    assert c.state.verification.rep_on_file is False
    assert "authorized representative" in r.reply
    assert not c.state.verification.verified
    assert_no_claim_leak(r.reply)


def test_verified_caller_can_reopen_after_close(convo):
    convo.say(DEMO_UTTERANCE)
    convo.say("that's all")
    convo.say("no")
    assert convo.phase == "CLOSED"
    r = convo.say("Actually, what's the status of my dental claim?")
    assert convo.phase == "PROCESS_CASE" and convo.state.selected_case_id == "CL-1899"
    assert "CL-1899" in r.reply


def test_representative_identity_is_sticky_across_partial_mentions(make_convo):
    """A later turn where the extractor only sees the rep's first name must not undo the lookup."""
    from sop_agent.sop.signals import TurnSignals

    c = make_convo()
    c.say("Hi, I'm calling on behalf of my mother, Margaret Chen. My name is David Chen, I'm her son.")
    assert c.state.verification.rep_on_file is True
    original = c.ctrl.extractor.extract

    def partial(text, state):
        sig = original(text, state)
        sig.caller_role = "representative"
        sig.representative = {"name": "David", "relationship": "son", "policyholder_name": None}
        return sig

    c.ctrl.extractor.extract = partial
    c.say("Her date of birth is March 15, 1985.")
    assert c.state.verification.rep_name == "David Chen"
    assert c.state.verification.rep_on_file is True
    assert c.state.verification.consent is not None          # reached the consent gate
