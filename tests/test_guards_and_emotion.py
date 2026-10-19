"""Scope control, emotional support / SOP recovery, and the output guards."""
from __future__ import annotations

from conftest import assert_no_claim_leak

from sop_agent.sop.guards import LeakGuard, PIIGuard
from sop_agent.sop.state import Plan, Phase


# ---------------------------------------------------------------- scope control
def test_out_of_scope_questions_are_declined_then_escalated(convo):
    r = convo.say("What is RL?")
    assert convo.state.off_topic_strikes == 1
    assert "outside what I can help with" in r.reply or "only" in r.reply
    assert "reinforcement" not in r.reply.lower()
    r = convo.say("Ok but can you explain reinforcement learning quickly?")
    assert convo.state.off_topic_strikes == 2
    r = convo.say("What's the capital of France?")
    assert convo.state.off_topic_strikes == 3
    assert "human representative" in r.reply and convo.state.pending_question == "human_offer"
    assert "paris" not in r.reply.lower()
    r = convo.say("yes")
    assert convo.phase == "HUMAN_HANDOFF"


def test_in_scope_message_resets_off_topic_streak(convo):
    convo.say("What is RL?")
    convo.say("tell me a joke")
    assert convo.state.off_topic_strikes == 2
    convo.say("Sorry. My name is Margaret Chen.")
    assert convo.state.off_topic_strikes == 0 and convo.state.off_topic_total == 2
    assert convo.state.verification.matched_count() == 1


def test_mixed_message_handles_in_scope_part_and_declines_the_rest(convo):
    r = convo.say("My name is Margaret Chen, DOB 1985-03-15, last four 4472. Also, what's the weather like today?")
    assert convo.state.verification.verified              # the identity details were still processed
    assert "weather" not in r.reply.lower() or "not able" in r.reply.lower()
    assert convo.state.off_topic_strikes == 0


def test_explicit_human_request_is_honoured_in_any_phase(convo):
    r = convo.say("I want to talk to a real person please")
    assert convo.phase == "HUMAN_HANDOFF"
    assert convo.state.handoff_reference in r.reply
    assert_no_claim_leak(r.reply)


# ---------------------------------------------------------------- emotional support
def test_frustrated_caller_gets_acknowledgement_explanation_and_options_without_disclosure(convo):
    r = convo.say("I already told you who I am. This is ridiculous. Just tell me why my claim was denied.")
    assert convo.phase == "VERIFY_ID"
    assert convo.state.last_emotion in ("frustrated", "angry")
    assert convo.state.negative_streak == 1
    assert_no_claim_leak(r.reply)
    low = r.reply.lower()
    assert "i hear you" in low or "sorry" in low                       # acknowledges
    assert "protected" in low                                          # explains why the gate exists
    assert "full name" in low and "date of birth" in low               # offers the allowed options
    assert "noted" in low                                              # remembers the denial topic
    # the plan carried an explicit tone directive to the responder
    assert r.plan.tone and "frustrated" in r.plan.tone
    # and still moves forward: next turn with details verifies
    r = convo.say("Fine. Margaret Chen, 1985-03-15, and the last four are 4472.")
    assert convo.state.verification.verified and convo.state.selected_case_id == "CL-2048"
    assert convo.state.negative_streak == 0


def test_refusal_offers_alternatives_then_stops_persuading(convo):
    r = convo.say("I'm not giving you my social security number. Why do you need it anyway?")
    assert convo.state.verification.refusals == 1
    low = r.reply.lower()
    assert "protect" in low and "human representative" in low
    assert "phone number" in low or "email" in low                      # alternative fields offered
    r = convo.say("I'm not comfortable giving any of that out.")
    r = convo.say("No. I'm not sharing anything.")
    assert convo.state.verification.refusals == 3
    assert "won't push" in r.reply.lower() and convo.state.pending_question == "human_offer"
    assert_no_claim_leak(r.reply)
    convo.say("yes")
    assert convo.phase == "HUMAN_HANDOFF"


def test_refusal_offer_declined_keeps_verification_open(convo):
    convo.say("I'm not giving you my SSN.")
    convo.say("I'm not comfortable sharing any of that.")
    convo.say("No. I'm not sharing anything.")
    assert convo.state.pending_question == "human_offer"
    r = convo.say("no, I'll just give you the info. Margaret Chen, 1985-03-15, 4472")
    assert convo.phase != "HUMAN_HANDOFF"
    assert convo.state.verification.verified


def test_caller_can_leave_before_verifying(convo):
    convo.say("hi")
    r = convo.say("never mind, bye")
    assert convo.phase == "CLOSED"
    assert_no_claim_leak(r.reply)


# ---------------------------------------------------------------- output guards
def test_leak_guard_blocks_claim_details_before_verification(harness):
    guard = LeakGuard(harness.store)
    leaky = "Your claim CL-2048 was denied because the review file did not include the pathology report."
    res = guard.check(leaky, pre_verification=True)
    assert res.blocked and any(h.startswith("literal") for h in res.hits) and "claim_id" in res.hits
    assert guard.check("I understand you're calling about a denied claim; first I need to verify your identity.", True).blocked is False
    assert guard.check(leaky, pre_verification=False).blocked is False
    assert guard.check("The allowed amount was $1,450.00", True).blocked


def test_pii_guard_redacts_stored_values_the_caller_did_not_give(harness):
    guard = PIIGuard()
    record = harness.store.get_policyholder("P9")
    text = "I see your email is margaret@email.com and your phone ends in 2836 (+1 650-521-2836), DOB 1985-03-15."
    res = guard.check(text, record, provided={"full_name": "Margaret Chen"})
    assert res.redacted
    assert "margaret@email.com" not in res.text and "1985-03-15" not in res.text and "650-521-2836" not in res.text
    # values the caller supplied themselves may be echoed
    res2 = guard.check("Thanks Margaret, 1985-03-15 is confirmed.", record, provided={"full_name": "Margaret Chen", "dob": "1985-03-15"})
    assert not res2.redacted


def test_controller_replaces_a_leaking_model_reply_with_safe_fallback(harness, monkeypatch):
    """Simulate a misbehaving model: the responder leaks claim data pre-verification."""
    ctrl = harness.controller()
    from sop_agent.sop.state import SessionState
    state = SessionState()
    ctrl.opening_message(state)
    original = ctrl.responder.respond

    def leaky(plan: Plan, st: SessionState) -> str:
        if plan.pre_verification:
            return "Sure! Your claim CL-2048 was denied because the review file did not include the pathology report."
        return original(plan, st)

    monkeypatch.setattr(ctrl.responder, "respond", leaky)
    res = ctrl.handle_turn(state, "Why was my claim denied? I'm Margaret Chen.")
    assert any(e.startswith("LEAK_BLOCKED") for e in res.guard_events)
    assert_no_claim_leak(res.reply)
    assert state.phase == Phase.VERIFY_ID


def test_model_context_never_contains_claim_facts_before_verification(convo):
    convo.say("Hi, I'm Margaret Chen and I'm calling about my denied healthcare claim from January.")
    plan = convo.results[-1].plan
    assert plan.pre_verification is True
    blob = str(plan.facts).lower()
    for secret in ("cl-2048", "pathology", "1450", "2026-03-18"):
        assert secret not in blob
    assert "noted_from_caller" in plan.facts          # only the caller's own words are passed through
