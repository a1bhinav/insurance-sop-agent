"""The SOP controller: a deterministic state machine that owns phase order,
safety gates and allowed actions. The LLM never decides *whether* a gate is
passed — it only reads the caller's words (extractor) and phrases the reply
(responder) inside the Plan the controller hands it.

Phase freedom dial
------------------
VERIFY_ID       strict    no claim facts in context; ≥N PII factors; attempt limits
RESOLVE_INTENT  guided    claim *briefs* in context; LLM phrases confirmation / disambiguation
PROCESS_CASE    flexible  full claim + guideline facts in context; LLM answers grounded questions
POST_PROCESS    guided    email-summary offer; explicit yes/no consent required
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional

from ..config import Settings
from ..data.store import Claim, DataStore
from ..data.tools import (
    ConsentService,
    EmailOutbox,
    claim_fact_bundle,
    hints_are_specific,
    humanize_date,
    resolve_claims,
)
from ..utils.normalize import mask_email, normalize_name
from .guards import OutputGuards
from .signals import PII_FIELDS, TurnSignals
from .state import (
    PENDING_ANYTHING_ELSE,
    PENDING_CHOOSE_CLAIM,
    PENDING_CONFIRM_CLAIM,
    PENDING_CONSENT_WAIT,
    PENDING_EMAIL_OFFER,
    PENDING_HUMAN_OFFER,
    ConsentState,
    Phase,
    Plan,
    SessionState,
    TurnRecord,
)
from .verification import VerifyOutcome, apply_identity_update, describe_mismatch, describe_options

CLAIM_INTENTS = {
    "status_inquiry", "denial_question", "document_submission", "next_steps",
    "appeal", "payment_question", "general_claim_question",
}

PRE_VERIFICATION_MUST_NOT = [
    "disclose, confirm or hint at any claim information (status, reason, amounts, dates, documents, claim numbers)",
    "state, confirm or correct any value from the record on file (do not say what the right DOB/phone/email/ID is)",
    "skip, shorten or promise to skip identity verification",
    "ask for the full SSN or full ID number — only the last four digits are ever needed",
]


@dataclass
class TurnResult:
    reply: str
    state: SessionState
    plan: Plan
    signals: TurnSignals
    guard_events: list[str] = field(default_factory=list)

    def trace(self) -> dict[str, Any]:
        return {
            "turn": self.state.turn_count,
            "phase_after": self.state.phase.value,
            "signals": self.signals.to_dict(),
            "events": list(self.plan.events) + list(self.guard_events),
            "directive": list(self.plan.directive),
            "facts_keys": sorted(self.plan.facts.keys()),
            "tone": self.plan.tone,
            "template": self.plan.template,
        }


class SOPController:
    def __init__(
        self,
        store: DataStore,
        extractor: Any,
        responder: Any,
        outbox: EmailOutbox,
        consent: ConsentService,
        settings: Settings,
    ):
        self.store = store
        self.extractor = extractor
        self.responder = responder
        self.outbox = outbox
        self.consent = consent
        self.settings = settings
        self.guards = OutputGuards(store)
        self.handlers: dict[Phase, Callable[[SessionState, TurnSignals, Plan], bool]] = {
            Phase.VERIFY_ID: self._verify_id,
            Phase.RESOLVE_INTENT: self._resolve_intent,
            Phase.PROCESS_CASE: self._process_case,
            Phase.POST_PROCESS: self._post_process,
        }

    # ------------------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------------------
    def today(self) -> str:
        return self.settings.demo_date or date.today().isoformat()

    def opening_message(self, state: SessionState) -> str:
        text = self.responder.opening(state)
        state.history.append(TurnRecord("agent", text, state.phase.value))
        return text

    def handle_turn(self, state: SessionState, user_text: str) -> TurnResult:
        user_text = (user_text or "").strip()
        state.turn_count += 1
        state.history.append(TurnRecord("user", user_text, state.phase.value))

        sig = self.extractor.extract(user_text, state)
        plan = Plan(phase=state.phase)
        plan.event(f"EXTRACTED ({sig.source}): scope={sig.scope} emotion={sig.emotion}/{sig.emotion_intensity}"
                   + (f" intent={sig.intent}" if sig.intent else "")
                   + (f" pii={sorted(sig.pii.keys())}" if sig.pii else ""))
        for ev in state.memory.absorb(sig, hold_questions=not state.verification.verified):
            plan.event(f"MEMORY {ev}")
        self._track_emotion(state, sig, plan)

        # A message that carries real content (identity details, a claim question, an
        # answer to our question) is in scope even if it also contains an aside.
        if sig.scope == "out_of_scope" and (sig.pii or sig.has_substantive_request or sig.answer in ("yes", "no")):
            sig.off_topic_part = sig.off_topic_part or user_text
            sig.scope = "in_scope"
            plan.event("SCOPE mixed message treated as in-scope (aside noted)")

        # ---- global interrupts (apply in every phase) --------------------------------
        if state.is_terminal:
            self._terminal(state, sig, plan)
        elif sig.requests_human:
            self._handoff(state, plan, reason="caller asked for a human representative")
        elif sig.scope == "out_of_scope":
            self._off_topic(state, sig, plan)
        else:
            if state.off_topic_strikes:
                plan.event("OFF_TOPIC streak reset")
            state.off_topic_strikes = 0
            if state.pending_question == PENDING_HUMAN_OFFER and sig.answer == "yes" and not sig.pii and not sig.questions:
                self._handoff(state, plan, reason="caller accepted the offer of a human representative")
            else:
                if state.pending_question == PENDING_HUMAN_OFFER:
                    # The offer is only live for one turn; a caller who moves on has implicitly declined.
                    state.pending_question = None
                    if sig.answer == "no":
                        plan.add("The caller declined the human transfer for now; continue the current step.")
                if sig.off_topic_part:
                    plan.facts["off_topic_aside"] = sig.off_topic_part
                    plan.add("The message also contained an unrelated question (off_topic_aside). Decline that part in one short clause "
                             "without answering it, and handle the in-scope part normally.")
                if sig.scope == "small_talk" and not sig.pii and not sig.has_substantive_request and sig.answer is None:
                    plan.add("The caller made small talk; respond warmly in one short sentence, then return to the current step.")
                self._run_phases(state, sig, plan)

        plan.tone = self._tone(state, sig)
        plan.phase = state.phase
        plan.facts.setdefault("caller_first_name", state.caller_first_name())
        plan.facts.setdefault("current_step", self._step_summary(state))

        # ---- speak, then guard -------------------------------------------------------
        text = self.responder.respond(plan, state)
        guard_events: list[str] = []
        guard = self.guards.apply(text, state, pre_verification=plan.pre_verification)
        if guard.blocked:
            guard_events.append(f"LEAK_BLOCKED {guard.hits} -> safe fallback used")
            text = self.responder.safe_fallback(plan, state)
        elif guard.redacted:
            guard_events.append(f"PII_REDACTED {guard.hits}")
            text = guard.text
        state.history.append(TurnRecord("agent", text, state.phase.value))
        result = TurnResult(reply=text, state=state, plan=plan, signals=sig, guard_events=guard_events)
        state.traces.append(result.trace())
        return result

    # ------------------------------------------------------------------------------
    # phase loop
    # ------------------------------------------------------------------------------
    def _run_phases(self, state: SessionState, sig: TurnSignals, plan: Plan) -> None:
        # The caller may wrap up from any phase.
        if sig.wants_to_end and not sig.pii and not sig.questions and not sig.has_case_hints:
            if state.phase == Phase.VERIFY_ID:
                state.pending_question = None
                plan.pre_verification = True
                plan.template = "closed_unverified"
                plan.add("The caller is ending the conversation before verification. Thank them, and mention they can come back any time "
                         "with their identifying details handy. No claim information.")
                state.transition(Phase.CLOSED, plan, "caller ended before verification")
                return
            if state.phase in (Phase.RESOLVE_INTENT, Phase.PROCESS_CASE):
                state.pending_question = None
                state.transition(Phase.POST_PROCESS, plan, "caller is wrapping up")

        # A single caller turn may legitimately move through several phases
        # (verified -> claim resolved -> question answered). Cap the chain.
        for _ in range(4):
            handler = self.handlers.get(state.phase)
            if handler is None:
                break
            if not handler(state, sig, plan):
                break

    # ------------------------------------------------------------------------------
    # VERIFY_ID (strict)
    # ------------------------------------------------------------------------------
    def _verify_id(self, state: SessionState, sig: TurnSignals, plan: Plan) -> bool:
        v = state.verification
        s = self.settings
        plan.pre_verification = True
        plan.allow_claim_details = False
        plan.must_not.extend(PRE_VERIFICATION_MUST_NOT)
        plan.facts["verification"] = {"required_factors": s.min_pii_factors}

        if v.locked:
            return self._verification_locked(state, plan)

        # -- representative path -------------------------------------------------------
        if sig.caller_role == "representative" or sig.representative:
            self._note_representative(state, sig, plan)

        # -- waiting on the policyholder's consent (representative calls) ----------------
        if v.caller_role == "representative" and v.consent and state.pending_question == PENDING_CONSENT_WAIT:
            return self._poll_consent(state, sig, plan)

        # -- absorb identity fields ------------------------------------------------------
        new_pii = dict(sig.pii)
        if new_pii:
            if state.pending_question == PENDING_HUMAN_OFFER:
                state.pending_question = None
            plan.event(f"VERIFY received fields {sorted(new_pii.keys())}")
        min_factors = s.rep_min_pii_factors if v.caller_role == "representative" else s.min_pii_factors
        if new_pii or v.provided:
            outcome = apply_identity_update(self.store, v, new_pii, min_factors, s.max_verify_attempts)
        else:
            outcome = VerifyOutcome(remaining=list(PII_FIELDS))
        plan.event(f"GATE identity: matched {outcome.matched_count}/{min_factors}"
                   + (f", mismatched {outcome.mismatched}" if outcome.mismatched else "")
                   + (", record not found" if outcome.unknown_record else "")
                   + (f", attempts {v.attempts}/{s.max_verify_attempts}" if v.attempts else ""))

        if v.caller_role == "representative":
            return self._verify_representative(state, sig, plan, outcome, min_factors)

        if outcome.verified:
            return self._mark_verified(state, plan, as_role="policyholder")

        if v.locked:
            return self._verification_locked(state, plan)

        self._ask_for_more_identity(state, sig, plan, outcome, min_factors)
        return False

    def _mark_verified(self, state: SessionState, plan: Plan, as_role: str) -> bool:
        v = state.verification
        v.verified = True
        v.verified_at = date.today().isoformat()
        state.pending_question = None
        plan.event(f"GATE identity PASSED ({as_role})")
        state.transition(Phase.RESOLVE_INTENT, plan, "identity verified")
        plan.add("Identity has just been verified: say so briefly and thank the caller by the first name they gave.")
        plan.template = "verified"
        return True

    def _verification_locked(self, state: SessionState, plan: Plan) -> bool:
        state.pending_question = PENDING_HUMAN_OFFER
        state.human_offered = True
        plan.template = "verify_locked"
        plan.facts["verification"]["locked"] = True
        plan.add(
            "Verification cannot continue over this channel: the allowed number of attempts has been used. "
            "Do not accept more identity details. Say that for the caller's protection a human representative "
            "needs to complete verification, and offer to transfer them now (yes/no)."
        )
        plan.event("GATE identity LOCKED -> human offer")
        return False

    def _ask_for_more_identity(self, state: SessionState, sig: TurnSignals, plan: Plan,
                               outcome: VerifyOutcome, min_factors: int) -> None:
        v = state.verification
        s = self.settings
        need = max(0, min_factors - outcome.matched_count)
        remaining_labels = describe_options(outcome.remaining)
        plan.template = "verify_ask"
        plan.template_args = {
            "need": need,
            "matched": outcome.matched_count,
            "options": remaining_labels,
            "mismatched": outcome.newly_mismatched,
            "unknown_record": outcome.unknown_record,
            "ambiguous": outcome.ambiguous,
            "attempts_left": max(0, s.max_verify_attempts - v.attempts),
        }
        plan.facts["verification"].update({
            "matched_so_far": outcome.matched_count,
            "still_needed": need,
            "fields_caller_can_offer": remaining_labels,
            "attempts_left": max(0, s.max_verify_attempts - v.attempts),
        })

        if outcome.ambiguous:
            plan.add("More than one record matches that name; ask for the policy number to locate the right one (do not list records).")
        elif outcome.needs_locator:
            plan.template = "verify_need_locator"
            plan.add("The caller gave some identifying details but not yet a full name or policy number, so no record can be looked up. "
                     "Thank them, and ask for their full name (or policy number) so you can check the details against the right record.")
        elif outcome.unknown_record:
            plan.add("Nothing on file matches the identifying details given so far. Ask the caller to double-check the spelling "
                     "or provide their policy number. Do not speculate about why.")
        elif outcome.newly_mismatched:
            labels = describe_mismatch(outcome.newly_mismatched)
            plan.facts["verification"]["fields_that_did_not_match"] = labels
            plan.add(f"Tell the caller that the {', '.join(labels)} they gave does not match our records (never say what the record holds). "
                     "Invite them to re-check it or offer a different identifying field instead.")
        if need > 0 and not outcome.ambiguous and not outcome.needs_locator:
            if outcome.matched_count:
                plan.add(f"{outcome.matched_count} identifying detail(s) have matched; {need} more are still required. "
                         f"Ask for {need} of: {', '.join(remaining_labels)}.")
            else:
                plan.add(f"Identity is not verified yet. Ask for {min_factors} identifying details, choosing from: {', '.join(remaining_labels)}. "
                         "If the caller already gave a full name, still ask for the remaining ones.")

        # memory-aware acknowledgement: we heard the reason for the call, we're just not acting on it yet
        meaningful_intent = [i for i in state.memory.intent_hints if i not in ("general_claim_question", "other")]
        if (state.memory.case_hints or meaningful_intent or state.memory.pending_questions) and not state.memory.hint_acknowledged:
            state.memory.hint_acknowledged = True
            plan.facts["noted_from_caller"] = self._hint_summary(state)
            plan.add("The caller already said what they are calling about (see noted_from_caller). Acknowledge in one clause that you've "
                     "noted it and will pull it up the moment verification is done — using only the caller's own words, adding no claim facts.")
            plan.event("MEMORY hint acknowledged (held until verification)")

        if sig.refuses_verification:
            v.refusals += 1
            plan.event(f"REFUSAL {v.refusals}/{s.refusal_limit}")
            if v.refusals >= s.refusal_limit or state.negative_streak >= 3:
                state.pending_question = PENDING_HUMAN_OFFER
                state.human_offered = True
                plan.template = "verify_refused_offer_human"
                plan.add("Stop persuading. Acknowledge their choice without judgement, say plainly that claim details cannot be shared "
                         "without verification on any channel, and offer a human representative (yes/no). Keep the door open to continue later.")
            else:
                plan.add("The caller is reluctant to share identifying details. Explain in one sentence why the step exists (it protects "
                         "their own claim information from anyone who isn't them), then offer the alternatives: any of the listed fields "
                         "works, or a human representative can help.")
        elif sig.asks_why_needed or (sig.is_negative and not v.why_explained):
            v.why_explained = True
            plan.add("Briefly explain why verification is required: claim details are protected information, and the check exists so "
                     "nobody else can access their claim. Then restate what is still needed.")
        elif state.negative_streak >= 3 and not state.human_offered:
            state.pending_question = PENDING_HUMAN_OFFER
            state.human_offered = True
            plan.add("The caller has been upset for several turns. Offer a human representative as an alternative (yes/no) while making "
                     "clear that the human will also need to verify identity.")

    def _hint_summary(self, state: SessionState) -> dict[str, Any]:
        mem = state.memory
        out: dict[str, Any] = {}
        if mem.case_hints:
            out["about"] = {k: v for k, v in mem.case_hints.items() if k in ("case_type", "status", "time_reference", "case_id")}
        if mem.intent_hints:
            out["wants"] = mem.intent_hints[-1].replace("_", " ")
        if mem.pending_questions:
            out["asked"] = mem.pending_questions[-1]
        return out

    # -- representative sub-path ----------------------------------------------------------
    def _note_representative(self, state: SessionState, sig: TurnSignals, plan: Plan) -> None:
        v = state.verification
        v.caller_role = "representative"
        rep = sig.representative
        new_name = rep.get("name")
        # A full name is needed to look anyone up; never let a later partial mention
        # ("Thanks, David") overwrite the full name we already have.
        if new_name and (len(new_name.split()) >= 2 or not v.rep_name):
            changed = normalize_name(new_name) != normalize_name(v.rep_name or "")
            v.rep_name = new_name
            if changed:
                v.rep_on_file = None   # a genuinely different name must be looked up again
        if rep.get("relationship"):
            v.rep_relationship = rep["relationship"]
        holder = rep.get("policyholder_name")
        if holder and (len(holder.split()) >= 2 or not v.provided.get("full_name")):
            if normalize_name(holder) != normalize_name(v.provided.get("full_name", "")):
                v.rep_on_file = None
            v.provided["full_name"] = holder
        # Look up once per (rep, policyholder) pair; the result is sticky after that.
        if v.rep_on_file is None and v.rep_name and len(v.rep_name.split()) >= 2 and v.provided.get("full_name"):
            found = self.store.find_representative(v.rep_name, v.provided["full_name"])
            v.rep_on_file = bool(found)
            if found and not v.rep_relationship:
                v.rep_relationship = found[0].relationship
            plan.event(f"REP lookup {v.rep_name!r} for policyholder: {'on file' if found else 'NOT on file'}")

    def _verify_representative(self, state: SessionState, sig: TurnSignals, plan: Plan,
                               outcome: VerifyOutcome, min_factors: int) -> bool:
        v = state.verification
        plan.facts["verification"]["caller_role"] = "representative"
        plan.template = "rep_ask"
        if not v.rep_name:
            plan.add("The caller says they are calling on someone's behalf. Ask for their own full name, their relationship to the "
                     "policyholder, and the policyholder's full name.")
            return False
        if not v.provided.get("full_name"):
            plan.add("Ask for the policyholder's full name so the record can be located.")
            return False
        if v.rep_on_file is False:
            state.pending_question = PENDING_HUMAN_OFFER
            state.human_offered = True
            plan.template = "rep_not_on_file"
            plan.add("Explain that you don't see the caller listed as an authorized representative for that policyholder, so you cannot "
                     "discuss the account with them. Options: the policyholder can call directly, or a human representative can explain how "
                     "to add an authorized representative (offer to transfer, yes/no). Do not confirm whether the policyholder exists.")
            return False
        if outcome.unknown_record:
            plan.add("No record matches the policyholder details given. Ask them to double-check the policyholder's name or give the policy number.")
            return False
        if outcome.newly_mismatched:
            labels = describe_mismatch(outcome.newly_mismatched)
            plan.facts["verification"]["fields_that_did_not_match"] = labels
            plan.add(f"The policyholder's {', '.join(labels)} given does not match our records (never reveal the stored value). Ask them to re-check.")
        if outcome.matched_count < min_factors:
            need = min_factors - outcome.matched_count
            options = describe_options([f for f in outcome.remaining if f != "full_name"])
            plan.facts["verification"].update({"matched_so_far": outcome.matched_count, "still_needed": need,
                                               "fields_caller_can_offer": options})
            plan.add(f"Before consent can be requested, {need} more identifying detail(s) about the policyholder are needed. "
                     f"Ask for {need} of: {', '.join(options)}.")
            return False
        if v.locked:
            return self._verification_locked(state, plan)
        # authorized rep + enough factors -> request the policyholder's consent (a gate the LLM cannot skip)
        req = self.consent.request(v.candidate_party_id or "", state.consent_scenario)
        initial = req.check()  # the request itself reads the first status (normally "pending")
        v.consent = ConsentState(consent_id=req.consent_id, status=initial, checks=req.checks, scenario=state.consent_scenario)
        state.consent_request = req
        state.pending_question = PENDING_CONSENT_WAIT
        plan.template = "consent_requested"
        plan.facts["consent"] = {"status": "requested", "sent_to": "the policyholder's phone on file", "reference": req.consent_id}
        plan.add("The caller is listed as an authorized representative and the policyholder's details check out. Explain that, for the "
                 "policyholder's protection, a one-time consent request has just been sent to the policyholder's phone on file, and ask the "
                 "caller to let you know once it has been approved. No claim details yet.")
        plan.event(f"CONSENT requested {req.consent_id} scenario={state.consent_scenario}")
        return False

    def _poll_consent(self, state: SessionState, sig: TurnSignals, plan: Plan) -> bool:
        v = state.verification
        req = state.consent_request
        if req is None or v.consent is None:
            state.pending_question = None
            return False
        status = req.check()
        v.consent.status, v.consent.checks = status, req.checks
        plan.facts["consent"] = {"status": status, "reference": v.consent.consent_id}
        plan.event(f"CONSENT check #{req.checks}: {status}")
        if status == "approved":
            plan.template = "consent_approved"
            plan.add("The policyholder's consent has been approved.")
            return self._mark_verified(state, plan, as_role="representative")
        if status in ("denied", "rejected"):
            plan.template = "consent_denied"
            state.pending_question = None
            v.locked = True
            plan.add("The policyholder declined the consent request. Explain that you cannot discuss the account without it; the "
                     "policyholder can call directly.")
            return False
        if req.checks >= self.settings.consent_max_checks:
            self._handoff(state, plan, reason="policyholder consent not received in time")
            return False
        plan.template = "consent_pending"
        plan.add("Consent is still showing as pending. Say so, suggest the policyholder check their phone, and offer to wait or to "
                 "transfer to a human representative if it keeps not arriving.")
        return False

    # ------------------------------------------------------------------------------
    # RESOLVE_INTENT (guided)
    # ------------------------------------------------------------------------------
    def _party_id(self, state: SessionState) -> str:
        return state.verification.candidate_party_id or ""

    def _select_claim(self, state: SessionState, claim: Claim, plan: Plan, how: str) -> None:
        state.selected_case_id = claim.case_id
        if claim.case_id not in state.claims_discussed:
            state.claims_discussed.append(claim.case_id)
        state.candidate_claims = []
        state.pending_question = None
        plan.event(f"CLAIM selected {claim.case_id} ({how})")

    def _resolve_intent(self, state: SessionState, sig: TurnSignals, plan: Plan) -> bool:
        plan.pre_verification = False
        plan.allow_claim_details = True
        plan.must_not.append("quote denial reasons, amounts or documents in this step — only identify which claim is meant")
        party = self._party_id(state)
        claims = self.store.claims_for_party(party)
        mem = state.memory

        if sig.intent == "update_contact_info" or (mem.primary_intent == "update_contact_info" and not sig.has_case_hints):
            return self._offer_human_for_unsupported(state, plan, "updating contact or account details")
        if sig.intent == "coverage_question":
            return self._offer_human_for_unsupported(state, plan, "questions about what a policy covers")

        # answers to a previous disambiguation question
        if state.pending_question in (PENDING_CHOOSE_CLAIM, PENDING_CONFIRM_CLAIM):
            if state.pending_question == PENDING_CONFIRM_CLAIM and sig.answer == "no" and not sig.has_case_hints:
                mem.clear_case_hints()
                state.pending_question = None
                plan.add("The caller says that is not the claim they meant; clear the earlier assumption.")

        hints = dict(mem.case_hints)
        candidates = resolve_claims(self.store, party, hints) if hints_are_specific(hints) else list(claims)
        briefs = [c.brief() for c in claims]

        if not claims:
            plan.template = "no_claims"
            plan.facts["claims_on_file"] = []
            plan.add("There are no claims on file for this policyholder. Say so and offer a human representative for anything else.")
            state.pending_question = PENDING_HUMAN_OFFER
            state.human_offered = True
            return False

        if hints_are_specific(hints) and len(candidates) == 1:
            claim = candidates[0]
            self._select_claim(state, claim, plan, how="resolved from remembered hints" if mem.hint_acknowledged else "resolved from hints")
            plan.add(f"You matched the claim the caller described ({claim.case_type} claim {claim.case_id}, filed {humanize_date(claim.created_at)}, "
                     f"status {claim.status}). Name it so they can correct you, then continue.")
            state.transition(Phase.PROCESS_CASE, plan, "single claim matched the caller's description")
            return True

        if len(claims) == 1 and not hints_are_specific(hints):
            self._select_claim(state, claims[0], plan, how="only claim on file")
            plan.add(f"The caller has one claim on file ({claims[0].case_type} claim {claims[0].case_id}); say you've pulled it up.")
            state.transition(Phase.PROCESS_CASE, plan, "only one claim on file")
            return True

        state.pending_question = PENDING_CHOOSE_CLAIM
        plan.template = "choose_claim"
        state.candidate_claims = [c.brief() for c in (candidates if hints_are_specific(hints) and len(candidates) > 1 else claims)]
        if hints_are_specific(hints) and len(candidates) > 1:
            plan.facts["candidate_claims"] = [c.brief() for c in candidates]
            plan.template_args = {"candidates": plan.facts["candidate_claims"], "mode": "several_match"}
            plan.add("Several claims match the caller's description. List the matching claims briefly (type, filing date, status, claim number) "
                     "and ask which one they mean. Do not go into denial reasons or amounts yet.")
        elif hints_are_specific(hints) and not candidates:
            plan.facts["claims_on_file"] = briefs
            plan.facts["caller_described"] = {k: v for k, v in hints.items() if k in ("case_type", "status", "time_reference", "case_id")}
            plan.template_args = {"candidates": briefs, "mode": "none_match"}
            plan.add("No claim matches the description the caller gave (caller_described). Say that plainly, list the claims that are on "
                     "file briefly, and ask which one they'd like to discuss.")
            mem.clear_case_hints()  # the description didn't work; start fresh next turn
        else:
            plan.facts["claims_on_file"] = briefs
            plan.template_args = {"candidates": briefs, "mode": "list"}
            plan.add("Ask what the caller needs help with today. List their claims briefly (type, filing date, status, claim number) so "
                     "they can pick one.")
        return False

    def _offer_human_for_unsupported(self, state: SessionState, plan: Plan, what: str) -> bool:
        state.pending_question = PENDING_HUMAN_OFFER
        state.human_offered = True
        plan.template = "unsupported_intent"
        plan.facts["unsupported_request"] = what
        plan.add(f"This assistant handles questions about existing claims; {what} must be done by a human representative. "
                 "Say so kindly and offer to transfer (yes/no), or to continue with a claim question instead.")
        return False

    # ------------------------------------------------------------------------------
    # PROCESS_CASE (flexible, grounded)
    # ------------------------------------------------------------------------------
    def _process_case(self, state: SessionState, sig: TurnSignals, plan: Plan) -> bool:
        plan.pre_verification = False
        plan.allow_claim_details = True
        party = self._party_id(state)
        claim = self.store.get_claim(state.selected_case_id or "")
        if claim is None:
            state.transition(Phase.RESOLVE_INTENT, plan, "no claim selected")
            return True

        if sig.intent == "update_contact_info":
            return self._offer_human_for_unsupported(state, plan, "updating contact or account details")
        if sig.intent == "coverage_question":
            return self._offer_human_for_unsupported(state, plan, "questions about what a policy covers")

        # caller may switch to another claim mid-conversation
        if sig.has_case_hints:
            cands = resolve_claims(self.store, party, sig.case_hints)
            if cands and claim.case_id not in [c.case_id for c in cands]:
                if len(cands) == 1:
                    claim = cands[0]
                    self._select_claim(state, claim, plan, how="caller switched claims")
                    plan.add(f"The caller is now asking about a different claim: {claim.case_type} claim {claim.case_id} "
                             f"(filed {humanize_date(claim.created_at)}, {claim.status}). Say you've switched to it.")
                else:
                    state.pending_question = PENDING_CHOOSE_CLAIM
                    state.candidate_claims = [c.brief() for c in cands]
                    plan.facts["candidate_claims"] = [c.brief() for c in cands]
                    plan.template = "choose_claim"
                    plan.template_args = {"candidates": plan.facts["candidate_claims"], "mode": "several_match"}
                    plan.add("Several claims match what the caller just described; list them briefly and ask which one.")
                    state.transition(Phase.RESOLVE_INTENT, plan, "ambiguous claim reference")
                    return False
            elif not cands and sig.case_hints.get("case_id"):
                plan.add(f"The caller mentioned claim {sig.case_hints['case_id']}, which is not on their file; say so and stay with the current claim.")

        # wrapping up?
        answered_no_more = state.pending_question == PENDING_ANYTHING_ELSE and sig.answer == "no" and not sig.questions
        if sig.wants_to_end or answered_no_more:
            state.pending_question = None
            state.transition(Phase.POST_PROCESS, plan, "caller has no further questions")
            return True

        # "yes, something else" with no question yet
        if (state.pending_question == PENDING_ANYTHING_ELSE and sig.answer == "yes"
                and not sig.questions and not sig.has_case_hints and sig.intent in (None, "other")):
            plan.template = "ask_what_else"
            plan.facts["claim_brief"] = claim.brief()
            plan.add("The caller has another question but hasn't said what it is yet; invite them to go ahead (one sentence).")
            state.pending_question = None
            return False

        # grounded Q&A
        question_text = state.history[-1].text if state.history else ""
        intent = sig.intent if sig.intent in CLAIM_INTENTS else (state.memory.primary_intent or "general_claim_question")
        plan.facts.update(claim_fact_bundle(self.store, claim, question_text, intent, today=self.today()))
        plan.template = "answer_claim"
        first_answer = any(e.startswith("CLAIM selected") for e in plan.events)
        plan.template_args = {"intent": intent, "questions": list(sig.questions), "first_answer": first_answer}
        plan.must_not.extend([
            "invent, estimate or extrapolate anything that is not in FACTS (no made-up timelines, policies or amounts)",
            "promise outcomes (approval, payment, reversal of a denial)",
        ])
        topic = intent
        if topic and topic not in state.topics_covered:
            state.topics_covered.append(topic)

        earlier = [q for q in state.memory.pending_questions]
        if earlier and not state.earlier_questions_served:
            state.earlier_questions_served = True
            plan.facts["questions_asked_before_verification"] = earlier
            plan.add("Also answer the question(s) the caller asked before verification (questions_asked_before_verification) — they should not have to repeat themselves.")

        if sig.questions or intent in CLAIM_INTENTS:
            plan.add("Answer the caller's question(s) directly from FACTS in plain language. Lead with the answer, then the one or two "
                     "most useful next steps. If FACTS do not cover something, say you don't have that detail and offer a human representative.")
        else:
            plan.add("The caller hasn't asked a specific question about this claim yet; give a one-sentence status from FACTS and ask what they'd like to know.")
        if claim.status == "denied" and claim.documents_needed:
            plan.add("Since the claim is denied for missing documents, make sure the caller leaves knowing which documents are needed and the appeal deadline (from FACTS).")
        if plan.facts.get("appeal_deadline_has_passed"):
            plan.add("Note: the appeal deadline in FACTS is already in the past relative to today; say so plainly and offer a human representative to review options rather than guessing.")
        plan.add("Finish by asking whether there is anything else you can help with on this or another claim.")
        state.pending_question = PENDING_ANYTHING_ELSE
        return False

    # ------------------------------------------------------------------------------
    # POST_PROCESS (guided)
    # ------------------------------------------------------------------------------
    def _post_process(self, state: SessionState, sig: TurnSignals, plan: Plan) -> bool:
        plan.pre_verification = False
        plan.allow_claim_details = True
        record = self.store.get_policyholder(self._party_id(state))
        masked = mask_email(record.email if record else None)
        plan.facts["email_summary_offer"] = {"to": masked, "contains": ["what was discussed", "claim status/outcome", "follow-up items and next steps"]}
        if state.verification.caller_role == "representative":
            plan.facts["email_summary_offer"]["note"] = "the summary goes to the policyholder's email on file, not the representative's"

        if state.pending_question != PENDING_EMAIL_OFFER:
            pref = state.memory.email_preference
            if pref == "yes":
                return self._send_summary(state, plan, record)
            if pref == "no":
                return self._close(state, plan, emailed=False, note="the caller said earlier they did not want an email")
            state.pending_question = PENDING_EMAIL_OFFER
            plan.template = "offer_email"
            plan.add("Wrap up: in one sentence recap what was covered, then offer to email a summary of the conversation (what was "
                     "discussed, the claim status/outcome, follow-up items) to the masked address in FACTS. Make clear they can say yes or skip it.")
            return False

        # the offer is on the table
        if sig.answer == "yes" or sig.email_preference == "yes":
            return self._send_summary(state, plan, record)
        if sig.answer == "no" or sig.email_preference == "no" or (sig.wants_to_end and not sig.questions):
            return self._close(state, plan, emailed=False)
        if sig.questions or sig.has_case_hints or (sig.intent in CLAIM_INTENTS):
            state.pending_question = None
            plan.add("The caller has another claim question after all; go back to helping with it.")
            state.transition(Phase.PROCESS_CASE if state.selected_case_id else Phase.RESOLVE_INTENT, plan, "caller re-opened a claim question")
            return True
        plan.template = "offer_email_clarify"
        plan.add("It isn't clear whether they want the email summary. Ask again plainly: send it to the masked address, or skip it?")
        return False

    def _send_summary(self, state: SessionState, plan: Plan, record) -> bool:
        to = record.email if record else "unknown@example.com"
        subject, body = self.responder.compose_email(state, self._summary_context(state))
        email = self.outbox.deliver(to=to, subject=subject, body=body, session_id=state.session_id,
                                    meta={"claims": list(state.claims_discussed), "topics": list(state.topics_covered)})
        state.email_sent_id = email.message_id
        plan.facts["email_sent"] = {"to": mask_email(to), "message_id": email.message_id}
        plan.event(f"EMAIL_SENT {email.message_id} -> {mask_email(to)}")
        return self._close(state, plan, emailed=True)

    def _close(self, state: SessionState, plan: Plan, emailed: bool, note: str = "") -> bool:
        state.pending_question = None
        state.transition(Phase.CLOSED, plan, "conversation complete")
        plan.template = "closed_with_email" if emailed else "closed_no_email"
        if emailed:
            plan.add("Confirm the summary email has been sent to the masked address, mention the key next step once, and close warmly.")
        else:
            plan.add("Confirm that no email will be sent" + (f" ({note})" if note else "") + ", mention the key next step once, and close warmly.")
        return False

    def _summary_context(self, state: SessionState) -> dict[str, Any]:
        claims = [self.store.get_claim(cid) for cid in state.claims_discussed]
        bundles = [claim_fact_bundle(self.store, c, today=self.today()) for c in claims if c]
        transcript = [{"role": t.role, "text": t.text} for t in state.history if t.role in ("user", "agent")]
        return {
            "company": self.settings.company_name,
            "agent_name": self.settings.agent_name,
            "caller_first_name": state.caller_first_name(),
            "caller_role": state.verification.caller_role,
            "topics_covered": list(state.topics_covered),
            "claims": bundles,
            "transcript": transcript,
            "today": self.today(),
        }

    # ------------------------------------------------------------------------------
    # terminal phases, interrupts, tone
    # ------------------------------------------------------------------------------
    def _terminal(self, state: SessionState, sig: TurnSignals, plan: Plan) -> None:
        plan.pre_verification = not state.verification.verified
        if state.phase == Phase.HUMAN_HANDOFF:
            plan.template = "handoff_wait"
            plan.facts["handoff"] = {"reference": state.handoff_reference}
            plan.add("This conversation has been handed to a human representative. Say they'll be with the caller shortly, repeat the "
                     "reference number, and do not attempt to help with the request yourself.")
            return
        # CLOSED: allow a verified caller to re-open with a new claim question
        if state.verification.verified and sig.has_substantive_request:
            plan.add("The caller came back with another question after we wrapped up; pick it up naturally.")
            state.transition(Phase.PROCESS_CASE if state.selected_case_id else Phase.RESOLVE_INTENT, plan, "re-opened after close")
            self._run_phases(state, sig, plan)
            return
        plan.template = "closed_followup"
        plan.add("The conversation has already been wrapped up. Thank them again briefly; if they need anything else they can simply ask "
                 "(a verified caller may continue) or start a new session.")

    def _off_topic(self, state: SessionState, sig: TurnSignals, plan: Plan) -> None:
        s = self.settings
        state.off_topic_strikes += 1
        state.off_topic_total += 1
        plan.pre_verification = not state.verification.verified
        plan.facts["off_topic"] = {"strike": state.off_topic_strikes, "limit": s.off_topic_limit,
                                   "question": sig.off_topic_part or (state.history[-1].text if state.history else "")}
        plan.event(f"OFF_TOPIC strike {state.off_topic_strikes}/{s.off_topic_limit}")
        if state.off_topic_strikes >= s.off_topic_limit:
            if state.human_offered and state.pending_question == PENDING_HUMAN_OFFER:
                self._handoff(state, plan, reason="repeated out-of-scope requests after a human was offered")
                return
            state.pending_question = PENDING_HUMAN_OFFER
            state.human_offered = True
            plan.template = "off_topic_offer_human"
            plan.add("This is the caller's repeated out-of-scope request. Decline politely in one sentence (no answer, no hints), explain that "
                     "this line only handles insurance claims matters, and offer to connect them with a human representative who may be able to "
                     "point them elsewhere (yes/no).")
        else:
            plan.template = "off_topic"
            plan.add("The caller asked something unrelated to insurance claims support. Decline politely in one sentence without answering it "
                     "(not even partially), then steer back to the current step (see current_step). Keep any acknowledgement of in-scope "
                     "content in the same message.")

    def _handoff(self, state: SessionState, plan: Plan, reason: str) -> None:
        state.handoff_reference = state.handoff_reference or f"HR-{uuid.uuid4().hex[:6].upper()}"
        state.handoff_reason = reason
        state.pending_question = None
        plan.pre_verification = not state.verification.verified
        plan.template = "handoff"
        plan.facts["handoff"] = {
            "reference": state.handoff_reference,
            "reason": reason,
            "identity_verified": state.verification.verified,
            "what_the_human_will_do": "complete identity verification if needed, then help with the request",
        }
        plan.add("Transfer the caller to a human representative: say you're doing it now, give the reference number, and note that the "
                 "representative will complete verification first if it isn't done. Keep it to two or three sentences; no claim details "
                 "unless identity_verified is true.")
        state.transition(Phase.HUMAN_HANDOFF, plan, reason)

    def _track_emotion(self, state: SessionState, sig: TurnSignals, plan: Plan) -> None:
        state.last_emotion = sig.emotion
        if sig.is_negative:
            state.negative_streak += 1
            plan.event(f"EMOTION {sig.emotion} (intensity {sig.emotion_intensity}), negative streak {state.negative_streak}")
        else:
            state.negative_streak = 0

    def _tone(self, state: SessionState, sig: TurnSignals) -> Optional[str]:
        e, i = sig.emotion, sig.emotion_intensity
        lines: list[str] = []
        if e in ("frustrated", "angry") and i >= 1:
            lines.append("The caller is frustrated. Open with one short, specific acknowledgement of what is annoying them (not a generic "
                         "'I understand'), do not get defensive or apologise repeatedly, then move to the concrete next action.")
        elif e == "anxious" and i >= 1:
            lines.append("The caller sounds worried. Be calm and reassuring, be concrete about what happens next, avoid jargon.")
        elif e == "confused" and i >= 1:
            lines.append("The caller is confused. Slow down: one step at a time, short sentences, no lists of more than three items.")
        elif e == "refusing":
            lines.append("The caller is pushing back. Don't argue; explain the 'why' in one sentence, offer alternatives, and make clear you cannot bypass the step.")
        elif e == "grateful":
            lines.append("The caller is appreciative; be warm and brief.")
        if state.negative_streak >= 2:
            lines.append("They have been upset for more than one turn: stop re-explaining policy at length, keep the reply short, and make "
                         "the human-representative option explicit.")
        return " ".join(lines) if lines else None

    def _step_summary(self, state: SessionState) -> str:
        p = state.phase
        v = state.verification
        if p == Phase.VERIFY_ID:
            if state.pending_question == PENDING_CONSENT_WAIT:
                return "waiting for the policyholder's consent approval"
            need = max(0, (self.settings.rep_min_pii_factors if v.caller_role == "representative" else self.settings.min_pii_factors) - v.matched_count())
            return f"identity verification ({need} more identifying detail(s) needed)"
        if p == Phase.RESOLVE_INTENT:
            return "finding out which claim the caller needs help with"
        if p == Phase.PROCESS_CASE:
            return f"helping with claim {state.selected_case_id}"
        if p == Phase.POST_PROCESS:
            return "wrapping up and offering an email summary"
        if p == Phase.HUMAN_HANDOFF:
            return "transferred to a human representative"
        return "conversation closed"
