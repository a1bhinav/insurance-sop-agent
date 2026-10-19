"""Responders turn a Plan into the agent's words.

TemplateResponder – deterministic phrasing. Used in mock mode, as the fallback when
                    the model call fails, and as the *safe fallback* whenever the
                    output guards block a generated reply.
LLMResponder      – phase-constrained generation with the model; delegates to the
                    template responder for the opening line and on any failure.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..config import Settings
from ..data.tools import humanize_date
from ..sop.state import Plan, SessionState
from .json_utils import parse_json_object
from .prompts import EMAIL_SYSTEM, email_user_prompt, responder_system_prompt
from .provider import LLMError, LLMProvider
from .schemas import EMAIL_SCHEMA

log = logging.getLogger(__name__)


def _join(items: list[str], final: str = "or") -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {final} {items[1]}"
    return ", ".join(items[:-1]) + f", {final} " + items[-1]


def _claim_line(c: dict[str, Any]) -> str:
    return f"{c['case_type']} claim {c['case_id']} filed {humanize_date(c['created_at'])} ({c['status']})"


class TemplateResponder:
    name = "template"

    def __init__(self, settings: Settings):
        self.s = settings

    # ---- entry points -----------------------------------------------------------------
    def opening(self, state: SessionState) -> str:
        return (f"Hi, this is {self.s.agent_name} with {self.s.company_name} claims support. I can help with questions about your claims. "
                f"Before I can look at any claim details I need to verify your identity — could you start with your full name?")

    def respond(self, plan: Plan, state: SessionState) -> str:
        return self.render(plan, state)

    def safe_fallback(self, plan: Plan, state: SessionState) -> str:
        return self.render(plan, state)

    # ---- rendering ---------------------------------------------------------------------
    def render(self, plan: Plan, state: SessionState) -> str:
        parts: list[str] = []
        empathy = self._empathy(plan, state)
        if empathy:
            parts.append(empathy)
        if plan.facts.get("off_topic_aside"):
            parts.append("I'm not able to help with that one — I can only assist with insurance claims matters here.")
        if any(e.startswith("GATE identity PASSED") for e in plan.events):
            name = state.caller_first_name()
            parts.append(f"Thank you{', ' + name if name else ''} — your identity is verified.")
        body = getattr(self, f"_t_{plan.template}", self._t_generic)(plan, state)
        if body:
            parts.append(body)
        return " ".join(p.strip() for p in parts if p and p.strip())

    def _empathy(self, plan: Plan, state: SessionState) -> str:
        e = state.last_emotion
        if plan.template in ("handoff", "handoff_wait"):
            return ""
        if e in ("frustrated", "angry"):
            return "I hear you, and I'm sorry this is taking extra steps."
        if e == "anxious":
            return "I understand this is stressful — let's take it one step at a time."
        if e == "confused":
            return "No problem, let me make this simpler."
        if e == "refusing":
            return "That's completely your call."
        return ""

    def _noted(self, plan: Plan) -> str:
        noted = plan.facts.get("noted_from_caller")
        if not noted:
            return ""
        about = noted.get("about") or {}
        bits = [about.get("status"), about.get("case_type"), "claim"]
        desc = " ".join(b for b in bits if b)
        when = about.get("time_reference")
        if when:
            desc += f" from {when}"
        if about.get("case_id"):
            desc += f" ({about['case_id']})"
        if not about and noted.get("wants"):
            wants = {"denial question": "why your claim was denied", "status inquiry": "the status of your claim",
                     "document submission": "submitting documents for your claim", "next steps": "the next steps on your claim",
                     "appeal": "appealing your claim", "payment question": "the payment on your claim"}.get(noted["wants"], "your claim")
            return f"I've noted that you're asking about {wants} — I'll get to that as soon as we've finished verifying your identity."
        if not about and noted.get("asked"):
            return "I've noted your question and I'll come back to it as soon as we've finished verifying your identity."
        return f"I've noted that you're calling about your {desc} — I'll pull that up as soon as we've finished verifying your identity."

    # ---- VERIFY_ID ----------------------------------------------------------------------
    def _t_verify_ask(self, plan: Plan, state: SessionState) -> str:
        a = plan.template_args
        v = plan.facts.get("verification", {})
        name = state.caller_first_name()
        bits: list[str] = []
        if a.get("ambiguous"):
            bits.append("I'm finding more than one record under that name — could you give me your policy number so I pick the right one?")
            return " ".join([self._noted(plan), *bits]).strip()
        if a.get("unknown_record"):
            bits.append("I'm not finding a record that matches those details. Could you double-check the spelling of your name, or give me your policy number?")
        if a.get("mismatched"):
            labels = v.get("fields_that_did_not_match") or a["mismatched"]
            bits.append(f"The {_join(labels, 'and')} you gave doesn't match what we have on file — could you re-check it, or offer a different detail instead?"
                        + (f" (You have {a['attempts_left']} more attempt{'s' if a['attempts_left'] != 1 else ''} before I'd need to hand you to a human representative.)"
                           if a.get("attempts_left", 3) <= 1 else ""))
        need, matched = a.get("need", 3), a.get("matched", 0)
        options = a.get("options", [])
        why = ""
        if any("explain why" in d.lower() or "why verification" in d.lower() for d in plan.directive):
            why = "Claim details are protected information, so I have to confirm I'm speaking with the right person before I can open anything — it's there to protect your own account."
        if any("stop persuading" in d.lower() for d in plan.directive):
            return " ".join(filter(None, [
                why, "I can't share any claim details without verification on this or any other channel. If you'd prefer, I can transfer you to a human representative now — would you like that?"]))
        if any("reluctant" in d.lower() for d in plan.directive):
            bits.append(why or "The verification step exists to protect your claim information from anyone who isn't you.")
            bits.append(f"Any {need} of these work: {_join(options, 'or')}. Or, if you'd rather, I can connect you with a human representative.")
            return " ".join(filter(None, [self._noted(plan), *bits]))
        if why:
            bits.append(why)
        if need > 0 and not a.get("unknown_record"):
            if not options:
                bits.append(f"I still need {need} more matching detail{'s' if need != 1 else ''} — please re-check the one that didn't match and give it to me again.")
            elif matched:
                bits.append(f"{'Thanks' + (', ' + name if name else '')} — I still need {need} more detail{'s' if need != 1 else ''}: {_join(options, 'or')}.")
            else:
                bits.append(f"To get started{', ' + name if name else ''}, could you give me {need} identifying details — for example your {_join(options[:3], 'and')}?")
        if any("upset for several turns" in d.lower() for d in plan.directive):
            bits.append("If you'd prefer, I can also transfer you to a human representative — they'll need to verify identity too, but they may be able to help differently. Would you like that?")
        return " ".join(filter(None, [self._noted(plan), *bits]))

    def _t_verify_need_locator(self, plan: Plan, state: SessionState) -> str:
        return " ".join(filter(None, [self._noted(plan),
                                      "Thanks for those details. To check them against the right record I'll need your full name (or your policy number) — could you give me that?"]))

    def _t_verify_locked(self, plan: Plan, state: SessionState) -> str:
        return ("For your protection I can't keep trying to verify over this channel after that many unsuccessful attempts. "
                "A human representative can complete verification with you — would you like me to transfer you now?")

    def _t_verify_refused_offer_human(self, plan: Plan, state: SessionState) -> str:
        return ("I won't push further. Just so it's clear: claim details can't be shared without verification on this or any channel, but the moment you're "
                "ready, any three identifying details will do. In the meantime, would you like me to transfer you to a human representative?")

    def _t_rep_ask(self, plan: Plan, state: SessionState) -> str:
        v = plan.facts.get("verification", {})
        if v.get("fields_caller_can_offer") and v.get("still_needed"):
            need = v["still_needed"]
            return (f"Thanks. Before I can send the consent request I need {need} more detail{'s' if need != 1 else ''} about the policyholder: "
                    f"{_join(v['fields_caller_can_offer'], 'or')}.")
        if "policyholder's full name" in " ".join(plan.directive).lower():
            return "Happy to help on their behalf. What is the policyholder's full name?"
        return ("Happy to help on their behalf. To get started I'll need your own full name, your relationship to the policyholder, "
                "and the policyholder's full name.")

    def _t_rep_not_on_file(self, plan: Plan, state: SessionState) -> str:
        return ("I'm sorry — I don't see you listed as an authorized representative for that policyholder, so I'm not able to discuss the account "
                "with you. The policyholder can call us directly, or a human representative can explain how to add you as an authorized "
                "representative. Would you like me to transfer you?")

    def _t_consent_requested(self, plan: Plan, state: SessionState) -> str:
        ref = plan.facts.get("consent", {}).get("reference", "")
        return (f"Thanks — you're listed as an authorized representative and the details check out. For the policyholder's protection I've just sent a "
                f"one-time consent request to their phone on file (reference {ref}). Let me know once they've approved it and we'll continue.")

    def _t_consent_pending(self, plan: Plan, state: SessionState) -> str:
        return ("The consent request is still showing as pending on my end. Could you ask the policyholder to check their phone? "
                "I can wait, or if it keeps not arriving I can transfer you to a human representative.")

    def _t_consent_approved(self, plan: Plan, state: SessionState) -> str:
        return "The consent has come through."

    def _t_consent_denied(self, plan: Plan, state: SessionState) -> str:
        return ("The policyholder declined the consent request, so I'm not able to discuss the account with you. "
                "They're welcome to call us directly.")

    # ---- RESOLVE_INTENT ------------------------------------------------------------------
    def _t_choose_claim(self, plan: Plan, state: SessionState) -> str:
        a = plan.template_args
        cands = a.get("candidates", [])
        listing = "; ".join(_claim_line(c) for c in cands)
        mode = a.get("mode")
        if mode == "several_match":
            return f"I see more than one claim matching that: {listing}. Which one would you like to go over?"
        if mode == "none_match":
            return f"I don't see a claim matching that description. Here's what's on file: {listing}. Which one did you mean?"
        return f"What can I help you with today? Here's what I see on file: {listing}."

    def _t_no_claims(self, plan: Plan, state: SessionState) -> str:
        return "I don't see any claims on file under this policy. Would you like me to connect you with a human representative for anything else?"

    def _t_unsupported_intent(self, plan: Plan, state: SessionState) -> str:
        what = plan.facts.get("unsupported_request", "that")
        return (f"I can help with questions about your existing claims, but {what} needs a human representative. "
                "Would you like me to transfer you, or is there a claim I can help with?")

    # ---- PROCESS_CASE --------------------------------------------------------------------
    def _t_answer_claim(self, plan: Plan, state: SessionState) -> str:
        f = plan.facts
        c = f.get("claim", {})
        a = plan.template_args
        intent = a.get("intent") or "general_claim_question"
        first = a.get("first_answer", False)
        switched = any("switched" in d.lower() for d in plan.directive)
        docs = [d["document"] for d in f.get("documents_needed_detail", [])]
        followups = f.get("followup_guidance", [])
        specific_topics = {fu["topic"] for fu in followups}
        parts: list[str] = []

        # Lead with the claim identity only when it was just selected / switched.
        if first or switched:
            verb = "Switching to" if switched else "I've pulled up"
            parts.append(f"{verb} your {c.get('case_type')} claim {c.get('case_id')} from {humanize_date(c.get('created_at'))} — current status: {c.get('status')}.")
        if f.get("questions_asked_before_verification"):
            parts.append("Coming back to what you asked earlier:")

        if intent in ("denial_question", "status_inquiry", "general_claim_question", "appeal", "next_steps") or first:
            if c.get("status") == "denied" and c.get("denial_reason") and (first or intent in ("denial_question", "appeal", "status_inquiry")):
                parts.append(f"It was denied because {c['denial_reason']}.")
            if docs and (first or intent in ("denial_question", "next_steps", "appeal")):
                parts.append(f"To move it forward we need the {_join(docs, 'and')}.")
                if c.get("appeal_deadline"):
                    passed = f.get("appeal_deadline_has_passed")
                    parts.append(f"The appeal deadline is {humanize_date(c['appeal_deadline'])}" +
                                 (" — that date has already passed, so a human representative would need to review your options with you." if passed else "."))
            if c.get("status") == "open" and not docs and intent in ("status_inquiry", "general_claim_question"):
                parts.append(f"It's still in progress: {c.get('summary')}.")
            if c.get("status") == "closed" and intent in ("status_inquiry", "general_claim_question"):
                parts.append(f"{c.get('summary')}; the amount paid was ${c.get('net_pay_usd')}.")

        if intent == "document_submission" and not specific_topics - {"submission_method"}:
            if docs and not first:
                parts.append(f"We need the {_join(docs, 'and')}.")
            for d in f.get("documents_needed_detail", []):
                if d.get("requirements"):
                    parts.append(f"For the {d['document']}: {d['requirements']}")
            if f.get("general_submission_guidance") and "submission_method" not in specific_topics:
                parts.append(f["general_submission_guidance"])

        if intent == "payment_question":
            parts.append(f"The allowed maximum for this claim is ${c.get('allowed_max_amount_usd')}, the expected reimbursement is "
                         f"${c.get('expected_reimbursement_amount_usd')}, and the amount paid so far is ${c.get('net_pay_usd')}.")

        for fu in followups[:2]:
            parts.append(fu["guidance"])

        if len(parts) <= (1 if (first or switched) else 0):
            parts.append(f"{c.get('summary')}. What would you like to know about it?")
        else:
            parts.append("Is there anything else I can help you with on this or another claim?")
        return " ".join(parts)

    def _t_ask_what_else(self, plan: Plan, state: SessionState) -> str:
        return "Of course — what else would you like to know?"

    # ---- POST_PROCESS --------------------------------------------------------------------
    def _t_offer_email(self, plan: Plan, state: SessionState) -> str:
        offer = plan.facts.get("email_summary_offer", {})
        note = f" (It would go to the policyholder's email on file.)" if offer.get("note") else ""
        return (f"Glad I could help. Would you like me to email a summary of what we covered — what we discussed, the claim status, and the next steps — "
                f"to {offer.get('to', 'the email on file')}?{note} Just say yes to send it or skip it if you'd rather not.")

    def _t_offer_email_clarify(self, plan: Plan, state: SessionState) -> str:
        offer = plan.facts.get("email_summary_offer", {})
        return f"Just to confirm — should I send the summary email to {offer.get('to', 'the email on file')}, or skip it?"

    def _t_closed_with_email(self, plan: Plan, state: SessionState) -> str:
        sent = plan.facts.get("email_sent", {})
        name = state.caller_first_name()
        return (f"Done — the summary is on its way to {sent.get('to', 'the email on file')}. Thanks for calling{', ' + name if name else ''}, "
                "and take care.")

    def _t_closed_no_email(self, plan: Plan, state: SessionState) -> str:
        name = state.caller_first_name()
        return f"No problem, I won't send an email. Thanks for calling{', ' + name if name else ''} — take care."

    def _t_closed_unverified(self, plan: Plan, state: SessionState) -> str:
        return ("Understood — thanks for reaching out. Whenever you're ready, come back with a few identifying details handy and "
                "I'll be glad to pull up your claim. Take care.")

    def _t_closed_followup(self, plan: Plan, state: SessionState) -> str:
        return "Thanks again — we've wrapped up for today, but if anything else comes up about your claims just ask and I'll pick it back up."

    # ---- interrupts ------------------------------------------------------------------------
    def _t_off_topic(self, plan: Plan, state: SessionState) -> str:
        step = plan.facts.get("current_step", "where we were")
        return f"I'm sorry, that's outside what I can help with — I'm here for insurance claims questions only. Let's get back to {step}."

    def _t_off_topic_offer_human(self, plan: Plan, state: SessionState) -> str:
        return ("I'm not able to help with that — this line is only for insurance claims matters. If you'd like, I can connect you with a human "
                "representative who may be able to point you in the right direction. Would you like that?")

    def _t_handoff(self, plan: Plan, state: SessionState) -> str:
        h = plan.facts.get("handoff", {})
        extra = "" if h.get("identity_verified") else " They'll complete identity verification with you first."
        return (f"I'm transferring you to a human representative now. Your reference number is {h.get('reference')}.{extra} "
                "Thanks for your patience.")

    def _t_handoff_wait(self, plan: Plan, state: SessionState) -> str:
        h = plan.facts.get("handoff", {})
        return f"A human representative will be with you shortly — your reference number is {h.get('reference')}."

    def _t_verified(self, plan: Plan, state: SessionState) -> str:
        return "What can I help you with today?"

    def _t_generic(self, plan: Plan, state: SessionState) -> str:
        return "How can I help you with your claim today?"

    # ---- email -----------------------------------------------------------------------------
    def compose_email(self, state: SessionState, ctx: dict[str, Any]) -> tuple[str, str]:
        name = ctx.get("caller_first_name") or ""
        claims = ctx.get("claims", [])
        lines = [f"Hello{' ' + name if name else ''},", "",
                 "Thank you for contacting " + ctx["company"] + " claims support today. Here is a summary of our conversation.", "",
                 "What we discussed"]
        topics = [t.replace("_", " ") for t in ctx.get("topics_covered", [])] or ["your claim"]
        lines.append(f"We went over your {_join(topics, 'and')}" + (f" for claim {', '.join(c['claim']['case_id'] for c in claims)}." if claims else "."))
        lines += ["", "Claim status / outcome"]
        for b in claims:
            c = b["claim"]
            line = f"- {c['case_id']} ({c['case_type']}, filed {humanize_date(c['created_at'])}): {c['status']}"
            if c.get("denial_reason"):
                line += f" — denied because {c['denial_reason']}"
            lines.append(line)
        lines += ["", "Next steps"]
        n = 1
        for b in claims:
            c = b["claim"]
            for d in b.get("documents_needed_detail", []):
                lines.append(f"{n}. Submit the {d['document']} for {c['case_id']}." + (f" {d['requirements']}" if d.get("requirements") else ""))
                n += 1
            if c.get("appeal_deadline"):
                lines.append(f"{n}. Appeal deadline for {c['case_id']}: {humanize_date(c['appeal_deadline'])}.")
                n += 1
            if b.get("general_submission_guidance"):
                lines.append(f"{n}. How to submit: {b['general_submission_guidance']}")
                n += 1
        if n == 1:
            lines.append("1. No further action is needed from you at this time.")
        lines += ["", f"{ctx['agent_name']}, {ctx['company']} Claims Support"]
        subject = f"Summary of your call with {ctx['company']} claims support" + (f" — {claims[0]['claim']['case_id']}" if claims else "")
        return subject, "\n".join(lines)


class LLMResponder:
    name = "llm"

    def __init__(self, provider: LLMProvider, settings: Settings):
        self.provider = provider
        self.s = settings
        self.templates = TemplateResponder(settings)

    def opening(self, state: SessionState) -> str:
        return self.templates.opening(state)

    def safe_fallback(self, plan: Plan, state: SessionState) -> str:
        return self.templates.render(plan, state)

    def _history_messages(self, state: SessionState, n: int = 10) -> list[dict[str, str]]:
        msgs = []
        for rec in state.recent_history(n):
            msgs.append({"role": "assistant" if rec.role == "agent" else "user", "content": rec.text})
        return msgs

    def respond(self, plan: Plan, state: SessionState) -> str:
        system = responder_system_prompt(
            plan.phase.value, plan.directive, plan.facts, plan.must_not, plan.tone,
            agent_name=self.s.agent_name, company=self.s.company_name, required_factors=self.s.min_pii_factors,
        )
        try:
            # Current Claude models may spend hidden reasoning tokens that count toward max_tokens; keep headroom.
            text = self.provider.complete(system, self._history_messages(state), max_tokens=1500, temperature=0.4)
        except (LLMError, Exception) as exc:  # noqa: BLE001
            log.warning("LLM response failed (%s); using template", exc)
            return self.templates.render(plan, state)
        text = (text or "").strip().strip('"').strip()
        if not text:
            return self.templates.render(plan, state)
        return text

    def compose_email(self, state: SessionState, ctx: dict[str, Any]) -> tuple[str, str]:
        system = EMAIL_SYSTEM.format(company=self.s.company_name, agent_name=self.s.agent_name)
        try:
            raw = self.provider.complete(system, [{"role": "user", "content": email_user_prompt(ctx)}],
                                         max_tokens=3000, temperature=0.2, json_mode=True, json_schema=EMAIL_SCHEMA)
            data = parse_json_object(raw) or {}
            subject, body = (data.get("subject") or "").strip(), (data.get("body") or "").strip()
            if subject and body:
                return subject, body
        except (LLMError, Exception) as exc:  # noqa: BLE001
            log.warning("LLM email composition failed (%s); using template", exc)
        return self.templates.compose_email(state, ctx)


def build_components(provider: LLMProvider, settings: Settings, today: str):
    """Return (extractor, responder) matching the provider type."""
    from .extractor import LLMExtractor, RuleExtractor  # local import to avoid cycles

    if provider.cfg.is_mock:
        return RuleExtractor(settings), TemplateResponder(settings)
    return LLMExtractor(provider, settings, today), LLMResponder(provider, settings)
