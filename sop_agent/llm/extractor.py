"""Turn extractors.

LLMExtractor  – asks the model for the TurnSignals JSON (see prompts.EXTRACTOR_SYSTEM)
RuleExtractor – regex/keyword heuristics; used in mock mode, as the fallback when the
                model call fails, and to cross-check unambiguous fields (email, phone,
                policy number) in LLM output.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..config import Settings
from ..sop.signals import TurnSignals
from ..sop.state import (
    PENDING_ANYTHING_ELSE,
    PENDING_CHOOSE_CLAIM,
    PENDING_CONFIRM_CLAIM,
    PENDING_CONSENT_WAIT,
    PENDING_EMAIL_OFFER,
    PENDING_HUMAN_OFFER,
    Phase,
    SessionState,
)
from ..sop.verification import describe_options
from ..utils.normalize import parse_dob
from .json_utils import parse_json_object
from .prompts import EXTRACTOR_SYSTEM, extractor_user_prompt
from .provider import LLMError, LLMProvider
from .schemas import TURN_SIGNALS_SCHEMA

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------
# Rule-based extractor
# ----------------------------------------------------------------------------------------
NAME_RE = re.compile(
    r"(?i:my name is|my name's|the name is|name is|this is|i am|i'm|it's|calling,? this is)\s+"
    r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})"
)
NAME_LABEL_RE = re.compile(r"(?i:\bname)\s*[:\-]\s*([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})")
BARE_NAME_RE = re.compile(r"^\s*([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\s*[.!]?\s*$")
CAPITALISED_RUN_RE = re.compile(r"\b([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,2})\b")
NAME_STOPWORDS = {
    "i", "hi", "hello", "hey", "ok", "okay", "fine", "the", "my", "dob", "ssn", "pol", "yes", "no", "thanks", "thank", "you",
    "please", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
    "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "social", "security", "number",
    "date", "birth", "phone", "email", "policy", "claim", "last", "four", "this", "that", "just", "tell", "why", "what",
    "sure", "here", "it", "is", "its", "also", "and", "or", "but", "so", "well", "look", "listen", "hold", "on", "sam",
    "harbor", "mutual", "insurance", "id", "national", "healthcare", "dental", "auto", "denied", "open", "closed",
}
DOB_KEY_RE = re.compile(r"(?:dob|date of birth|birth ?date|birthday|born(?: on)?|d\.o\.b\.?)\s*(?:is|was|:)?\s*(.{4,30})", re.I)
DATE_ANYWHERE_RE = re.compile(
    r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s+\d{4}\b", re.I)
CLAIM_DATE_CONTEXT_RE = re.compile(r"\b(claim|filed|submitted|accident|visit|service|appointment|denied|incident|happened|treatment|surgery|since|deadline)\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
LAST4_RE = re.compile(
    r"\b(?:ssn|social(?: security)?(?: number)?|last (?:four|4)(?: digits)?|national id|id(?: number)?|ends? in|ending in)\b"
    r"[^\d\n]{0,40}?(?<![\d/\-])(\d{4})(?![\d/\-])", re.I)
BARE_LAST4_RE = re.compile(r"^\D*(\d{4})\D*$")
POLICY_RE = re.compile(r"\bPOL[\s\-]?(\d{3,8})\b", re.I)
CASE_ID_RE = re.compile(r"\bCL[\s\-]?(\d{3,6})\b", re.I)
YEAR_RE = re.compile(r"\b(20[0-3]\d)\b")
MONTH_RE = re.compile(
    r"\b(january|february|march|april|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\b", re.I)
MAY_RE = re.compile(r"\b(?:in|from|since|last|of|back in)\s+may\b|\bmay\s+20\d\d\b", re.I)
MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
          "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
          "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}

REP_RE = re.compile(
    r"(?:on behalf of|calling for|speaking for|for my)\s+(?:my\s+)?"
    r"(mother|mom|father|dad|wife|husband|spouse|son|daughter|parent|grandmother|grandfather|sister|brother|client|friend)?"
    r"[,\s]*([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})?", re.I)
REL_RE = re.compile(r"\b(?:i am|i'm|this is)\s+(?:her|his|their)\s+(son|daughter|husband|wife|spouse|caregiver|guardian|attorney|representative|agent)\b", re.I)
REP_SELF_RE = re.compile(r"\b(?:on behalf of|authori[sz]ed representative|power of attorney|calling for my|speaking for)\b", re.I)

IN_SCOPE_WORDS = (
    "claim", "polic", "insur", "deni", "appeal", "cover", "deductible", "premium", "reimburs", "payment", "paid", "pay",
    "document", "paperwork", "submit", "upload", "fax", "mail", "verify", "verification", "identity", "name", "dob", "birth",
    "ssn", "social", "phone", "email", "representative", "human", "agent", "status", "pathology", "report", "note",
    "hospital", "medical", "health", "dental", "auto", "car", "accident", "summary", "benefit", "bill", "invoice",
    "provider", "doctor", "clinic", "portal", "deadline", "amount", "money", "check", "copay", "coinsurance", "account",
    "member", "id", "policy", "transfer", "supervisor", "manager", "address", "consent", "approve", "approved",
    "pending", "denied", "rejected", "case", "cl-", "pol-", "settle", "settled", "diagnosis", "estimate", "repair",
    "why", "what do you need", "last four", "on file", "on behalf", "mother", "father", "son", "daughter",
)
OUT_PATTERNS = re.compile(
    r"\b(what is|what's|whats|who is|who's|define|explain|tell me about|how do i|how to|write (?:me )?(?:a|an|some)|"
    r"weather|joke|recipe|capital of|python|javascript|code|program|reinforcement learning|machine learning|\brl\b|ai model|"
    r"stock|bitcoin|crypto|sports|football|movie|song|poem|translate|math|calculate|president|election|news|history of|"
    r"meaning of life|chatgpt|gpt|llm|neural)\b", re.I)
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good (?:morning|afternoon|evening)|thanks?|thank you|cheers|ok(?:ay)?|cool|great|alright|sure)[\s!.,]*$", re.I)

YES_RE = re.compile(r"^\s*(?:yes|yeah|yep|yup|ya|sure|ok|okay|please|correct|right|that's right|that is right|exactly|go ahead|absolutely|"
                    r"of course|do it|send it|sounds good|that works|that's the one|that one|yes please|please do|fine|affirmative|"
                    r"let's do it|i do|i would|i'd like that|sure thing|definitely|certainly|perfect)\b", re.I)
NO_RE = re.compile(r"^\s*(?:no|nope|nah|skip|don't|do not|not now|no thanks|no thank you|that's all|that is all|nothing else|"
                   r"i'm good|im good|i'm fine|not needed|that's not|negative|not really|no need|pass|i'll pass|not that one|neither)\b", re.I)
SKIP_RE = re.compile(r"\b(skip|no email|don't email|do not email|don't send|do not send|no need to send|not necessary)\b", re.I)
WANT_EMAIL_RE = re.compile(r"\b(email me|send me (?:an |the )?(?:email|summary)|send the summary|yes,? (?:please )?send|i'd like the (?:email|summary))\b", re.I)
END_RE = re.compile(r"\b(that's all|that is all|that'?s everything|nothing else|no more questions|no other questions|i'm done|im done|"
                    r"that's it|that is it|bye|goodbye|have a (?:good|nice|great) (?:day|one)|all set|we're done|i'm all set|take care)\b", re.I)
HUMAN_RE = re.compile(r"\b(?:(?:speak|talk|connect|transfer|get|put)(?: me)?(?: through)? (?:to|with) (?:a |an |the )?(?:human|real person|person|live agent|"
                      r"representative|rep|agent|supervisor|manager|someone)|(?:a )?human(?: being)?(?: please)?|real person|live agent|supervisor|"
                      r"escalate|someone else who can help)\b", re.I)
WHY_RE = re.compile(r"\b(why do you need|why should i|why is (?:this|that) necessary|why do i have to|what do you need (?:that|this|it) for|"
                    r"why can't you just|why won't you|why does it matter)\b", re.I)
REFUSE_RE = re.compile(r"\b(not giving|won't give|will not give|i refuse|not comfortable|don't want to (?:give|share|provide)|"
                       r"not going to (?:give|share|provide)|no way i'm|i'm not sharing|not providing|i'm not telling|not telling you|"
                       r"i won't (?:share|provide|tell)|i don't give out|i'm not going to tell)\b", re.I)
ANGRY_RE = re.compile(r"\b(ridiculous|unacceptable|insane|absurd|angry|furious|pissed|sick of|fed up|wasting my time|waste of time|"
                      r"seriously\?|come on|already told you|i told you|this is a joke|terrible|worst|useless|stupid|incompetent|"
                      r"outrageous|enough|for god's sake|unbelievable|frustrat|annoy|infuriat|"
                      r"how many times|again\?|stop asking)\b", re.I)
ANXIOUS_RE = re.compile(r"\b(worried|scared|anxious|stressed|nervous|afraid|panick?|desperate|urgent|can't afford|cannot afford|"
                        r"terrified|freaking out|overwhelmed|need this|what am i going to do|what if)\b", re.I)
CONFUSED_RE = re.compile(r"\b(confused|don't understand|do not understand|what do you mean|i'm lost|makes no sense|doesn't make sense|"
                         r"not sure what|i don't get it|huh\?)\b", re.I)
GRATEFUL_RE = re.compile(r"\b(thank|thanks|appreciate|grateful|you've been (?:great|helpful)|very helpful)\b", re.I)
QUESTION_START_RE = re.compile(r"^\s*(what|why|when|where|which|who|how|can|could|would|will|is|are|do|does|did|should|am)\b", re.I)

INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("update_contact_info", re.compile(r"\b(change|update|correct)\s+(?:my\s+)?(address|phone|email|contact|number)\b", re.I)),
    ("appeal", re.compile(r"\bappeal", re.I)),
    ("denial_question", re.compile(r"\b(denied|denial|rejected|turned down|declined|why (?:was|did|is) .*(?:deni|reject))", re.I)),
    ("document_submission", re.compile(r"\b(submit|upload|send (?:in |over |you )?(?:the |my )?(?:document|paperwork|report|note|file|photo|estimate)|"
                                        r"documents?|paperwork|fax|portal|upload link|where (?:do|should) i send|what (?:do|should) i (?:send|provide|include))\b", re.I)),
    ("payment_question", re.compile(r"\b(how much|amount|pay(?:ment|out|ing)?\b|paid|reimburse|money|check|owe|cover(?:ed)? amount|net pay|allowed)\b", re.I)),
    ("status_inquiry", re.compile(r"\b(status|update on|where (?:is|are) (?:my|the|we)|progress|what's happening|whats happening|any news|still (?:pending|open|processing))\b", re.I)),
    ("next_steps", re.compile(r"\b(next step|what (?:do|should) i do|how do i proceed|what happens (?:now|next)|what now|how do i fix|what can i do)\b", re.I)),
    ("coverage_question", re.compile(r"\b(cover(?:ed|age)?\b(?! amount)|deductible|premium|copay|in[- ]network)\b", re.I)),
    ("general_claim_question", re.compile(r"\bclaim", re.I)),
]

CASE_TYPE_RULES = [
    ("healthcare", re.compile(r"\b(health ?care|health|medical|hospital|doctor|clinic|surgery|pathology|diagnosis|treatment|physician)\b", re.I)),
    ("dental", re.compile(r"\b(dental|dentist|teeth|tooth|orthodont)\b", re.I)),
    ("auto", re.compile(r"\b(auto|car|vehicle|accident|collision|crash|repair|body shop|windshield|fender)\b", re.I)),
]
STATUS_RULES = [
    ("denied", re.compile(r"\b(denied|denial|rejected|declined|turned down)\b", re.I)),
    ("open", re.compile(r"\b(open|pending|in progress|processing|ongoing|still waiting)\b", re.I)),
    ("closed", re.compile(r"\b(closed|settled|completed|finished|resolved|paid out)\b", re.I)),
]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _in_scope_sentence(sentence: str) -> bool:
    low = sentence.lower()
    return any(w in low for w in IN_SCOPE_WORDS)


class RuleExtractor:
    """Deterministic extractor. Good enough to drive the SOP end-to-end without a model."""

    name = "mock"

    def __init__(self, settings: Settings):
        self.settings = settings

    # -- helpers ------------------------------------------------------------------------
    @staticmethod
    def _last_agent_text(state: SessionState) -> str:
        for rec in reversed(state.history):
            if rec.role == "agent":
                return rec.text.lower()
        return ""

    def _pii(self, text: str, state: SessionState, data: dict[str, Any]) -> None:
        pii: dict[str, Any] = {}
        awaiting_identity = state.phase == Phase.VERIFY_ID
        last_agent = self._last_agent_text(state)

        m = NAME_RE.search(text) or NAME_LABEL_RE.search(text)
        if m and not re.match(r"^(the|a|an|her|his|calling|not|just)\b", m.group(1), re.I):
            pii["full_name"] = m.group(1)
        elif awaiting_identity and BARE_NAME_RE.match(text) and not YES_RE.match(text) and not NO_RE.match(text):
            cand = BARE_NAME_RE.match(text).group(1)
            if len(cand.split()) >= 2:
                pii["full_name"] = cand
        elif awaiting_identity:
            # "Fine. Margaret Chen, 1985-03-15, last four 4472" — a capitalised 2-3 word run that isn't ordinary prose
            for cand in CAPITALISED_RUN_RE.findall(text):
                words = cand.split()
                if all(w.lower() not in NAME_STOPWORDS for w in words) and 2 <= len(words) <= 3:
                    pii["full_name"] = cand
                    break

        dm = DOB_KEY_RE.search(text)
        dob = parse_dob(dm.group(1)) if dm else None
        if not dob and awaiting_identity:
            # During verification a date is almost always the DOB — unless it clearly describes a claim event.
            dmatch = DATE_ANYWHERE_RE.search(text)
            if dmatch and not CLAIM_DATE_CONTEXT_RE.search(text[max(0, dmatch.start() - 30):dmatch.start()]):
                dob = parse_dob(dmatch.group(0))
        if dob:
            pii["dob"] = dob

        em = EMAIL_RE.search(text)
        if em:
            pii["email"] = em.group(0)

        pm = PHONE_RE.search(text)
        if pm:
            pii["phone"] = pm.group(0)

        lm = LAST4_RE.search(text)
        if lm:
            pii["id_last4"] = lm.group(1)
        elif awaiting_identity:
            # a lone 4-digit token that is not a year, not part of a date/phone/policy number
            scrubbed = POLICY_RE.sub(" ", PHONE_RE.sub(" ", DATE_ANYWHERE_RE.sub(" ", text)))
            for tok in re.findall(r"(?<![\d\-/])(\d{4})(?![\d\-/])", scrubbed):
                if not (1900 <= int(tok) <= 2099):
                    pii["id_last4"] = tok
                    break

        pol = POLICY_RE.search(text)
        if pol:
            pii["policy_number"] = f"POL-{pol.group(1)}"
        data["pii"] = pii

        # representative?
        rep: dict[str, Any] = {}
        rm = REP_RE.search(text)
        if REP_SELF_RE.search(text) or (rm is not None and rm.group(1)):
            if rm:
                if rm.group(2):
                    rep["policyholder_name"] = rm.group(2)
                rel = rm.group(1)
                relm = REL_RE.search(text)
                if relm:
                    rep["relationship"] = relm.group(1).lower()
                elif rel:
                    rel = rel.lower()
                    rep["relationship"] = {"mother": "child", "mom": "child", "father": "child", "dad": "child"}.get(rel, rel)
            if pii.get("full_name") and rep.get("policyholder_name") and pii["full_name"] != rep["policyholder_name"]:
                rep["name"] = pii["full_name"]
                pii["full_name"] = rep["policyholder_name"]
            elif pii.get("full_name") and not rep.get("policyholder_name"):
                rep["name"] = pii.pop("full_name")
            data["caller_role"] = "representative"
        elif re.search(r"\b(i am|i'm) the policyholder\b", text, re.I):
            data["caller_role"] = "policyholder"
        data["representative"] = rep

    def _intent_and_hints(self, text: str, data: dict[str, Any]) -> None:
        intent = None
        for name, rx in INTENT_RULES:
            if rx.search(text):
                intent = name
                break
        data["intent"] = intent
        # A date of birth is not a claim date: strip it before looking for month/year hints.
        if data.get("pii", {}).get("dob"):
            dm = DOB_KEY_RE.search(text)
            text = text[:dm.start()] + text[dm.end():] if dm else ("" if len(text) < 40 else text)
        hints: dict[str, Any] = {}
        cm = CASE_ID_RE.search(text)
        if cm:
            hints["case_id"] = f"CL-{cm.group(1)}"
        for ct, rx in CASE_TYPE_RULES:
            if rx.search(text):
                hints["case_type"] = ct
                break
        for st, rx in STATUS_RULES:
            if rx.search(text):
                hints["status"] = st
                break
        mm = MONTH_RE.search(text)
        if mm:
            hints["month"] = MONTHS[mm.group(1).lower()]
            hints["time_reference"] = mm.group(0)
        elif MAY_RE.search(text):
            hints["month"] = 5
            hints["time_reference"] = "May"
        ym = YEAR_RE.search(text)
        if ym and not DOB_KEY_RE.search(text):
            hints["year"] = int(ym.group(1))
        data["case_hints"] = hints

    def _scope(self, text: str, data: dict[str, Any], state: SessionState) -> None:
        sentences = _split_sentences(text) or [text]
        in_s = [s for s in sentences if _in_scope_sentence(s)]
        out_s = [s for s in sentences if not _in_scope_sentence(s) and OUT_PATTERNS.search(s)]
        if GREETING_RE.match(text):
            data["scope"] = "small_talk"
        elif in_s or data["pii"] or data.get("answer") or data.get("requests_human"):
            data["scope"] = "in_scope"
            if out_s:
                data["off_topic_part"] = out_s[0]
        elif out_s:
            data["scope"] = "out_of_scope"
            data["off_topic_part"] = out_s[0]
        elif QUESTION_START_RE.match(text) and len(text.split()) >= 3 and state.phase != Phase.PROCESS_CASE:
            data["scope"] = "out_of_scope"
            data["off_topic_part"] = text
        else:
            data["scope"] = "in_scope"

    def _emotion(self, text: str, data: dict[str, Any]) -> None:
        label, intensity = "neutral", 0
        exclaims = text.count("!")
        caps_words = len([w for w in text.split() if len(w) > 3 and w.isupper()])
        if REFUSE_RE.search(text):
            label, intensity = "refusing", 2
        elif ANGRY_RE.search(text):
            hits = len(ANGRY_RE.findall(text))
            label = "angry" if hits >= 2 or exclaims >= 2 or caps_words >= 2 else "frustrated"
            intensity = min(3, 1 + hits + (1 if exclaims else 0))
        elif ANXIOUS_RE.search(text):
            label, intensity = "anxious", 2
        elif CONFUSED_RE.search(text):
            label, intensity = "confused", 1
        elif GRATEFUL_RE.search(text) and not QUESTION_START_RE.match(text):
            label, intensity = "grateful", 1
        data["emotion"] = {"label": label, "intensity": intensity}

    def _flags(self, text: str, data: dict[str, Any], state: SessionState) -> None:
        data["requests_human"] = bool(HUMAN_RE.search(text)) and not re.search(r"\b(authori[sz]ed representative|on behalf)\b", text, re.I)
        data["refuses_verification"] = bool(REFUSE_RE.search(text)) and state.phase == Phase.VERIFY_ID
        data["asks_why_needed"] = bool(WHY_RE.search(text))
        data["wants_to_end"] = bool(END_RE.search(text)) and not QUESTION_START_RE.match(text)
        answer = None
        if state.pending_question in (PENDING_EMAIL_OFFER, PENDING_ANYTHING_ELSE, PENDING_HUMAN_OFFER, PENDING_CONFIRM_CLAIM,
                                      PENDING_CHOOSE_CLAIM, PENDING_CONSENT_WAIT):
            if YES_RE.match(text) or WANT_EMAIL_RE.search(text):
                answer = "yes"
            elif NO_RE.match(text) or SKIP_RE.search(text) or END_RE.search(text):
                answer = "no"
            if state.pending_question == PENDING_ANYTHING_ELSE and (data.get("intent") or "?" in text) and not END_RE.search(text):
                answer = None if not YES_RE.match(text) else "yes"
        data["answer"] = answer
        if SKIP_RE.search(text):
            data["email_preference"] = "no"
        elif WANT_EMAIL_RE.search(text):
            data["email_preference"] = "yes"
        # Claim questions worth remembering — not questions about the verification process itself.
        questions = [s for s in _split_sentences(text)
                     if ("?" in s or QUESTION_START_RE.match(s)) and _in_scope_sentence(s)
                     and not WHY_RE.search(s) and not re.search(r"\b(verif|identity|need (?:my|that|this|it)|who i am)\b", s, re.I)]
        data["questions"] = questions[:3]

    def extract(self, text: str, state: SessionState) -> TurnSignals:
        data: dict[str, Any] = {}
        self._pii(text, state, data)
        self._intent_and_hints(text, data)
        self._emotion(text, data)
        self._flags(text, data, state)
        self._scope(text, data, state)
        return TurnSignals.from_dict(data, source="mock")


# ----------------------------------------------------------------------------------------
# LLM extractor
# ----------------------------------------------------------------------------------------
class LLMExtractor:
    name = "llm"

    def __init__(self, provider: LLMProvider, settings: Settings, today: str):
        self.provider = provider
        self.settings = settings
        self.today = today
        self.rules = RuleExtractor(settings)

    def _context(self, state: SessionState) -> dict[str, Any]:
        v = state.verification
        last_agent = next((r.text for r in reversed(state.history) if r.role == "agent"), None)
        pending_desc = {
            PENDING_EMAIL_OFFER: "agent offered to email a conversation summary (yes = send, no = skip)",
            PENDING_ANYTHING_ELSE: "agent asked whether there is anything else they need help with",
            PENDING_HUMAN_OFFER: "agent offered to transfer to a human representative (yes/no)",
            PENDING_CONFIRM_CLAIM: "agent asked whether a specific claim is the one they mean (yes/no)",
            PENDING_CHOOSE_CLAIM: "agent asked which of several claims they mean",
            PENDING_CONSENT_WAIT: "agent is waiting for the policyholder's consent approval",
        }.get(state.pending_question or "", None)
        need = max(0, self.settings.min_pii_factors - v.matched_count())
        ctx: dict[str, Any] = {
            "TODAY": self.today,
            "PHASE": state.phase.value,
            "PENDING_QUESTION": pending_desc,
            "AGENT_LAST_MESSAGE": last_agent,
            "CALLER_ROLE_SO_FAR": v.caller_role,
            "IDENTITY_FIELDS_ALREADY_GIVEN": sorted(v.provided.keys()),
        }
        if state.phase == Phase.VERIFY_ID:
            ctx["IDENTITY_FIELDS_STILL_NEEDED"] = need
            ctx["ACCEPTABLE_FIELDS"] = describe_options([f for f in ("full_name", "dob", "phone", "email", "id_last4") if f not in v.matched])
        if state.selected_case_id:
            ctx["CLAIM_BEING_DISCUSSED"] = state.selected_case_id
        if state.candidate_claims and state.pending_question == PENDING_CHOOSE_CLAIM:
            ctx["CLAIMS_OFFERED_TO_CHOOSE_FROM"] = state.candidate_claims
            ctx["NOTE"] = "If the caller picks one of these (by number, type, date, status or position), put its case_id in case_hints.case_id."
        return ctx

    def extract(self, text: str, state: SessionState) -> TurnSignals:
        rule_sig = self.rules.extract(text, state)
        try:
            raw = self.provider.complete(
                EXTRACTOR_SYSTEM,
                [{"role": "user", "content": extractor_user_prompt(text, self._context(state))}],
                model=self.provider.cfg.extractor_model,
                max_tokens=2000,   # structured JSON + any hidden reasoning the model does
                temperature=0.0,
                json_mode=True,
                json_schema=TURN_SIGNALS_SCHEMA,
            )
            data = parse_json_object(raw)
            if data is None:
                raise LLMError("extractor returned non-JSON output")
            sig = TurnSignals.from_dict(data, source="llm")
        except (LLMError, Exception) as exc:  # noqa: BLE001 - any failure degrades to rules
            log.warning("LLM extraction failed (%s); using rule-based extractor", exc)
            rule_sig.source = "fallback"
            return rule_sig

        # Cross-check unambiguous fields the model may have skipped.
        for key in ("email", "phone", "policy_number", "id_last4"):
            if key not in sig.pii and key in rule_sig.pii:
                sig.pii[key] = rule_sig.pii[key]
        if "dob" not in sig.pii and "dob" in rule_sig.pii and re.search(r"dob|birth|born", text, re.I):
            sig.pii["dob"] = rule_sig.pii["dob"]
        if not sig.case_hints.get("case_id") and rule_sig.case_hints.get("case_id"):
            sig.case_hints["case_id"] = rule_sig.case_hints["case_id"]
        return sig
