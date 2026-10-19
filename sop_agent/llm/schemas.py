"""JSON schemas for structured model output.

Used with Anthropic's `output_config.format` (json_schema) and OpenAI's
`response_format` (json_schema, strict). Every property is required and
objects forbid additional properties, as both APIs expect; "absent" values are
expressed with explicit sentinels ("none" / "unknown") or nullable types.
"""
from __future__ import annotations

from ..sop.signals import EMOTIONS, INTENTS, SCOPES


def _nullable(t: str) -> dict:
    return {"type": [t, "null"]}


TURN_SIGNALS_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pii": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "full_name": _nullable("string"),
                "dob": _nullable("string"),
                "phone": _nullable("string"),
                "email": _nullable("string"),
                "id_last4": _nullable("string"),
                "policy_number": _nullable("string"),
            },
            "required": ["full_name", "dob", "phone", "email", "id_last4", "policy_number"],
        },
        "caller_role": {"type": "string", "enum": ["policyholder", "representative", "unknown"]},
        "representative": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": _nullable("string"),
                "relationship": _nullable("string"),
                "policyholder_name": _nullable("string"),
            },
            "required": ["name", "relationship", "policyholder_name"],
        },
        "intent": {"type": "string", "enum": list(INTENTS) + ["none"]},
        "case_hints": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "case_id": _nullable("string"),
                "case_type": {"type": "string", "enum": ["healthcare", "dental", "auto", "unknown"]},
                "status": {"type": "string", "enum": ["denied", "open", "closed", "unknown"]},
                "time_reference": _nullable("string"),
                "month": _nullable("integer"),
                "year": _nullable("integer"),
            },
            "required": ["case_id", "case_type", "status", "time_reference", "month", "year"],
        },
        "questions": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "string", "enum": list(SCOPES)},
        "off_topic_part": _nullable("string"),
        "emotion": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": list(EMOTIONS)},
                "intensity": {"type": "integer"},
            },
            "required": ["label", "intensity"],
        },
        "requests_human": {"type": "boolean"},
        "refuses_verification": {"type": "boolean"},
        "asks_why_needed": {"type": "boolean"},
        "answer": {"type": "string", "enum": ["yes", "no", "unclear", "none"]},
        "wants_to_end": {"type": "boolean"},
        "email_preference": {"type": "string", "enum": ["yes", "no", "none"]},
    },
    "required": [
        "pii", "caller_role", "representative", "intent", "case_hints", "questions", "notes", "scope",
        "off_topic_part", "emotion", "requests_human", "refuses_verification", "asks_why_needed", "answer",
        "wants_to_end", "email_preference",
    ],
}

EMAIL_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "subject": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["subject", "body"],
}
