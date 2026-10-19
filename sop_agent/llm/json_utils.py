"""Forgiving JSON extraction for LLM output."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    candidates = [text.strip()]
    m = _FENCE.search(text)
    if m:
        candidates.insert(0, m.group(1).strip())
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start, end = cand.find("{"), cand.rfind("}")
        if start != -1 and end > start:
            chunk = cand[start:end + 1]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                # try to repair trailing commas
                repaired = re.sub(r",\s*([}\]])", r"\1", chunk)
                try:
                    obj = json.loads(repaired)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
    return None
