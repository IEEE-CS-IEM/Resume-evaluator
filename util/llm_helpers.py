import json
from typing import Any, Dict, Tuple


def coerce_text(response: Any) -> str:
    """
    Normalise responses from different client SDKs into plain text.
    """
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return getattr(response, "text")
    if hasattr(response, "content"):
        return getattr(response, "content")
    return str(response)


def coerce_json(response: Any) -> Tuple[Dict[str, Any], str]:
    """
    Attempt to parse an LLM response into JSON while also returning the raw text.
    """
    raw_text = coerce_text(response)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to salvage JSON embedded within text fences
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw_text[start : end + 1]
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
    return payload, raw_text
