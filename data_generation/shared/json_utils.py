"""JSON extraction helpers for LLM outputs."""

from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> dict:
    """Parse the first complete JSON object from an LLM response.

    Strips optional markdown fences and trailing commas before decoding.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth, end = 0, -1
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("Unmatched braces in JSON")

    raw = text[start : end + 1]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return json.loads(raw)


def save_json(obj, path, indent: int = 2) -> None:
    """Write ``obj`` as UTF-8 JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
