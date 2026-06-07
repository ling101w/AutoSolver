"""Utilities for handling LLM JSON responses."""

from __future__ import annotations

import json
import re
from typing import Any, Iterator


class JSONExtractionError(ValueError):
    """Raised when a JSON document cannot be recovered from text."""


def load_json_document(text_or_value: Any) -> Any:
    """Load JSON directly or recover the first balanced JSON document from text."""

    if isinstance(text_or_value, (dict, list)):
        return text_or_value
    text = str(text_or_value).strip()
    try:
        return json.loads(text)
    except Exception as direct_error:
        for candidate in _json_candidates(text):
            try:
                return json.loads(candidate)
            except Exception:
                continue
        raise JSONExtractionError(f"response is not valid JSON: {direct_error}") from direct_error


def _json_candidates(text: str) -> Iterator[str]:
    for match in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL):
        candidate = match.group(1).strip()
        if candidate:
            yield candidate

    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    for start, char in enumerate(cleaned):
        if char in "{[":
            candidate = _balanced_json_at(cleaned, start)
            if candidate:
                yield candidate


def _balanced_json_at(text: str, start: int) -> str:
    openers = {"{": "}", "[": "]"}
    stack = []
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in openers:
            stack.append(openers[char])
            continue
        if char in "}]":
            if not stack or char != stack[-1]:
                return ""
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return ""
