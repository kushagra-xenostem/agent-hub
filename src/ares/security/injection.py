"""Heuristic prompt-injection detection for untrusted tool output.

A mitigation, not a guarantee: patterns catch common phrasings of instructions aimed at the model, and
paraphrases can evade them. Guarantees come from permissions, approval, egress and output checks.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

PATTERNS: dict[str, re.Pattern[str]] = {name: re.compile(rx, re.IGNORECASE) for name, rx in {
    "ignore_instructions": r"\b(ignore|disregard|override|forget)\b.{0,30}\b(previous|prior|above|earlier|other|all)\b"
                           r".{0,20}\b(instructions?|sources?|directions?|rules?|figures?)",
    "addressed_to_ai": r"\b(note|notice|message|instructions?|attention)\b.{0,15}\b(to|for)\b.{0,10}"
                       r"\b(ai|llm|language models?|assistants?|agents?|bots?)\b",
    "ai_must": r"\b(ai|llm)\s+(assistants?|agents?|models?|systems?)\b.{0,80}\b(must|should|are required|need to|have to)\b",
    "system_prompt": r"\bsystem\s+(prompt|instructions?|message)\b|\bsession reference codes?\b",
    "tool_invocation": r"\b(send_email|save_note|fetch_page|search_web|submit_report)\b|"
                       r"\b(send|forward|email|mail)\b.{0,60}\b[\w.+-]+@[\w-]+\.[\w.]+",
    "citation_hijack": r"\b(must|always)\s+cite\b|\bcite\b.{0,60}\bas the source\b",
    "secret_request": r"\b(include|reveal|share|output|print)\b.{0,40}\b(internal|confidential|secret|deal codes?|api keys?|"
                      r"passwords?|credentials?)\b",
    "excessive_actions": r"\b(at least|minimum of)\s+\d{2,}\s+(searches|queries|calls|requests|times)\b",
}.items()}

# a sentence ends at . ! ? followed by whitespace, not at the dots inside URLs or e-mail addresses
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?=\s)|(?<=\n)")
REDACTION = "[removed: text that looked like instructions to the AI]"


class Detection(BaseModel):
    pattern: str
    excerpt: str


def scan(text: str) -> list[Detection]:
    found = []
    for name, rx in PATTERNS.items():
        m = rx.search(text)
        if m:
            found.append(Detection(pattern=name, excerpt=text[max(0, m.start() - 20): m.end() + 20]))
    return found


def redact(text: str) -> tuple[str, int]:
    """Replace every sentence that matches a pattern. Returns (text, sentences removed)."""
    out, removed = [], 0
    for sentence in _SENTENCE_BREAK.split(text):
        if sentence.strip() and any(rx.search(sentence) for rx in PATTERNS.values()):
            out.append(sentence[: len(sentence) - len(sentence.lstrip())] + REDACTION)
            removed += 1
        else:
            out.append(sentence)
    return "".join(out), removed


def scan_value(value: Any) -> list[Detection]:
    if isinstance(value, str):
        return scan(value)
    if isinstance(value, dict):
        return [d for k, v in value.items() if k != "url" for d in scan_value(v)]
    if isinstance(value, list):
        return [d for v in value for d in scan_value(v)]
    return []


def redact_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        total, out = 0, {}
        for k, v in value.items():
            if k == "url":
                out[k] = v
                continue
            out[k], n = redact_value(v)
            total += n
        return out, total
    if isinstance(value, list):
        total, out = 0, []
        for v in value:
            r, n = redact_value(v)
            out.append(r)
            total += n
        return out, total
    return value, 0
