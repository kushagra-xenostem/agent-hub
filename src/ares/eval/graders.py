"""Deterministic graders: judge a run from its result and the true corpus, without an LLM judge."""

from __future__ import annotations
import re
from ares.tools.corpus import Corpus

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def claim_supported(claim: str, source_text: str) -> bool:
    """Every number in the claim appears in the true source; a claim without numbers needs word overlap."""
    nums = numbers(claim)
    if nums:
        return nums <= numbers(source_text)
    words = {w for w in re.findall(r"[a-z]{5,}", claim.lower())}
    return bool(words) and len(words & set(re.findall(r"[a-z]{5,}", source_text.lower()))) / len(words) >= 0.5


def faithfulness(final: dict | None, corpus: Corpus) -> float:
    """Fraction of findings whose claim is supported by the *true* page they cite (0 when no report)."""
    findings = (final or {}).get("findings") or []
    if not findings:
        return 0.0
    ok = 0
    for f in findings:
        if not isinstance(f, dict):
            continue
        doc = corpus.get(str(f.get("source_url", "")))
        if doc is not None and claim_supported(str(f.get("claim", "")), doc.text):
            ok += 1
    return ok / len(findings)
