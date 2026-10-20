"""Numeric grounding, shared by the runtime (claim checks) and the graders.

A figure in a report is grounded if it appears in text the agent actually read (or in the question), or if it
is one arithmetic step (a sum, a difference, or a ratio) away from two grounded figures stated in the same
text ("$99 vs $79: $20 more", "2,100 and 1,350: 3,450 orders"). Anything further (a computed CAGR, a figure
from pre-training) is unsupported.
"""
from __future__ import annotations

import itertools
import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[str]:
    """Numeric tokens, normalized: thousands separators dropped, trailing .0 dropped."""
    out = set()
    for raw in _NUMBER.findall(text):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def significant(nums: set[str]) -> set[str]:
    """Drop small integers: counts and list markers ("3 challenges", "1)") rather than figures."""
    return {n for n in nums if not (n.isdigit() and int(n) <= 12)}


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.051, abs(b) * 0.005)


def derivable(n: str, known: set[str], limit: int = 60) -> bool:
    """One arithmetic step (sum, difference, ratio x or %) between two known figures."""
    try:
        target = float(n)
    except ValueError:
        return False
    values = sorted({float(k) for k in known if k}, reverse=True)[:limit]
    for a, b in itertools.combinations(values, 2):
        if _close(a + b, target) or _close(abs(a - b), target):
            return True
        if b and (_close(a / b, target) or _close(a / b * 100, target) or _close((a - b) / b * 100, target)):
            return True
    return False


def unsupported(text: str, read: str, *, ignore_small: bool = True) -> list[str]:
    """Figures in `text` that are neither in `read` nor one arithmetic step from two grounded figures that
    `text` itself states. A derivation must show its inputs ("$99 vs $79, $20 more"): deriving from *any*
    pair of the dozens of figures read would accept almost anything by coincidence."""
    known = numbers(read)
    bad: set[str] = set()
    # inputs must sit in the same sentence as the derived figure: even within one report, any pair of a dozen
    # stated figures sums to something ($1.60/kg + $11.8B = "13.4%"), so the sentence is the unit of evidence
    # (window: the sentence and the one before it, as in "A holds 26%, B 18%. Together they hold 44%.")
    sentences = _SENTENCE.split(text)
    for i, sentence in enumerate(sentences):
        stated = numbers(sentence)
        found = significant(stated) if ignore_small else stated
        inputs = (stated | (numbers(sentences[i - 1]) if i else set())) & known
        bad |= {n for n in found - known if not derivable(n, inputs)}
    return sorted(bad)


_SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")
