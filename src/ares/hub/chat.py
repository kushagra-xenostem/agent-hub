"""Minimal JSON-friendly chat helper for the hub's database and browser agents."""

from __future__ import annotations
import re
import json
import time
import httpx
from typing import Any
from ares.config import LLMSettings


class ChatError(RuntimeError):
    pass


class Chat:
    def __init__(self, settings: LLMSettings, http: httpx.Client | None = None):
        self.s = settings
        self.http = http or httpx.Client(timeout=httpx.Timeout(settings.timeout_s, connect=15))
        self.calls = 0

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int = 800, temperature: float = 0.0) -> str:
        payload = {"model": self.s.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        last = None
        for attempt in range(3):
            try:
                r = self.http.post(f"{self.s.base_url}/chat/completions", json=payload,
                                   headers={"Authorization": f"Bearer {self.s.api_key}"})
                if r.status_code == 200:
                    self.calls += 1
                    return (r.json()["choices"][0]["message"].get("content") or "").strip()
                last = ChatError(f"LLM server returned {r.status_code}: {r.text[:200]}")
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
                wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
            except httpx.HTTPError as e:
                last, wait = ChatError(f"Cannot reach LLM server: {e}"), 2 * (attempt + 1)
            time.sleep(min(wait, 20))
        raise last or ChatError("LLM call failed")


def parse_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    cand = (m.group(1) if m else text).strip()
    starts = sorted((i, o, c) for o, c in (("{", "}"), ("[", "]")) if (i := cand.find(o)) >= 0)
    for i, o, c in starts:
        j = cand.rfind(c)
        if j > i:
            try:
                return json.loads(cand[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ChatError("Model did not return valid JSON")
