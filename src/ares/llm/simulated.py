"""A rule-based stand-in for the model, for large offline experiments.

It researches like a competent but imperfect agent: search, fetch the top results, then report findings
quoted from the fetched pages. Like an LLM it reacts to what it sees in the transcript: it re-issues a
call that failed (correcting bad arguments), moves on from pages that keep failing, obeys the wrap-up
nudge, and trusts whatever page text it receives (so corrupted or altered pages flow into its report).
Its own mistakes (bad arguments, unknown tools, prose replies) happen at configurable rates.

Fully deterministic given its seed and the transcript, so experiments are reproducible and fast; results
are then confirmed on the real model at smaller scale.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any

from ares.llm.base import LLMRequest, LLMResponse, ToolCall, Usage
from ares.tools.corpus import tokenize


class SimulatedResearcher:
    model = "simulated"

    def __init__(self, *, seed: int = 0, p_bad_args: float = 0.0, p_unknown_tool: float = 0.0,
                 p_text_reply: float = 0.0, fetch_k: int = 2, max_findings: int = 4):
        self.seed = seed
        self.p_bad_args = p_bad_args
        self.p_unknown_tool = p_unknown_tool
        self.p_text_reply = p_text_reply
        self.fetch_k = fetch_k
        self.max_findings = max_findings
        self.n = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.n += 1
        rng = random.Random(f"{self.seed}:{self.n}")
        view = _Transcript(request.messages)
        name, args = self._next_action(view)

        usage = Usage(prompt_tokens=int((len(json.dumps(request.messages)) + len(json.dumps(request.tools))) / 3.5),
                      completion_tokens=20 + len(json.dumps(args)) // 4)
        roll = rng.random()
        if roll < self.p_text_reply:
            return LLMResponse(content="Based on my research the market looks promising.", finish_reason="stop",
                               usage=usage)
        if roll < self.p_text_reply + self.p_unknown_tool:
            name = {"search_web": "web_search", "fetch_page": "open_url"}.get(name, "finish")
        elif roll < self.p_text_reply + self.p_unknown_tool + self.p_bad_args:
            args = _break_args(name, args)
        return LLMResponse(tool_calls=[ToolCall(id=f"sim_{self.seed}_{self.n}", name=name, arguments=json.dumps(args))],
                           finish_reason="tool_calls", usage=usage)

    def _next_action(self, t: _Transcript) -> tuple[str, dict[str, Any]]:
        # Errors are handled by re-planning from the transcript (like a model reading the error):
        # failed pages count against that page, failed searches are simply searched again.
        # 1. wrap-up nudge, or enough evidence: report
        candidates = [u for u in t.hits if u not in t.fetched and t.fetch_failures(u) < 3]
        if t.wrapup or len(t.fetched) >= self.fetch_k or (t.searches and not candidates and t.fetched):
            return "submit_report", self._report(t)
        # 2. search first (again with a different query if the last search found nothing)
        if not t.hits:
            query = " ".join(tokenize(t.task)[:6]) if t.searches == 0 else " ".join(tokenize(t.task)[1:5])
            return "search_web", {"query": query or "market outlook", "max_results": 5}
        if candidates:
            return "fetch_page", {"url": candidates[0]}
        return "submit_report", self._report(t)

    def _report(self, t: _Transcript) -> dict[str, Any]:
        findings = []
        for url, text in t.fetched.items():
            text = re.sub(r"\s*\.\.\.\[truncated \d+ chars\]", "", text)  # a notice, not content
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if re.search(r"\d", sentence) and len(sentence) >= 20 and len(findings) < self.max_findings:
                    findings.append({"claim": sentence.strip()[:500], "source_url": url})
        if not findings:  # nothing usable was fetched: an ungrounded report the runtime should reject
            findings = [{"claim": "The market is expected to grow strongly.",
                         "source_url": t.hits[0] if t.hits else "https://unknown.example/"}]
        return {"title": f"Research report: {t.task[:80]}", "summary": " ".join(f["claim"] for f in findings)[:2000]
                or "No findings.", "findings": findings}


class _Transcript:
    """What the simulated researcher can see: the task, tool results, errors and nudges."""

    def __init__(self, messages: list[dict[str, Any]]):
        self.task = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.calls: dict[str, dict[str, Any]] = {}
        self.results: dict[str, Any] = {}
        self.order: list[str] = []
        self.wrapup = False
        for m in messages:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls", []):
                    self.calls[tc["id"]] = {"id": tc["id"], "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"]}
                    self.order.append(tc["id"])
            elif m["role"] == "tool":
                self.results[m["tool_call_id"]] = _loads(m["content"])
            elif m["role"] == "user" and "steps left" in str(m.get("content")):
                self.wrapup = True
        self.hits: list[str] = []
        self.fetched: dict[str, str] = {}
        self.searches = 0
        self._fetch_failures: dict[str, int] = {}
        for cid in self.order:
            call, res = self.calls[cid], self.results.get(cid)
            if res is None:
                continue
            payload = res
            for _ in range(3):  # unwrap loop-guard, injection-warning and untrusted-content wrappers
                if isinstance(payload, dict) and payload.get("untrusted") and "content" in payload:
                    payload = payload["content"]
                elif isinstance(payload, dict) and "result" in payload and ("note" in payload or "warning" in payload):
                    payload = payload["result"]
            if call["name"] == "search_web" and _ok(res):
                self.searches += 1
                hits = (payload.get("results") or payload.get("hits") or []) if isinstance(payload, dict) else []
                for hit in hits:
                    if isinstance(hit, dict) and hit.get("url") not in self.hits:
                        self.hits.append(hit.get("url"))
            elif call["name"] == "fetch_page":
                url = self.intended_args(call).get("url", "")
                if isinstance(payload, dict) and payload.get("compacted"):
                    if payload.get("preview"):
                        self.fetched[payload.get("url") or url] = payload["preview"]  # only the preview survives
                    continue
                if _ok(res) and isinstance(payload, dict) and isinstance(payload.get("text"), str) and payload["text"]:
                    self.fetched[url] = payload["text"]
                elif _ok(res) and isinstance(payload, (dict, str)) and not isinstance(payload, dict):
                    self.fetched[url] = str(payload)  # trusts a malformed page if nothing flags it
                else:
                    self._fetch_failures[url] = self._fetch_failures.get(url, 0) + 1

    def fetch_failures(self, url: str) -> int:
        return self._fetch_failures.get(url, 0)

    @staticmethod
    def intended_args(call: dict[str, Any]) -> dict[str, Any]:
        """What the call was meant to be: broken arguments are repaired the way a model fixes them."""
        args = _loads(call["arguments"])
        if not isinstance(args, dict):
            args = {}
        if call["name"] == "search_web":
            return {"query": str(args.get("query") or "market outlook")[:200],
                    "max_results": min(10, max(1, int(args.get("max_results", 5)) if str(args.get("max_results", 5)).isdigit() else 5))}
        if call["name"] == "fetch_page":
            url = str(args.get("url") or args.get("link") or "")
            return {"url": url if url.startswith("http") else "https://" + url.lstrip("/")}
        return args


def _loads(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def _ok(res: Any) -> bool:
    return not (isinstance(res, dict) and ("error" in res or res.get("status") == "skipped"))


def _break_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "search_web":
        return {**args, "max_results": 50}
    if name == "fetch_page":
        return {"link": args.get("url", "")}  # wrong parameter name
    if name == "submit_report":
        return {k: v for k, v in args.items() if k != "title"}
    return {**args, "unexpected": True}
