"""Applies a FaultPlan by wrapping tool handlers and the LLM client.

The agent itself is unchanged: faults arrive through the same interfaces real failures do, so recovery
code cannot tell them apart. Every injection is recorded and handed to the agent's trace.
"""

from __future__ import annotations
import re
import json
import random
import threading
from typing import Any
from collections import Counter
from pydantic import BaseModel, Field
from ares.tools.registry import ToolRegistry
from ares.errors import AgentError, ErrorKind
from ares.faults.plan import FaultKind, FaultPlan, FaultSpec
from ares.llm.base import LLMClient, LLMRequest, LLMResponse
from ares.tools.contracts import ToolContext, ToolFailure, ToolSpec


class Injection(BaseModel):
    target: str
    kind: FaultKind
    call: int  # 1-based call number of the target
    detail: dict[str, Any] = Field(default_factory=dict)


class FaultInjector:
    """Use one injector per run; its call counters are part of what makes injection reproducible."""

    def __init__(self, plan: FaultPlan, seed: int | str = 0):
        self.plan = plan
        self.seed = seed
        self.calls: Counter[str] = Counter()
        self.injected: Counter[int] = Counter()  # by fault index
        self.log: list[Injection] = []
        self._undrained: list[Injection] = []
        self._lock = threading.Lock()

    # --- decisions ------------------------------------------------------------------------------
    def decide(self, target: str) -> tuple[FaultSpec, random.Random] | None:
        """Count a call to `target` and return the fault to inject (with its own RNG), if any."""
        with self._lock:
            self.calls[target] += 1
            n = self.calls[target]
            for i, f in enumerate(self.plan.faults):
                if f.kind == FaultKind.LLM_CONTEXT_LIMIT or not f.matches(target):
                    continue
                if f.max_injections is not None and self.injected[i] >= f.max_injections:
                    continue
                if f.on_calls is not None and n not in f.on_calls:
                    continue
                rng = random.Random(f"{self.seed}:{i}:{target}:{n}")
                if rng.random() < f.rate:
                    self.injected[i] += 1
                    return f, rng
            return None

    def record(self, target: str, kind: FaultKind, **detail: Any) -> None:
        with self._lock:
            inj = Injection(target=target, kind=kind, call=self.calls[target], detail=detail)
            self.log.append(inj)
            self._undrained.append(inj)

    def drain(self) -> list[Injection]:
        with self._lock:
            out, self._undrained = self._undrained, []
            return out

    # --- wrapping -------------------------------------------------------------------------------
    def wrap_registry(self, registry: ToolRegistry) -> ToolRegistry:
        targeted = {f.target for f in self.plan.faults if not f.is_llm}
        names = registry.names() if "*" in targeted else [n for n in registry.names() if n in targeted]
        return registry.with_overrides(**{n: {"handler": self._faulty_handler(registry.get(n))} for n in names})

    def wrap_llm(self, llm: LLMClient) -> LLMClient:
        return FaultyLLM(llm, self) if any(f.is_llm for f in self.plan.faults) else llm

    def _faulty_handler(self, spec: ToolSpec):
        original = spec.handler

        def handler(args: Any, ctx: ToolContext) -> Any:
            decision = self.decide(spec.name)
            if decision is None:
                return original(args, ctx)
            f, rng = decision
            if f.kind == FaultKind.TOOL_ERROR:
                self.record(spec.name, f.kind, transient=f.transient)
                raise ToolFailure("injected: upstream returned 503" if f.transient else "injected: 404 Not Found",
                                  transient=f.transient)
            if f.kind == FaultKind.TOOL_TIMEOUT:
                if f.after_effect:
                    original(args, ctx)  # the side effect happens; the caller just never hears back
                self.record(spec.name, f.kind, after_effect=f.after_effect)
                raise AgentError(ErrorKind.TOOL_TIMEOUT, f"injected: '{spec.name}' did not respond",
                                 detail={"injected": True})
            output = _as_dict(original(args, ctx))
            if f.kind == FaultKind.TOOL_MALFORMED:
                variant, bad = _malform(output, rng)
                self.record(spec.name, f.kind, variant=variant)
                return bad
            if f.kind == FaultKind.TOOL_EMPTY:
                self.record(spec.name, f.kind)
                return _emptied(output)
            if f.kind == FaultKind.TOOL_CONFLICT:
                factor = rng.choice([0.6, 0.75, 1.3, 1.5])
                self.record(spec.name, f.kind, factor=factor)
                return _scale_numbers(output, factor)
            if f.kind == FaultKind.TOOL_OVERSIZED:
                self.record(spec.name, f.kind, added_chars=f.oversize_chars)
                return _oversized(output, f.oversize_chars)
            raise ValueError(f"not a tool fault: {f.kind}")

        return handler

    def context_limits(self) -> list[FaultSpec]:
        return [f for f in self.plan.faults if f.kind == FaultKind.LLM_CONTEXT_LIMIT]


class FaultyLLM:
    def __init__(self, inner: LLMClient, injector: FaultInjector):
        self.inner = inner
        self.injector = injector
        self.model = inner.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        inj = self.injector
        for f in inj.context_limits():
            size = int((len(json.dumps(request.messages)) + len(json.dumps(request.tools))) / 3.5)
            if size > f.limit_tokens:
                inj.calls["llm"] += 1
                inj.record("llm", f.kind, estimated_tokens=size, limit=f.limit_tokens)
                raise AgentError(ErrorKind.LLM_CONTEXT_OVERFLOW,
                                 f"injected: prompt of ~{size} tokens exceeds the {f.limit_tokens}-token window",
                                 detail={"injected": True})
        decision = inj.decide("llm")
        if decision:
            f, _ = decision
            if f.kind == FaultKind.LLM_RATE_LIMIT:
                inj.record("llm", f.kind, retry_after_s=f.retry_after_s)
                raise AgentError(ErrorKind.LLM_RATE_LIMITED, "injected: rate limit exceeded",
                                 retry_after_s=f.retry_after_s, detail={"injected": True})
            if f.kind == FaultKind.LLM_UNAVAILABLE:
                inj.record("llm", f.kind)
                raise AgentError(ErrorKind.LLM_UNAVAILABLE, "injected: 503 model loading", detail={"injected": True})
            if f.kind == FaultKind.LLM_TIMEOUT:
                inj.record("llm", f.kind)
                raise AgentError(ErrorKind.LLM_TIMEOUT, "injected: request timed out", detail={"injected": True})

        resp = self.inner.complete(request)
        if decision and decision[0].kind == FaultKind.LLM_GARBLED and resp.tool_calls:
            tc = resp.tool_calls[0]
            tc.arguments = tc.arguments[: max(1, len(tc.arguments) // 2)]
            inj.record("llm", FaultKind.LLM_GARBLED, tool=tc.name)
        elif decision and decision[0].kind == FaultKind.LLM_NO_TOOL:
            inj.record("llm", FaultKind.LLM_NO_TOOL)
            resp = LLMResponse(content="I have gathered enough information to answer.", finish_reason="stop",
                               usage=resp.usage, latency_s=resp.latency_s)
        return resp


# --- output corruption helpers ------------------------------------------------------------------
def _as_dict(output: Any) -> Any:
    return output.model_dump() if isinstance(output, BaseModel) else output


def _malform(output: Any, rng: random.Random) -> tuple[str, Any]:
    variant = rng.choice(["html_error_page", "missing_field", "wrong_type"])
    if variant == "html_error_page" or not isinstance(output, dict) or not output:
        return "html_error_page", "<html><body><h1>502 Bad Gateway</h1></body></html>"
    if variant == "missing_field":
        bad = dict(output)
        bad.pop(sorted(bad)[-1])
        return variant, bad
    bad = dict(output)
    key = sorted(bad)[0]
    bad[key] = 12345 if isinstance(bad[key], str) else "corrupted"
    return variant, bad


def _emptied(output: Any) -> Any:
    if not isinstance(output, dict):
        return output
    return {k: ([] if isinstance(v, list) else "" if k == "text" else v) for k, v in output.items()}


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _scale(text: str, factor: float) -> str:
    def repl(m: re.Match) -> str:
        if len(m.group()) == 4 and m.group().startswith(("19", "20")):  # keep years
            return m.group()
        value = float(m.group()) * factor
        return f"{value:.1f}" if "." in m.group() else str(round(value))
    return _NUMBER.sub(repl, text)


def _scale_numbers(output: Any, factor: float) -> Any:
    if isinstance(output, str):
        return _scale(output, factor)
    if isinstance(output, list):
        return [_scale_numbers(v, factor) for v in output]
    if isinstance(output, dict):
        return {k: (v if k in ("url", "query") else _scale_numbers(v, factor)) for k, v in output.items()}
    return output


def _oversized(output: Any, chars: int) -> Any:
    if isinstance(output, dict) and isinstance(output.get("text"), str):
        filler = " Additional archived commentary follows." * (chars // 40 + 1)
        return {**output, "text": output["text"] + filler[:chars]}
    return output
