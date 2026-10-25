"""Client for OpenAI-compatible chat-completions APIs (the project #1 llama.cpp server)."""

from __future__ import annotations
import json
import time
import httpx
from typing import Any
from ares.config import LLMSettings
from ares.errors import AgentError, ErrorKind
from ares.llm.base import LLMRequest, LLMResponse, ToolCall, Usage


class OpenAICompatClient:
    def __init__(self, settings: LLMSettings, http: httpx.Client | None = None):
        self.model = settings.model
        self._http = http or httpx.Client(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.timeout_s, connect=10.0),
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )

    @classmethod
    def from_env(cls) -> OpenAICompatClient:
        return cls(LLMSettings.from_env())

    def list_models(self) -> list[str]:
        r = self._send("GET", "/models")
        data = r.json()
        items = data.get("data") or data.get("models") or []
        return [m.get("id") or m.get("model") or m.get("name") for m in items]

    def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.tools:
            body["tools"] = request.tools
        if request.seed is not None:
            body["seed"] = request.seed

        t0 = time.perf_counter()
        r = self._send("POST", "/chat/completions", json=body)
        latency = time.perf_counter() - t0
        try:
            data = r.json()
            choice = data["choices"][0]
            msg = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise AgentError(ErrorKind.LLM_UNAVAILABLE, f"unparseable LLM response: {r.text[:200]}") from e

        fp = request.fingerprint()
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):  # some servers return an object instead of JSON text
                args = json.dumps(args if args is not None else {})
            # deterministic fallback id keeps replays stable when the server omits one
            calls.append(ToolCall(id=tc.get("id") or f"call_{fp[:8]}_{i}", name=fn.get("name", ""), arguments=args))

        usage = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content"),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            reasoning=msg.get("reasoning_content"),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            ),
            latency_s=round(latency, 3),
            server_timings=data.get("timings"),
        )

    def _send(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.TimeoutException as e:
            raise AgentError(ErrorKind.LLM_TIMEOUT, f"LLM request timed out: {e}") from e
        except httpx.TransportError as e:
            raise AgentError(ErrorKind.LLM_UNAVAILABLE, f"cannot reach the LLM API: {e}") from e
        if r.status_code != 200:
            raise http_error(r)
        return r


def http_error(r: httpx.Response) -> AgentError:
    try:
        message = r.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = r.text[:300]
    status = r.status_code
    detail = {"status": status}
    if status in (401, 403):
        return AgentError(ErrorKind.LLM_AUTH, f"LLM API rejected the key ({status}): {message}", detail=detail)
    if status == 429:
        retry_after = None
        try:
            retry_after = float(r.headers.get("Retry-After", ""))
        except ValueError:
            pass
        return AgentError(ErrorKind.LLM_RATE_LIMITED, f"rate limited: {message}", retry_after_s=retry_after, detail=detail)
    lowered = message.lower()
    if status == 413 or (status == 400 and ("context" in lowered or "too long" in lowered or "exceed" in lowered)):
        return AgentError(ErrorKind.LLM_CONTEXT_OVERFLOW, f"prompt exceeds the context window: {message}", detail=detail)
    if status == 500 and "parse" in lowered and "tool call" in lowered:
        # llama.cpp re-parses tool-call arguments in the history; the same request will always fail
        return AgentError(ErrorKind.LLM_BAD_REQUEST, f"server rejected the conversation history: {message}",
                          detail=detail)
    if status == 504:
        return AgentError(ErrorKind.LLM_TIMEOUT, f"upstream timeout: {message}", detail=detail)
    if status >= 500:
        return AgentError(ErrorKind.LLM_UNAVAILABLE, f"LLM API error {status}: {message}", detail=detail)
    return AgentError(ErrorKind.LLM_BAD_REQUEST, f"LLM API rejected the request ({status}): {message}", detail=detail)
