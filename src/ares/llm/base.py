"""Provider-neutral LLM request/response types and the client protocol."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # raw JSON text as produced by the model; parsing it is the runtime's job


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class LLMRequest(BaseModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float = 0.0
    seed: int | None = None
    max_tokens: int = 1024

    def fingerprint(self) -> str:
        """Stable hash of everything that determines the response; the replay key."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    reasoning: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    server_timings: dict[str, Any] | None = None

    def assistant_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        return msg


class LLMClient(Protocol):
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the model's response, or raise ares.errors.AgentError. Never retries by itself:
        retrying is the runtime's recovery policy, so it stays visible and measurable."""
        ...
