"""Structured tool contracts.

A tool is a typed function plus the metadata the runtime needs to call it safely: input and output
schemas, the permission scope it requires, whether it has side effects (so retries must be idempotent),
whether its output is untrusted (external content that may carry prompt injections), and a timeout.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ares.tools.corpus import Corpus


class Scope(StrEnum):
    WEB_READ = "web.read"
    NOTES_READ = "notes.read"
    NOTES_WRITE = "notes.write"
    EXTERNAL_SEND = "external.send"


class StrictModel(BaseModel):
    """Base for tool inputs/outputs: unknown fields are an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


class ToolFailure(Exception):
    """Raised by a tool handler for an expected failure (404, upstream error, ...)."""

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


@dataclass
class ToolContext:
    """The world a run's tools act on. Fresh per run, so runs cannot affect each other."""

    run_id: str
    corpus: Corpus
    notes: dict[str, str] = field(default_factory=dict)
    outbox: list[dict[str, Any]] = field(default_factory=list)
    idempotency: dict[str, Any] = field(default_factory=dict)  # completed side-effect results by key
    idempotency_key: str | None = None  # key of the call currently executing, for service-side dedup


Handler = Callable[[Any, ToolContext], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Handler
    scope: Scope | None = None  # None: always allowed
    side_effects: bool = False
    untrusted_output: bool = False
    requires_approval: bool = False
    timeout_s: float = 10.0
    source_field: str | None = None  # output field holding a URL the agent has now read (for grounding)
    recipient_field: str | None = None  # argument naming an outbound recipient (egress allowlist)

    def schema(self) -> dict[str, Any]:
        """OpenAI tool definition, generated from the input model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strip_titles(self.input_model.model_json_schema()),
            },
        }


def _strip_titles(node: Any) -> Any:
    """Drop pydantic's auto-generated "title" annotations, but never a *property* called "title"."""
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "title" and isinstance(v, str):
            continue  # an annotation; property schemas are dicts, so they survive
        if k in ("properties", "$defs"):
            out[k] = {name: _strip_titles(sub) for name, sub in v.items()}
        else:
            out[k] = _strip_titles(v)
    return out
