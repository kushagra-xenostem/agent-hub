"""Runs one tool call: idempotency, timeout, error classification and output-contract validation."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from ares.errors import AgentError, ErrorKind, format_validation_error
from ares.tools.contracts import ToolContext, ToolFailure, ToolSpec


@dataclass
class ExecResult:
    output: Any
    deduplicated: bool = False


def idempotency_key(tool: str, raw_args: dict[str, Any]) -> str:
    blob = json.dumps({"tool": tool, "args": raw_args}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def execute_tool(spec: ToolSpec, args: Any, ctx: ToolContext, *, raw_args: dict[str, Any],
                 validate_output: bool) -> ExecResult:
    key = None
    if spec.side_effects:
        key = idempotency_key(spec.name, raw_args)
        if key in ctx.idempotency:  # already completed in this run: return the recorded result
            return ExecResult(ctx.idempotency[key], deduplicated=True)

    ctx.idempotency_key = key
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{spec.name}")
    try:
        future = pool.submit(spec.handler, args, ctx)
        try:
            output = future.result(timeout=spec.timeout_s)
        except FuturesTimeout:
            raise AgentError(ErrorKind.TOOL_TIMEOUT, f"'{spec.name}' did not finish within {spec.timeout_s:g}s") from None
        except AgentError:
            raise
        except ToolFailure as e:
            raise AgentError(ErrorKind.TOOL_ERROR, f"'{spec.name}' failed: {e}", transient=e.transient) from e
        except Exception as e:  # a bug or an unexpected input reaching the handler
            raise AgentError(ErrorKind.TOOL_ERROR, f"'{spec.name}' crashed: {type(e).__name__}: {e}") from e
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # a timed-out handler is abandoned, not awaited
        ctx.idempotency_key = None

    if validate_output:
        try:
            payload = output.model_dump() if isinstance(output, BaseModel) else output
            output = spec.output_model.model_validate(payload)
        except ValidationError as e:
            raise AgentError(ErrorKind.TOOL_OUTPUT_INVALID,
                             f"'{spec.name}' returned output that violates its contract: {format_validation_error(e)}") from e

    if key:
        ctx.idempotency[key] = output
    return ExecResult(output)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)
