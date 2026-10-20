from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any

from ares.tools.contracts import ToolSpec


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()):
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._specs.values())

    def schemas(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(self._specs) if names is None else set(names)
        return [s.schema() for s in self._specs.values() if s.name in wanted]

    def with_overrides(self, **changes_by_tool: dict[str, Any]) -> ToolRegistry:
        """A copy with some specs changed, e.g. a fault-injecting handler or a shorter timeout."""
        return ToolRegistry(
            replace(s, **changes_by_tool[s.name]) if s.name in changes_by_tool else s for s in self._specs.values()
        )
