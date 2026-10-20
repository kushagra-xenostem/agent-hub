"""Settings, read from the project's .env file. Process environment variables win over the file."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


class LLMSettings(BaseModel):
    base_url: str
    api_key: str = Field(repr=False)
    model: str
    timeout_s: float = 300.0

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> LLMSettings:
        values = read_env_file(env_file or PROJECT_ROOT / ".env")
        values.update({k: v for k, v in os.environ.items() if k.startswith("LLM_")})
        missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not values.get(k)]
        if missing:
            raise RuntimeError(f"missing settings {', '.join(missing)}: set them in .env")
        return cls(
            base_url=values["LLM_BASE_URL"].rstrip("/"),
            api_key=values["LLM_API_KEY"],
            model=values["LLM_MODEL"],
            timeout_s=float(values.get("LLM_TIMEOUT_S", 300)),
        )
