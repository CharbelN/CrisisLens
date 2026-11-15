"""Central settings for the app.

Reads API keys and switches from environment variables, which are loaded
from a local `.env` file if one exists. The `.env` file is never committed.

- GROQ_API_KEY   : main language-model engine (needed from build stage 3).
- GEMINI_API_KEY : optional backup engine. Missing is fine, never an error.
- RISKDESK_SHARE : "true" to create a temporary public Gradio link. Off by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TRUE_WORDS = {"1", "true", "yes", "on"}


def _clean(value: str | None) -> str | None:
    """Treat a missing or blank value as 'not set'."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def parse_bool(value: str | None) -> bool:
    """Read an on/off switch from text. Anything not clearly 'on' counts as off."""
    return (value or "").strip().lower() in _TRUE_WORDS


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    gemini_api_key: str | None
    share_enabled: bool

    @property
    def has_groq(self) -> bool:
        return self.groq_api_key is not None

    @property
    def has_gemini(self) -> bool:
        return self.gemini_api_key is not None


def load_settings(env: dict[str, str] | None = None, *, read_dotenv: bool = True) -> Settings:
    """Build the settings.

    `env` lets tests pass a fake environment; by default the real one is used.
    """
    if env is None:
        if read_dotenv:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = dict(os.environ)
    return Settings(
        groq_api_key=_clean(env.get("GROQ_API_KEY")),
        gemini_api_key=_clean(env.get("GEMINI_API_KEY")),
        share_enabled=parse_bool(env.get("RISKDESK_SHARE")),
    )
