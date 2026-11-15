"""A thin wrapper around the Groq chat API that always asks for JSON and caches answers.

Why a cache: the free tier allows about 1,000 requests and 200,000 tokens per day.
Every answer is saved to cache/dev/ under a hash of (model, messages), so asking the
same question twice costs nothing and returns exactly the same text.

Why a fallback model: if the main model hits its rate limit (HTTP 429), the same
question is sent once to the smaller model instead of failing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from riskdesk.config import PROJECT_ROOT, load_settings

MAIN_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
DEV_CACHE_DIR = PROJECT_ROOT / "cache" / "dev"

Messages = list[dict[str, str]]


class LLMUnavailableError(RuntimeError):
    """Raised when no answer can be obtained (no key, network down, both models limited)."""


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


def cache_key(model: str, messages: Messages) -> str:
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GroqClient:
    """Ask a Groq chat model for a JSON answer.

    `sdk` can be passed in tests to replace the real Groq SDK object.
    """

    def __init__(
        self,
        *,
        model: str = MAIN_MODEL,
        fallback_model: str | None = FALLBACK_MODEL,
        cache_dir: Path | None = DEV_CACHE_DIR,
        sdk: Any = None,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self.cache_dir = cache_dir
        self._sdk = sdk

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            settings = load_settings()
            if not settings.has_groq:
                raise LLMUnavailableError("GROQ_API_KEY is not set. Add it to the .env file.")
            from groq import Groq

            self._sdk = Groq(api_key=settings.groq_api_key)
        return self._sdk

    # --- cache -----------------------------------------------------------
    def _cache_path(self, model: str, messages: Messages) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{cache_key(model, messages)}.json"

    def _read_cache(self, model: str, messages: Messages) -> LLMReply | None:
        path = self._cache_path(model, messages)
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMReply(
            text=data["text"],
            model=data["model"],
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            latency_s=0.0,
            cached=True,
        )

    def _write_cache(self, messages: Messages, reply: LLMReply) -> None:
        path = self._cache_path(reply.model, messages)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": reply.model,
            "messages": messages,
            "text": reply.text,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- calls -----------------------------------------------------------
    def _call(self, model: str, messages: Messages) -> LLMReply:
        sdk = self._get_sdk()
        start = time.perf_counter()
        response = sdk.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        latency = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        return LLMReply(
            text=response.choices[0].message.content or "",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
            cached=False,
        )

    def chat_json(self, messages: Messages) -> LLMReply:
        """Return the model's answer text (expected to be JSON), from cache if possible."""
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        for model in models:
            hit = self._read_cache(model, messages)
            if hit is not None:
                return hit

        last_error: Exception | None = None
        for model in models:
            try:
                reply = self._call(model, messages)
            except LLMUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - SDK raises many error types
                last_error = exc
                if _is_rate_limit(exc):
                    continue  # try the fallback model
                if _is_invalid_json(exc):
                    # The model answered, but not with valid JSON (Groq rejects it server-side). That is a
                    # bad answer, not an outage: return it uncached so the caller's checks ask again.
                    return LLMReply(_failed_generation(exc), model, 0, 0, 0.0, False)
                raise LLMUnavailableError(f"Groq request failed: {exc}") from exc
            self._write_cache(messages, reply)
            return reply
        raise LLMUnavailableError(f"All models are rate-limited: {last_error}")


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError"


def _is_invalid_json(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 400 and "json_validate_failed" in str(exc)


def _failed_generation(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("failed_generation") or "")
    return ""
