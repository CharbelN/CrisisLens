"""One client for every language-model provider: always asks for JSON, caches every answer.

Groq and Gemini both offer an OpenAI-compatible endpoint, so a provider is only settings
(a URL, a key and a model name) and one `openai` SDK does all the calls.
The client tries the providers in order: Groq 120b, Groq 20b, then Gemini Flash-Lite as the
backup if GEMINI_API_KEY is set.

When it moves on to the next provider:
- rate limit (HTTP 429), server error (5xx), network failure, or the provider's key is
  missing or rejected (401 / 403).
Any other error (a bad request) stops at once with LLMUnavailableError.
A broken-JSON answer is NOT an outage: Groq rejects it server-side (`json_validate_failed`) and
returns the bad text, which is handed back so the caller's checks ask the SAME provider again.

Why a cache: the free tiers are small (Groq: about 1,000 requests and 200,000 tokens per day).
Every answer is saved to cache/dev/ under a hash of (model, messages), so asking the same
question twice costs nothing and returns exactly the same text. Each record also stores the
provider and model that produced it.

Demo cache: answers for the demo scenarios are saved in cache/demo/ (committed, read-only for
the app) and looked up FIRST, so the demo works with no live call (see scripts/build_demo_cache.py).

Evaluations use `pinned_client()` / `PacedClient`, so an answer from another provider never
slips silently into the measured numbers.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from riskdesk.config import PROJECT_ROOT, Settings, load_settings

DEV_CACHE_DIR = PROJECT_ROOT / "cache" / "dev"
DEMO_CACHE_DIR = PROJECT_ROOT / "cache" / "demo"

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class Provider:
    """Where to send a question: an OpenAI-compatible endpoint, its key and one model."""

    name: str  # "groq", "gemini"
    model: str
    base_url: str
    key_setting: str  # attribute of Settings that holds the key
    extra: dict[str, Any] = field(default_factory=dict)  # extra request fields for this provider

    def key(self, settings: Settings) -> str | None:
        return getattr(settings, self.key_setting)


GROQ_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

GROQ_MAIN = Provider("groq", "openai/gpt-oss-120b", GROQ_URL, "groq_api_key")
GROQ_SMALL = Provider("groq", "openai/gpt-oss-20b", GROQ_URL, "groq_api_key")
# Gemini's OpenAI layer is still in beta, so it comes last. Gemini 3 always reasons: keep it short.
GEMINI_LITE = Provider("gemini", "gemini-flash-lite-latest", GEMINI_URL, "gemini_api_key",
                       {"reasoning_effort": "low"})

MAIN_MODEL = GROQ_MAIN.model


class LLMUnavailableError(RuntimeError):
    """Raised when no answer can be obtained (no key, network down, every provider limited)."""


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    provider: str = ""


def cache_key(model: str, messages: Messages) -> str:
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_providers(settings: Settings | None = None) -> list[Provider]:
    """Groq first, then Gemini as the backup if its key is set."""
    settings = settings or load_settings()
    providers = [GROQ_MAIN, GROQ_SMALL]
    if settings.has_gemini:
        providers.append(GEMINI_LITE)
    return providers


class LLMClient:
    """Ask the providers in order for a JSON answer, from cache if possible.

    `sdk` can be passed in tests: one fake object used for every provider.
    """

    def __init__(
        self,
        providers: list[Provider] | None = None,
        *,
        cache_dir: Path | None = DEV_CACHE_DIR,
        demo_dir: Path | None = DEMO_CACHE_DIR,
        sdk: Any = None,
        settings: Settings | None = None,
    ) -> None:
        self.providers = providers if providers is not None else default_providers(settings)
        self.cache_dir = cache_dir  # new answers are saved here
        self.demo_dir = demo_dir  # read first, never written by the app
        self._fake_sdk = sdk
        self._settings = settings
        self._sdks: dict[tuple[str, str], Any] = {}

    def _get_sdk(self, provider: Provider) -> Any:
        if self._fake_sdk is not None:
            return self._fake_sdk
        if self._settings is None:
            self._settings = load_settings()
        key = provider.key(self._settings)
        if key is None:
            raise LLMUnavailableError(f"{provider.key_setting.upper()} is not set.")
        cache_id = (provider.base_url, key)
        if cache_id not in self._sdks:
            from openai import OpenAI

            # max_retries=0: on a limit we move to the next provider instead of waiting.
            self._sdks[cache_id] = OpenAI(api_key=key, base_url=provider.base_url, max_retries=0)
        return self._sdks[cache_id]

    # --- cache -----------------------------------------------------------
    def _read_cache(self, model: str, messages: Messages) -> LLMReply | None:
        for folder in (self.demo_dir, self.cache_dir):
            if folder is None:
                continue
            path = folder / f"{cache_key(model, messages)}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return LLMReply(
                    text=data["text"],
                    model=data["model"],
                    prompt_tokens=data.get("prompt_tokens", 0),
                    completion_tokens=data.get("completion_tokens", 0),
                    latency_s=0.0,
                    cached=True,
                    provider=data.get("provider", "groq"),  # older records have no provider field
                )
        return None

    def _write_cache(self, messages: Messages, reply: LLMReply, folder: Path | None = None) -> None:
        folder = folder or self.cache_dir
        if folder is None:
            return
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            "provider": reply.provider,
            "model": reply.model,
            "messages": messages,
            "text": reply.text,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
        }
        path = folder / f"{cache_key(reply.model, messages)}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- calls -----------------------------------------------------------
    def _call(self, provider: Provider, messages: Messages) -> LLMReply:
        sdk = self._get_sdk(provider)
        start = time.perf_counter()
        response = sdk.chat.completions.create(
            model=provider.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            **provider.extra,
        )
        latency = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        return LLMReply(
            text=response.choices[0].message.content or "",
            model=provider.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
            cached=False,
            provider=provider.name,
        )

    def chat_json(self, messages: Messages) -> LLMReply:
        """Return the answer text (expected to be JSON) from the first provider that gives one."""
        for provider in self.providers:
            hit = self._read_cache(provider.model, messages)
            if hit is not None:
                return hit

        problems: list[str] = []
        for provider in self.providers:
            label = f"{provider.name} {provider.model}"
            try:
                reply = self._call(provider, messages)
            except LLMUnavailableError as exc:  # key not set: try the next provider
                problems.append(f"{label}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - the SDK raises many error types
                if _is_invalid_json(exc):
                    # The model answered, but not with valid JSON. That is a bad answer, not an outage:
                    # return it uncached so the caller's checks ask this same provider again.
                    return LLMReply(_failed_generation(exc), provider.model, 0, 0, 0.0, False, provider.name)
                if _should_switch(exc):
                    problems.append(f"{label}: {exc}")
                    continue
                raise LLMUnavailableError(f"{label} request failed: {exc}") from exc
            self._write_cache(messages, reply)
            return reply
        raise LLMUnavailableError("No provider could answer. " + " | ".join(problems))


def default_client() -> LLMClient:
    """For the app: Groq, then Gemini if its key is set."""
    return LLMClient()


def pinned_client(*providers: Provider) -> LLMClient:
    """For evaluations: only the providers named (default Groq 120b), so no silent switch."""
    return LLMClient(list(providers) or [GROQ_MAIN])


def save_reply(folder: Path, messages: Messages, reply: LLMReply) -> None:
    """Save one answer into a cache folder (used to build the demo cache)."""
    LLMClient([], cache_dir=folder, demo_dir=None)._write_cache(messages, reply)


class PacedClient:
    """Groq 120b only; waits after each new answer so tokens per minute stay under the free limit.

    For long batch runs (evaluations, building the demo cache), not for the app.
    """

    def __init__(self, tokens_per_minute: int = 7000, inner: LLMClient | None = None) -> None:
        self.inner = inner or pinned_client(GROQ_MAIN)
        self.tokens_per_minute = tokens_per_minute

    def chat_json(self, messages: Messages) -> LLMReply:
        for wait in (0, 60, 120):  # on a rate limit, wait and try again
            time.sleep(wait)
            try:
                reply = self.inner.chat_json(messages)
            except LLMUnavailableError as exc:
                if "tokens per day" in str(exc).lower():
                    raise SystemExit("Groq's daily token limit is used up. Answers so far are cached: "
                                     "run again later and it resumes where it stopped.") from exc
                if "rate" not in str(exc).lower() and "429" not in str(exc):
                    raise
                print(f"   rate-limited, waiting {wait + 60} s")
                continue
            if not reply.cached:
                time.sleep(60 * (reply.prompt_tokens + reply.completion_tokens) / self.tokens_per_minute)
            return reply
        raise LLMUnavailableError("still rate-limited after waiting")


# --- error sorting -----------------------------------------------------------
_NETWORK_ERRORS = {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"}


def _should_switch(exc: Exception) -> bool:
    """Rate limit, server error, rejected key or network failure: worth trying the next provider."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403, 429) or (isinstance(status, int) and status >= 500):
        return True
    return type(exc).__name__ in _NETWORK_ERRORS or isinstance(exc, (ConnectionError, TimeoutError))


def _is_invalid_json(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 400 and "json_validate_failed" in str(exc)


def _failed_generation(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)  # the openai SDK already unwraps "error"; others may not
        if isinstance(err, dict):
            return str(err.get("failed_generation") or "")
    return ""
