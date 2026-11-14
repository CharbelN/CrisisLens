"""Translator retry logic and the client cache, with fake models (no network)."""

import json
from types import SimpleNamespace

import pytest

from riskdesk.checks.moves import Limits
from riskdesk.engine.drivers import DRIVER_IDS
from riskdesk.llm.client import GroqClient, LLMReply, LLMUnavailableError
from riskdesk.llm.translator import MAX_RETRIES, translate

LIMITS = {d: Limits(-90.0, 500.0) for d in DRIVER_IDS} | {"us10y": Limits(-300.0, 300.0)}
GOOD = json.dumps({"moves": {d: 0 for d in DRIVER_IDS} | {"oil": 40, "equity": -8}, "reasoning": "r", "events_used": []})
TOO_BIG = json.dumps({"moves": {d: 0 for d in DRIVER_IDS} | {"equity": -150}})


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def chat_json(self, messages):
        self.calls.append(messages)
        return LLMReply(self.answers.pop(0), "fake", 10, 5, 0.1, False)


def test_good_first_try():
    res = translate("Oil jumps", FakeClient([GOOD]), LIMITS)
    assert res.ok and res.attempts == 1 and res.moves["oil"] == 40
    assert res.prompt_tokens == 10 and res.completion_tokens == 5


def test_bad_json_then_good():
    client = FakeClient(["oops", GOOD])
    res = translate("Oil jumps", client, LIMITS)
    assert res.ok and res.attempts == 2
    # the second request shows the model its own answer and the problem
    assert client.calls[1][-2] == {"role": "assistant", "content": "oops"}
    assert "not valid JSON" in client.calls[1][-1]["content"]


def test_out_of_range_then_good():
    res = translate("Stocks crash", FakeClient([TOO_BIG, GOOD]), LIMITS)
    assert res.ok and res.attempts == 2
    assert "outside the plausible range" in res.errors[0][0]


def test_always_bad_fails_after_retries():
    res = translate("x", FakeClient(["bad"] * 3), LIMITS)
    assert res.status == "failed" and res.moves is None
    assert res.attempts == 1 + MAX_RETRIES == 3


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        translate("  ", FakeClient([]), LIMITS)


# --- client ------------------------------------------------------------------


class FakeSDK:
    """Mimics groq.Groq().chat.completions.create."""

    def __init__(self, behaviour):
        self.behaviour = behaviour  # model -> text or Exception
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kwargs):
        self.calls.append(model)
        out = self.behaviour[model]
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=out))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )


class RateLimitError(Exception):
    status_code = 429


MSG = [{"role": "user", "content": "hi"}]


def test_cache_second_call_is_free(tmp_path):
    sdk = FakeSDK({"big": "{}"})
    client = GroqClient(model="big", fallback_model=None, cache_dir=tmp_path, sdk=sdk)
    first = client.chat_json(MSG)
    second = client.chat_json(MSG)
    assert not first.cached and second.cached
    assert second.text == first.text and sdk.calls == ["big"]


def test_rate_limit_falls_back(tmp_path):
    sdk = FakeSDK({"big": RateLimitError("429"), "small": '{"a": 1}'})
    client = GroqClient(model="big", fallback_model="small", cache_dir=tmp_path, sdk=sdk)
    reply = client.chat_json(MSG)
    assert reply.model == "small" and sdk.calls == ["big", "small"]


def test_other_errors_raise_clearly(tmp_path):
    sdk = FakeSDK({"big": ConnectionError("down")})
    client = GroqClient(model="big", fallback_model="small", cache_dir=tmp_path, sdk=sdk)
    with pytest.raises(LLMUnavailableError):
        client.chat_json(MSG)
