"""Translator retry logic and the client cache, with fake models (no network)."""

import json
from types import SimpleNamespace

import pytest

from riskdesk.checks.moves import Limits
from riskdesk.engine.drivers import DRIVER_IDS
from riskdesk.config import load_settings
from riskdesk.llm.client import LLMClient, LLMReply, LLMUnavailableError, Provider, default_providers
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
    """Mimics openai.OpenAI().chat.completions.create (the same for every provider)."""

    def __init__(self, behaviour):
        self.behaviour = behaviour  # model -> text or Exception
        self.calls = []
        self.kwargs = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kwargs):
        self.calls.append(model)
        self.kwargs.append(kwargs)
        out = self.behaviour[model]
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=out))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )


class StatusError(Exception):
    def __init__(self, status, msg="", body=None):
        super().__init__(msg or f"Error code: {status}")
        self.status_code, self.body = status, body


def providers(*models, extra=None):
    return [Provider(f"p{i}", m, "http://x", "groq_api_key", extra or {}) for i, m in enumerate(models, 1)]


def make(models, sdk, tmp_path, **kw):
    return LLMClient(providers(*models), cache_dir=tmp_path, demo_dir=None, sdk=sdk, **kw)


MSG = [{"role": "user", "content": "hi"}]


def test_cache_second_call_is_free(tmp_path):
    sdk = FakeSDK({"big": "{}"})
    client = make(["big"], sdk, tmp_path)
    first = client.chat_json(MSG)
    second = client.chat_json(MSG)
    assert not first.cached and second.cached
    assert second.text == first.text and sdk.calls == ["big"]
    assert first.provider == second.provider == "p1"
    record = json.loads(next(tmp_path.iterdir()).read_text(encoding="utf-8"))
    assert record["provider"] == "p1" and record["model"] == "big"


@pytest.mark.parametrize("err", [StatusError(429), StatusError(503), ConnectionError("down")])
def test_limit_server_or_network_error_moves_to_next_provider(tmp_path, err):
    sdk = FakeSDK({"big": err, "small": '{"a": 1}'})
    reply = make(["big", "small"], sdk, tmp_path).chat_json(MSG)
    assert reply.model == "small" and reply.provider == "p2" and sdk.calls == ["big", "small"]


def test_other_errors_stop_at_once(tmp_path):
    sdk = FakeSDK({"big": StatusError(401, "bad key"), "small": "{}"})
    with pytest.raises(LLMUnavailableError):
        make(["big", "small"], sdk, tmp_path).chat_json(MSG)
    assert sdk.calls == ["big"]  # no switch on a non-outage error


def test_all_providers_down_raises_with_every_reason(tmp_path):
    sdk = FakeSDK({"big": StatusError(429), "small": StatusError(500)})
    with pytest.raises(LLMUnavailableError, match="big.*small"):
        make(["big", "small"], sdk, tmp_path).chat_json(MSG)


def test_missing_key_skips_provider(tmp_path):
    settings = load_settings({"GEMINI_API_KEY": "k"})  # no Groq key
    gem = Provider("gemini", "g", "http://x", "gemini_api_key")
    groq = Provider("groq", "q", "http://x", "groq_api_key")
    client = LLMClient([groq, gem], cache_dir=tmp_path, demo_dir=None, settings=settings)
    client._sdks[("http://x", "k")] = FakeSDK({"g": '{"ok": 1}'})  # stands in for the real SDK
    reply = client.chat_json(MSG)
    assert reply.provider == "gemini"


def test_provider_extra_fields_are_sent(tmp_path):
    sdk = FakeSDK({"big": "{}"})
    LLMClient(providers("big", extra={"reasoning_effort": "low"}), cache_dir=tmp_path, demo_dir=None,
              sdk=sdk).chat_json(MSG)
    assert sdk.kwargs[0]["reasoning_effort"] == "low" and sdk.kwargs[0]["temperature"] == 0


def test_default_order_and_optional_backups():
    names = lambda env: [(p.name, p.model) for p in default_providers(load_settings(env))]  # noqa: E731
    assert names({}) == [("groq", "openai/gpt-oss-120b"), ("groq", "openai/gpt-oss-20b")]
    full = names({"OPENROUTER_API_KEY": "a", "GEMINI_API_KEY": "b"})
    assert [n for n, _ in full] == ["groq", "groq", "openrouter", "gemini"]
    assert full[2][1] == "openai/gpt-oss-120b:free"


def test_invalid_json_is_a_bad_answer_not_an_outage(tmp_path):
    # the openai SDK unwraps {"error": {...}} into body; accept both shapes
    for body in ({"code": "json_validate_failed", "failed_generation": "{oops"},
                 {"error": {"code": "json_validate_failed", "failed_generation": "{oops"}}):
        err = StatusError(400, "Error code: 400 - json_validate_failed", body)
        sdk = FakeSDK({"big": err, "small": "{}"})
        reply = make(["big", "small"], sdk, tmp_path).chat_json(MSG)
        assert reply.text == "{oops" and reply.provider == "p1" and sdk.calls == ["big"]  # same provider
        assert not any(tmp_path.iterdir())  # never cached
    # the translator then asks again with the problem
    res = translate("Oil jumps", FakeClient([reply.text, GOOD]), LIMITS)
    assert res.ok and "not valid JSON" in res.errors[0][0]


def test_demo_cache_is_read_first_and_never_written(tmp_path):
    from riskdesk.llm.client import save_reply

    demo, dev = tmp_path / "demo", tmp_path / "dev"
    msgs = [{"role": "user", "content": "q"}]
    save_reply(demo, msgs, LLMReply('{"from": "demo"}', "big", 1, 1, 0.0, False))
    sdk = FakeSDK({"big": '{"from": "live"}'})
    client = LLMClient(providers("big"), cache_dir=dev, demo_dir=demo, sdk=sdk)
    reply = client.chat_json(msgs)
    assert reply.cached and reply.text == '{"from": "demo"}' and sdk.calls == []
    client.chat_json([{"role": "user", "content": "other"}])  # a new question: live, saved to dev only
    assert len(list(demo.iterdir())) == 1 and len(list(dev.iterdir())) == 1
