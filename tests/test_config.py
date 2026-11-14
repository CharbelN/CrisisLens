from riskdesk.config import load_settings, parse_bool


def test_gemini_missing_is_not_an_error():
    s = load_settings({"GROQ_API_KEY": "abc"})
    assert s.has_groq
    assert s.gemini_api_key is None
    assert not s.has_gemini


def test_blank_keys_count_as_missing():
    s = load_settings({"GROQ_API_KEY": "  ", "GEMINI_API_KEY": ""})
    assert not s.has_groq
    assert not s.has_gemini


def test_share_off_by_default():
    assert load_settings({}).share_enabled is False


def test_share_flag_parsing():
    for on in ["true", "TRUE", "1", "yes", "on", " True "]:
        assert parse_bool(on) is True
    for off in [None, "", "false", "0", "no", "maybe"]:
        assert parse_bool(off) is False
