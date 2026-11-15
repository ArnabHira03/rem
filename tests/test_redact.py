import pytest

from rem.redact import redact


@pytest.mark.parametrize("secret,kind", [
    ("sk-ant-api03-" + "a" * 40, "anthropic_key"),
    ("sk-proj-" + "b" * 40, "openai_key"),
    ("ghp_" + "c" * 36, "github_token"),
    ("AKIA" + "D" * 16, "aws_access_key"),
    ("eyJhbGciOi.eyJzdWIiOi.c2lnbmF0dXJl", "jwt"),
])
def test_known_token_shapes(secret, kind):
    assert redact(f"use {secret} please") == f"use [REDACTED:{kind}] please"


def test_env_assignment_keeps_name():
    assert redact("export OPENAI_API_KEY=abc123\nDEBUG=1") == "export OPENAI_API_KEY=[REDACTED:secret_assignment]\nDEBUG=1"


def test_private_key_block():
    pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\ndef\n-----END OPENSSH PRIVATE KEY-----"
    assert redact(f"key:\n{pem}\nend") == "key:\n[REDACTED:private_key]\nend"


def test_ordinary_text_untouched():
    s = "I prefer uv over pip; my laptop has 15 GB RAM and the token count is 400."
    assert redact(s) == s


def test_bearer_token():
    assert redact("Authorization: Bearer abcdef0123456789xyz") == "Authorization: Bearer [REDACTED:bearer_token]"


def test_json_style_key_value():
    assert redact('{"api_key": "abcd1234efgh"}') == '{"api_key": "[REDACTED:secret_value]"}'


def test_inline_password_assignment():
    assert redact("connect with user=bob password=hunter22 now") == "connect with user=bob password=[REDACTED:secret_value] now"
