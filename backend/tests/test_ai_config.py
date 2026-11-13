"""Offline AI settings checks using dummy credentials; no model calls."""

import os

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


DUMMY_KEY = "test-only-api-key-never-use"
DUMMY_MODEL = "test-model"


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("REPOFIX_"):
            monkeypatch.delenv(name, raising=False)


def test_ai_configuration_is_optional():
    settings = Settings(_env_file=None)

    assert settings.ai_configured is False
    assert settings.openai_api_key is None
    assert settings.ai_model is None
    assert settings.ai_max_turns == 10
    assert settings.ai_timeout_seconds == 90


@pytest.mark.parametrize(
    "values",
    [{"openai_api_key": DUMMY_KEY}, {"ai_model": DUMMY_MODEL}],
)
def test_partial_configuration_keeps_ai_unconfigured(values):
    settings = Settings(_env_file=None, **values)

    assert settings.ai_configured is False


def test_key_and_model_mark_ai_as_configured():
    settings = Settings(
        _env_file=None,
        openai_api_key=DUMMY_KEY,
        ai_model=DUMMY_MODEL,
    )

    assert settings.ai_configured is True
    assert isinstance(settings.openai_api_key, SecretStr)
    assert settings.openai_api_key.get_secret_value() == DUMMY_KEY
    assert settings.ai_model == DUMMY_MODEL


def test_api_key_is_excluded_from_normal_output():
    settings = Settings(_env_file=None, openai_api_key=DUMMY_KEY)

    assert "openai_api_key" not in settings.model_dump()
    for output in (settings.model_dump_json(), repr(settings), str(settings)):
        assert DUMMY_KEY not in output


@pytest.mark.parametrize(
    "value",
    ["", "   ", "test key", "test\nkey", "test\x00key", "test\x7fkey", "x" * 4097],
)
def test_invalid_api_keys_are_rejected_without_echoing_input(value):
    with pytest.raises(ValidationError, match="OpenAI API key") as error:
        Settings(_env_file=None, openai_api_key=value)

    message = str(error.value)
    assert "input_value=" not in message
    if value.strip():
        assert value not in message


@pytest.mark.parametrize(
    "value",
    ["", "   ", "https://example.com/model", "two models", "x" * 101],
)
def test_invalid_model_identifiers_are_rejected(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_model=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ai_max_turns", 0),
        ("ai_max_turns", 21),
        ("ai_timeout_seconds", 4),
        ("ai_timeout_seconds", 301),
    ],
)
def test_out_of_range_limits_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("turns,seconds", [(1, 5), (20, 300)])
def test_limit_boundaries_are_accepted(turns, seconds):
    settings = Settings(
        _env_file=None,
        ai_max_turns=turns,
        ai_timeout_seconds=seconds,
    )

    assert settings.ai_max_turns == turns
    assert settings.ai_timeout_seconds == seconds


def test_environment_overrides_temporary_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / "ai-test.env"
    dotenv.write_text(
        f"REPOFIX_OPENAI_API_KEY={DUMMY_KEY}\n"
        f"REPOFIX_AI_MODEL={DUMMY_MODEL}\n"
        "REPOFIX_AI_MAX_TURNS=7\n"
        "REPOFIX_AI_TIMEOUT_SECONDS=45\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REPOFIX_AI_MODEL", "environment-test-model")
    monkeypatch.setenv("REPOFIX_AI_MAX_TURNS", "8")

    settings = Settings(_env_file=dotenv)

    assert settings.ai_configured is True
    assert settings.openai_api_key.get_secret_value() == DUMMY_KEY
    assert settings.ai_model == "environment-test-model"
    assert settings.ai_max_turns == 8
    assert settings.ai_timeout_seconds == 45


def test_prefixed_environment_variables_configure_ai(monkeypatch):
    monkeypatch.setenv("REPOFIX_OPENAI_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("REPOFIX_AI_MODEL", DUMMY_MODEL)

    settings = Settings(_env_file=None)

    assert settings.ai_configured is True
    assert settings.openai_api_key.get_secret_value() == DUMMY_KEY


def test_github_configuration_works_without_ai_credentials():
    settings = Settings(
        _env_file=None,
        github_client_id="test-client-id",
        github_client_secret="test-github-secret",
    )

    assert settings.github_auth_configured is True
    assert settings.ai_configured is False
