"""Offline OAuth settings tests using dummy credentials only."""

import pytest
from pydantic import ValidationError

from app.config import Settings


DUMMY_CLIENT_ID = "test-client-id"
DUMMY_SECRET = "test-secret-never-use-for-login"
CALLBACK_PATH = "/api/auth/github/callback"


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch):
    for name in (
        "REPOFIX_APP_NAME",
        "REPOFIX_ENVIRONMENT",
        "REPOFIX_GITHUB_CLIENT_ID",
        "REPOFIX_GITHUB_CLIENT_SECRET",
        "REPOFIX_GITHUB_REDIRECT_URI",
    ):
        monkeypatch.delenv(name, raising=False)


def configured_settings(**overrides):
    values = {
        "github_client_id": DUMMY_CLIENT_ID,
        "github_client_secret": DUMMY_SECRET,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_oauth_is_optional():
    settings = Settings(_env_file=None)

    assert settings.github_auth_configured is False
    assert settings.github_client_id is None
    assert settings.github_client_secret is None
    assert settings.github_redirect_uri == f"http://127.0.0.1:8000{CALLBACK_PATH}"


def test_complete_credentials_enable_configuration():
    settings = configured_settings()

    assert settings.github_auth_configured is True
    assert settings.github_client_id == DUMMY_CLIENT_ID
    assert settings.github_client_secret.get_secret_value() == DUMMY_SECRET


def test_secret_is_excluded_from_normal_output():
    settings = configured_settings()

    assert "github_client_secret" not in settings.model_dump()

    for output in (settings.model_dump_json(), repr(settings), str(settings)):
        assert DUMMY_SECRET not in output


@pytest.mark.parametrize("missing", ["github_client_id", "github_client_secret"])
def test_incomplete_credentials_are_rejected(missing):
    with pytest.raises(ValidationError, match="Set both GitHub client ID") as error:
        configured_settings(**{missing: None})

    assert DUMMY_SECRET not in str(error.value)


@pytest.mark.parametrize("field", ["github_client_id", "github_client_secret"])
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_credentials_are_rejected(field, value):
    with pytest.raises(ValidationError):
        configured_settings(**{field: value})


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://[::1]:8000",
        "https://example.com",
    ],
)
def test_valid_callback_is_accepted(origin):
    callback = origin + CALLBACK_PATH

    assert configured_settings(github_redirect_uri=callback).github_redirect_uri == callback


@pytest.mark.parametrize(
    "callback",
    [
        f"http://example.com{CALLBACK_PATH}",
        f"https://user:password@example.com{CALLBACK_PATH}",
        f"https://example.com{CALLBACK_PATH}?next=other",
        f"https://example.com{CALLBACK_PATH}#fragment",
        "https://example.com/wrong-path",
        CALLBACK_PATH,
        f"http://127.0.0.1:invalid{CALLBACK_PATH}",
        f"http://127.0.0.1:0{CALLBACK_PATH}",
        f"https://exam ple.com{CALLBACK_PATH}",
        f"https://example.com/api/auth/github/\ncallback",
    ],
)
def test_invalid_callback_is_rejected(callback):
    with pytest.raises(ValidationError):
        configured_settings(github_redirect_uri=callback)


def test_enabled_production_oauth_requires_https():
    with pytest.raises(ValidationError, match="requires an HTTPS callback") as error:
        configured_settings(environment="production")

    assert DUMMY_SECRET not in str(error.value)


def test_production_https_callback_is_accepted():
    settings = configured_settings(
        environment="production",
        github_redirect_uri=f"https://example.com{CALLBACK_PATH}",
    )

    assert settings.github_auth_configured is True


def test_environment_overrides_temporary_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / "oauth-test.env"
    dotenv.write_text(
        f"REPOFIX_GITHUB_CLIENT_ID={DUMMY_CLIENT_ID}\n"
        f"REPOFIX_GITHUB_CLIENT_SECRET={DUMMY_SECRET}\n"
        f"REPOFIX_GITHUB_REDIRECT_URI=http://localhost:8000{CALLBACK_PATH}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REPOFIX_GITHUB_CLIENT_ID", "environment-test-id")

    settings = Settings(_env_file=dotenv)

    assert settings.github_auth_configured is True
    assert settings.github_client_id == "environment-test-id"
    assert settings.github_client_secret.get_secret_value() == DUMMY_SECRET
    assert settings.github_redirect_uri == f"http://localhost:8000{CALLBACK_PATH}"