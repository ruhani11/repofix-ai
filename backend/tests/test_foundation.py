"""Tests for the backend health endpoint and configuration."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app


@pytest.fixture(autouse=True)
def clear_config_environment(monkeypatch):
    """Keep each test independent of existing environment variables."""
    monkeypatch.delenv("REPOFIX_APP_NAME", raising=False)
    monkeypatch.delenv("REPOFIX_ENVIRONMENT", raising=False)


def test_health_response():
    settings = Settings(_env_file=None, environment="test")

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "RepoFix AI",
        "version": "0.1.0",
    }


def test_environment_variable_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "REPOFIX_APP_NAME=From file\n"
        "REPOFIX_ENVIRONMENT=development\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("REPOFIX_APP_NAME", "From environment")

    settings = Settings(_env_file=env_file)

    assert settings.app_name == "From environment"
    assert settings.environment == "development"


def test_invalid_environment_is_rejected():
    with pytest.raises(ValidationError, match="environment"):
        Settings(_env_file=None, environment="unsupported")


def test_blank_application_name_is_rejected():
    with pytest.raises(ValidationError, match="app_name"):
        Settings(_env_file=None, app_name="   ")


def test_unknown_dotenv_setting_is_rejected(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "REPOFIX_ENVIROMENT=production\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Settings(_env_file=env_file)