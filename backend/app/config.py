"""Validated application settings; environment values override backend/.env."""

from functools import lru_cache
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REPOFIX_",
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )

    app_name: str = Field(default="RepoFix AI", min_length=1, max_length=80)
    environment: Literal["development", "test", "production"] = "development"
    enable_local_validation: bool = False

    github_client_id: str | None = Field(default=None, min_length=1, max_length=100)
    github_client_secret: SecretStr | None = Field(
        default=None, exclude=True, repr=False
    )
    github_inspector_token: SecretStr | None = Field(default=None, exclude=True, repr=False)
    github_draft_pr_token: SecretStr | None = Field(default=None, exclude=True, repr=False)
    github_redirect_uri: str = Field(
        default="http://127.0.0.1:8000/api/auth/github/callback",
        min_length=1,
        max_length=2048,
        validate_default=True,
    )

    openai_api_key: SecretStr | None = Field(
        default=None, exclude=True, repr=False
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, exclude=True, repr=False
    )
    ai_provider: Literal["openai", "gemini"] = "openai"
    ai_model: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    ai_max_turns: int = Field(default=10, ge=1, le=20)
    ai_timeout_seconds: int = Field(default=90, ge=5, le=300)
    history_database_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "data" / "investigations.db",
        validate_default=True,
    )
    proposal_database_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "data" / "proposals.db",
        validate_default=True,
    )

    @field_validator("history_database_path", "proposal_database_path", mode="before")
    @classmethod
    def validate_history_database_path(cls, value: object) -> object:
        if not isinstance(value, (str, Path)):
            return value
        raw = str(value)
        if not raw.strip() or raw == ":memory:" or any(ord(c) < 32 for c in raw):
            raise ValueError("Use a file path for investigation history.")
        path = Path(raw).expanduser()
        # Relative settings are stable even when the server's cwd changes.
        return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path

    @model_validator(mode="after")
    def separate_history_databases(self) -> "Settings":
        if self.history_database_path.resolve() == self.proposal_database_path.resolve():
            raise ValueError("Use separate database files for investigations and proposals.")
        return self

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            secret = value.get_secret_value()
            if not secret or len(secret) > 4096 or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in secret
            ):
                raise ValueError("OpenAI API key must be nonblank and contain no whitespace or controls.")
        return value

    @field_validator("github_inspector_token", "github_draft_pr_token")
    @classmethod
    def validate_inspector_token(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not re.fullmatch(
            r"github_pat_[A-Za-z0-9_]{1,256}", value.get_secret_value()
        ):
            raise ValueError("Set a fine-grained GitHub personal access token.")
        return value

    @field_validator("github_client_secret")
    @classmethod
    def validate_client_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("GitHub client secret must not be blank.")
        return value

    @field_validator("gemini_api_key")
    @classmethod
    def validate_gemini_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            secret = value.get_secret_value()
            if not secret or len(secret) > 4096 or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in secret
            ):
                raise ValueError(
                    "Gemini API key must be nonblank and contain no whitespace or controls."
                )
        return value

    @field_validator("github_redirect_uri")
    @classmethod
    def validate_redirect_uri(cls, value: str) -> str:
        if any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            raise ValueError("GitHub callback URI must not contain whitespace or controls.")
        try:
            parsed = urlsplit(value)
            port = parsed.port
            valid = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and parsed.path == "/api/auth/github/callback"
                and (port is None or port > 0)
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Use an absolute URL ending in /api/auth/github/callback.")
        if parsed.scheme == "http" and parsed.hostname not in {
            "127.0.0.1", "localhost", "::1"
        }:
            raise ValueError("HTTP callbacks are allowed only on loopback hosts.")
        return value

    @model_validator(mode="after")
    def validate_github_configuration(self) -> "Settings":
        if (self.github_client_id is None) != (self.github_client_secret is None):
            raise ValueError("Set both GitHub client ID and client secret, or neither.")
        if (
            self.github_auth_configured
            and self.environment == "production"
            and urlsplit(self.github_redirect_uri).scheme != "https"
        ):
            raise ValueError("Production GitHub authentication requires an HTTPS callback.")
        return self

    @model_validator(mode="after")
    def validate_ai_provider_model(self) -> "Settings":
        if self.ai_model is not None:
            is_gemini_model = self.ai_model.startswith("gemini-")
            if is_gemini_model != (self.ai_provider == "gemini"):
                raise ValueError("The selected AI model must match its provider.")
        return self

    @property
    def github_auth_configured(self) -> bool:
        return self.github_client_id is not None and self.github_client_secret is not None

    @property
    def ai_configured(self) -> bool:
        """Selected-provider configuration only; does not verify API access."""
        key = (
            self.gemini_api_key
            if self.ai_provider == "gemini"
            else self.openai_api_key
        )
        return key is not None and self.ai_model is not None

    @property
    def gemini_key_configured(self) -> bool:
        """Key presence only; does not verify credentials, quota or model access."""
        return self.gemini_api_key is not None


@lru_cache
def get_settings() -> Settings:
    """Load once per process. Restart the server after changing settings."""
    return Settings()
