from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["groq", "ollama", "huggingface"]


class Settings(BaseSettings):
    """Settings for the Paper Pilot application."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    llm_provider: Provider = "ollama"
    llm_model: str | None = None
    llm_fallback_model: str | None = None
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 60
    model_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    hugging_face_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Get the settings for the Paper Pilot application."""
    return Settings()
