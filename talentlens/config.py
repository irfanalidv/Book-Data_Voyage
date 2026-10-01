"""Application settings loaded from environment and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from talentlens.paths import REPO_ROOT


class Settings(BaseSettings):
    """TalentLens configuration - validated at import time, not at first HTTP request."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    data_dir: Path = REPO_ROOT / "data"
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    def mask_secrets(self) -> dict[str, str]:
        """Return settings safe to print (API keys redacted)."""
        data = self.model_dump()
        for key in ("openai_api_key", "groq_api_key"):
            value = data.get(key)
            if value is not None:
                data[key] = "***set***"
        data["data_dir"] = str(data["data_dir"])
        return {k: str(v) for k, v in data.items()}


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - one parse per process."""
    return Settings()
