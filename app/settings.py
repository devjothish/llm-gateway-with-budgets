"""Typed config from the environment. Fails loudly at import if malformed.

Split from `config.py` on a security line, not an arbitrary one: secrets live
here and come from the environment, policy lives there and comes from a
committed YAML file. Nothing that belongs in one can accidentally be written to
the other.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider credentials. Absent keys mean that provider is simply not
    # registered; see providers.build_providers.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Ollama needs no credential, so setting this URL is what enables it.
    # 127.0.0.1 rather than localhost, deliberately: on macOS `localhost`
    # resolves to ::1 first, and a machine running both Ollama.app and an
    # Ollama container reaches a different one over IPv6 than over IPv4, with
    # a different set of models. The failure reads as "model not found".
    ollama_base_url: str | None = "http://127.0.0.1:11434/v1"

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://gateway:gateway@localhost:5433/gateway"
    config_path: Path = Path("gateway.yaml")

    service_name: str = "llm-gateway"
    environment: str = "dev"
    logfire_token: str | None = None

    # Admission needs a token estimate before the response exists. Output is the
    # unknown half, so this is what gets reserved and later reconciled against
    # the real count.
    # ponytail: a flat guess. Replace with a per-team rolling median once there
    # is traffic to compute one from; the reconcile path already handles error.
    estimated_output_tokens: int = 500


settings = Settings()
