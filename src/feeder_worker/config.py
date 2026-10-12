"""Configuração tipada do worker, carregada de variáveis de ambiente / .env.

Todas as configurações obrigatórias são validadas na inicialização: a
aplicação deve falhar cedo e claramente se algo essencial estiver ausente ou
inválido, em vez de falhar de forma obscura no meio do processamento.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        protected_namespaces=(),
    )

    # AWS
    aws_profile: str | None = Field(default=None, alias="AWS_PROFILE")
    aws_region: str = Field("us-east-1", alias="AWS_REGION")
    s3_bucket: str = Field(..., alias="S3_BUCKET")

    # API Aves
    api_base_url: str = Field(..., alias="API_BASE_URL")
    api_user_name: str = Field(..., alias="API_USER_NAME")
    api_user_email: str = Field(..., alias="API_USER_EMAIL")
    api_user_password: str = Field(..., alias="API_USER_PASSWORD")

    # Modelo e estado
    model_path: Path = Field(..., alias="MODEL_PATH")
    state_db_path: Path = Field(..., alias="STATE_DB_PATH")

    # Parâmetros operacionais
    poll_interval_seconds: int = Field(60, alias="POLL_INTERVAL_SECONDS", gt=0)
    notification_processing_enabled: bool = Field(False, alias="NOTIFICATION_PROCESSING_ENABLED")
    max_images_per_cycle: int = Field(10, alias="MAX_IMAGES_PER_CYCLE", gt=0)
    confidence_threshold: float = Field(0.85, alias="CONFIDENCE_THRESHOLD", gt=0, le=1)
    capture_timezone: str = Field("America/Sao_Paulo", alias="CAPTURE_TIMEZONE")

    @field_validator("api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("capture_timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"Fuso horário inválido em CAPTURE_TIMEZONE: {value!r}. "
                "No Windows, instale o pacote 'tzdata' (já declarado como dependência)."
            ) from exc
        return value

    def zone_info(self) -> ZoneInfo:
        return ZoneInfo(self.capture_timezone)


def load_settings() -> Settings:
    """Carrega e valida as configurações. Lança ValidationError com mensagem
    clara sobre qual variável está ausente/inválida caso algo esteja errado."""
    return Settings()  # type: ignore[call-arg]
