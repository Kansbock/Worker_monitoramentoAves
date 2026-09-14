import pytest
from pydantic import ValidationError

from feeder_worker.config import Settings


REQUIRED_ENV = {
    "AWS_PROFILE": "aves-worker",
    "S3_BUCKET": "comedouro-aves-fotos-tcc-2026",
    "API_BASE_URL": "https://api.example.com",
    "API_USER_NAME": "Worker",
    "API_USER_EMAIL": "worker@example.com",
    "API_USER_PASSWORD": "secret",
    "MODEL_PATH": "/tmp/model.keras",
    "STATE_DB_PATH": "/tmp/state.db",
}


def _set_env(monkeypatch, overrides=None):
    env = {**REQUIRED_ENV, **(overrides or {})}
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_settings_load_with_all_required_env_vars(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.aws_profile == "aves-worker"
    assert settings.s3_bucket == "comedouro-aves-fotos-tcc-2026"
    assert settings.api_base_url == "https://api.example.com"


def test_settings_apply_defaults_for_optional_fields(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.aws_region == "us-east-1"
    assert settings.poll_interval_seconds == 60
    assert settings.max_images_per_cycle == 10
    assert settings.confidence_threshold == 0.85
    assert settings.capture_timezone == "America/Sao_Paulo"


def test_settings_missing_required_field_raises(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.delenv("S3_BUCKET")

    with pytest.raises(ValidationError, match="S3_BUCKET"):
        Settings(_env_file=None)


def test_settings_strips_trailing_slash_from_api_base_url(monkeypatch):
    _set_env(monkeypatch, {"API_BASE_URL": "https://api.example.com/"})

    settings = Settings(_env_file=None)

    assert settings.api_base_url == "https://api.example.com"


def test_settings_rejects_invalid_timezone(monkeypatch):
    _set_env(monkeypatch, {"CAPTURE_TIMEZONE": "Not/AZone"})

    with pytest.raises(ValidationError, match="Fuso hor.rio inv.lido"):
        Settings(_env_file=None)


def test_settings_zone_info_matches_configured_timezone(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert str(settings.zone_info()) == "America/Sao_Paulo"


@pytest.mark.parametrize("value", ["0", "-0.1", "1.5"])
def test_settings_rejects_confidence_threshold_out_of_range(monkeypatch, value):
    _set_env(monkeypatch, {"CONFIDENCE_THRESHOLD": value})

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_rejects_non_positive_poll_interval(monkeypatch):
    _set_env(monkeypatch, {"POLL_INTERVAL_SECONDS": "0"})

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
