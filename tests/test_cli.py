import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx

from feeder_worker import cli
from feeder_worker.api_client import AvesApiClient
from feeder_worker.catalog import load_default_catalog
from feeder_worker.cli import build_parser


def test_parser_requires_a_subcommand():
    parser = build_parser()

    assert parser.parse_args(["run"]).command == "run"


def test_run_subcommand_defaults_and_flags():
    parser = build_parser()

    args = parser.parse_args(["run"])
    assert args.dry_run is False
    assert args.once is False
    assert args.func.__name__ == "_run"

    args = parser.parse_args(["run", "--dry-run", "--once"])
    assert args.dry_run is True
    assert args.once is True


def test_bootstrap_service_user_subcommand_is_registered():
    parser = build_parser()

    args = parser.parse_args(["bootstrap-service-user"])

    assert args.func.__name__ == "_bootstrap_service_user"


def test_seed_species_subcommand_is_registered():
    parser = build_parser()

    args = parser.parse_args(["seed-species"])

    assert args.func.__name__ == "_seed_species"


def test_parser_errors_on_unknown_subcommand():
    parser = build_parser()

    try:
        parser.parse_args(["unknown-command"])
        assert False, "esperava SystemExit para subcomando desconhecido"
    except SystemExit as exc:
        assert exc.code == 2


def test_processor_passes_species_labels_to_classifier(monkeypatch, tmp_path):
    captured = {}

    def fake_classifier(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "BirdClassifier", fake_classifier)
    monkeypatch.setattr(cli, "S3Gateway", lambda **kwargs: object())
    monkeypatch.setattr(cli, "AvesApiClient", lambda **kwargs: object())
    monkeypatch.setattr(cli, "resolve_species_ids", lambda catalog, client: {})
    monkeypatch.setattr(cli, "StateStore", lambda path: object())
    settings = SimpleNamespace(
        s3_bucket="test-bucket", aws_region="us-east-1", aws_profile="test-profile",
        api_base_url="https://example.com", api_user_email="test@example.com",
        api_user_password="test-password", model_path=tmp_path / "model.keras",
        state_db_path=tmp_path / "state.db", confidence_threshold=0.85,
        zone_info=lambda: ZoneInfo("UTC"),
    )

    deps = cli._build_processor_dependencies(settings, dry_run=True)

    assert deps.classifier is not None
    assert captured == {
        "model_path": settings.model_path,
        "num_classes": 3,
        "species_class_names": ["Avoante", "Bem-Te-Vi", "Sabia-Laranjeira"],
    }


def test_seed_registers_unidentified_category_once_without_registering_background(monkeypatch):
    catalog = load_default_catalog()
    registered = [
        {"id": f"sp-{entry.class_index}", "name": entry.common_name,
         "scientificName": entry.scientific_name}
        for entry in catalog.entries
    ]
    created = []

    def handler(request):
        if request.method == "GET" and request.url.path == "/api/species":
            return httpx.Response(200, json=list(registered))
        if request.method == "POST" and request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"token": "synthetic-session-token"})
        if request.method == "POST" and request.url.path == "/api/species":
            body = json.loads(request.content)
            created.append(body)
            record = {**body, "id": "sp-outros"}
            registered.append(record)
            return httpx.Response(201, json=record)
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    settings = SimpleNamespace(
        api_base_url="https://example.com", api_user_email="test@example.com",
        api_user_password="synthetic-password",
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(
        cli, "AvesApiClient",
        lambda **kwargs: AvesApiClient(**kwargs, transport=httpx.MockTransport(handler)),
    )

    assert cli._seed_species(SimpleNamespace()) == 0
    assert cli._seed_species(SimpleNamespace()) == 0

    assert len(created) == 1
    assert created[0]["name"] == "Ave não identificada"
    assert created[0]["scientificName"] == "Espécie não identificada"
    assert len(registered) == 4
    assert all(record["name"] != "Background" for record in registered)
