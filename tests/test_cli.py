from types import SimpleNamespace
from zoneinfo import ZoneInfo

from feeder_worker import cli
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
