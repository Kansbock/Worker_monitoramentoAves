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
