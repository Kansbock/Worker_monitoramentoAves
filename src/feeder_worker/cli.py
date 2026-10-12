"""Interface de linha de comando do worker (`feeder-worker`)."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import httpx
from types import FrameType

from .api_client import ApiError, AvesApiClient
from .catalog import load_default_catalog
from .classifier import BirdClassifier
from .config import Settings, load_settings
from .processor import (
    ProcessorDependencies,
    SpeciesRegistrationError,
    resolve_species_ids,
    run_cycle,
)
from .s3_gateway import S3Gateway
from .state import StateStore

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_processor_dependencies(settings: Settings, dry_run: bool) -> ProcessorDependencies:
    catalog = load_default_catalog()

    s3_gateway = S3Gateway(
        bucket=settings.s3_bucket,
        region=settings.aws_region,
        profile=settings.aws_profile,
        excluded_prefixes=catalog.destination_prefixes(),
    )
    api_client = AvesApiClient(
        base_url=settings.api_base_url,
        email=settings.api_user_email,
        password=settings.api_user_password,
    )
    species_ids = resolve_species_ids(catalog, api_client)
    classifier = BirdClassifier(
        model_path=settings.model_path,
        num_classes=catalog.num_classes,
        species_class_names=[entry.model_label for entry in catalog.entries],
    )
    state_store = StateStore(settings.state_db_path)

    return ProcessorDependencies(
        s3_gateway=s3_gateway,
        api_client=api_client,
        classifier=classifier,
        state_store=state_store,
        catalog=catalog,
        species_ids_by_scientific_name=species_ids,
        capture_timezone=settings.zone_info(),
        confidence_threshold=settings.confidence_threshold,
        dry_run=dry_run,
    )


def _run(args: argparse.Namespace) -> int:
    settings = load_settings()

    try:
        deps = _build_processor_dependencies(settings, dry_run=args.dry_run)
    except SpeciesRegistrationError as exc:
        logger.error(str(exc))
        return 1

    stop_requested = False

    def _handle_signal(signum: int, frame: FrameType | None) -> None:
        nonlocal stop_requested
        logger.info("Sinal %s recebido, encerrando após o ciclo atual", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_requested:
            _process_notifications(deps, settings.notification_processing_enabled)
            examined = run_cycle(deps, settings.max_images_per_cycle)
            logger.info("Ciclo concluído: %d foto(s) examinada(s)", examined)
            _process_notifications(deps, settings.notification_processing_enabled)
            if args.once:
                break
            for _ in range(settings.poll_interval_seconds):
                if stop_requested:
                    break
                time.sleep(1)
    finally:
        deps.api_client.close()
        deps.state_store.close()

    return 0


def _process_notifications(deps: ProcessorDependencies, enabled: bool) -> None:
    if not enabled or deps.dry_run:
        return
    try:
        batch = deps.api_client.process_notifications()
    except (ApiError, httpx.HTTPError) as exc:
        logger.error("Executor de notificações indisponível: category=%s status=%s",
                     type(exc).__name__, getattr(exc, "status_code", "network"))
        return
    logger.info("Notificações: concluídas=%d adiadas=%d falhas=%d",
                batch.completed, batch.deferred, batch.failed)
    if batch.failed:
        logger.warning("Há trabalhos de notificação com falha persistida; revisar a causa na API")


def _bootstrap_service_user(args: argparse.Namespace) -> int:
    settings = load_settings()
    with AvesApiClient(
        base_url=settings.api_base_url,
        email=settings.api_user_email,
        password=settings.api_user_password,
    ) as client:
        created = client.register_user(
            name=settings.api_user_name,
            email=settings.api_user_email,
            password=settings.api_user_password,
        )
        if created:
            logger.info("Conta de serviço criada: %s", settings.api_user_email)
        else:
            logger.info("Conta de serviço já existia, mantendo: %s", settings.api_user_email)

        # Valida que o login funciona com as credenciais configuradas, sem
        # persistir senha ou token em nenhum arquivo.
        client.authenticate()
        logger.info("Login validado com sucesso para %s", settings.api_user_email)
    return 0


def _seed_species(args: argparse.Namespace) -> int:
    settings = load_settings()
    catalog = load_default_catalog()

    with AvesApiClient(
        base_url=settings.api_base_url,
        email=settings.api_user_email,
        password=settings.api_user_password,
    ) as client:
        existing = {record.scientific_name: record for record in client.list_species()}

        for entry in catalog.entries:
            if entry.scientific_name in existing:
                logger.info("Espécie já cadastrada, mantendo: %s", entry.scientific_name)
                continue
            created = client.create_species(
                name=entry.common_name,
                scientific_name=entry.scientific_name,
                description=entry.description,
                tips=entry.tips,
                image_url=entry.image_url,
            )
            logger.info("Espécie criada: %s (id=%s)", created.scientific_name, created.id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="feeder-worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Executa o loop contínuo de classificação")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classifica e registra a ação planejada sem chamar a API nem alterar o S3",
    )
    run_parser.add_argument(
        "--once", action="store_true", help="Executa um único ciclo e encerra (útil para testes)"
    )
    run_parser.set_defaults(func=_run)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap-service-user", help="Cria (se necessário) a conta de serviço do worker"
    )
    bootstrap_parser.set_defaults(func=_bootstrap_service_user)

    seed_parser = subparsers.add_parser(
        "seed-species", help="Cria (se necessário) o catálogo inicial de espécies na API"
    )
    seed_parser.set_defaults(func=_seed_species)

    return parser


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    exit_code = 1
    try:
        exit_code = args.func(args)
    except ApiError as exc:
        logger.error("Erro da API Aves: %s", exc.message)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
