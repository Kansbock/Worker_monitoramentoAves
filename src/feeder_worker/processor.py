"""Orquestra o processamento de cada foto: classificar, registrar o
avistamento ou rejeitar, mover e permitir retomada segura em caso de falha.

Cada foto é tratada isoladamente: uma falha em uma imagem nunca interrompe o
processamento das demais no mesmo ciclo. O estado local garante que, ao
retomar após uma falha entre a criação do avistamento e o movimento físico
no S3, o worker não repita a criação (idempotência) e apenas conclua o
movimento pendente.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from .api_client import AvesApiClient
from .catalog import SpeciesCatalog
from .classifier import OTHER_CLASS_LABEL, BirdClassifier, InvalidImageError
from .s3_gateway import S3Gateway, S3Object
from .state import (
    PHASE_COMPLETED,
    PHASE_FAILED_RETRY,
    PHASE_INVALID,
    PHASE_REJECTED,
    PHASE_SIGHTING_CREATED,
    PhotoState,
    StateStore,
    compute_idempotency_key,
)
from .timestamps import extract_capture_timestamp

logger = logging.getLogger(__name__)

_MIN_BACKOFF_SECONDS = 30
_MAX_BACKOFF_SECONDS = 3600


class SpeciesRegistrationError(RuntimeError):
    """Levantado quando o catálogo local referencia uma espécie ausente na
    API. O daemon não deve iniciar o loop nesse estado (execute
    'feeder-worker seed-species' antes)."""


def resolve_species_ids(catalog: SpeciesCatalog, api_client: AvesApiClient) -> dict[str, str]:
    """Mapeia nome científico -> id retornado pela API."""
    registered = {record.scientific_name: record.id for record in api_client.list_species()}
    missing = [
        entry.scientific_name for entry in catalog.entries if entry.scientific_name not in registered
    ]
    if missing:
        raise SpeciesRegistrationError(
            "Espécies do catálogo local ausentes na API (execute "
            "'feeder-worker seed-species' antes de iniciar o worker): " + ", ".join(missing)
        )
    return {entry.scientific_name: registered[entry.scientific_name] for entry in catalog.entries}


def _compute_backoff_seconds(attempt_count: int) -> float:
    return min(_MIN_BACKOFF_SECONDS * (2 ** max(attempt_count - 1, 0)), _MAX_BACKOFF_SECONDS)


@dataclass
class ProcessorDependencies:
    s3_gateway: S3Gateway
    api_client: AvesApiClient
    classifier: BirdClassifier
    state_store: StateStore
    catalog: SpeciesCatalog
    species_ids_by_scientific_name: dict[str, str]
    capture_timezone: ZoneInfo
    confidence_threshold: float
    dry_run: bool = False


def process_photo(deps: ProcessorDependencies, obj: S3Object) -> None:
    """Processa uma única foto até um estado terminal ou até uma falha
    isolada e recuperável (registrada para nova tentativa)."""
    idempotency_key = compute_idempotency_key(deps.s3_gateway.bucket, obj.key, obj.etag)
    state = deps.state_store.get_or_create(obj.key, obj.etag, idempotency_key)

    if state.is_terminal():
        logger.debug("Foto já finalizada, ignorando: %s (fase=%s)", obj.key, state.phase)
        return

    if not state.is_ready_for_retry(time.time()):
        logger.debug("Foto aguardando novo horário de retentativa: %s", obj.key)
        return

    try:
        _process_photo_inner(deps, obj, state)
    except Exception as exc:  # noqa: BLE001 - isola a falha nesta foto; o ciclo continua
        logger.exception("Falha ao processar %s, será retentado", obj.key)
        state.attempt_count += 1
        state.phase = PHASE_FAILED_RETRY
        state.last_error = str(exc)
        state.next_retry_at = time.time() + _compute_backoff_seconds(state.attempt_count)
        if not deps.dry_run:
            deps.state_store.save(state)


def _process_photo_inner(deps: ProcessorDependencies, obj: S3Object, state: PhotoState) -> None:
    # Retomada: o avistamento já foi criado numa tentativa anterior — não
    # reclassificar nem chamar a API de novo, apenas concluir o movimento.
    if state.phase == PHASE_SIGHTING_CREATED:
        _finish_move(deps, obj, state)
        return

    image_bytes = deps.s3_gateway.download(obj.key)

    try:
        result = deps.classifier.classify(image_bytes)
    except InvalidImageError as exc:
        logger.warning("Arquivo inválido em %s: %s", obj.key, exc)
        if not deps.dry_run:
            deps.s3_gateway.move_to_errors(obj.key)
        state.phase = PHASE_INVALID
        state.last_error = str(exc)
        if not deps.dry_run:
            deps.state_store.save(state)
        return

    if result.class_index == deps.catalog.num_classes:
        state.model_label = OTHER_CLASS_LABEL
        state.confidence = result.confidence
        logger.info(
            "Predição %s (confiança %.4f) em %s; movendo para background/ sem criar avistamento",
            OTHER_CLASS_LABEL,
            result.confidence,
            obj.key,
        )
        _reject_photo(deps, obj, state)
        return

    species_entry = deps.catalog.by_class_index(result.class_index)
    state.model_label = species_entry.model_label
    state.confidence = result.confidence

    if result.confidence < deps.confidence_threshold:
        logger.info(
            "Confiança %.4f abaixo do limiar %.2f em %s (mais provável: %s); movendo para background/",
            result.confidence,
            deps.confidence_threshold,
            obj.key,
            species_entry.model_label,
        )
        _reject_photo(deps, obj, state)
        return

    species_id = deps.species_ids_by_scientific_name[species_entry.scientific_name]
    destination_key = f"{species_entry.s3_destination_prefix}{obj.filename}"
    image_url = deps.s3_gateway.public_url(destination_key)
    capture = extract_capture_timestamp(obj.key, obj.last_modified, deps.capture_timezone)

    logger.info(
        "Reconhecido %s (confiança %.4f) em %s; registrando avistamento",
        species_entry.model_label,
        result.confidence,
        obj.key,
    )

    if deps.dry_run:
        logger.info("[dry-run] criaria avistamento e moveria %s -> %s", obj.key, destination_key)
        return

    sighting = deps.api_client.create_sighting(
        species_id=species_id,
        sighting_date=capture.date,
        sighting_time=capture.time,
        image_url=image_url,
        idempotency_key=state.idempotency_key,
    )

    state.sighting_id = sighting.id
    state.destination_key = destination_key
    state.phase = PHASE_SIGHTING_CREATED
    deps.state_store.save(state)

    _finish_move(deps, obj, state)


def _reject_photo(deps: ProcessorDependencies, obj: S3Object, state: PhotoState) -> None:
    if not deps.dry_run:
        deps.s3_gateway.move_to_background(obj.key)
    state.phase = PHASE_REJECTED
    if not deps.dry_run:
        deps.state_store.save(state)


def _finish_move(deps: ProcessorDependencies, obj: S3Object, state: PhotoState) -> None:
    assert state.destination_key is not None  # garantido por _process_photo_inner
    if not deps.dry_run:
        deps.s3_gateway.move(obj.key, state.destination_key)
    state.phase = PHASE_COMPLETED
    if not deps.dry_run:
        deps.state_store.save(state)


def run_cycle(deps: ProcessorDependencies, max_images: int) -> int:
    """Processa até `max_images` fotos da raiz do bucket. Retorna quantas
    foram examinadas neste ciclo (para fins de log/observabilidade)."""
    objects = deps.s3_gateway.list_root_images()[:max_images]
    for obj in objects:
        process_photo(deps, obj)
    return len(objects)
