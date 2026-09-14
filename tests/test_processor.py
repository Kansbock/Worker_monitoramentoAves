from datetime import date, datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo

import pytest

from feeder_worker.api_client import ApiError, SightingRecord, SpeciesRecord
from feeder_worker.catalog import load_default_catalog
from feeder_worker.classifier import ClassificationResult, InvalidImageError
from feeder_worker.processor import (
    ProcessorDependencies,
    SpeciesRegistrationError,
    _compute_backoff_seconds,
    process_photo,
    resolve_species_ids,
    run_cycle,
)
from feeder_worker.s3_gateway import S3Object
from feeder_worker.state import (
    PHASE_COMPLETED,
    PHASE_DISCOVERED,
    PHASE_FAILED_RETRY,
    PHASE_INVALID,
    PHASE_REJECTED,
    PHASE_SIGHTING_CREATED,
    PhotoState,
    StateStore,
)

SP_TZ = ZoneInfo("America/Sao_Paulo")

CATALOG = load_default_catalog()

SPECIES_IDS = {
    "Zenaida auriculata": "sp-avoante",
    "Pitangus sulphuratus": "sp-bem-te-vi",
    "Turdus rufiventris": "sp-sabia",
}


class FakeS3Gateway:
    def __init__(self, bucket="test-bucket", objects=None):
        self.bucket = bucket
        self._objects = objects or []
        self.downloaded: list[str] = []
        self.moved: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.moved_to_errors: list[str] = []
        self.moved_to_background: list[str] = []

    def list_root_images(self):
        return list(self._objects)

    def download(self, key):
        self.downloaded.append(key)
        return b"fake-bytes"

    def move(self, source_key, destination_key):
        self.moved.append((source_key, destination_key))

    def move_to_errors(self, key):
        self.moved_to_errors.append(key)
        return f"erros/{key}"

    def move_to_background(self, key):
        self.moved_to_background.append(key)
        return f"background/{key}"

    def delete(self, key):
        self.deleted.append(key)

    def public_url(self, key):
        return f"https://{self.bucket}.s3.amazonaws.com/{key}"


class FakeClassifier:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def classify(self, image_bytes):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class FakeApiClient:
    def __init__(self, sighting_id="sig-1", error=None):
        self.sighting_id = sighting_id
        self.error = error
        self.calls: list[dict] = []

    def create_sighting(self, *, species_id, sighting_date, sighting_time, image_url, idempotency_key):
        self.calls.append(
            {
                "species_id": species_id,
                "sighting_date": sighting_date,
                "sighting_time": sighting_time,
                "image_url": image_url,
                "idempotency_key": idempotency_key,
            }
        )
        if self.error is not None:
            raise self.error
        return SightingRecord(id=self.sighting_id, species_ids=[species_id])

    def list_species(self):
        return [
            SpeciesRecord(id=species_id, name=name, scientific_name=name)
            for name, species_id in SPECIES_IDS.items()
        ]


def _s3_object(key="foto_20260910_101500_1.jpg", etag="etag-1"):
    return S3Object(key=key, etag=etag, last_modified=datetime(2026, 9, 10, 13, 15, tzinfo=timezone.utc), size=100)


def _deps(s3_gateway=None, classifier=None, api_client=None, state_store=None, dry_run=False):
    return ProcessorDependencies(
        s3_gateway=s3_gateway or FakeS3Gateway(),
        api_client=api_client or FakeApiClient(),
        classifier=classifier or FakeClassifier(result=ClassificationResult(0, 0.99, [0.99, 0.005, 0.005])),
        state_store=state_store,
        catalog=CATALOG,
        species_ids_by_scientific_name=dict(SPECIES_IDS),
        capture_timezone=SP_TZ,
        confidence_threshold=0.85,
        dry_run=dry_run,
    )


@pytest.fixture
def state_store(tmp_path):
    store = StateStore(tmp_path / "state.db")
    yield store
    store.close()


# ---- reconhecimento acima do limiar ----

def test_recognized_species_creates_sighting_and_moves_photo(state_store):
    obj = _s3_object()
    api_client = FakeApiClient(sighting_id="sig-99")
    s3_gateway = FakeS3Gateway()
    deps = _deps(s3_gateway=s3_gateway, api_client=api_client, state_store=state_store)

    process_photo(deps, obj)

    assert len(api_client.calls) == 1
    call = api_client.calls[0]
    assert call["species_id"] == "sp-avoante"
    assert call["sighting_date"] == date(2026, 9, 10)
    assert call["sighting_time"] == dt_time(10, 15, 0)
    assert call["idempotency_key"].startswith("feeder-")

    assert s3_gateway.moved == [(obj.key, "Avoante/foto_20260910_101500_1.jpg")]
    assert s3_gateway.deleted == []

    saved = state_store.get(obj.key, obj.etag)
    assert saved.phase == PHASE_COMPLETED
    assert saved.sighting_id == "sig-99"


# ---- confiança abaixo do limiar ----

def test_low_confidence_moves_to_background_without_creating_sighting(state_store):
    obj = _s3_object()
    classifier = FakeClassifier(result=ClassificationResult(0, 0.5, [0.5, 0.3, 0.2]))
    api_client = FakeApiClient()
    s3_gateway = FakeS3Gateway()
    deps = _deps(s3_gateway=s3_gateway, classifier=classifier, api_client=api_client, state_store=state_store)

    process_photo(deps, obj)

    assert api_client.calls == []
    assert s3_gateway.moved_to_background == [obj.key]
    assert s3_gateway.deleted == []
    assert s3_gateway.moved == []

    saved = state_store.get(obj.key, obj.etag)
    assert saved.phase == PHASE_REJECTED


# ---- arquivo inválido ----

def test_invalid_image_moves_to_errors_and_marks_invalid(state_store):
    obj = _s3_object()
    classifier = FakeClassifier(error=InvalidImageError("corrupted"))
    api_client = FakeApiClient()
    s3_gateway = FakeS3Gateway()
    deps = _deps(s3_gateway=s3_gateway, classifier=classifier, api_client=api_client, state_store=state_store)

    process_photo(deps, obj)

    assert api_client.calls == []
    assert s3_gateway.moved_to_errors == [obj.key]
    assert s3_gateway.deleted == []

    saved = state_store.get(obj.key, obj.etag)
    assert saved.phase == PHASE_INVALID


# ---- retomada após avistamento já criado ----

def test_resume_from_sighting_created_skips_reclassification_and_api_call(state_store):
    obj = _s3_object()
    state_store.save(
        PhotoState(
            s3_key=obj.key,
            etag=obj.etag,
            idempotency_key="feeder-existing",
            phase=PHASE_SIGHTING_CREATED,
            sighting_id="sig-already-created",
            destination_key="Avoante/foto_20260910_101500_1.jpg",
        )
    )
    classifier = FakeClassifier(error=AssertionError("não deveria reclassificar"))
    api_client = FakeApiClient(error=AssertionError("não deveria recriar avistamento"))
    s3_gateway = FakeS3Gateway()
    deps = _deps(s3_gateway=s3_gateway, classifier=classifier, api_client=api_client, state_store=state_store)

    process_photo(deps, obj)

    assert classifier.calls == 0
    assert api_client.calls == []
    assert s3_gateway.downloaded == []
    assert s3_gateway.moved == [(obj.key, "Avoante/foto_20260910_101500_1.jpg")]

    saved = state_store.get(obj.key, obj.etag)
    assert saved.phase == PHASE_COMPLETED
    assert saved.sighting_id == "sig-already-created"


# ---- estados terminais e de retentativa ----

def test_terminal_state_is_never_reprocessed(state_store):
    obj = _s3_object()
    state_store.save(
        PhotoState(s3_key=obj.key, etag=obj.etag, idempotency_key="feeder-x", phase=PHASE_COMPLETED)
    )
    classifier = FakeClassifier(error=AssertionError("não deveria classificar"))
    s3_gateway = FakeS3Gateway()
    deps = _deps(s3_gateway=s3_gateway, classifier=classifier, state_store=state_store)

    process_photo(deps, obj)

    assert s3_gateway.downloaded == []
    assert s3_gateway.moved == []
    assert s3_gateway.deleted == []


def test_failed_retry_before_backoff_window_is_skipped(state_store):
    obj = _s3_object()
    state_store.save(
        PhotoState(
            s3_key=obj.key,
            etag=obj.etag,
            idempotency_key="feeder-x",
            phase=PHASE_FAILED_RETRY,
            next_retry_at=time_in_future(),
        )
    )
    classifier = FakeClassifier(error=AssertionError("não deveria classificar ainda"))
    deps = _deps(classifier=classifier, state_store=state_store)

    process_photo(deps, obj)

    assert classifier.calls == 0


def time_in_future():
    import time as _time

    return _time.time() + 3600


# ---- isolamento de falhas por foto ----

def test_exception_during_processing_marks_failed_retry_with_backoff(state_store):
    obj = _s3_object()
    api_client = FakeApiClient(error=RuntimeError("falha transitória de rede"))
    deps = _deps(api_client=api_client, state_store=state_store)

    process_photo(deps, obj)

    saved = state_store.get(obj.key, obj.etag)
    assert saved.phase == PHASE_FAILED_RETRY
    assert saved.attempt_count == 1
    assert saved.last_error == "falha transitória de rede"
    assert saved.next_retry_at is not None and saved.next_retry_at > 0


def test_compute_backoff_seconds_doubles_and_caps():
    assert _compute_backoff_seconds(1) == 30
    assert _compute_backoff_seconds(2) == 60
    assert _compute_backoff_seconds(3) == 120
    assert _compute_backoff_seconds(20) == 3600  # respeita o teto


# ---- dry-run ----

def test_dry_run_never_calls_api_or_mutates_s3(state_store):
    obj = _s3_object()
    api_client = FakeApiClient()
    s3_gateway = FakeS3Gateway()
    deps = _deps(s3_gateway=s3_gateway, api_client=api_client, state_store=state_store, dry_run=True)

    process_photo(deps, obj)

    assert api_client.calls == []
    assert s3_gateway.moved == []
    assert s3_gateway.deleted == []
    assert s3_gateway.moved_to_errors == []

    saved = state_store.get(obj.key, obj.etag)
    assert saved.phase == PHASE_DISCOVERED


# ---- resolução de espécies do catálogo ----

def test_resolve_species_ids_maps_scientific_name_to_api_id():
    api_client = FakeApiClient()

    mapping = resolve_species_ids(CATALOG, api_client)

    assert mapping == SPECIES_IDS


def test_resolve_species_ids_raises_when_catalog_species_missing_from_api():
    class PartialApiClient:
        def list_species(self):
            return [SpeciesRecord(id="sp-avoante", name="Avoante", scientific_name="Zenaida auriculata")]

    with pytest.raises(SpeciesRegistrationError, match="seed-species"):
        resolve_species_ids(CATALOG, PartialApiClient())


# ---- limite de imagens por ciclo ----

def test_run_cycle_processes_at_most_max_images(state_store):
    objects = [_s3_object(key=f"foto_2026091{i}_101500_1.jpg", etag=f"etag-{i}") for i in range(5)]
    s3_gateway = FakeS3Gateway(objects=objects)
    classifier = FakeClassifier(result=ClassificationResult(0, 0.99, [0.99, 0.005, 0.005]))
    deps = _deps(s3_gateway=s3_gateway, classifier=classifier, state_store=state_store)

    examined = run_cycle(deps, max_images=2)

    assert examined == 2
    assert classifier.calls == 2
