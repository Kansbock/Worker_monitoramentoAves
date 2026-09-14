import time

from feeder_worker.state import (
    PHASE_COMPLETED,
    PHASE_DISCOVERED,
    PHASE_FAILED_RETRY,
    PHASE_INVALID,
    PHASE_REJECTED,
    PhotoState,
    StateStore,
    compute_idempotency_key,
)


def test_compute_idempotency_key_is_deterministic_and_prefixed():
    key1 = compute_idempotency_key("bucket", "foto.jpg", "etag123")
    key2 = compute_idempotency_key("bucket", "foto.jpg", "etag123")

    assert key1 == key2
    assert key1.startswith("feeder-")


def test_compute_idempotency_key_differs_per_input():
    key1 = compute_idempotency_key("bucket", "foto.jpg", "etag123")
    key2 = compute_idempotency_key("bucket", "foto.jpg", "etag456")
    key3 = compute_idempotency_key("bucket", "outra.jpg", "etag123")

    assert len({key1, key2, key3}) == 3


def test_get_or_create_persists_discovered_row_on_first_call(tmp_path):
    store = StateStore(tmp_path / "state.db")

    state = store.get_or_create("foto.jpg", "etag1", "idem-1")

    assert state.phase == PHASE_DISCOVERED
    reloaded = store.get("foto.jpg", "etag1")
    assert reloaded is not None
    assert reloaded.phase == PHASE_DISCOVERED
    assert reloaded.idempotency_key == "idem-1"
    store.close()


def test_get_or_create_returns_existing_row_without_overwriting(tmp_path):
    store = StateStore(tmp_path / "state.db")
    first = store.get_or_create("foto.jpg", "etag1", "idem-1")
    first.phase = PHASE_COMPLETED
    store.save(first)

    second = store.get_or_create("foto.jpg", "etag1", "idem-1")

    assert second.phase == PHASE_COMPLETED
    store.close()


def test_save_round_trips_all_fields(tmp_path):
    store = StateStore(tmp_path / "state.db")
    state = PhotoState(
        s3_key="foto.jpg",
        etag="etag1",
        idempotency_key="idem-1",
        phase=PHASE_COMPLETED,
        model_label="Avoante",
        confidence=0.97,
        sighting_id="sig-1",
        destination_key="Avoante/foto.jpg",
        attempt_count=2,
        next_retry_at=123.0,
        last_error="boom",
    )

    store.save(state)
    reloaded = store.get("foto.jpg", "etag1")

    assert reloaded == state
    store.close()


def test_get_returns_none_when_not_found(tmp_path):
    store = StateStore(tmp_path / "state.db")

    assert store.get("missing.jpg", "etag") is None
    store.close()


def test_is_terminal_true_for_completed_rejected_and_invalid():
    for phase in (PHASE_COMPLETED, PHASE_REJECTED, PHASE_INVALID):
        state = PhotoState(s3_key="k", etag="e", idempotency_key="i", phase=phase)
        assert state.is_terminal() is True


def test_is_terminal_false_for_discovered_and_failed_retry():
    for phase in (PHASE_DISCOVERED, PHASE_FAILED_RETRY):
        state = PhotoState(s3_key="k", etag="e", idempotency_key="i", phase=phase)
        assert state.is_terminal() is False


def test_is_ready_for_retry_true_when_not_failed_retry_phase():
    state = PhotoState(s3_key="k", etag="e", idempotency_key="i", phase=PHASE_DISCOVERED)

    assert state.is_ready_for_retry(time.time()) is True


def test_is_ready_for_retry_respects_next_retry_at():
    now = time.time()
    state = PhotoState(
        s3_key="k", etag="e", idempotency_key="i", phase=PHASE_FAILED_RETRY, next_retry_at=now + 100
    )

    assert state.is_ready_for_retry(now) is False
    assert state.is_ready_for_retry(now + 200) is True


def test_is_ready_for_retry_true_when_next_retry_at_missing():
    state = PhotoState(s3_key="k", etag="e", idempotency_key="i", phase=PHASE_FAILED_RETRY, next_retry_at=None)

    assert state.is_ready_for_retry(time.time()) is True
