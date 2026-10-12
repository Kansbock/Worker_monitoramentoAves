from types import SimpleNamespace

import httpx
import pytest
import sqlite3

from feeder_worker.api_client import ApiError, NotificationBatch
from feeder_worker.cli import _process_notifications
from feeder_worker import cli
from feeder_worker.state import StateStore, PhotoState, PHASE_COMPLETED


class Executor:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def process_notifications(self):
        self.calls += 1
        if self.error:
            raise self.error
        return NotificationBatch(1, 2, 0)


def test_idle_cycle_can_process_notifications_without_photo_state(caplog):
    client = Executor()
    deps = SimpleNamespace(api_client=client, dry_run=False)
    with caplog.at_level("INFO"):
        _process_notifications(deps, True)
    assert client.calls == 1
    assert "concluídas=1" in caplog.text
    assert vars(deps) == {"api_client": client, "dry_run": False}


@pytest.mark.parametrize("error", [ApiError(503, "disabled"), httpx.ReadTimeout("synthetic")])
def test_executor_failure_is_logged_and_does_not_fail_photo_cycle(error, caplog):
    client = Executor(error)
    deps = SimpleNamespace(api_client=client, dry_run=False)
    _process_notifications(deps, True)
    assert client.calls == 1
    assert "Executor de notificações indisponível" in caplog.text


@pytest.mark.parametrize("enabled,dry_run", [(False, False), (True, True)])
def test_disabled_or_dry_run_does_not_call_api(enabled, dry_run):
    client = Executor()
    _process_notifications(SimpleNamespace(api_client=client, dry_run=dry_run), enabled)
    assert client.calls == 0


@pytest.mark.parametrize("dry_run", [False, True])
def test_run_once_processes_idle_queue_before_and_after_cycle_without_reopening_completed_photo(
    monkeypatch, tmp_path, dry_run
):
    from feeder_worker.api_client import AvesApiClient

    calls = []
    def handler(request):
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"token": "synthetic-token"})
        calls.append(request.url.path)
        return httpx.Response(503, json={"message": "synthetic-unavailable"})
    client = AvesApiClient(base_url="https://example.test", email="runner@example.test",
                          password="synthetic-test-only", transport=httpx.MockTransport(handler))
    path = tmp_path / "state.db"
    state = StateStore(path)
    state.save(PhotoState("finished.jpg", "synthetic-etag", "synthetic-idempotency", PHASE_COMPLETED))
    deps = SimpleNamespace(api_client=client, state_store=state, dry_run=dry_run,
                           s3_gateway=SimpleNamespace(list_root_images=lambda: []))
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(
        notification_processing_enabled=True, max_images_per_cycle=1, poll_interval_seconds=60))
    monkeypatch.setattr(cli, "_build_processor_dependencies", lambda settings, dry_run: deps)
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    args = cli.build_parser().parse_args(["run", "--once"] + (["--dry-run"] if dry_run else []))
    assert args.func(args) == 0
    assert calls == ([] if dry_run else ["/api/internal/notification-jobs/process"] * 2)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT phase FROM photo_state").fetchall() == [(PHASE_COMPLETED,)]
