"""Persistência local de estado do worker (SQLite).

Cada foto processada é rastreada por `(key, etag)` para permitir retomar
depois de uma falha sem duplicar avistamentos nem perder imagens. A chave de
idempotência enviada à API é derivada deterministicamente de
`bucket + key + etag`, garantindo que reprocessar a mesma foto após uma
queda produza sempre o mesmo valor.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Fases do ciclo de vida de uma foto no worker.
PHASE_DISCOVERED = "discovered"
PHASE_SIGHTING_CREATED = "sighting_created"
PHASE_COMPLETED = "completed"
PHASE_REJECTED = "rejected"
PHASE_INVALID = "invalid"
PHASE_FAILED_RETRY = "failed_retry"

_TERMINAL_PHASES = {PHASE_COMPLETED, PHASE_REJECTED, PHASE_INVALID}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photo_state (
    s3_key TEXT NOT NULL,
    etag TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    phase TEXT NOT NULL,
    model_label TEXT,
    confidence REAL,
    sighting_id TEXT,
    destination_key TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at REAL,
    last_error TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (s3_key, etag)
);
"""


def compute_idempotency_key(bucket: str, key: str, etag: str) -> str:
    digest = hashlib.sha256(f"{bucket}:{key}:{etag}".encode("utf-8")).hexdigest()
    return f"feeder-{digest}"


@dataclass
class PhotoState:
    s3_key: str
    etag: str
    idempotency_key: str
    phase: str
    model_label: str | None = None
    confidence: float | None = None
    sighting_id: str | None = None
    destination_key: str | None = None
    attempt_count: int = 0
    next_retry_at: float | None = None
    last_error: str | None = None

    def is_terminal(self) -> bool:
        return self.phase in _TERMINAL_PHASES

    def is_ready_for_retry(self, now: float) -> bool:
        if self.phase != PHASE_FAILED_RETRY:
            return True
        return self.next_retry_at is None or now >= self.next_retry_at


class StateStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cursor:
            cursor.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def get(self, s3_key: str, etag: str) -> PhotoState | None:
        with closing(self._conn.cursor()) as cursor:
            row = cursor.execute(
                "SELECT * FROM photo_state WHERE s3_key = ? AND etag = ?", (s3_key, etag)
            ).fetchone()
        if row is None:
            return None
        return PhotoState(
            s3_key=row["s3_key"],
            etag=row["etag"],
            idempotency_key=row["idempotency_key"],
            phase=row["phase"],
            model_label=row["model_label"],
            confidence=row["confidence"],
            sighting_id=row["sighting_id"],
            destination_key=row["destination_key"],
            attempt_count=row["attempt_count"],
            next_retry_at=row["next_retry_at"],
            last_error=row["last_error"],
        )

    def save(self, state: PhotoState) -> None:
        with closing(self._conn.cursor()) as cursor:
            cursor.execute(
                """
                INSERT INTO photo_state (
                    s3_key, etag, idempotency_key, phase, model_label, confidence,
                    sighting_id, destination_key, attempt_count, next_retry_at,
                    last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(s3_key, etag) DO UPDATE SET
                    idempotency_key = excluded.idempotency_key,
                    phase = excluded.phase,
                    model_label = excluded.model_label,
                    confidence = excluded.confidence,
                    sighting_id = excluded.sighting_id,
                    destination_key = excluded.destination_key,
                    attempt_count = excluded.attempt_count,
                    next_retry_at = excluded.next_retry_at,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (
                    state.s3_key,
                    state.etag,
                    state.idempotency_key,
                    state.phase,
                    state.model_label,
                    state.confidence,
                    state.sighting_id,
                    state.destination_key,
                    state.attempt_count,
                    state.next_retry_at,
                    state.last_error,
                    time.time(),
                ),
            )
        self._conn.commit()

    def get_or_create(self, s3_key: str, etag: str, idempotency_key: str) -> PhotoState:
        existing = self.get(s3_key, etag)
        if existing is not None:
            return existing
        state = PhotoState(
            s3_key=s3_key, etag=etag, idempotency_key=idempotency_key, phase=PHASE_DISCOVERED
        )
        self.save(state)
        return state
