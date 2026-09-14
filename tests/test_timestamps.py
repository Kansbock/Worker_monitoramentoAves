from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from feeder_worker.timestamps import extract_capture_timestamp

SP_TZ = ZoneInfo("America/Sao_Paulo")


def test_extracts_date_and_time_from_filename_pattern():
    last_modified = datetime(2020, 1, 1, tzinfo=timezone.utc)

    result = extract_capture_timestamp("foto_20260910_101500_1.jpg", last_modified, SP_TZ)

    assert result.date.isoformat() == "2026-09-10"
    assert result.time.isoformat() == "10:15:00"
    assert result.source == "filename"


def test_filename_pattern_is_case_insensitive_and_ignores_extension():
    last_modified = datetime(2020, 1, 1, tzinfo=timezone.utc)

    result = extract_capture_timestamp("FOTO_20260910_101500_2.JPEG", last_modified, SP_TZ)

    assert result.source == "filename"


def test_falls_back_to_last_modified_when_filename_does_not_match():
    last_modified = datetime(2026, 9, 10, 13, 15, 0, tzinfo=timezone.utc)

    result = extract_capture_timestamp("qualquer-outro-nome.jpg", last_modified, SP_TZ)

    assert result.source == "s3_last_modified"
    # 13:15 UTC -> 10:15 em America/Sao_Paulo (UTC-3)
    assert result.time.isoformat() == "10:15:00"
    assert result.date.isoformat() == "2026-09-10"


def test_falls_back_to_last_modified_when_filename_has_invalid_date_components():
    last_modified = datetime(2026, 9, 10, 13, 15, 0, tzinfo=timezone.utc)

    # mês 13 não existe: o regex casa, mas a construção do datetime falha.
    result = extract_capture_timestamp("foto_20261310_101500_1.jpg", last_modified, SP_TZ)

    assert result.source == "s3_last_modified"


def test_handles_nested_key_by_using_only_the_filename_part():
    last_modified = datetime(2020, 1, 1, tzinfo=timezone.utc)

    result = extract_capture_timestamp("some/prefix/foto_20260910_101500_1.jpg", last_modified, SP_TZ)

    assert result.source == "filename"
    assert result.date.isoformat() == "2026-09-10"
