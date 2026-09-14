"""Extração de data/hora de captura a partir do nome do arquivo, com
fallback para o metadado LastModified do S3.

As fotos seguem o padrão `foto_YYYYMMDD_HHMMSS_n.jpg`. Quando o nome não
corresponde a esse padrão, usa-se o LastModified do objeto (que já vem em
UTC do S3) convertido para o fuso configurado.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_FILENAME_PATTERN = re.compile(
    r"^foto_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})_\d+\.\w+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CaptureTimestamp:
    date: date
    time: dt_time
    source: str  # "filename" ou "s3_last_modified"


def extract_capture_timestamp(
    key: str, last_modified: datetime, capture_timezone: ZoneInfo
) -> CaptureTimestamp:
    """Extrai a data/hora de captura de uma chave S3 (ex.: 'foto_...jpg')."""
    filename = key.rsplit("/", maxsplit=1)[-1]
    match = _FILENAME_PATTERN.match(filename)

    if match:
        try:
            parsed = datetime(
                year=int(match["year"]),
                month=int(match["month"]),
                day=int(match["day"]),
                hour=int(match["hour"]),
                minute=int(match["minute"]),
                second=int(match["second"]),
                tzinfo=capture_timezone,
            )
            return CaptureTimestamp(date=parsed.date(), time=parsed.time(), source="filename")
        except ValueError:
            logger.warning(
                "Nome de arquivo com padrão de data/hora inválido, usando LastModified: %s", key
            )

    localized = last_modified.astimezone(capture_timezone)
    return CaptureTimestamp(date=localized.date(), time=localized.time(), source="s3_last_modified")
