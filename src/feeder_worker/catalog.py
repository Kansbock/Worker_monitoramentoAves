"""Carregamento e validação do catálogo de espécies versionado.

O catálogo mapeia apenas os índices das espécies reconhecidas para o nome
científico usado como chave de integração com a API Aves, além do prefixo S3
e do conteúdo curatorial do seed. Outros é uma categoria de ave não
identificada, separada das espécies; Background não gera avistamentos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

OTHER_CLASS_LABEL = "Outros"
BACKGROUND_CLASS_LABEL = "Background"
EXTRA_CLASS_LABELS = (OTHER_CLASS_LABEL, BACKGROUND_CLASS_LABEL)


@dataclass(frozen=True)
class SpeciesEntry:
    model_label: str
    class_index: int
    scientific_name: str
    common_name: str
    s3_destination_prefix: str
    description: str
    tips: str
    image_url: str


class SpeciesCatalog:
    """Catálogo validado e indexado por índice de classe e por rótulo."""

    def __init__(self, entries: list[SpeciesEntry], unidentified_bird: SpeciesEntry | None = None):
        self._by_index: dict[int, SpeciesEntry] = {}
        self._by_label: dict[str, SpeciesEntry] = {}
        all_entries = entries + ([unidentified_bird] if unidentified_bird is not None else [])

        if any(entry.model_label in EXTRA_CLASS_LABELS for entry in entries):
            raise ValueError("Outros e Background não devem integrar a lista de espécies identificadas")
        if unidentified_bird is not None and (
            unidentified_bird.model_label != OTHER_CLASS_LABEL
            or unidentified_bird.class_index != len(entries)
        ):
            raise ValueError("A categoria de ave não identificada deve ser Outros, após as espécies")

        for entry in all_entries:
            if entry.class_index in self._by_index:
                raise ValueError(f"classIndex duplicado no catálogo: {entry.class_index}")
            if entry.model_label in self._by_label:
                raise ValueError(f"modelLabel duplicado no catálogo: {entry.model_label}")
            if not entry.s3_destination_prefix.endswith("/"):
                raise ValueError(
                    f"s3DestinationPrefix deve terminar com '/': {entry.s3_destination_prefix!r}"
                )
            if entry.s3_destination_prefix.count("/") != 1:
                raise ValueError(
                    "s3DestinationPrefix deve ser um único nível de pasta na raiz do bucket: "
                    f"{entry.s3_destination_prefix!r}"
                )
            self._by_index[entry.class_index] = entry
            self._by_label[entry.model_label] = entry

        expected_indices = set(range(len(all_entries)))
        if set(self._by_index.keys()) != expected_indices:
            raise ValueError(
                "classIndex do catálogo deve formar uma sequência contígua começando em 0: "
                f"encontrado {sorted(self._by_index.keys())}"
            )

        self._ordered = [self._by_index[i] for i in range(len(entries))]
        self._sighting_entries = [self._by_index[i] for i in range(len(all_entries))]

    @property
    def entries(self) -> list[SpeciesEntry]:
        return list(self._ordered)

    @property
    def num_classes(self) -> int:
        return len(self._ordered)

    @property
    def sighting_entries(self) -> list[SpeciesEntry]:
        return list(self._sighting_entries)

    def by_class_index(self, index: int) -> SpeciesEntry:
        try:
            return self._by_index[index]
        except KeyError as exc:
            raise ValueError(f"Nenhuma espécie associada ao índice de classe {index}") from exc

    def destination_prefixes(self) -> set[str]:
        """Prefixos de pasta de destino, usados para excluir do escopo de
        entrada qualquer objeto já organizado por espécie."""
        return {entry.s3_destination_prefix for entry in self._sighting_entries}


def _parse_entry(item: dict) -> SpeciesEntry:
    return SpeciesEntry(
        model_label=item["modelLabel"],
        class_index=item["classIndex"],
        scientific_name=item["scientificName"],
        common_name=item["commonName"],
        s3_destination_prefix=item["s3DestinationPrefix"],
        description=item["description"],
        tips=item["tips"],
        image_url=item["imageUrl"],
    )


def _parse_catalog(payload: dict) -> SpeciesCatalog:
    raw_entries = payload.get("species")
    if not raw_entries:
        raise ValueError("Catálogo de espécies vazio ou ausente (chave 'species')")

    entries = [_parse_entry(item) for item in raw_entries]
    raw_unidentified = payload.get("unidentifiedBird")
    unidentified = _parse_entry(raw_unidentified) if raw_unidentified is not None else None

    scientific_names = [entry.scientific_name for entry in entries]
    if unidentified is not None:
        scientific_names.append(unidentified.scientific_name)
    if len(scientific_names) != len(set(scientific_names)):
        raise ValueError("Nomes científicos duplicados no catálogo")

    return SpeciesCatalog(entries, unidentified)


def load_default_catalog() -> SpeciesCatalog:
    """Carrega o catálogo versionado embutido no pacote."""
    package_resources = resources.files("feeder_worker.resources")
    payload = json.loads((package_resources / "species_catalog.json").read_text(encoding="utf-8"))
    return _parse_catalog(payload)


def load_catalog_from_path(path: Path) -> SpeciesCatalog:
    """Carrega um catálogo de um caminho arbitrário (usado em testes)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _parse_catalog(payload)
