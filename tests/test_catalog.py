import json

import pytest

from feeder_worker.catalog import SpeciesCatalog, load_catalog_from_path, load_default_catalog


def _entry(**overrides):
    base = {
        "modelLabel": "Avoante",
        "classIndex": 0,
        "scientificName": "Zenaida auriculata",
        "commonName": "Avoante",
        "s3DestinationPrefix": "Avoante/",
        "description": "desc",
        "tips": "tips",
        "imageUrl": "https://example/img.jpg",
    }
    base.update(overrides)
    return base


def _write_catalog(tmp_path, species):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"species": species}), encoding="utf-8")
    return path


def test_load_default_catalog_has_three_species_in_order():
    catalog = load_default_catalog()

    assert catalog.num_classes == 3
    assert [e.model_label for e in catalog.entries] == ["Avoante", "Bem-Te-Vi", "Sabia-Laranjeira"]
    assert [e.class_index for e in catalog.entries] == [0, 1, 2]


def test_by_class_index_returns_matching_entry():
    catalog = load_default_catalog()

    entry = catalog.by_class_index(1)

    assert entry.model_label == "Bem-Te-Vi"
    assert entry.scientific_name == "Pitangus sulphuratus"


def test_unidentified_bird_is_a_sighting_category_not_an_identified_species():
    catalog = load_default_catalog()

    entry = catalog.by_class_index(3)

    assert catalog.num_classes == 3
    assert entry.model_label == "Outros"
    assert entry.common_name == "Ave não identificada"
    assert entry.scientific_name == "Espécie não identificada"
    assert entry.s3_destination_prefix == "outros/"
    assert len(catalog.sighting_entries) == 4
    assert all(entry.model_label != "Background" for entry in catalog.sighting_entries)


def test_by_class_index_raises_for_unknown_index():
    catalog = load_default_catalog()

    with pytest.raises(ValueError):
        catalog.by_class_index(99)


def test_destination_prefixes_match_catalog_entries():
    catalog = load_default_catalog()

    assert catalog.destination_prefixes() == {
        "Avoante/", "Bem-te-vi/", "Sabia-Laranjeira/", "outros/",
    }


def test_valid_two_entry_catalog_loads_from_path(tmp_path):
    species = [
        _entry(),
        _entry(modelLabel="Bem-Te-Vi", classIndex=1, scientificName="Pitangus sulphuratus",
               s3DestinationPrefix="Bem-te-vi/"),
    ]
    path = _write_catalog(tmp_path, species)

    catalog = load_catalog_from_path(path)

    assert catalog.num_classes == 2


def test_rejects_duplicate_class_index(tmp_path):
    species = [_entry(), _entry(modelLabel="Outro", scientificName="Outro nome")]
    path = _write_catalog(tmp_path, species)

    with pytest.raises(ValueError, match="classIndex duplicado"):
        load_catalog_from_path(path)


def test_rejects_duplicate_model_label(tmp_path):
    species = [_entry(), _entry(classIndex=1, scientificName="Outro nome")]
    path = _write_catalog(tmp_path, species)

    with pytest.raises(ValueError, match="modelLabel duplicado"):
        load_catalog_from_path(path)


def test_rejects_duplicate_scientific_name(tmp_path):
    species = [_entry(), _entry(modelLabel="Outro", classIndex=1)]
    path = _write_catalog(tmp_path, species)

    with pytest.raises(ValueError, match="Nomes científicos duplicados"):
        load_catalog_from_path(path)


def test_rejects_destination_prefix_without_trailing_slash(tmp_path):
    species = [_entry(s3DestinationPrefix="Avoante")]
    path = _write_catalog(tmp_path, species)

    with pytest.raises(ValueError, match="deve terminar com"):
        load_catalog_from_path(path)


def test_rejects_destination_prefix_with_nested_folders(tmp_path):
    species = [_entry(s3DestinationPrefix="Avoante/sub/")]
    path = _write_catalog(tmp_path, species)

    with pytest.raises(ValueError, match="único nível de pasta"):
        load_catalog_from_path(path)


def test_rejects_non_contiguous_class_indices(tmp_path):
    species = [_entry(classIndex=0), _entry(modelLabel="Outro", classIndex=2, scientificName="Outro nome")]
    path = _write_catalog(tmp_path, species)

    with pytest.raises(ValueError, match="sequência contígua"):
        load_catalog_from_path(path)


def test_rejects_empty_catalog(tmp_path):
    path = _write_catalog(tmp_path, [])

    with pytest.raises(ValueError, match="vazio"):
        load_catalog_from_path(path)


@pytest.mark.parametrize("label", ["Outros", "Background"])
def test_rejects_special_class_in_identified_species_list(tmp_path, label):
    path = _write_catalog(tmp_path, [_entry(modelLabel=label)])

    with pytest.raises(ValueError, match="lista de espécies identificadas"):
        load_catalog_from_path(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"classIndex": 0},
        {"modelLabel": "Background"},
        {"scientificName": "Zenaida auriculata"},
        {"s3DestinationPrefix": "outros/subfolder/"},
    ],
)
def test_rejects_invalid_unidentified_category(tmp_path, overrides):
    unidentified = _entry(
        modelLabel="Outros", classIndex=1, scientificName="Espécie não identificada",
        s3DestinationPrefix="outros/",
    )
    unidentified.update(overrides)
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"species": [_entry()], "unidentifiedBird": unidentified}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_catalog_from_path(path)


def test_catalog_class_rejects_duplicates_directly():
    from feeder_worker.catalog import SpeciesEntry

    entries = [
        SpeciesEntry("A", 0, "Sci A", "A", "A/", "d", "t", "u"),
        SpeciesEntry("B", 0, "Sci B", "B", "B/", "d", "t", "u"),
    ]

    with pytest.raises(ValueError, match="classIndex duplicado"):
        SpeciesCatalog(entries)
