from datetime import date, time

import httpx
import pytest

from feeder_worker.api_client import ApiError, AvesApiClient


def _client_with_handler(handler) -> AvesApiClient:
    transport = httpx.MockTransport(handler)
    return AvesApiClient(
        base_url="https://api.test",
        email="worker@example.com",
        password="secret",
        transport=transport,
    )


def test_authenticate_posts_credentials_and_returns_token():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == "/api/auth/login"
        assert request.method == "POST"
        return httpx.Response(200, json={"token": "jwt-123"})

    client = _client_with_handler(handler)

    token = client.authenticate()

    assert token == "jwt-123"
    assert len(calls) == 1


def test_authorized_request_logs_in_first_when_no_token_cached():
    login_calls = 0
    species_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_calls, species_calls
        if request.url.path == "/api/auth/login":
            login_calls += 1
            return httpx.Response(200, json={"token": "jwt-1"})
        if request.url.path == "/api/species" and request.method == "POST":
            species_calls += 1
            assert request.headers["authorization"] == "Bearer jwt-1"
            return httpx.Response(
                201,
                json={"id": "sp-1", "name": "Arara", "scientificName": "Ara ararauna"},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    client = _client_with_handler(handler)

    result = client.create_species("Arara", "Ara ararauna", "desc", "tips", "https://img")

    assert result.id == "sp-1"
    assert login_calls == 1
    assert species_calls == 1


def test_authorized_request_relogs_in_once_after_401_and_retries():
    tokens_issued = []
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            token = f"jwt-{len(tokens_issued) + 1}"
            tokens_issued.append(token)
            return httpx.Response(200, json={"token": token})
        if request.url.path == "/api/species" and request.method == "POST":
            attempts.append(request.headers["authorization"])
            if request.headers["authorization"] == "Bearer jwt-1":
                return httpx.Response(401, json={"message": "expirado"})
            return httpx.Response(
                201,
                json={"id": "sp-1", "name": "Arara", "scientificName": "Ara ararauna"},
            )
        raise AssertionError("unexpected request")

    client = _client_with_handler(handler)

    species = client.create_species("Arara", "Ara ararauna", "desc", "tips", "https://img")

    assert species.id == "sp-1"
    assert attempts == ["Bearer jwt-1", "Bearer jwt-2"]
    assert len(tokens_issued) == 2


def test_register_user_returns_false_on_409_conflict():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/users"
        return httpx.Response(409, json={"message": "já existe"})

    client = _client_with_handler(handler)

    created = client.register_user("Worker", "worker@example.com", "secret")

    assert created is False


def test_register_user_returns_true_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": "user-1"})

    client = _client_with_handler(handler)

    created = client.register_user("Worker", "worker@example.com", "secret")

    assert created is True


def test_list_species_maps_response_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"id": "sp-1", "name": "Arara", "scientificName": "Ara ararauna"},
                {"id": "sp-2", "name": "Tucano", "scientificName": "Ramphastos toco"},
            ],
        )

    client = _client_with_handler(handler)

    species = client.list_species()

    assert [s.id for s in species] == ["sp-1", "sp-2"]
    assert [s.scientific_name for s in species] == ["Ara ararauna", "Ramphastos toco"]


def test_create_sighting_sends_idempotency_key_header_and_single_species_id():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"token": "jwt-1"})
        captured["headers"] = request.headers
        captured["body"] = request.content
        return httpx.Response(
            201,
            json={
                "id": "sig-1",
                "species": [{"id": "sp-1", "name": "Arara", "scientificName": "Ara ararauna"}],
            },
        )

    client = _client_with_handler(handler)

    sighting = client.create_sighting(
        species_id="sp-1",
        sighting_date=date(2026, 9, 10),
        sighting_time=time(10, 15, 0),
        image_url="https://img/sig-1.jpg",
        idempotency_key="feeder-abc123",
    )

    assert sighting.id == "sig-1"
    assert sighting.species_ids == ["sp-1"]
    assert captured["headers"]["idempotency-key"] == "feeder-abc123"
    import json as _json

    body = _json.loads(captured["body"])
    assert body["speciesIds"] == ["sp-1"]
    assert body["date"] == "2026-09-10"
    assert body["time"] == "10:15:00"
    assert body["imageUrl"] == "https://img/sig-1.jpg"


def test_create_sighting_raises_api_error_with_parsed_message_on_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"token": "jwt-1"})
        return httpx.Response(409, json={"message": "Idempotency-Key já foi usada"})

    client = _client_with_handler(handler)

    with pytest.raises(ApiError) as exc_info:
        client.create_sighting(
            species_id="sp-1",
            sighting_date=date(2026, 9, 10),
            sighting_time=time(10, 15, 0),
            image_url="https://img",
            idempotency_key="feeder-abc123",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.message == "Idempotency-Key já foi usada"


def test_create_sighting_raises_api_error_with_raw_text_when_body_is_not_json():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"token": "jwt-1"})
        return httpx.Response(500, text="internal server error")

    client = _client_with_handler(handler)

    with pytest.raises(ApiError) as exc_info:
        client.create_sighting(
            species_id="sp-1",
            sighting_date=date(2026, 9, 10),
            sighting_time=time(10, 15, 0),
            image_url="https://img",
            idempotency_key="feeder-abc123",
        )

    assert exc_info.value.status_code == 500
    assert "internal server error" in exc_info.value.message


def test_context_manager_closes_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token": "jwt-1"})

    transport = httpx.MockTransport(handler)
    with AvesApiClient(
        base_url="https://api.test", email="a@b.com", password="secret", transport=transport
    ) as client:
        client.authenticate()

    assert client._client.is_closed
