"""Cliente HTTP da API Aves.

Autentica como a conta de serviço do worker, mantém o JWT somente em
memória (renovado sob demanda após 401) e expõe as operações necessárias:
registro/consulta de usuário de serviço, catálogo de espécies e criação
idempotente de avistamentos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time

import httpx

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class SpeciesRecord:
    id: str
    name: str
    scientific_name: str


@dataclass(frozen=True)
class SightingRecord:
    id: str
    species_ids: list[str]


@dataclass(frozen=True)
class NotificationBatch:
    completed: int
    deferred: int
    failed: int


class AvesApiClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        password: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        # `transport` permite injetar um `httpx.MockTransport` nos testes,
        # sem abrir conexões de rede reais.
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._email = email
        self._password = password
        self._token: str | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AvesApiClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- autenticação ----

    def authenticate(self) -> str:
        """Executa login e retorna o JWT. Usado explicitamente pelo comando
        de bootstrap para validar as credenciais da conta de serviço."""
        return self._login()

    def _login(self) -> str:
        response = self._client.post(
            "/api/auth/login", json={"email": self._email, "password": self._password}
        )
        self._raise_for_status(response)
        token = response.json()["token"]
        self._token = token
        return token

    def _authorized_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if self._token is None:
            self._login()

        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._token}"
        response = self._client.request(method, path, headers=headers, **kwargs)

        if response.status_code == 401:
            # Token expirado/inválido: renova uma vez e tenta novamente.
            logger.info("Token expirado ao chamar %s %s, renovando login", method, path)
            self._login()
            headers["Authorization"] = f"Bearer {self._token}"
            response = self._client.request(method, path, headers=headers, **kwargs)

        return response

    # ---- usuário de serviço (bootstrap) ----

    def process_notifications(self) -> NotificationBatch:
        response = self._authorized_request("POST", "/api/internal/notification-jobs/process")
        self._raise_for_status(response)
        try:
            body = response.json()
            values = [body[key] for key in ("completed", "deferred", "failed")]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("invalid counters")
        except (ValueError, KeyError, TypeError):
            raise ApiError(502, "Resposta inválida do executor de notificações") from None
        return NotificationBatch(*values)

    def register_user(self, name: str, email: str, password: str) -> bool:
        """Cria o usuário de serviço. Retorna True se criado agora, False se
        já existia (409 tratado como estado esperado no bootstrap)."""
        response = self._client.post(
            "/api/users", json={"name": name, "email": email, "password": password}
        )
        if response.status_code == 409:
            return False
        self._raise_for_status(response)
        return True

    # ---- catálogo de espécies ----

    def list_species(self) -> list[SpeciesRecord]:
        response = self._client.get("/api/species")
        self._raise_for_status(response)
        return [
            SpeciesRecord(id=item["id"], name=item["name"], scientific_name=item["scientificName"])
            for item in response.json()
        ]

    def create_species(
        self, name: str, scientific_name: str, description: str, tips: str, image_url: str
    ) -> SpeciesRecord:
        response = self._authorized_request(
            "POST",
            "/api/species",
            json={
                "name": name,
                "scientificName": scientific_name,
                "description": description,
                "tips": tips,
                "imageUrl": image_url,
            },
        )
        self._raise_for_status(response)
        item = response.json()
        return SpeciesRecord(id=item["id"], name=item["name"], scientific_name=item["scientificName"])

    # ---- avistamentos ----

    def create_sighting(
        self,
        *,
        species_id: str,
        sighting_date: date,
        sighting_time: time,
        image_url: str,
        idempotency_key: str,
    ) -> SightingRecord:
        response = self._authorized_request(
            "POST",
            "/api/sightings",
            json={
                "speciesIds": [species_id],
                "date": sighting_date.isoformat(),
                "time": sighting_time.strftime("%H:%M:%S"),
                "imageUrl": image_url,
            },
            headers={"Idempotency-Key": idempotency_key},
        )
        self._raise_for_status(response)
        item = response.json()
        return SightingRecord(id=item["id"], species_ids=[s["id"] for s in item["species"]])

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            message = response.json().get("message", response.text)
        except ValueError:
            message = response.text
        raise ApiError(response.status_code, message)
