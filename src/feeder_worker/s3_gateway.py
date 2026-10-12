"""Gateway de acesso ao bucket S3 do comedouro.

Regras de escopo (definidas no plano):
  * Entrada: somente objetos diretamente na raiz do bucket (chave sem "/").
  * "app/", "Upload/", "Fotos salvas/", "erros/", "background/", "outros/" e qualquer
    prefixo de destino por espécie nunca são tratados como entrada.
  * Mover = copiar para o destino e só então excluir a origem, nessa ordem,
    para nunca perder uma imagem por falha a meio caminho.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ERRORS_PREFIX = "erros/"
BACKGROUND_PREFIX = "background/"


@dataclass(frozen=True)
class S3Object:
    key: str
    etag: str
    last_modified: datetime
    size: int

    @property
    def filename(self) -> str:
        return PurePosixPath(self.key).name


class S3Gateway:
    def __init__(self, bucket: str, region: str, profile: str, excluded_prefixes: set[str]):
        session = boto3.Session(profile_name=profile, region_name=region)
        self._client = session.client("s3", config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}))
        self._bucket = bucket
        # Prefixos nunca tratados como entrada: pastas de destino por espécie
        # somadas às pastas fixas do acervo/app.
        self._excluded_prefixes = excluded_prefixes | {
            "app/",
            "Upload/",
            "Fotos salvas/",
            ERRORS_PREFIX,
            BACKGROUND_PREFIX,
        }

    @classmethod
    def with_client(cls, client, bucket: str, excluded_prefixes: set[str]) -> "S3Gateway":
        """Constrói o gateway com um cliente S3 já pronto (usado nos testes
        para injetar um cliente fake, sem depender de credenciais/rede reais)."""
        instance = cls.__new__(cls)
        instance._client = client
        instance._bucket = bucket
        instance._excluded_prefixes = excluded_prefixes | {
            "app/",
            "Upload/",
            "Fotos salvas/",
            ERRORS_PREFIX,
            BACKGROUND_PREFIX,
        }
        return instance

    @property
    def bucket(self) -> str:
        return self._bucket

    def list_root_images(self) -> list[S3Object]:
        """Lista objetos de imagem diretamente na raiz do bucket (sem '/' na chave)."""
        paginator = self._client.get_paginator("list_objects_v2")
        objects: list[S3Object] = []

        for page in paginator.paginate(Bucket=self._bucket):
            for item in page.get("Contents", []):
                key = item["Key"]
                if self._is_out_of_scope(key):
                    continue
                if PurePosixPath(key).suffix.lower() not in _SUPPORTED_EXTENSIONS:
                    continue
                objects.append(
                    S3Object(
                        key=key,
                        etag=item["ETag"].strip('"'),
                        last_modified=item["LastModified"],
                        size=item["Size"],
                    )
                )

        objects.sort(key=lambda obj: obj.last_modified)
        return objects

    def _is_out_of_scope(self, key: str) -> bool:
        if "/" in key:
            return True
        return any(key.startswith(prefix) for prefix in self._excluded_prefixes)

    def download(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def move(self, source_key: str, destination_key: str) -> None:
        """Copia o objeto para o destino e só então remove a origem. Não
        remove a origem se a cópia falhar."""
        self._copy(source_key, destination_key)
        self._delete(source_key)

    def move_to_species_folder(self, key: str, destination_prefix: str) -> str:
        """Copia o objeto para a pasta de destino e remove a origem.
        Retorna a chave final."""
        destination_key = f"{destination_prefix}{PurePosixPath(key).name}"
        self.move(key, destination_key)
        return destination_key

    def move_to_errors(self, key: str) -> str:
        destination_key = f"{ERRORS_PREFIX}{PurePosixPath(key).name}"
        self.move(key, destination_key)
        return destination_key

    def move_to_background(self, key: str) -> str:
        """Move para 'background/' uma cena vazia ou previsão abaixo do limiar."""
        destination_key = f"{BACKGROUND_PREFIX}{PurePosixPath(key).name}"
        self.move(key, destination_key)
        return destination_key

    def delete(self, key: str) -> None:
        self._delete(key)

    def public_url(self, key: str) -> str:
        return f"https://{self._bucket}.s3.{self._client.meta.region_name}.amazonaws.com/{key}"

    def _copy(self, source_key: str, destination_key: str) -> None:
        logger.info("Copiando s3://%s/%s -> %s", self._bucket, source_key, destination_key)
        self._client.copy_object(
            Bucket=self._bucket,
            Key=destination_key,
            CopySource={"Bucket": self._bucket, "Key": source_key},
            MetadataDirective="COPY",
        )

    def _delete(self, key: str) -> None:
        logger.info("Removendo s3://%s/%s", self._bucket, key)
        self._client.delete_object(Bucket=self._bucket, Key=key)
