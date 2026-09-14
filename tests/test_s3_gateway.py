from datetime import datetime, timezone

import pytest
from botocore.exceptions import ClientError

from feeder_worker.s3_gateway import BACKGROUND_PREFIX, ERRORS_PREFIX, S3Gateway


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return iter(self._pages)


class _FakeMeta:
    region_name = "us-east-1"


class _FakeS3Client:
    """Stub mínimo do cliente boto3 S3 usado pelo gateway, registrando as
    chamadas para asserção nos testes."""

    def __init__(self, pages):
        self._pages = pages
        self.meta = _FakeMeta()
        self.copy_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []
        self.fail_copy_for: set[str] = set()

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self._pages)

    def get_object(self, Bucket, Key):
        return {"Body": _FakeBody(b"fake-bytes")}

    def copy_object(self, Bucket, Key, CopySource, MetadataDirective):
        source_key = CopySource["Key"]
        if source_key in self.fail_copy_for:
            raise ClientError({"Error": {"Code": "InternalError", "Message": "boom"}}, "CopyObject")
        self.copy_calls.append((source_key, Key))

    def delete_object(self, Bucket, Key):
        self.delete_calls.append(Key)


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _page(*items):
    return {"Contents": list(items)}


def _item(key, etag='"abc123"', last_modified=None, size=100):
    return {
        "Key": key,
        "ETag": etag,
        "LastModified": last_modified or datetime(2026, 9, 10, tzinfo=timezone.utc),
        "Size": size,
    }


def test_list_root_images_excludes_nested_keys_and_known_prefixes():
    pages = [
        _page(
            _item("foto_1.jpg"),
            _item("app/AVO.JPG"),
            _item("Upload/x.jpg"),
            _item("Fotos salvas/y.jpg"),
            _item("Avoante/already-moved.jpg"),
            _item("erros/bad.jpg"),
            _item("nested/dir/z.jpg"),
        )
    ]
    client = _FakeS3Client(pages)
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes={"Avoante/"})

    objects = gateway.list_root_images()

    assert [obj.key for obj in objects] == ["foto_1.jpg"]


def test_list_root_images_filters_unsupported_extensions():
    pages = [_page(_item("foto_1.jpg"), _item("document.pdf"), _item("foto_2.png"), _item("clip.mp4"))]
    client = _FakeS3Client(pages)
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    objects = gateway.list_root_images()

    assert sorted(obj.key for obj in objects) == ["foto_1.jpg", "foto_2.png"]


def test_list_root_images_strips_etag_quotes():
    pages = [_page(_item("foto_1.jpg", etag='"quoted-etag"'))]
    client = _FakeS3Client(pages)
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    [obj] = gateway.list_root_images()

    assert obj.etag == "quoted-etag"


def test_list_root_images_sorts_by_last_modified_ascending():
    newer = _item("foto_new.jpg", last_modified=datetime(2026, 9, 10, tzinfo=timezone.utc))
    older = _item("foto_old.jpg", last_modified=datetime(2026, 1, 1, tzinfo=timezone.utc))
    client = _FakeS3Client([_page(newer, older)])
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    objects = gateway.list_root_images()

    assert [obj.key for obj in objects] == ["foto_old.jpg", "foto_new.jpg"]


def test_move_copies_then_deletes_in_order():
    client = _FakeS3Client([])
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    gateway.move("foto_1.jpg", "Avoante/foto_1.jpg")

    assert client.copy_calls == [("foto_1.jpg", "Avoante/foto_1.jpg")]
    assert client.delete_calls == ["foto_1.jpg"]


def test_move_does_not_delete_source_when_copy_fails():
    client = _FakeS3Client([])
    client.fail_copy_for.add("foto_1.jpg")
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    with pytest.raises(ClientError):
        gateway.move("foto_1.jpg", "Avoante/foto_1.jpg")

    assert client.delete_calls == []


def test_move_to_species_folder_preserves_original_filename():
    client = _FakeS3Client([])
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    destination = gateway.move_to_species_folder("foto_1.jpg", "Avoante/")

    assert destination == "Avoante/foto_1.jpg"
    assert client.copy_calls == [("foto_1.jpg", "Avoante/foto_1.jpg")]
    assert client.delete_calls == ["foto_1.jpg"]


def test_move_to_errors_uses_errors_prefix():
    client = _FakeS3Client([])
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    destination = gateway.move_to_errors("foto_1.jpg")

    assert destination == f"{ERRORS_PREFIX}foto_1.jpg"


def test_move_to_background_uses_background_prefix():
    client = _FakeS3Client([])
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    destination = gateway.move_to_background("foto_1.jpg")

    assert destination == f"{BACKGROUND_PREFIX}foto_1.jpg"
    assert client.copy_calls == [("foto_1.jpg", f"{BACKGROUND_PREFIX}foto_1.jpg")]
    assert client.delete_calls == ["foto_1.jpg"]


def test_delete_calls_delete_object_directly():
    client = _FakeS3Client([])
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    gateway.delete("foto_1.jpg")

    assert client.delete_calls == ["foto_1.jpg"]
    assert client.copy_calls == []


def test_public_url_uses_bucket_and_region():
    client = _FakeS3Client([])
    gateway = S3Gateway.with_client(client, bucket="my-bucket", excluded_prefixes=set())

    url = gateway.public_url("Avoante/foto_1.jpg")

    assert url == "https://my-bucket.s3.us-east-1.amazonaws.com/Avoante/foto_1.jpg"


def test_download_returns_object_bytes():
    client = _FakeS3Client([])
    gateway = S3Gateway.with_client(client, bucket="bucket", excluded_prefixes=set())

    assert gateway.download("foto_1.jpg") == b"fake-bytes"
