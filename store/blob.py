"""Object storage for uploaded datasets and generated PDF reports.

Wraps the MinIO client so callers work with plain bytes and never deal with
stream lifecycles. The underlying client is synchronous, so every call is
dispatched to a worker thread to keep the async call sites non-blocking.
"""

from __future__ import annotations

import asyncio
import io

import minio
from minio.error import S3Error

from config import settings


class BlobStore:
    """Reads and writes binary artifacts in a single MinIO bucket.

    The bucket is created on first use if it does not already exist.
    """

    def __init__(self) -> None:
        self._client = minio.Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )
        self._bucket = settings.MINIO_BUCKET
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    @staticmethod
    def dataset_key(dataset_id: str, filename: str = "data.parquet") -> str:
        """Build the object key under which a dataset is stored."""
        return f"datasets/{dataset_id}/{filename}"

    @staticmethod
    def report_key(analysis_id: str, filename: str = "report.pdf") -> str:
        """Build the object key under which a generated report is stored."""
        return f"reports/{analysis_id}/{filename}"

    def _upload(self, key: str, data: bytes, content_type: str) -> str:
        self._client.put_object(
            bucket_name=self._bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def _download(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def _exists(self, key: str) -> bool:
        try:
            self._client.stat_object(self._bucket, key)
            return True
        except S3Error:
            return False

    async def upload_bytes(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Store ``data`` under ``key`` and return the key."""
        return await asyncio.to_thread(self._upload, key, data, content_type)

    async def download_bytes(self, key: str) -> bytes:
        """Return the object stored under ``key``."""
        return await asyncio.to_thread(self._download, key)

    async def object_exists(self, key: str) -> bool:
        """Return whether an object exists at ``key``."""
        return await asyncio.to_thread(self._exists, key)

    async def delete_object(self, key: str) -> None:
        """Remove the object stored under ``key``."""
        await asyncio.to_thread(self._client.remove_object, self._bucket, key)

    async def list_objects(self, prefix: str = "") -> list[str]:
        """Return every object key beneath ``prefix``."""

        def _list() -> list[str]:
            return [
                obj.object_name
                for obj in self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
            ]

        return await asyncio.to_thread(_list)
