import logging
from typing import Any

import boto3  # type: ignore
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from tqdm import tqdm
from botocore.exceptions import ClientError  # type: ignore

from queryboost.exceptions import QueryboostBatchHandlerError

from .base import BatchHandler

logger = logging.getLogger(__name__)
""" :meta private: """


class S3ParquetBatchHandler(BatchHandler):
    """Handler that uploads batches to S3 as Parquet files with buffering.

    This handler accumulates record batches in memory and writes them to S3 Parquet files
    when the buffer exceeds ``target_write_bytes``. Each flush operation creates a separate
    Parquet file (part-00000.parquet, part-00001.parquet, etc.) in the specified S3 location.

    This buffering approach reduces S3 API calls and improves upload efficiency by combining
    multiple small batches into fewer, larger Parquet files.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        target_write_bytes: int = 256 * 1024 * 1024,
        metadata: dict[str, Any] = {},
    ):
        """Initialize the S3 Parquet batch handler.

        Args:
            bucket: S3 bucket name where files will be uploaded
            prefix: Required prefix/folder path in the bucket (must be empty initially)
            target_write_bytes: Target size in bytes for each Parquet file. When the
                accumulated buffer size exceeds this threshold, batches are written to
                a new Parquet file in S3. Defaults to 256 MB.
            metadata: Optional metadata dictionary
        """

        super().__init__(target_write_bytes, metadata)

        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._s3_client = boto3.client("s3")
        self._fs = pafs.S3FileSystem()

        # Ensure bucket exists or create it
        self._ensure_bucket_exists()

        # Ensure prefix is non-empty and currently empty in S3
        self._check_prefix()

        tqdm.write(f"Uploading results as parquet files to: s3://{bucket}/{self._prefix}/")
        tqdm.write("")

    def _ensure_bucket_exists(self) -> None:
        """Check if bucket exists; create it if not."""

        try:
            self._s3_client.head_bucket(Bucket=self._bucket)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchBucket"):
                tqdm.write(f"Bucket '{self._bucket}' not found. Creating it...")

                # Let AWS config determine region which mirrors CLI behavior
                self._s3_client.create_bucket(Bucket=self._bucket)
            else:
                raise QueryboostBatchHandlerError(f"Failed to access S3 bucket '{self._bucket}'.") from e

    def _check_prefix(self) -> None:
        """Check if the S3 prefix is valid."""

        if not self._prefix:
            raise QueryboostBatchHandlerError("S3 prefix is required.")

        resp = self._s3_client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=self._prefix + "/",
            MaxKeys=1,
        )
        if "Contents" in resp:
            raise QueryboostBatchHandlerError(
                f"The S3 prefix 's3://{self._bucket}/{self._prefix}' already contains files. "
                "Please specify an empty prefix or delete existing files before continuing."
            )

    def _flush(self) -> None:
        """Write buffered batches to a Parquet file in S3.

        Combines all batches in the buffer into a single PyArrow Table and uploads it
        to a sequentially numbered Parquet file (e.g., part-00000.parquet) in S3.
        """

        path = f"{self._bucket}/{self._prefix}/part-{self._write_idx:05d}.parquet"

        # Combine buffered batches into a single table
        table = pa.Table.from_batches(self._buffer)

        # Stream Parquet bytes directly to S3
        with self._fs.open_output_stream(path) as out:
            pq.write_table(table, out)
