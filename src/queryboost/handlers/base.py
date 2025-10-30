from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa

from queryboost.exceptions import QueryboostBatchHandlerError


class BatchHandler(ABC):
    """Base class for batch handlers with buffering support.

    Batch handlers define how to process record batches received from the Queryboost server.
    The base class implements buffering logic that accumulates batches in memory and flushes
    them when the buffer exceeds ``target_write_bytes``.

    Subclasses implement the ``_flush`` method to define how accumulated batches are written
    to their destination:

    - Saving to local storage (e.g., Parquet files)
    - Uploading to remote storage (e.g., S3)
    - Inserting into a database (e.g., PostgreSQL)
    - Streaming to another service
    - Custom processing logic

    The buffering approach reduces write overhead by combining multiple small record batches
    into fewer, larger writes. This is especially important for network-based destinations
    like S3 or databases.

    Queryboost provides built-in handlers like ``LocalParquetBatchHandler``, but you can also create
    custom handlers by subclassing ``BatchHandler`` and implementing the ``_flush`` method.
    """

    def __init__(
        self,
        name: str,
        target_write_bytes: int = 256 * 1024 * 1024,
        metadata: dict[str, Any] = {},
    ):
        """Initialize the batch handler.

        Args:
            name: Name for this handler run. Used to organize output files/keys.
                Can include path separators for hierarchical organization (e.g., "prod/2025/my-run").
            target_write_bytes: Target size in bytes for each flush operation. When the
                accumulated buffer size exceeds this threshold, batches are flushed to
                the destination. Defaults to 256 MB. Must be greater than 0.
            metadata: Optional metadata dictionary that will be stored with the handler.
                This can include any additional information needed for processing.
                Defaults to an empty dictionary.
        """

        if not name.strip():
            raise QueryboostBatchHandlerError("name cannot be empty.")

        if target_write_bytes < 1:
            raise QueryboostBatchHandlerError("target_write_bytes must be greater than 0.")

        self._name = name.strip("/")
        self._target_write_bytes = target_write_bytes
        self._metadata = metadata

        # Buffer state
        self._buffer: list[pa.RecordBatch] = []
        self._buffer_bytes: int = 0

        self._write_idx: int = 0

    def handle(self, batch: pa.RecordBatch) -> None:  # pragma: no cover
        """Handle a single record batch by adding it to the buffer.

        Batches are accumulated in memory until the buffer size exceeds ``target_write_bytes``,
        at which point all buffered batches are flushed to the destination via ``_flush()``.

        Args:
            batch: The PyArrow RecordBatch to process.
        """

        self._buffer.append(batch)
        self._buffer_bytes += batch.nbytes

        if self._buffer_bytes >= self._target_write_bytes:
            self._flush()
            self._reset_buffer()
            self._write_idx += 1

    def _reset_buffer(self) -> None:
        """Reset the buffer."""

        self._buffer.clear()
        self._buffer_bytes = 0

    def close(self) -> None:
        """Flush any remaining buffered data and reset the buffer."""

        if self._buffer:
            self._flush()
            self._reset_buffer()

    @abstractmethod
    def _flush(self) -> None:  # pragma: no cover
        """Flush the buffered batches to the destination.

        This method is called when the buffer exceeds ``target_write_bytes`` or when
        ``close()`` is called with remaining data. Subclasses must implement this method
        to define how accumulated batches in ``self._buffer`` are written to their
        specific destination (e.g., disk, S3, database).

        The implementation can access:
        - ``self._name``: Name for this handler run
        - ``self._buffer``: List of PyArrow RecordBatches to write
        - ``self._write_idx``: Sequential index for naming/tracking flush operations
        - ``self._metadata``: User-provided metadata dictionary
        """
        pass
