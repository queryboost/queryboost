from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa

from queryboost.exceptions import QueryboostError


class BatchHandler(ABC):
    """Base class for batch handlers.

    Batch handlers define how to process record batches received from the Queryboost server.

    Each batch handler implements a ``handle`` method that processes a single record batch. The handler
    can perform any operation on the batch, such as:

    - Saving to local storage (e.g., Parquet files)
    - Uploading to remote storage (e.g., S3)
    - Inserting into a database (e.g., PostgreSQL)
    - Streaming to another service
    - Custom processing logic

    Queryboost provides built-in handlers like ``LocalParquetBatchHandler``, but you can also create
    custom handlers by subclassing ``BatchHandler`` and implementing the ``handle`` method.
    """

    def __init__(
        self,
        target_write_bytes: int = 256 * 1024 * 1024,
        metadata: dict[str, Any] = {},
    ):
        """Initialize the batch handler.

        Args:
            metadata: Optional metadata dictionary that will be stored with the handler.
                This can include any additional information needed for processing.
                Defaults to an empty dictionary.
        """

        if target_write_bytes < 1:
            raise QueryboostError("Target write bytes must be greater than 0.")

        self._target_write_bytes = target_write_bytes
        self._metadata = metadata

        # Buffer state
        self._buffer: list[pa.RecordBatch] = []
        self._buffer_bytes: int = 0

        self._write_idx: int = 0

    def handle(
        self,
        batch: pa.RecordBatch,
        batch_idx: int,
    ) -> None:  # pragma: no cover

        self._buffer.append(batch)
        self._buffer_bytes += batch.nbytes

        if self._buffer_bytes >= self._target_write_bytes:
            self._flush()

            self._buffer.clear()
            self._buffer_bytes = 0

            self._write_idx += 1

    def close(self) -> None:
        """Flush any remaining buffered data. Call when you're done reading batches."""

        if self._buffer:
            self._flush()

    @abstractmethod
    def _flush(self) -> None:  # pragma: no cover
        pass
