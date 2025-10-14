from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa


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

    def __init__(self, metadata: dict[str, Any] = {}):
        """Initialize the batch handler.

        Args:
            metadata: Optional metadata dictionary that will be stored with the handler.
                This can include any additional information needed for processing.
                Defaults to an empty dictionary.
        """

        self._metadata = metadata
        """ :meta private: """

    @abstractmethod
    def handle(
        self,
        batch: pa.RecordBatch,
        batch_idx: int,
    ) -> None:  # pragma: no cover
        """Process a single record batch received from the Queryboost server.

        This abstract method must be implemented by concrete handler classes to define
        how record batches should be processed. The implementation can perform operations
        like saving to storage, uploading to databases, or custom processing logic.

        Args:
            batch: PyArrow RecordBatch containing the data to be processed. The batch
                structure determines the schema of the data.
            batch_idx: Integer index of the batch in the sequence of batches being processed.
                Used to track the order and position of batches.
        """

        pass
