import logging
from typing import Any
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .base import BatchHandler

logger = logging.getLogger(__name__)
""" :meta private: """


class LocalParquetBatchHandler(BatchHandler):
    """Handler that saves batches to local Parquet files.

    This handler processes record batches by saving each one to a separate Parquet file
    in a specified local directory.
    """

    def __init__(
        self,
        output_dir: Path | str,
        metadata: dict[str, Any] = {},
    ):
        """Initialize the local Parquet batch handler.

        Args:
            output_dir: Directory path where Parquet files will be saved. The directory
                will be created if it doesn't exist.
            metadata: Optional metadata dictionary that will be stored with the handler.
                This can include any additional information needed for processing.
                Defaults to an empty dictionary.
        """

        super().__init__(metadata)

        self._output_dir = Path(output_dir)
        """ :meta private: """

        self._output_dir.mkdir(parents=True, exist_ok=True)

        if any(self._output_dir.iterdir()):
            logger.warning(f"Output directory {self._output_dir} already contains files")

        tqdm.write(f"Saving results as parquet files to: {self._output_dir}")
        tqdm.write("Pass a custom BatchHandler to save results elsewhere (e.g., database, S3)")
        tqdm.write("")  # Visual separation: batch handler setup complete, AI data processing begins

    def handle(
        self,
        batch: pa.RecordBatch,
        batch_idx: int,
    ) -> None:
        """Process and save a record batch to a Parquet file.

        This method takes a PyArrow RecordBatch and saves it to a Parquet file in the
        configured output directory. The filename is generated using the batch_idx.

        Args:
            batch: PyArrow RecordBatch containing the data to be saved. The batch
                structure determines the schema of the resulting Parquet file.
            batch_idx: Integer index of the batch in the sequence of batches being processed.
                Used to track the order and position of batches.
        """

        output_file = Path(self._output_dir) / f"batch_{batch_idx}.parquet"
        pq.write_table(pa.Table.from_batches([batch]), output_file)
