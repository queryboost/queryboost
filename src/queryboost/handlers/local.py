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
        target_write_bytes: int = 256 * 1024 * 1024,
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

        super().__init__(target_write_bytes, metadata)

        self._output_dir = Path(output_dir)
        """ :meta private: """

        self._output_dir.mkdir(parents=True, exist_ok=True)

        if any(self._output_dir.iterdir()):
            logger.warning(
                f"Output directory {self._output_dir} already contains files"
            )

        tqdm.write(f"Saving results as parquet files to: {self._output_dir}")
        tqdm.write(
            "Pass a custom BatchHandler to save results elsewhere (e.g., database, S3)"
        )
        tqdm.write(
            ""
        )  # Visual separation: batch handler setup complete, AI data processing begins

    def _flush(self) -> None:

        output_file = Path(self._output_dir) / f"part-{self._write_idx:05d}.parquet"
        table = pa.Table.from_batches(self._buffer)
        pq.write_table(table, output_file)
