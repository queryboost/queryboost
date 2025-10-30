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
    """Handler that saves batches to local Parquet files with buffering.

    This handler accumulates record batches in memory and writes them to Parquet files
    when the buffer exceeds ``target_write_bytes``. Each flush operation creates a separate
    Parquet file (part-00000.parquet, part-00001.parquet, etc.) in a directory organized by name.

    This buffering approach reduces file I/O overhead by combining multiple small batches
    into fewer, larger Parquet files.
    """

    def __init__(
        self,
        name: str,
        cache_dir: Path | str,
        target_write_bytes: int = 256 * 1024 * 1024,
        metadata: dict[str, Any] = {},
    ):
        """Initialize the local Parquet batch handler.

        Args:
            name: Name for this handler run. The final output directory will be cache_dir/name.
                Can include path separators for hierarchical organization (e.g., "prod/2025/my-run").
            cache_dir: Base cache directory path where Parquet files will be saved. The directory
                will be created if it doesn't exist.
            target_write_bytes: Target size in bytes for each Parquet file. When the
                accumulated buffer size exceeds this threshold, batches are written to
                a new Parquet file. Defaults to 256 MB.
            metadata: Optional metadata dictionary that will be stored with the handler.
                This can include any additional information needed for processing.
                Defaults to an empty dictionary.
        """

        super().__init__(name, target_write_bytes, metadata)

        # Construct final output path: cache_dir / name
        self._output_dir = Path(cache_dir) / self._name
        """ :meta private: """

        self._output_dir.mkdir(parents=True, exist_ok=True)

        if any(self._output_dir.iterdir()):
            logger.warning(f"Output directory {self._output_dir} already contains files")

        tqdm.write(f"Saving results as parquet files to: {self._output_dir}")
        tqdm.write("Pass a custom BatchHandler to save results elsewhere (e.g., database, S3)")
        tqdm.write("")  # Visual separation: batch handler setup complete, AI data processing begins

    def _flush(self) -> None:
        """Write buffered batches to a Parquet file.

        Combines all batches in the buffer into a single PyArrow Table and writes it
        to a sequentially numbered Parquet file (e.g., part-00000.parquet).
        """

        path = Path(self._output_dir) / f"part-{self._write_idx:05d}.parquet"

        table = pa.Table.from_batches(self._buffer)

        pq.write_table(table, path)
