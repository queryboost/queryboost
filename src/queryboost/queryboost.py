import json
import uuid
import logging
from typing import Optional
from pathlib import Path

import pyarrow.flight as flight

from .auth import ApiKeyClientAuth
from .types import BatchableData
from .utils import DataBatcher, validate_prompt
from .config import ConfigBuilder
from .stream import BatchStreamer
from .handlers import BatchHandler, LocalParquetBatchHandler

logger = logging.getLogger(__name__)
""" :meta private: """


DEFAULT_CACHE_DIR = "~/.cache/queryboost/"


class Queryboost:
    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs,
    ):
        """The ``Queryboost`` class is the main entry point for interacting with Queryboost via this Python SDK."""

        self._config = ConfigBuilder.build(
            api_key=api_key,
            url=url,
            port=port,
        )
        """ :meta private: """

        self._auth = ApiKeyClientAuth(self._config.api_key)
        """ :meta private: """

        self._location = f"{self._config.url}:{self._config.port}"
        """ :meta private: """

        self._client = flight.FlightClient(
            location=self._location,
            **kwargs,
        )
        """ :meta private: """

        # Authenticate with Flight handshake
        self._client.authenticate(self._auth)

    def run(
        self,
        data: BatchableData,
        prompt: str,
        name: str = str(uuid.uuid4()),
        num_gpus: int | None = None,
        batch_size: int = 32,
        batch_handler: Optional[BatchHandler] = None,
    ) -> None:
        """Process data with a prompt using Queryboost.

        This method streams input data to the Queryboost server for processing with a prompt template, then
        handles the results as they arrive. The prompt template should reference columns in the input data
        using Python string formatting syntax (e.g., "{column_name}").

        Args:
            data: The input data to process. Can be a Hugging Face Dataset, IterableDataset, list of dictionaries,
                or iterator of dictionaries.
            prompt: A string template that references columns in the input data.
            name: A unique identifier for this run. Used for organizing output files. Defaults to a random UUID.
            num_gpus: The number of GPUs to allocate for processing. If None, automatically selects based on workload.
                Defaults to None.
            batch_size: The number of rows to send to the server in each write operation. The server performs
                dynamic and continuous batching to maximize GPU utilization and throughput. Defaults to 32.
            batch_handler: A BatchHandler instance that determines how to handle the processed batches.
                Defaults to LocalParquetBatchHandler which saves results to parquet files.
        """

        if not batch_handler:
            batch_handler = LocalParquetBatchHandler(output_dir=Path(DEFAULT_CACHE_DIR).expanduser() / name)

        data_batcher = DataBatcher(data, batch_size)

        # Validate the prompt client-side before sending to the server for additional validation
        validate_prompt(prompt, data_batcher.schema.names)

        command = {
            "prompt": prompt,
            "num_gpus": num_gpus,
            "num_rows": data_batcher.num_rows,
        }
        descriptor = flight.FlightDescriptor.for_command(json.dumps(command).encode("utf-8"))

        batch_streamer = BatchStreamer(data_batcher, batch_handler)
        batch_streamer.stream(self._client, descriptor)
