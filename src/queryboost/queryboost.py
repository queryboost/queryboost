import json
import uuid
import logging
from typing import Any, Optional
from pathlib import Path

import certifi
import pyarrow.flight as flight

from .auth import ApiKeyClientAuth
from .types import BatchableData
from .utils import DataBatcher, validate_prompt
from .config import ConfigBuilder
from .stream import BatchStreamer
from .handlers import BatchHandler, LocalParquetBatchHandler
from .exceptions import QueryboostError

logger = logging.getLogger(__name__)
""" :meta private: """


DEFAULT_CACHE_DIR = "~/.cache/queryboost/"


class Queryboost:
    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = None,
        flight_client_kwargs: Optional[dict[str, Any]] = None,
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

        self._flight_client_kwargs = flight_client_kwargs or {}
        """ :meta private: """

        self._connect()

    def _connect(self) -> None:
        """Connect to the Queryboost server and authenticate with the Flight handshake.

        :meta private:
        """

        # Load the trusted CA bundle (PEM-encoded) used for TLS certificate verification
        # certifi provides Mozilla's root CA bundle, ensuring consistent trust across platforms
        with open(certifi.where(), "rb") as f:
            tls_root_certs = f.read()

        self._client = flight.FlightClient(
            location=self._location,
            tls_root_certs=tls_root_certs,
            **self._flight_client_kwargs,
        )

        # Authenticate with Flight handshake
        self._client.authenticate(self._auth)

    def run(
        self,
        data: BatchableData,
        prompt: str,
        name: Optional[str] = None,
        num_gpus: Optional[int] = None,
        batch_size: int = 16,
        batch_handler: Optional[BatchHandler] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> None:
        """Process data with a prompt using Queryboost.

        This method streams input data to the Queryboost server for processing with a prompt template, then
        handles the results as they arrive. The prompt template should reference columns in the input data
        using Python string formatting syntax (e.g., "{column_name}").

        Args:
            data: The input data to process. Can be a Hugging Face Dataset, IterableDataset, list of dictionaries,
                or iterator of dictionaries.
            prompt: A string template that references columns in the input data.
            name: Name for this run. Used with default LocalParquetBatchHandler to organize output files.
                Mutually exclusive with ``batch_handler``.
            num_gpus: The number of GPUs to allocate for processing. If None, automatically selects based on workload.
                Defaults to None.
            batch_size: The number of rows to send to the server in each write operation. The server performs
                dynamic and continuous batching to maximize GPU utilization and throughput. Defaults to 16.
            batch_handler: Custom batch handler instance. Mutually exclusive with ``name``.
            json_schema: Optional JSON schema dict that specifies the schema for the structured output. Must follow
                the JSON Schema specification. For Queryboost, the schema must include ``additionalProperties: false``
                and all properties must be listed in the ``required`` array. If not provided, a recommended schema
                will be inferred based on the prompt. Defaults to None.

                Example::

                    {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "additionalProperties": false,
                        "required": ["answer"],
                    }

        Raises:
            QueryboostError: If both ``name`` and ``batch_handler`` are specified.
        """

        if name is not None and batch_handler is not None:
            raise QueryboostError("Cannot specify both 'name' and 'batch_handler'.")

        if batch_handler is None:
            batch_handler = LocalParquetBatchHandler(
                name=name or str(uuid.uuid4()),
                cache_dir=Path(DEFAULT_CACHE_DIR).expanduser(),
            )

        data_batcher = DataBatcher(data, batch_size)

        # Validate the prompt client-side before sending to the server for additional validation
        validate_prompt(prompt, data_batcher.schema.names)

        command = {
            "prompt": prompt,
            "num_gpus": num_gpus,
            "num_rows": data_batcher.num_rows,
            "json_schema": json_schema,
        }
        descriptor = flight.FlightDescriptor.for_command(json.dumps(command).encode("utf-8"))

        batch_streamer = BatchStreamer(data_batcher, batch_handler)

        self._client.wait_for_available()

        batch_streamer.stream(self._client, descriptor)
