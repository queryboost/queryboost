from typing import Any, Callable, Iterator, Generator
from itertools import islice
from collections import defaultdict

import pyarrow as pa
from datasets import Dataset, IterableDataset
from more_itertools import peekable  # pyright: ignore

from queryboost.types import BatchableData
from queryboost.exceptions import (
    QueryboostDataTypeError,
    QueryboostBatchSizeError,
    QueryboostDataColumnError,
)

_RESERVED_COLUMN_NAMES = frozenset(["_inference", "_error"])


class DataBatcher:
    def __init__(
        self,
        data: BatchableData,
        batch_size: int = 16,
    ):
        """The ``DataBatcher`` class generates batches from ``BatchableData``.

        This class handles batching from ``BatchableData`` types including:
        - Hugging Face Datasets
        - Hugging Face IterableDatasets
        - Lists of dictionaries (for small datasets that fit into memory)
        - Iterators of dictionaries (for large datasets that don't fit into memory and should be streamed)

        The DataBatcher can only be iterated once since the underlying iterator is a generator that gets consumed.

        Args:
            data: The data source to batch. Must be a ``BatchableData`` type.
            batch_size: The size of batches to yield. Defaults to 16.

        Attributes:
            schema: The PyArrow schema of the data, inferred from the input data source.
        """

        if batch_size < 1:
            raise QueryboostBatchSizeError("Batch size must be greater than 0.")

        if isinstance(data, list):
            self._data = Dataset.from_list(data)
        else:
            self._data = data

        self._batch_size = batch_size

        if isinstance(self._data, Dataset):
            _batch_generator_func: Callable[
                ...,
                Generator[dict[str, list[Any]], None, None],
            ] = self._batch_dataset
            self.schema = self._data.data.table.schema
            self.num_rows = self._data.num_rows

        elif isinstance(self._data, IterableDataset):
            _batch_generator_func = self._batch_iterable_dataset
            self.schema = self._data.features.arrow_schema  # pyright: ignore

            # For Hugging Face IterableDataset, check the metadata in the info attribute
            try:
                split = self._data.split
                self.num_rows = self._data.info.splits[split].num_examples  # type: ignore[reportOptionalSubscript]
            except (AttributeError, KeyError, TypeError):
                self.num_rows = None

        elif isinstance(self._data, Iterator):
            _batch_generator_func = self._batch_iterator

            # Peek at the first element to get the schema
            self._data = peekable(self._data)
            first_element = self._data.peek()
            self.schema = pa.Table.from_pylist([first_element]).schema
            self.num_rows = None  # TODO: allow users to set the number of rows

        else:
            raise QueryboostDataTypeError(
                f"Unsupported data type: {type(self._data)}. Please use one of the following supported types: {BatchableData}."
            )

        if _RESERVED_COLUMN_NAMES.intersection(self.schema.names):
            raise QueryboostDataColumnError(
                f"Reserved column names are not allowed: {', '.join(_RESERVED_COLUMN_NAMES)}."
            )

        self._batch_generator = _batch_generator_func(
            self._data,
            self._batch_size,
        )

    @staticmethod
    def _batch_dataset(
        data: Dataset,
        batch_size: int,
    ) -> Generator[dict[str, list[Any]], None, None]:
        """Batch generator for Hugging Face Dataset.

        :meta private:
        """

        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    @staticmethod
    def _batch_iterable_dataset(
        data: IterableDataset,
        batch_size: int,
    ) -> Generator[dict[str, list[Any]], None, None]:
        """Batch generator for Hugging Face IterableDataset.

        :meta private:
        """

        for batch in data.iter(batch_size=batch_size):
            yield batch

    @staticmethod
    def _batch_iterator(
        data: Iterator,
        batch_size: int,
    ) -> Generator[dict[str, list[Any]], None, None]:
        """Batch generator for Iterator.

        :meta private:
        """

        while batch := list(islice(data, batch_size)):
            yield invert_list_of_dicts(batch)

    def __iter__(self) -> Generator[dict[str, list[Any]], None, None]:
        return self._batch_generator

    def __next__(self) -> dict[str, list[Any]]:
        return next(self._batch_generator)


def invert_list_of_dicts(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Invert a list of dicts into a dict of lists.

    This is a simple and efficient approach that works well for small to moderately sized lists.
    The keys are taken from the first dict in the list and used for all subsequent dicts.

    Args:
        data: list of dicts

    Returns:
        dict of lists

    :meta private:
    """

    if not data:
        return {}

    # Use the keys from the first dict
    keys = data[0].keys()

    inverse_data = defaultdict(list)
    for row in data:
        for key in keys:
            inverse_data[key].append(row[key])

    return dict(inverse_data)
