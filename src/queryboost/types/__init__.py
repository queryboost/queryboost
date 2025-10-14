from typing import Any, Union, Iterator, TypeAlias

from datasets import Dataset, IterableDataset

BatchableData: TypeAlias = Union[
    Dataset,
    IterableDataset,
    list[dict[str, Any]],
    Iterator[dict[str, Any]],
]


__all__ = ["BatchableData"]
