from typing import Any, Union, Callable, Iterator, TypeAlias
from dataclasses import dataclass

from datasets import Dataset, IterableDataset

BatchableData: TypeAlias = Union[
    Dataset,
    IterableDataset,
    list[dict[str, Any]],
    Iterator[dict[str, Any]],
]


@dataclass
class ProgressEvent:
    """Progress update emitted during a Queryboost run.

    Attributes:
        rows_sent: Number of rows sent to the server so far.
        rows_received: Number of result rows received so far.
        total_rows: Total number of rows to process, or None if unknown (e.g. iterator input).
        event: The event type that triggered this update (e.g. ``"write"``, ``"read"``,
            ``"processing_started"``, ``"done_reading"``, ``"failed"``).
        message: Optional message from the server.
        error: Error message if event is ``"failed"``, otherwise None.
    """

    rows_sent: int
    rows_received: int
    total_rows: int | None
    event: str | None
    message: str | None = None
    error: str | None = None


ProgressCallback: TypeAlias = Callable[[ProgressEvent], None]


__all__ = ["BatchableData", "ProgressEvent", "ProgressCallback"]
