import json
import queue
import threading
from typing import Any

import pyarrow as pa
import pyarrow.flight as flight
from tqdm import tqdm

from queryboost.utils import DataBatcher
from queryboost.handlers import BatchHandler
from queryboost.exceptions import QueryboostServerError, clean_flight_error_message

# Events to skip when calling tqdm.write()
_EVENTS_TO_SKIP_TQDM_WRITE = frozenset(["processing_started"])


class BatchStreamer:
    """Coordinates bidirectional streaming between client and server.

    Manages concurrent sending of batches to the server and receiving of processed results,
    with real-time progress tracking and exception handling.

    :meta private:
    """

    def __init__(
        self,
        data_batcher: DataBatcher,
        batch_handler: BatchHandler,
    ):
        self._data_batcher = data_batcher
        self._schema = self._data_batcher.schema
        self._num_rows = self._data_batcher.num_rows

        self._batch_handler = batch_handler

        # Initialize thread-safe queues for tracking progress and exceptions
        self._progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._exception_queue: queue.Queue[Exception] = queue.Queue()

        self._pbar: tqdm | None = None

        self._stop_event = threading.Event()

    def _write(self, writer: flight.FlightStreamWriter) -> None:
        """Write record batches to the server.

        Sends the schema followed by each batch sequentially, emitting progress events
        for tracking. Responds to stop signals for graceful shutdown.

        Args:
            writer: The Flight stream writer to send batches through
        """

        try:
            writer.begin(self._schema)

            for batch in self._data_batcher:
                if self._stop_event.is_set():
                    break

                record_batch = pa.RecordBatch.from_pydict(batch)
                writer.write(record_batch)

                num_rows = record_batch.num_rows
                self._progress_queue.put({"event": "write", "num_rows": num_rows})

        except Exception as e:
            if self._exception_queue.empty():
                self._exception_queue.put(e)

        finally:
            writer.done_writing()
            self._progress_queue.put({"event": "done_writing"})

    def _read(self, reader: flight.FlightStreamReader) -> None:
        """Read and process record batches from the server.

        Handles chunks containing metadata, record batches, or both. Processes batches
        through the handler and emits progress events. Responds to stop signals.

        Args:
            reader: The Flight stream reader to read chunks from
        """

        try:
            for chunk in reader:
                if self._stop_event.is_set():
                    break

                if metadata := chunk.app_metadata:
                    metadata = json.loads(metadata.to_pybytes().decode("utf-8"))
                    self._progress_queue.put(metadata)

                if record_batch := chunk.data:
                    self._batch_handler.handle(record_batch)

                    num_rows = record_batch.num_rows
                    self._progress_queue.put({"event": "read", "num_rows": num_rows})

        except StopIteration:
            # StopIteration is raised when the stream is finished
            pass

        except Exception as e:
            if self._exception_queue.empty():
                self._exception_queue.put(e)

        finally:
            self._progress_queue.put({"event": "done_reading"})

    def _track_progress_and_exceptions(self) -> None:
        """Monitor streaming progress and surface exceptions from worker threads.

        Runs until both send and receive operations complete, displaying real-time
        progress via a progress bar and propagating any worker thread exceptions.
        """

        done_writing = False
        done_reading = False
        sent_num_rows = 0
        received_num_rows = 0

        while not (done_writing and done_reading):
            try:
                item = self._progress_queue.get(timeout=0.2)

                event = item.get("event")
                message = item.get("message")
                num_rows = item.get("num_rows", 0)

                if event == "write":
                    sent_num_rows += num_rows

                    if self._pbar is not None:
                        self._pbar.set_postfix_str(f"Sent: {sent_num_rows}")

                elif event == "read":
                    received_num_rows += num_rows

                    if self._pbar is not None:
                        self._pbar.update(num_rows)

                elif event == "done_writing":
                    done_writing = True
                elif event == "done_reading":
                    done_reading = True
                elif event == "processing_started":
                    # Initialize the progress bar when the server indicates processing started state
                    self._pbar = tqdm(
                        total=self._num_rows,
                        desc=message,
                    )
                    self._pbar.set_postfix_str(f"Sent: {sent_num_rows}")

                elif event == "processing_done":
                    if self._pbar is not None:
                        self._pbar.close()

                if message and event not in _EVENTS_TO_SKIP_TQDM_WRITE:
                    tqdm.write(message)

                if not self._exception_queue.empty():
                    e = self._exception_queue.get()

                    if isinstance(e, flight.FlightError):
                        error_message = clean_flight_error_message(e)

                        raise QueryboostServerError(error_message) from None

                    else:
                        raise e

            except queue.Empty:
                continue

    def stream(
        self,
        client: flight.FlightClient,
        descriptor: flight.FlightDescriptor,
    ) -> None:
        """Stream record batches bidirectionally with the server.

        Starts concurrent threads for sending and receiving batches while tracking
        progress. Handles interruptions gracefully by signaling threads to stop
        and ensuring proper resource cleanup.

        Args:
            client: The Flight client instance to use for communication
            descriptor: Flight descriptor containing metadata about the stream
        """

        writer, reader = client.do_exchange(descriptor)

        write_thread = threading.Thread(target=self._write, kwargs={"writer": writer})
        read_thread = threading.Thread(target=self._read, kwargs={"reader": reader})

        write_thread.start()
        read_thread.start()

        try:
            self._track_progress_and_exceptions()

        finally:
            self._batch_handler.close()

            self._stop_event.set()

            if self._pbar is not None:
                self._pbar.close()

            reader.cancel()

            write_thread.join()
            read_thread.join()
