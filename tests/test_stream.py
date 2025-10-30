import queue
from unittest.mock import Mock, patch

import pytest
import pyarrow as pa
import pyarrow.flight as flight

from queryboost.utils import DataBatcher
from queryboost.stream import BatchStreamer
from queryboost.handlers import BatchHandler


class MockBatchHandler(BatchHandler):
    """Mock batch handler for testing."""

    def __init__(self, name: str = "test-stream"):
        super().__init__(name)
        self.handled_batches: list[pa.RecordBatch] = []

    def _flush(self):
        """Store flushed batches for verification."""
        self.handled_batches.extend(self._buffer)


class TestBatchStreamer:
    """Test suite for BatchStreamer class."""

    def test_initialization(self):
        """Test BatchStreamer initialization."""
        data = [{"x": 1}, {"x": 2}]
        data_batcher = DataBatcher(data, batch_size=2)
        handler = MockBatchHandler()

        streamer = BatchStreamer(data_batcher, handler)

        assert streamer._data_batcher == data_batcher
        assert streamer._batch_handler == handler
        assert streamer._schema == data_batcher.schema
        assert streamer._num_rows == 2
        assert isinstance(streamer._progress_queue, queue.Queue)
        assert isinstance(streamer._exception_queue, queue.Queue)
        assert streamer._pbar is None

    def test_write_sends_batches(self):
        """Test _write method sends batches to writer."""
        data = [{"val": i} for i in range(5)]
        data_batcher = DataBatcher(data, batch_size=2)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Mock writer
        mock_writer = Mock(spec=flight.FlightStreamWriter)

        streamer._write(mock_writer)

        # Verify writer.begin was called with schema
        mock_writer.begin.assert_called_once_with(streamer._schema)

        # Verify writer.write was called for each batch
        assert mock_writer.write.call_count == 3  # 2 + 2 + 1

        # Verify done_writing was called
        mock_writer.done_writing.assert_called_once()

        # Verify queue received write events
        events = []
        while not streamer._progress_queue.empty():
            events.append(streamer._progress_queue.get())

        write_events = [e for e in events if e["event"] == "write"]
        assert len(write_events) == 3
        assert any(e["event"] == "done_writing" for e in events)

    def test_read_processes_batches(self):
        """Test _read method processes batches from reader."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        # Use small target to trigger flush for each batch
        handler = MockBatchHandler()
        handler._target_write_bytes = 1
        streamer = BatchStreamer(data_batcher, handler)

        # Create mock reader with batches
        batch1 = pa.RecordBatch.from_pydict({"result": [1, 2]})
        batch2 = pa.RecordBatch.from_pydict({"result": [3, 4]})

        chunk1 = Mock()
        chunk1.data = batch1
        chunk1.app_metadata = None

        chunk2 = Mock()
        chunk2.data = batch2
        chunk2.app_metadata = None

        mock_reader = Mock(spec=flight.FlightStreamReader)
        mock_reader.__iter__ = Mock(return_value=iter([chunk1, chunk2]))

        streamer._read(mock_reader)

        # Verify handler received both batches after flush
        assert len(handler.handled_batches) == 2

        # Verify queue received read events
        events = []
        while not streamer._progress_queue.empty():
            events.append(streamer._progress_queue.get())

        read_events = [e for e in events if e["event"] == "read"]
        assert len(read_events) == 2
        assert any(e["event"] == "done_reading" for e in events)

    def test_read_handles_server_messages(self):
        """Test _read processes server messages from metadata."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Create chunk with metadata (server message)
        ready_chunk = Mock()
        ready_chunk.data = None
        ready_chunk.app_metadata = Mock()
        ready_chunk.app_metadata.to_pybytes.return_value = (
            b'{"event": "processing_started", "message": "Processing started"}'
        )

        # Create chunk with data
        data_chunk = Mock()
        data_chunk.data = pa.RecordBatch.from_pydict({"result": [1]})
        data_chunk.app_metadata = None

        mock_reader = Mock(spec=flight.FlightStreamReader)
        mock_reader.__iter__ = Mock(return_value=iter([ready_chunk, data_chunk]))

        streamer._read(mock_reader)

        # Verify metadata was queued
        events = []
        while not streamer._progress_queue.empty():
            events.append(streamer._progress_queue.get())

        assert any(e.get("event") == "processing_started" for e in events)

    def test_read_handles_stop_iteration(self):
        """Test _read handles StopIteration gracefully."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        mock_reader = Mock(spec=flight.FlightStreamReader)
        mock_reader.__iter__ = Mock(side_effect=StopIteration)

        # Should not raise exception
        streamer._read(mock_reader)

        # Should have done_reading event
        events = []
        while not streamer._progress_queue.empty():
            events.append(streamer._progress_queue.get())

        assert any(e["event"] == "done_reading" for e in events)

    def test_track_progress_monitors_queue(self):
        """Test _track_progress monitors the queue correctly."""
        data = [{"x": i} for i in range(10)]
        data_batcher = DataBatcher(data, batch_size=5)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Simulate progress events
        streamer._progress_queue.put({"event": "write", "num_rows": 5})
        streamer._progress_queue.put({"event": "write", "num_rows": 5})
        streamer._progress_queue.put({"event": "done_writing"})
        streamer._progress_queue.put({"event": "processing_started", "message": "Processing"})
        streamer._progress_queue.put({"event": "read", "num_rows": 5})
        streamer._progress_queue.put({"event": "read", "num_rows": 5})
        streamer._progress_queue.put({"event": "done_reading"})

        with patch("queryboost.stream.tqdm") as mock_tqdm:
            with patch("queryboost.stream.tqdm.write"):
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                streamer._track_progress_and_exceptions()

                # Verify progress bar was created
                mock_tqdm.assert_called_once()

                # Verify updates were called
                assert mock_pbar.update.call_count == 2  # Two read events

    def test_stream_creates_threads(self):
        """Test stream method creates and manages threads."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Mock client and descriptor
        mock_client = Mock()
        mock_writer = Mock()
        mock_reader = Mock()
        mock_reader.__iter__ = Mock(return_value=iter([]))
        mock_client.do_exchange.return_value = (mock_writer, mock_reader)

        mock_descriptor = Mock(spec=flight.FlightDescriptor)

        with patch.object(streamer, "_track_progress_and_exceptions") as mock_track:
            with patch("queryboost.stream.threading.Thread") as mock_thread_cls:
                mock_write_thread = Mock()
                mock_read_thread = Mock()
                mock_thread_cls.side_effect = [mock_write_thread, mock_read_thread]

                streamer.stream(mock_client, mock_descriptor)

                # Verify do_exchange was called
                mock_client.do_exchange.assert_called_once_with(mock_descriptor)

                # Verify threads were created
                assert mock_thread_cls.call_count == 2

                # Verify threads were started
                mock_write_thread.start.assert_called_once()
                mock_read_thread.start.assert_called_once()

                # Verify threads were joined
                mock_write_thread.join.assert_called_once()
                mock_read_thread.join.assert_called_once()

                # Verify progress tracking was called
                mock_track.assert_called_once()

    def test_queue_is_thread_safe(self):
        """Test that the queues are thread-safe."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Queues should be thread-safe Queues
        assert isinstance(streamer._progress_queue, queue.Queue)
        assert isinstance(streamer._exception_queue, queue.Queue)

    def test_write_converts_batches_to_record_batches(self):
        """Test that _write converts dict batches to PyArrow RecordBatches."""
        data = [{"text": "a"}, {"text": "b"}]
        data_batcher = DataBatcher(data, batch_size=2)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        mock_writer = Mock(spec=flight.FlightStreamWriter)

        streamer._write(mock_writer)

        # Verify write was called with RecordBatch
        write_calls = mock_writer.write.call_args_list
        assert len(write_calls) == 1

        written_batch = write_calls[0][0][0]
        assert isinstance(written_batch, pa.RecordBatch)
        assert written_batch.num_rows == 2

    def test_track_progress_handles_queue_empty_timeout(self):
        """Test _track_progress handles queue.Empty timeout gracefully."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Mock the queue to simulate Empty exception followed by done events
        call_count = 0

        def mock_get(timeout=None):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First two calls raise Empty to test the continue path
                raise queue.Empty()
            elif call_count == 3:
                return {"event": "done_writing"}
            else:
                return {"event": "done_reading"}

        with patch.object(streamer._progress_queue, "get", side_effect=mock_get):
            with patch("queryboost.stream.tqdm"):
                streamer._track_progress_and_exceptions()

    def test_track_progress_writes_messages(self):
        """Test _track_progress writes messages using tqdm.write for non-skipped events."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Add event with message that should be written
        streamer._progress_queue.put({"event": "custom_event", "message": "Custom message"})
        streamer._progress_queue.put({"event": "done_writing"})
        streamer._progress_queue.put({"event": "done_reading"})

        with patch("queryboost.stream.tqdm") as mock_tqdm_cls:
            with patch("queryboost.stream.tqdm.write") as mock_tqdm_write:
                mock_pbar = Mock()
                mock_tqdm_cls.return_value = mock_pbar

                streamer._track_progress_and_exceptions()

                # Verify tqdm.write was called with the message
                mock_tqdm_write.assert_called_with("Custom message")

    def test_write_stops_on_stop_event(self):
        """Test _write stops when stop_event is set."""
        data = [{"val": i} for i in range(10)]
        data_batcher = DataBatcher(data, batch_size=2)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        mock_writer = Mock(spec=flight.FlightStreamWriter)

        # Set stop event after first batch
        def set_stop_after_first(*args, **kwargs):  # noqa: ARG001
            streamer._stop_event.set()

        mock_writer.write.side_effect = set_stop_after_first

        streamer._write(mock_writer)

        # Should only write once before stopping
        assert mock_writer.write.call_count == 1
        mock_writer.done_writing.assert_called_once()

    def test_write_handles_exception(self):
        """Test _write handles exceptions and queues them."""
        data = [{"val": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        mock_writer = Mock(spec=flight.FlightStreamWriter)
        test_error = Exception("Write failed")
        mock_writer.write.side_effect = test_error

        streamer._write(mock_writer)

        # Exception should be queued
        assert not streamer._exception_queue.empty()
        queued_error = streamer._exception_queue.get()
        assert queued_error == test_error

    def test_read_stops_on_stop_event(self):
        """Test _read stops when stop_event is set."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        # Use small target to trigger flush immediately
        handler._target_write_bytes = 1
        streamer = BatchStreamer(data_batcher, handler)

        batch1 = pa.RecordBatch.from_pydict({"result": [1]})
        batch2 = pa.RecordBatch.from_pydict({"result": [2]})

        chunk1 = Mock()
        chunk1.data = batch1
        chunk1.app_metadata = None

        chunk2 = Mock()
        chunk2.data = batch2
        chunk2.app_metadata = None

        # Set stop event after first chunk
        def chunk_iter():
            yield chunk1
            streamer._stop_event.set()
            yield chunk2

        mock_reader = Mock(spec=flight.FlightStreamReader)
        mock_reader.__iter__ = Mock(return_value=chunk_iter())

        streamer._read(mock_reader)

        # Should only process first batch
        assert len(handler.handled_batches) == 1

    def test_read_handles_exception(self):
        """Test _read handles exceptions and queues them."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        test_error = Exception("Read failed")
        mock_reader = Mock(spec=flight.FlightStreamReader)
        mock_reader.__iter__ = Mock(side_effect=test_error)

        streamer._read(mock_reader)

        # Exception should be queued
        assert not streamer._exception_queue.empty()
        queued_error = streamer._exception_queue.get()
        assert queued_error == test_error

    def test_track_progress_raises_flight_error_as_server_error(self):
        """Test _track_progress_and_exceptions wraps FlightError as QueryboostServerError."""
        from queryboost.exceptions import QueryboostServerError

        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Queue a FlightError
        flight_error = flight.FlightServerError("Flight error: Test error")
        streamer._exception_queue.put(flight_error)
        streamer._progress_queue.put({"event": "done_writing"})
        streamer._progress_queue.put({"event": "done_reading"})

        with patch("queryboost.stream.tqdm"):
            with pytest.raises(QueryboostServerError) as exc_info:
                streamer._track_progress_and_exceptions()

            assert "Test error" in str(exc_info.value)
            assert "Flight error:" not in str(exc_info.value)

    def test_track_progress_raises_non_flight_errors(self):
        """Test _track_progress_and_exceptions re-raises non-FlightError exceptions."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Queue a regular exception
        test_error = ValueError("Some error")
        streamer._exception_queue.put(test_error)
        streamer._progress_queue.put({"event": "done_writing"})
        streamer._progress_queue.put({"event": "done_reading"})

        with patch("queryboost.stream.tqdm"):
            with pytest.raises(ValueError) as exc_info:
                streamer._track_progress_and_exceptions()

            assert str(exc_info.value) == "Some error"

    def test_stream_closes_progress_bar_on_keyboard_interrupt(self):
        """Test that stream closes progress bar when KeyboardInterrupt is raised."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        mock_client = Mock()
        mock_writer = Mock()
        mock_reader = Mock()
        mock_client.do_exchange.return_value = (mock_writer, mock_reader)
        mock_descriptor = Mock(spec=flight.FlightDescriptor)

        # Create a mock progress bar
        mock_pbar = Mock()
        streamer._pbar = mock_pbar

        with patch.object(streamer, "_track_progress_and_exceptions", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                streamer.stream(mock_client, mock_descriptor)

        # Verify progress bar was closed
        mock_pbar.close.assert_called_once()

    def test_stream_closes_progress_bar_on_exception(self):
        """Test that stream closes progress bar when exception is raised."""
        data = [{"x": 1}]
        data_batcher = DataBatcher(data, batch_size=1)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        mock_client = Mock()
        mock_writer = Mock()
        mock_reader = Mock()
        mock_client.do_exchange.return_value = (mock_writer, mock_reader)
        mock_descriptor = Mock(spec=flight.FlightDescriptor)

        # Create a mock progress bar
        mock_pbar = Mock()
        streamer._pbar = mock_pbar

        test_error = ValueError("Test error")
        with patch.object(streamer, "_track_progress_and_exceptions", side_effect=test_error):
            with pytest.raises(ValueError):
                streamer.stream(mock_client, mock_descriptor)

        # Verify progress bar was closed even on exception
        mock_pbar.close.assert_called_once()

    def test_track_progress_updates_postfix_on_write_after_pbar_init(self):
        """Test _track_progress updates postfix when write events occur after pbar is initialized."""
        data = [{"x": i} for i in range(10)]
        data_batcher = DataBatcher(data, batch_size=5)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Simulate progress events with write happening AFTER processing_started
        streamer._progress_queue.put({"event": "processing_started", "message": "AI processing"})
        streamer._progress_queue.put({"event": "write", "num_rows": 5})
        streamer._progress_queue.put({"event": "write", "num_rows": 5})
        streamer._progress_queue.put({"event": "done_writing"})
        streamer._progress_queue.put({"event": "read", "num_rows": 5})
        streamer._progress_queue.put({"event": "read", "num_rows": 5})
        streamer._progress_queue.put({"event": "done_reading"})

        with patch("queryboost.stream.tqdm") as mock_tqdm:
            with patch("queryboost.stream.tqdm.write"):
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                streamer._track_progress_and_exceptions()

                # Verify progress bar was created
                mock_tqdm.assert_called_once()

                # Verify set_postfix_str was called on processing_started and write events
                assert mock_pbar.set_postfix_str.call_count == 3
                mock_pbar.set_postfix_str.assert_any_call("Sent: 0")  # Initial call on processing_started
                mock_pbar.set_postfix_str.assert_any_call("Sent: 5")  # First write event
                mock_pbar.set_postfix_str.assert_any_call("Sent: 10")  # Second write event

    def test_track_progress_closes_pbar_on_processing_done(self):
        """Test _track_progress closes progress bar when processing_done event is received."""
        data = [{"x": i} for i in range(10)]
        data_batcher = DataBatcher(data, batch_size=5)
        handler = MockBatchHandler()
        streamer = BatchStreamer(data_batcher, handler)

        # Simulate progress events including processing_done
        streamer._progress_queue.put({"event": "processing_started", "message": "Processing"})
        streamer._progress_queue.put({"event": "write", "num_rows": 5})
        streamer._progress_queue.put({"event": "done_writing"})
        streamer._progress_queue.put({"event": "read", "num_rows": 5})
        streamer._progress_queue.put({"event": "processing_done"})
        streamer._progress_queue.put({"event": "done_reading"})

        with patch("queryboost.stream.tqdm") as mock_tqdm:
            with patch("queryboost.stream.tqdm.write"):
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                streamer._track_progress_and_exceptions()

                # Verify progress bar was created
                mock_tqdm.assert_called_once()

                # Verify close was called when processing_done event was received
                mock_pbar.close.assert_called_once()
