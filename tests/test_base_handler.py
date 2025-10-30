import pytest
import pyarrow as pa

from queryboost.handlers import BatchHandler


class MockBatchHandler(BatchHandler):
    """Mock implementation of BatchHandler for testing."""

    def __init__(self, name: str = "test-handler", target_write_bytes: int = 256 * 1024 * 1024, metadata={}):
        super().__init__(name, target_write_bytes, metadata)
        self.flushed_batches: list[list[pa.RecordBatch]] = []

    def _flush(self):
        """Store flushed batches for verification."""
        # Append shallow copy of buffer since base class clears it after flush
        self.flushed_batches.append(list(self._buffer))


class TestBatchHandler:
    """Test suite for BatchHandler base class."""

    def test_initialization_default_metadata(self):
        """Test BatchHandler initialization with default metadata and target_write_bytes."""
        handler = MockBatchHandler()

        assert handler._metadata == {}
        assert handler._target_write_bytes == 256 * 1024 * 1024
        assert handler._buffer == []
        assert handler._buffer_bytes == 0
        assert handler._write_idx == 0

    def test_initialization_with_metadata(self):
        """Test BatchHandler initialization with custom metadata and target_write_bytes."""
        metadata = {"key": "value", "count": 42}
        target_bytes = 128 * 1024 * 1024
        handler = MockBatchHandler(target_write_bytes=target_bytes, metadata=metadata)

        assert handler._metadata == metadata
        assert handler._target_write_bytes == target_bytes

    def test_flush_method_must_be_implemented(self):
        """Test that BatchHandler cannot be instantiated without implementing _flush."""
        with pytest.raises(TypeError):
            BatchHandler()  # type: ignore

    def test_buffer_accumulation_and_flush(self):
        """Test that handler accumulates batches in buffer and flushes when threshold exceeded."""
        # Use very small target to trigger flush after two small batches
        handler = MockBatchHandler(target_write_bytes=40)

        # Create small batches
        batch1 = pa.RecordBatch.from_pydict({"col": [1, 2, 3]})  # Small batch
        batch2 = pa.RecordBatch.from_pydict({"col": [4, 5, 6]})  # Small batch

        # First batch should accumulate
        handler.handle(batch1)
        assert len(handler._buffer) == 1
        assert len(handler.flushed_batches) == 0

        # Second batch should trigger flush
        handler.handle(batch2)
        assert len(handler._buffer) == 0  # Buffer cleared after flush
        assert len(handler.flushed_batches) == 1  # One flush occurred
        assert len(handler.flushed_batches[0]) == 2  # Both batches were in the flush
        assert handler._write_idx == 1

    def test_close_flushes_remaining_buffer(self):
        """Test that close() flushes any remaining batches in buffer."""
        handler = MockBatchHandler(target_write_bytes=1000000)  # Large threshold

        batch = pa.RecordBatch.from_pydict({"col": [1, 2, 3]})
        handler.handle(batch)

        # Buffer should contain batch, no flush yet
        assert len(handler._buffer) == 1
        assert len(handler.flushed_batches) == 0

        # Close should flush
        handler.close()
        assert len(handler.flushed_batches) == 1
        assert len(handler.flushed_batches[0]) == 1

    def test_close_with_empty_buffer(self):
        """Test that close() with empty buffer doesn't flush."""
        handler = MockBatchHandler()

        handler.close()
        assert len(handler.flushed_batches) == 0

    def test_invalid_target_write_bytes(self):
        """Test that target_write_bytes must be greater than 0."""
        from queryboost.exceptions import QueryboostBatchHandlerError

        with pytest.raises(
            QueryboostBatchHandlerError,
            match="target_write_bytes must be greater than 0.",
        ):
            MockBatchHandler(target_write_bytes=0)
