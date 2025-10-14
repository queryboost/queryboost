import tempfile
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from queryboost.handlers import BatchHandler, LocalParquetBatchHandler


class MockBatchHandler(BatchHandler):
    """Mock implementation of BatchHandler for testing."""

    def __init__(self, metadata={}):
        super().__init__(metadata)
        self.handled_batches = []

    def handle(self, batch: pa.RecordBatch, batch_idx: int):
        """Store handled batches for verification."""
        self.handled_batches.append(
            {
                "batch": batch,
                "batch_idx": batch_idx,
            }
        )


class TestBatchHandler:
    """Test suite for BatchHandler base class."""

    def test_initialization_default_metadata(self):
        """Test BatchHandler initialization with default metadata."""
        handler = MockBatchHandler()

        assert handler._metadata == {}

    def test_initialization_with_metadata(self):
        """Test BatchHandler initialization with custom metadata."""
        metadata = {"key": "value", "count": 42}
        handler = MockBatchHandler(metadata=metadata)

        assert handler._metadata == metadata

    def test_handle_method_must_be_implemented(self):
        """Test that BatchHandler cannot be instantiated without implementing handle."""
        with pytest.raises(TypeError):
            BatchHandler()  # type: ignore

    def test_mock_handler_tracks_batches(self):
        """Test that mock handler correctly tracks handled batches."""
        handler = MockBatchHandler()

        # Create test batches
        batch1 = pa.RecordBatch.from_pydict({"col": [1, 2, 3]})
        batch2 = pa.RecordBatch.from_pydict({"col": [4, 5, 6]})

        handler.handle(batch1, 0)
        handler.handle(batch2, 1)

        assert len(handler.handled_batches) == 2
        assert handler.handled_batches[0]["batch_idx"] == 0
        assert handler.handled_batches[1]["batch_idx"] == 1


class TestLocalParquetBatchHandler:
    """Test suite for LocalParquetBatchHandler class."""

    def test_initialization_with_path_string(self):
        """Test initialization with path as string."""
        handler = LocalParquetBatchHandler(output_dir="/tmp/test")

        assert handler._output_dir == Path("/tmp/test")
        assert handler._metadata == {}

    def test_initialization_with_path_object(self):
        """Test initialization with Path object."""
        path = Path("/tmp/test")
        handler = LocalParquetBatchHandler(output_dir=path)

        assert handler._output_dir == path

    def test_initialization_with_metadata(self):
        """Test initialization with custom metadata."""
        metadata = {"source": "test", "version": "1.0"}
        handler = LocalParquetBatchHandler(
            output_dir="/tmp/test",
            metadata=metadata,
        )

        assert handler._metadata == metadata

    def test_handle_creates_parquet_file(self, sample_batch_simple):
        """Test that handle method creates a Parquet file using mock data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalParquetBatchHandler(output_dir=tmpdir)

            # Use mock batch fixture (auto-injected by pytest)
            handler.handle(sample_batch_simple, 0)

            # Check file was created
            output_file = Path(tmpdir) / "batch_0.parquet"
            assert output_file.exists()

            # Verify file contents match mock data
            table = pq.read_table(output_file)
            assert table.num_rows == 2
            assert "prompt" in table.column_names
            assert "response" in table.column_names

    def test_handle_multiple_batches(self):
        """Test handling multiple batches creates separate files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalParquetBatchHandler(output_dir=tmpdir)

            # Handle multiple batches
            for i in range(3):
                batch = pa.RecordBatch.from_pydict(
                    {
                        "value": [i * 10, i * 10 + 1],
                    }
                )
                handler.handle(batch, i)

            # Check all files were created
            for i in range(3):
                output_file = Path(tmpdir) / f"batch_{i}.parquet"
                assert output_file.exists()

    def test_handle_with_various_data_types(self):
        """Test handling batches with various PyArrow data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalParquetBatchHandler(output_dir=tmpdir)

            # Create batch with various types
            batch = pa.RecordBatch.from_pydict(
                {
                    "string_col": ["a", "b"],
                    "int_col": [1, 2],
                    "float_col": [1.5, 2.5],
                    "bool_col": [True, False],
                }
            )
            handler.handle(batch, 0)

            # Verify data persisted correctly
            output_file = Path(tmpdir) / "batch_0.parquet"
            table = pq.read_table(output_file)

            assert table.column("string_col").to_pylist() == ["a", "b"]
            assert table.column("int_col").to_pylist() == [1, 2]
            assert table.column("float_col").to_pylist() == [1.5, 2.5]
            assert table.column("bool_col").to_pylist() == [True, False]

    def test_handle_creates_output_directory(self):
        """Test that __init__ creates the output directory automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a subdirectory that doesn't exist yet
            output_dir = Path(tmpdir) / "subdir" / "nested"

            # Directory should be created during initialization
            handler = LocalParquetBatchHandler(output_dir=output_dir)

            assert output_dir.exists()
            assert output_dir.is_dir()

            batch = pa.RecordBatch.from_pydict({"x": [1]})

            # Should work since directory was created in __init__
            handler.handle(batch, 0)

            # Now file should be created
            assert (output_dir / "batch_0.parquet").exists()

    def test_warns_when_directory_contains_files(self, caplog):
        """Test that warning is logged when output directory contains files."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing file
            (Path(tmpdir) / "existing.txt").write_text("data")

            with caplog.at_level(logging.WARNING):
                LocalParquetBatchHandler(output_dir=tmpdir)

            assert "already contains files" in caplog.text

    def test_handle_overwrites_existing_file(self):
        """Test that handling batch with same index overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalParquetBatchHandler(output_dir=tmpdir)

            # Write first batch
            batch1 = pa.RecordBatch.from_pydict({"value": [1, 2]})
            handler.handle(batch1, 0)

            # Write second batch with same index
            batch2 = pa.RecordBatch.from_pydict({"value": [99, 100]})
            handler.handle(batch2, 0)

            # Verify file contains second batch data
            output_file = Path(tmpdir) / "batch_0.parquet"
            table = pq.read_table(output_file)
            assert table.column("value").to_pylist() == [99, 100]

    def test_handle_empty_batch(self):
        """Test handling empty batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalParquetBatchHandler(output_dir=tmpdir)

            # Create empty batch with schema
            schema = pa.schema([("col", pa.int64())])
            batch = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], schema=schema)
            handler.handle(batch, 0)

            # Verify empty file was created
            output_file = Path(tmpdir) / "batch_0.parquet"
            assert output_file.exists()

            table = pq.read_table(output_file)
            assert table.num_rows == 0
            assert table.column_names == ["col"]

    def test_handle_large_batch_idx(self):
        """Test handling batch with large index number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalParquetBatchHandler(output_dir=tmpdir)

            batch = pa.RecordBatch.from_pydict({"data": [1, 2, 3]})
            handler.handle(batch, 9999)

            output_file = Path(tmpdir) / "batch_9999.parquet"
            assert output_file.exists()

    def test_is_subclass_of_batch_handler(self):
        """Test that LocalParquetBatchHandler is a BatchHandler."""
        assert issubclass(LocalParquetBatchHandler, BatchHandler)

        handler = LocalParquetBatchHandler(output_dir="/tmp/test")
        assert isinstance(handler, BatchHandler)
