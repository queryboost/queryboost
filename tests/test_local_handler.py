import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from queryboost.handlers import BatchHandler, LocalParquetBatchHandler


class TestLocalParquetBatchHandler:
    """Test suite for LocalParquetBatchHandler class."""

    def test_initialization_with_path_string(self):
        """Test initialization with path as string."""
        handler = LocalParquetBatchHandler(name="my-run", cache_dir="/tmp/test")

        assert handler._output_dir == Path("/tmp/test/my-run")
        assert handler._metadata == {}

    def test_initialization_with_path_object(self):
        """Test initialization with Path object."""
        path = Path("/tmp/test")
        handler = LocalParquetBatchHandler(name="my-run", cache_dir=path)

        assert handler._output_dir == path / "my-run"

    def test_initialization_with_metadata(self):
        """Test initialization with custom metadata."""
        metadata = {"source": "test", "version": "1.0"}
        handler = LocalParquetBatchHandler(
            name="my-run",
            cache_dir="/tmp/test",
            metadata=metadata,
        )

        assert handler._metadata == metadata

    def test_handle_creates_parquet_file(self, sample_batch_simple):
        """Test that handle method buffers and flushes to Parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use small target to trigger flush with single batch
            handler = LocalParquetBatchHandler(name="my-run", cache_dir=tmpdir, target_write_bytes=1)

            # Use mock batch fixture (auto-injected by pytest)
            handler.handle(sample_batch_simple)

            # Check file was created after flush
            output_file = Path(tmpdir) / "my-run" / "part-00000.parquet"
            assert output_file.exists()

            # Verify file contents match mock data
            table = pq.read_table(output_file)
            assert table.num_rows == 2
            assert "prompt" in table.column_names
            assert "response" in table.column_names

    def test_handle_multiple_batches(self):
        """Test handling multiple batches creates separate files when flushed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use small target to trigger flush after each batch
            handler = LocalParquetBatchHandler(name="my-run", cache_dir=tmpdir, target_write_bytes=1)

            # Handle multiple batches
            for i in range(3):
                batch = pa.RecordBatch.from_pydict(
                    {
                        "value": [i * 10, i * 10 + 1],
                    }
                )
                handler.handle(batch)

            # Check all files were created with proper naming
            for i in range(3):
                output_file = Path(tmpdir) / "my-run" / f"part-{i:05d}.parquet"
                assert output_file.exists()

    def test_handle_with_various_data_types(self):
        """Test handling batches with various PyArrow data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalParquetBatchHandler(name="my-run", cache_dir=tmpdir, target_write_bytes=1)

            # Create batch with various types
            batch = pa.RecordBatch.from_pydict(
                {
                    "string_col": ["a", "b"],
                    "int_col": [1, 2],
                    "float_col": [1.5, 2.5],
                    "bool_col": [True, False],
                }
            )
            handler.handle(batch)

            # Verify data persisted correctly
            output_file = Path(tmpdir) / "my-run" / "part-00000.parquet"
            table = pq.read_table(output_file)

            assert table.column("string_col").to_pylist() == ["a", "b"]
            assert table.column("int_col").to_pylist() == [1, 2]
            assert table.column("float_col").to_pylist() == [1.5, 2.5]
            assert table.column("bool_col").to_pylist() == [True, False]

    def test_handle_creates_output_directory(self):
        """Test that __init__ creates the output directory automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a subdirectory that doesn't exist yet
            cache_dir = Path(tmpdir) / "subdir" / "nested"

            # Directory should be created during initialization
            handler = LocalParquetBatchHandler(name="my-run", cache_dir=cache_dir, target_write_bytes=1)

            output_dir = cache_dir / "my-run"
            assert output_dir.exists()
            assert output_dir.is_dir()

            batch = pa.RecordBatch.from_pydict({"x": [1]})

            # Should work since directory was created in __init__
            handler.handle(batch)

            # Now file should be created after flush
            assert (output_dir / "part-00000.parquet").exists()

    def test_warns_when_directory_contains_files(self, caplog):
        """Test that warning is logged when output directory contains files."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the my-run subdirectory with an existing file
            run_dir = Path(tmpdir) / "my-run"
            run_dir.mkdir()
            (run_dir / "existing.txt").write_text("data")

            with caplog.at_level(logging.WARNING):
                LocalParquetBatchHandler(name="my-run", cache_dir=tmpdir)

            assert "already contains files" in caplog.text

    def test_buffering_combines_multiple_batches(self):
        """Test that buffering combines multiple batches into single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use large target so batches accumulate
            handler = LocalParquetBatchHandler(name="my-run", cache_dir=tmpdir, target_write_bytes=1000000)

            # Add two batches
            batch1 = pa.RecordBatch.from_pydict({"value": [1, 2]})
            batch2 = pa.RecordBatch.from_pydict({"value": [3, 4]})
            handler.handle(batch1)
            handler.handle(batch2)

            # No file should exist yet (not flushed)
            output_file = Path(tmpdir) / "my-run" / "part-00000.parquet"
            assert not output_file.exists()

            # Close should flush
            handler.close()
            assert output_file.exists()

            # Verify file contains both batches combined
            table = pq.read_table(output_file)
            assert table.column("value").to_pylist() == [1, 2, 3, 4]

    def test_handle_empty_batch(self):
        """Test handling empty batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalParquetBatchHandler(name="my-run", cache_dir=tmpdir, target_write_bytes=1)

            # Create empty batch with schema
            schema = pa.schema([("col", pa.int64())])
            batch = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], schema=schema)
            handler.handle(batch)

            # Empty batch won't auto-flush (0 bytes < threshold), need to close
            handler.close()

            # Verify empty file was created after flush
            output_file = Path(tmpdir) / "my-run" / "part-00000.parquet"
            assert output_file.exists()

            table = pq.read_table(output_file)
            assert table.num_rows == 0
            assert table.column_names == ["col"]

    def test_multiple_flushes_create_multiple_files(self):
        """Test that multiple flushes create sequentially numbered files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use small target to trigger multiple flushes
            handler = LocalParquetBatchHandler(name="my-run", cache_dir=tmpdir, target_write_bytes=1)

            for i in range(3):
                batch = pa.RecordBatch.from_pydict({"data": [i]})
                handler.handle(batch)

            # Verify files exist with sequential numbering
            assert (Path(tmpdir) / "my-run" / "part-00000.parquet").exists()
            assert (Path(tmpdir) / "my-run" / "part-00001.parquet").exists()
            assert (Path(tmpdir) / "my-run" / "part-00002.parquet").exists()

    def test_is_subclass_of_batch_handler(self):
        """Test that LocalParquetBatchHandler is a BatchHandler."""
        assert issubclass(LocalParquetBatchHandler, BatchHandler)

        handler = LocalParquetBatchHandler(name="my-run", cache_dir="/tmp/test")
        assert isinstance(handler, BatchHandler)
