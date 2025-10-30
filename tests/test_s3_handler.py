from io import BytesIO
from unittest.mock import Mock, MagicMock, patch

import boto3  # type: ignore
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from moto import mock_aws  # type: ignore

from queryboost.handlers import BatchHandler
from queryboost.exceptions import QueryboostBatchHandlerError
from queryboost.handlers.s3 import S3ParquetBatchHandler


@pytest.fixture
def s3_setup():
    """Set up mock S3 environment with moto for boto3."""
    with mock_aws():
        # Create S3 client and bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket = "test-bucket"
        s3_client.create_bucket(Bucket=bucket)

        yield {
            "bucket": bucket,
            "s3_client": s3_client,
        }


@pytest.fixture
def mock_s3_fs():
    """Mock PyArrow S3FileSystem."""
    with patch("queryboost.handlers.s3.pafs.S3FileSystem") as mock_fs_class:
        mock_fs = Mock()
        mock_fs_class.return_value = mock_fs

        # Track written data in memory
        written_files = {}

        def mock_open_output_stream(path):
            """Mock output stream that captures written data."""
            stream = BytesIO()

            def mock_exit(_exc_type, _exc_val, _exc_tb):
                # Save data when context manager exits
                written_files[path] = stream.getvalue()
                return False

            return MagicMock(__enter__=Mock(return_value=stream), __exit__=Mock(side_effect=mock_exit))

        mock_fs.open_output_stream.side_effect = mock_open_output_stream

        yield {
            "mock_fs": mock_fs,
            "written_files": written_files,
        }


class TestS3ParquetBatchHandler:
    """Test suite for S3ParquetBatchHandler."""

    def test_initialization_default_target_bytes(self, s3_setup, mock_s3_fs):  # noqa: ARG002
        """Test S3ParquetBatchHandler initialization with default target_write_bytes."""
        handler = S3ParquetBatchHandler(
            name="test-run",
            bucket=s3_setup["bucket"],
        )

        assert handler._bucket == s3_setup["bucket"]
        assert handler._name == "test-run"
        assert handler._target_write_bytes == 256 * 1024 * 1024
        assert handler._metadata == {}

    def test_initialization_with_custom_target_bytes(self, s3_setup, mock_s3_fs):  # noqa: ARG002
        """Test initialization with custom target_write_bytes."""
        target_bytes = 128 * 1024 * 1024
        handler = S3ParquetBatchHandler(
            name="test-run",
            bucket=s3_setup["bucket"],
            target_write_bytes=target_bytes,
        )

        assert handler._target_write_bytes == target_bytes

    def test_initialization_with_metadata(self, s3_setup, mock_s3_fs):  # noqa: ARG002
        """Test initialization with custom metadata."""
        metadata = {"source": "test", "version": "1.0"}
        handler = S3ParquetBatchHandler(
            name="test-run",
            bucket=s3_setup["bucket"],
            metadata=metadata,
        )

        assert handler._metadata == metadata

    def test_flush_creates_s3_file(self, s3_setup, mock_s3_fs):
        """Test that flush creates a Parquet file in S3."""
        handler = S3ParquetBatchHandler(
            name="test-run",
            bucket=s3_setup["bucket"],
            target_write_bytes=1,  # Trigger flush immediately
        )

        # Create and handle a batch
        batch = pa.RecordBatch.from_pydict({"col": [1, 2, 3]})
        handler.handle(batch)

        # Verify file was written with correct path (bucket/key format)
        mock_fs = mock_s3_fs["mock_fs"]
        expected_path = "test-bucket/test-run/part-00000.parquet"
        mock_fs.open_output_stream.assert_called_once_with(expected_path)

        # Verify data was written
        written_files = mock_s3_fs["written_files"]
        assert expected_path in written_files
        assert len(written_files[expected_path]) > 0

    def test_buffering_combines_batches(self, s3_setup, mock_s3_fs):
        """Test that buffering combines multiple batches into single S3 file."""
        handler = S3ParquetBatchHandler(
            name="test-run",
            bucket=s3_setup["bucket"],
            target_write_bytes=1000000,  # Large threshold
        )

        # Add two batches
        batch1 = pa.RecordBatch.from_pydict({"value": [1, 2]})
        batch2 = pa.RecordBatch.from_pydict({"value": [3, 4]})
        handler.handle(batch1)
        handler.handle(batch2)

        # No file should be written yet (not flushed)
        mock_fs = mock_s3_fs["mock_fs"]
        assert mock_fs.open_output_stream.call_count == 0

        # Close should flush
        handler.close()

        # Now file should be written
        assert mock_fs.open_output_stream.call_count == 1
        expected_path = "test-bucket/test-run/part-00000.parquet"
        mock_fs.open_output_stream.assert_called_with(expected_path)

        # Verify combined data
        written_files = mock_s3_fs["written_files"]
        assert expected_path in written_files

        # Read back the parquet data to verify
        parquet_bytes = BytesIO(written_files[expected_path])
        table = pq.read_table(parquet_bytes)
        assert table.column("value").to_pylist() == [1, 2, 3, 4]

    def test_multiple_flushes_create_multiple_files(self, s3_setup, mock_s3_fs):
        """Test that multiple flushes create sequentially numbered files."""
        handler = S3ParquetBatchHandler(
            bucket=s3_setup["bucket"],
            name="test-run",
            target_write_bytes=1,  # Trigger flush for each batch
        )

        # Handle three batches
        for i in range(3):
            batch = pa.RecordBatch.from_pydict({"data": [i]})
            handler.handle(batch)

        # Verify three files were written with correct naming
        mock_fs = mock_s3_fs["mock_fs"]
        assert mock_fs.open_output_stream.call_count == 3

        expected_calls = [
            (("test-bucket/test-run/part-00000.parquet",),),
            (("test-bucket/test-run/part-00001.parquet",),),
            (("test-bucket/test-run/part-00002.parquet",),),
        ]
        mock_fs.open_output_stream.assert_has_calls(expected_calls)

    def test_name_must_not_be_empty(self, s3_setup, mock_s3_fs):  # noqa: ARG002
        """Test that empty name raises QueryboostBatchHandlerError."""
        with pytest.raises(QueryboostBatchHandlerError, match="name cannot be empty"):
            S3ParquetBatchHandler(
                name="",
                bucket=s3_setup["bucket"],
            )

    def test_prefix_must_be_empty_initially(self, s3_setup, mock_s3_fs):  # noqa: ARG002
        """Test that non-empty prefix raises QueryboostBatchHandlerError."""
        # Upload a file to the prefix using boto3 (moto will track this)
        s3_client = s3_setup["s3_client"]
        s3_client.put_object(
            Bucket=s3_setup["bucket"],
            Key="existing-run/file.txt",
            Body=b"data",
        )

        # Try to create handler with existing prefix
        with pytest.raises(QueryboostBatchHandlerError, match="already contains files"):
            S3ParquetBatchHandler(
                name="existing-run",
                bucket=s3_setup["bucket"],
            )

    def test_strips_leading_trailing_slashes(self, s3_setup, mock_s3_fs):  # noqa: ARG002
        """Test that name slashes are normalized."""
        handler = S3ParquetBatchHandler(
            name="/test-run/",
            bucket=s3_setup["bucket"],
        )

        assert handler._name == "test-run"

    def test_handle_with_various_data_types(self, s3_setup, mock_s3_fs):
        """Test handling batches with various PyArrow data types."""
        handler = S3ParquetBatchHandler(
            bucket=s3_setup["bucket"],
            name="test-run",
            target_write_bytes=1,
        )

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

        # Verify data was written and can be read back
        written_files = mock_s3_fs["written_files"]
        expected_path = "test-bucket/test-run/part-00000.parquet"
        assert expected_path in written_files

        parquet_bytes = BytesIO(written_files[expected_path])
        table = pq.read_table(parquet_bytes)
        assert table.column("string_col").to_pylist() == ["a", "b"]
        assert table.column("int_col").to_pylist() == [1, 2]
        assert table.column("float_col").to_pylist() == [1.5, 2.5]
        assert table.column("bool_col").to_pylist() == [True, False]

    def test_handle_empty_batch(self, s3_setup, mock_s3_fs):
        """Test handling empty batch."""
        handler = S3ParquetBatchHandler(
            bucket=s3_setup["bucket"],
            name="test-run",
            target_write_bytes=1,
        )

        # Create empty batch with schema
        schema = pa.schema([("col", pa.int64())])
        batch = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], schema=schema)
        handler.handle(batch)

        # Empty batch won't auto-flush (0 bytes < threshold), need to close
        handler.close()

        # Verify file was written
        written_files = mock_s3_fs["written_files"]
        expected_path = "test-bucket/test-run/part-00000.parquet"
        assert expected_path in written_files

        # Verify empty file structure
        parquet_bytes = BytesIO(written_files[expected_path])
        table = pq.read_table(parquet_bytes)
        assert table.num_rows == 0
        assert table.column_names == ["col"]

    def test_close_flushes_remaining_buffer(self, s3_setup, mock_s3_fs):
        """Test that close() flushes any remaining batches in buffer."""
        handler = S3ParquetBatchHandler(
            bucket=s3_setup["bucket"],
            name="test-run",
            target_write_bytes=1000000,  # Large threshold
        )

        batch = pa.RecordBatch.from_pydict({"col": [1, 2, 3]})
        handler.handle(batch)

        # Buffer should contain batch, no flush yet
        mock_fs = mock_s3_fs["mock_fs"]
        assert mock_fs.open_output_stream.call_count == 0

        # Close should flush
        handler.close()
        assert mock_fs.open_output_stream.call_count == 1

    def test_close_with_empty_buffer(self, s3_setup, mock_s3_fs):
        """Test that close() with empty buffer doesn't create files."""
        handler = S3ParquetBatchHandler(
            bucket=s3_setup["bucket"],
            name="test-run",
        )

        handler.close()

        # No files should be written
        mock_fs = mock_s3_fs["mock_fs"]
        assert mock_fs.open_output_stream.call_count == 0

    def test_is_subclass_of_batch_handler(self, s3_setup, mock_s3_fs):  # noqa: ARG002
        """Test that S3ParquetBatchHandler is a BatchHandler."""
        assert issubclass(S3ParquetBatchHandler, BatchHandler)

        handler = S3ParquetBatchHandler(
            bucket=s3_setup["bucket"],
            name="test-run",
        )
        assert isinstance(handler, BatchHandler)

    def test_bucket_creation_if_not_exists(self, mock_s3_fs):  # noqa: ARG002
        """Test that handler creates bucket if it doesn't exist."""
        with mock_aws():
            s3_client = boto3.client("s3", region_name="us-east-1")

            # Create handler with non-existent bucket
            handler = S3ParquetBatchHandler(
                bucket="new-bucket",
                name="test-run",
            )

            # Verify bucket was created
            response = s3_client.list_buckets()
            bucket_names = [b["Name"] for b in response["Buckets"]]
            assert "new-bucket" in bucket_names
            assert handler._bucket == "new-bucket"

    def test_bucket_access_error(self, mock_s3_fs):  # noqa: ARG002
        """Test that handler raises error when bucket access fails."""
        from botocore.exceptions import ClientError  # type: ignore

        with mock_aws():
            with patch("queryboost.handlers.s3.boto3.client") as mock_boto_client:
                mock_s3_client = Mock()
                mock_boto_client.return_value = mock_s3_client

                # Simulate access denied error (not 404)
                error_response = {"Error": {"Code": "403", "Message": "Forbidden"}}
                mock_s3_client.head_bucket.side_effect = ClientError(error_response, "HeadBucket")

                # Should raise QueryboostBatchHandlerError
                with pytest.raises(QueryboostBatchHandlerError, match="Failed to access S3 bucket"):
                    S3ParquetBatchHandler(
                        bucket="test-bucket",
                        name="test-run",
                    )
