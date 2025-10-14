# Contributing to Queryboost Python SDK

Thank you for your interest in contributing to Queryboost! This guide will help you develop custom batch handlers and contribute to the SDK.

## Table of Contents

- [Development Setup](#development-setup)
- [Creating Custom Batch Handlers](#creating-custom-batch-handlers)
- [Testing Without the Production Server](#testing-without-the-production-server)
- [Testing Your Handler](#testing-your-handler)
- [Submitting Your Contribution](#submitting-your-contribution)

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/queryboost-python-client.git
   cd queryboost-python-client
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

3. **Run tests to verify setup:**
   ```bash
   uv run pytest tests/
   ```

## Creating Custom Batch Handlers

Batch handlers define how processed data is saved or transmitted. Queryboost provides `LocalParquetBatchHandler` by default, but you can create custom handlers for:

- **Databases**: PostgreSQL, MySQL, MongoDB, etc.
- **Object Stores**: S3, GCS, Azure Blob Storage
- **Data Warehouses**: Snowflake, BigQuery, Redshift
- **Streaming**: Kafka, Kinesis, Pub/Sub
- **Custom Logic**: Any processing or storage you need

### Handler Structure

All handlers must:
1. Subclass `BatchHandler`
2. Implement the `handle()` method
3. Accept a PyArrow `RecordBatch` and `batch_idx`

### Example: S3 Batch Handler

```python
# src/queryboost/handlers/s3.py
from typing import Any
from pathlib import Path
import logging

import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from tqdm import tqdm

from .base import BatchHandler

logger = logging.getLogger(__name__)


class S3ParquetBatchHandler(BatchHandler):
    """Handler that uploads batches to S3 as Parquet files."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        metadata: dict[str, Any] = {},
    ):
        """Initialize the S3 Parquet batch handler.

        Args:
            bucket: S3 bucket name where files will be uploaded
            prefix: Optional prefix/folder path in the bucket
            metadata: Optional metadata dictionary
        """
        super().__init__(metadata)

        self._bucket = bucket
        self._prefix = prefix
        self._s3_client = boto3.client('s3')

        tqdm.write(f"Uploading results as parquet files to: s3://{bucket}/{prefix}")

    def handle(
        self,
        batch: pa.RecordBatch,
        batch_idx: int,
    ) -> None:
        """Process and upload a record batch to S3.

        Args:
            batch: PyArrow RecordBatch containing the data to upload
            batch_idx: Integer index of the batch
        """
        # Write to local temp file first
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            pq.write_table(pa.Table.from_batches([batch]), tmp.name)

            # Upload to S3
            key = f"{self._prefix}/batch_{batch_idx}.parquet"
            self._s3_client.upload_file(tmp.name, self._bucket, key)

            logger.info(f"Uploaded batch {batch_idx} to s3://{self._bucket}/{key}")
```

## Testing Without the Production Server

We provide mock data fixtures so you can develop handlers locally without connecting to the production Queryboost server (which uses GPU reservations).

### Using Mock Fixtures

Mock data fixtures are defined in `tests/conftest.py` and are automatically available in all tests (no imports needed!).

Available fixtures:
- `sample_batch_simple` - Basic batch with prompt/response data (2 rows)
- `sample_batch_with_metadata` - Batch with additional metadata fields (tokens, timestamps)
- `sample_batch_large` - 100-row batch for performance testing
- `sample_batch_empty` - Empty batch for edge case testing

These fixtures are automatically injected by pytest when you add them as test parameters:

## Testing Your Handler

### Test Structure

Create tests in `tests/test_handlers.py`:

```python
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from queryboost.handlers.s3 import S3ParquetBatchHandler


class TestS3ParquetBatchHandler:
    """Test suite for S3ParquetBatchHandler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = S3ParquetBatchHandler(
            bucket="my-bucket",
            prefix="results"
        )

        assert handler._bucket == "my-bucket"
        assert handler._prefix == "results"

    @patch('boto3.client')
    def test_handle_uploads_to_s3(self, mock_boto_client, sample_batch_simple):
        """Test that handle uploads batch to S3.

        Note: sample_batch_simple is auto-injected by pytest - no import needed!
        """
        # Setup mock S3 client
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3

        handler = S3ParquetBatchHandler(bucket="test-bucket")

        # Use mock batch fixture (auto-injected by pytest)
        handler.handle(sample_batch_simple, batch_idx=0)

        # Verify S3 upload was called
        mock_s3.upload_file.assert_called_once()
        args = mock_s3.upload_file.call_args
        assert args[0][1] == "test-bucket"  # bucket
        assert "batch_0.parquet" in args[0][2]  # key

    def test_handle_with_metadata_batch(self, sample_batch_with_metadata):
        """Test handler with metadata-rich batch.

        Note: sample_batch_with_metadata is auto-injected by pytest!
        """
        with patch('boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_boto.return_value = mock_s3

            handler = S3ParquetBatchHandler(bucket="test-bucket")

            # Use metadata batch fixture (auto-injected)
            handler.handle(sample_batch_with_metadata, batch_idx=5)

            # Verify correct batch index in filename
            args = mock_s3.upload_file.call_args
            assert "batch_5.parquet" in args[0][2]
```

**Key points:**
- Fixtures are automatically discovered from `tests/conftest.py`
- No imports needed - just add fixture name as test parameter
- Pytest automatically injects the fixture value

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src/queryboost --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_handlers.py

# Run specific test
uv run pytest tests/test_handlers.py::TestS3ParquetBatchHandler::test_initialization
```

### Testing Best Practices

#### Unit Tests (Required)

Unit tests mock external services and run quickly without any external dependencies:

1. **Use mock fixtures** - Never require production server access for unit tests
2. **Mock external services** - Use `unittest.mock` or `pytest-mock` for S3, databases, etc.
3. **Test edge cases** - Empty batches, large batches, network failures
4. **Test error handling** - Connection failures, permission errors, etc.
5. **Aim for 100% coverage** - All handlers should have comprehensive tests

#### Integration Tests (Recommended)

Integration tests verify handlers work with real service implementations, but use **local test infrastructure** instead of production services:

**For AWS services (S3, etc.):**
Use [LocalStack](https://github.com/localstack/localstack) - a local AWS cloud emulator:

```python
# tests/integration/test_s3_handler.py
import pytest
from queryboost.handlers.s3 import S3ParquetBatchHandler

@pytest.mark.integration
def test_s3_handler_with_localstack(sample_batch_simple):
    """Test S3 handler against LocalStack (local AWS emulator)."""
    # Assumes LocalStack running: docker run -p 4566:4566 localstack/localstack
    handler = S3ParquetBatchHandler(
        bucket="test-bucket",
        endpoint_url="http://localhost:4566",  # LocalStack endpoint
        aws_access_key_id="test",
        aws_secret_access_key="test"
    )
    handler.handle(sample_batch_simple, 0)

    # Verify upload succeeded
    import boto3
    s3 = boto3.client('s3', endpoint_url='http://localhost:4566')
    response = s3.head_object(Bucket='test-bucket', Key='batch_0.parquet')
    assert response['ResponseMetadata']['HTTPStatusCode'] == 200
```

**For databases:**
Use [pytest-docker](https://github.com/avast/pytest-docker) or [testcontainers](https://testcontainers-python.readthedocs.io/):

```python
# tests/integration/test_postgres_handler.py
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.mark.integration
def test_postgres_handler(sample_batch_simple):
    """Test PostgreSQL handler against real Postgres in Docker."""
    with PostgresContainer("postgres:15") as postgres:
        handler = PostgresBatchHandler(
            connection_string=postgres.get_connection_url()
        )
        handler.handle(sample_batch_simple, 0)

        # Verify data was inserted
        # ... verification logic ...
```

**Running integration tests:**

```bash
# Run only unit tests (fast, no external dependencies)
uv run pytest tests/ -m "not integration"

# Run integration tests (requires Docker)
docker run -d -p 4566:4566 localstack/localstack
uv run pytest tests/ -m integration

# Run all tests
uv run pytest tests/
```

**Why use local test infrastructure?**
- ✅ No production credentials needed
- ✅ Fast enough for CI/CD pipelines
- ✅ Tests actual service protocols (not just mocks)
- ✅ Catches real-world integration issues
- ✅ Safe - no risk of polluting production data

**Integration tests are optional but recommended** for handlers that interact with external services. They provide higher confidence than mocked unit tests while remaining practical for development.

## Submitting Your Contribution

### Before Submitting

1. **Run all tests:**
   ```bash
   uv run pytest tests/
   ```

2. **Check code formatting:**
   ```bash
   uv run ruff check .
   uv run ruff format .
   ```

3. **Verify coverage:**
   ```bash
   uv run pytest tests/ --cov=src/queryboost --cov-report=term-missing
   ```

### Pull Request Guidelines

1. **Create a new branch:**
   ```bash
   git checkout -b feature/add-s3-handler
   ```

2. **Include in your PR:**
   - New handler implementation in `src/queryboost/handlers/`
   - Comprehensive tests in `tests/test_handlers.py`
   - Update to handler exports in `src/queryboost/handlers/__init__.py`
   - Documentation/docstrings following existing patterns
   - Example usage in your PR description

3. **PR Title Format:**
   - `feat: Add S3ParquetBatchHandler for S3 uploads`
   - `fix: Handle connection errors in PostgresBatchHandler`
   - `docs: Add example for custom batch handlers`

4. **Description should include:**
   - What the handler does
   - Example usage code
   - Any external dependencies added
   - Testing approach

### Example PR Description

```markdown
## Summary
Adds `S3ParquetBatchHandler` for uploading processed batches directly to S3.

## Usage
```python
from queryboost import Queryboost
from queryboost.handlers import S3ParquetBatchHandler

qb = Queryboost()
handler = S3ParquetBatchHandler(bucket="my-bucket", prefix="results/")
qb.run(data, prompt, batch_handler=handler)
```

## Dependencies
- Adds `boto3` as an optional dependency

## Testing
- All tests use mock S3 client (no AWS credentials required)
- Includes tests for upload, error handling, and edge cases
- Coverage: 100%
```

## Questions?

- **Issues**: https://github.com/yourusername/queryboost-python-client/issues
- **Discussions**: https://github.com/yourusername/queryboost-python-client/discussions

Thank you for contributing to Queryboost! 🚀
