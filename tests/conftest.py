"""Pytest configuration and shared fixtures for Queryboost tests."""

from unittest.mock import Mock

import pytest
import pyarrow as pa

# Mock data fixtures for testing batch handlers
# These simulate responses from the Queryboost server for local testing


@pytest.fixture
def sample_batch_simple():
    """Create a simple sample batch with basic data types.

    Returns:
        A RecordBatch with columns: prompt, response, id
    """
    return pa.RecordBatch.from_pydict(
        {
            "prompt": ["What is Python?", "Explain recursion"],
            "response": [
                "Python is a high-level programming language.",
                "Recursion is when a function calls itself.",
            ],
            "id": [1, 2],
        }
    )


@pytest.fixture
def sample_batch_with_metadata():
    """Create a sample batch with metadata fields.

    Returns:
        A RecordBatch with columns: prompt, response, tokens, timestamp
    """
    return pa.RecordBatch.from_pydict(
        {
            "prompt": ["Hello world", "Test prompt", "Another query"],
            "response": ["Hi there!", "Test response", "Query result"],
            "tokens": [15, 20, 18],
            "timestamp": ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
        }
    )


@pytest.fixture
def sample_batch_large():
    """Create a larger sample batch for performance testing.

    Returns:
        A RecordBatch with 100 rows
    """
    size = 100
    return pa.RecordBatch.from_pydict(
        {
            "prompt": [f"Prompt {i}" for i in range(size)],
            "response": [f"Response {i}" for i in range(size)],
            "id": list(range(size)),
            "score": [float(i) / 10 for i in range(size)],
        }
    )


@pytest.fixture
def sample_batch_empty():
    """Create an empty batch with schema but no rows.

    Returns:
        An empty RecordBatch with the standard schema
    """
    return pa.RecordBatch.from_pydict(
        {
            "prompt": [],
            "response": [],
            "id": [],
        }
    )


# Infrastructure fixtures


@pytest.fixture
def mock_flight_client():
    """Fixture providing a mock FlightClient for integration tests."""
    mock = Mock()
    mock.authenticate = Mock()
    mock.do_exchange = Mock()
    return mock


@pytest.fixture(autouse=True)
def reset_env_vars(monkeypatch):
    """Automatically reset environment variables before each test."""
    monkeypatch.delenv("QUERYBOOST_API_KEY", raising=False)
    monkeypatch.delenv("QUERYBOOST_URL", raising=False)
    monkeypatch.delenv("QUERYBOOST_PORT", raising=False)
