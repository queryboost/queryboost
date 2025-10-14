import pytest
import pyarrow.flight as flight

from queryboost.exceptions import (
    QueryboostPromptError,
    QueryboostServerError,
    QueryboostDataTypeError,
    QueryboostBatchSizeError,
    QueryboostConfigurationError,
    clean_flight_error_message,
)
from queryboost.exceptions.exceptions import QueryboostError


def test_queryboost_error_is_exception():
    """Test that QueryboostError is an Exception subclass."""
    assert issubclass(QueryboostError, Exception)


def test_queryboost_error_can_be_raised():
    """Test that QueryboostError can be raised with a message."""
    with pytest.raises(QueryboostError) as exc_info:
        raise QueryboostError("Test error message")
    assert str(exc_info.value) == "Test error message"


def test_configuration_error_is_queryboost_error():
    """Test that QueryboostConfigurationError inherits from QueryboostError."""
    assert issubclass(QueryboostConfigurationError, QueryboostError)


def test_configuration_error_can_be_raised():
    """Test that QueryboostConfigurationError can be raised."""
    with pytest.raises(QueryboostConfigurationError) as exc_info:
        raise QueryboostConfigurationError("Config error")
    assert str(exc_info.value) == "Config error"


def test_data_type_error_is_queryboost_error():
    """Test that QueryboostDataTypeError inherits from QueryboostError."""
    assert issubclass(QueryboostDataTypeError, QueryboostError)


def test_data_type_error_can_be_raised():
    """Test that QueryboostDataTypeError can be raised."""
    with pytest.raises(QueryboostDataTypeError) as exc_info:
        raise QueryboostDataTypeError("Data type error")
    assert str(exc_info.value) == "Data type error"


def test_batch_size_error_is_queryboost_error():
    """Test that QueryboostBatchSizeError inherits from QueryboostError."""
    assert issubclass(QueryboostBatchSizeError, QueryboostError)


def test_batch_size_error_can_be_raised():
    """Test that QueryboostBatchSizeError can be raised."""
    with pytest.raises(QueryboostBatchSizeError) as exc_info:
        raise QueryboostBatchSizeError("Batch size error")
    assert str(exc_info.value) == "Batch size error"


def test_prompt_error_is_queryboost_error():
    """Test that QueryboostPromptError inherits from QueryboostError."""
    assert issubclass(QueryboostPromptError, QueryboostError)


def test_prompt_error_can_be_raised():
    """Test that QueryboostPromptError can be raised."""
    with pytest.raises(QueryboostPromptError) as exc_info:
        raise QueryboostPromptError("Prompt error")
    assert str(exc_info.value) == "Prompt error"


def test_all_exceptions_can_be_caught_as_queryboost_error():
    """Test that all specific exceptions can be caught as QueryboostError."""
    exceptions = [
        QueryboostConfigurationError("config"),
        QueryboostDataTypeError("data"),
        QueryboostBatchSizeError("batch"),
        QueryboostPromptError("prompt"),
        QueryboostServerError("server"),
    ]

    for exc in exceptions:
        with pytest.raises(QueryboostError):
            raise exc


def test_server_error_is_queryboost_error():
    """Test that QueryboostServerError inherits from QueryboostError."""
    assert issubclass(QueryboostServerError, QueryboostError)


def test_server_error_can_be_raised():
    """Test that QueryboostServerError can be raised."""
    with pytest.raises(QueryboostServerError) as exc_info:
        raise QueryboostServerError("Server error")
    assert str(exc_info.value) == "Server error"


def test_clean_flight_error_message_removes_grpc_context():
    """Test that clean_flight_error_message removes gRPC debug context."""
    error = flight.FlightServerError(
        'Flight error: Invalid input. gRPC client debug context: UNKNOWN:Error received from peer ipv4:34.232.20.189:443 {grpc_message:"Flight error: Invalid input.", grpc_status:2, created_time:"2025-10-07T21:15:21.522313-07:00"}'
    )

    cleaned = clean_flight_error_message(error)

    assert "gRPC client debug context" not in cleaned
    assert cleaned == "Invalid input"


def test_clean_flight_error_message_removes_flight_error_prefix():
    """Test that clean_flight_error_message removes 'Flight error:' prefix."""
    error = flight.FlightServerError("Flight error: Number of GPUs must be at least 1, got 0.")

    cleaned = clean_flight_error_message(error)

    assert cleaned == "Number of GPUs must be at least 1, got 0."
    assert not cleaned.startswith("Flight error:")


def test_clean_flight_error_message_handles_plain_message():
    """Test that clean_flight_error_message handles messages without prefixes."""
    error = Exception("Simple error message")

    cleaned = clean_flight_error_message(error)

    assert cleaned == "Simple error message"
