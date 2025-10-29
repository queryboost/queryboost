from pyarrow.flight import FlightError


class QueryboostError(Exception):
    """The base exception class for all exceptions in the Queryboost Python SDK."""


class QueryboostConfigurationError(QueryboostError):
    """Raised when a configuration error occurs."""


class QueryboostDataTypeError(QueryboostError):
    """Raised when a data type error occurs."""


class QueryboostDataColumnError(QueryboostError):
    """Raised when a data column error occurs."""


class QueryboostBatchSizeError(QueryboostError):
    """Raised when a batch size error occurs."""


class QueryboostPromptError(QueryboostError):
    """Raised when a prompt error occurs."""


class QueryboostBatchHandlerError(QueryboostError):
    """Raised when a batch handler error occurs."""


class QueryboostServerError(QueryboostError):
    """Raised when a server-side error occurs."""


def clean_flight_error_message(flight_error: FlightError) -> str:
    """Extract clean error message from PyArrow Flight errors.

    Removes gRPC debug context and unnecessary prefixes from Flight error messages
    to provide cleaner, more user-friendly error output.

    Args:
        flight_error: The Flight error exception to clean

    Returns:
        Cleaned error message string

    Example:
        >>> error = FlightError("Flight error: Invalid input. gRPC client debug context: ...")
        >>> clean_flight_error_message(error)
        "Invalid input"
    """

    message = str(flight_error)

    # Remove gRPC debug context
    message = message.split(". gRPC client debug context:")[0]

    # Remove Flight error prefix
    message = message.replace("Flight error: ", "")

    return message
