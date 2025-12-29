from unittest.mock import Mock, patch

import pytest
import pyarrow.flight as flight

from queryboost import Queryboost
from queryboost.exceptions import QueryboostError, QueryboostConfigurationError
from queryboost.config.config import Config


class TestQueryboost:
    """Test suite for Queryboost client class."""

    @patch("queryboost.queryboost.flight.FlightClient")
    def test_initialization_with_api_key(self, mock_flight_client):
        """Test Queryboost initialization with API key."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        client = Queryboost(api_key="test_key")

        assert client._config.api_key == "test_key"
        assert client._config.url == "grpc+tls://api.queryboost.com"
        assert client._config.port == 443
        assert client._location == "grpc+tls://api.queryboost.com:443"

        # Verify FlightClient was created with location and tls_root_certs
        call_kwargs = mock_flight_client.call_args[1]
        assert call_kwargs["location"] == client._location
        assert "tls_root_certs" in call_kwargs

        # Verify authentication was called
        mock_client.authenticate.assert_called_once()

    @patch("queryboost.queryboost.flight.FlightClient")
    def test_initialization_with_all_params(self, mock_flight_client):
        """Test Queryboost initialization with all parameters."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        client = Queryboost(
            api_key="my_key",
            url="grpc+tls://custom.host.com",
            port=9443,
        )

        assert client._config.api_key == "my_key"
        assert client._config.url == "grpc+tls://custom.host.com"
        assert client._config.port == 9443
        assert client._location == "grpc+tls://custom.host.com:9443"

    @patch("queryboost.queryboost.flight.FlightClient")
    def test_initialization_without_api_key_raises_error(self, _mock_flight_client):
        """Test that missing API key raises error."""
        with pytest.raises(QueryboostConfigurationError):
            Queryboost()

    @patch("queryboost.queryboost.flight.FlightClient")
    def test_initialization_creates_auth_handler(self, mock_flight_client):
        """Test that initialization creates ApiKeyClientAuth."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        client = Queryboost(api_key="test_key")

        assert client._auth is not None
        assert client._auth._api_key == "test_key"

    @patch("queryboost.queryboost.flight.FlightClient")
    @patch("queryboost.queryboost.BatchStreamer")
    @patch("queryboost.queryboost.DataBatcher")
    @patch("queryboost.queryboost.validate_prompt")
    def test_run_with_default_handler(
        self,
        mock_validate,
        mock_data_batcher_cls,
        mock_batch_streamer_cls,
        mock_flight_client,
    ):
        """Test run method with default batch handler."""
        # Setup mocks
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        mock_data_batcher = Mock()
        mock_data_batcher.schema.names = ["col1", "col2"]
        mock_data_batcher.num_rows = 1000
        mock_data_batcher_cls.return_value = mock_data_batcher

        mock_batch_streamer = Mock()
        mock_batch_streamer_cls.return_value = mock_batch_streamer

        # Create client
        client = Queryboost(api_key="test_key")

        # Run
        data = [{"col1": "a", "col2": "b"}]
        prompt = "Process {col1} and {col2}"

        client.run(data=data, prompt=prompt)

        # Verify DataBatcher was created
        mock_data_batcher_cls.assert_called_once_with(data, 16)

        # Verify prompt validation
        mock_validate.assert_called_once_with(prompt, ["col1", "col2"])

        # Verify BatchStreamer was created
        assert mock_batch_streamer_cls.called

        # Verify stream was called
        mock_batch_streamer.stream.assert_called_once()

    @patch("queryboost.queryboost.flight.FlightClient")
    @patch("queryboost.queryboost.BatchStreamer")
    @patch("queryboost.queryboost.DataBatcher")
    @patch("queryboost.queryboost.validate_prompt")
    def test_run_with_custom_batch_size(
        self,
        _mock_validate,
        mock_data_batcher_cls,
        mock_batch_streamer_cls,
        mock_flight_client,
    ):
        """Test run method with custom batch size."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        mock_data_batcher = Mock()
        mock_data_batcher.schema.names = ["text"]
        mock_data_batcher.num_rows = 1000
        mock_data_batcher_cls.return_value = mock_data_batcher

        mock_batch_streamer = Mock()
        mock_batch_streamer_cls.return_value = mock_batch_streamer

        client = Queryboost(api_key="test_key")

        data = [{"text": "hello"}]
        prompt = "Analyze {text}"

        client.run(data=data, prompt=prompt, batch_size=64)

        # Verify custom batch size was used
        mock_data_batcher_cls.assert_called_once_with(data, 64)

    @patch("queryboost.queryboost.flight.FlightClient")
    @patch("queryboost.queryboost.BatchStreamer")
    @patch("queryboost.queryboost.DataBatcher")
    @patch("queryboost.queryboost.validate_prompt")
    def test_run_with_custom_handler(
        self,
        _mock_validate,
        mock_data_batcher_cls,
        mock_batch_streamer_cls,
        mock_flight_client,
    ):
        """Test run method with custom batch handler."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        mock_data_batcher = Mock()
        mock_data_batcher.schema.names = ["text"]
        mock_data_batcher.num_rows = 1000
        mock_data_batcher_cls.return_value = mock_data_batcher

        mock_batch_streamer = Mock()
        mock_batch_streamer_cls.return_value = mock_batch_streamer

        client = Queryboost(api_key="test_key")

        # Custom handler
        custom_handler = Mock()

        data = [{"text": "test"}]
        prompt = "Process {text}"

        client.run(data=data, prompt=prompt, batch_handler=custom_handler)

        # Verify BatchStreamer was created with custom handler
        call_args = mock_batch_streamer_cls.call_args
        assert call_args[0][1] == custom_handler

    @patch("queryboost.queryboost.flight.FlightClient")
    @patch("queryboost.queryboost.BatchStreamer")
    @patch("queryboost.queryboost.DataBatcher")
    @patch("queryboost.queryboost.validate_prompt")
    def test_run_with_custom_name(
        self,
        _mock_validate,
        mock_data_batcher_cls,
        mock_batch_streamer_cls,
        mock_flight_client,
    ):
        """Test run method with custom name parameter."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        mock_data_batcher = Mock()
        mock_data_batcher.schema.names = ["text"]
        mock_data_batcher.num_rows = 1000
        mock_data_batcher_cls.return_value = mock_data_batcher

        mock_batch_streamer = Mock()
        mock_batch_streamer_cls.return_value = mock_batch_streamer

        client = Queryboost(api_key="test_key")

        data = [{"text": "test"}]
        prompt = "Process {text}"

        client.run(data=data, prompt=prompt, name="my_custom_name")

        # The name is used in the default handler creation
        # We can't easily verify it without inspecting the handler
        # but we ensure no errors occur
        assert mock_batch_streamer.stream.called

    @patch("queryboost.queryboost.flight.FlightClient")
    @patch("queryboost.queryboost.DataBatcher")
    @patch("queryboost.queryboost.validate_prompt")
    def test_run_creates_flight_descriptor(
        self,
        _mock_validate,
        mock_data_batcher_cls,
        mock_flight_client,
    ):
        """Test that run creates proper FlightDescriptor with command."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        mock_data_batcher = Mock()
        mock_data_batcher.schema.names = ["text"]
        mock_data_batcher.num_rows = 1000
        mock_data_batcher_cls.return_value = mock_data_batcher

        with patch("queryboost.queryboost.BatchStreamer") as mock_streamer_cls:
            mock_streamer = Mock()
            mock_streamer_cls.return_value = mock_streamer

            client = Queryboost(api_key="test_key")

            data = [{"text": "test"}]
            prompt = "Process {text}"

            client.run(data=data, prompt=prompt)

            # Verify stream was called with client and descriptor
            assert mock_streamer.stream.called
            call_args = mock_streamer.stream.call_args[0]
            assert call_args[0] == mock_client
            assert isinstance(call_args[1], flight.FlightDescriptor)

    @patch("queryboost.queryboost.flight.FlightClient")
    def test_config_is_built_correctly(self, mock_flight_client):
        """Test that config is built using ConfigBuilder."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        client = Queryboost(
            api_key="test_key",
            url="grpc+tls://test.com",
            port=8080,
        )

        assert isinstance(client._config, Config)
        assert client._config.api_key == "test_key"
        assert client._config.url == "grpc+tls://test.com"
        assert client._config.port == 8080

    @patch("queryboost.queryboost.flight.FlightClient")
    @patch("queryboost.queryboost.BatchStreamer")
    @patch("queryboost.queryboost.DataBatcher")
    @patch("queryboost.queryboost.validate_prompt")
    def test_run_with_num_gpus(
        self,
        _mock_validate,
        mock_data_batcher_cls,
        mock_batch_streamer_cls,
        mock_flight_client,
    ):
        """Test run method with num_gpus parameter.

        Validation of num_gpus (e.g., max GPU limits) is done server-side.
        """
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        mock_data_batcher = Mock()
        mock_data_batcher.schema.names = ["text"]
        mock_data_batcher.num_rows = 1000
        mock_data_batcher_cls.return_value = mock_data_batcher

        mock_batch_streamer = Mock()
        mock_batch_streamer_cls.return_value = mock_batch_streamer

        client = Queryboost(api_key="test_key")

        data = [{"text": "test"}]
        prompt = "Process {text}"

        # Test with num_gpus specified
        client.run(data=data, prompt=prompt, num_gpus=4)

        # Verify it doesn't raise an error and stream is called
        assert mock_batch_streamer.stream.called

    @patch("queryboost.queryboost.flight.FlightClient")
    def test_run_raises_error_when_both_name_and_handler_provided(self, mock_flight_client):
        """Test that providing both name and batch_handler raises QueryboostError."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        client = Queryboost(api_key="test_key")

        # Create a mock handler
        custom_handler = Mock()

        data = [{"text": "test"}]
        prompt = "Process {text}"

        # Verify that providing both name and batch_handler raises error
        with pytest.raises(QueryboostError, match="Cannot specify both 'name' and 'batch_handler'"):
            client.run(data=data, prompt=prompt, name="my-run", batch_handler=custom_handler)

    @patch("queryboost.queryboost.flight.FlightClient")
    @patch("queryboost.queryboost.BatchStreamer")
    @patch("queryboost.queryboost.DataBatcher")
    @patch("queryboost.queryboost.validate_prompt")
    @patch("queryboost.queryboost.json.dumps")
    def test_run_with_json_schema(
        self,
        mock_json_dumps,
        _mock_validate,
        mock_data_batcher_cls,
        mock_batch_streamer_cls,
        mock_flight_client,
    ):
        """Test run method with json_schema parameter."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        mock_data_batcher = Mock()
        mock_data_batcher.schema.names = ["text"]
        mock_data_batcher.num_rows = 1000
        mock_data_batcher_cls.return_value = mock_data_batcher

        mock_batch_streamer = Mock()
        mock_batch_streamer_cls.return_value = mock_batch_streamer

        mock_json_dumps.return_value = '{"test": "json"}'

        client = Queryboost(api_key="test_key")

        data = [{"text": "test"}]
        prompt = "Process {text}"
        json_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
            "required": ["answer"],
        }

        client.run(data=data, prompt=prompt, json_schema=json_schema)

        # Verify json_schema was included in the command dict
        call_args = mock_json_dumps.call_args[0]
        command_dict = call_args[0]
        assert command_dict["json_schema"] == json_schema
        assert command_dict["prompt"] == prompt
        assert command_dict["num_rows"] == 1000

    @patch("queryboost.queryboost.flight.FlightClient")
    @patch("queryboost.queryboost.BatchStreamer")
    @patch("queryboost.queryboost.DataBatcher")
    @patch("queryboost.queryboost.validate_prompt")
    @patch("queryboost.queryboost.json.dumps")
    def test_run_without_json_schema(
        self,
        mock_json_dumps,
        _mock_validate,
        mock_data_batcher_cls,
        mock_batch_streamer_cls,
        mock_flight_client,
    ):
        """Test run method without json_schema parameter (None by default)."""
        mock_client = Mock()
        mock_flight_client.return_value = mock_client

        mock_data_batcher = Mock()
        mock_data_batcher.schema.names = ["text"]
        mock_data_batcher.num_rows = 1000
        mock_data_batcher_cls.return_value = mock_data_batcher

        mock_batch_streamer = Mock()
        mock_batch_streamer_cls.return_value = mock_batch_streamer

        mock_json_dumps.return_value = '{"test": "json"}'

        client = Queryboost(api_key="test_key")

        data = [{"text": "test"}]
        prompt = "Process {text}"

        client.run(data=data, prompt=prompt)

        # Verify json_schema is None in the command dict
        call_args = mock_json_dumps.call_args[0]
        command_dict = call_args[0]
        assert command_dict["json_schema"] is None
        assert command_dict["prompt"] == prompt
        assert command_dict["num_rows"] == 1000
