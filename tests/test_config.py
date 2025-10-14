import pytest

from queryboost.config import ConfigBuilder
from queryboost.exceptions import QueryboostConfigurationError
from queryboost.config.config import Config


class TestConfigBuilder:
    """Test suite for ConfigBuilder class."""

    def test_build_with_all_parameters(self):
        """Test building config with all parameters provided."""
        config = ConfigBuilder.build(
            api_key="test_api_key",
            url="grpc+tls://test.example.com",
            port=8443,
        )

        assert isinstance(config, Config)
        assert config.api_key == "test_api_key"
        assert config.url == "grpc+tls://test.example.com"
        assert config.port == 8443

    def test_build_with_minimal_parameters(self, monkeypatch):
        """Test building config with only api_key provided."""
        monkeypatch.delenv("QUERYBOOST_API_KEY", raising=False)
        monkeypatch.delenv("QUERYBOOST_URL", raising=False)
        monkeypatch.delenv("QUERYBOOST_PORT", raising=False)

        config = ConfigBuilder.build(api_key="test_key")

        assert config.api_key == "test_key"
        assert config.url == "grpc+tls://api.queryboost.com"
        assert config.port == 443

    def test_build_with_env_variables(self, monkeypatch):
        """Test building config from environment variables."""
        monkeypatch.setenv("QUERYBOOST_API_KEY", "env_api_key")
        monkeypatch.setenv("QUERYBOOST_URL", "grpc+tls://env.example.com")
        monkeypatch.setenv("QUERYBOOST_PORT", "9443")

        config = ConfigBuilder.build()

        assert config.api_key == "env_api_key"
        assert config.url == "grpc+tls://env.example.com"
        assert config.port == "9443"

    def test_parameter_overrides_env_variable(self, monkeypatch):
        """Test that explicit parameters override environment variables."""
        monkeypatch.setenv("QUERYBOOST_API_KEY", "env_key")
        monkeypatch.setenv("QUERYBOOST_URL", "grpc+tls://env.example.com")
        monkeypatch.setenv("QUERYBOOST_PORT", "9443")

        config = ConfigBuilder.build(
            api_key="param_key",
            url="grpc+tls://param.example.com",
            port=8443,
        )

        assert config.api_key == "param_key"
        assert config.url == "grpc+tls://param.example.com"
        assert config.port == 8443

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises QueryboostConfigurationError."""
        monkeypatch.delenv("QUERYBOOST_API_KEY", raising=False)

        with pytest.raises(QueryboostConfigurationError) as exc_info:
            ConfigBuilder.build()

        assert "API key" in str(exc_info.value)
        assert "QUERYBOOST_API_KEY" in str(exc_info.value)

    def test_partial_env_variables(self, monkeypatch):
        """Test with only some environment variables set."""
        monkeypatch.setenv("QUERYBOOST_API_KEY", "test_key")
        monkeypatch.delenv("QUERYBOOST_URL", raising=False)
        monkeypatch.setenv("QUERYBOOST_PORT", "8080")

        config = ConfigBuilder.build()

        assert config.api_key == "test_key"
        assert config.url == "grpc+tls://api.queryboost.com"  # Default
        assert config.port == "8080"

    def test_port_as_string(self):
        """Test that port can be provided as a string."""
        config = ConfigBuilder.build(
            api_key="test_key",
            port="8443",
        )

        assert config.port == "8443"
        assert isinstance(config.port, str)

    def test_port_as_int(self):
        """Test that port can be provided as an integer."""
        config = ConfigBuilder.build(
            api_key="test_key",
            port=8443,
        )

        assert config.port == 8443
        assert isinstance(config.port, int)

    def test_config_is_named_tuple(self):
        """Test that Config is a NamedTuple with expected attributes."""
        config = ConfigBuilder.build(api_key="test_key")

        assert hasattr(config, "api_key")
        assert hasattr(config, "url")
        assert hasattr(config, "port")
        assert isinstance(config, tuple)

    def test_config_immutability(self):
        """Test that Config is immutable (NamedTuple behavior)."""
        config = ConfigBuilder.build(api_key="test_key")

        with pytest.raises(AttributeError):
            config.api_key = "new_key"  # type: ignore
