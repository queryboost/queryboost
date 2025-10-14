import io

from queryboost.auth import ApiKeyClientAuth


class TestApiKeyClientAuth:
    """Test suite for ApiKeyClientAuth class."""

    def test_initialization(self):
        """Test ApiKeyClientAuth initialization."""
        auth = ApiKeyClientAuth("test_api_key")

        assert auth._api_key == "test_api_key"
        assert auth._token is None

    def test_authenticate_handshake(self):
        """Test authenticate method performs handshake correctly."""
        auth = ApiKeyClientAuth("my_secret_key")

        # Mock the outgoing and incoming streams
        outgoing = io.BytesIO()
        incoming = io.BytesIO(b"jwt_token_from_server")

        # Perform authentication
        auth.authenticate(outgoing, incoming)

        # Verify the API key was written to outgoing stream
        outgoing.seek(0)
        assert outgoing.read() == b"my_secret_key"

        # Verify the token was read from incoming stream
        assert auth._token == "jwt_token_from_server"

    def test_get_token_before_authentication(self):
        """Test get_token returns None before authentication."""
        auth = ApiKeyClientAuth("test_key")

        assert auth.get_token() is None

    def test_get_token_after_authentication(self):
        """Test get_token returns token after authentication."""
        auth = ApiKeyClientAuth("test_key")

        # Simulate authentication
        outgoing = io.BytesIO()
        incoming = io.BytesIO(b"received_token")
        auth.authenticate(outgoing, incoming)

        assert auth.get_token() == "received_token"

    def test_authenticate_with_empty_token(self):
        """Test authenticate with empty token from server."""
        auth = ApiKeyClientAuth("test_key")

        outgoing = io.BytesIO()
        incoming = io.BytesIO(b"")
        auth.authenticate(outgoing, incoming)

        assert auth._token == ""
        assert auth.get_token() == ""

    def test_authenticate_with_special_characters(self):
        """Test authenticate with special characters in API key and token."""
        special_key = "key_with_!@#$%^&*()"
        auth = ApiKeyClientAuth(special_key)

        outgoing = io.BytesIO()
        special_token = "token_with_特殊字符"
        incoming = io.BytesIO(special_token.encode("utf-8"))

        auth.authenticate(outgoing, incoming)

        outgoing.seek(0)
        assert outgoing.read() == special_key.encode("utf-8")
        assert auth._token == special_token

    def test_multiple_authentications(self):
        """Test that multiple authentications update the token."""
        auth = ApiKeyClientAuth("test_key")

        # First authentication
        outgoing1 = io.BytesIO()
        incoming1 = io.BytesIO(b"first_token")
        auth.authenticate(outgoing1, incoming1)
        assert auth.get_token() == "first_token"

        # Second authentication (token should be updated)
        outgoing2 = io.BytesIO()
        incoming2 = io.BytesIO(b"second_token")
        auth.authenticate(outgoing2, incoming2)
        assert auth.get_token() == "second_token"
