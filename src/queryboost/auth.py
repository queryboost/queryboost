from typing import BinaryIO

import pyarrow.flight as flight


class ApiKeyClientAuth(flight.ClientAuthHandler):
    """The Flight client authentication handler for API keys."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        """ :meta private: """

        self._token: str | None = None
        """ :meta private: """

    def authenticate(self, outgoing: BinaryIO, incoming: BinaryIO) -> None:
        """Conduct the handshake with the server to get a JWT for API calls."""

        outgoing.write(self._api_key.encode("utf-8"))

        self._token = incoming.read().decode("utf-8")
        """ :meta private: """

    def get_token(self) -> str | None:
        """Get the JWT for API calls."""

        return self._token
