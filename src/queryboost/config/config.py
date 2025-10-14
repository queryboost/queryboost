import os
import logging
from typing import Optional, NamedTuple

from queryboost.exceptions import QueryboostConfigurationError

logger = logging.getLogger(__name__)
""" :meta private: """


DEFAULT_URL = "grpc+tls://api.queryboost.com"
DEFAULT_PORT = 443


class Config(NamedTuple):
    api_key: str
    url: str
    port: int | str


class ConfigBuilder:
    """The ConfigBuilder class handles configuration resolution for the Queryboost client.

    Configurations are resolved in the following order of precedence:
    1. Configs passed directly as keyword parameters to build()
    2. Configs specified in environment variables:
       - QUERYBOOST_API_KEY
       - QUERYBOOST_URL
       - QUERYBOOST_PORT
    3. Default values (for url and port only)
    """

    @staticmethod
    def build(
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int | str] = None,
    ) -> Config:
        api_key = api_key or os.getenv("QUERYBOOST_API_KEY")
        url = url or os.getenv("QUERYBOOST_URL") or DEFAULT_URL
        port = port or os.getenv("QUERYBOOST_PORT") or DEFAULT_PORT

        if not api_key:
            raise QueryboostConfigurationError(
                "You haven't specified an API key. Please either set the QUERYBOOST_API_KEY environment variable or pass the 'api_key' keyword argument to the Queryboost client constructor."
            )

        return Config(
            api_key=api_key,
            url=url,
            port=port,
        )
