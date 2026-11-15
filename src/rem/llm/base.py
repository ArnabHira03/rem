from abc import ABC, abstractmethod
from ipaddress import ip_address
from urllib.parse import urlparse


class LLM(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier such as "anthropic:claude-haiku-5-5"."""

    @abstractmethod
    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """Return a JSON object matching `schema`. Raises LLMError on failure."""


def is_loopback(url: str) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False
