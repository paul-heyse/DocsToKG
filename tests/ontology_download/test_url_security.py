"""URL safety tests for validate_url_security."""

from __future__ import annotations

import socket
from typing import List, Tuple

import pytest

from DocsToKG.OntologyDownload import api as api_mod
from DocsToKG.OntologyDownload.errors import ConfigError
from DocsToKG.OntologyDownload.io import network as network_mod
from DocsToKG.OntologyDownload.settings import DownloadConfiguration


@pytest.fixture(autouse=True)
def _clear_dns_stubs():
    network_mod.clear_dns_stubs()
    yield
    network_mod.clear_dns_stubs()


def _fake_getaddrinfo(address: str) -> List[Tuple]:
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0))]


def test_validate_url_security_rejects_mixed_script_idn() -> None:
    """IDNs mixing confusable scripts should raise ConfigError."""

    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https://раypal.com/resource")


def test_validate_url_security_requires_port_allowlist() -> None:
    """Non-default ports require the host to be explicitly allowlisted."""

    network_mod.register_dns_stub("example.org", lambda host: _fake_getaddrinfo(host))
    url = "https://example.org:8443/data"

    with pytest.raises(ConfigError):
        api_mod.validate_url_security(url, DownloadConfiguration())

    config = DownloadConfiguration(allowed_hosts=["example.org:8443"])
    validated = api_mod.validate_url_security(url, config)
    assert validated.endswith(":8443/data")


def test_validate_url_security_blocks_private_cidr() -> None:
    """Private IP literals should be rejected unless explicitly allowed."""

    url = "https://10.0.0.5/file.owl"
    with pytest.raises(ConfigError):
        api_mod.validate_url_security(url, DownloadConfiguration())

    # Hostnames resolving to private space are also rejected unless allowlisted.
    def _private_getaddrinfo(host: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.1.2.3", 0))]

    network_mod.register_dns_stub("internal.example.org", _private_getaddrinfo)
    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https://internal.example.org/data", DownloadConfiguration())

    allowed_config = DownloadConfiguration(allowed_hosts=["internal.example.org"])
    validated = api_mod.validate_url_security("https://internal.example.org/data", allowed_config)
    assert validated.startswith("https://internal.example.org")


def test_validate_url_security_dns_failure_strict_mode() -> None:
    """DNS lookups should raise when strict_dns is enabled."""

    def _failing_getaddrinfo(host: str):
        raise socket.gaierror("host not found")

    network_mod.register_dns_stub("missing.example.org", _failing_getaddrinfo)

    strict_config = DownloadConfiguration(strict_dns=True)
    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https://missing.example.org/data", strict_config)

    permissive_config = DownloadConfiguration(strict_dns=False)
    result = api_mod.validate_url_security("https://missing.example.org/data", permissive_config)
    assert result.startswith("https://missing.example.org")
