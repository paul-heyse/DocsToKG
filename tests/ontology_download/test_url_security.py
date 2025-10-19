"""URL safety and allowlist enforcement tests.

Exercises ``validate_url_security`` against IPv4/v6 edge cases, private ranges,
port overrides, IPv6 literals, and DNS stubbing to ensure resolvers can only
contact approved endpoints.
"""

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


def test_clear_dns_stubs_purges_cached_stub_results():
    """Clearing DNS stubs should also evict any cached stubbed resolutions."""

    host = "cache-test.example.org"
    stub_result: List[Tuple] = [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.0.2.10", 0))
    ]

    network_mod.register_dns_stub(host, lambda _host: stub_result)

    def _unexpected_getaddrinfo(*args, **kwargs):  # pragma: no cover - defensive
        pytest.fail("socket.getaddrinfo should not be called while stub is active")

    original_getaddrinfo = network_mod.socket.getaddrinfo
    try:
        network_mod.socket.getaddrinfo = _unexpected_getaddrinfo  # type: ignore[assignment]

        first_lookup = network_mod._cached_getaddrinfo(host)
        assert first_lookup == stub_result

        network_mod.clear_dns_stubs()

        real_result: List[Tuple] = [
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("2001:db8::1", 0, 0, 0))
        ]
        calls: List[str] = []

        def _real_getaddrinfo(host_arg, port):
            calls.append(host_arg)
            assert port is None
            return real_result

        network_mod.socket.getaddrinfo = _real_getaddrinfo  # type: ignore[assignment]

        second_lookup = network_mod._cached_getaddrinfo(host)
        assert second_lookup == real_result
        assert calls == [host]
    finally:
        network_mod.socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
