# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_url_security",
#   "purpose": "URL safety and allowlist enforcement tests.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""URL safety and allowlist enforcement tests.

Exercises ``validate_url_security`` against IPv4/v6 edge cases, private ranges,
port overrides, IPv6 literals, and DNS stubbing to ensure resolvers can only
contact approved endpoints."""

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

    # Hostnames resolving to private space are also rejected unless explicitly opted in.
    def _private_getaddrinfo(host: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.1.2.3", 0))]

    network_mod.register_dns_stub("internal.example.org", _private_getaddrinfo)
    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https://internal.example.org/data", DownloadConfiguration())


def test_allowlisted_domain_resolving_to_public_ip_is_allowed() -> None:
    """Allowlisted domains remain subject to DNS checks and pass for public IPs."""

    def _public_getaddrinfo(host: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0))]

    network_mod.register_dns_stub("allowed.example.org", _public_getaddrinfo)
    config = DownloadConfiguration(allowed_hosts=["allowed.example.org"])

    validated = api_mod.validate_url_security("https://allowed.example.org/data", config)
    assert validated.startswith("https://allowed.example.org")


def test_allowlisted_http_url_upgrades_without_opt_in() -> None:
    """HTTP URLs for allowlisted hosts are upgraded to HTTPS unless explicitly allowed."""

    def _public_getaddrinfo(host: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("203.0.113.5", 0))]

    network_mod.register_dns_stub("http-only.example.org", _public_getaddrinfo)
    config = DownloadConfiguration(allowed_hosts=["http-only.example.org"])

    validated = api_mod.validate_url_security("http://http-only.example.org/data", config)
    assert validated.startswith("https://http-only.example.org")


def test_allowlisted_domain_resolving_to_private_ip_is_rejected() -> None:
    """Allowlisted domains resolving to private space should still be blocked."""

    def _private_getaddrinfo(host: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 0))]

    network_mod.register_dns_stub("loopback.example.org", _private_getaddrinfo)
    config = DownloadConfiguration(allowed_hosts=["loopback.example.org"])

    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https://loopback.example.org/data", config)


def test_allowlisted_private_ip_literal_allowed_with_opt_in() -> None:
    """Explicit IP literals can opt into private networks when configured."""

    url = "https://10.1.2.3/secret"
    config = DownloadConfiguration(
        allowed_hosts=["10.1.2.3"], allow_private_networks_for_host_allowlist=True
    )

    validated = api_mod.validate_url_security(url, config)
    assert validated.startswith("https://10.1.2.3")


def test_allowlisted_domain_private_resolution_allowed_when_opted_in() -> None:
    """The opt-in flag restores legacy behaviour for DNS hostnames."""

    def _private_getaddrinfo(host: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.9.8.7", 0))]

    network_mod.register_dns_stub("legacy.example.org", _private_getaddrinfo)
    config = DownloadConfiguration(
        allowed_hosts=["legacy.example.org"],
        allow_private_networks_for_host_allowlist=True,
    )

    validated = api_mod.validate_url_security("https://legacy.example.org/data", config)
    assert validated.startswith("https://legacy.example.org")


def test_allow_plain_http_flag_allows_http_but_blocks_private_resolution() -> None:
    """Allowing HTTP for allowlisted hosts should not weaken private-IP checks."""

    def _public_getaddrinfo(host: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("198.51.100.23", 0))]

    host = "plain-http.example.org"
    config = DownloadConfiguration(
        allowed_hosts=[host],
        allow_plain_http_for_host_allowlist=True,
    )

    network_mod.register_dns_stub(host, _public_getaddrinfo)
    validated = api_mod.validate_url_security(f"http://{host}/data", config)
    assert validated.startswith(f"http://{host}")

    def _private_getaddrinfo(hostname: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.4.3.2", 0))]

    network_mod.register_dns_stub(host, _private_getaddrinfo)
    with pytest.raises(ConfigError):
        api_mod.validate_url_security(f"http://{host}/data", config)


def test_private_network_flag_keeps_https_upgrade() -> None:
    """Allowing private networks should not implicitly allow HTTP."""

    def _private_getaddrinfo(host: str) -> List[Tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.20.30.40", 0))]

    host = "legacy-http.example.org"
    network_mod.register_dns_stub(host, _private_getaddrinfo)
    config = DownloadConfiguration(
        allowed_hosts=[host],
        allow_private_networks_for_host_allowlist=True,
    )

    validated = api_mod.validate_url_security(f"http://{host}/data", config)
    assert validated.startswith(f"https://{host}")


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


def test_validate_url_security_rejects_missing_scheme():
    """URLs without schemes should be rejected."""

    with pytest.raises(ConfigError):
        api_mod.validate_url_security("example.com/path")


def test_validate_url_security_rejects_missing_netloc():
    """URLs without hostnames should be rejected."""

    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https:///path")


def test_validate_url_security_normalizes_backslashes():
    """Backslashes in authority and path should be normalized to forward slashes."""

    result = api_mod.validate_url_security("https://example.com\\path\\to\\file")
    assert result == "https://example.com/path/to/file"

    # Backslashes in netloc break URL parsing - this is expected behavior
    # The URL becomes malformed and gets normalized to just the host
    result = api_mod.validate_url_security("https://example.com\\:8080\\path")
    assert result == "https://example.com/"


def test_validate_url_security_trims_trailing_dots():
    """Trailing dots in hostnames should be trimmed."""

    result = api_mod.validate_url_security("https://example.com./path")
    assert result == "https://example.com/path"

    result = api_mod.validate_url_security("https://sub.example.com../path")
    assert result == "https://sub.example.com/path"


def test_validate_url_security_preserves_percent_encoding():
    """Percent-encoded characters should be preserved without lossy decoding."""

    result = api_mod.validate_url_security("https://example.com/path%20with%20spaces")
    assert result == "https://example.com/path%20with%20spaces"

    result = api_mod.validate_url_security("https://example.com/path?query%3Dvalue%26other")
    assert result == "https://example.com/path?query%3Dvalue%26other"


def test_validate_url_security_comprehensive_ipv6_handling():
    """IPv6 literals should be handled correctly with proper bracket normalization."""

    # IPv6 without brackets should be rejected
    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https://2001:db8::1/path")

    # IPv6 with brackets should work
    config = DownloadConfiguration(allowed_hosts=["[2001:db8::1]"])
    result = api_mod.validate_url_security("https://[2001:db8::1]/path", config)
    assert result == "https://[2001:db8::1]/path"

    # IPv6 with port
    config = DownloadConfiguration(allowed_hosts=["[2001:db8::1]:8443"])
    result = api_mod.validate_url_security("https://[2001:db8::1]:8443/path", config)
    assert result == "https://[2001:db8::1]:8443/path"

    # IPv6 with default port should omit port
    config = DownloadConfiguration(allowed_hosts=["[2001:db8::1]"])
    result = api_mod.validate_url_security("https://[2001:db8::1]:443/path", config)
    assert result == "https://[2001:db8::1]/path"


def test_validate_url_security_registrable_domain_wildcards():
    """Wildcard suffix matching should work with registrable domain awareness."""

    config = DownloadConfiguration(allowed_hosts=["*.example.org"])

    # Subdomain should match
    result = api_mod.validate_url_security("https://sub.example.org/path", config)
    assert result == "https://sub.example.org/path"

    # Nested subdomain should match
    result = api_mod.validate_url_security("https://deep.sub.example.org/path", config)
    assert result == "https://deep.sub.example.org/path"

    # Exact match should work
    config = DownloadConfiguration(allowed_hosts=["example.org", "*.example.org"])
    result = api_mod.validate_url_security("https://example.org/path", config)
    assert result == "https://example.org/path"

    # Non-matching domain should fail
    config = DownloadConfiguration(allowed_hosts=["*.example.org"])
    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https://other.com/path", config)


def test_validate_url_security_ip_literal_private_network_handling():
    """IP literals should properly handle private network classification."""

    # Private IP should be rejected by default
    with pytest.raises(ConfigError):
        api_mod.validate_url_security("https://10.0.0.1/path")

    # Private IP should be allowed when explicitly allowlisted
    config = DownloadConfiguration(allowed_hosts=["10.0.0.1"])
    result = api_mod.validate_url_security("https://10.0.0.1/path", config)
    assert result == "https://10.0.0.1/path"

    # Private IP should be allowed with private network flag
    config = DownloadConfiguration(
        allowed_hosts=["private.example.org"], allow_private_networks_for_host_allowlist=True
    )
    # This would require DNS stubbing to test properly
    network_mod.register_dns_stub(
        "private.example.org",
        lambda host: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.1", 0))
        ],
    )
    result = api_mod.validate_url_security("https://private.example.org/path", config)
    assert result == "https://private.example.org/path"


def test_validate_url_security_edge_case_paths():
    """Edge cases in path normalization should be handled correctly."""

    # Root path
    result = api_mod.validate_url_security("https://example.com/")
    assert result == "https://example.com/"

    # Empty path
    result = api_mod.validate_url_security("https://example.com")
    assert result == "https://example.com"

    # Path with multiple slashes
    result = api_mod.validate_url_security("https://example.com//path//to//file")
    assert result == "https://example.com//path//to//file"  # urljoin preserves multiple slashes

    # Path with encoded characters
    result = api_mod.validate_url_security("https://example.com/path%2Fwith%2Fencoded%2Fslashes")
    assert result == "https://example.com/path%2Fwith%2Fencoded%2Fslashes"


def test_validate_url_security_query_fragment_preservation():
    """Query and fragment should be preserved verbatim without reordering."""

    # Complex query with multiple parameters
    result = api_mod.validate_url_security(
        "https://example.com/path?param1=value1&param2=value2&param3=value3"
    )
    assert result == "https://example.com/path?param1=value1&param2=value2&param3=value3"

    # Query with fragment
    result = api_mod.validate_url_security("https://example.com/path?query=value#fragment")
    assert result == "https://example.com/path?query=value#fragment"

    # Fragment only
    result = api_mod.validate_url_security("https://example.com/path#fragment")
    assert result == "https://example.com/path#fragment"

    # Empty query and fragment - urljoin drops empty query/fragment
    result = api_mod.validate_url_security("https://example.com/path?#")
    assert result == "https://example.com/path"
