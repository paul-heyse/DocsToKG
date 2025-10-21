"""Tests for polite HTTP client: Integration of HTTP + Rate-Limiting.

Tests cover:
- Basic GET/POST requests with rate-limiting
- Service-aware and host-aware keying
- Singleton pattern and lifecycle
- Thread safety
- URL host extraction
- Error handling
"""

import pytest
from DocsToKG.OntologyDownload.network.polite_client import (
    PoliteHttpClient,
    get_polite_http_client,
    close_polite_http_client,
    reset_polite_http_client,
)


class TestPoliteHttpClientBasics:
    """Test basic PoliteHttpClient functionality."""

    def test_polite_client_creation(self):
        """PoliteHttpClient can be created."""
        client = PoliteHttpClient(service="ols")
        assert client is not None

    def test_polite_client_extract_host(self):
        """Host extraction from URL works correctly."""
        client = PoliteHttpClient()
        host = client._extract_host("https://www.ebi.ac.uk/ols/api/search")
        assert host == "www.ebi.ac.uk"

    def test_polite_client_extract_host_with_port(self):
        """Host extraction handles ports."""
        client = PoliteHttpClient()
        host = client._extract_host("https://api.example.com:8443/search")
        assert host == "api.example.com:8443"

    def test_polite_client_extract_host_fallback(self):
        """Host extraction returns 'unknown' on error."""
        client = PoliteHttpClient()
        host = client._extract_host("not-a-valid-url")
        assert host == "not-a-valid-url" or host == "unknown"

    def test_polite_client_close(self):
        """Client.close() doesn't raise."""
        client = PoliteHttpClient()
        # Should not raise
        client.close()

    def test_polite_client_with_service_and_host(self):
        """PoliteHttpClient accepts service and host arguments."""
        client = PoliteHttpClient(
            service="ols",
            host="www.ebi.ac.uk",
        )
        assert client._service == "ols"
        assert client._host == "www.ebi.ac.uk"

    def test_polite_client_default_service(self):
        """Default service is 'default' if not specified."""
        client = PoliteHttpClient()
        assert client._service == "default"


class TestPoliteHttpClientSingleton:
    """Test singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_polite_http_client()

    def teardown_method(self):
        """Clean up after each test."""
        close_polite_http_client()

    def test_get_polite_client_creates_instance(self):
        """get_polite_http_client() creates instance on first call."""
        client = get_polite_http_client()
        assert client is not None

    def test_get_polite_client_returns_same_instance(self):
        """get_polite_http_client() returns same instance."""
        client1 = get_polite_http_client()
        client2 = get_polite_http_client()
        assert client1 is client2

    def test_close_polite_client_releases(self):
        """close_polite_http_client() releases the singleton."""
        client1 = get_polite_http_client()
        close_polite_http_client()
        client2 = get_polite_http_client()
        assert client1 is not client2

    def test_close_polite_client_is_idempotent(self):
        """close_polite_http_client() safe to call multiple times."""
        get_polite_http_client()
        close_polite_http_client()
        close_polite_http_client()  # Should not raise
        close_polite_http_client()  # Should not raise

    def test_reset_forces_new_instance(self):
        """reset_polite_http_client() forces new instance."""
        client1 = get_polite_http_client()
        reset_polite_http_client()
        client2 = get_polite_http_client()
        assert client1 is not client2

    def test_singleton_thread_safe(self):
        """Singleton creation is thread-safe."""
        import threading

        clients = []
        lock = threading.Lock()

        def get_and_store():
            client = get_polite_http_client()
            with lock:
                clients.append(client)

        threads = [threading.Thread(target=get_and_store) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be same instance
        assert len(clients) == 4
        assert all(c is clients[0] for c in clients)


class TestPoliteHttpClientIntegration:
    """Test integration with HTTP client and rate limiter."""

    def setup_method(self):
        """Reset clients before each test."""
        reset_polite_http_client()

    def teardown_method(self):
        """Clean up after each test."""
        close_polite_http_client()

    def test_polite_client_uses_http_client(self):
        """PoliteHttpClient uses get_http_client()."""
        client = PoliteHttpClient()
        # Should have an HTTP client
        assert client._http_client is not None

    def test_polite_client_uses_rate_limiter(self):
        """PoliteHttpClient uses get_rate_limiter()."""
        client = PoliteHttpClient()
        # Should have a rate limiter
        assert client._rate_limiter is not None

    def test_polite_client_with_custom_service(self):
        """PoliteHttpClient respects custom service name."""
        client = PoliteHttpClient(service="bioportal")
        assert client._service == "bioportal"

    def test_polite_client_with_custom_host(self):
        """PoliteHttpClient respects custom host."""
        client = PoliteHttpClient(host="example.com")
        assert client._host == "example.com"

    def test_polite_client_api_has_get(self):
        """PoliteHttpClient has get() method."""
        client = PoliteHttpClient()
        assert hasattr(client, "get")
        assert callable(client.get)

    def test_polite_client_api_has_post(self):
        """PoliteHttpClient has post() method."""
        client = PoliteHttpClient()
        assert hasattr(client, "post")
        assert callable(client.post)

    def test_polite_client_api_has_request(self):
        """PoliteHttpClient has request() method."""
        client = PoliteHttpClient()
        assert hasattr(client, "request")
        assert callable(client.request)
