"""Tests for networking layer breaker integration."""

import pytest
from unittest.mock import Mock, patch
import httpx

try:
    import pybreaker
except ImportError:
    pybreaker = None
    pytest.skip("pybreaker not available", allow_module_level=True)

from DocsToKG.ContentDownload.networking import (
    BreakerOpenError,
    get_breaker_registry,
    request_with_retries,
    reset_breaker_registry,
    set_breaker_registry,
)
from DocsToKG.ContentDownload.breakers import (
    BreakerRegistry,
    BreakerConfig,
    BreakerPolicy,
    RequestRole,
)


class TestNetworkingBreakerIntegration:
    """Test breaker integration in networking layer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset global breaker registry
        reset_breaker_registry()

    def teardown_method(self):
        """Clean up after tests."""
        # Reset global breaker registry
        reset_breaker_registry()

    @patch("DocsToKG.ContentDownload.networking.get_http_client")
    def test_request_with_retries_no_breaker(self, mock_get_client):
        """Test request_with_retries when no breaker registry is set."""
        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.extensions = {}
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Should work normally without breaker
        response = request_with_retries(None, "GET", "https://example.com")
        assert response == mock_response

    @patch("DocsToKG.ContentDownload.networking.get_http_client")
    def test_request_with_retries_breaker_allows(self, mock_get_client):
        """Test request_with_retries when breaker allows the request."""
        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.extensions = {}
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Set up breaker registry
        config = BreakerConfig()
        registry = BreakerRegistry(config)
        set_breaker_registry(registry)

        # Should work normally
        response = request_with_retries(None, "GET", "https://example.com")
        assert response == mock_response

    @patch("DocsToKG.ContentDownload.networking.get_http_client")
    def test_request_with_retries_breaker_blocks(self, mock_get_client):
        """Test request_with_retries when breaker blocks the request."""
        # Set up breaker registry with low threshold
        config = BreakerConfig(defaults=BreakerPolicy(fail_max=1, reset_timeout_s=60))
        registry = BreakerRegistry(config)

        # Trigger failure to open breaker
        registry.on_failure("example.com", role=RequestRole.METADATA, status=503)
        set_breaker_registry(registry)

        # Should raise BreakerOpenError
        with pytest.raises(BreakerOpenError):
            request_with_retries(None, "GET", "https://example.com")

    @patch("DocsToKG.ContentDownload.networking.get_http_client")
    def test_request_with_retries_updates_breaker_on_success(self, mock_get_client):
        """Test that successful requests update breaker state."""
        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.extensions = {}
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Set up breaker registry
        config = BreakerConfig(defaults=BreakerPolicy(fail_max=1, reset_timeout_s=60))
        registry = BreakerRegistry(config)
        set_breaker_registry(registry)

        # Mock successful response
        with patch.object(registry, "on_success") as mock_on_success:
            response = request_with_retries(None, "GET", "https://example.com")
            assert response == mock_response
            mock_on_success.assert_called_once()

    @patch("DocsToKG.ContentDownload.networking.get_http_client")
    def test_request_with_retries_updates_breaker_on_failure(self, mock_get_client):
        """Test that failed requests update breaker state."""
        # Mock HTTP client returning failure status
        mock_client = Mock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.headers = {}
        mock_response.extensions = {}
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Set up breaker registry
        config = BreakerConfig()
        registry = BreakerRegistry(config)
        set_breaker_registry(registry)

        # Mock on_failure to verify it's called
        with patch.object(registry, "on_failure") as mock_on_failure:
            response = request_with_retries(
                None, "GET", "https://example.com", max_retries=0
            )
            assert response == mock_response

            # Should have called on_failure
            mock_on_failure.assert_called_once()
            kwargs = mock_on_failure.call_args.kwargs
            assert kwargs.get("status") == 503

    @patch("DocsToKG.ContentDownload.networking.get_http_client")
    def test_request_with_retries_handles_retry_after(self, mock_get_client):
        """Test that Retry-After headers are handled correctly."""
        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "5"}
        mock_response.extensions = {}
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Set up breaker registry
        config = BreakerConfig()
        registry = BreakerRegistry(config)
        set_breaker_registry(registry)

        # Mock on_failure to verify retry_after_s is passed
        with patch.object(registry, "on_failure") as mock_on_failure:
            request_with_retries(None, "GET", "https://example.com")

            # Should have called on_failure with retry_after_s
            mock_on_failure.assert_called_once()
            call_args = mock_on_failure.call_args
            assert call_args[1]["retry_after_s"] == 5.0

    @patch("DocsToKG.ContentDownload.networking.get_http_client")
    def test_request_with_retries_resolver_parameter(self, mock_get_client):
        """Test that resolver parameter is passed to breaker."""
        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.extensions = {}
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Set up breaker registry
        config = BreakerConfig()
        registry = BreakerRegistry(config)
        set_breaker_registry(registry)

        # Mock allow to verify resolver parameter
        with patch.object(registry, "allow") as mock_allow:
            request_with_retries(None, "GET", "https://example.com", resolver="test_resolver")

            # Should have called allow with resolver
            mock_allow.assert_called_once()
            call_args = mock_allow.call_args
            assert call_args[1]["resolver"] == "test_resolver"

    @patch("DocsToKG.ContentDownload.networking.get_http_client")
    def test_request_with_retries_role_mapping(self, mock_get_client):
        """Test that role strings are mapped to RequestRole enum."""
        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.extensions = {}
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Set up breaker registry
        config = BreakerConfig()
        registry = BreakerRegistry(config)
        set_breaker_registry(registry)

        # Mock allow to verify role mapping
        with patch.object(registry, "allow") as mock_allow:
            # Test different role mappings
            request_with_retries(None, "GET", "https://example.com", role="metadata")
            call_args = mock_allow.call_args
            assert call_args[1]["role"] == RequestRole.METADATA

            request_with_retries(None, "GET", "https://example.com", role="landing")
            call_args = mock_allow.call_args
            assert call_args[1]["role"] == RequestRole.LANDING

            request_with_retries(None, "GET", "https://example.com", role="artifact")
            call_args = mock_allow.call_args
            assert call_args[1]["role"] == RequestRole.ARTIFACT

    def test_breaker_registry_global_functions(self):
        """Test global breaker registry functions."""
        # Initially no registry
        assert get_breaker_registry() is None

        # Set registry
        config = BreakerConfig()
        registry = BreakerRegistry(config)
        set_breaker_registry(registry)

        # Should return the registry
        assert get_breaker_registry() is registry

        # Reset registry
        set_breaker_registry(None)
        assert get_breaker_registry() is None
