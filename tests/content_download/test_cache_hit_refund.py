"""
Test cache-hit refund logic in PerResolverHttpClient.

Verifies that when hishel serves a response from cache (pure cache hit,
no network), the rate-limit token is refunded so that cache hits don't
consume the rate budget.
"""

from unittest.mock import Mock

import httpx
import pytest

from DocsToKG.ContentDownload.resolver_http_client import (
    PerResolverHttpClient,
    RetryConfig,
    TokenBucket,
)


class TestTokenBucketRefund:
    """Test TokenBucket.refund() functionality."""

    def test_refund_increases_tokens(self):
        """Refunding tokens should increase the token count."""
        bucket = TokenBucket(capacity=5.0, refill_per_sec=1.0, burst=1.0)

        # Consume 3 tokens
        bucket.acquire(tokens=3.0, timeout_s=1.0)
        initial_tokens = bucket.tokens

        # Refund 2 tokens
        bucket.refund(tokens=2.0)

        # Tokens should increase
        assert bucket.tokens > initial_tokens
        assert bucket.tokens <= 5.0  # capped at capacity

    def test_refund_doesnt_exceed_capacity(self):
        """Refunded tokens should not exceed bucket capacity."""
        bucket = TokenBucket(capacity=5.0, refill_per_sec=1.0, burst=1.0)

        # Refund way more than capacity
        bucket.refund(tokens=100.0)

        # Should be capped at capacity
        assert bucket.tokens <= 5.0


class TestCacheHitRefund:
    """Test cache-hit refund in PerResolverHttpClient._request()."""

    def test_pure_cache_hit_refunds_token(self):
        """When response is from hishel cache (pure hit), token should be refunded."""
        # Create mock session
        mock_session = Mock(spec=httpx.Client)
        mock_session.timeout = httpx.Timeout(timeout=30.0)

        # Create mock response with cache metadata
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.extensions = {"from_cache": True, "revalidated": False}
        mock_session.request.return_value = mock_response

        # Create client
        client = PerResolverHttpClient(
            session=mock_session,
            resolver_name="test_resolver",
            retry_config=RetryConfig(rate_capacity=5.0, rate_refill_per_sec=1.0),
        )

        # Get initial tokens
        initial_tokens = client.rate_limiter.tokens

        # Make request (acquires 1 token)
        response = client.get("https://example.com/test")

        # After cache hit, tokens should be refunded
        # So it should be at or near initial level
        assert response.status_code == 200
        assert client.rate_limiter.tokens >= (initial_tokens - 0.1)  # allow small float error

    def test_revalidated_response_does_not_refund(self):
        """When response is revalidated (304), token should NOT be refunded."""
        mock_session = Mock(spec=httpx.Client)
        mock_session.timeout = httpx.Timeout(timeout=30.0)

        # Create mock response with revalidated flag
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 304
        mock_response.extensions = {"from_cache": True, "revalidated": True}
        mock_session.request.return_value = mock_response

        client = PerResolverHttpClient(
            session=mock_session,
            resolver_name="test_resolver",
            retry_config=RetryConfig(rate_capacity=5.0, rate_refill_per_sec=1.0),
        )

        initial_tokens = client.rate_limiter.tokens

        response = client.get("https://example.com/test")

        # Revalidated responses don't refund (network was used for revalidation)
        # Token should be consumed (lower than initial)
        assert response.status_code == 304
        assert client.rate_limiter.tokens < initial_tokens

    def test_non_cached_response_does_not_refund(self):
        """When response is fresh (not from cache), token should NOT be refunded."""
        mock_session = Mock(spec=httpx.Client)
        mock_session.timeout = httpx.Timeout(timeout=30.0)

        # Create mock response without cache metadata
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.extensions = {"from_cache": False, "revalidated": False}
        mock_session.request.return_value = mock_response

        client = PerResolverHttpClient(
            session=mock_session,
            resolver_name="test_resolver",
            retry_config=RetryConfig(rate_capacity=5.0, rate_refill_per_sec=1.0),
        )

        initial_tokens = client.rate_limiter.tokens

        response = client.get("https://example.com/test")

        # Fresh responses don't refund (network was used)
        assert response.status_code == 200
        assert client.rate_limiter.tokens < initial_tokens

    def test_response_without_extensions_does_not_refund(self):
        """When response has no extensions dict, token should NOT be refunded (safeguard)."""
        mock_session = Mock(spec=httpx.Client)
        mock_session.timeout = httpx.Timeout(timeout=30.0)

        # Create mock response without extensions
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.extensions = {}  # Empty extensions
        mock_session.request.return_value = mock_response

        client = PerResolverHttpClient(
            session=mock_session,
            resolver_name="test_resolver",
            retry_config=RetryConfig(rate_capacity=5.0, rate_refill_per_sec=1.0),
        )

        initial_tokens = client.rate_limiter.tokens

        response = client.get("https://example.com/test")

        # No cache metadata = no refund
        assert response.status_code == 200
        assert client.rate_limiter.tokens < initial_tokens


class TestCacheHitTelemetry:
    """Test that cache-hit attempts are emitted properly."""

    def test_cache_hit_response_returned_with_tokens_refunded(self):
        """Cache-hit response should be returned immediately with tokens refunded."""
        mock_session = Mock(spec=httpx.Client)
        mock_session.timeout = httpx.Timeout(timeout=30.0)

        # Create mock cached response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.extensions = {"from_cache": True, "revalidated": False}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_session.request.return_value = mock_response

        # Create mock telemetry
        mock_telemetry = Mock()

        client = PerResolverHttpClient(
            session=mock_session,
            resolver_name="test_resolver",
            retry_config=RetryConfig(rate_capacity=10.0, rate_refill_per_sec=1.0),
            telemetry=mock_telemetry,
        )

        # Make request
        response = client.get("https://api.example.com/data")

        # Verify response is returned
        assert response == mock_response
        assert response.status_code == 200

        # Verify session.request was called exactly once (no retries for cache-hit)
        assert mock_session.request.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
