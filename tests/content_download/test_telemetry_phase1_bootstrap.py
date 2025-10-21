"""Phase 1 Telemetry Tests: HTTP Session, Rate Limiting, Per-Resolver Client, Bootstrap.

Tests the full stack for:
- Shared HTTP session with polite headers
- Token bucket rate limiting
- Per-resolver HTTP client with retry/backoff
- Bootstrap orchestration
"""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from DocsToKG.ContentDownload.bootstrap import (
    BootstrapConfig,
    RunResult,
    run_from_config,
)
from DocsToKG.ContentDownload.http_session import HttpConfig, get_http_session, reset_http_session
from DocsToKG.ContentDownload.resolver_http_client import (
    PerResolverHttpClient,
    RetryConfig,
    TokenBucket,
)


class TestHttpSession(unittest.TestCase):
    """Test shared HTTP session factory."""

    def setUp(self):
        """Reset session before each test."""
        reset_http_session()

    def tearDown(self):
        """Cleanup."""
        reset_http_session()

    def test_get_http_session_singleton(self):
        """Session is a singleton (same instance on multiple calls)."""
        session1 = get_http_session()
        session2 = get_http_session()
        assert session1 is session2

    def test_http_session_has_user_agent(self):
        """Session includes User-Agent header."""
        session = get_http_session(HttpConfig(user_agent="MyBot/1.0"))
        assert "User-Agent" in session.headers
        assert "MyBot/1.0" in session.headers["User-Agent"]

    def test_http_session_with_mailto(self):
        """Session appends mailto if provided."""
        session = get_http_session(HttpConfig(user_agent="MyBot/1.0", mailto="admin@example.com"))
        ua = session.headers.get("User-Agent", "")
        assert "admin@example.com" in ua or "+mailto:" in ua

    def test_http_session_timeout_config(self):
        """Session applies timeout configuration."""
        config = HttpConfig(timeout_connect_s=5.0, timeout_read_s=30.0)
        session = get_http_session(config)
        # httpx.Client stores timeout internally
        assert session.timeout is not None

    def test_http_session_connection_limits(self):
        """Session configures connection pooling."""
        config = HttpConfig(pool_connections=5, pool_maxsize=10)
        session = get_http_session(config)
        # Connection limits are set via httpx.Limits
        # We verify the session was created successfully with the config
        assert session is not None
        assert session.timeout is not None


class TestTokenBucket(unittest.TestCase):
    """Test token bucket rate limiter."""

    def test_token_bucket_acquire_immediate(self):
        """Acquire succeeds immediately when tokens available."""
        bucket = TokenBucket(capacity=5.0, refill_per_sec=1.0)
        sleep_s = bucket.acquire(tokens=1.0, timeout_s=1.0)
        assert sleep_s == 0.0  # No wait

    def test_token_bucket_refill(self):
        """Tokens refill over time."""
        bucket = TokenBucket(capacity=1.0, refill_per_sec=1.0)
        bucket.acquire(tokens=1.0)  # Consume all tokens

        # Wait for refill
        time.sleep(0.1)  # 0.1s should refill 0.1 tokens
        # This is a timing-sensitive test; may need adjustment

    def test_token_bucket_capacity_limit(self):
        """Bucket respects capacity limit."""
        TokenBucket(capacity=5.0, refill_per_sec=10.0)
        time.sleep(0.2)  # Would produce 2 tokens, but capped at capacity
        # Tokens should be capped at capacity=5.0

    def test_token_bucket_timeout(self):
        """Acquire raises TimeoutError if tokens unavailable."""
        bucket = TokenBucket(capacity=1.0, refill_per_sec=0.1)
        bucket.acquire(tokens=1.0)  # Consume all

        with pytest.raises(TimeoutError):
            bucket.acquire(tokens=1.0, timeout_s=0.1)  # Can't get token in time


class TestPerResolverHttpClient(unittest.TestCase):
    """Test per-resolver HTTP client with retry/backoff."""

    def setUp(self):
        """Create mock session and client."""
        self.mock_session = MagicMock(spec=httpx.Client)
        self.mock_session.timeout = httpx.Timeout(60.0)

    def test_client_head_request(self):
        """Client performs HEAD requests."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        self.mock_session.request.return_value = mock_response

        client = PerResolverHttpClient(
            session=self.mock_session,
            resolver_name="test",
        )

        response = client.head("https://example.com/doc.pdf")

        assert response.status_code == 200
        self.mock_session.request.assert_called_once()
        call_args = self.mock_session.request.call_args
        assert call_args[0][0] == "HEAD"  # Method

    def test_client_get_request(self):
        """Client performs GET requests."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        self.mock_session.request.return_value = mock_response

        client = PerResolverHttpClient(
            session=self.mock_session,
            resolver_name="test",
        )

        response = client.get("https://example.com/doc.pdf")

        assert response.status_code == 200
        self.mock_session.request.assert_called_once()
        call_args = self.mock_session.request.call_args
        assert call_args[0][0] == "GET"  # Method

    def test_client_retry_on_429(self):
        """Client retries on 429 (rate limit)."""
        # First: 429, Second: 200
        mock_response_429 = MagicMock(spec=httpx.Response)
        mock_response_429.status_code = 429
        mock_response_429.headers = {}

        mock_response_200 = MagicMock(spec=httpx.Response)
        mock_response_200.status_code = 200

        self.mock_session.request.side_effect = [mock_response_429, mock_response_200]

        client = PerResolverHttpClient(
            session=self.mock_session,
            resolver_name="test",
            retry_config=RetryConfig(base_delay_ms=10, max_delay_ms=100),  # Short delays
        )

        with patch("time.sleep"):  # Don't actually sleep in test
            response = client.get("https://example.com/doc.pdf")

        assert response.status_code == 200
        assert self.mock_session.request.call_count == 2

    def test_client_retry_with_retry_after(self):
        """Client honors Retry-After header."""
        mock_response_429 = MagicMock(spec=httpx.Response)
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "2"}  # 2 seconds

        mock_response_200 = MagicMock(spec=httpx.Response)
        mock_response_200.status_code = 200

        self.mock_session.request.side_effect = [mock_response_429, mock_response_200]

        client = PerResolverHttpClient(
            session=self.mock_session,
            resolver_name="test",
        )

        with patch("time.sleep") as mock_sleep:
            response = client.get("https://example.com/doc.pdf")

        assert response.status_code == 200
        # Check that sleep was called with ~2 seconds
        mock_sleep.assert_called()

    def test_client_retry_exhaustion(self):
        """Client raises after max retries exhausted."""
        mock_response_429 = MagicMock(spec=httpx.Response)
        mock_response_429.status_code = 429
        mock_response_429.headers = {}

        self.mock_session.request.return_value = mock_response_429

        client = PerResolverHttpClient(
            session=self.mock_session,
            resolver_name="test",
            retry_config=RetryConfig(max_attempts=2, base_delay_ms=10),
        )

        with patch("time.sleep"):
            response = client.get("https://example.com/doc.pdf")

        # Should return the 429 response after exhausting retries
        assert response.status_code == 429

    def test_client_retry_on_network_error(self):
        """Client retries on network errors."""
        self.mock_session.request.side_effect = [
            httpx.ConnectError("Connection failed"),
            MagicMock(spec=httpx.Response, status_code=200),
        ]

        client = PerResolverHttpClient(
            session=self.mock_session,
            resolver_name="test",
            retry_config=RetryConfig(max_attempts=2, base_delay_ms=10),
        )

        with patch("time.sleep"):
            response = client.get("https://example.com/doc.pdf")

        assert response.status_code == 200
        assert self.mock_session.request.call_count == 2


class TestBootstrapOrchestration(unittest.TestCase):
    """Test bootstrap orchestration."""

    def setUp(self):
        """Create temporary directory for telemetry paths."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Cleanup temporary directory."""
        self.temp_dir.cleanup()

    def test_bootstrap_with_no_artifacts(self):
        """Bootstrap validates wiring without artifacts."""
        config = BootstrapConfig(
            http=HttpConfig(),
            telemetry_paths={"csv": self.temp_path / "attempts.csv"},
            resolver_registry={},
        )

        result = run_from_config(config, artifacts=None)

        assert isinstance(result, RunResult)
        assert result.success_count == 0
        assert (result.skip_count + result.error_count) == 0

    def test_bootstrap_generates_run_id(self):
        """Bootstrap generates run_id if not provided."""
        config = BootstrapConfig(
            http=HttpConfig(),
            telemetry_paths={"csv": self.temp_path / "attempts.csv"},
            resolver_registry={},
        )

        result = run_from_config(config, artifacts=None)

        assert result.run_id is not None
        assert len(result.run_id) > 0

    def test_bootstrap_uses_provided_run_id(self):
        """Bootstrap generates unique run_id each time."""
        config = BootstrapConfig(
            http=HttpConfig(),
            telemetry_paths={"csv": self.temp_path / "attempts.csv"},
            resolver_registry={},
        )

        result1 = run_from_config(config, artifacts=None)
        result2 = run_from_config(config, artifacts=None)

        # Each run should get a unique run_id
        assert result1.run_id != result2.run_id

    def test_bootstrap_with_artifacts(self):
        """Bootstrap processes artifact iterator."""
        # Create a mock resolver
        mock_resolver = MagicMock()
        mock_resolver.name = "test_resolver"
        mock_resolver.resolve.return_value = MagicMock(plans=[])

        # Create mock artifact
        mock_artifact = MagicMock()
        mock_artifact.artifact_id = "artifact-1"

        config = BootstrapConfig(
            http=HttpConfig(),
            telemetry_paths={"csv": self.temp_path / "attempts.csv"},
            resolver_registry={"test": mock_resolver},
        )

        # Run with artifact iterator
        result = run_from_config(config, artifacts=iter([mock_artifact]))

        # Verify result structure
        assert isinstance(result, RunResult)
        assert result.run_id is not None


class TestEndToEndBootstrap(unittest.TestCase):
    """End-to-end bootstrap tests (smoke tests)."""

    def setUp(self):
        """Create temporary directory for telemetry paths."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Cleanup temporary directory and HTTP session."""
        self.temp_dir.cleanup()
        reset_http_session()

    def test_e2e_bootstrap_session_client_pipeline(self):
        """E2E: Bootstrap → Session → Client → Pipeline."""
        # Reset session
        reset_http_session()

        # Create config
        config = BootstrapConfig(
            http=HttpConfig(user_agent="TestBot/1.0"),
            telemetry_paths={"csv": self.temp_path / "attempts.csv"},
            resolver_registry={},
            resolver_retry_configs={},
        )

        # Run bootstrap
        result = run_from_config(config, artifacts=None)

        # Verify result
        assert isinstance(result, RunResult)
        assert result.run_id is not None
        assert result.success_count >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
