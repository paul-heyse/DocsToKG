"""
Smoke test for cache-hit and http-304 tokens in ContentDownload.

This test verifies that the ContentDownload module correctly emits
cache-aware attempt tokens when responses are served from cache or
when servers respond with 304 Not Modified.

To run:
  pytest -xvs tests/content_download/test_smoke_cache_tokens.py

To add to CI:
  pytest tests/content_download/test_smoke_cache_tokens.py -q
"""

import pytest
from unittest.mock import MagicMock, patch
from DocsToKG.ContentDownload.download_execution import stream_candidate_payload
from DocsToKG.ContentDownload.api.types import DownloadPlan


class TestCacheHitTokens:
    """Verify cache-hit and http-304 tokens are emitted."""

    def test_cache_hit_token_emitted(self):
        """Verify 'cache-hit' token is emitted for pure cache hits."""
        # Mock response that indicates cache hit (from_cache=True, revalidated=False)
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.extensions = {"from_cache": True, "revalidated": False}
        mock_resp.iter_bytes = MagicMock(return_value=[b"test data"])
        mock_session.get.return_value = mock_resp

        # Mock telemetry sink to capture attempts
        emitted_attempts = []
        mock_telemetry = MagicMock()
        mock_telemetry.log_attempt = lambda **kwargs: emitted_attempts.append(kwargs)

        # Create download plan
        plan = DownloadPlan(
            work_id="test-work",
            artifact_id="test-artifact",
            resolver_name="test-resolver",
            url="https://example.com/test.pdf",
        )

        # Execute stream
        with patch("builtins.open", MagicMock()):
            stream_candidate_payload(
                plan,
                session=mock_session,
                telemetry=mock_telemetry,
                run_id="test-run",
            )

        # Verify cache-hit token was emitted
        cache_hit_attempts = [
            a for a in emitted_attempts if a.get("status") == "cache-hit"
        ]
        assert (
            len(cache_hit_attempts) > 0
        ), "Expected 'cache-hit' token to be emitted for cache hits"

    def test_http_304_token_emitted(self):
        """Verify 'http-304' token is emitted for 304 Not Modified responses."""
        # Mock response indicating 304 (revalidated=True, status=304)
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 304
        mock_resp.headers = {"ETag": "some-etag"}
        mock_resp.extensions = {"from_cache": False, "revalidated": True}
        mock_resp.iter_bytes = MagicMock(return_value=[])
        mock_session.get.return_value = mock_resp

        # Mock telemetry sink
        emitted_attempts = []
        mock_telemetry = MagicMock()
        mock_telemetry.log_attempt = lambda **kwargs: emitted_attempts.append(kwargs)

        # Create download plan
        plan = DownloadPlan(
            work_id="test-work",
            artifact_id="test-artifact",
            resolver_name="test-resolver",
            url="https://example.com/test.pdf",
        )

        # Execute stream
        with patch("builtins.open", MagicMock()):
            stream_candidate_payload(
                plan,
                session=mock_session,
                telemetry=mock_telemetry,
                run_id="test-run",
            )

        # Verify http-304 token was emitted
        http_304_attempts = [
            a for a in emitted_attempts if a.get("status") == "http-304"
        ]
        assert (
            len(http_304_attempts) > 0
        ), "Expected 'http-304' token to be emitted for 304 responses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
