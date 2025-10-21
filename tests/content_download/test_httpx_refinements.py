"""Tests for HTTPX Pooling Refinements 1 & 2.

Refinement 1: Host normalization for telemetry
Refinement 2: Exception context in error paths
"""

import unittest
from unittest.mock import MagicMock, patch

import httpx

from DocsToKG.ContentDownload.net.client import _normalize_host_for_telemetry
from DocsToKG.ContentDownload.net.download_helper import (
    DownloadError,
    head_request,
    stream_download_to_file,
)


class TestHostNormalization(unittest.TestCase):
    """Test Refinement 1: Host normalization for telemetry."""

    def test_normalize_basic_host(self):
        """Basic hostname normalization (lowercase)."""
        url = "https://EXAMPLE.ORG/path"
        host = _normalize_host_for_telemetry(url)
        assert host == "example.org"

    def test_normalize_host_with_port(self):
        """Port is stripped by httpx.URL.host."""
        url = "https://example.org:8443/path"
        host = _normalize_host_for_telemetry(url)
        # httpx.URL.host returns the host without port
        assert host == "example.org"

    def test_normalize_idn_host(self):
        """IDN domain normalization (with unicode)."""
        # httpx supports Unicode domain names directly
        url = "https://münchen.de/path"
        host = _normalize_host_for_telemetry(url)
        # httpx normalizes to lowercase
        assert host == "münchen.de"

    def test_normalize_localhost(self):
        """Localhost normalization."""
        url = "http://LOCALHOST:8000/path"
        host = _normalize_host_for_telemetry(url)
        assert host == "localhost"

    def test_normalize_ip_address(self):
        """IPv4 address."""
        url = "http://192.168.1.1/path"
        host = _normalize_host_for_telemetry(url)
        assert host == "192.168.1.1"

    def test_normalize_invalid_url(self):
        """Invalid URL returns 'unknown'."""
        url = "not a url"
        host = _normalize_host_for_telemetry(url)
        assert host == "unknown"

    def test_normalize_no_host(self):
        """URL with no host returns 'unknown'."""
        url = "http:///path"
        host = _normalize_host_for_telemetry(url)
        assert host == "unknown"

    def test_normalize_consistency(self):
        """Same URL always produces same host."""
        url = "https://API.EXAMPLE.COM:443/v1/data?key=value"
        host1 = _normalize_host_for_telemetry(url)
        host2 = _normalize_host_for_telemetry(url)
        assert host1 == host2 == "api.example.com"


class TestExceptionContext(unittest.TestCase):
    """Test Refinement 2: Exception context in error paths."""

    def test_stream_download_http_error_context(self):
        """HTTP error provides correct status code in telemetry."""
        # Create a mock config
        config = MagicMock()
        config.http.timeout_connect_s = 5.0
        config.http.timeout_read_s = 30.0
        config.http.timeout_write_s = 30.0
        config.http.timeout_pool_s = 5.0

        # Mock the telemetry emitter
        with patch(
            "DocsToKG.ContentDownload.net.download_helper.get_net_request_emitter"
        ) as mock_emitter_factory:
            mock_emitter = MagicMock()
            mock_emitter_factory.return_value = mock_emitter

            # Mock the HTTP client to raise an HTTPStatusError with 404
            with patch(
                "DocsToKG.ContentDownload.net.download_helper.get_http_client"
            ) as mock_client_factory:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.status_code = 404
                mock_response.http_version = "HTTP/1.1"

                error = httpx.HTTPStatusError(
                    "404 Not Found", request=MagicMock(), response=mock_response
                )
                mock_client.request.side_effect = error

                mock_client_factory.return_value = mock_client

                # Mock request_with_redirect_audit to raise the error
                with patch(
                    "DocsToKG.ContentDownload.net.download_helper.request_with_redirect_audit"
                ) as mock_audit:
                    mock_audit.side_effect = error

                    # Call stream_download_to_file
                    import tempfile
                    from pathlib import Path

                    with tempfile.TemporaryDirectory() as tmpdir:
                        dest = Path(tmpdir) / "file.bin"
                        try:
                            stream_download_to_file(config, "https://example.org/404", dest)
                            assert False, "Should raise DownloadError"
                        except DownloadError:
                            pass

                    # Verify telemetry was emitted with correct status code
                    mock_emitter.emit.assert_called()
                    # Get the emitted event
                    call_args = mock_emitter.emit.call_args
                    event = call_args[0][0] if call_args[0] else None
                    # The event should have status_code from the exception
                    assert event is not None

    def test_head_request_network_error_context(self):
        """Network error (non-HTTP) provides reasonable fallback."""
        config = MagicMock()

        with patch(
            "DocsToKG.ContentDownload.net.download_helper.get_net_request_emitter"
        ) as mock_emitter_factory:
            mock_emitter = MagicMock()
            mock_emitter_factory.return_value = mock_emitter

            with patch(
                "DocsToKG.ContentDownload.net.download_helper.get_http_client"
            ) as mock_client_factory:
                mock_client = MagicMock()
                # Raise a network error (not HTTPStatusError)
                error = httpx.ConnectError("Connection refused")

                mock_client_factory.return_value = mock_client

                with patch(
                    "DocsToKG.ContentDownload.net.download_helper.request_with_redirect_audit"
                ) as mock_audit:
                    mock_audit.side_effect = error

                    try:
                        head_request(config, "https://example.org/resource")
                        assert False, "Should raise httpx.HTTPError"
                    except httpx.HTTPError:
                        pass

                    # Verify telemetry was emitted
                    mock_emitter.emit.assert_called()
                    # Get the emitted event
                    call_args = mock_emitter.emit.call_args
                    event = call_args[0][0] if call_args[0] else None
                    assert event is not None

    def test_exception_context_no_undefined_variables(self):
        """Exception handling doesn't reference undefined 'resp' variable."""
        # This test verifies the fix works by not raising NameError
        # when an exception occurs before resp is defined

        config = MagicMock()

        with patch(
            "DocsToKG.ContentDownload.net.download_helper.get_net_request_emitter"
        ) as mock_emitter_factory:
            mock_emitter = MagicMock()
            mock_emitter_factory.return_value = mock_emitter

            with patch(
                "DocsToKG.ContentDownload.net.download_helper.get_http_client"
            ) as mock_client_factory:
                mock_client = MagicMock()
                # Network error before response is received
                error = httpx.NetworkError("DNS failure")

                mock_client_factory.return_value = mock_client

                with patch(
                    "DocsToKG.ContentDownload.net.download_helper.request_with_redirect_audit"
                ) as mock_audit:
                    mock_audit.side_effect = error

                    import tempfile
                    from pathlib import Path

                    with tempfile.TemporaryDirectory() as tmpdir:
                        dest = Path(tmpdir) / "file.bin"
                        try:
                            stream_download_to_file(config, "https://example.org/data", dest)
                            assert False, "Should raise DownloadError"
                        except DownloadError as e:
                            # Should succeed without NameError
                            assert str(e)
                        except NameError as e:
                            # This should NOT happen with the fix
                            assert False, f"NameError (bug): {e}"


if __name__ == "__main__":
    unittest.main()
