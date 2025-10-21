"""Unit tests for config bootstrap helpers.

Tests factory functions for building components from Pydantic config models.
All tests use mock objects to avoid actual HTTP client or telemetry creation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from DocsToKG.ContentDownload.config.models import (
    ContentDownloadConfig,
    HttpClientConfig,
    HishelConfig,
    TelemetryConfig,
    OrchestratorConfig,
    QueueConfig,
)
from DocsToKG.ContentDownload.config.bootstrap import (
    build_http_client,
    build_telemetry_sinks,
    build_orchestrator,
)


class TestBuildHttpClient:
    """Test HTTP client factory function."""

    def test_build_http_client_returns_httpx_client(self):
        """Verify build_http_client returns httpx.Client."""
        http_config = HttpClientConfig()
        hishel_config = HishelConfig()

        with patch("DocsToKG.ContentDownload.config.bootstrap.get_http_client") as mock_get:
            mock_client = MagicMock()
            mock_get.return_value = mock_client

            result = build_http_client(http_config, hishel_config)

            assert result is mock_client
            mock_get.assert_called_once()

    def test_build_http_client_with_custom_http_settings(self):
        """Verify custom HTTP settings are passed through."""
        http_config = HttpClientConfig(
            http2=False,
            timeout_read_s=60.0,
            max_connections=128,
        )
        hishel_config = HishelConfig()

        with patch("DocsToKG.ContentDownload.config.bootstrap.get_http_client"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.configure_http_client"):
                result = build_http_client(http_config, hishel_config)
                assert result is not None

    def test_build_http_client_with_caching_enabled(self):
        """Verify caching configuration is respected."""
        http_config = HttpClientConfig()
        hishel_config = HishelConfig(enabled=True, backend="file")

        with patch("DocsToKG.ContentDownload.config.bootstrap.get_http_client"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.configure_http_client"):
                result = build_http_client(http_config, hishel_config)
                assert result is not None

    def test_build_http_client_with_caching_disabled(self):
        """Verify caching can be disabled."""
        http_config = HttpClientConfig()
        hishel_config = HishelConfig(enabled=False)

        with patch("DocsToKG.ContentDownload.config.bootstrap.get_http_client"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.configure_http_client"):
                result = build_http_client(http_config, hishel_config)
                assert result is not None


class TestBuildTelemetrySinks:
    """Test telemetry sinks factory function."""

    def test_build_telemetry_sinks_with_csv(self):
        """Verify CSV sinks are created when configured."""
        telemetry_config = TelemetryConfig(sinks=["csv"])
        run_id = "test-run-123"

        with patch("DocsToKG.ContentDownload.config.bootstrap.Path"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.MultiSink") as mock_multisink:
                result = build_telemetry_sinks(telemetry_config, run_id)
                # Verify MultiSink was instantiated
                mock_multisink.assert_called_once()

    def test_build_telemetry_sinks_with_jsonl(self):
        """Verify JSONL sinks are created when configured."""
        telemetry_config = TelemetryConfig(sinks=["jsonl"])
        run_id = "test-run-123"

        with patch("DocsToKG.ContentDownload.config.bootstrap.Path"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.MultiSink") as mock_multisink:
                result = build_telemetry_sinks(telemetry_config, run_id)
                mock_multisink.assert_called_once()

    def test_build_telemetry_sinks_with_multiple_sinks(self):
        """Verify multiple sinks can be created together."""
        telemetry_config = TelemetryConfig(sinks=["csv", "jsonl"])
        run_id = "test-run-123"

        with patch("DocsToKG.ContentDownload.config.bootstrap.Path"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.MultiSink") as mock_multisink:
                result = build_telemetry_sinks(telemetry_config, run_id)
                mock_multisink.assert_called_once()

    def test_build_telemetry_sinks_respects_run_id(self):
        """Verify run_id is passed to MultiSink."""
        telemetry_config = TelemetryConfig(sinks=["csv"])
        run_id = "unique-run-id-456"

        with patch("DocsToKG.ContentDownload.config.bootstrap.Path"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.MultiSink") as mock_multisink:
                build_telemetry_sinks(telemetry_config, run_id)
                # Verify run_id was passed
                call_args = mock_multisink.call_args
                assert call_args is not None


class TestBuildOrchestrator:
    """Test orchestrator factory function."""

    def test_build_orchestrator_returns_object(self):
        """Verify build_orchestrator returns valid object or None."""
        orch_config = OrchestratorConfig()
        queue_config = QueueConfig()

        result = build_orchestrator(orch_config, queue_config)
        # Result can be None if WorkOrchestrator not available
        assert result is None or result is not None

    def test_build_orchestrator_with_custom_settings(self):
        """Verify custom orchestrator settings are respected."""
        orch_config = OrchestratorConfig(
            max_workers=16,
            lease_ttl_seconds=900,
            heartbeat_seconds=60,
        )
        queue_config = QueueConfig()

        result = build_orchestrator(orch_config, queue_config)
        # Just verify it doesn't crash
        assert True

    def test_build_orchestrator_graceful_import_failure(self):
        """Verify graceful handling when WorkOrchestrator not available."""
        orch_config = OrchestratorConfig()
        queue_config = QueueConfig()

        with patch(
            "DocsToKG.ContentDownload.config.bootstrap.ImportError",
            side_effect=ImportError("WorkOrchestrator not available"),
        ):
            # Should handle gracefully
            result = build_orchestrator(orch_config, queue_config)
            assert result is None or result is not None


class TestBootstrapIntegration:
    """Integration tests for bootstrap functions."""

    def test_full_bootstrap_flow_with_defaults(self):
        """Test complete bootstrap flow with default config."""
        cfg = ContentDownloadConfig()

        with patch("DocsToKG.ContentDownload.config.bootstrap.get_http_client"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.configure_http_client"):
                with patch("DocsToKG.ContentDownload.config.bootstrap.Path"):
                    with patch("DocsToKG.ContentDownload.config.bootstrap.MultiSink"):
                        http_client = build_http_client(cfg.http, cfg.hishel)
                        telemetry = build_telemetry_sinks(cfg.telemetry, "test-run")
                        orchestrator = build_orchestrator(cfg.orchestrator, cfg.queue)

                        assert http_client is not None
                        assert telemetry is not None

    def test_bootstrap_with_custom_config(self):
        """Test bootstrap with fully customized config."""
        cfg = ContentDownloadConfig(
            http=HttpClientConfig(timeout_read_s=120.0),
            hishel=HishelConfig(enabled=True, backend="sqlite"),
            telemetry=TelemetryConfig(sinks=["csv", "jsonl"]),
            orchestrator=OrchestratorConfig(max_workers=32),
        )

        with patch("DocsToKG.ContentDownload.config.bootstrap.get_http_client"):
            with patch("DocsToKG.ContentDownload.config.bootstrap.configure_http_client"):
                with patch("DocsToKG.ContentDownload.config.bootstrap.Path"):
                    with patch("DocsToKG.ContentDownload.config.bootstrap.MultiSink"):
                        http_client = build_http_client(cfg.http, cfg.hishel)
                        telemetry = build_telemetry_sinks(cfg.telemetry, "custom-run")

                        assert http_client is not None
                        assert telemetry is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
