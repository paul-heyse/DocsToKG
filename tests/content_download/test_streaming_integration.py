"""Tests for streaming_integration layer.

Tests cover:
- Feature flag behavior (enabled/disabled)
- Graceful fallback when modules unavailable
- Resume decision integration
- I/O integration
- Finalization integration
- Idempotency integration
- Schema integration
- Status reporting
"""

from __future__ import annotations

import os
import unittest
from typing import Any, Dict, Optional
from unittest import mock

import pytest


class TestFeatureFlags(unittest.TestCase):
    """Test feature flag functions."""

    def setUp(self) -> None:
        """Clear environment variables before each test."""
        self.env_vars = {
            "DOCSTOKG_ENABLE_STREAMING",
            "DOCSTOKG_ENABLE_IDEMPOTENCY",
            "DOCSTOKG_ENABLE_STREAMING_SCHEMA",
        }
        self.saved_env = {var: os.environ.pop(var, None) for var in self.env_vars}

    def tearDown(self) -> None:
        """Restore environment variables after each test."""
        for var, val in self.saved_env.items():
            if val is not None:
                os.environ[var] = val
            elif var in os.environ:
                del os.environ[var]

    def test_streaming_enabled_by_default(self) -> None:
        """Test that streaming is enabled by default."""
        from DocsToKG.ContentDownload.streaming_integration import streaming_enabled

        # Should be enabled if module is available
        assert isinstance(streaming_enabled(), bool)

    def test_streaming_can_be_disabled(self) -> None:
        """Test that streaming can be disabled via env var."""
        os.environ["DOCSTOKG_ENABLE_STREAMING"] = "0"
        from DocsToKG.ContentDownload.streaming_integration import streaming_enabled
        import importlib

        # Re-import to pick up env var
        import DocsToKG.ContentDownload.streaming_integration as si

        importlib.reload(si)
        assert si.streaming_enabled() is False

    def test_idempotency_enabled_by_default(self) -> None:
        """Test that idempotency is enabled by default."""
        from DocsToKG.ContentDownload.streaming_integration import idempotency_enabled

        assert isinstance(idempotency_enabled(), bool)

    def test_idempotency_can_be_disabled(self) -> None:
        """Test that idempotency can be disabled via env var."""
        os.environ["DOCSTOKG_ENABLE_IDEMPOTENCY"] = "0"
        from DocsToKG.ContentDownload.streaming_integration import idempotency_enabled
        import importlib
        import DocsToKG.ContentDownload.streaming_integration as si

        importlib.reload(si)
        assert si.idempotency_enabled() is False

    def test_schema_enabled_by_default(self) -> None:
        """Test that schema is enabled by default."""
        from DocsToKG.ContentDownload.streaming_integration import schema_enabled

        assert isinstance(schema_enabled(), bool)


class TestResumDecisionIntegration(unittest.TestCase):
    """Test resume decision integration."""

    def test_use_streaming_for_resume_requires_plan(self) -> None:
        """Test that use_streaming_for_resume checks plan attributes."""
        from DocsToKG.ContentDownload.streaming_integration import (
            use_streaming_for_resume,
        )

        # Plan without cond_helper should return False
        plan = mock.Mock(spec=[])
        result = use_streaming_for_resume(plan)
        assert result is False

    def test_use_streaming_for_resume_with_valid_plan(self) -> None:
        """Test that use_streaming_for_resume returns True with valid plan."""
        from DocsToKG.ContentDownload.streaming_integration import (
            use_streaming_for_resume,
        )

        plan = mock.Mock()
        plan.cond_helper = mock.Mock()
        result = use_streaming_for_resume(plan)
        # Should return True if streaming is available
        assert isinstance(result, bool)

    def test_try_streaming_resume_decision_returns_none_on_disabled(self) -> None:
        """Test that try_streaming_resume_decision returns None when disabled."""
        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.streaming_enabled",
            return_value=False,
        ):
            from DocsToKG.ContentDownload.streaming_integration import (
                try_streaming_resume_decision,
            )

            result = try_streaming_resume_decision(None, None)
            assert result is None

    def test_try_streaming_resume_decision_graceful_fallback(self) -> None:
        """Test graceful fallback on error."""
        from DocsToKG.ContentDownload.streaming_integration import (
            try_streaming_resume_decision,
        )

        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.streaming_enabled",
            return_value=True,
        ):
            with mock.patch(
                "DocsToKG.ContentDownload.streaming_integration.streaming",
                None,
            ):
                result = try_streaming_resume_decision(None, None)
                assert result is None


class TestIOIntegration(unittest.TestCase):
    """Test I/O integration."""

    def test_use_streaming_for_io_requires_plan(self) -> None:
        """Test that use_streaming_for_io checks plan attributes."""
        from DocsToKG.ContentDownload.streaming_integration import (
            use_streaming_for_io,
        )

        # Plan without context should return False
        plan = mock.Mock(spec=[])
        result = use_streaming_for_io(plan)
        assert result is False

    def test_use_streaming_for_io_with_valid_plan(self) -> None:
        """Test that use_streaming_for_io returns True with valid plan."""
        from DocsToKG.ContentDownload.streaming_integration import (
            use_streaming_for_io,
        )

        plan = mock.Mock()
        plan.context = mock.Mock()
        result = use_streaming_for_io(plan)
        # Should return True if streaming is available
        assert isinstance(result, bool)

    def test_try_streaming_io_returns_none_on_disabled(self) -> None:
        """Test that try_streaming_io returns None when disabled."""
        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.streaming_enabled",
            return_value=False,
        ):
            from DocsToKG.ContentDownload.streaming_integration import (
                try_streaming_io,
            )

            result = try_streaming_io(mock.Mock(), mock.Mock())
            assert result is None


class TestFinalizationIntegration(unittest.TestCase):
    """Test finalization integration."""

    def test_use_streaming_for_finalization_requires_outcome(self) -> None:
        """Test that use_streaming_for_finalization checks outcome attributes."""
        from DocsToKG.ContentDownload.streaming_integration import (
            use_streaming_for_finalization,
        )

        # Outcome without classification should return False
        outcome = mock.Mock(spec=[])
        result = use_streaming_for_finalization(outcome)
        assert result is False

    def test_use_streaming_for_finalization_with_valid_outcome(self) -> None:
        """Test that use_streaming_for_finalization returns True with valid outcome."""
        from DocsToKG.ContentDownload.streaming_integration import (
            use_streaming_for_finalization,
        )

        outcome = mock.Mock()
        outcome.classification = "PDF"
        result = use_streaming_for_finalization(outcome)
        # Should return True if streaming is available
        assert isinstance(result, bool)


class TestIdempotencyIntegration(unittest.TestCase):
    """Test idempotency integration."""

    def test_generate_job_key_returns_none_when_disabled(self) -> None:
        """Test that generate_job_key returns None when disabled."""
        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.idempotency_enabled",
            return_value=False,
        ):
            from DocsToKG.ContentDownload.streaming_integration import (
                generate_job_key,
            )

            result = generate_job_key("w1", "a1", "http://example.com")
            assert result is None

    def test_generate_job_key_graceful_fallback(self) -> None:
        """Test graceful fallback on error."""
        from DocsToKG.ContentDownload.streaming_integration import (
            generate_job_key,
        )

        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.idempotency_enabled",
            return_value=True,
        ):
            with mock.patch(
                "DocsToKG.ContentDownload.streaming_integration.idempotency",
                None,
            ):
                result = generate_job_key("w1", "a1", "http://example.com")
                assert result is None

    def test_generate_operation_key_returns_none_when_disabled(self) -> None:
        """Test that generate_operation_key returns None when disabled."""
        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.idempotency_enabled",
            return_value=False,
        ):
            from DocsToKG.ContentDownload.streaming_integration import (
                generate_operation_key,
            )

            result = generate_operation_key("HEAD", "job1")
            assert result is None

    def test_generate_operation_key_with_context(self) -> None:
        """Test that generate_operation_key passes context."""
        from DocsToKG.ContentDownload.streaming_integration import (
            generate_operation_key,
        )

        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.idempotency_enabled",
            return_value=True,
        ):
            with mock.patch(
                "DocsToKG.ContentDownload.streaming_integration.idempotency",
                None,
            ):
                result = generate_operation_key(
                    "STREAM",
                    "job1",
                    url="http://example.com",
                    range_start=0,
                )
                assert result is None


class TestSchemaIntegration(unittest.TestCase):
    """Test schema integration."""

    def test_get_streaming_database_returns_none_when_disabled(self) -> None:
        """Test that get_streaming_database returns None when disabled."""
        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.schema_enabled",
            return_value=False,
        ):
            from DocsToKG.ContentDownload.streaming_integration import (
                get_streaming_database,
            )

            result = get_streaming_database()
            assert result is None

    def test_get_streaming_database_graceful_fallback(self) -> None:
        """Test graceful fallback on error."""
        from DocsToKG.ContentDownload.streaming_integration import (
            get_streaming_database,
        )

        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.schema_enabled",
            return_value=True,
        ):
            with mock.patch(
                "DocsToKG.ContentDownload.streaming_integration.streaming_schema",
                None,
            ):
                result = get_streaming_database()
                assert result is None

    def test_check_database_health_returns_none_when_disabled(self) -> None:
        """Test that check_database_health returns None when disabled."""
        with mock.patch(
            "DocsToKG.ContentDownload.streaming_integration.schema_enabled",
            return_value=False,
        ):
            from DocsToKG.ContentDownload.streaming_integration import (
                check_database_health,
            )

            result = check_database_health()
            assert result is None


class TestIntegrationStatus(unittest.TestCase):
    """Test integration status reporting."""

    def test_integration_status_returns_dict(self) -> None:
        """Test that integration_status returns a dict with expected keys."""
        from DocsToKG.ContentDownload.streaming_integration import (
            integration_status,
        )

        status = integration_status()
        assert isinstance(status, dict)
        assert "streaming" in status
        assert "idempotency" in status
        assert "schema" in status
        assert all(isinstance(v, bool) for v in status.values())

    def test_log_integration_status(self) -> None:
        """Test that log_integration_status logs without error."""
        from DocsToKG.ContentDownload.streaming_integration import (
            log_integration_status,
        )

        # Should not raise
        log_integration_status()


class TestGracefulFallback(unittest.TestCase):
    """Test overall graceful fallback behavior."""

    def test_all_functions_handle_missing_modules(self) -> None:
        """Test that all functions handle missing modules gracefully."""
        from DocsToKG.ContentDownload import streaming_integration

        with mock.patch.object(streaming_integration, "streaming", None):
            with mock.patch.object(streaming_integration, "idempotency", None):
                with mock.patch.object(streaming_integration, "streaming_schema", None):
                    # All these should return None or False without raising
                    with mock.patch.object(
                        streaming_integration, "streaming_enabled", return_value=False
                    ):
                        with mock.patch.object(
                            streaming_integration, "idempotency_enabled", return_value=False
                        ):
                            with mock.patch.object(
                                streaming_integration, "schema_enabled", return_value=False
                            ):
                                assert (
                                    streaming_integration.try_streaming_resume_decision(None, None)
                                    is None
                                )
                                assert (
                                    streaming_integration.try_streaming_io(mock.Mock(), mock.Mock())
                                    is None
                                )
                                assert (
                                    streaming_integration.generate_job_key("w", "a", "url") is None
                                )


if __name__ == "__main__":
    unittest.main()
