# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.test_orchestrator_config",
#   "purpose": "Tests for OrchestratorConfig and QueueConfig Pydantic models",
#   "sections": [
#     {"id": "queueconfig", "name": "TestQueueConfig", "anchor": "#class-testqueueconfig", "kind": "test"},
#     {"id": "orchestratorconfig", "name": "TestOrchestratorConfig", "anchor": "#class-testorchestratorconfig", "kind": "test"}
#   ]
# }
# === /NAVMAP ===

"""Tests for work orchestration configuration models.

Tests OrchestratorConfig and QueueConfig Pydantic models for:
- Valid configurations
- Default values
- Validation rules
- Type safety
- Integration with ContentDownloadConfig
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from DocsToKG.ContentDownload.config.models import (
    ContentDownloadConfig,
    OrchestratorConfig,
    QueueConfig,
)


class TestQueueConfig(unittest.TestCase):
    """Tests for QueueConfig model."""

    def test_queue_config_defaults(self) -> None:
        """QueueConfig has sensible defaults."""
        config = QueueConfig()
        assert config.backend == "sqlite"
        assert config.path == "state/workqueue.sqlite"
        assert config.wal_mode is True
        assert config.timeout_sec == 10

    def test_queue_config_custom_path(self) -> None:
        """QueueConfig accepts custom path."""
        config = QueueConfig(path="/custom/queue.sqlite")
        assert config.path == "/custom/queue.sqlite"

    def test_queue_config_disable_wal(self) -> None:
        """QueueConfig can disable WAL mode."""
        config = QueueConfig(wal_mode=False)
        assert config.wal_mode is False

    def test_queue_config_custom_timeout(self) -> None:
        """QueueConfig accepts custom timeout."""
        config = QueueConfig(timeout_sec=30)
        assert config.timeout_sec == 30

    def test_queue_config_invalid_timeout_zero(self) -> None:
        """QueueConfig rejects timeout_sec=0."""
        with self.assertRaises(ValidationError):
            QueueConfig(timeout_sec=0)

    def test_queue_config_invalid_timeout_negative(self) -> None:
        """QueueConfig rejects negative timeout_sec."""
        with self.assertRaises(ValidationError):
            QueueConfig(timeout_sec=-1)

    def test_queue_config_extra_forbid(self) -> None:
        """QueueConfig forbids extra fields."""
        with self.assertRaises(ValidationError):
            QueueConfig(unknown_field="value")  # type: ignore


class TestOrchestratorConfig(unittest.TestCase):
    """Tests for OrchestratorConfig model."""

    def test_orchestrator_config_defaults(self) -> None:
        """OrchestratorConfig has sensible defaults."""
        config = OrchestratorConfig()
        assert config.max_workers == 8
        assert config.max_per_resolver == {}
        assert config.max_per_host == 4
        assert config.lease_ttl_seconds == 600
        assert config.heartbeat_seconds == 30
        assert config.max_job_attempts == 3
        assert config.retry_backoff_seconds == 60
        assert config.jitter_seconds == 15

    def test_orchestrator_config_custom_workers(self) -> None:
        """OrchestratorConfig accepts custom worker count."""
        config = OrchestratorConfig(max_workers=16)
        assert config.max_workers == 16

    def test_orchestrator_config_max_workers_bounds(self) -> None:
        """OrchestratorConfig enforces max_workers bounds (1-256)."""
        # Valid boundaries
        assert OrchestratorConfig(max_workers=1).max_workers == 1
        assert OrchestratorConfig(max_workers=256).max_workers == 256

        # Invalid boundaries
        with self.assertRaises(ValidationError):
            OrchestratorConfig(max_workers=0)
        with self.assertRaises(ValidationError):
            OrchestratorConfig(max_workers=257)

    def test_orchestrator_config_per_resolver_limits(self) -> None:
        """OrchestratorConfig accepts per-resolver limits."""
        config = OrchestratorConfig(
            max_per_resolver={"unpaywall": 2, "crossref": 4, "openalex": 3}
        )
        assert config.max_per_resolver == {"unpaywall": 2, "crossref": 4, "openalex": 3}

    def test_orchestrator_config_per_resolver_validation(self) -> None:
        """OrchestratorConfig validates per-resolver limits are positive."""
        # Valid
        OrchestratorConfig(max_per_resolver={"resolver": 1})

        # Invalid (zero)
        with self.assertRaises(ValidationError) as ctx:
            OrchestratorConfig(max_per_resolver={"resolver": 0})
        assert "must be > 0" in str(ctx.exception)

        # Invalid (negative)
        with self.assertRaises(ValidationError) as ctx:
            OrchestratorConfig(max_per_resolver={"resolver": -5})
        assert "must be > 0" in str(ctx.exception)

    def test_orchestrator_config_max_per_host_bounds(self) -> None:
        """OrchestratorConfig enforces max_per_host >= 1."""
        assert OrchestratorConfig(max_per_host=1).max_per_host == 1
        assert OrchestratorConfig(max_per_host=8).max_per_host == 8

        with self.assertRaises(ValidationError):
            OrchestratorConfig(max_per_host=0)

    def test_orchestrator_config_lease_ttl_bounds(self) -> None:
        """OrchestratorConfig enforces lease_ttl_seconds >= 30."""
        assert OrchestratorConfig(lease_ttl_seconds=30).lease_ttl_seconds == 30
        assert OrchestratorConfig(lease_ttl_seconds=600).lease_ttl_seconds == 600

        with self.assertRaises(ValidationError):
            OrchestratorConfig(lease_ttl_seconds=29)

    def test_orchestrator_config_heartbeat_bounds(self) -> None:
        """OrchestratorConfig enforces heartbeat_seconds >= 5."""
        assert OrchestratorConfig(heartbeat_seconds=5).heartbeat_seconds == 5
        assert OrchestratorConfig(heartbeat_seconds=30).heartbeat_seconds == 30

        with self.assertRaises(ValidationError):
            OrchestratorConfig(heartbeat_seconds=4)

    def test_orchestrator_config_max_attempts_bounds(self) -> None:
        """OrchestratorConfig enforces max_job_attempts >= 1."""
        assert OrchestratorConfig(max_job_attempts=1).max_job_attempts == 1
        assert OrchestratorConfig(max_job_attempts=5).max_job_attempts == 5

        with self.assertRaises(ValidationError):
            OrchestratorConfig(max_job_attempts=0)

    def test_orchestrator_config_retry_backoff_bounds(self) -> None:
        """OrchestratorConfig enforces retry_backoff_seconds >= 1."""
        assert OrchestratorConfig(retry_backoff_seconds=1).retry_backoff_seconds == 1

        with self.assertRaises(ValidationError):
            OrchestratorConfig(retry_backoff_seconds=0)

    def test_orchestrator_config_jitter_bounds(self) -> None:
        """OrchestratorConfig allows jitter_seconds >= 0."""
        assert OrchestratorConfig(jitter_seconds=0).jitter_seconds == 0
        assert OrchestratorConfig(jitter_seconds=15).jitter_seconds == 15

    def test_orchestrator_config_extra_forbid(self) -> None:
        """OrchestratorConfig forbids extra fields."""
        with self.assertRaises(ValidationError):
            OrchestratorConfig(unknown_field="value")  # type: ignore

    def test_orchestrator_config_realistic_scenario(self) -> None:
        """OrchestratorConfig works with realistic multi-resolver setup."""
        config = OrchestratorConfig(
            max_workers=32,
            max_per_resolver={"unpaywall": 2, "crossref": 4, "arxiv": 3, "pmc": 2},
            max_per_host=6,
            lease_ttl_seconds=900,
            heartbeat_seconds=45,
            max_job_attempts=5,
            retry_backoff_seconds=120,
            jitter_seconds=30,
        )
        assert config.max_workers == 32
        assert len(config.max_per_resolver) == 4
        assert config.max_per_host == 6


class TestConfigIntegration(unittest.TestCase):
    """Tests for integration with ContentDownloadConfig."""

    def test_content_download_config_includes_queue(self) -> None:
        """ContentDownloadConfig includes QueueConfig."""
        config = ContentDownloadConfig()
        assert isinstance(config.queue, QueueConfig)
        assert config.queue.path == "state/workqueue.sqlite"

    def test_content_download_config_includes_orchestrator(self) -> None:
        """ContentDownloadConfig includes OrchestratorConfig."""
        config = ContentDownloadConfig()
        assert isinstance(config.orchestrator, OrchestratorConfig)
        assert config.orchestrator.max_workers == 8

    def test_content_download_config_custom_orchestrator(self) -> None:
        """ContentDownloadConfig accepts custom OrchestratorConfig."""
        config = ContentDownloadConfig(
            orchestrator=OrchestratorConfig(
                max_workers=16,
                max_per_host=8,
            )
        )
        assert config.orchestrator.max_workers == 16
        assert config.orchestrator.max_per_host == 8

    def test_content_download_config_custom_queue(self) -> None:
        """ContentDownloadConfig accepts custom QueueConfig."""
        config = ContentDownloadConfig(
            queue=QueueConfig(path="/custom/path/queue.sqlite", wal_mode=False)
        )
        assert config.queue.path == "/custom/path/queue.sqlite"
        assert config.queue.wal_mode is False


if __name__ == "__main__":
    unittest.main()
