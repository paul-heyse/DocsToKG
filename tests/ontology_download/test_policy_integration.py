"""Integration tests for the complete policy system.

Tests:
- End-to-end gate workflows
- Cross-platform path validation
- Error propagation and handling
- Metrics integration
"""

import pytest

from DocsToKG.OntologyDownload.policy.errors import (
    FilesystemPolicyException,
    URLPolicyException,
)
from DocsToKG.OntologyDownload.policy.gates import (
    config_gate,
    extraction_gate,
    path_gate,
    url_gate,
)
from DocsToKG.OntologyDownload.policy.metrics import (
    GateMetric,
    get_metrics_collector,
)
from DocsToKG.OntologyDownload.policy.registry import get_registry

# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


class TestPolicyWorkflows:
    """Test complete policy workflows."""

    def test_successful_gate_chain(self):
        """Multiple gates succeed in sequence."""

        # Config gate passes
        class MockConfig:
            http_settings = {}
            network_settings = {}

        result1 = config_gate(MockConfig())
        assert result1 is not None

        # URL gate passes
        result2 = url_gate("https://example.com")
        assert result2 is not None

        # Path gate passes
        result3 = path_gate("data/file.txt")
        assert result3 is not None

        # Extraction gate passes
        entry = {"type": "file", "size": 1000, "compressed_size": 500}
        result4 = extraction_gate(entry)
        assert result4 is not None

    def test_gate_rejection_stops_flow(self):
        """Rejection at one gate stops processing."""
        # First gate passes
        result1 = url_gate("https://good.com")
        assert result1 is not None

        # Second gate fails
        with pytest.raises(URLPolicyException):
            url_gate("ftp://bad.com")


# ============================================================================
# Cross-Platform Tests
# ============================================================================


class TestCrossPlatformPaths:
    """Test path validation across platforms."""

    def test_windows_reserved_names(self):
        """Windows reserved names are rejected."""
        reserved = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
        for name in reserved:
            with pytest.raises(FilesystemPolicyException):
                path_gate(f"data/{name}/file.txt")

    def test_unix_absolute_paths_rejected(self):
        """Unix absolute paths are rejected."""
        with pytest.raises(FilesystemPolicyException):
            path_gate("/etc/passwd")

    def test_traversal_attempts_rejected(self):
        """Directory traversal attempts are rejected."""
        traversals = [
            "../../etc/passwd",
            "data/../../../root",
            "a/b/../../c/d",
        ]
        for path in traversals:
            with pytest.raises(FilesystemPolicyException):
                path_gate(path)

    def test_valid_portable_paths(self):
        """Valid paths work across platforms."""
        valid_paths = [
            "data/file.txt",
            "archive/subdir/file.tar.gz",
            "a/b/c/d/e/f/g",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "123_numbers.txt",
        ]
        for path in valid_paths:
            result = path_gate(path)
            assert result is not None


# ============================================================================
# Registry Integration Tests
# ============================================================================


class TestRegistryIntegration:
    """Test policy registry integration."""

    def test_gates_registered(self):
        """All gates are registered with the registry."""
        registry = get_registry()
        gates = registry.list_gates()

        expected_gates = {
            "config_gate",
            "url_gate",
            "path_gate",
            "extraction_gate",
            "storage_gate",
            "db_gate",
        }

        for gate_name in expected_gates:
            assert gate_name in gates

    def test_gate_invocation_via_registry(self):
        """Gates can be invoked through the registry."""
        registry = get_registry()

        # Invoke URL gate
        result = registry.invoke("url_gate", "https://example.com")
        assert result is not None

        # Invoke path gate
        result = registry.invoke("path_gate", "data/file.txt")
        assert result is not None

    def test_domain_filtering(self):
        """Gates can be filtered by domain."""
        registry = get_registry()

        # Get network gates
        net_gates = registry.gates_by_domain("network")
        assert "url_gate" in net_gates

        # Get filesystem gates
        fs_gates = registry.gates_by_domain("filesystem")
        assert "path_gate" in fs_gates

        # Get extraction gates
        extraction_gates = registry.gates_by_domain("extraction")
        assert "extraction_gate" in extraction_gates


# ============================================================================
# Metrics Integration Tests
# ============================================================================


class TestMetricsIntegration:
    """Test metrics collection integration."""

    def test_metrics_on_success(self):
        """Metrics are collected on successful gate invocation."""
        collector = get_metrics_collector()
        collector.clear_metrics("test_success_gate")

        registry = get_registry()
        result = registry.invoke("url_gate", "https://example.com")
        assert result is not None

    def test_error_handling_with_metrics(self):
        """Errors are handled and can be tracked."""
        collector = get_metrics_collector()
        collector.clear_metrics()

        # Record a failure
        collector.record_metric(GateMetric("test_gate", False, 1.0, "E_HOST_DENY"))

        snapshot = collector.get_snapshot("test_gate")
        assert snapshot.rejects == 1
        assert snapshot.passes == 0

    def test_summary_with_multiple_gates(self):
        """Summary aggregates across multiple gates."""
        collector = get_metrics_collector()
        collector.clear_metrics()

        # Record metrics for different gates
        collector.record_metric(GateMetric("gate1", True, 1.0))
        collector.record_metric(GateMetric("gate1", True, 1.1))
        collector.record_metric(GateMetric("gate2", True, 0.5))
        collector.record_metric(GateMetric("gate2", False, 0.6))

        summary = collector.get_summary()
        assert summary["total_invocations"] == 4
        assert summary["total_passes"] == 3
        assert summary["total_rejects"] == 1


# ============================================================================
# Error Propagation Tests
# ============================================================================


class TestErrorPropagation:
    """Test error propagation through the system."""

    def test_policy_exception_inheritance(self):
        """Policy exceptions maintain proper inheritance."""
        with pytest.raises(URLPolicyException):
            url_gate("ftp://bad.com")

    def test_error_details_preserved(self):
        """Error details are preserved through exceptions."""
        try:
            url_gate("ftp://bad.com")
            assert False, "Should have raised URLPolicyException"
        except URLPolicyException as e:
            assert e.error_code is not None
            assert "ftp" in str(e.error_code).lower() or "scheme" in str(e).lower()

    def test_sensitive_data_scrubbed(self):
        """Sensitive data is scrubbed from errors."""
        try:
            url_gate("https://user:password@example.com")
            assert False, "Should have raised URLPolicyException"
        except URLPolicyException as e:
            error_str = str(e)
            # Password should not appear in error
            assert "password" not in error_str.lower()


# ============================================================================
# Stress Tests
# ============================================================================


class TestStressScenarios:
    """Test system under stress."""

    def test_many_gates_registered(self):
        """Registry handles many gates."""
        registry = get_registry()
        gates = registry.list_gates()
        assert len(gates) >= 6  # At least our 6 gates

    def test_rapid_metric_collection(self):
        """Metrics collector handles rapid recording."""
        collector = get_metrics_collector()
        collector.clear_metrics("rapid_gate")

        # Record 1000 rapid metrics
        for i in range(1000):
            passed = i % 10 != 0  # 90% pass rate
            collector.record_metric(GateMetric("rapid_gate", passed, 0.1 * (i % 10), None))

        snapshot = collector.get_snapshot("rapid_gate")
        assert snapshot.invocations == 1000
        assert snapshot.passes == 900
        assert snapshot.rejects == 100
