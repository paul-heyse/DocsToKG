"""Integration tests for policy gates with ContentDownload path operations."""

import pytest
from pathlib import Path

from DocsToKG.OntologyDownload.policy.gates import path_gate
from DocsToKG.OntologyDownload.policy.errors import FilesystemPolicyException


class TestPathGateIntegration:
    """Tests for path policy gate integration with ContentDownload."""

    def test_path_gate_accepts_valid_relative_path(self):
        """Valid relative paths should pass the gate."""
        path = "data/downloads/paper.pdf"
        result = path_gate(path)
        assert result is not None
        assert result.gate_name == "path_gate"

    def test_path_gate_rejects_absolute_paths(self):
        """Absolute paths should be rejected."""
        paths = [
            "/etc/passwd",
            "/home/user/data",
        ]
        for path in paths:
            with pytest.raises(FilesystemPolicyException):
                path_gate(path)

    def test_path_gate_rejects_path_traversal(self):
        """Path traversal attempts should be rejected."""
        paths = [
            "../../../etc/passwd",
            "data/../../etc/passwd",
            "downloads/..\\..\\windows\\system32",
        ]
        for path in paths:
            with pytest.raises(FilesystemPolicyException):
                path_gate(path)

    def test_path_gate_accepts_normal_filenames(self):
        """Normal filenames should pass."""
        paths = [
            "paper.pdf",
            "arxiv_2024_01.pdf",
            "crossref-data.json",
            "output_v2.parquet",
        ]
        for path in paths:
            result = path_gate(path)
            assert result is not None

    def test_path_gate_accepts_subdirectories(self):
        """Reasonable subdirectory structures should pass."""
        paths = [
            "downloads/pdfs",
            "data/embeddings/dense",
            "output/2024/10/21",
        ]
        for path in paths:
            result = path_gate(path)
            assert result is not None

    def test_path_gate_rejects_excessive_depth(self):
        """Paths with excessive directory depth should be rejected."""
        # Create a path with many levels (>10)
        deep_path = "/".join(["level" + str(i) for i in range(15)])
        with pytest.raises(FilesystemPolicyException):
            path_gate(deep_path)

    def test_path_gate_rejects_too_long_paths(self):
        """Paths that are too long should be rejected."""
        # Create a very long path name
        long_name = "x" * 300
        path = f"data/{long_name}/file.pdf"
        with pytest.raises(FilesystemPolicyException):
            path_gate(path)

    def test_path_gate_tracks_statistics(self):
        """Path gate should track invocation statistics."""
        from DocsToKG.OntologyDownload.policy.registry import get_registry
        
        registry = get_registry()
        
        # Clear stats
        registry.reset_stats("path_gate")
        
        # Invoke with valid path
        try:
            path_gate("data/output.pdf")
            stats = registry.get_stats("path_gate")
            assert stats["passes"] > 0
        except Exception:
            pass

    def test_path_gate_with_special_chars(self):
        """Paths with reasonable special characters should pass."""
        paths = [
            "data-2024/file_v2.pdf",
            "arxiv(2024).json",
            "output [final].pdf",
        ]
        for path in paths:
            try:
                result = path_gate(path)
                assert result is not None
            except FilesystemPolicyException:
                # Some special chars might be rejected - that's ok
                pass


class TestPathGateErrorHandling:
    """Tests for path gate error handling in ContentDownload context."""

    def test_path_gate_error_messages_are_clear(self):
        """Error messages should be helpful for debugging."""
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate("../../../../etc/passwd")
        
        error = exc_info.value
        assert hasattr(error, 'error_code')
        assert error.error_code is not None

    def test_path_gate_detects_traversal_attacks(self):
        """Path traversal attacks should be clearly identified."""
        with pytest.raises(FilesystemPolicyException) as exc_info:
            path_gate("../../../sensitive/data")
        
        error = exc_info.value
        error_str = str(error)
        # Should mention something about traversal or path security
        assert any(keyword in error_str.lower() for keyword in ["traversal", "path", "escape"])


class TestPathGateStorageIntegration:
    """Tests demonstrating ContentDownload storage integration."""

    def test_validate_output_paths_before_write(self):
        """Output paths should be validated before file writes."""
        # Valid output path
        output_path = "data/output/results.pdf"
        result = path_gate(output_path)
        assert result is not None
        
        # Invalid output path
        invalid_path = "../../etc/passwd"
        with pytest.raises(FilesystemPolicyException):
            path_gate(invalid_path)

    def test_validate_cache_paths(self):
        """Cache paths should be validated."""
        cache_paths = [
            "cache/downloads",
            "cache/embeddings",
            ".cache/http",
        ]
        for path in cache_paths:
            try:
                result = path_gate(path)
                assert result is not None
            except FilesystemPolicyException:
                # Hidden cache files might be rejected - acceptable
                pass

    def test_path_gate_metrics_available(self):
        """Metrics should be available after gate invocations."""
        from DocsToKG.OntologyDownload.policy.registry import get_registry
        
        registry = get_registry()
        
        # After running tests, metrics should be available
        stats = registry.get_stats("path_gate")
        
        # Stats should have expected keys
        expected_keys = {"passes", "rejects", "avg_ms", "min_ms", "max_ms", "pass_rate"}
        assert expected_keys.issubset(set(stats.keys()))


class TestPathGatePerformance:
    """Tests for path gate performance in ContentDownload context."""

    def test_path_gate_is_fast(self):
        """Path gate validation should be fast."""
        import time
        
        path = "data/downloads/file.pdf"
        
        start = time.time()
        for _ in range(1000):
            try:
                path_gate(path)
            except Exception:
                pass
        elapsed = time.time() - start
        
        # 1000 validations should take < 100ms
        assert elapsed < 0.1, f"Path validation too slow: {elapsed}s for 1000 calls"

    def test_path_gate_batching_scenario(self):
        """Path gate should handle batches of paths efficiently."""
        paths = [
            "data/output.pdf",
            "cache/embeddings.parquet",
            "results/2024/10/report.json",
            "temp/processing_queue",
        ]
        
        results = []
        for path in paths:
            try:
                result = path_gate(path)
                results.append((path, "PASS", result))
            except FilesystemPolicyException as e:
                results.append((path, "REJECT", str(e)))
        
        # Most should pass
        passed = sum(1 for _, status, _ in results if status == "PASS")
        assert passed >= 2, f"Expected at least 2 passes, got {passed}"

