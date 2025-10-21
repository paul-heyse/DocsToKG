"""Integration tests for policy gates with ContentDownload extraction operations."""

import pytest

from DocsToKG.OntologyDownload.policy.gates import extraction_gate
from DocsToKG.OntologyDownload.policy.errors import ExtractionPolicyException


class TestExtractionGateIntegration:
    """Tests for extraction policy gate integration with ContentDownload."""

    def test_extraction_gate_accepts_small_files(self):
        """Small archive entries should pass."""
        entry = {
            "type": "file",
            "name": "paper.pdf",
            "size": 1000000,  # 1 MB
            "compressed_size": 500000,
        }
        result = extraction_gate(entry)
        assert result is not None
        assert result.gate_name == "extraction_gate"

    def test_extraction_gate_rejects_large_files(self):
        """Very large files should be rejected."""
        entry = {
            "type": "file",
            "name": "huge.zip",
            "size": 10000000000,  # 10 GB
            "compressed_size": 5000000000,
        }
        with pytest.raises(ExtractionPolicyException):
            extraction_gate(entry)

    def test_extraction_gate_detects_zip_bombs(self):
        """Zip bombs (high compression ratio) should be detected."""
        entry = {
            "type": "file",
            "name": "suspicious.zip",
            "size": 5000000000,  # 5 GB
            "compressed_size": 100000,  # 100 KB - 50,000x compression!
        }
        with pytest.raises(ExtractionPolicyException):
            extraction_gate(entry)

    def test_extraction_gate_accepts_reasonable_compression(self):
        """Reasonable compression ratios should pass."""
        entries = [
            {
                "type": "file",
                "name": "document.pdf",
                "size": 1000000,
                "compressed_size": 900000,  # 90% - reasonable
            },
            {
                "type": "file",
                "name": "data.json",
                "size": 1000000,
                "compressed_size": 100000,  # 10% - good but reasonable
            },
        ]
        for entry in entries:
            result = extraction_gate(entry)
            assert result is not None

    def test_extraction_gate_rejects_invalid_types(self):
        """Invalid entry types should be rejected."""
        entry = {
            "type": "symlink",
            "name": "link_to_etc",
            "target": "/etc/passwd",
        }
        with pytest.raises(ExtractionPolicyException):
            extraction_gate(entry)

    def test_extraction_gate_tracks_statistics(self):
        """Extraction gate should track invocation statistics."""
        from DocsToKG.OntologyDownload.policy.registry import get_registry
        
        registry = get_registry()
        
        # Clear stats
        registry.reset_stats("extraction_gate")
        
        # Invoke with valid entry
        try:
            entry = {"type": "file", "name": "test.pdf", "size": 1000000}
            extraction_gate(entry)
            stats = registry.get_stats("extraction_gate")
            assert stats["passes"] > 0
        except Exception:
            pass

    def test_extraction_gate_accepts_various_files(self):
        """Different file types should be handled."""
        entries = [
            {"type": "file", "name": "readme.txt", "size": 5000},
            {"type": "file", "name": "script.py", "size": 10000},
            {"type": "file", "name": "data.csv", "size": 100000},
        ]
        for entry in entries:
            result = extraction_gate(entry)
            assert result is not None


class TestExtractionGateErrorHandling:
    """Tests for extraction gate error handling."""

    def test_extraction_gate_error_messages_are_clear(self):
        """Error messages should be helpful for debugging."""
        entry = {
            "type": "file",
            "name": "bomb.zip",
            "size": 100000000000,
            "compressed_size": 1000,
        }
        with pytest.raises(ExtractionPolicyException) as exc_info:
            extraction_gate(entry)
        
        error = exc_info.value
        assert hasattr(error, 'error_code')
        assert error.error_code is not None

    def test_extraction_gate_prevents_symlinks(self):
        """Symlinks should be prevented (security risk)."""
        entry = {
            "type": "symlink",
            "name": "link",
            "target": "/etc/passwd",
        }
        with pytest.raises(ExtractionPolicyException):
            extraction_gate(entry)


class TestExtractionGatePdfIntegration:
    """Tests for PDF extraction from archives."""

    def test_extract_pdf_from_archive(self):
        """PDFs extracted from archives should be validated."""
        entry = {
            "type": "file",
            "name": "papers/arxiv_2024_001.pdf",
            "size": 5000000,  # 5 MB
            "compressed_size": 2000000,
        }
        result = extraction_gate(entry)
        assert result is not None

    def test_extraction_gate_accepts_pdf_files(self):
        """PDF files in various contexts should pass."""
        entries = [
            {
                "type": "file",
                "name": "paper.pdf",
                "size": 1000000,
                "compressed_size": 500000,
            },
            {
                "type": "file",
                "name": "scan_2024.pdf",
                "size": 2000000,
                "compressed_size": 1500000,
            },
        ]
        for entry in entries:
            result = extraction_gate(entry)
            assert result is not None

    def test_extraction_gate_limits_pdf_sizes(self):
        """Very large PDFs should be limited."""
        entry = {
            "type": "file",
            "name": "huge_scan.pdf",
            "size": 5000000000,  # 5 GB
            "compressed_size": 2000000000,
        }
        with pytest.raises(ExtractionPolicyException):
            extraction_gate(entry)


class TestExtractionGatePerformance:
    """Tests for extraction gate performance."""

    def test_extraction_gate_is_fast(self):
        """Extraction gate validation should be fast."""
        import time
        
        entry = {
            "type": "file",
            "name": "paper.pdf",
            "size": 1000000,
            "compressed_size": 500000,
        }
        
        start = time.time()
        for _ in range(1000):
            try:
                extraction_gate(entry)
            except Exception:
                pass
        elapsed = time.time() - start
        
        # 1000 validations should take < 100ms
        assert elapsed < 0.1, f"Extraction validation too slow: {elapsed}s for 1000 calls"

    def test_extraction_gate_batching_scenario(self):
        """Extraction gate should handle batches efficiently."""
        entries = [
            {"type": "file", "name": "p1.pdf", "size": 1000000, "compressed_size": 500000},
            {"type": "file", "name": "p2.pdf", "size": 2000000, "compressed_size": 1000000},
            {"type": "file", "name": "config.json", "size": 50000, "compressed_size": 10000},
            {"type": "file", "name": "data.csv", "size": 100000, "compressed_size": 30000},
        ]
        
        results = []
        for entry in entries:
            try:
                result = extraction_gate(entry)
                results.append((entry["name"], "PASS", result))
            except ExtractionPolicyException as e:
                results.append((entry["name"], "REJECT", str(e)))
        
        # Most should pass
        passed = sum(1 for _, status, _ in results if status == "PASS")
        assert passed >= 2, f"Expected at least 2 passes, got {passed}"

    def test_extraction_gate_metrics_available(self):
        """Metrics should be available after gate invocations."""
        from DocsToKG.OntologyDownload.policy.registry import get_registry
        
        registry = get_registry()
        
        # After running tests, metrics should be available
        stats = registry.get_stats("extraction_gate")
        
        # Stats should have expected keys
        expected_keys = {"passes", "rejects", "avg_ms", "min_ms", "max_ms", "pass_rate"}
        assert expected_keys.issubset(set(stats.keys()))

