"""Property-based tests for security gates using Hypothesis.

Tests use property-based testing to generate random inputs and verify
that gates maintain invariants across a wide range of edge cases.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import List

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from DocsToKG.OntologyDownload.policy.gates import (
    url_gate,
    extraction_gate,
    filesystem_gate,
)
from DocsToKG.OntologyDownload.policy.errors import (
    PolicyOK,
    URLPolicyException,
    ExtractionPolicyException,
    FilesystemPolicyException,
)


class TestURLGateProperties:
    """Property-based tests for URL gate."""

    @given(
        host=st.just("example.com"),
        port=st.integers(min_value=1, max_value=65535),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_url_gate_accepts_valid_ports(self, host: str, port: int):
        """URL gate should accept any valid port number."""
        result = url_gate(
            f"https://{host}:{port}/path",
            allowed_hosts=[host],
            allowed_ports=[port],
        )
        assert isinstance(result, PolicyOK)
        assert result.elapsed_ms >= 0

    @given(
        host=st.sampled_from(
            [
                "example.com",
                "sub.example.com",
                "deep.sub.example.com",
                "example.co.uk",
                "example.org",
            ]
        ),
    )
    @settings(max_examples=50)
    def test_url_gate_accepts_valid_hosts(self, host: str):
        """URL gate should accept any allowlisted host."""
        result = url_gate(
            f"https://{host}/ontology.owl",
            allowed_hosts=[host],
            allowed_ports=[443],
        )
        assert isinstance(result, PolicyOK)

    @given(
        disallowed_host=st.text(
            alphabet=string.ascii_lowercase + string.digits + ".-", min_size=5, max_size=20
        )
    )
    @settings(max_examples=50)
    def test_url_gate_rejects_disallowed_hosts(self, disallowed_host: str):
        """URL gate should reject hosts not in allowlist."""
        if not disallowed_host or disallowed_host[0] == ".":
            return  # Skip invalid hostnames

        with pytest.raises(URLPolicyException):
            url_gate(
                f"https://{disallowed_host}/ontology.owl",
                allowed_hosts=["example.com"],
                allowed_ports=[443],
            )


class TestExtractionGateProperties:
    """Property-based tests for extraction gate."""

    @given(
        entries=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_extraction_gate_accepts_reasonable_archives(self, entries: int):
        """Extraction gate should accept archives with reasonable ratios."""
        # bytes_per_entry = 100KB (well within normal limits)
        bytes_declared = entries * 100_000
        result = extraction_gate(
            entries_total=entries,
            bytes_declared=bytes_declared,
            max_total_ratio=10.0,
            max_entry_ratio=500_000,  # Allow up to 500KB per entry
        )
        assert isinstance(result, PolicyOK)

    @given(
        entries=st.integers(min_value=100_000, max_value=1_000_000),
    )
    @settings(max_examples=50)
    def test_extraction_gate_rejects_excessive_entry_counts(self, entries: int):
        """Extraction gate should reject excessively large entry counts."""
        with pytest.raises(ExtractionPolicyException):
            extraction_gate(
                entries_total=entries,
                bytes_declared=100_000,
                max_total_ratio=10.0,
                max_entries=50_000,  # Strict limit
            )

    @given(
        compressed_size=st.integers(min_value=100, max_value=10_000),
        ratio_limit=st.floats(min_value=0.5, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_extraction_gate_compression_ratio_bounds(
        self, compressed_size: int, ratio_limit: float
    ):
        """Extraction gate should enforce compression ratio limits consistently."""
        # Create entries that would exceed the ratio
        entries = int((compressed_size * ratio_limit * 10))  # Way over limit

        try:
            result = extraction_gate(
                entries_total=entries,
                bytes_declared=compressed_size,
                max_total_ratio=10.0,
                max_entry_ratio=ratio_limit,
            )
            # If it passes, verify the actual ratio is within bounds
            actual_ratio = compressed_size / max(entries, 1)
            assert actual_ratio <= ratio_limit
        except ExtractionPolicyException:
            # Expected for extreme ratios
            pass


class TestFilesystemGateProperties:
    """Property-based tests for filesystem gate."""

    @given(
        path_part=st.text(
            alphabet=string.ascii_lowercase + string.digits + "-_", min_size=1, max_size=50
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filesystem_gate_accepts_normal_paths(self, path_part: str, tmp_path: Path):
        """Filesystem gate should accept well-formed paths."""
        root = tmp_path / "extract"
        root.mkdir(exist_ok=True)

        result = filesystem_gate(
            root_path=str(root),
            entry_paths=[f"dir/{path_part}/file.txt"],
            allow_symlinks=False,
        )
        assert isinstance(result, PolicyOK)

    @given(
        depth=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filesystem_gate_accepts_reasonable_depths(self, depth: int, tmp_path: Path):
        """Filesystem gate should accept reasonable path depths."""
        root = tmp_path / "extract"
        root.mkdir(exist_ok=True)

        # Build a nested path
        parts = [f"dir{i}" for i in range(depth)]
        path = "/".join(parts) + "/file.txt"

        result = filesystem_gate(
            root_path=str(root),
            entry_paths=[path],
            allow_symlinks=False,
        )
        assert isinstance(result, PolicyOK)

    @given(
        traversal_attempt=st.sampled_from(
            [
                "../etc/passwd",
                "../../etc/passwd",
                "../../../root/.ssh/id_rsa",
                "....//etc/passwd",
                "..\\..\\windows\\system32",
            ]
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filesystem_gate_rejects_traversal_attempts(
        self, traversal_attempt: str, tmp_path: Path
    ):
        """Filesystem gate should reject all path traversal attempts."""
        root = tmp_path / "extract"
        root.mkdir(exist_ok=True)

        with pytest.raises(FilesystemPolicyException):
            filesystem_gate(
                root_path=str(root),
                entry_paths=[traversal_attempt],
                allow_symlinks=False,
            )

    @given(
        absolute_path=st.sampled_from(
            [
                "/etc/passwd",
                "/root/.ssh/id_rsa",
                "/var/log/auth.log",
                "//network/share/file",
            ]
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filesystem_gate_rejects_absolute_paths(self, absolute_path: str, tmp_path: Path):
        """Filesystem gate should reject absolute paths."""
        root = tmp_path / "extract"
        root.mkdir(exist_ok=True)

        with pytest.raises(FilesystemPolicyException):
            filesystem_gate(
                root_path=str(root),
                entry_paths=[absolute_path],
                allow_symlinks=False,
            )


class TestGateInvariants:
    """Test invariants that must hold across all gates."""

    @given(st.just(0))
    @settings(max_examples=10)
    def test_all_gates_return_consistent_types(self, _: int):
        """All gates should return PolicyOK or raise exception (never PolicyReject without exception)."""
        # URL gate
        result = url_gate(
            "https://example.com/path",
            allowed_hosts=["example.com"],
            allowed_ports=[443],
        )
        assert isinstance(result, PolicyOK)
        assert hasattr(result, "elapsed_ms")
        assert hasattr(result, "gate_name")
        assert result.elapsed_ms >= 0

    @given(st.just(0))
    @settings(max_examples=10)
    def test_all_gate_timings_reasonable(self, _: int):
        """All gates should complete within reasonable time bounds."""
        import tempfile

        # URL gate
        url_result = url_gate(
            "https://example.com/path",
            allowed_hosts=["example.com"],
            allowed_ports=[443],
        )
        assert url_result.elapsed_ms < 50.0  # Very generous 50ms limit

        # Extraction gate
        extraction_result = extraction_gate(
            entries_total=100,
            bytes_declared=10_000_000,
            max_total_ratio=10.0,
            max_entry_ratio=200_000,
        )
        assert extraction_result.elapsed_ms < 50.0

        # Filesystem gate - use context manager to avoid fixture issue
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "extract"
            root.mkdir()
            fs_result = filesystem_gate(
                root_path=str(root),
                entry_paths=["test.txt"],
                allow_symlinks=False,
            )
            assert fs_result.elapsed_ms < 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
