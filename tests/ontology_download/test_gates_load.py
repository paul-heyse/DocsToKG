"""Load testing for security gates - performance under stress.

Tests measure gate performance under high load to ensure they maintain
sub-millisecond response times and scalability.
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import pytest

from DocsToKG.OntologyDownload.policy.gates import (
    url_gate,
    extraction_gate,
    filesystem_gate,
    db_boundary_gate,
)
from DocsToKG.OntologyDownload.policy.errors import PolicyOK


class TestURLGateLoadTesting:
    """Load tests for URL gate under concurrent access."""

    def test_url_gate_concurrent_calls_1000(self):
        """URL gate should handle 1000 concurrent requests."""

        def invoke_gate() -> float:
            result = url_gate(
                "https://example.com/path",
                allowed_hosts=["example.com"],
                allowed_ports=[443],
            )
            return result.elapsed_ms

        timings: List[float] = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(invoke_gate) for _ in range(1000)]
            for future in as_completed(futures):
                timings.append(future.result())

        assert len(timings) == 1000
        assert all(0 <= t < 10 for t in timings)  # All under 10ms
        avg = sum(timings) / len(timings)
        p95 = sorted(timings)[int(len(timings) * 0.95)]
        p99 = sorted(timings)[int(len(timings) * 0.99)]

        print(f"\nURL Gate Load Test (1000 calls):")
        print(f"  Average: {avg:.4f}ms")
        print(f"  P95: {p95:.4f}ms")
        print(f"  P99: {p99:.4f}ms")
        assert avg < 1.0  # Average under 1ms
        assert p95 < 5.0  # 95th percentile under 5ms
        assert p99 < 10.0  # 99th percentile under 10ms

    def test_url_gate_sequential_1000(self):
        """URL gate should handle 1000 sequential calls."""
        timings: List[float] = []
        start = time.perf_counter()

        for _ in range(1000):
            result = url_gate(
                "https://example.com/path",
                allowed_hosts=["example.com"],
                allowed_ports=[443],
            )
            timings.append(result.elapsed_ms)

        total_time = (time.perf_counter() - start) * 1000
        avg = sum(timings) / len(timings)

        print(f"\nURL Gate Sequential Test (1000 calls):")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Average: {avg:.4f}ms")
        assert total_time < 10000  # 10 seconds total
        assert avg < 1.0


class TestExtractionGateLoadTesting:
    """Load tests for extraction gate under high volume."""

    def test_extraction_gate_concurrent_calls_500(self):
        """Extraction gate should handle 500 concurrent requests."""

        def invoke_gate() -> float:
            result = extraction_gate(
                entries_total=1000,
                bytes_declared=100_000_000,
                max_total_ratio=10.0,
                max_entry_ratio=200_000,
            )
            return result.elapsed_ms

        timings: List[float] = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(invoke_gate) for _ in range(500)]
            for future in as_completed(futures):
                timings.append(future.result())

        assert len(timings) == 500
        avg = sum(timings) / len(timings)
        p95 = sorted(timings)[int(len(timings) * 0.95)]

        print(f"\nExtraction Gate Load Test (500 calls):")
        print(f"  Average: {avg:.4f}ms")
        print(f"  P95: {p95:.4f}ms")
        assert avg < 2.0  # Average under 2ms
        assert p95 < 10.0

    def test_extraction_gate_varying_archive_sizes(self):
        """Extraction gate should handle varying archive sizes efficiently."""
        sizes = [100, 1_000, 10_000, 100_000]
        timings: Dict[int, List[float]] = {size: [] for size in sizes}

        for size in sizes:
            for _ in range(100):
                result = extraction_gate(
                    entries_total=size,
                    bytes_declared=size * 100_000,
                    max_total_ratio=10.0,
                    max_entry_ratio=200_000,
                )
                timings[size].append(result.elapsed_ms)

        print(f"\nExtraction Gate Performance by Archive Size:")
        for size in sizes:
            avg = sum(timings[size]) / len(timings[size])
            print(f"  {size} entries: {avg:.4f}ms avg")
            assert avg < 5.0  # All should be fast


class TestFilesystemGateLoadTesting:
    """Load tests for filesystem gate."""

    def test_filesystem_gate_concurrent_calls_500(self):
        """Filesystem gate should handle 500 concurrent path validations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "extract"
            root.mkdir()

            def invoke_gate() -> float:
                result = filesystem_gate(
                    root_path=str(root),
                    entry_paths=["test.txt"],
                    allow_symlinks=False,
                )
                return result.elapsed_ms

            timings: List[float] = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(invoke_gate) for _ in range(500)]
                for future in as_completed(futures):
                    timings.append(future.result())

            assert len(timings) == 500
            avg = sum(timings) / len(timings)

            print(f"\nFilesystem Gate Load Test (500 calls):")
            print(f"  Average: {avg:.4f}ms")
            assert avg < 2.0

    def test_filesystem_gate_deep_paths(self):
        """Filesystem gate should handle deeply nested paths efficiently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "extract"
            root.mkdir()

            depths = [5, 10, 15, 20]  # Stay within filesystem gate limit of 20 levels
            timings: Dict[int, List[float]] = {d: [] for d in depths}

            for depth in depths:
                parts = [f"dir{i}" for i in range(depth)]
                path = "/".join(parts) + "/file.txt"

                for _ in range(50):
                    result = filesystem_gate(
                        root_path=str(root),
                        entry_paths=[path],
                        allow_symlinks=False,
                    )
                    timings[depth].append(result.elapsed_ms)

            print(f"\nFilesystem Gate Performance by Path Depth:")
            for depth in depths:
                avg = sum(timings[depth]) / len(timings[depth])
                print(f"  Depth {depth}: {avg:.4f}ms avg")
                assert avg < 5.0


class TestDBBoundaryGateLoadTesting:
    """Load tests for DB boundary gate."""

    def test_db_boundary_gate_concurrent_calls_1000(self):
        """DB boundary gate should handle 1000 concurrent calls."""

        def invoke_gate() -> float:
            result = db_boundary_gate(
                operation="pre_commit",
                tables_affected=["extracted_files"],
                fs_success=True,
            )
            return result.elapsed_ms

        timings: List[float] = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(invoke_gate) for _ in range(1000)]
            for future in as_completed(futures):
                timings.append(future.result())

        assert len(timings) == 1000
        avg = sum(timings) / len(timings)
        p95 = sorted(timings)[int(len(timings) * 0.95)]

        print(f"\nDB Boundary Gate Load Test (1000 calls):")
        print(f"  Average: {avg:.4f}ms")
        print(f"  P95: {p95:.4f}ms")
        assert avg < 1.0
        assert p95 < 5.0


class TestCombinedGateLoad:
    """Test gates under combined realistic load."""

    def test_all_gates_concurrent_mixed_load(self):
        """All gates should handle concurrent mixed-gate load."""

        def url_call() -> tuple[str, float]:
            result = url_gate(
                "https://example.com/path",
                allowed_hosts=["example.com"],
                allowed_ports=[443],
            )
            return ("url", result.elapsed_ms)

        def extraction_call() -> tuple[str, float]:
            result = extraction_gate(
                entries_total=1000,
                bytes_declared=100_000_000,
                max_total_ratio=10.0,
                max_entry_ratio=200_000,
            )
            return ("extraction", result.elapsed_ms)

        def db_call() -> tuple[str, float]:
            result = db_boundary_gate(
                operation="pre_commit",
                tables_affected=["extracted_files"],
                fs_success=True,
            )
            return ("db_boundary", result.elapsed_ms)

        results: Dict[str, List[float]] = {"url": [], "extraction": [], "db_boundary": []}

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            # 300 URL gate calls
            for _ in range(300):
                futures.append(executor.submit(url_call))
            # 200 Extraction gate calls
            for _ in range(200):
                futures.append(executor.submit(extraction_call))
            # 300 DB boundary gate calls
            for _ in range(300):
                futures.append(executor.submit(db_call))

            for future in as_completed(futures):
                gate_type, elapsed = future.result()
                results[gate_type].append(elapsed)

        print(f"\nMixed Gate Load Test (800 total calls):")
        for gate_type, timings in results.items():
            avg = sum(timings) / len(timings)
            max_time = max(timings)
            print(f"  {gate_type}: avg={avg:.4f}ms, max={max_time:.4f}ms")
            assert avg < 5.0  # All gates should average under 5ms under load

    def test_gates_sustained_load_30_seconds(self):
        """Gates should maintain performance under 30-second sustained load."""

        def mixed_calls() -> List[float]:
            """Make mixed gate calls for one iteration."""
            times = []

            # URL gate call
            result = url_gate(
                "https://example.com/path",
                allowed_hosts=["example.com"],
                allowed_ports=[443],
            )
            times.append(result.elapsed_ms)

            # Extraction gate call
            result = extraction_gate(
                entries_total=500,
                bytes_declared=50_000_000,
                max_total_ratio=10.0,
                max_entry_ratio=200_000,
            )
            times.append(result.elapsed_ms)

            # DB boundary gate call
            result = db_boundary_gate(
                operation="pre_commit",
                tables_affected=["extracted_files"],
                fs_success=True,
            )
            times.append(result.elapsed_ms)

            return times

        start = time.perf_counter()
        all_timings: List[float] = []
        iteration = 0

        while time.perf_counter() - start < 30:  # 30 second sustained load
            timings = mixed_calls()
            all_timings.extend(timings)
            iteration += 1

        elapsed = time.perf_counter() - start
        total_calls = len(all_timings)
        avg = sum(all_timings) / len(all_timings)
        max_time = max(all_timings)

        print(f"\n30-Second Sustained Load Test:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Total calls: {total_calls}")
        print(f"  Calls/sec: {total_calls / elapsed:.2f}")
        print(f"  Average time: {avg:.4f}ms")
        print(f"  Max time: {max_time:.4f}ms")

        assert avg < 5.0  # Average under 5ms
        assert max_time < 50.0  # No call exceeded 50ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
