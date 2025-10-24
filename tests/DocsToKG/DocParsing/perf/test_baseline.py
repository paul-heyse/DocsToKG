"""Unit tests for DocParsing performance baseline comparison."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[4] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from DocsToKG.DocParsing.perf.baseline import compare_metrics, load_baseline  # noqa: E402
from DocsToKG.DocParsing.perf.runner import StageMetrics  # noqa: E402


def _write_metrics(
    tmp_path: Path,
    stage: str,
    wall: float,
    cpu: float,
    rss: int | None,
) -> StageMetrics:
    stdout = tmp_path / f"{stage}.stdout"
    stderr = tmp_path / f"{stage}.stderr"
    stdout.write_text("stdout", encoding="utf-8")
    stderr.write_text("stderr", encoding="utf-8")
    return StageMetrics(
        stage=stage,
        command=[stage],
        wall_time_s=wall,
        cpu_time_s=cpu,
        max_rss_bytes=rss,
        exit_code=0,
        timestamp=datetime.now(UTC),
        stdout_path=stdout,
        stderr_path=stderr,
    )


def test_compare_metrics_detects_regressions(tmp_path: Path) -> None:
    """Regressions above the configured threshold should be reported."""

    baseline_payload = {
        "stages": [
            {
                "stage": "chunk",
                "wall_time_s": 10.0,
                "cpu_time_s": 8.0,
                "max_rss_bytes": 1_000,
            }
        ]
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    current_metrics = [
        _write_metrics(tmp_path, "chunk", wall=12.5, cpu=9.5, rss=1_400),
    ]

    result = compare_metrics(
        current=current_metrics,
        baseline=load_baseline(baseline_path),
        wall_threshold=0.1,
        cpu_threshold=0.1,
        rss_threshold=0.2,
    )

    assert result.regressions, "Expected regression to be reported"
    assert not result.improvements
    assert not result.unchanged


def test_compare_metrics_flags_improvements(tmp_path: Path) -> None:
    """Large improvements should be highlighted to aid triage."""

    baseline_payload = {
        "stages": [
            {
                "stage": "embed",
                "wall_time_s": 10.0,
                "cpu_time_s": 9.0,
                "max_rss_bytes": 2_000,
            }
        ]
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    current_metrics = [
        _write_metrics(tmp_path, "embed", wall=6.0, cpu=6.5, rss=1_200),
    ]

    result = compare_metrics(
        current=current_metrics,
        baseline=load_baseline(baseline_path),
        wall_threshold=0.1,
        cpu_threshold=0.1,
        rss_threshold=0.2,
    )

    assert not result.regressions
    assert any("wall_time_s" in entry for entry in result.improvements)
    assert any("max_rss_bytes" in entry for entry in result.improvements)
