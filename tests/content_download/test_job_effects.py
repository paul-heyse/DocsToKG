"""Regression tests for job effect caching."""

import sqlite3

from DocsToKG.ContentDownload.job_effects import run_effect


def _create_artifact_ops_table(cx: sqlite3.Connection) -> None:
    cx.executescript(
        """
        CREATE TABLE artifact_ops (
            op_key TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            op_type TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            result_code TEXT,
            result_json TEXT
        );
        """
    )


def test_run_effect_handles_default_row_factory() -> None:
    """run_effect should replay cached results even without sqlite3.Row support."""

    cx = sqlite3.connect(":memory:")
    _create_artifact_ops_table(cx)

    calls: list[dict[str, str]] = []

    def effect_fn() -> dict[str, str]:
        payload = {"code": "OK", "value": "cached"}
        calls.append(payload)
        return payload

    first_result = run_effect(
        cx,
        job_id="job-1",
        kind="HEAD",
        opkey="op-1",
        effect_fn=effect_fn,
    )

    second_result = run_effect(
        cx,
        job_id="job-1",
        kind="HEAD",
        opkey="op-1",
        effect_fn=effect_fn,
    )

    assert first_result == second_result == {"code": "OK", "value": "cached"}
    # Without normalization run_effect would attempt row["result_json"] and fail.
    assert len(calls) == 1
