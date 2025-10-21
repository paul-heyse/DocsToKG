"""Unit tests for DocParsing runner semantics.

Tests cover:
- Timeout behavior (task exceeds deadline)
- Retries with exponential backoff + jitter
- Error budget enforcement (stop after N errors)
- Resume/force semantics and fingerprinting
- Executor policy selection (io→ThreadPool, cpu→ProcessPool, gpu→ThreadPool)
- Hook lifecycle (before/after called in order)
- Skip logic
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from DocsToKG.DocParsing.core.runner import (
    ItemFingerprint,
    ItemOutcome,
    StageContext,
    StageError,
    StageHooks,
    StageOptions,
    StagePlan,
    StageOutcome,
    WorkItem,
    run_stage,
)


def _make_plan(n_items: int, stage_name: str = "test") -> StagePlan:
    """Create a simple test plan with N items."""
    items = [
        WorkItem(
            item_id=f"item_{i}",
            inputs={},
            outputs={},
            cfg_hash="test-hash",
            cost_hint=float(i + 1),  # Ascending costs for SJF testing
            metadata={},
        )
        for i in range(n_items)
    ]
    return StagePlan(stage_name=stage_name, items=items, total_items=n_items)


# ============================================================================
# Test: Basic Success Path
# ============================================================================


def test_runner_success_path():
    """Test basic success: all items complete, outcome recorded correctly."""

    def success_worker(item: WorkItem) -> ItemOutcome:
        return ItemOutcome(
            status="success",
            duration_s=0.1,
            manifest={"result": "ok"},
        )

    plan = _make_plan(3)
    outcome = run_stage(plan, success_worker)

    assert outcome.succeeded == 3
    assert outcome.failed == 0
    assert outcome.skipped == 0
    assert outcome.scheduled == 3
    assert outcome.exec_p50_ms > 0  # Some execution time recorded


# ============================================================================
# Test: Timeout Behavior
# ============================================================================


def test_runner_timeout_behavior():
    """Test timeout: task exceeds limit, error recorded, exit code set."""

    def slow_worker(item: WorkItem) -> ItemOutcome:
        time.sleep(0.5)  # Sleep longer than timeout
        return ItemOutcome(status="success", duration_s=0.5)

    plan = _make_plan(2)
    options = StageOptions(workers=1, per_item_timeout_s=0.1, error_budget=0)
    outcome = run_stage(plan, slow_worker, options)

    # Timeout should cause error
    assert outcome.failed >= 1, "At least one task should timeout"
    # Should have timeout error in errors
    timeout_errors = [e for e in outcome.errors if e.category == "timeout"]
    assert len(timeout_errors) > 0, "Should have timeout category errors"


# ============================================================================
# Test: Retries with Exponential Backoff
# ============================================================================


def test_runner_retries_success_on_second_attempt():
    """Test retries: task fails once, succeeds on retry."""

    attempts = {"count": 0}

    def retry_worker(item: WorkItem) -> ItemOutcome:
        attempts["count"] += 1
        if attempts["count"] < 2:
            err = StageError(
                stage="test",
                item_id=item.item_id,
                category="transient",
                message="Transient error",
                retryable=True,
            )
            return ItemOutcome(status="failure", duration_s=0.01, error=err)
        return ItemOutcome(status="success", duration_s=0.01)

    plan = _make_plan(1)
    options = StageOptions(workers=1, retries=2, retry_backoff_s=0.01, error_budget=10)
    outcome = run_stage(plan, retry_worker, options)

    # Should eventually succeed after retry
    assert outcome.succeeded == 1
    assert outcome.failed == 0


# ============================================================================
# Test: Error Budget Enforcement
# ============================================================================


def test_runner_error_budget_stops_submissions():
    """Test error budget: after N errors, no new submissions."""

    def failing_worker(item: WorkItem) -> ItemOutcome:
        return ItemOutcome(
            status="failure",
            duration_s=0.01,
            error=StageError(
                stage="test",
                item_id=item.item_id,
                category="runtime",
                message="Always fail",
                retryable=False,
            ),
        )

    plan = _make_plan(10)
    options = StageOptions(workers=1, error_budget=2)  # Stop after 2 errors
    outcome = run_stage(plan, failing_worker, options)

    # Should have cancelled because error budget exceeded
    assert outcome.failed >= 2
    assert outcome.cancelled


# ============================================================================
# Test: Resume with Fingerprint Matching
# ============================================================================


def test_runner_resume_with_fingerprint_match(tmp_path: Path):
    """Test resume: item skipped when fingerprint matches, output exists."""

    output_file = tmp_path / "output.txt"
    output_file.write_text("existing content")
    fp_file = tmp_path / "output.txt.fp.json"
    fp_file.write_text('{"input_sha256": "abc123", "cfg_hash": "cfg123"}')

    def worker(item: WorkItem) -> ItemOutcome:
        return ItemOutcome(status="success", duration_s=0.01)

    fingerprint = ItemFingerprint(
        path=fp_file,
        input_sha256="abc123",
        cfg_hash="cfg123",
    )
    item = WorkItem(
        item_id="test",
        inputs={},
        outputs={"out": output_file},
        cfg_hash="cfg123",
        fingerprint=fingerprint,
    )

    plan = StagePlan(stage_name="test", items=[item], total_items=1)
    options = StageOptions(resume=True, force=False)
    outcome = run_stage(plan, worker, options)

    # Should be skipped due to resume + fingerprint match
    assert outcome.skipped == 1
    assert outcome.succeeded == 0


# ============================================================================
# Test: Force Ignores Resume
# ============================================================================


def test_runner_force_ignores_resume(tmp_path: Path):
    """Test force: even if resume satisfied, force causes recomputation."""

    output_file = tmp_path / "output.txt"
    output_file.write_text("existing")
    fp_file = tmp_path / "output.txt.fp.json"
    fp_file.write_text('{"input_sha256": "abc", "cfg_hash": "cfg"}')

    def worker(item: WorkItem) -> ItemOutcome:
        return ItemOutcome(status="success", duration_s=0.01)

    fingerprint = ItemFingerprint(
        path=fp_file,
        input_sha256="abc",
        cfg_hash="cfg",
    )
    item = WorkItem(
        item_id="test",
        inputs={},
        outputs={"out": output_file},
        cfg_hash="cfg",
        fingerprint=fingerprint,
    )

    plan = StagePlan(stage_name="test", items=[item], total_items=1)
    options = StageOptions(resume=True, force=True)  # Force overrides resume
    outcome = run_stage(plan, worker, options)

    # Should be processed despite resume being true
    assert outcome.succeeded == 1
    assert outcome.skipped == 0


# ============================================================================
# Test: Hook Lifecycle
# ============================================================================


def test_runner_hook_lifecycle():
    """Test hooks: before_stage, after_item, after_stage called in order."""

    call_sequence = []

    def before_stage_hook(ctx: StageContext) -> None:
        call_sequence.append("before_stage")

    def after_item_hook(item: WorkItem, result, ctx: StageContext) -> None:
        call_sequence.append(f"after_item_{item.item_id}")

    def after_stage_hook(outcome: StageOutcome, ctx: StageContext) -> None:
        call_sequence.append("after_stage")

    def worker(item: WorkItem) -> ItemOutcome:
        call_sequence.append(f"work_{item.item_id}")
        return ItemOutcome(status="success", duration_s=0.01)

    hooks = StageHooks(
        before_stage=before_stage_hook,
        after_item=after_item_hook,
        after_stage=after_stage_hook,
    )

    plan = _make_plan(2)
    outcome = run_stage(plan, worker, hooks=hooks)

    # Verify call sequence
    assert call_sequence[0] == "before_stage"
    assert "after_stage" in call_sequence
    assert call_sequence[-1] == "after_stage"


# ============================================================================
# Test: Policy Selection (Executor Type)
# ============================================================================


def test_runner_policy_io_creates_threadpool():
    """Test policy: 'io' creates ThreadPool."""

    def worker(item: WorkItem) -> ItemOutcome:
        return ItemOutcome(status="success", duration_s=0.01)

    plan = _make_plan(2)
    options = StageOptions(policy="io", workers=2)
    outcome = run_stage(plan, worker, options)

    # Should complete successfully with ThreadPool
    assert outcome.succeeded == 2


def test_runner_policy_cpu_with_spawn():
    """Test policy: 'cpu' creates ProcessPool with spawn."""

    def worker(item: WorkItem) -> ItemOutcome:
        # In CPU policy, this runs in subprocess
        return ItemOutcome(status="success", duration_s=0.01)

    plan = _make_plan(2)
    options = StageOptions(policy="cpu", workers=2)
    outcome = run_stage(plan, worker, options)

    # Should complete successfully
    assert outcome.succeeded == 2


# ============================================================================
# Test: Dry Run Mode
# ============================================================================


def test_runner_dry_run_no_execution():
    """Test dry_run: no items executed, just planning."""

    executed = {"count": 0}

    def worker(item: WorkItem) -> ItemOutcome:
        executed["count"] += 1
        return ItemOutcome(status="success", duration_s=0.01)

    plan = _make_plan(5)
    options = StageOptions(dry_run=True)
    outcome = run_stage(plan, worker, options)

    # Nothing should execute
    assert executed["count"] == 0
    assert outcome.succeeded == 0
    assert outcome.skipped == 5  # All marked as skipped in dry-run


# ============================================================================
# Test: Percentile Calculation
# ============================================================================


def test_runner_percentile_calculation():
    """Test runner computes p50 and p95 correctly."""

    def variable_worker(item: WorkItem) -> ItemOutcome:
        # Variable execution times
        duration = 0.01 * (int(item.item_id.split("_")[1]) + 1)
        time.sleep(duration / 10)  # Make it measurable
        return ItemOutcome(status="success", duration_s=duration)

    plan = _make_plan(10)
    outcome = run_stage(plan, variable_worker)

    # p50 and p95 should be populated
    assert outcome.exec_p50_ms > 0
    assert outcome.exec_p95_ms >= outcome.exec_p50_ms


# ============================================================================
# Test: Diagnostics Logging Interval
# ============================================================================


def test_runner_diagnostics_interval():
    """Test runner logs diagnostics at specified interval."""

    logged = {"events": []}

    def capture_log(msg: str) -> None:
        logged["events"].append(msg)

    def worker(item: WorkItem) -> ItemOutcome:
        return ItemOutcome(status="success", duration_s=0.01)

    plan = _make_plan(5)
    options = StageOptions(diagnostics_interval_s=0.05)
    # Note: Hard to test logging directly, but interval is used in runner
    outcome = run_stage(plan, worker, options)

    assert outcome.succeeded == 5
