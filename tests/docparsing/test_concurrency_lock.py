"""Stress-test advisory lock behaviour used throughout DocParsing.

The chunking and embedding pipelines rely on ``acquire_lock`` (wrapping
``filelock.FileLock``) to guard manifest writes and temporary resources. These
tests spawn real processes to ensure the lock enforces mutual exclusion, keeps
its `.lock` file tidy, and honours timeout semantics.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import pytest

from DocsToKG.DocParsing.core.concurrency import acquire_lock


def _lock_worker(
    lock_target: str,
    start_event: Any,
    queue: Any,
    hold_time: float,
) -> None:
    """Attempt to acquire ``lock_target`` and report entry/exit events."""

    path = Path(lock_target)
    start_event.wait()
    with acquire_lock(path, timeout=5.0):
        queue.put(("acquired", os.getpid(), time.time()))
        time.sleep(hold_time)
        queue.put(("released", os.getpid(), time.time()))


def test_acquire_lock_is_mutually_exclusive(tmp_path: Path) -> None:
    """Processes contending for a lock should enter the critical section sequentially."""

    lock_target = tmp_path / "shared-resource"
    ctx = multiprocessing.get_context("spawn")
    start_event = ctx.Event()
    queue: multiprocessing.queues.Queue = ctx.Queue()
    hold_time = 0.2

    processes = [
        ctx.Process(
            target=_lock_worker,
            args=(str(lock_target), start_event, queue, hold_time),
        )
        for _ in range(2)
    ]

    try:
        for proc in processes:
            proc.start()

        start_event.set()

        events: List[Tuple[str, int, float]] = []
        for _ in range(4):
            events.append(queue.get(timeout=10.0))

        for proc in processes:
            proc.join(timeout=5.0)
            assert proc.exitcode == 0

    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        queue.close()
        queue.join_thread()

    _assert_events_sequential(events)
    assert not lock_target.with_suffix(lock_target.suffix + ".lock").exists()


def _assert_events_sequential(events: Iterable[Tuple[str, int, float]]) -> None:
    """Ensure lock acquisition events never overlap in time."""

    active: set[int] = set()
    for kind, pid, _ in events:
        if kind == "acquired":
            assert not active, f"Lock acquired by {pid} while held by {active}"
            active.add(pid)
        elif kind == "released":
            assert pid in active, f"Process {pid} released lock it did not hold"
            active.remove(pid)
        else:  # pragma: no cover - defensive guard
            raise AssertionError(f"Unexpected event kind: {kind}")

    assert not active, f"Lock still held by {active} at test completion"


def test_acquire_lock_creates_nested_directory(tmp_path: Path) -> None:
    """Locks should create their directories and clean up after release."""

    target_path = tmp_path / "nested" / "deeper" / "artifact.jsonl"
    lock_path = target_path.with_suffix(target_path.suffix + ".lock")

    with acquire_lock(target_path):
        assert lock_path.exists()

    assert not lock_path.exists()


@pytest.mark.parametrize("raw_pid", ["", "not-a-pid"])
def test_acquire_lock_recovers_from_invalid_pid(tmp_path: Path, raw_pid: str) -> None:
    """Invalid lock contents should be treated as stale and recovered quickly."""

    target_path = tmp_path / f"resource-{raw_pid or 'empty'}"
    lock_path = target_path.with_suffix(target_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(raw_pid, encoding="utf-8")

    timeout = 0.5
    start = time.time()
    with acquire_lock(target_path, timeout=timeout):
        assert lock_path.exists()
    elapsed = time.time() - start

    assert elapsed < timeout - 0.1
    assert not lock_path.exists()
