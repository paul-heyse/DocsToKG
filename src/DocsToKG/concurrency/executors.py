"""Executor factory utilities used by DocParsing stage runner."""

from __future__ import annotations

from concurrent import futures
from multiprocessing import get_context
from typing import Optional, Tuple

Executor = futures.Executor


def create_executor(policy: str, workers: int) -> Tuple[Optional[Executor], bool]:
    """
    Return an executor configured for the given policy.

    Args:
        policy: Execution policy; ``"cpu"`` selects a process pool, anything
            else defaults to a thread-based pool suitable for IO-bound work.
        workers: Desired concurrency level.

    Returns:
        Tuple of (executor, needs_shutdown). Caller is responsible for shutting
        down the returned executor when ``needs_shutdown`` is ``True``.
    """
    normalized = (policy or "io").lower()
    if workers <= 1:
        return None, False
    if normalized == "cpu":
        mp_ctx = get_context("spawn")
        return futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx), True
    return futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="docparse-stage"), True

