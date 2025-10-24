# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.perf.runner",
#   "purpose": "Execute DocParsing stages and collect timing/memory metrics.",
#   "sections": [
#     {
#       "id": "stage-metrics",
#       "name": "StageMetrics",
#       "anchor": "class-stage-metrics",
#       "kind": "class"
#     },
#     {
#       "id": "run-stage",
#       "name": "run_stage",
#       "anchor": "function-run-stage",
#       "kind": "function"
#     },
#     {
#       "id": "convert-profile",
#       "name": "convert_profile_to_collapsed",
#       "anchor": "function-convert-profile",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Execute DocParsing stages and collect timing/memory metrics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - platform without resource (e.g. Windows)
    resource = None  # type: ignore[assignment]

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import pstats
except Exception as exc:  # pragma: no cover - standard lib should always exist
    raise RuntimeError("pstats module is required for profiling support") from exc


@dataclass(slots=True)
class StageMetrics:
    """Structured metrics captured for a single DocParsing stage."""

    stage: str
    command: Sequence[str]
    wall_time_s: float
    cpu_time_s: float
    max_rss_bytes: int | None
    exit_code: int
    timestamp: datetime
    stdout_path: Path
    stderr_path: Path
    profile_path: Path | None = None
    collapsed_profile_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON serialisable payload."""

        return {
            "stage": self.stage,
            "command": list(self.command),
            "wall_time_s": self.wall_time_s,
            "cpu_time_s": self.cpu_time_s,
            "max_rss_bytes": self.max_rss_bytes,
            "exit_code": self.exit_code,
            "timestamp": self.timestamp.isoformat(),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "profile_path": str(self.profile_path) if self.profile_path else None,
            "collapsed_profile_path": (
                str(self.collapsed_profile_path) if self.collapsed_profile_path else None
            ),
            "extra": self.extra,
        }


def _format_func(func: tuple[str, int, str]) -> str:
    """Format a cProfile function tuple into a human friendly label."""

    filename, line_no, func_name = func
    base = Path(filename).name
    return f"{base}:{line_no}:{func_name}"


def convert_profile_to_collapsed(profile_path: Path, collapsed_path: Path) -> None:
    """Convert a cProfile dump to a simple collapsed stack representation."""

    stats = pstats.Stats(str(profile_path))
    stats.calc_callees()
    callers = {func: details[4] for func, details in stats.stats.items()}
    callees = getattr(stats, "all_callees", {})

    roots: set[tuple[str, int, str]] = set(stats.stats)
    for func_callers in callers.values():
        for caller in func_callers:
            roots.discard(caller)

    collapsed_lines: list[str] = []

    def dfs(
        func: tuple[str, int, str],
        prefix: tuple[str, ...],
        weight: float,
        visited: set[tuple[str, int, str]],
    ) -> None:
        if func in visited:
            return
        visited.add(func)
        node_label = _format_func(func)
        entry = prefix + (node_label,)
        cc, nc, tt, _ct, _callers = stats.stats[func]
        exclusive = max(tt, 0.0) * max(weight, 0.0)
        if exclusive > 0:
            collapsed_lines.append(f"{'/'.join(entry)} {exclusive:.6f}")
        children = callees.get(func, {})
        total_calls = sum(children.values())
        if total_calls > 0:
            for child, count in children.items():
                child_weight = weight * (count / total_calls) if total_calls else weight
                dfs(child, entry, child_weight, visited.copy())
        visited.discard(func)

    if not roots:
        roots = set(stats.stats)

    for root in roots:
        dfs(root, tuple(), 1.0, set())

    collapsed_path.write_text("\n".join(collapsed_lines), encoding="utf-8")


def _monitor_process_memory(proc: subprocess.Popen[str]) -> Callable[[], int | None] | None:
    """Start a psutil-based memory sampler and return a finaliser."""

    if psutil is None:
        return None

    try:
        process = psutil.Process(proc.pid)
    except psutil.Error:  # pragma: no cover - process exited quickly
        return None

    rss_holder: dict[str, int] = {"value": 0}
    stop_event = threading.Event()

    def _poll() -> None:
        try:
            while not stop_event.is_set():
                try:
                    rss = process.memory_info().rss
                    rss_holder["value"] = max(rss_holder["value"], int(rss))
                except psutil.Error:
                    break
                for child in process.children(recursive=True):
                    try:
                        rss = child.memory_info().rss
                        rss_holder["value"] = max(rss_holder["value"], int(rss))
                    except psutil.Error:
                        continue
                if proc.poll() is not None:
                    break
                if stop_event.wait(0.1):
                    break
        finally:
            stop_event.set()

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()

    def _finalise() -> int | None:
        stop_event.set()
        thread.join(timeout=1.0)
        return rss_holder["value"] or None

    return _finalise


def run_stage(
    *,
    stage: str,
    command: Sequence[str],
    output_dir: Path,
    profile: bool,
    env: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> StageMetrics:
    """Execute a DocParsing CLI stage and collect metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / f"{stage}.stdout.log"
    stderr_path = output_dir / f"{stage}.stderr.log"
    profile_path = output_dir / f"{stage}.pstats"
    collapsed_path = output_dir / f"{stage}.collapsed.txt"

    exe = sys.executable
    if profile:
        cmd = [exe, "-m", "cProfile", "-o", str(profile_path), "-m", "DocsToKG.DocParsing.cli"]
    else:
        cmd = [exe, "-m", "DocsToKG.DocParsing.cli"]
    cmd.extend(command)

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    start_usage = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **(env or {})},
    )

    monitor = _monitor_process_memory(proc)
    stdout, stderr = proc.communicate()
    exit_code = proc.returncode or 0

    max_rss = monitor() if monitor else None

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    end_usage = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None

    wall_time = end_wall - start_wall
    if start_usage and end_usage:
        cpu_time = (end_usage.ru_utime + end_usage.ru_stime) - (
            start_usage.ru_utime + start_usage.ru_stime
        )
    else:
        cpu_time = end_cpu - start_cpu
    if max_rss is None and end_usage:
        max_rss = end_usage.ru_maxrss * 1024

    collapsed_profile_path = None
    if profile and profile_path.exists() and exit_code == 0:
        convert_profile_to_collapsed(profile_path, collapsed_path)
        collapsed_profile_path = collapsed_path
    else:
        profile_path = profile_path if profile and profile_path.exists() else None

    metrics = StageMetrics(
        stage=stage,
        command=command,
        wall_time_s=wall_time,
        cpu_time_s=cpu_time,
        max_rss_bytes=max_rss,
        exit_code=exit_code,
        timestamp=datetime.utcnow(),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        profile_path=profile_path if profile and profile_path else None,
        collapsed_profile_path=collapsed_profile_path,
        extra=extra or {},
    )

    metrics_path = output_dir / f"{stage}.metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_json(), indent=2), encoding="utf-8")

    return metrics
