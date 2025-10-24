"""Unit tests covering DocParsing stage runner metrics emission."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "jsonlines" not in sys.modules:
    jsonlines_stub = types.ModuleType("jsonlines")

    class _Reader:
        def __init__(self, *args, **kwargs) -> None:
            self._closed = False

        def read(self) -> None:
            raise EOFError

        def close(self) -> None:
            self._closed = True

    class _InvalidLineError(Exception):
        pass

    jsonlines_stub.Reader = _Reader
    jsonlines_stub.InvalidLineError = _InvalidLineError
    sys.modules["jsonlines"] = jsonlines_stub

if "filelock" not in sys.modules:
    filelock_stub = types.ModuleType("filelock")

    class _Timeout(Exception):
        pass

    class _FileLock:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def acquire(self, *args, **kwargs) -> None:
            return None

        def release(self) -> None:
            return None

    filelock_stub.FileLock = _FileLock
    filelock_stub.Timeout = _Timeout
    sys.modules["filelock"] = filelock_stub

if "certifi" not in sys.modules:
    certifi_stub = types.ModuleType("certifi")

    def _where() -> str:
        return ""

    certifi_stub.where = _where  # type: ignore[attr-defined]
    sys.modules["certifi"] = certifi_stub

logging_stub = types.ModuleType("DocsToKG.DocParsing.logging")


class _StubStructuredLogger:
    def __init__(self, *args, **kwargs) -> None:
        self.logger = self

    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


def _get_logger(*args, **kwargs) -> _StubStructuredLogger:
    return _StubStructuredLogger()


def _log_event(*args, **kwargs) -> None:
    return None


logging_stub.StructuredLogger = _StubStructuredLogger  # type: ignore[attr-defined]
logging_stub.get_logger = _get_logger  # type: ignore[attr-defined]
logging_stub.log_event = _log_event  # type: ignore[attr-defined]
sys.modules["DocsToKG.DocParsing.logging"] = logging_stub

import DocsToKG  # noqa: E402
import DocsToKG.DocParsing  # noqa: E402

core_stub = types.ModuleType("DocsToKG.DocParsing.core")
core_stub.__path__ = [str((SRC / "DocsToKG" / "DocParsing" / "core").resolve())]
sys.modules["DocsToKG.DocParsing.core"] = core_stub

runner_module = importlib.import_module("DocsToKG.DocParsing.core.runner")

ItemOutcome = runner_module.ItemOutcome
StageContext = runner_module.StageContext
StageHooks = runner_module.StageHooks
StageOptions = runner_module.StageOptions
StageOutcome = runner_module.StageOutcome
StagePlan = runner_module.StagePlan
WorkItem = runner_module.WorkItem
run_stage = runner_module.run_stage


def _cpu_bound_worker(item: WorkItem) -> ItemOutcome:
    """Simple worker that performs CPU work to exercise timing metrics."""

    n = int(item.metadata.get("iterations", 75_000))
    accumulator = 0
    for idx in range(n):
        accumulator += idx % 7
    return ItemOutcome(
        status="success",
        duration_s=0.0,
        manifest={},
        result={"acc": accumulator},
    )


def _build_plan(iterations: list[int]) -> StagePlan:
    items: list[WorkItem] = []
    for index, count in enumerate(iterations):
        metadata: dict[str, Any] = {"iterations": count}
        items.append(
            WorkItem(
                item_id=f"item-{index}",
                inputs={},
                outputs={},
                cfg_hash="cfg",
                metadata=metadata,
            )
        )
    return StagePlan(stage_name="metrics-test", items=tuple(items), total_items=len(items))


def test_run_stage_exposes_extended_metrics() -> None:
    """``run_stage`` returns and surfaces extended latency + CPU metrics."""

    plan = _build_plan([90_000, 120_000, 150_000])
    options = StageOptions(workers=1)

    captured: list[StageOutcome] = []

    def _after_stage(outcome: StageOutcome, context: StageContext) -> None:
        captured.append(outcome)

    hooks = StageHooks(after_stage=_after_stage)
    outcome = run_stage(plan, _cpu_bound_worker, options, hooks)

    assert captured, "Stage hooks should receive the computed outcome"
    recorded = captured[0]
    assert recorded == outcome

    assert outcome.exec_p50_ms > 0.0
    assert outcome.exec_p95_ms >= outcome.exec_p50_ms
    assert outcome.exec_p99_ms >= outcome.exec_p95_ms
    assert outcome.queue_p50_ms >= 0.0
    assert outcome.queue_p95_ms >= outcome.queue_p50_ms
    assert outcome.cpu_time_total_ms >= outcome.exec_p50_ms
    assert outcome.cpu_time_total_ms > 0.0
