import sys
import importlib.util
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "jsonlines" not in sys.modules:
    jsonlines_stub = types.ModuleType("jsonlines")
    jsonlines_stub.open = lambda *args, **kwargs: None
    sys.modules["jsonlines"] = jsonlines_stub

if "certifi" not in sys.modules:
    certifi_stub = types.ModuleType("certifi")
    certifi_stub.where = lambda: ""
    sys.modules["certifi"] = certifi_stub

if "DocsToKG.DocParsing.logging" not in sys.modules:
    logging_stub = types.ModuleType("DocsToKG.DocParsing.logging")

    class _StubLogger:
        def __init__(self) -> None:
            self.logger = self

        def log(self, *args, **kwargs):  # pragma: no cover - stub
            pass

        def warning(self, *args, **kwargs):  # pragma: no cover - stub
            pass

        def info(self, *args, **kwargs):  # pragma: no cover - stub
            pass

    def _get_logger(*args, **kwargs):  # pragma: no cover - stub
        return _StubLogger()

    def _log_event(*args, **kwargs):  # pragma: no cover - stub
        pass

    logging_stub.get_logger = _get_logger
    logging_stub.log_event = _log_event
    sys.modules["DocsToKG.DocParsing.logging"] = logging_stub


def _load_runner_module():
    runner_path = SRC / "DocsToKG" / "DocParsing" / "core" / "runner.py"
    spec = importlib.util.spec_from_file_location("docparsing_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("docparsing_runner", module)
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()
ItemOutcome = runner.ItemOutcome
StageHooks = runner.StageHooks
StageOptions = runner.StageOptions
StagePlan = runner.StagePlan
WorkItem = runner.WorkItem
run_stage = runner.run_stage


def _make_plan() -> StagePlan:
    item = WorkItem(
        item_id="doc-1",
        inputs={},
        outputs={},
        cfg_hash="cfg",
    )
    return StagePlan(stage_name="timeout", items=[item], total_items=1)


def test_runner_timeout_cancels_and_retries():
    plan = _make_plan()
    options = StageOptions(
        workers=2,
        per_item_timeout_s=0.05,
        retries=1,
        retry_backoff_s=0.0,
    )

    active_events: list[threading.Event] = []
    attempt_count = 0

    def worker(_: WorkItem) -> ItemOutcome:
        nonlocal attempt_count
        attempt_count += 1
        release = threading.Event()
        active_events.append(release)
        while not release.is_set():
            time.sleep(0.005)
        return ItemOutcome(status="success", duration_s=0.0)

    def after_item(_: WorkItem, outcome_or_error, __):
        if isinstance(outcome_or_error, Exception) and getattr(
            outcome_or_error, "category", None
        ) == "timeout":
            # Unblock the worker associated with the current attempt.
            active_events[-1].set()

    hooks = StageHooks(after_item=after_item)

    outcome = run_stage(plan, worker, options=options, hooks=hooks)

    assert attempt_count == 2
    assert outcome.failed == 1
    assert outcome.succeeded == 0
    assert not outcome.cancelled
    assert [error.category for error in outcome.errors] == ["timeout", "timeout"]
    assert all(error.retryable for error in outcome.errors)
