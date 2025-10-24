import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest

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
StageError = runner.StageError
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
    return StagePlan(stage_name="backoff", items=[item], total_items=1)


def test_runner_backoff_uses_seeded_rng(monkeypatch):
    plan = _make_plan()
    options = StageOptions(
        workers=1,
        retries=1,
        retry_backoff_s=0.2,
        seed=1337,
    )

    sleep_calls: list[float] = []
    monkeypatch.setattr(runner.time, "sleep", lambda value: sleep_calls.append(value))

    attempts: dict[str, int] = {plan.items[0].item_id: 0}

    def worker(item: WorkItem) -> ItemOutcome:
        attempts[item.item_id] += 1
        if attempts[item.item_id] == 1:
            return ItemOutcome(
                status="failure",
                duration_s=0.01,
                error=StageError(
                    stage=plan.stage_name,
                    item_id=item.item_id,
                    category="test",
                    message="boom",
                    retryable=True,
                ),
            )
        return ItemOutcome(status="success", duration_s=0.01)

    outcome = run_stage(plan, worker, options=options)

    assert attempts[plan.items[0].item_id] == 2
    assert outcome.succeeded == 1
    assert outcome.failed == 0
    assert len(sleep_calls) == 1

    expected_delay = runner._apply_backoff(0.2, 1, random.Random(1337))
    assert sleep_calls[0] == pytest.approx(expected_delay)
