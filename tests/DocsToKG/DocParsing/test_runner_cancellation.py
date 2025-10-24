"""Tests covering cancellation semantics for the DocParsing stage runner."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"

if "jsonlines" not in sys.modules:
    jsonlines_stub = types.ModuleType("jsonlines")

    class _Reader:
        def __init__(self, *_args, **_kwargs) -> None:
            self._rows: list[dict[str, object]] = []

        def __iter__(self):  # pragma: no cover - defensive stub
            return iter(self._rows)

    jsonlines_stub.Reader = _Reader
    jsonlines_stub.InvalidLineError = ValueError
    sys.modules["jsonlines"] = jsonlines_stub

if "filelock" not in sys.modules:
    filelock_stub = types.ModuleType("filelock")

    class _FileLock:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):  # pragma: no cover - defensive stub
            return self

        def __exit__(self, *_exc) -> None:  # pragma: no cover - defensive stub
            return None

    class _Timeout(Exception):
        pass

    filelock_stub.FileLock = _FileLock
    filelock_stub.Timeout = _Timeout
    sys.modules["filelock"] = filelock_stub

if "certifi" not in sys.modules:
    certifi_stub = types.ModuleType("certifi")
    certifi_stub.where = lambda: ""
    sys.modules["certifi"] = certifi_stub

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _Timeout:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _Limits:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _Request:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _Response:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _HTTPTransport:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    httpx_stub.Client = _Client
    httpx_stub.Timeout = _Timeout
    httpx_stub.Limits = _Limits
    httpx_stub.Request = _Request
    httpx_stub.Response = _Response
    httpx_stub.HTTPTransport = _HTTPTransport
    sys.modules["httpx"] = httpx_stub

if "hishel" not in sys.modules:
    hishel_stub = types.ModuleType("hishel")

    class _CacheTransport:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _Controller:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def configure(self, *_args, **_kwargs) -> None:  # pragma: no cover - defensive stub
            return None

    class _FileStorage:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    hishel_stub.CacheTransport = _CacheTransport
    hishel_stub.Controller = _Controller
    hishel_stub.FileStorage = _FileStorage
    sys.modules["hishel"] = hishel_stub

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(*_args, **_kwargs):  # pragma: no cover - defensive stub
        return {}

    def _safe_dump(*_args, **_kwargs) -> str:  # pragma: no cover - defensive stub
        return ""

    yaml_stub.safe_load = _safe_load
    yaml_stub.safe_dump = _safe_dump
    sys.modules["yaml"] = yaml_stub

if "pydantic" not in sys.modules:
    pydantic_stub = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _ConfigDict(dict):
        pass

    def _Field(*_args, **_kwargs):  # pragma: no cover - defensive stub
        return None

    class _PrivateAttr:
        def __init__(self, default=None) -> None:
            self.default = default

    def _field_validator(*_args, **_kwargs):  # pragma: no cover - defensive stub
        def decorator(func):
            return func

        return decorator

    class _AliasChoices:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    pydantic_stub.BaseModel = _BaseModel
    pydantic_stub.ConfigDict = _ConfigDict
    pydantic_stub.Field = _Field
    pydantic_stub.PrivateAttr = _PrivateAttr
    pydantic_stub.field_validator = _field_validator
    pydantic_stub.AliasChoices = _AliasChoices
    pydantic_stub.ValidationError = type("ValidationError", (Exception,), {})
    sys.modules["pydantic"] = pydantic_stub

if "pydantic_core" not in sys.modules:
    pydantic_core_stub = types.ModuleType("pydantic_core")
    pydantic_core_stub.ValidationError = type("ValidationError", (Exception,), {})
    sys.modules["pydantic_core"] = pydantic_core_stub

if "pydantic_settings" not in sys.modules:
    pydantic_settings_stub = types.ModuleType("pydantic_settings")

    class _BaseSettings:
        model_config = {}

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _SettingsConfigDict(dict):
        pass

    pydantic_settings_stub.BaseSettings = _BaseSettings
    pydantic_settings_stub.SettingsConfigDict = _SettingsConfigDict
    sys.modules["pydantic_settings"] = pydantic_settings_stub


def _install_runner_module():
    module_name = "DocsToKG.DocParsing.core.runner"
    if module_name in sys.modules:
        return sys.modules[module_name]

    names_to_restore = {
        "DocsToKG": sys.modules.get("DocsToKG"),
        "DocsToKG.concurrency": sys.modules.get("DocsToKG.concurrency"),
        "DocsToKG.concurrency.executors": sys.modules.get("DocsToKG.concurrency.executors"),
        "DocsToKG.DocParsing": sys.modules.get("DocsToKG.DocParsing"),
        "DocsToKG.DocParsing.logging": sys.modules.get("DocsToKG.DocParsing.logging"),
        "DocsToKG.DocParsing.core": sys.modules.get("DocsToKG.DocParsing.core"),
    }

    docs_pkg = sys.modules.setdefault("DocsToKG", types.ModuleType("DocsToKG"))
    docs_pkg.__path__ = []

    concurrency_pkg = sys.modules.setdefault(
        "DocsToKG.concurrency", types.ModuleType("DocsToKG.concurrency")
    )
    concurrency_pkg.__path__ = []
    docs_pkg.concurrency = concurrency_pkg

    executors_mod = types.ModuleType("DocsToKG.concurrency.executors")

    def _create_executor_stub(_policy: str, _workers: int):
        return None, False

    executors_mod.create_executor = _create_executor_stub
    sys.modules["DocsToKG.concurrency.executors"] = executors_mod
    concurrency_pkg.executors = executors_mod

    docparse_pkg = sys.modules.setdefault(
        "DocsToKG.DocParsing", types.ModuleType("DocsToKG.DocParsing")
    )
    docparse_pkg.__path__ = []
    docs_pkg.DocParsing = docparse_pkg

    logging_mod = types.ModuleType("DocsToKG.DocParsing.logging")

    class _StubLogger:
        def __init__(self) -> None:
            self.logger = self

        def info(self, *_args, **_kwargs) -> None:  # pragma: no cover - stub behaviour
            return None

        def warning(self, *_args, **_kwargs) -> None:  # pragma: no cover - stub behaviour
            return None

    def _get_logger(*_args, **_kwargs) -> _StubLogger:
        return _StubLogger()

    def _log_event(*_args, **_kwargs) -> None:  # pragma: no cover - stub behaviour
        return None

    logging_mod.get_logger = _get_logger
    logging_mod.log_event = _log_event
    sys.modules["DocsToKG.DocParsing.logging"] = logging_mod
    docparse_pkg.logging = logging_mod

    core_pkg = sys.modules.setdefault("DocsToKG.DocParsing.core", types.ModuleType("DocsToKG.DocParsing.core"))
    core_pkg.__path__ = []
    docparse_pkg.core = core_pkg

    runner_path = SRC_DIR / "DocsToKG/DocParsing/core/runner.py"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError("Unable to load runner module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    for name, original in names_to_restore.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    return module


_runner_module = _install_runner_module()

ItemOutcome = _runner_module.ItemOutcome
StageError = _runner_module.StageError
StageHooks = _runner_module.StageHooks
StageOptions = _runner_module.StageOptions
StagePlan = _runner_module.StagePlan
WorkItem = _runner_module.WorkItem
run_stage = _runner_module.run_stage

def _make_work_item(identifier: str) -> WorkItem:
    """Return a minimal WorkItem suitable for runner tests."""

    dummy_path = Path(f"/tmp/{identifier}")
    return WorkItem(
        item_id=identifier,
        inputs={"source": dummy_path},
        outputs={"dest": dummy_path},
        cfg_hash="cfg",
        metadata={},
    )


def test_run_stage_marks_cancelled_when_error_budget_exceeded():
    """The runner should surface cancellation metadata when the error budget trips."""

    items = [_make_work_item(f"doc-{idx}") for idx in range(3)]
    plan = StagePlan(stage_name="test-stage", items=items, total_items=len(items))

    failing_ids = {items[0].item_id, items[1].item_id}

    def worker(item: WorkItem) -> ItemOutcome:
        if item.item_id in failing_ids:
            error = StageError(
                stage="test-stage",
                item_id=item.item_id,
                category="test",
                message="boom",
                retryable=False,
            )
            return ItemOutcome(
                status="failure",
                duration_s=0.01,
                manifest={},
                result={},
                error=error,
            )
        return ItemOutcome(
            status="success",
            duration_s=0.01,
            manifest={},
            result={},
            error=None,
        )

    captured_metadata: dict[str, object] = {}

    def after_stage(outcome, context) -> None:
        captured_metadata.update(context.metadata)

    hooks = StageHooks(after_stage=after_stage)
    options = StageOptions(workers=1, error_budget=1)

    outcome = run_stage(plan, worker, options, hooks)

    assert outcome.cancelled is True
    assert outcome.cancelled_reason == "error-budget"
    assert outcome.failed == 2
    assert outcome.succeeded == 0
    assert outcome.planned == len(items)
    assert outcome.scheduled == 2
    assert captured_metadata["stage_cancelled"] is True
    assert captured_metadata["stage_cancel_reason"] == "error-budget"
    assert captured_metadata["stage_planned_items"] == len(items)
    assert captured_metadata["stage_scheduled_items"] == 2


def test_run_stage_sets_keyboard_interrupt_cancellation():
    """KeyboardInterrupt from hooks should mark the run as cancelled."""

    items = [_make_work_item(f"doc-{idx}") for idx in range(2)]
    plan = StagePlan(stage_name="interrupt-stage", items=items, total_items=len(items))

    def worker(item: WorkItem) -> ItemOutcome:
        return ItemOutcome(
            status="success",
            duration_s=0.01,
            manifest={},
            result={},
            error=None,
        )

    call_count = 0
    captured_metadata: dict[str, object] = {}

    def after_item(item, outcome, context) -> None:  # pragma: no cover - exercised indirectly
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise KeyboardInterrupt()

    def after_stage(outcome, context) -> None:
        captured_metadata.update(context.metadata)

    hooks = StageHooks(after_item=after_item, after_stage=after_stage)
    options = StageOptions(workers=1)

    outcome = run_stage(plan, worker, options, hooks)

    assert outcome.cancelled is True
    assert outcome.cancelled_reason == "keyboard-interrupt"
    assert outcome.succeeded == 1
    assert outcome.failed == 0
    assert outcome.planned == len(items)
    assert outcome.scheduled == 1
    assert captured_metadata["stage_cancelled"] is True
    assert captured_metadata["stage_cancel_reason"] == "keyboard-interrupt"
    assert captured_metadata["stage_planned_items"] == len(items)
    assert captured_metadata["stage_scheduled_items"] == 1


@pytest.mark.parametrize(
    "field",
    [
        "stage_planned_items",
        "stage_scheduled_items",
        "stage_completed_items",
        "stage_cancelled",
        "stage_cancel_reason",
    ],
)
def test_stage_context_metadata_contains_cancellation_keys(field: str) -> None:
    """All cancellation metadata keys should be populated even without failures."""

    items = [_make_work_item("doc-final")]
    plan = StagePlan(stage_name="metadata-stage", items=items, total_items=1)

    def worker(item: WorkItem) -> ItemOutcome:
        return ItemOutcome(status="success", duration_s=0.0, manifest={}, result={}, error=None)

    captured_metadata: dict[str, object] = {}

    def after_stage(outcome, context) -> None:
        captured_metadata.update(context.metadata)

    hooks = StageHooks(after_stage=after_stage)
    outcome = run_stage(plan, worker, StageOptions(workers=1), hooks)

    assert outcome.cancelled is False
    assert field in captured_metadata
