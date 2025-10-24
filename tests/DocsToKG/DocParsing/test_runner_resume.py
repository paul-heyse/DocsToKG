"""Tests covering resume semantics enforced by the DocParsing runner."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_original_docparsing = sys.modules.get("DocsToKG.DocParsing")
_original_core = sys.modules.get("DocsToKG.DocParsing.core")
_original_ontology = sys.modules.get("DocsToKG.OntologyDownload")
_original_logging_utils = sys.modules.get("DocsToKG.OntologyDownload.logging_utils")

if "DocsToKG" not in sys.modules:
    import DocsToKG  # noqa: F401  # ensure base package is registered

docparsing_pkg = sys.modules.setdefault(
    "DocsToKG.DocParsing", types.ModuleType("DocsToKG.DocParsing")
)
docparsing_pkg.__path__ = [str(SRC / "DocsToKG" / "DocParsing")]
core_pkg = sys.modules.setdefault("DocsToKG.DocParsing.core", types.ModuleType("DocsToKG.DocParsing.core"))
core_pkg.__path__ = [str(SRC / "DocsToKG" / "DocParsing" / "core")]

ontology_pkg = types.ModuleType("DocsToKG.OntologyDownload")
ontology_pkg.__path__ = []
sys.modules.setdefault("DocsToKG.OntologyDownload", ontology_pkg)

logging_utils_stub = types.ModuleType("DocsToKG.OntologyDownload.logging_utils")


class _StubJSONFormatter:
    """Minimal JSON formatter stub for DocParsing logging imports."""

    def format(self, record) -> str:  # pragma: no cover - stub behaviour
        return str(getattr(record, "message", ""))


logging_utils_stub.JSONFormatter = _StubJSONFormatter
logging_utils_stub.setup_logging = lambda *args, **kwargs: None  # pragma: no cover
sys.modules.setdefault("DocsToKG.OntologyDownload.logging_utils", logging_utils_stub)


manifest_spec = importlib.util.spec_from_file_location(
    "DocsToKG.DocParsing.core.manifest", SRC / "DocsToKG" / "DocParsing" / "core" / "manifest.py"
)
manifest_module = importlib.util.module_from_spec(manifest_spec)
assert manifest_spec.loader is not None
sys.modules[manifest_spec.name] = manifest_module
manifest_spec.loader.exec_module(manifest_module)
ResumeController = manifest_module.ResumeController

runner_spec = importlib.util.spec_from_file_location(
    "DocsToKG.DocParsing.core.runner", SRC / "DocsToKG" / "DocParsing" / "core" / "runner.py"
)
runner_module = importlib.util.module_from_spec(runner_spec)
assert runner_spec.loader is not None
sys.modules[runner_spec.name] = runner_module
runner_spec.loader.exec_module(runner_module)
StageOptions = runner_module.StageOptions
WorkItem = runner_module.WorkItem
_should_skip = runner_module._should_skip

if _original_docparsing is None:
    sys.modules.pop("DocsToKG.DocParsing", None)
else:
    sys.modules["DocsToKG.DocParsing"] = _original_docparsing
if _original_core is None:
    sys.modules.pop("DocsToKG.DocParsing.core", None)
else:
    sys.modules["DocsToKG.DocParsing.core"] = _original_core
if _original_logging_utils is None:
    sys.modules.pop("DocsToKG.OntologyDownload.logging_utils", None)
else:
    sys.modules["DocsToKG.OntologyDownload.logging_utils"] = _original_logging_utils
if _original_ontology is None:
    sys.modules.pop("DocsToKG.OntologyDownload", None)
else:
    sys.modules["DocsToKG.OntologyDownload"] = _original_ontology


def _make_work_item(doc_id: str, output_path: Path, input_hash: str) -> WorkItem:
    return WorkItem(
        item_id=doc_id,
        inputs={},
        outputs={"output": output_path},
        cfg_hash="cfg",
        metadata={
            "input_hash": input_hash,
            "output_path": str(output_path),
        },
    )


def test_should_skip_when_manifest_matches(tmp_path) -> None:
    output_path = tmp_path / "doc.jsonl"
    output_path.write_text("ok", encoding="utf-8")
    manifest_index = {
        "doc-1": {
            "status": "success",
            "input_hash": "abc",
            "output_path": str(output_path),
        }
    }
    controller = ResumeController(True, False, manifest_index)
    options = StageOptions(resume=True, force=False, resume_controller=controller)
    item = _make_work_item("doc-1", output_path, "abc")

    assert _should_skip(item, options) is True


def test_should_not_skip_when_hash_differs(tmp_path) -> None:
    output_path = tmp_path / "doc.jsonl"
    output_path.write_text("ok", encoding="utf-8")
    manifest_index = {
        "doc-1": {
            "status": "success",
            "input_hash": "expected",
            "output_path": str(output_path),
        }
    }
    controller = ResumeController(True, False, manifest_index)
    options = StageOptions(resume=True, force=False, resume_controller=controller)
    item = _make_work_item("doc-1", output_path, "different")

    assert _should_skip(item, options) is False


def test_should_not_skip_without_controller(tmp_path) -> None:
    output_path = tmp_path / "doc.jsonl"
    output_path.write_text("ok", encoding="utf-8")
    manifest_index = {
        "doc-1": {
            "status": "success",
            "input_hash": "abc",
            "output_path": str(output_path),
        }
    }
    controller = ResumeController(True, False, manifest_index)
    item = _make_work_item("doc-1", output_path, "abc")

    options_without_controller = StageOptions(resume=True, force=False, resume_controller=None)
    assert _should_skip(item, options_without_controller) is False

    forced_options = StageOptions(resume=True, force=True, resume_controller=controller)
    assert _should_skip(item, forced_options) is False
