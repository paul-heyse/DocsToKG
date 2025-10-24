from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_PKG_ROOT = SRC_ROOT / "DocsToKG"
_DOCSTOKG = types.ModuleType("DocsToKG")
_DOCSTOKG.__path__ = [str(_PKG_ROOT)]
sys.modules.setdefault("DocsToKG", _DOCSTOKG)

_DOCPARSING = types.ModuleType("DocsToKG.DocParsing")
_DOCPARSING.__path__ = [str(_PKG_ROOT / "DocParsing")]
sys.modules.setdefault("DocsToKG.DocParsing", _DOCPARSING)

_CORE = types.ModuleType("DocsToKG.DocParsing.core")
_CORE.__path__ = [str(_PKG_ROOT / "DocParsing" / "core")]
sys.modules.setdefault("DocsToKG.DocParsing.core", _CORE)

_MANIFEST_MODULE_PATH = Path(_CORE.__path__[0]) / "manifest.py"
_MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "DocsToKG.DocParsing.core.manifest",
    _MANIFEST_MODULE_PATH,
)
_MANIFEST_MODULE = importlib.util.module_from_spec(_MANIFEST_SPEC)
sys.modules.setdefault("DocsToKG.DocParsing.core.manifest", _MANIFEST_MODULE)
assert _MANIFEST_SPEC and _MANIFEST_SPEC.loader
_MANIFEST_SPEC.loader.exec_module(_MANIFEST_MODULE)

ResumeController = _MANIFEST_MODULE.ResumeController
should_skip_output = _MANIFEST_MODULE.should_skip_output


def _make_manifest_entry(output_path: Path, status: str = "success") -> dict[str, object]:
    return {
        "stage": "chunk",
        "doc_id": "doc-1",
        "status": status,
        "duration_s": 0.1,
        "input_path": str(output_path),
        "output_path": str(output_path),
        "schema_version": "v1",
        "input_hash": "hash-1",
    }


def test_can_skip_without_hash_accepts_symlinked_output(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_output = real_dir / "doc.jsonl"
    real_output.write_text("payload", encoding="utf-8")

    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    symlink_output = link_dir / real_output.name

    entry = _make_manifest_entry(real_output)
    manifest_index = {"doc-1": entry}
    controller = ResumeController(resume=True, force=False, manifest_index=manifest_index)

    can_skip, returned_entry = controller.can_skip_without_hash("doc-1", symlink_output)

    assert can_skip is True
    assert returned_entry is entry


def test_should_skip_output_uses_normalized_manifest_path(tmp_path: Path) -> None:
    real_output = tmp_path / "real.jsonl"
    real_output.write_text("payload", encoding="utf-8")

    symlink_output = tmp_path / "alias.jsonl"
    symlink_output.symlink_to(real_output)

    manifest_entry = _make_manifest_entry(symlink_output)

    should_skip = should_skip_output(
        real_output,
        manifest_entry,
        input_hash="hash-1",
        resume=True,
        force=False,
    )

    assert should_skip is True
