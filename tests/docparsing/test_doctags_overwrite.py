"""Verify DocTags overwrite semantics and resume short-circuiting.

When operators re-run DocTags with overwrite flags, the pipeline should bypass
costly hashing and manifest lookups. These tests build minimal HTML fixtures,
simulate manifest state, and ensure the orchestrator only recomputes work when
necessary while still emitting the correct telemetry events.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from DocsToKG.DocParsing.doctags import HtmlConversionResult, html_main
from tests.conftest import PatchManager


def test_html_main_resume_overwrite_skips_hash(patcher: PatchManager, tmp_path: Path) -> None:
    """HTML execution path skips hashing when overwrite short-circuits resume."""

    html_dir = tmp_path / "html"
    output_dir = tmp_path / "doctags"
    html_dir.mkdir()
    output_dir.mkdir()

    html_path = html_dir / "doc.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    manifest_index = {"doc.html": {"input_hash": "cached", "status": "success"}}
    events: list[tuple[str, dict[str, object]]] = []

    def _raise_hash(_path: Path) -> str:
        raise AssertionError("compute_content_hash should not run when overwrite disables resume")

    @contextmanager
    def _noop_scope(_telemetry: object) -> None:
        yield

    class _DummyFuture:
        def __init__(self, payload: HtmlConversionResult) -> None:
            self._payload = payload

        def result(self) -> HtmlConversionResult:
            return self._payload

    class _DummyExecutor:
        def __init__(self, *_args, **_kwargs) -> None:
            self.tasks: list[object] = []

        def __enter__(self) -> "_DummyExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def submit(self, _fn, task) -> _DummyFuture:
            self.tasks.append(task)
            result = HtmlConversionResult(
                doc_id=task.relative_id,
                status="success",
                duration_s=0.01,
                input_path=str(task.html_path),
                input_hash=task.input_hash,
                output_path=str(task.output_path),
                sanitizer_profile=task.sanitizer_profile,
            )
            return _DummyFuture(result)

    def _finalize_cfg(cfg: object) -> None:
        cfg.data_root = tmp_path
        cfg.input = html_dir
        cfg.output = output_dir
        workers = getattr(cfg, "workers", 1)
        cfg.workers = 1 if not isinstance(workers, int) or workers < 1 else workers
        cfg.mode = "html"
        cfg.html_sanitizer = "balanced"
        cfg.http_timeout = (5.0, 30.0)

    patcher.setenv("DOCSTOKG_DOCTAGS_HTML_SANITIZER", "balanced")
    patcher.setattr("DocsToKG.DocParsing.doctags.compute_content_hash", _raise_hash)
    patcher.setattr(
        "DocsToKG.DocParsing.doctags.load_manifest_index",
        lambda *_args, **_kwargs: manifest_index,
    )
    patcher.setattr("DocsToKG.DocParsing.doctags.list_htmls", lambda _path: [html_path])
    patcher.setattr("DocsToKG.DocParsing.doctags.detect_data_root", lambda: tmp_path)
    patcher.setattr("DocsToKG.DocParsing.doctags.data_html", lambda _root: None)
    patcher.setattr("DocsToKG.DocParsing.doctags.data_doctags", lambda _root: None)
    patcher.setattr("DocsToKG.DocParsing.doctags.telemetry_scope", _noop_scope)
    patcher.setattr(
        "DocsToKG.DocParsing.doctags.StageTelemetry", lambda *_args, **_kwargs: object()
    )
    patcher.setattr("DocsToKG.DocParsing.doctags.TelemetrySink", lambda *_args, **_kwargs: object())
    patcher.setattr(
        "DocsToKG.DocParsing.doctags.manifest_log_success",
        lambda *args, **kwargs: events.append(("success", kwargs)),
    )
    patcher.setattr(
        "DocsToKG.DocParsing.doctags.manifest_log_skip",
        lambda *args, **kwargs: events.append(("skip", kwargs)),
    )
    patcher.setattr("DocsToKG.DocParsing.doctags.DoctagsCfg.finalize", _finalize_cfg)
    patcher.setattr("DocsToKG.DocParsing.doctags.ProcessPoolExecutor", _DummyExecutor)
    patcher.setattr("DocsToKG.DocParsing.doctags.as_completed", lambda futures: futures)
    patcher.setattr("DocsToKG.DocParsing.doctags.tqdm", lambda iterable, **_kwargs: iterable)

    exit_code = html_main(
        [
            "--input",
            str(html_dir),
            "--output",
            str(output_dir),
            "--workers",
            "1",
            "--resume",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert any(event[0] == "success" for event in events)
