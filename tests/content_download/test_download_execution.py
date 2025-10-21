"""End-to-end exercises for download execution primitives.

The tests simulate robots enforcement, range-resume, cached payload reuse,
streaming failures, and manifest bookkeeping. They cover both the low-level
streaming helpers (`stream_candidate_payload`, `finalize_candidate_download`)
and orchestration utilities (`download_candidate`, `process_one_work`) to ensure
partial files, digests, and HTML sidecars are managed correctly under success
and error conditions.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx
import pytest

from DocsToKG.ContentDownload import download as downloader
from DocsToKG.ContentDownload.core import Classification, ReasonCode, WorkArtifact
from DocsToKG.ContentDownload.download import (
    DownloadConfig,
    DownloadOutcome,
    finalize_candidate_download,
    prepare_candidate_download,
    stream_candidate_payload,
)
from DocsToKG.ContentDownload.networking import CachedResult, ConditionalRequestHelper
from DocsToKG.ContentDownload.api.types import DownloadOutcome
from tests.conftest import PatchManager


def _build_mock_client() -> httpx.Client:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    return httpx.Client(transport=transport)


def _build_response(
    content: bytes,
    *,
    status: int = 200,
    content_type: str = "application/pdf",
) -> httpx.Response:
    headers = {"Content-Type": content_type}
    request = httpx.Request("GET", "https://example.com/foo.pdf")
    return httpx.Response(status, headers=headers, content=content, request=request)


download_impl = downloader


class _ResponseContext:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def __enter__(self) -> httpx.Response:
        return self._response

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._response.close()
        return False


@pytest.fixture
def artifact(tmp_path: Path) -> WorkArtifact:
    art = WorkArtifact(
        work_id="W1",
        title="Example",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.com/foo.pdf"],
        open_access_url=None,
        source_display_names=[],
        base_stem="example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
    )
    art.pdf_dir.mkdir(parents=True, exist_ok=True)
    art.html_dir.mkdir(parents=True, exist_ok=True)
    art.xml_dir.mkdir(parents=True, exist_ok=True)
    return art


class _DenyRobots:
    def is_allowed(self, client: httpx.Client, url: str, timeout: float) -> bool:
        return False


def test_prepare_candidate_download_blocks_robots(artifact: WorkArtifact) -> None:
    config = DownloadConfig(run_id="run", robots_checker=_DenyRobots())
    ctx = config.to_context({})
    client = _build_mock_client()
    try:
        plan = prepare_candidate_download(
            client,
            artifact,
            "https://example.com/foo.pdf",
            None,
            5.0,
            ctx,
        )
    finally:
        client.close()
    assert plan.skip_outcome is not None
    assert plan.skip_outcome.reason is ReasonCode.ROBOTS_DISALLOWED


def test_prepare_candidate_download_skip_head_precheck(
    patcher: PatchManager, artifact: WorkArtifact
) -> None:
    calls: List[str] = []

    def fake_head(
        client: httpx.Client,
        url: str,
        timeout: float,
        *,
        content_policy: Optional[Dict[str, Any]] = None,
    ) -> bool:
        calls.append(url)
        return True

    patcher.setattr(downloader, "head_precheck", fake_head)
    config = DownloadConfig(run_id="run", skip_head_precheck=True)
    ctx = config.to_context({})
    client = _build_mock_client()
    try:
        plan = prepare_candidate_download(
            client,
            artifact,
            "https://example.com/foo.pdf",
            None,
            5.0,
            ctx,
        )
    finally:
        client.close()
    assert not calls
    assert not plan.head_precheck_passed
def test_stream_candidate_payload_returns_cached(
    patcher: PatchManager, artifact: WorkArtifact, tmp_path: Path
) -> None:
    config = DownloadConfig(run_id="run")
    ctx = config.to_context({})
    client = _build_mock_client()
    try:
        plan = prepare_candidate_download(
            client,
            artifact,
            "https://example.com/foo.pdf",
            None,
            5.0,
            ctx,
        )
    finally:
        client.close()

    cached_bytes = b"cacheddata"
    cached = CachedResult(
        path=str(tmp_path / "cached.pdf"),
        sha256=hashlib.sha256(cached_bytes).hexdigest(),
        content_length=len(cached_bytes),
        etag="etag",
        last_modified="yesterday",
    )
    Path(cached.path).write_bytes(cached_bytes)

    class StubHelper(ConditionalRequestHelper):
        def build_headers(self) -> Dict[str, str]:
            return {}

        def interpret_response(self, response: httpx.Response) -> CachedResult:
            return cached

    plan.cond_helper = StubHelper()

    patcher.setattr(
        downloader,
        "request_with_retries",
        lambda *args, **kwargs: _ResponseContext(_build_response(b"", status=304)),
    )
    result = stream_candidate_payload(plan)
    assert result.outcome is not None
    assert result.outcome.classification is Classification.CACHED
    assert result.outcome.reason is ReasonCode.CONDITIONAL_NOT_MODIFIED
    assert result.outcome.reason_detail == "not-modified"


def test_stream_candidate_payload_streams_pdf(
    patcher: PatchManager, artifact: WorkArtifact
) -> None:
    progress: List[int] = []

    config = DownloadConfig(
        run_id="run",
        progress_callback=lambda transferred, total, url: progress.append(transferred),
        skip_head_precheck=True,
        min_pdf_bytes=16,
    )
    ctx = config.to_context({})
    client = _build_mock_client()
    try:
        plan = prepare_candidate_download(
            client,
            artifact,
            "https://example.com/foo.pdf",
            None,
            5.0,
            ctx,
        )
    finally:
        client.close()

    pdf_bytes = b"%PDF-1.4\n" + (b"x" * 32) + b"\n%%EOF\n"

    patcher.setattr(
        downloader,
        "request_with_retries",
        lambda *args, **kwargs: _ResponseContext(_build_response(pdf_bytes)),
    )
    result = stream_candidate_payload(plan)
    assert result.outcome is None
    assert result.strategy is not None
    outcome = finalize_candidate_download(plan, result)
    assert outcome.classification is Classification.PDF
    assert outcome.path is not None
    assert Path(outcome.path).exists()
    assert progress


def test_download_candidate_retries_and_cleans_partial(
    patcher: PatchManager,
    artifact: WorkArtifact,
) -> None:
    pdf_bytes = b"%PDF-1.4\n" + (b"y" * 64) + b"\n%%EOF\n"

    request_calls: List[Tuple[str, bool]] = []

    def _request_with_retries(client, method, url, **kwargs):
        request_calls.append((method.upper(), bool(kwargs.get("stream"))))
        return _ResponseContext(_build_response(pdf_bytes))

    patcher.setattr(downloader, "request_with_retries", _request_with_retries, raising=False)
    patcher.setattr(download_impl, "request_with_retries", _request_with_retries)

    call_state = {"count": 0}

    def flaky_atomic_write(
        dest_path: Path,
        chunks: Iterator[bytes],
        *,
        hasher: Optional[Any] = None,
        keep_partial_on_error: bool = False,
    ) -> None:
        call_state["count"] += 1
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")
        data = b"".join(chunks)
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(data)
        if call_state["count"] == 1:
            raise httpx.ReadError(
                "stream error",
                request=httpx.Request("GET", "https://example.com/foo.pdf"),
            )
        dest_path.write_bytes(data)
        part_path.unlink(missing_ok=True)

    patcher.setattr(downloader, "atomic_write", flaky_atomic_write)

    progress: List[int] = []
    config = DownloadConfig(
        run_id="run",
        progress_callback=lambda transferred, total, url: progress.append(transferred),
        skip_head_precheck=True,
        min_pdf_bytes=16,
    )
    context = config.to_context({})
    client = _build_mock_client()
    try:
        outcome = downloader.download_candidate(
            client,
            artifact,
            "https://example.com/foo.pdf",
            None,
            5.0,
            context,
        )
    finally:
        client.close()
    assert outcome.classification is Classification.PDF
    assert Path(outcome.path or "").exists()
    assert not Path(str(outcome.path) + ".part").exists()
    assert progress
    assert request_calls == [("GET", True), ("GET", True)]
def test_cleanup_sidecar_files_removes_partial_pdf(
    tmp_path: Path, artifact: WorkArtifact
) -> None:
    target = artifact.pdf_dir / "example.pdf"
    part_path = target.with_suffix(".pdf.part")
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"data")
    part_path.write_bytes(b"temp")

    config = DownloadConfig(run_id="run")
    downloader.cleanup_sidecar_files(
        artifact,
        Classification.PDF,
        config,
    )
    assert not part_path.exists()
    assert not target.exists()


def test_cleanup_sidecar_files_removes_html(tmp_path: Path, artifact: WorkArtifact) -> None:
    target = artifact.html_dir / "example.html"
    part_path = target.with_suffix(".html.part")
    artifact.html_dir.mkdir(parents=True, exist_ok=True)
    target.write_text("<html></html>")
    part_path.write_text("partial")
    downloader.cleanup_sidecar_files(
        artifact,
        Classification.HTML,
        DownloadConfig(run_id="run"),
    )
    assert not part_path.exists()
    assert not target.exists()


def test_download_config_round_trip() -> None:
    progress_calls: List[int] = []
    config = DownloadConfig(
        run_id="run",
        domain_content_rules={"example.com": {"allow": ["pdf"]}},
        host_accept_overrides={"example.com": "application/pdf"},
        progress_callback=lambda transferred, total, url: progress_calls.append(transferred),
        skip_head_precheck=True,
        head_precheck_passed=True,
    )
    ctx = config.to_context({"previous": {}})
    assert ctx.domain_content_rules == config.domain_content_rules
    assert ctx.host_accept_overrides == config.host_accept_overrides
    assert ctx.skip_head_precheck
    assert ctx.progress_callback is config.progress_callback

    round_trip = DownloadConfig.from_options(ctx)
    assert round_trip.host_accept_overrides == config.host_accept_overrides
    legacy = DownloadConfig.from_options({"run_id": "legacy", "max_bytes": 123})
    assert "max_bytes" not in legacy.extra


def test_build_download_outcome_html_tail_reason(tmp_path: Path, artifact: WorkArtifact) -> None:
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n" + (b"z" * 32))
    tail = b"<html>"
    response = _build_response(b"", content_type="application/pdf")
    outcome = downloader.build_download_outcome(
        artifact=artifact,
        classification=Classification.PDF,
        dest_path=pdf_path,
        response=response,
        elapsed_ms=10.0,
        flagged_unknown=False,
        sha256="abc",
        content_length=32,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        dry_run=False,
        tail_bytes=tail,
        head_precheck_passed=True,
        min_pdf_bytes=8,
        tail_check_bytes=2048,
        retry_after=None,
        options=DownloadConfig(run_id="run"),
    )
    assert outcome.classification is Classification.MISS
    assert outcome.reason is ReasonCode.HTML_TAIL_DETECTED
    assert outcome.reason_detail == "html-tail-detected"


def test_process_one_work_preserves_reason(patcher: PatchManager, artifact: WorkArtifact) -> None:
    class StubLogger:
        def __init__(self) -> None:
            self.manifest_reason: Optional[Any] = None
            self.manifest_detail: Optional[Any] = None

        def record_pipeline_result(self, *args: Any, **kwargs: Any) -> None:
            return None

        def record_manifest(
            self,
            artifact: WorkArtifact,
            *,
            resolver: Optional[str],
            url: Optional[str],
            outcome: DownloadOutcome,
            html_paths: List[str],
            dry_run: bool,
            run_id: str,
            reason: Optional[Any],
            reason_detail: Optional[Any],
        ) -> None:
            self.manifest_reason = reason
            self.manifest_detail = reason_detail

        def log_attempt(self, *args: Any, **kwargs: Any) -> None:
            return None

    outcome = DownloadOutcome(
        classification=Classification.MISS,
        path=None,
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=12.0,
        reason=ReasonCode.HTML_TAIL_DETECTED,
        reason_detail="html-tail-detected",
    )
    pipeline_result = DownloadOutcome(
        classification=Classification.MISS,
        path=None,
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=12.0,
        reason=ReasonCode.HTML_TAIL_DETECTED,
        reason_detail="html-tail-detected",
    )

    class StubPipeline:
        def run(self, *args: Any, **kwargs: Any) -> DownloadOutcome:
            return pipeline_result

    logger = StubLogger()
    metrics = ResolverMetrics()
    config = DownloadConfig(run_id="run")
    client = _build_mock_client()
    try:
        result = downloader.process_one_work(
            artifact,
            client,
            artifact.pdf_dir,
            artifact.html_dir,
            artifact.xml_dir,
            StubPipeline(),
            logger,
            metrics,
            options=config,
        )
    finally:
        client.close()
    assert not result["saved"]
    assert logger.manifest_reason == ReasonCode.HTML_TAIL_DETECTED
    assert logger.manifest_detail == "html_tail_detected"
