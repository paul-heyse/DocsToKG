"""Regression tests for resolver download preflight behaviour.

These tests focus on the edge case where `prepare_candidate_download` must honour
robots.txt decisions before attempting HEAD preflights. They ensure that a disallowed
URL short-circuits the pipeline without touching the network or mutating download
context, preventing wasted requests and preserving accurate skip reasons.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from DocsToKG.ContentDownload import download as downloader
from DocsToKG.ContentDownload.core import ReasonCode, WorkArtifact
from DocsToKG.ContentDownload.download import DownloadConfig, prepare_candidate_download
from tests.conftest import PatchManager

requests = downloader.requests


def _build_artifact(tmp_path: Path) -> WorkArtifact:
    artifact = WorkArtifact(
        work_id="W-robots",
        title="Robots Skip",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.com/blocked.pdf"],
        open_access_url=None,
        source_display_names=[],
        base_stem="robots-skip",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
    )
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
    artifact.html_dir.mkdir(parents=True, exist_ok=True)
    artifact.xml_dir.mkdir(parents=True, exist_ok=True)
    return artifact


def test_head_precheck_not_called_when_robots_disallow(
    patcher: PatchManager, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_head_precheck(*args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        calls.append("called")
        return True

    patcher.setattr(downloader, "head_precheck", fake_head_precheck)

    robots_checker = mock.Mock(spec=downloader.RobotsCache)
    robots_checker.is_allowed.return_value = False

    config = DownloadConfig(run_id="run", robots_checker=robots_checker)
    ctx = config.to_context({})

    artifact = _build_artifact(tmp_path)

    plan = prepare_candidate_download(
        requests.Session(),
        artifact,
        "https://example.com/blocked.pdf",
        None,
        5.0,
        ctx,
    )

    assert plan.skip_outcome is not None
    assert plan.skip_outcome.reason is ReasonCode.ROBOTS_DISALLOWED
    assert calls == []
    robots_checker.is_allowed.assert_called_once()
