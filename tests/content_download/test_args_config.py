"""Tests covering configuration resolution purity helpers."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.args import (
    ResolvedConfig,
    bootstrap_run_environment,
    build_parser,
    parse_args,
    resolve_config,
)
from DocsToKG.ContentDownload.core import Classification
from DocsToKG.ContentDownload.pipeline import ResolverPipeline


def _build_args(tmp_path: Path):
    parser = build_parser()
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(tmp_path / "pdfs"),
    ]
    args = parse_args(parser, argv)
    return parser, args


def test_resolve_config_is_pure(tmp_path: Path) -> None:
    parser, args = _build_args(tmp_path)
    snapshot = {key: getattr(args, key) for key in vars(args)}

    resolved = resolve_config(args, parser)

    assert isinstance(resolved, ResolvedConfig)
    assert not resolved.pdf_dir.exists()
    assert not resolved.html_dir.exists()
    assert not resolved.xml_dir.exists()

    current = {key: getattr(args, key) for key in vars(args)}
    assert current == snapshot
    assert not hasattr(args, "extract_html_text")
    assert resolved.extract_html_text is False

    with pytest.raises(FrozenInstanceError):
        resolved.run_id = "override"  # type: ignore[misc]


def test_bootstrap_run_environment_creates_directories(tmp_path: Path) -> None:
    parser, args = _build_args(tmp_path)
    resolved = resolve_config(args, parser)

    assert not resolved.pdf_dir.exists()
    assert not resolved.html_dir.exists()
    assert not resolved.xml_dir.exists()

    bootstrap_run_environment(resolved)

    assert resolved.pdf_dir.exists()
    assert resolved.html_dir.exists()
    assert resolved.xml_dir.exists()

    # Idempotent behaviour should not raise when repeated.
    bootstrap_run_environment(resolved)


def test_resolve_config_skips_manifest_index_when_dedup_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    parser = build_parser()
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(tmp_path / "pdfs"),
        "--no-global-url-dedup",
    ]
    args = parse_args(parser, argv)

    def _unexpected_iter(*_args, **_kwargs):
        raise AssertionError("ManifestUrlIndex.iter_existing_paths should not be called")

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.args.ManifestUrlIndex.iter_existing_paths",
        _unexpected_iter,
    )

    resolved = resolve_config(args, parser)

    assert resolved.resolver_config.enable_global_url_dedup is False
    assert resolved.persistent_seen_urls == set()


def test_resolver_pipeline_receives_seen_urls_when_dedup_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    parser = build_parser()
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(tmp_path / "pdfs"),
    ]
    args = parse_args(parser, argv)

    manifest_entries = [
        ("https://example.org/pdf", {"classification": Classification.PDF.value}),
        ("https://example.org/cached", {"classification": Classification.CACHED.value}),
        ("https://example.org/html", {"classification": Classification.HTML.value}),
        ("https://example.org/xml", {"classification": Classification.XML.value}),
    ]

    def _iter_existing_paths(*_args, **_kwargs):
        for entry in manifest_entries:
            yield entry

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.args.ManifestUrlIndex.iter_existing_paths",
        _iter_existing_paths,
    )

    resolved = resolve_config(args, parser)

    expected_urls = {
        "https://example.org/pdf",
        "https://example.org/cached",
        "https://example.org/xml",
    }
    assert resolved.resolver_config.enable_global_url_dedup is True
    assert resolved.persistent_seen_urls == expected_urls

    class _Logger:
        def log_attempt(self, *_, **__):
            return None

    pipeline = ResolverPipeline(
        resolvers=resolved.resolver_instances,
        config=resolved.resolver_config,
        download_func=lambda *_, **__: None,
        logger=_Logger(),
        initial_seen_urls=resolved.persistent_seen_urls,
    )

    assert pipeline._global_seen_urls == expected_urls
