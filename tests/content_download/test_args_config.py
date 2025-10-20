"""Tests covering configuration resolution purity helpers."""

import logging
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Mapping

import pytest

from DocsToKG.ContentDownload import args as args_module, httpx_transport
from DocsToKG.ContentDownload.args import (
    ResolvedConfig,
    bootstrap_run_environment,
    build_parser,
    parse_args,
    resolve_config,
)
from DocsToKG.ContentDownload.core import Classification
from DocsToKG.ContentDownload.pipeline import ResolverConfig, ResolverPipeline
from DocsToKG.ContentDownload.ratelimit import (
    BackendConfig,
    clone_policies,
    get_rate_limiter_manager,
    RolePolicy,
)
from DocsToKG.ContentDownload.runner import DownloadRun
from DocsToKG.ContentDownload.urls import get_url_policy, reset_url_policy_for_tests
from pyrate_limiter import Duration
from tests.conftest import PatchManager


@pytest.fixture(autouse=True)
def _reset_url_policy() -> None:
    reset_url_policy_for_tests()
    yield
    reset_url_policy_for_tests()


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


def _clone_backend_config() -> BackendConfig:
    manager = get_rate_limiter_manager()
    options = manager.backend.options
    if isinstance(options, dict):
        cloned_options = dict(options)
    else:
        cloned_options = dict(options or {})
    return BackendConfig(backend=manager.backend.backend, options=cloned_options)


def _restore_manager_state(
    manager, policies: Mapping[str, RolePolicy], backend: BackendConfig
) -> None:
    manager.configure_backend(backend)
    manager.configure_policies(clone_policies(policies))
    manager.reset_metrics()


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


@pytest.mark.parametrize(
    ("manifest_arg", "csv_arg"),
    [
        ("manifest.jsonl", "attempts.csv"),
        ("~/logs/manifest.jsonl", "~/logs/attempts.csv"),
    ],
)
def test_resolve_config_expands_csv_path_like_manifest(
    tmp_path: Path, patcher: PatchManager, manifest_arg: str, csv_arg: str
) -> None:
    home_dir = tmp_path / "home"
    work_dir = tmp_path / "work"
    home_dir.mkdir()
    work_dir.mkdir()
    patcher.setenv("HOME", str(home_dir))
    patcher.chdir(work_dir)

    if manifest_arg.startswith("~/") or csv_arg.startswith("~/"):
        (home_dir / "logs").mkdir(exist_ok=True)

    parser = build_parser()
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(work_dir / "pdfs"),
        "--manifest",
        manifest_arg,
        "--log-format",
        "csv",
        "--log-csv",
        csv_arg,
    ]
    args = parse_args(parser, argv)

    resolved = resolve_config(args, parser)

    expected_manifest = Path(manifest_arg).expanduser().resolve(strict=False)
    expected_csv = Path(csv_arg).expanduser().resolve(strict=False)

    assert resolved.manifest_path == expected_manifest
    assert resolved.csv_path == expected_csv


def test_resolve_config_skips_manifest_index_when_dedup_disabled(tmp_path: Path, patcher) -> None:
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

    patcher.setattr(
        "DocsToKG.ContentDownload.args.ManifestUrlIndex.iter_existing_paths",
        _unexpected_iter,
    )

    resolved = resolve_config(args, parser)

    assert resolved.resolver_config.enable_global_url_dedup is False
    assert resolved.persistent_seen_urls == set()


def test_resolver_pipeline_receives_seen_urls_when_dedup_enabled(tmp_path: Path, patcher) -> None:
    parser = build_parser()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.touch()
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(tmp_path / "pdfs"),
        "--manifest",
        str(manifest_path),
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

    patcher.setattr(
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
    assert pipeline._global_seen_urls == resolved.persistent_seen_urls


def test_resolve_config_truncates_persistent_seen_urls_at_cap(
    tmp_path: Path, patcher, caplog
) -> None:
    parser = build_parser()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.touch()
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(tmp_path / "pdfs"),
        "--manifest",
        str(manifest_path),
        "--global-url-dedup-cap",
        "2",
    ]
    args = parse_args(parser, argv)

    manifest_entries = [
        ("https://example.org/1", {"classification": Classification.PDF.value}),
        ("https://example.org/2", {"classification": Classification.PDF.value}),
        ("https://example.org/3", {"classification": Classification.PDF.value}),
    ]

    def _iter_existing_paths(*_args, **_kwargs):
        yield from manifest_entries

    patcher.setattr(
        "DocsToKG.ContentDownload.args.ManifestUrlIndex.iter_existing_paths",
        _iter_existing_paths,
    )

    caplog.set_level(logging.INFO)
    caplog.clear()

    resolved = resolve_config(args, parser)

    assert resolved.persistent_seen_urls == {
        "https://example.org/1",
        "https://example.org/2",
    }
    assert any(
        "truncated to configured cap" in record.message for record in caplog.records
    )


def test_lookup_topic_id_requests_minimal_payload(patcher, caplog) -> None:
    args_module._lookup_topic_id.cache_clear()

    class FakeTopics:
        def __init__(self) -> None:
            self.calls = []

        def search(self, text: str):
            self.calls.append(("search", text))
            return self

        def select(self, fields):
            self.calls.append(("select", tuple(fields)))
            return self

        def per_page(self, value: int):
            self.calls.append(("per_page", value))
            return self

        def get(self):
            self.calls.append(("get",))
            return [{"id": "https://openalex.org/T999"}]

    fake = FakeTopics()
    patcher.setattr(args_module, "Topics", lambda: fake)
    caplog.set_level(logging.INFO)
    caplog.clear()

    resolved = args_module._lookup_topic_id("quantum photonics")

    assert resolved == "https://openalex.org/T999"
    assert fake.calls == [
        ("search", "quantum photonics"),
        ("select", ("id",)),
        ("per_page", 1),
        ("get",),
    ]
    assert (
        "DocsToKG.ContentDownload",
        logging.INFO,
        "Resolved topic 'quantum photonics' -> https://openalex.org/T999",
    ) in caplog.record_tuples


def test_resolve_config_applies_url_policy_overrides(tmp_path: Path) -> None:
    parser = build_parser()
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2021",
        "--year-end",
        "2021",
        "--out",
        str(tmp_path / "pdfs"),
        "--url-default-scheme",
        "http",
        "--url-param-allowlist",
        "example.com:id,token;page",
        "--no-url-filter-landing",
    ]
    args = parse_args(parser, argv)

    resolve_config(args, parser)

    policy = get_url_policy()
    assert policy.default_scheme == "http"
    assert policy.filter_for["landing"] is False
    assert policy.param_allowlist_global == ("page",)
    assert policy.param_allowlist_per_domain["example.com"] == ("id", "token")


def test_lookup_topic_id_handles_empty_results(patcher, caplog) -> None:
    args_module._lookup_topic_id.cache_clear()

    class EmptyTopics:
        def __init__(self) -> None:
            self.calls = []

        def search(self, text: str):
            self.calls.append(("search", text))
            return self

        def select(self, fields):
            self.calls.append(("select", tuple(fields)))
            return self

        def per_page(self, value: int):
            self.calls.append(("per_page", value))
            return self

        def get(self):
            self.calls.append(("get",))
            return []

    fake = EmptyTopics()
    patcher.setattr(args_module, "Topics", lambda: fake)
    caplog.set_level(logging.INFO)
    caplog.clear()

    resolved = args_module._lookup_topic_id("obscure field")

    assert resolved is None
    assert fake.calls == [
        ("search", "obscure field"),
        ("select", ("id",)),
        ("per_page", 1),
        ("get",),
    ]
    assert not any(
        record[0] == "DocsToKG.ContentDownload" and record[1] == logging.INFO
        for record in caplog.record_tuples
    )


def test_lookup_topic_id_handles_request_exception(patcher, caplog) -> None:
    args_module._lookup_topic_id.cache_clear()

    class ErrorTopics:
        def search(self, text: str):
            raise requests.RequestException("boom")

    patcher.setattr(args_module, "Topics", lambda: ErrorTopics())
    caplog.set_level(logging.WARNING)
    caplog.clear()

    resolved = args_module._lookup_topic_id("failing topic")

    assert resolved is None
    assert any(
        record[0] == "DocsToKG.ContentDownload" and record[1] == logging.WARNING
        for record in caplog.record_tuples
    )
    assert any("Topic lookup failed" in record[2] for record in caplog.record_tuples)


def test_rate_disable_flag_disables_limiter(tmp_path: Path, caplog) -> None:
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
        "--rate-disable",
    ]
    args = parse_args(parser, argv)

    manager = get_rate_limiter_manager()
    original_policies = clone_policies(manager.policies())
    original_backend = _clone_backend_config()

    try:
        caplog.set_level(logging.INFO)
        resolved = resolve_config(args, parser)

        assert resolved.rate_policies == {}
        assert resolved.rate_backend.backend == "disabled"
        assert manager.policies() == {}
        assert manager.backend.backend == "disabled"
        assert any(
            "Centralized rate limiter disabled" in record.message for record in caplog.records
        )
    finally:
        _restore_manager_state(manager, original_policies, original_backend)


def test_legacy_domain_flags_rejected(tmp_path: Path, capsys) -> None:
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
        "--domain-token-bucket",
        "example.org=0.5:capacity=3",
    ]

    with pytest.raises(SystemExit):
        parse_args(parser, argv)

    _, err = capsys.readouterr()
    assert "--domain-token-bucket" in err


def test_legacy_domain_min_interval_flag_rejected(tmp_path: Path, capsys) -> None:
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
        "--domain-min-interval",
        "example.org=1.5",
    ]

    with pytest.raises(SystemExit):
        parse_args(parser, argv)

    _, err = capsys.readouterr()
    assert "--domain-min-interval" in err

def test_rate_disable_conflicts_with_overrides(tmp_path: Path) -> None:
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
        "--rate-disable",
        "--rate",
        "example.org=1/s",
    ]
    args = parse_args(parser, argv)
    with pytest.raises(SystemExit):
        resolve_config(args, parser)


def test_cli_rate_overrides_update_policies_and_logging(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    parser = build_parser()
    rate_db = tmp_path / "limits" / "policy.sqlite"
    rate_db.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "--topic-id",
        "https://openalex.org/T12345",
        "--year-start",
        "2020",
        "--year-end",
        "2020",
        "--out",
        str(tmp_path / "pdfs"),
        "--rate",
        "example.org=3/s,180/h",
        "--rate-mode",
        "example.org.artifact=wait:750",
        "--rate-backend",
        f"sqlite:path={rate_db},use_file_lock=true",
    ]
    args = parse_args(parser, argv)

    manager = get_rate_limiter_manager()
    original_policies = clone_policies(manager.policies())
    original_backend = _clone_backend_config()

    try:
        resolved = resolve_config(args, parser)

        policy = resolved.rate_policies["example.org"]
        metadata_rates = [
            (rate.limit, int(rate.interval)) for rate in policy.rates["metadata"]
        ]
        artifact_delay = policy.max_delay_ms["artifact"]
        assert metadata_rates == [
            (3, Duration.SECOND),
            (180, Duration.HOUR),
        ]
        assert artifact_delay == 750
        assert resolved.rate_backend.backend == "sqlite"
        assert resolved.rate_backend.options["path"] == str(rate_db)

        bootstrap_run_environment(resolved)
        caplog.set_level("INFO", logger="DocsToKG.ContentDownload")

        def _abort_setup(self, stack):
            raise RuntimeError("stop-run")

        monkeypatch.setattr(DownloadRun, "setup_sinks", _abort_setup)

        run = DownloadRun(resolved)
        with pytest.raises(RuntimeError, match="stop-run"):
            run.run()

        messages = [
            record.message
            for record in caplog.records
            if record.name == "DocsToKG.ContentDownload"
        ]
        assert any(
            "Rate limiter configured with backend=sqlite" in message for message in messages
        ), messages
    finally:
        _restore_manager_state(manager, original_policies, original_backend)
