"""Focused coverage for CLI argument validation helpers."""

from __future__ import annotations

import argparse
import types
import typing
from types import SimpleNamespace

import pytest

if not hasattr(typing, "TracebackType"):  # pragma: no cover - python<3.13 shim
    typing.TracebackType = types.TracebackType

from DocsToKG.ContentDownload import args as download_args


class _StubWorks:
    """Minimal Works stub implementing the chainable API used by build_query."""

    def filter(self, **_: object) -> "_StubWorks":
        return self

    def search(self, _: object) -> "_StubWorks":
        return self

    def select(self, _: object) -> "_StubWorks":
        return self

    def sort(self, **_: object) -> "_StubWorks":
        return self


class _StubManifestIndex:
    """Stub manifest index that exposes the iterator expected by resolve_config."""

    def __init__(self, path: object, eager: object) -> None:  # pragma: no cover - trivial
        self.path = path
        self.eager = eager

    def iter_existing_paths(self):  # pragma: no cover - trivial
        return []

    def iter_existing(self):  # pragma: no cover - trivial
        return self.iter_existing_paths()


@pytest.fixture(autouse=True)
def _patch_dependencies(monkeypatch):
    """Replace heavy dependencies with lightweight test doubles."""

    monkeypatch.setattr(download_args, "Works", _StubWorks)
    monkeypatch.setattr(download_args, "default_resolvers", lambda: [SimpleNamespace(name="openalex")])
    monkeypatch.setattr(
        download_args,
        "load_resolver_config",
        lambda *_: SimpleNamespace(max_concurrent_resolvers=1, polite_headers={"User-Agent": "stub"}),
    )
    monkeypatch.setattr(download_args, "ManifestUrlIndex", _StubManifestIndex)


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    """Provide a fresh parser instance for each test."""

    return download_args.build_parser()


def _base_args() -> list[str]:
    return [
        "--topic",
        "ai",
        "--year-start",
        "2020",
        "--year-end",
        "2021",
        "--ignore-robots",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["--year-start", "2020", "--year-end", "2021", "--ignore-robots"],
        ["--topic", "", "--year-start", "2020", "--year-end", "2021", "--ignore-robots"],
        [
            "--topic-id",
            "   ",
            "--year-start",
            "2020",
            "--year-end",
            "2021",
            "--ignore-robots",
        ],
    ],
)
def test_resolve_config_requires_topic_or_topic_id(parser, capfd, argv):
    args = download_args.parse_args(parser, argv)

    with pytest.raises(SystemExit) as excinfo:
        download_args.resolve_config(args, parser)

    assert excinfo.value.code == 2
    _, err = capfd.readouterr()
    assert "Provide --topic or --topic-id." in err


@pytest.mark.parametrize("per_page", [0, 201])
def test_resolve_config_rejects_invalid_per_page(parser, capfd, per_page):
    argv = _base_args() + ["--per-page", str(per_page)]
    args = download_args.parse_args(parser, argv)

    with pytest.raises(SystemExit) as excinfo:
        download_args.resolve_config(args, parser)

    assert excinfo.value.code == 2
    _, err = capfd.readouterr()
    assert "--per-page must be between 1 and 200" in err


def test_resolve_config_rejects_negative_sleep(parser, capfd):
    argv = _base_args() + ["--sleep", "-1"]
    args = download_args.parse_args(parser, argv)

    with pytest.raises(SystemExit) as excinfo:
        download_args.resolve_config(args, parser)

    assert excinfo.value.code == 2
    _, err = capfd.readouterr()
    assert "--sleep must be greater than or equal to 0" in err


@pytest.mark.parametrize("max_value", [-10])
def test_resolve_config_rejects_non_positive_max(parser, capfd, max_value):
    argv = _base_args() + ["--max", str(max_value)]
    args = download_args.parse_args(parser, argv)

    with pytest.raises(SystemExit) as excinfo:
        download_args.resolve_config(args, parser)

    assert excinfo.value.code == 2
    _, err = capfd.readouterr()
    assert "--max must be greater than or equal to 0" in err


def test_resolve_config_rejects_inverted_year_range(parser, capfd):
    argv = [
        "--topic",
        "ai",
        "--year-start",
        "2022",
        "--year-end",
        "2020",
        "--ignore-robots",
    ]
    args = download_args.parse_args(parser, argv)

    with pytest.raises(SystemExit) as excinfo:
        download_args.resolve_config(args, parser)

    assert excinfo.value.code == 2
    _, err = capfd.readouterr()
    assert "--year-start must be less than or equal to --year-end" in err


@pytest.mark.parametrize("per_page", [1, 200])
def test_resolve_config_accepts_per_page_boundaries(parser, per_page):
    argv = _base_args() + ["--per-page", str(per_page)]
    args = download_args.parse_args(parser, argv)

    resolved = download_args.resolve_config(args, parser)

    assert resolved.args.per_page == per_page


@pytest.mark.parametrize("max_value", [5, 0])
def test_resolve_config_accepts_non_negative_max(parser, max_value):
    argv = _base_args() + ["--max", str(max_value)]
    args = download_args.parse_args(parser, argv)

    resolved = download_args.resolve_config(args, parser)

    assert resolved.args.max == max_value


def test_resolve_config_sets_pyalex_email_from_config(parser, monkeypatch):
    from pyalex import config as oa_config

    monkeypatch.setattr(
        download_args,
        "load_resolver_config",
        lambda *_: SimpleNamespace(
            max_concurrent_resolvers=1,
            polite_headers={"User-Agent": "stub"},
            mailto="config@example.org",
        ),
    )
    monkeypatch.setattr(oa_config, "email", "initial@example.org", raising=False)

    args = download_args.parse_args(parser, _base_args())
    download_args.resolve_config(args, parser)

    assert oa_config.email == "config@example.org"
