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


@pytest.mark.parametrize(
    ("manifest_arg", "csv_arg"),
    [
        ("manifest.jsonl", "attempts.csv"),
        ("~/logs/manifest.jsonl", "~/logs/attempts.csv"),
    ],
)
def test_resolve_config_expands_csv_path_like_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_arg: str, csv_arg: str
) -> None:
    home_dir = tmp_path / "home"
    work_dir = tmp_path / "work"
    home_dir.mkdir()
    work_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(work_dir)

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
