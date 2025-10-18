"""CLI validation tests for the ``docparse manifest`` entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from DocsToKG.DocParsing.cli_errors import CLIValidationError

from tests.docparsing.test_manifest_streaming_cli import _prepare_manifest_cli_stubs


def test_manifest_accepts_known_stage(monkeypatch, tmp_path) -> None:
    """CLI should accept known manifest stages and reject unsupported ones."""

    _prepare_manifest_cli_stubs(monkeypatch)

    from DocsToKG.DocParsing.core import cli

    manifests_dir = tmp_path / "Manifests"
    manifests_dir.mkdir()

    captured: dict[str, object] = {}
    call_count = {"value": 0}

    def fake_iter_manifest_entries(stages, data_root: Path):
        call_count["value"] += 1
        captured["stages"] = list(stages)
        captured["data_root"] = data_root
        return iter(())

    monkeypatch.setattr(cli, "iter_manifest_entries", fake_iter_manifest_entries)
    monkeypatch.setattr(
        cli,
        "data_manifests",
        lambda _root, *, ensure=False: manifests_dir,
    )

    exit_code = cli.manifest(["--stage", "doctags", "--data-root", str(tmp_path)])

    assert exit_code == 0
    assert captured["stages"] == ["doctags-html", "doctags-pdf"]
    assert captured["data_root"] == tmp_path

    with pytest.raises(CLIValidationError) as excinfo:
        cli._manifest_main(["--stage", "unknown-stage", "--data-root", str(tmp_path)])

    assert "Unsupported stage" in str(excinfo.value)
    assert call_count["value"] == 1


def test_docparse_manifest_reports_structured_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI wrapper should format validation errors consistently."""

    _prepare_manifest_cli_stubs(monkeypatch)

    from DocsToKG.DocParsing.core import cli

    manifests_dir = tmp_path / "Manifests"
    manifests_dir.mkdir()

    monkeypatch.setattr(cli, "data_manifests", lambda _root, *, ensure=False: manifests_dir)

    exit_code = cli.manifest(["--stage", "bad"])

    assert exit_code == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    expected = (
        "[cli] --stage: Unsupported stage 'bad'. Canonical stages: doctags-html, "
        "doctags-pdf, chunks, embeddings. Discovered stages: none. Hint: Choose a supported "
        "manifest stage."
    )
    assert captured.err.strip() == expected
