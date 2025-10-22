from __future__ import annotations

from pathlib import Path

import pytest

from DocsToKG.ContentDownload.api import DownloadPlan
from DocsToKG.ContentDownload.api.exceptions import DownloadError
from DocsToKG.ContentDownload.download_execution import (
    finalize_candidate_download,
    stream_candidate_payload,
)


class _StubHeadResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class _StubGetResponse:
    def __init__(self, payload: bytes) -> None:
        self.status_code = 200
        self._payload = payload
        self.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": str(len(payload)),
        }
        self.extensions = {"from_cache": False, "revalidated": False}

    def iter_bytes(self, *, chunk_size: int):  # pragma: no cover - signature parity
        del chunk_size
        yield self._payload


class _StubSession:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def head(self, url: str, **_: object) -> _StubHeadResponse:  # pragma: no cover - signature parity
        assert url.startswith("https://")
        return _StubHeadResponse(200)

    def get(self, url: str, **_: object) -> _StubGetResponse:  # pragma: no cover - signature parity
        assert url.startswith("https://")
        return _StubGetResponse(self._payload)


def _run_stream(
    plan: DownloadPlan,
    *,
    payload: bytes,
    max_bytes: int | None = None,
):
    session = _StubSession(payload)
    return stream_candidate_payload(
        plan,
        session=session,
        telemetry=None,
        run_id="run-1",
        max_bytes=max_bytes,
    )


def test_stream_and_finalize_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("DOCSTOKG_STORAGE_TMP", str(tmp_path / "tmp"))

    plan = DownloadPlan(url="https://example.org/file.pdf", resolver_name="resolver")
    payload = b"payload"

    result = _run_stream(plan, payload=payload)

    tmp_file = Path(result.path_tmp)
    assert result.staging_path is not None
    staging_dir = Path(result.staging_path)
    assert tmp_file.exists()
    assert staging_dir.exists()

    final_dest = storage_root / "final.pdf"
    outcome = finalize_candidate_download(
        plan,
        result,
        final_path=str(final_dest),
        telemetry=None,
        run_id="run-1",
        storage_root=str(storage_root),
    )

    assert outcome.ok is True
    assert outcome.path == str(final_dest.resolve())
    assert final_dest.exists()
    assert final_dest.read_bytes() == payload
    assert not staging_dir.exists()


def test_stream_enforces_size_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tmp_env = tmp_path / "tmp"
    monkeypatch.setenv("DOCSTOKG_STORAGE_TMP", str(tmp_env))

    plan = DownloadPlan(url="https://example.org/file.pdf", resolver_name="resolver")
    payload = b"0123456789"

    with pytest.raises(DownloadError) as excinfo:
        _run_stream(plan, payload=payload, max_bytes=5)

    assert excinfo.value.reason == "too-large"

    staging_root = tmp_env / "staging"
    if staging_root.exists():
        assert not any(staging_root.rglob("*"))


def test_finalize_cleans_staging_on_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tmp_env = tmp_path / "tmp"
    monkeypatch.setenv("DOCSTOKG_STORAGE_TMP", str(tmp_env))

    plan = DownloadPlan(url="https://example.org/file.pdf", resolver_name="resolver")
    payload = b"payload"

    result = _run_stream(plan, payload=payload)
    tmp_file = Path(result.path_tmp)
    assert result.staging_path is not None
    staging_dir = Path(result.staging_path)
    assert tmp_file.exists()
    assert staging_dir.exists()

    def _fail_replace(src: str, dst: str) -> None:  # pragma: no cover - signature parity
        raise OSError(f"cannot move {src} -> {dst}")

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.download_execution.os.replace",
        _fail_replace,
    )

    with pytest.raises(DownloadError) as excinfo:
        finalize_candidate_download(
            plan,
            result,
            final_path=str(tmp_path / "final.pdf"),
            storage_root=str(tmp_path),
        )

    assert excinfo.value.reason == "download-error"
    assert not tmp_file.exists()
    assert not staging_dir.exists()
