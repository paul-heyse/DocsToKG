"""Minimal stub of the :mod:`pooch` API used in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

__all__ = ["HTTPDownloader", "retrieve"]


class HTTPDownloader:
    def __init__(
        self,
        *,
        headers: Dict[str, str] | None = None,
        progressbar: bool = False,
        timeout: float | None = None,
    ) -> None:
        self.headers = headers or {}
        self.progressbar = progressbar
        self.timeout = timeout

    def __call__(self, url: str, output_file: str, pooch_logger: Any | None = None) -> None:
        # Real implementation streams to ``output_file``. Tests patch the download
        # pipeline before this method is exercised, so the stub can no-op safely.
        return None


def retrieve(
    url: str,
    fname: str,
    *,
    path: str | Path | None = None,
    known_hash: str | None = None,
    downloader: HTTPDownloader | None = None,
    **_: Any,
) -> str:
    """Return a deterministic local path for the requested download."""

    destination_dir = Path(path) if path is not None else Path.cwd()
    destination = destination_dir / fname
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"")
    if downloader is not None:
        downloader(url, str(destination), None)
    return str(destination)
