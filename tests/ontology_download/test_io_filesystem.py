"""Regression tests for filesystem helpers."""

from __future__ import annotations

import errno
from pathlib import Path
from unittest.mock import patch

from DocsToKG.OntologyDownload.io.filesystem import _materialize_cached_file


def test_materialize_cached_file_preserves_cache_on_cross_device(tmp_path_factory) -> None:
    """When linking fails, copies should keep the original cache file intact."""

    cache_dir = tmp_path_factory.mktemp("cache-root")
    destination_dir = tmp_path_factory.mktemp("dest-root")

    source = Path(cache_dir) / "example.bin"
    source.write_bytes(b"payload")

    destination = Path(destination_dir) / "example.bin"

    with patch(
        "DocsToKG.OntologyDownload.io.filesystem.os.link",
        side_effect=OSError(errno.EXDEV, "Invalid cross-device link"),
    ):
        artifact_path, cache_path = _materialize_cached_file(source, destination)

    assert artifact_path == destination
    assert cache_path == source
    assert destination.read_bytes() == b"payload"
    assert source.exists(), "cache entry should remain after materialization"
