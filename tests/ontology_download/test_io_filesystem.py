# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_io_filesystem",
#   "purpose": "Filesystem helper regression tests.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Filesystem helper regression tests.

Covers cached file materialisation, cross-device fallback paths, archive
extraction safety, sanitized filenames, and correlation ID generation for
logging. Ensures ontology artefacts are staged securely before validation."""

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
