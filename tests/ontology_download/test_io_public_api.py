# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_io_public_api",
#   "purpose": "Public re-export smoke tests for the IO package.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Public re-export smoke tests for the IO package.

Ensures convenience imports from ``DocsToKG.OntologyDownload.io`` expose
critical helpers expected by downstream callers."""

from __future__ import annotations

from DocsToKG.OntologyDownload import io
from DocsToKG.OntologyDownload.io.extraction_constraints import (
    validate_file_size as _impl_validate_file_size,
)


def test_validate_file_size_is_reexported() -> None:
    """``validate_file_size`` should remain available via the io package."""

    assert hasattr(io, "validate_file_size")
    assert io.validate_file_size is _impl_validate_file_size
