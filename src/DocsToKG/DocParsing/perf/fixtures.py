# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.perf.fixtures",
#   "purpose": "Reusable fixtures for DocParsing performance runs.",
#   "sections": [
#     {
#       "id": "fixture-paths",
#       "name": "FixturePaths",
#       "anchor": "class-fixture-paths",
#       "kind": "class"
#     },
#     {
#       "id": "prepare-synthetic-fixture",
#       "name": "prepare_synthetic_html_fixture",
#       "anchor": "function-prepare-synthetic-fixture",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Reusable fixtures for DocParsing performance runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from DocsToKG.DocParsing.env import prepare_data_root


@dataclass(slots=True)
class FixturePaths:
    """Resolved directories for a synthetic profiling dataset."""

    root: Path
    html_dir: Path
    doctags_dir: Path
    chunks_dir: Path
    vectors_dir: Path


def prepare_synthetic_html_fixture(
    *, data_root: Path, name: str = "synthetic-html", documents: int = 5
) -> FixturePaths:
    """Create a deterministic HTML fixture for profiling runs."""

    resolved_root = prepare_data_root(data_root, data_root)
    fixture_root = resolved_root / "Profiles" / name
    html_dir = fixture_root / "HTML"
    doctags_dir = fixture_root / "DocTags"
    chunks_dir = fixture_root / "Chunks"
    vectors_dir = fixture_root / "Vectors"

    html_dir.mkdir(parents=True, exist_ok=True)
    doctags_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    sample_block = "<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>"
    for index in range(documents):
        content = (
            "<html><head><title>DocParsing Fixture</title></head><body>"
            f"<h1>Fixture Document {index}</h1>"
            f"{sample_block * 5}"
            "</body></html>"
        )
        (html_dir / f"fixture-{index:02d}.html").write_text(content, encoding="utf-8")

    return FixturePaths(
        root=fixture_root,
        html_dir=html_dir,
        doctags_dir=doctags_dir,
        chunks_dir=chunks_dir,
        vectors_dir=vectors_dir,
    )


__all__ = ["FixturePaths", "prepare_synthetic_html_fixture"]
