# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.discovery",
#   "purpose": "Filesystem discovery utilities for DocParsing manifests and artifacts.",
#   "sections": [
#     {
#       "id": "chunkdiscovery",
#       "name": "ChunkDiscovery",
#       "anchor": "class-chunkdiscovery",
#       "kind": "class"
#     },
#     {
#       "id": "ensure-str-sequence",
#       "name": "_ensure_str_sequence",
#       "anchor": "function-ensure-str-sequence",
#       "kind": "function"
#     },
#     {
#       "id": "load-structural-marker-profile",
#       "name": "load_structural_marker_profile",
#       "anchor": "function-load-structural-marker-profile",
#       "kind": "function"
#     },
#     {
#       "id": "load-structural-marker-config",
#       "name": "load_structural_marker_config",
#       "anchor": "function-load-structural-marker-config",
#       "kind": "function"
#     },
#     {
#       "id": "normalise-chunk-relative",
#       "name": "_normalise_chunk_relative",
#       "anchor": "function-normalise-chunk-relative",
#       "kind": "function"
#     },
#     {
#       "id": "vector-artifact-name",
#       "name": "vector_artifact_name",
#       "anchor": "function-vector-artifact-name",
#       "kind": "function"
#     },
#     {
#       "id": "derive-doc-id-and-doctags-path",
#       "name": "derive_doc_id_and_doctags_path",
#       "anchor": "function-derive-doc-id-and-doctags-path",
#       "kind": "function"
#     },
#     {
#       "id": "derive-doc-id-and-chunks-path",
#       "name": "derive_doc_id_and_chunks_path",
#       "anchor": "function-derive-doc-id-and-chunks-path",
#       "kind": "function"
#     },
#     {
#       "id": "derive-doc-id-and-vectors-path",
#       "name": "derive_doc_id_and_vectors_path",
#       "anchor": "function-derive-doc-id-and-vectors-path",
#       "kind": "function"
#     },
#     {
#       "id": "compute-relative-doc-id",
#       "name": "compute_relative_doc_id",
#       "anchor": "function-compute-relative-doc-id",
#       "kind": "function"
#     },
#     {
#       "id": "compute-stable-shard",
#       "name": "compute_stable_shard",
#       "anchor": "function-compute-stable-shard",
#       "kind": "function"
#     },
#     {
#       "id": "iter-chunks",
#       "name": "iter_chunks",
#       "anchor": "function-iter-chunks",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Filesystem discovery utilities for DocParsing manifests and artifacts.

The discovery layer translates on-disk structures—DocTags, chunk JSONL files,
and embedding vectors—into stable logical identifiers that downstream stages
can reason about. It encapsulates structural marker loading, shard calculation,
and relative-path hashing so resume logic remains deterministic even when input
directories move or contain symlinks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from DocsToKG.DocParsing.config import load_toml_markers, load_yaml_markers

DEFAULT_HEADING_MARKERS: tuple[str, ...] = ("#",)
DEFAULT_CAPTION_MARKERS: tuple[str, ...] = (
    "Figure caption:",
    "Table:",
    "Picture description:",
    "<!-- image -->",
)

__all__ = [
    "DEFAULT_CAPTION_MARKERS",
    "DEFAULT_HEADING_MARKERS",
    "ChunkDiscovery",
    "compute_relative_doc_id",
    "compute_stable_shard",
    "derive_doc_id_and_chunks_path",
    "derive_doc_id_and_doctags_path",
    "derive_doc_id_and_vectors_path",
    "vector_artifact_name",
    "iter_chunks",
    "load_structural_marker_config",
    "load_structural_marker_profile",
]


@dataclass(frozen=True)
class ChunkDiscovery:
    """Discovery record that retains logical and resolved chunk paths."""

    logical_path: Path
    """Path of the chunk file relative to the traversal root."""

    resolved_path: Path
    """Canonical filesystem path of the chunk file (after symlink resolution)."""

    def __fspath__(self) -> str:
        """Return the resolved path for :func:`os.fspath` compatibility."""

        return str(self.resolved_path)


def _ensure_str_sequence(value: object, label: str) -> list[str]:
    """Normalise structural marker entries into string lists."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a list of strings for '{label}'")
    return [item for item in value if item]


def load_structural_marker_profile(path: Path) -> tuple[list[str], list[str]]:
    """Load heading/caption marker overrides from JSON, YAML, or TOML files."""

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    parsers: list[str] = []
    if suffix in {".yaml", ".yml"}:
        parsers = ["yaml"]
    elif suffix == ".toml":
        parsers = ["toml"]
    elif suffix == ".json":
        parsers = ["json"]
    else:
        parsers = ["json", "yaml", "toml"]

    data: object = None
    last_error: Exception | None = None

    for parser in parsers:
        try:
            if parser == "json":
                data = json.loads(raw)
            elif parser == "yaml":
                data = load_yaml_markers(raw)
            elif parser == "toml":
                data = load_toml_markers(raw)
            else:  # pragma: no cover - defensive branch
                continue
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            data = None
            continue
        except Exception as exc:
            last_error = exc
            data = None
            continue

        if data is not None:
            break

    if data is None:
        messages = ", ".join(parsers)
        detail = f"; last error: {last_error}" if last_error else ""
        raise ValueError(
            f"Unable to parse structural marker file {path} using supported formats ({messages}){detail}."
        )

    if isinstance(data, list):
        headings = _ensure_str_sequence(data, "headings")
        captions: list[str] = []
    elif isinstance(data, dict):
        headings = _ensure_str_sequence(data.get("headings"), "headings")
        captions = _ensure_str_sequence(data.get("captions"), "captions")
    else:
        raise ValueError(f"Unsupported structural marker format in {path}")

    return headings, captions


def load_structural_marker_config(path: Path) -> tuple[list[str], list[str]]:
    """Backward compatible alias for :func:`load_structural_marker_profile`."""

    return load_structural_marker_profile(path)


def _normalise_chunk_relative(path: Path) -> Path:
    """Return ``path`` with ``.chunks[.jsonl]`` suffixes removed."""

    base = path
    if base.suffix == ".jsonl":
        base = base.with_suffix("")
    if base.suffix == ".chunks":
        base = base.with_suffix("")
    return base


def vector_artifact_name(logical: Path, fmt: str) -> Path:
    """Return the vectors filename for a chunk ``logical`` path and format."""

    fmt_normalised = str(fmt or "jsonl").lower()
    if fmt_normalised == "jsonl":
        suffix = ".vectors.jsonl"
    elif fmt_normalised == "parquet":
        suffix = ".vectors.parquet"
    else:
        raise ValueError(f"Unsupported vector format: {fmt}")
    base = _normalise_chunk_relative(logical)
    return base.with_suffix(suffix)


def derive_doc_id_and_doctags_path(
    source_pdf: Path, pdfs_root: Path, doctags_root: Path
) -> tuple[str, Path]:
    """Return manifest doc identifier and DocTags output path for ``source_pdf``."""

    doc_id = compute_relative_doc_id(source_pdf, pdfs_root)
    relative = Path(doc_id)
    doctags_path = (doctags_root / relative).with_suffix(".doctags")
    return doc_id, doctags_path


def derive_doc_id_and_chunks_path(
    doctags_file: Path, doctags_root: Path, chunks_root: Path
) -> tuple[str, Path]:
    """Return manifest doc identifier and chunk output path for ``doctags_file``."""

    doc_id = compute_relative_doc_id(doctags_file, doctags_root)
    relative = Path(doc_id)
    chunk_path = (chunks_root / relative).with_suffix(".chunks.jsonl")
    return doc_id, chunk_path


def derive_doc_id_and_vectors_path(
    chunk_file: Path | ChunkDiscovery,
    chunks_root: Path,
    vectors_root: Path,
    *,
    vector_format: str = "jsonl",
) -> tuple[str, Path]:
    """Return manifest doc identifier and vectors output path for ``chunk_file``."""

    if isinstance(chunk_file, ChunkDiscovery):
        relative = chunk_file.logical_path
    else:
        if chunk_file.is_absolute():
            try:
                relative = chunk_file.relative_to(chunks_root)
            except ValueError as exc:
                raise ValueError(
                    "Chunk file is outside the provided chunks_root; pass a ChunkDiscovery "
                    "instance to preserve the logical tree path."
                ) from exc
        else:
            relative = chunk_file
    base = _normalise_chunk_relative(relative)
    doc_id = base.with_suffix(".doctags").as_posix()
    vector_relative = vector_artifact_name(base, vector_format)
    return doc_id, vectors_root / vector_relative


def compute_relative_doc_id(path: Path, root: Path) -> str:
    """Return POSIX-style relative identifier for a document path."""

    return path.relative_to(root).as_posix()


def compute_stable_shard(identifier: str, shard_count: int) -> int:
    """Deterministically map ``identifier`` to a shard in ``[0, shard_count)``."""

    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def iter_chunks(directory: Path) -> Iterator[ChunkDiscovery]:
    """Yield :class:`ChunkDiscovery` records for chunk files under ``directory``."""

    if not directory.exists():
        return

    if directory.is_file():
        if directory.name.endswith(".chunks.jsonl"):
            resolved = directory.resolve()
            yield ChunkDiscovery(Path(directory.name), resolved)
        return

    root = directory.resolve()
    yielded_symlink_targets: set[Path] = set()

    def _walk(current: Path) -> Iterator[ChunkDiscovery]:
        """Yield chunk files beneath ``current`` depth-first with symlink guards."""

        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except FileNotFoundError:  # pragma: no cover - defensive guard
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and not entry.is_symlink():
                yield from _walk(entry)
            elif entry.is_file() and entry.name.endswith(".chunks.jsonl"):
                resolved = entry.resolve()
                if entry.is_symlink():
                    if resolved in yielded_symlink_targets:
                        continue
                    yielded_symlink_targets.add(resolved)
                relative = entry.relative_to(root)
                yield ChunkDiscovery(relative, resolved)

    yield from _walk(root)
