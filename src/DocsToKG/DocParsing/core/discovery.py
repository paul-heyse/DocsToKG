"""Path and artifact discovery helpers for DocParsing pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, List, Tuple

from DocsToKG.DocParsing.config import load_toml_markers, load_yaml_markers

DEFAULT_HEADING_MARKERS: Tuple[str, ...] = ("#",)
DEFAULT_CAPTION_MARKERS: Tuple[str, ...] = (
    "Figure caption:",
    "Table:",
    "Picture description:",
    "<!-- image -->",
)

__all__ = [
    "DEFAULT_CAPTION_MARKERS",
    "DEFAULT_HEADING_MARKERS",
    "compute_relative_doc_id",
    "compute_stable_shard",
    "derive_doc_id_and_chunks_path",
    "derive_doc_id_and_doctags_path",
    "derive_doc_id_and_vectors_path",
    "iter_chunks",
    "load_structural_marker_config",
    "load_structural_marker_profile",
]


def _ensure_str_sequence(value: object, label: str) -> List[str]:
    """Normalise structural marker entries into string lists."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a list of strings for '{label}'")
    return [item for item in value if item]


def load_structural_marker_profile(path: Path) -> Tuple[List[str], List[str]]:
    """Load heading/caption marker overrides from JSON, YAML, or TOML files."""

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    parsers: List[str] = []
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
        captions: List[str] = []
    elif isinstance(data, dict):
        headings = _ensure_str_sequence(data.get("headings"), "headings")
        captions = _ensure_str_sequence(data.get("captions"), "captions")
    else:
        raise ValueError(f"Unsupported structural marker format in {path}")

    return headings, captions


def load_structural_marker_config(path: Path) -> Tuple[List[str], List[str]]:
    """Backward compatible alias for :func:`load_structural_marker_profile`."""

    return load_structural_marker_profile(path)


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
    chunk_file: Path, chunks_root: Path, vectors_root: Path
) -> tuple[str, Path]:
    """Return manifest doc identifier and vectors output path for ``chunk_file``."""

    if chunks_root.is_file():
        relative = Path(chunk_file.name)
    else:
        relative = chunk_file.relative_to(chunks_root)
    base = relative
    if base.suffix == ".jsonl":
        base = base.with_suffix("")
    if base.suffix == ".chunks":
        base = base.with_suffix("")
    doc_id = base.with_suffix(".doctags").as_posix()
    vector_relative = base.with_suffix(".vectors.jsonl")
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


def iter_chunks(directory: Path) -> Iterator[Path]:
    """Yield chunk JSONL files from ``directory`` and all descendants."""

    root = directory.resolve()
    if root.is_file():
        if root.name.endswith(".chunks.jsonl"):
            yield root
        return
    if not root.exists():
        return

    yielded_symlink_targets: set[Path] = set()

    def _walk(current: Path) -> Iterator[Path]:
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
                yield resolved

    yield from _walk(root)
