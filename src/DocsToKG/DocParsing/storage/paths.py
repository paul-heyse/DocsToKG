"""
Dataset Layout & Path Builders

Encapsulates the canonical directory hierarchy and rel_id normalization for DocTags,
Chunks, and Vectors (Dense, Sparse, Lexical) with Parquet/Arrow storage.

Directory Layout:
    Data/
      Doctags/{yyyy}/{mm}/{rel_id}.jsonl
      Chunks/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet
      Vectors/{family=dense|sparse|lexical}/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.parquet
      Manifests/...

Key Concepts:
- `rel_id`: Stable, filesystem-safe, relative identifier derived from source path (normalized NFC).
- Partitions: {yyyy}/{mm} based on write time; {family}, {fmt} as literal partition keys.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple


def normalize_rel_id(source_path: str | Path, max_length: int = 512) -> str:
    """
    Normalize a source path into a stable `rel_id` for storage.

    Rules:
    - Unicode NFC normalization.
    - Case-preserving (no automatic downcase).
    - Replace disallowed chars ([^A-Za-z0-9._~/ -]) with `_`.
    - Max length: 512 code points.
    - Remove leading/trailing slashes and `..` components.

    Args:
        source_path: Source file path (relative or absolute).
        max_length: Maximum length of rel_id in code points.

    Returns:
        Normalized rel_id string.

    Raises:
        ValueError: If rel_id exceeds max_length after normalization.
    """
    # Convert to string and normalize to NFC
    path_str = str(source_path)
    normalized = unicodedata.normalize("NFC", path_str)

    # Remove extension (keep path up to last dot that indicates extension)
    if normalized.endswith((".json", ".jsonl", ".parquet", ".pdf", ".html")):
        # Remove known extensions
        for ext in (".jsonl", ".json", ".parquet", ".pdf", ".html"):
            if normalized.endswith(ext):
                normalized = normalized[: -len(ext)]
                break

    # Strip leading/trailing slashes
    normalized = normalized.strip("/")

    # Replace disallowed characters with `_`
    allowed_set = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~/ -")
    normalized = "".join(c if c in allowed_set else "_" for c in normalized)

    # Remove `..` sequences (prevent directory traversal)
    while ".." in normalized:
        normalized = normalized.replace("..", "_")

    # Enforce max length
    if len(normalized) > max_length:
        raise ValueError(f"rel_id exceeds max length {max_length}: {len(normalized)} code points")

    return normalized


def _timestamp_to_partition(ts: Optional[datetime] = None) -> Tuple[str, str]:
    """
    Convert datetime to (yyyy, mm) partition strings.

    Args:
        ts: Datetime (UTC). If None, uses current UTC time.

    Returns:
        Tuple of (yyyy, mm) strings.
    """
    if ts is None:
        ts = datetime.now(timezone.utc)
    return ts.strftime("%Y"), ts.strftime("%m")


def chunks_output_path(
    data_root: str | Path,
    rel_id: str,
    fmt: Literal["parquet", "jsonl"] = "parquet",
    ts: Optional[datetime] = None,
) -> Path:
    """
    Build the output path for a Chunks artifact.

    Layout: `{data_root}/Chunks/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.{ext}`

    Args:
        data_root: Data root directory.
        rel_id: Normalized relative identifier.
        fmt: Output format ("parquet" or "jsonl").
        ts: Write timestamp (UTC). If None, uses current time.

    Returns:
        Path object for the Chunks file.
    """
    yyyy, mm = _timestamp_to_partition(ts)
    ext = "parquet" if fmt == "parquet" else "jsonl"
    root = Path(data_root)
    return root / "Chunks" / f"fmt={fmt}" / yyyy / mm / f"{rel_id}.{ext}"


def doctags_output_path(
    data_root: str | Path,
    rel_id: str,
    ts: Optional[datetime] = None,
) -> Path:
    """
    Build the output path for DocTags (always JSONL).

    Layout: `{data_root}/Doctags/{yyyy}/{mm}/{rel_id}.jsonl`

    Args:
        data_root: Data root directory.
        rel_id: Normalized relative identifier.
        ts: Write timestamp (UTC). If None, uses current time.

    Returns:
        Path object for the DocTags file.
    """
    yyyy, mm = _timestamp_to_partition(ts)
    root = Path(data_root)
    return root / "Doctags" / yyyy / mm / f"{rel_id}.jsonl"


def vectors_output_path(
    data_root: str | Path,
    family: Literal["dense", "sparse", "lexical"],
    rel_id: str,
    fmt: Literal["parquet", "jsonl"] = "parquet",
    ts: Optional[datetime] = None,
) -> Path:
    """
    Build the output path for a Vectors artifact.

    Layout:
    `{data_root}/Vectors/{family=dense|sparse|lexical}/{fmt=parquet}/{yyyy}/{mm}/{rel_id}.{ext}`

    Args:
        data_root: Data root directory.
        family: Vector family ("dense", "sparse", or "lexical").
        rel_id: Normalized relative identifier.
        fmt: Output format ("parquet" or "jsonl").
        ts: Write timestamp (UTC). If None, uses current time.

    Returns:
        Path object for the Vectors file.

    Raises:
        ValueError: If family is not in ("dense", "sparse", "lexical").
    """
    if family not in ("dense", "sparse", "lexical"):
        raise ValueError(f"Invalid family: {family}")

    yyyy, mm = _timestamp_to_partition(ts)
    ext = "parquet" if fmt == "parquet" else "jsonl"
    root = Path(data_root)
    return root / "Vectors" / f"family={family}" / f"fmt={fmt}" / yyyy / mm / f"{rel_id}.{ext}"


def chunk_file_glob_pattern(data_root: str | Path, family: Optional[str] = None) -> str:
    """
    Return a glob pattern to discover Chunks or Vectors Parquet files.

    Patterns:
    - Chunks: `{data_root}/Chunks/fmt=parquet/*/*.parquet`
    - Vectors/dense: `{data_root}/Vectors/family=dense/fmt=parquet/*/*.parquet`
    - Vectors/sparse: `{data_root}/Vectors/family=sparse/fmt=parquet/*/*.parquet`
    - Vectors/lexical: `{data_root}/Vectors/family=lexical/fmt=parquet/*/*.parquet`

    Args:
        data_root: Data root directory.
        family: Vector family or None for Chunks. One of ("dense", "sparse", "lexical", None).

    Returns:
        Glob pattern string.
    """
    root = Path(data_root)
    if family is None:
        return str(root / "Chunks" / "fmt=parquet" / "*" / "*" / "*.parquet")
    elif family in ("dense", "sparse", "lexical"):
        return str(root / "Vectors" / f"family={family}" / "fmt=parquet" / "*" / "*" / "*.parquet")
    else:
        raise ValueError(f"Invalid family: {family}")


def extract_partition_keys(file_path: str | Path) -> Optional[dict]:
    """
    Parse partition keys (family, fmt, yyyy, mm) from a file path.

    Returns a dict with keys: family (or None for Chunks), fmt, yyyy, mm.
    Returns None if the path doesn't match the expected layout.

    Args:
        file_path: File path to parse.

    Returns:
        Dict with partition keys, or None.
    """
    path = Path(file_path)
    parts = path.parts

    # Expected: .../{Chunks|Vectors}/[family=X/]fmt=Y/yyyy/mm/filename.parquet
    try:
        dataset_idx = None
        for i, part in enumerate(parts):
            if part in ("Chunks", "Vectors"):
                dataset_idx = i
                break

        if dataset_idx is None:
            return None

        remaining = parts[dataset_idx + 1 :]
        if len(remaining) < 4:
            return None

        family = None
        fmt_idx = 0

        # Check if first remaining part is family=X (for Vectors)
        if remaining[0].startswith("family="):
            family = remaining[0].split("=", 1)[1]
            fmt_idx = 1

        # Next should be fmt=Y
        if not remaining[fmt_idx].startswith("fmt="):
            return None
        fmt = remaining[fmt_idx].split("=", 1)[1]

        # Next should be yyyy, mm, filename
        if len(remaining) < fmt_idx + 4:
            return None

        yyyy = remaining[fmt_idx + 1]
        mm = remaining[fmt_idx + 2]

        return {
            "family": family,
            "fmt": fmt,
            "yyyy": yyyy,
            "mm": mm,
        }
    except (IndexError, ValueError):
        return None
