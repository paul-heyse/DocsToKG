# file: src/DocsToKG/DocParsing/storage/parquet_schemas.py
# Purpose: Executable Arrow schema declarations + Parquet footer contract
# Compatible with: pyarrow >= 9 (recommended >= 12)

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# Version tags (SemVer) – bump only with spec changes
# ============================================================

SCHEMA_VERSION_CHUNKS = "docparse/chunks/1.0.0"
SCHEMA_VERSION_DENSE = "docparse/vectors/dense/1.0.0"
SCHEMA_VERSION_SPARSE = "docparse/vectors/sparse/1.0.0"
SCHEMA_VERSION_LEXICAL = "docparse/vectors/lexical/1.0.0"

ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"
DOCPARSE_PREFIX = "docparse."

# ============================================================
# Arrow schema factories
# ============================================================


def chunks_schema(include_optional: bool = True) -> pa.Schema:
    """
    Chunks dataset schema (Parquet). Default includes optional columns.
    """
    fields = [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("tokens", pa.int32(), nullable=False),
        pa.field(
            "span",
            pa.struct(
                [
                    pa.field("start", pa.int32(), nullable=False),
                    pa.field("end", pa.int32(), nullable=False),
                ]
            ),
            nullable=False,
        ),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
    if include_optional:
        fields += [
            pa.field("section", pa.string()),
            pa.field("meta", pa.map_(pa.string(), pa.string())),
        ]
    return pa.schema(fields)


def dense_schema(dim: int, fixed_size: bool = True) -> pa.Schema:
    """
    Dense vectors schema.
    - Prefer fixed-size list when 'dim' is constant across all rows.
    - Fallback (fixed_size=False) uses variable-length lists with per-row 'dim' check.
    """
    if dim <= 0:
        raise ValueError("dim must be > 0")
    vec_type = pa.list_(pa.float32(), list_size=dim) if fixed_size else pa.list_(pa.float32())
    fields = [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("dim", pa.int32(), nullable=False),
        pa.field("vec", vec_type, nullable=False),
        pa.field("normalize_l2", pa.bool_(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
    return pa.schema(fields)


def sparse_schema_idspace() -> pa.Schema:
    """
    Sparse (SPLADE) vectors stored as index/weight lists.
    """
    fields = [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("nnz", pa.int32(), nullable=False),
        pa.field("indices", pa.list_(pa.int32()), nullable=False),
        pa.field("weights", pa.list_(pa.float32()), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
    return pa.schema(fields)


def lexical_schema_idspace() -> pa.Schema:
    """
    Lexical (BM25) vectors with integer term IDs.
    """
    fields = [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("nnz", pa.int32(), nullable=False),
        pa.field("indices", pa.list_(pa.int32()), nullable=False),
        pa.field("weights", pa.list_(pa.float32()), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
    return pa.schema(fields)


def lexical_schema_terms() -> pa.Schema:
    """
    Lexical (BM25) vectors with string terms (dictionary-encodable).
    """
    fields = [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("nnz", pa.int32(), nullable=False),
        pa.field("terms", pa.list_(pa.string()), nullable=False),
        pa.field("weights", pa.list_(pa.float32()), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
    return pa.schema(fields)


# ============================================================
# Parquet footer contract (key-value metadata)
# ============================================================

# Regexes & enums
SEMVER_RE = re.compile(r"^docparse/(chunks|vectors/(dense|sparse|lexical))/[0-9]+\.[0-9]+\.[0-9]+$")
ISO_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
FAMILY_ENUM = {"dense", "sparse", "lexical"}
DTYPE_ENUM = {"float32"}
LEX_REPR_ENUM = {"indices", "terms"}

# Common required footer keys present in ALL Parquet files we write
FOOTER_REQ_COMMON = (
    "docparse.schema_version",
    "docparse.cfg_hash",
    "docparse.created_by",
    "docparse.created_at",
)

# Required in *vector* families
FOOTER_REQ_VECTORS = (
    "docparse.family",
    "docparse.provider",
    "docparse.model_id",
    "docparse.dtype",
)

# Dense-only required keys
FOOTER_REQ_DENSE = ("docparse.dim",)

# Lexical-specific keys
FOOTER_REQ_LEXICAL = (
    "docparse.lexical.representation",
    "docparse.bm25.k1",
    "docparse.bm25.b",
    "docparse.stopwords_policy",
    "docparse.min_df",
    "docparse.max_df_ratio",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_UTC)


def _to_bytes_meta(meta: Mapping[str, str]) -> Mapping[str, bytes]:
    """Parquet schema metadata requires byte values."""
    return {k: (v if isinstance(v, bytes) else str(v).encode("utf-8")) for k, v in meta.items()}


def build_footer_common(
    schema_version: str,
    cfg_hash: str,
    created_by: str,
    created_at: Optional[str] = None,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    created_at = created_at or _utc_now_iso()
    base = {
        "docparse.schema_version": schema_version,
        "docparse.cfg_hash": cfg_hash,
        "docparse.created_by": created_by,
        "docparse.created_at": created_at,
    }
    if extra:
        base.update(extra)
    return base


def build_footer_dense(
    provider: str,
    model_id: str,
    dim: int,
    cfg_hash: str,
    dtype: str = "float32",
    device: Optional[str] = None,
    created_by: str = "DocsToKG-DocParsing",
    created_at: Optional[str] = None,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    meta = build_footer_common(SCHEMA_VERSION_DENSE, cfg_hash, created_by, created_at, extra)
    meta.update(
        {
            "docparse.family": "dense",
            "docparse.provider": provider,
            "docparse.model_id": model_id,
            "docparse.dim": str(int(dim)),
            "docparse.dtype": dtype,
        }
    )
    if device:
        meta["docparse.device"] = device
    return meta


def build_footer_sparse(
    provider: str,
    model_id: str,
    cfg_hash: str,
    vocab_id: Optional[str] = None,
    hash_scheme: Optional[str] = None,
    dtype: str = "float32",
    created_by: str = "DocsToKG-DocParsing",
    created_at: Optional[str] = None,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    meta = build_footer_common(SCHEMA_VERSION_SPARSE, cfg_hash, created_by, created_at, extra)
    meta.update(
        {
            "docparse.family": "sparse",
            "docparse.provider": provider,
            "docparse.model_id": model_id,
            "docparse.dtype": dtype,
        }
    )
    if vocab_id:
        meta["docparse.sparse.vocab_id"] = vocab_id
    if hash_scheme:
        meta["docparse.sparse.hash_scheme"] = hash_scheme
    return meta


def build_footer_lexical(
    representation: str,
    tokenizer_id: str,
    k1: float,
    b: float,
    stopwords_policy: str,
    min_df: int,
    max_df_ratio: float,
    cfg_hash: str,
    provider: str = "lexical.local_bm25",
    model_id: str = "bm25",
    dtype: str = "float32",
    created_by: str = "DocsToKG-DocParsing",
    created_at: Optional[str] = None,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    meta = build_footer_common(SCHEMA_VERSION_LEXICAL, cfg_hash, created_by, created_at, extra)
    meta.update(
        {
            "docparse.family": "lexical",
            "docparse.provider": provider,
            "docparse.model_id": model_id,
            "docparse.dtype": dtype,
            "docparse.lexical.representation": representation,
            "docparse.tokenizer_id": tokenizer_id,
            "docparse.bm25.k1": str(float(k1)),
            "docparse.bm25.b": str(float(b)),
            "docparse.stopwords_policy": stopwords_policy,
            "docparse.min_df": str(int(min_df)),
            "docparse.max_df_ratio": str(float(max_df_ratio)),
        }
    )
    return meta


# ============================================================
# Validators (metadata + quick table checks)
# ============================================================


@dataclass(frozen=True)
class FooterValidationResult:
    ok: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()


def _decode_file_metadata(md: Optional[Mapping[bytes, bytes]]) -> Dict[str, str]:
    if not md:
        return {}
    return {k.decode("utf-8", "replace"): v.decode("utf-8", "replace") for k, v in md.items()}


def validate_footer_common(meta: Mapping[str, str]) -> FooterValidationResult:
    errs = []
    warns = []

    # Required presence
    for k in FOOTER_REQ_COMMON:
        if k not in meta:
            errs.append(f"Missing required footer key: {k}")

    # Shapes/patterns
    sv = meta.get("docparse.schema_version")
    if sv and not SEMVER_RE.match(sv):
        errs.append(f"Invalid schema_version '{sv}' (expect 'docparse/.../x.y.z').")

    ts = meta.get("docparse.created_at")
    if ts and not ISO_DT_RE.match(ts):
        warns.append(f"created_at '{ts}' is not strict Zulu format ({ISO_UTC}); continuing.")

    return FooterValidationResult(ok=not errs, errors=tuple(errs), warnings=tuple(warns))


def validate_footer_vectors(meta: Mapping[str, str]) -> FooterValidationResult:
    """Apply vectors-level constraints."""
    base = validate_footer_common(meta)
    errs = list(base.errors)
    warns = list(base.warnings)

    for k in FOOTER_REQ_VECTORS:
        if k not in meta:
            errs.append(f"Missing required footer key: {k}")

    fam = meta.get("docparse.family")
    if fam and fam not in FAMILY_ENUM:
        errs.append(f"Invalid family '{fam}'. Valid: {sorted(FAMILY_ENUM)}")

    dt = meta.get("docparse.dtype")
    if dt and dt not in DTYPE_ENUM:
        warns.append(f"dtype '{dt}' not in recommended set {sorted(DTYPE_ENUM)}")

    return FooterValidationResult(ok=not errs, errors=tuple(errs), warnings=tuple(warns))


def validate_footer_dense(meta: Mapping[str, str]) -> FooterValidationResult:
    v = validate_footer_vectors(meta)
    errs = list(v.errors)
    warns = list(v.warnings)

    for k in FOOTER_REQ_DENSE:
        if k not in meta:
            errs.append(f"Missing dense footer key: {k}")

    # dim must be > 0 int
    try:
        if int(meta.get("docparse.dim", "0")) <= 0:
            errs.append("docparse.dim must be > 0")
    except ValueError:
        errs.append("docparse.dim must be an integer")

    return FooterValidationResult(ok=not errs, errors=tuple(errs), warnings=tuple(warns))


def validate_footer_sparse(meta: Mapping[str, str]) -> FooterValidationResult:
    v = validate_footer_vectors(meta)
    errs = list(v.errors)
    warns = list(v.warnings)

    # At least one of vocab_id/hash_scheme should be present
    if not any(k in meta for k in ("docparse.sparse.vocab_id", "docparse.sparse.hash_scheme")):
        warns.append("Sparse footer has neither vocab_id nor hash_scheme – add one for provenance.")

    return FooterValidationResult(ok=not errs, errors=tuple(errs), warnings=tuple(warns))


def validate_footer_lexical(meta: Mapping[str, str]) -> FooterValidationResult:
    v = validate_footer_vectors(meta)
    errs = list(v.errors)
    warns = list(v.warnings)

    # Required lexical keys
    for k in FOOTER_REQ_LEXICAL:
        if k not in meta:
            errs.append(f"Missing lexical footer key: {k}")

    rep = meta.get("docparse.lexical.representation")
    if rep and rep not in LEX_REPR_ENUM:
        errs.append(f"Invalid lexical.representation '{rep}' (expected {sorted(LEX_REPR_ENUM)})")

    # Numeric plausibility
    try:
        if float(meta.get("docparse.bm25.k1", "0")) <= 0:
            errs.append("bm25.k1 must be > 0")
    except ValueError:
        errs.append("bm25.k1 must be numeric")

    try:
        b = float(meta.get("docparse.bm25.b", "-1"))
        if not (0.0 <= b <= 1.0):
            errs.append("bm25.b must be in [0, 1]")
    except ValueError:
        errs.append("bm25.b must be numeric")

    try:
        if int(meta.get("docparse.min_df", "0")) < 0:
            errs.append("min_df must be >= 0")
    except ValueError:
        errs.append("min_df must be integer")

    try:
        r = float(meta.get("docparse.max_df_ratio", "0"))
        if not (0.0 < r <= 1.0):
            errs.append("max_df_ratio must be in (0, 1]")
    except ValueError:
        errs.append("max_df_ratio must be numeric")

    return FooterValidationResult(ok=not errs, errors=tuple(errs), warnings=tuple(warns))


# ============================================================
# Table metadata attach / read / file validate
# ============================================================


def attach_footer_metadata(table: pa.Table, meta: Mapping[str, str]) -> pa.Table:
    """
    Return a new table whose schema carries Parquet-compatible key-value metadata.
    """
    existing = table.schema.metadata or {}
    merged = dict(existing)
    for k, v in _to_bytes_meta(meta).items():
        merged[k] = v
    return table.replace_schema_metadata(merged)


def read_parquet_footer(path: str) -> Dict[str, str]:
    """
    Read a Parquet file's key_value_metadata as str->str.
    """
    pf = pq.ParquetFile(path)
    return _decode_file_metadata(pf.metadata.metadata)


def validate_parquet_file(path: str, family: Optional[str] = None) -> FooterValidationResult:
    """
    Validate a Parquet file against the footer contract.
    If family is None, infer from footer key 'docparse.family' when present.
    """
    meta = read_parquet_footer(path)
    if not meta:
        return FooterValidationResult(ok=False, errors=("Parquet file has no key_value_metadata.",))

    # Common
    common = validate_footer_common(meta)
    errs = list(common.errors)
    warns = list(common.warnings)
    if errs:
        return FooterValidationResult(ok=False, errors=tuple(errs), warnings=tuple(warns))

    fam = family or meta.get("docparse.family")
    if fam is None:
        # Chunks dataset has no family; treat as non-vectors
        return FooterValidationResult(ok=True, errors=(), warnings=tuple(warns))

    fam = str(fam)
    if fam == "dense":
        v = validate_footer_dense(meta)
    elif fam == "sparse":
        v = validate_footer_sparse(meta)
    elif fam == "lexical":
        v = validate_footer_lexical(meta)
    else:
        return FooterValidationResult(
            ok=False, errors=(f"Unknown family '{fam}'",), warnings=tuple(warns)
        )

    errs.extend(v.errors)
    warns.extend(v.warnings)
    return FooterValidationResult(ok=(len(errs) == 0), errors=tuple(errs), warnings=tuple(warns))


# ============================================================
# Quick table validators (structure only; not scanning all rows)
# ============================================================


def assert_table_matches_schema(table: pa.Table, expected: pa.Schema) -> None:
    """
    Raise ValueError if the table's visible fields don't match the expected schema
    (names and types). Allows extra columns if present, but they must not conflict.
    """
    actual = table.schema
    # Check required fields exist with identical types
    for f in expected:
        try:
            f2 = actual.field(f.name)
        except KeyError:
            raise ValueError(f"Missing required column: {f.name}")
        if f2.type != f.type:
            raise ValueError(f"Column {f.name} has type {f2.type}, expected {f.type}")


# ============================================================
# Writer hints (for your Parquet writer implementation)
# ============================================================


def recommended_parquet_writer_options(dataset: str) -> Dict[str, object]:
    """
    Returns a dictionary of recommended write options per dataset type.
    You can map these into your actual writer (e.g., pyarrow.parquet.write_table).
    NOTE: 'use_byte_stream_split' availability depends on pyarrow version.
    """
    if dataset == "chunks":
        return {
            "compression": "zstd",
            "compression_level": 5,
            "use_dictionary": {"section": True, "text": False},
            "write_statistics": True,
        }
    elif dataset == "dense":
        return {
            "compression": "zstd",
            "compression_level": 5,
            "use_dictionary": {},
            "write_statistics": True,
        }
    elif dataset in ("sparse", "lexical"):
        return {
            "compression": "zstd",
            "compression_level": 5,
            "use_dictionary": {},
            "write_statistics": True,
        }
    else:
        raise ValueError("dataset must be one of: 'chunks', 'dense', 'sparse', 'lexical'")


# ============================================================
# Legacy Vector Schema (for backward compatibility)
# ============================================================


def _legacy_vector_schema() -> pa.Schema:
    """
    Arrow schema for legacy vector rows (all families in one row).

    This maintains compatibility with existing JSONL vector export format
    when transitioning to Parquet. Contains dense (Qwen), sparse (SPLADE),
    and lexical (BM25) vectors in a single row.

    Returns:
        PyArrow schema matching the legacy vector row structure.
    """
    string_list = pa.list_(pa.string())
    float_list = pa.list_(pa.float32())

    return pa.schema(
        [
            pa.field("UUID", pa.string(), nullable=False),
            pa.field(
                "BM25",
                pa.struct(
                    [
                        pa.field("terms", string_list, nullable=True),
                        pa.field("weights", float_list, nullable=True),
                        pa.field("avgdl", pa.float64(), nullable=True),
                        pa.field("N", pa.int64(), nullable=True),
                    ]
                ),
                nullable=False,
            ),
            pa.field(
                "SPLADEv3",
                pa.struct(
                    [
                        pa.field("tokens", string_list, nullable=True),
                        pa.field("weights", float_list, nullable=True),
                    ]
                ),
                nullable=False,
            ),
            pa.field(
                "Qwen3-4B",
                pa.struct(
                    [
                        pa.field("model_id", pa.string(), nullable=False),
                        pa.field("vector", float_list, nullable=False),
                        pa.field("dimension", pa.int32(), nullable=True),
                    ]
                ),
                nullable=False,
            ),
            pa.field("model_metadata", pa.string(), nullable=True),
            pa.field("schema_version", pa.string(), nullable=False),
        ]
    )
