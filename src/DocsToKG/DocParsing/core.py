# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core",
#   "purpose": "Shared utilities and CLI glue for DocParsing workflows",
#   "sections": [
#     {
#       "id": "dedupe-preserve-order",
#       "name": "dedupe_preserve_order",
#       "anchor": "function-dedupe-preserve-order",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-str-sequence",
#       "name": "_ensure_str_sequence",
#       "anchor": "function-ensure-str-sequence",
#       "kind": "function"
#     },
#     {
#       "id": "load-yaml-markers",
#       "name": "_load_yaml_markers",
#       "anchor": "function-load-yaml-markers",
#       "kind": "function"
#     },
#     {
#       "id": "load-toml-markers",
#       "name": "_load_toml_markers",
#       "anchor": "function-load-toml-markers",
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
#       "id": "clioption",
#       "name": "CLIOption",
#       "anchor": "class-clioption",
#       "kind": "class"
#     },
#     {
#       "id": "build-subcommand",
#       "name": "build_subcommand",
#       "anchor": "function-build-subcommand",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-path",
#       "name": "_coerce_path",
#       "anchor": "function-coerce-path",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-optional-path",
#       "name": "_coerce_optional_path",
#       "anchor": "function-coerce-optional-path",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-bool",
#       "name": "_coerce_bool",
#       "anchor": "function-coerce-bool",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-int",
#       "name": "_coerce_int",
#       "anchor": "function-coerce-int",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-float",
#       "name": "_coerce_float",
#       "anchor": "function-coerce-float",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-str",
#       "name": "_coerce_str",
#       "anchor": "function-coerce-str",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-str-tuple",
#       "name": "_coerce_str_tuple",
#       "anchor": "function-coerce-str-tuple",
#       "kind": "function"
#     },
#     {
#       "id": "manifest-value",
#       "name": "_manifest_value",
#       "anchor": "function-manifest-value",
#       "kind": "function"
#     },
#     {
#       "id": "load-config-mapping",
#       "name": "load_config_mapping",
#       "anchor": "function-load-config-mapping",
#       "kind": "function"
#     },
#     {
#       "id": "stageconfigbase",
#       "name": "StageConfigBase",
#       "anchor": "class-stageconfigbase",
#       "kind": "class"
#     },
#     {
#       "id": "bm25stats",
#       "name": "BM25Stats",
#       "anchor": "class-bm25stats",
#       "kind": "class"
#     },
#     {
#       "id": "spladecfg",
#       "name": "SpladeCfg",
#       "anchor": "class-spladecfg",
#       "kind": "class"
#     },
#     {
#       "id": "qwencfg",
#       "name": "QwenCfg",
#       "anchor": "class-qwencfg",
#       "kind": "class"
#     },
#     {
#       "id": "chunkworkerconfig",
#       "name": "ChunkWorkerConfig",
#       "anchor": "class-chunkworkerconfig",
#       "kind": "class"
#     },
#     {
#       "id": "chunktask",
#       "name": "ChunkTask",
#       "anchor": "class-chunktask",
#       "kind": "class"
#     },
#     {
#       "id": "chunkresult",
#       "name": "ChunkResult",
#       "anchor": "class-chunkresult",
#       "kind": "class"
#     },
#     {
#       "id": "expand-path",
#       "name": "expand_path",
#       "anchor": "function-expand-path",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-hf-home",
#       "name": "resolve_hf_home",
#       "anchor": "function-resolve-hf-home",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-model-root",
#       "name": "resolve_model_root",
#       "anchor": "function-resolve-model-root",
#       "kind": "function"
#     },
#     {
#       "id": "looks-like-filesystem-path",
#       "name": "looks_like_filesystem_path",
#       "anchor": "function-looks-like-filesystem-path",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-pdf-model-path",
#       "name": "resolve_pdf_model_path",
#       "anchor": "function-resolve-pdf-model-path",
#       "kind": "function"
#     },
#     {
#       "id": "init-hf-env",
#       "name": "init_hf_env",
#       "anchor": "function-init-hf-env",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-model-environment",
#       "name": "ensure_model_environment",
#       "anchor": "function-ensure-model-environment",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-optional-dependency",
#       "name": "_ensure_optional_dependency",
#       "anchor": "function-ensure-optional-dependency",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-splade-dependencies",
#       "name": "ensure_splade_dependencies",
#       "anchor": "function-ensure-splade-dependencies",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-qwen-dependencies",
#       "name": "ensure_qwen_dependencies",
#       "anchor": "function-ensure-qwen-dependencies",
#       "kind": "function"
#     },
#     {
#       "id": "detect-data-root",
#       "name": "detect_data_root",
#       "anchor": "function-detect-data-root",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-dir",
#       "name": "_ensure_dir",
#       "anchor": "function-ensure-dir",
#       "kind": "function"
#     },
#     {
#       "id": "data-doctags",
#       "name": "data_doctags",
#       "anchor": "function-data-doctags",
#       "kind": "function"
#     },
#     {
#       "id": "data-chunks",
#       "name": "data_chunks",
#       "anchor": "function-data-chunks",
#       "kind": "function"
#     },
#     {
#       "id": "data-vectors",
#       "name": "data_vectors",
#       "anchor": "function-data-vectors",
#       "kind": "function"
#     },
#     {
#       "id": "data-manifests",
#       "name": "data_manifests",
#       "anchor": "function-data-manifests",
#       "kind": "function"
#     },
#     {
#       "id": "prepare-data-root",
#       "name": "prepare_data_root",
#       "anchor": "function-prepare-data-root",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-pipeline-path",
#       "name": "resolve_pipeline_path",
#       "anchor": "function-resolve-pipeline-path",
#       "kind": "function"
#     },
#     {
#       "id": "data-pdfs",
#       "name": "data_pdfs",
#       "anchor": "function-data-pdfs",
#       "kind": "function"
#     },
#     {
#       "id": "data-html",
#       "name": "data_html",
#       "anchor": "function-data-html",
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
#       "id": "should-skip-output",
#       "name": "should_skip_output",
#       "anchor": "function-should-skip-output",
#       "kind": "function"
#     },
#     {
#       "id": "stringify-path",
#       "name": "_stringify_path",
#       "anchor": "function-stringify-path",
#       "kind": "function"
#     },
#     {
#       "id": "manifest-log-skip",
#       "name": "manifest_log_skip",
#       "anchor": "function-manifest-log-skip",
#       "kind": "function"
#     },
#     {
#       "id": "manifest-log-success",
#       "name": "manifest_log_success",
#       "anchor": "function-manifest-log-success",
#       "kind": "function"
#     },
#     {
#       "id": "manifest-log-failure",
#       "name": "manifest_log_failure",
#       "anchor": "function-manifest-log-failure",
#       "kind": "function"
#     },
#     {
#       "id": "get-logger",
#       "name": "get_logger",
#       "anchor": "function-get-logger",
#       "kind": "function"
#     },
#     {
#       "id": "log-event",
#       "name": "log_event",
#       "anchor": "function-log-event",
#       "kind": "function"
#     },
#     {
#       "id": "find-free-port",
#       "name": "find_free_port",
#       "anchor": "function-find-free-port",
#       "kind": "function"
#     },
#     {
#       "id": "atomic-write",
#       "name": "atomic_write",
#       "anchor": "function-atomic-write",
#       "kind": "function"
#     },
#     {
#       "id": "iter-doctags",
#       "name": "iter_doctags",
#       "anchor": "function-iter-doctags",
#       "kind": "function"
#     },
#     {
#       "id": "iter-chunks",
#       "name": "iter_chunks",
#       "anchor": "function-iter-chunks",
#       "kind": "function"
#     },
#     {
#       "id": "jsonl-load",
#       "name": "jsonl_load",
#       "anchor": "function-jsonl-load",
#       "kind": "function"
#     },
#     {
#       "id": "jsonl-save",
#       "name": "jsonl_save",
#       "anchor": "function-jsonl-save",
#       "kind": "function"
#     },
#     {
#       "id": "jsonl-append-iter",
#       "name": "jsonl_append_iter",
#       "anchor": "function-jsonl-append-iter",
#       "kind": "function"
#     },
#     {
#       "id": "batcher",
#       "name": "Batcher",
#       "anchor": "class-batcher",
#       "kind": "class"
#     },
#     {
#       "id": "manifest-filename",
#       "name": "_manifest_filename",
#       "anchor": "function-manifest-filename",
#       "kind": "function"
#     },
#     {
#       "id": "manifest-append",
#       "name": "manifest_append",
#       "anchor": "function-manifest-append",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-hash-algorithm",
#       "name": "resolve_hash_algorithm",
#       "anchor": "function-resolve-hash-algorithm",
#       "kind": "function"
#     },
#     {
#       "id": "compute-chunk-uuid",
#       "name": "compute_chunk_uuid",
#       "anchor": "function-compute-chunk-uuid",
#       "kind": "function"
#     },
#     {
#       "id": "compute-content-hash",
#       "name": "compute_content_hash",
#       "anchor": "function-compute-content-hash",
#       "kind": "function"
#     },
#     {
#       "id": "load-manifest-index",
#       "name": "load_manifest_index",
#       "anchor": "function-load-manifest-index",
#       "kind": "function"
#     },
#     {
#       "id": "acquire-lock",
#       "name": "acquire_lock",
#       "anchor": "function-acquire-lock",
#       "kind": "function"
#     },
#     {
#       "id": "pid-is-running",
#       "name": "_pid_is_running",
#       "anchor": "function-pid-is-running",
#       "kind": "function"
#     },
#     {
#       "id": "set-spawn-or-warn",
#       "name": "set_spawn_or_warn",
#       "anchor": "function-set-spawn-or-warn",
#       "kind": "function"
#     },
#     {
#       "id": "run-chunk",
#       "name": "_run_chunk",
#       "anchor": "function-run-chunk",
#       "kind": "function"
#     },
#     {
#       "id": "run-embed",
#       "name": "_run_embed",
#       "anchor": "function-run-embed",
#       "kind": "function"
#     },
#     {
#       "id": "build-doctags-parser",
#       "name": "_build_doctags_parser",
#       "anchor": "function-build-doctags-parser",
#       "kind": "function"
#     },
#     {
#       "id": "scan-pdf-html",
#       "name": "_scan_pdf_html",
#       "anchor": "function-scan-pdf-html",
#       "kind": "function"
#     },
#     {
#       "id": "directory-contains-suffixes",
#       "name": "_directory_contains_suffixes",
#       "anchor": "function-directory-contains-suffixes",
#       "kind": "function"
#     },
#     {
#       "id": "detect-mode",
#       "name": "_detect_mode",
#       "anchor": "function-detect-mode",
#       "kind": "function"
#     },
#     {
#       "id": "merge-args",
#       "name": "_merge_args",
#       "anchor": "function-merge-args",
#       "kind": "function"
#     },
#     {
#       "id": "run-doctags",
#       "name": "_run_doctags",
#       "anchor": "function-run-doctags",
#       "kind": "function"
#     },
#     {
#       "id": "preview-list",
#       "name": "_preview_list",
#       "anchor": "function-preview-list",
#       "kind": "function"
#     },
#     {
#       "id": "plan-doctags",
#       "name": "_plan_doctags",
#       "anchor": "function-plan-doctags",
#       "kind": "function"
#     },
#     {
#       "id": "plan-chunk",
#       "name": "_plan_chunk",
#       "anchor": "function-plan-chunk",
#       "kind": "function"
#     },
#     {
#       "id": "plan-embed",
#       "name": "_plan_embed",
#       "anchor": "function-plan-embed",
#       "kind": "function"
#     },
#     {
#       "id": "display-plan",
#       "name": "_display_plan",
#       "anchor": "function-display-plan",
#       "kind": "function"
#     },
#     {
#       "id": "run-all",
#       "name": "_run_all",
#       "anchor": "function-run-all",
#       "kind": "function"
#     },
#     {
#       "id": "command",
#       "name": "_Command",
#       "anchor": "class-command",
#       "kind": "class"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     },
#     {
#       "id": "run-all",
#       "name": "run_all",
#       "anchor": "function-run-all",
#       "kind": "function"
#     },
#     {
#       "id": "chunk",
#       "name": "chunk",
#       "anchor": "function-chunk",
#       "kind": "function"
#     },
#     {
#       "id": "embed",
#       "name": "embed",
#       "anchor": "function-embed",
#       "kind": "function"
#     },
#     {
#       "id": "doctags",
#       "name": "doctags",
#       "anchor": "function-doctags",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
DocParsing Core Utilities

This module centralises lightweight helpers that power multiple DocParsing
pipeline stages. Utilities span path discovery, atomic file writes, JSONL
parsing, manifest bookkeeping, CLI glue, and structured logging so that
chunking, embedding, and conversion scripts can share consistent behaviour
without an additional dependency layer.

Key Features:
- Resolve DocsToKG data directories with environment and ancestor discovery
- Stream JSONL inputs and outputs with validation and error tolerance
- Emit structured JSON logs suited for machine ingestion and dashboards
- Manage pipeline manifests, batching helpers, and advisory file locks

Usage:
    from DocsToKG.DocParsing import core

    chunks_dir = core.data_chunks()
    with core.atomic_write(chunks_dir / \"example.jsonl\") as handle:
        handle.write(\"{}\")

Dependencies:
- json, pathlib, logging: Provide standard I/O and diagnostics primitives.
- typing: Supply type hints consumed by Sphinx documentation tooling and API generators.
- pydantic (optional): Some helpers integrate with schema validation routines.

All helpers are safe to import in multiprocessing contexts and avoid heavy
third-party dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import logging
import math
import os
import shutil
import socket
import time
import unicodedata
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    TypeVar,
)

T = TypeVar("T")

# Default structural marker configuration shared across stages.
DEFAULT_HEADING_MARKERS: Tuple[str, ...] = ("#",)
DEFAULT_CAPTION_MARKERS: Tuple[str, ...] = (
    "Figure caption:",
    "Table:",
    "Picture description:",
    "<!-- image -->",
)


def dedupe_preserve_order(markers: Sequence[str]) -> Tuple[str, ...]:
    """Return ``markers`` without duplicates while preserving input order."""

    seen: set[str] = set()
    ordered: List[str] = []
    for marker in markers:
        if not marker:
            continue
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(marker)
    return tuple(ordered)


def _ensure_str_sequence(value: object, label: str) -> List[str]:
    """Normalise structural marker entries into string lists."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a list of strings for '{label}'")
    return [item for item in value if item]


def _load_yaml_markers(raw: str) -> object:
    """Deserialize YAML marker overrides, raising when PyYAML is unavailable."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Loading YAML structural markers requires the 'PyYAML' package (pip install PyYAML)."
        ) from exc
    return yaml.safe_load(raw)


def _load_toml_markers(raw: str) -> object:
    """Deserialize TOML marker definitions with compatibility fallbacks."""
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - fallback path
        try:
            import tomli as tomllib  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Loading TOML structural markers requires the 'tomli' package (pip install tomli)."
            ) from exc
    return tomllib.loads(raw)


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
                data = _load_yaml_markers(raw)
            elif parser == "toml":
                data = _load_toml_markers(raw)
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


@dataclass(frozen=True)
class CLIOption:
    """Declarative CLI argument specification used by ``build_subcommand``."""

    flags: Tuple[str, ...]
    kwargs: Dict[str, Any]


def build_subcommand(
    parser: argparse.ArgumentParser, options: Sequence[CLIOption]
) -> argparse.ArgumentParser:
    """Attach CLI options described by ``options`` to ``parser``."""

    for option in options:
        parser.add_argument(*option.flags, **option.kwargs)
    return parser


def _coerce_path(value: object, base_dir: Optional[Path] = None) -> Path:
    """Convert ``value`` into an absolute :class:`Path`."""

    if isinstance(value, Path):
        path = value
    else:
        path = Path(str(value))
    if base_dir is not None and not path.is_absolute():
        path = (base_dir / path).expanduser()
    else:
        path = path.expanduser()
    return path.resolve()


def _coerce_optional_path(value: object, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Convert optional path-like values."""

    if value in (None, "", False):
        return None
    return _coerce_path(value, base_dir)


def _coerce_bool(value: object, _base_dir: Optional[Path] = None) -> bool:
    """Convert truthy strings or numbers to boolean."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(normalized)


def _coerce_int(value: object, _base_dir: Optional[Path] = None) -> int:
    """Convert ``value`` to ``int``."""

    if isinstance(value, int):
        return value
    return int(str(value).strip())


def _coerce_float(value: object, _base_dir: Optional[Path] = None) -> float:
    """Convert ``value`` to ``float``."""

    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    return float(str(value).strip())


def _coerce_str(value: object, _base_dir: Optional[Path] = None) -> str:
    """Return ``value`` coerced to string."""

    return str(value)


def _coerce_str_tuple(value: object, _base_dir: Optional[Path] = None) -> Tuple[str, ...]:
    """Return ``value`` as a tuple of strings."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        flattened: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (list, tuple, set)):
                for sub_item in item:
                    if sub_item is None:
                        continue
                    text = str(sub_item).strip()
                    if text:
                        flattened.append(text)
            else:
                text = str(item).strip()
                if text:
                    flattened.append(text)
        return tuple(flattened)
    text = str(value).strip()
    if not text:
        return ()
    # Support JSON-style arrays for convenience
    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("(") and text.endswith(")")
    ):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, tuple, set)):
                return tuple(str(item) for item in parsed)
        except json.JSONDecodeError:
            pass
    separators = [",", ";"]
    parts = [text]
    for sep in separators:
        if sep in text:
            parts = [segment.strip() for segment in text.split(sep)]
            break
    return tuple(part for part in parts if part)


def _manifest_value(value: Any) -> Any:
    """Convert values to manifest-friendly representations."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_manifest_value(item) for item in value]
    if isinstance(value, list):
        return [_manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _manifest_value(val) for key, val in value.items()}
    return value


def load_config_mapping(path: Path) -> Dict[str, Any]:
    """Load a configuration mapping from JSON, YAML, or TOML."""

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml_markers(raw)
    elif suffix == ".toml":
        data = _load_toml_markers(raw)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"Stage configuration file {path} must contain an object; received {type(data).__name__}."
        )
    return data


@dataclass
class StageConfigBase:
    """Base dataclass for stage configuration objects."""

    config: Optional[Path] = None
    overrides: Set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
        metadata={"manifest": False},
    )

    ENV_VARS: ClassVar[Dict[str, str]] = {}
    FIELD_PARSERS: ClassVar[Dict[str, Callable[[Any, Optional[Path]], Any]]] = {}

    def apply_env(self) -> None:
        """Overlay configuration from environment variables."""

        for field_name, env_name in self.ENV_VARS.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            new_value = self._coerce_field(field_name, raw, None)
            current = getattr(self, field_name, None)
            if new_value == current:
                continue
            setattr(self, field_name, new_value)
            self.overrides.add(field_name)

    def update_from_file(self, cfg_path: Path) -> None:
        """Overlay configuration from ``cfg_path``."""

        mapping = load_config_mapping(cfg_path)
        base_dir = cfg_path.parent
        for key, value in mapping.items():
            if not hasattr(self, key):
                continue
            new_value = self._coerce_field(key, value, base_dir)
            current = getattr(self, key, None)
            if new_value == current:
                continue
            setattr(self, key, new_value)
            self.overrides.add(key)
        self.config = _coerce_optional_path(cfg_path, None)
        self.overrides.add("config")

    def apply_args(self, args: Any) -> None:
        """Overlay configuration from an argparse namespace."""

        for field_def in fields(self):
            name = field_def.name
            if not hasattr(args, name):
                continue
            value = getattr(args, name)
            if value is None:
                continue
            new_value = self._coerce_field(name, value, None)
            current = getattr(self, name, None)
            if new_value == current:
                continue
            setattr(self, name, new_value)
            self.overrides.add(name)

    @classmethod
    def from_env(cls) -> "StageConfigBase":
        """Instantiate a configuration populated solely from environment variables."""
        cfg = cls()
        cfg.apply_env()
        cfg.finalize()
        return cfg

    def finalize(self) -> None:  # pragma: no cover - overridden by subclasses
        """Hook allowing subclasses to normalise derived fields."""

    def to_manifest(self) -> Dict[str, Any]:
        """Return a manifest-friendly snapshot of the configuration."""

        payload: Dict[str, Any] = {}
        for field_def in fields(self):
            if not field_def.metadata.get("manifest", True):
                continue
            payload[field_def.name] = _manifest_value(getattr(self, field_def.name))
        return payload

    def _coerce_field(self, name: str, value: Any, base_dir: Optional[Path]) -> Any:
        """Run field-specific coercion logic before manifest serialization."""
        parser = self.FIELD_PARSERS.get(name)
        if parser is None:
            return value
        return parser(value, base_dir)

    def is_overridden(self, field_name: str) -> bool:
        """Return ``True`` when ``field_name`` was explicitly overridden."""

        return field_name in self.overrides

    _coerce_path = staticmethod(_coerce_path)
    _coerce_optional_path = staticmethod(_coerce_optional_path)
    _coerce_bool = staticmethod(_coerce_bool)
    _coerce_int = staticmethod(_coerce_int)
    _coerce_float = staticmethod(_coerce_float)
    _coerce_str = staticmethod(_coerce_str)
    _coerce_str_tuple = staticmethod(_coerce_str_tuple)


# --- Globals ---

__all__ = [
    "detect_data_root",
    "data_doctags",
    "data_chunks",
    "data_vectors",
    "data_manifests",
    "data_pdfs",
    "data_html",
    "expand_path",
    "resolve_hf_home",
    "resolve_model_root",
    "find_free_port",
    "atomic_write",
    "iter_doctags",
    "iter_chunks",
    "jsonl_load",
    "jsonl_save",
    "jsonl_append_iter",
    "build_jsonl_split_map",
    "get_logger",
    "log_event",
    "Batcher",
    "compute_chunk_uuid",
    "quarantine_artifact",
    "manifest_append",
    "manifest_log_failure",
    "manifest_log_skip",
    "manifest_log_success",
    "compute_content_hash",
    "resolve_hash_algorithm",
    "load_manifest_index",
    "acquire_lock",
    "set_spawn_or_warn",
    "derive_doc_id_and_vectors_path",
    "derive_doc_id_and_doctags_path",
    "derive_doc_id_and_chunks_path",
    "compute_relative_doc_id",
    "compute_stable_shard",
    "should_skip_output",
    "relative_path",
    "init_hf_env",
    "ensure_model_environment",
    "ensure_splade_dependencies",
    "ensure_qwen_dependencies",
    "UUID_NAMESPACE",
    "BM25Stats",
    "SpladeCfg",
    "QwenCfg",
    "ChunkWorkerConfig",
    "ChunkTask",
    "ChunkResult",
    "DEFAULT_HEADING_MARKERS",
    "DEFAULT_CAPTION_MARKERS",
    "DEFAULT_SERIALIZER_PROVIDER",
    "DEFAULT_TOKENIZER",
    "dedupe_preserve_order",
    "CLIOption",
    "build_subcommand",
    "looks_like_filesystem_path",
    "resolve_pdf_model_path",
    "prepare_data_root",
    "resolve_pipeline_path",
    "load_structural_marker_profile",
    "load_structural_marker_config",
    "CommandHandler",
    "CLI_DESCRIPTION",
    "main",
    "run_all",
    "chunk",
    "embed",
    "doctags",
]

# --- Data Containers ---

UUID_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")
DEFAULT_SERIALIZER_PROVIDER = "DocsToKG.DocParsing.formats:RichSerializerProvider"
DEFAULT_TOKENIZER = "Qwen/Qwen3-Embedding-4B"
PDF_MODEL_SUBDIR = Path("granite-docling-258M")


@dataclass(slots=True)
class BM25Stats:
    """Corpus-level statistics required for BM25 weighting."""

    N: int
    avgdl: float
    df: Dict[str, int]


@dataclass(slots=True)
class SpladeCfg:
    """Runtime configuration for SPLADE sparse encoding."""

    model_dir: Path
    device: str = "cuda"
    batch_size: int = 32
    cache_folder: Optional[Path] = None
    max_active_dims: Optional[int] = None
    attn_impl: Optional[str] = None
    local_files_only: bool = True


@dataclass(slots=True)
class QwenCfg:
    """Configuration for generating dense embeddings with Qwen via vLLM."""

    model_dir: Path
    dtype: str = "bfloat16"
    tp: int = 1
    gpu_mem_util: float = 0.60
    batch_size: int = 32
    quantization: Optional[str] = None
    dim: int = 2560
    cache_enabled: bool = True


@dataclass(slots=True)
class ChunkWorkerConfig:
    """Lightweight configuration shared across chunker worker processes."""

    tokenizer_model: str
    min_tokens: int
    max_tokens: int
    soft_barrier_margin: int
    heading_markers: Tuple[str, ...]
    caption_markers: Tuple[str, ...]
    docling_version: str
    serializer_provider_spec: str = DEFAULT_SERIALIZER_PROVIDER
    inject_anchors: bool = False


@dataclass(slots=True)
class ChunkTask:
    """Work unit describing a single DocTags file to chunk."""

    doc_path: Path
    output_path: Path
    doc_id: str
    doc_stem: str
    input_hash: str
    parse_engine: str
    sanitizer_profile: Optional[str] = None


@dataclass(slots=True)
class ChunkResult:
    """Result envelope emitted by chunker workers."""

    doc_id: str
    doc_stem: str
    status: str
    duration_s: float
    input_path: Path
    output_path: Path
    input_hash: str
    chunk_count: int
    parse_engine: str
    sanitizer_profile: Optional[str] = None
    anchors_injected: bool = False
    error: Optional[str] = None


# --- Path Resolution ---


def expand_path(path: str | Path) -> Path:
    """Return ``path`` expanded to an absolute :class:`Path`.

    Args:
        path: Candidate filesystem path supplied as string or :class:`Path`.

    Returns:
        Absolute path with user home components resolved.
    """

    return Path(path).expanduser().resolve()


def resolve_hf_home() -> Path:
    """Resolve the HuggingFace cache directory respecting ``HF_HOME``.

    Args:
        None

    Returns:
        Path: Absolute path to the HuggingFace cache directory.
    """

    env = os.getenv("HF_HOME")
    if env:
        return expand_path(env)
    return expand_path(Path.home() / ".cache" / "huggingface")


def resolve_model_root(hf_home: Optional[Path] = None) -> Path:
    """Resolve the DocsToKG model root honoring ``DOCSTOKG_MODEL_ROOT``.

    Args:
        hf_home: Optional HuggingFace cache directory to treat as the base path.

    Returns:
        Path: Absolute directory where DocsToKG models should be stored.
    """

    env = os.getenv("DOCSTOKG_MODEL_ROOT")
    if env:
        return expand_path(env)
    base = hf_home if hf_home is not None else resolve_hf_home()
    return expand_path(base)


def looks_like_filesystem_path(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` appears to reference a local path."""

    expanded = Path(candidate).expanduser()
    drive, _ = os.path.splitdrive(candidate)
    if drive:
        return True
    if expanded.is_absolute() or expanded.exists():
        return True
    prefixes = ["~", "."]
    if os.sep not in prefixes:
        prefixes.append(os.sep)
    alt = os.altsep
    if alt and alt not in prefixes:
        prefixes.append(alt)
    return any(candidate.startswith(prefix) for prefix in prefixes)


def resolve_pdf_model_path(cli_value: str | None = None) -> str:
    """Determine PDF model path using CLI and environment precedence."""

    if cli_value:
        if looks_like_filesystem_path(cli_value):
            return str(expand_path(cli_value))
        return cli_value
    env_model = os.getenv("DOCLING_PDF_MODEL")
    if env_model:
        return str(expand_path(env_model))
    model_root = resolve_model_root()
    return str(expand_path(model_root / PDF_MODEL_SUBDIR))


def init_hf_env(
    hf_home: Optional[Path] = None,
    model_root: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Initialise Hugging Face and transformer cache environment variables.

    Args:
        hf_home: Optional explicit HF cache directory.
        model_root: Optional DocsToKG model root override.

    Returns:
        Tuple of ``(hf_home, model_root)`` paths after normalisation.
    """

    resolved_hf = expand_path(hf_home) if isinstance(hf_home, Path) else resolve_hf_home()
    resolved_model_root = (
        expand_path(model_root) if isinstance(model_root, Path) else resolve_model_root(resolved_hf)
    )

    os.environ["HF_HOME"] = str(resolved_hf)
    os.environ["HF_HUB_CACHE"] = str(resolved_hf / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(resolved_hf / "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(resolved_model_root)
    os.environ["DOCSTOKG_MODEL_ROOT"] = str(resolved_model_root)

    return resolved_hf, resolved_model_root


_MODEL_ENV: Tuple[Path, Path] | None = None


def ensure_model_environment(
    hf_home: Optional[Path] = None, model_root: Optional[Path] = None
) -> Tuple[Path, Path]:
    """Initialise and cache the HuggingFace/model-root environment settings."""

    global _MODEL_ENV
    if _MODEL_ENV is None or hf_home is not None or model_root is not None:
        _MODEL_ENV = init_hf_env(hf_home=hf_home, model_root=model_root)
    return _MODEL_ENV


SPLADE_DEPENDENCY_MESSAGE = (
    "Optional dependency 'sentence-transformers' is required for SPLADE embeddings. "
    "Install it with `pip install sentence-transformers` or disable SPLADE generation."
)
QWEN_DEPENDENCY_MESSAGE = (
    "Optional dependency 'vllm' is required for Qwen dense embeddings. "
    "Install it with `pip install vllm` before running the embedding pipeline."
)


def _ensure_optional_dependency(
    module_name: str, message: str, *, import_error: Exception | None = None
) -> None:
    """Import ``module_name`` or raise with ``message``."""

    try:
        importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - dependency missing
        cause = import_error or exc
        raise ImportError(message) from cause


def ensure_splade_dependencies(import_error: Exception | None = None) -> None:
    """Validate that SPLADE optional dependencies are importable."""

    _ensure_optional_dependency(
        "sentence_transformers", SPLADE_DEPENDENCY_MESSAGE, import_error=import_error
    )


def ensure_qwen_dependencies(import_error: Exception | None = None) -> None:
    """Validate that Qwen/vLLM optional dependencies are importable."""

    _ensure_optional_dependency("vllm", QWEN_DEPENDENCY_MESSAGE, import_error=import_error)


def detect_data_root(start: Optional[Path] = None) -> Path:
    """Locate the DocsToKG Data directory via env var or ancestor scan.

    Checks the ``DOCSTOKG_DATA_ROOT`` environment variable first. If not set,
    scans ancestor directories for a ``Data`` folder containing expected
    subdirectories (``PDFs``, ``HTML``, ``DocTagsFiles``, or
    ``ChunkedDocTagFiles``).

    Args:
        start: Starting directory for the ancestor scan. Defaults to the
            current working directory when ``None``.

    Returns:
        Absolute path to the resolved ``Data`` directory. When
        ``DOCSTOKG_DATA_ROOT`` is set but the directory does not yet exist,
        it is created automatically.

    Examples:
        >>> os.environ["DOCSTOKG_DATA_ROOT"] = "/tmp/data"
        >>> (Path("/tmp/data")).mkdir(parents=True, exist_ok=True)
        >>> detect_data_root()
        PosixPath('/tmp/data')

        >>> os.environ.pop("DOCSTOKG_DATA_ROOT")
        >>> detect_data_root(Path("/workspace/DocsToKG/src"))
        PosixPath('/workspace/DocsToKG/Data')
    """

    env_root = os.getenv("DOCSTOKG_DATA_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if not env_path.exists():
            env_path.mkdir(parents=True, exist_ok=True)
        return env_path

    start_path = Path.cwd() if start is None else Path(start).resolve()
    expected_dirs = ["PDFs", "HTML", "DocTagsFiles", "ChunkedDocTagFiles"]
    for ancestor in [start_path, *start_path.parents]:
        candidate = ancestor / "Data"
        if any((candidate / directory).is_dir() for directory in expected_dirs):
            return candidate.resolve()

    return (start_path / "Data").resolve()


def _ensure_dir(path: Path) -> Path:
    """Create ``path`` if needed and return its absolute form.

    Args:
        path: Directory to create when missing.

    Returns:
        Absolute path to the created directory.

    Examples:
        >>> _ensure_dir(Path("./tmp_dir"))
        PosixPath('tmp_dir')
    """

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def data_doctags(root: Optional[Path] = None) -> Path:
    """Return the DocTags directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the DocTags directory.

    Examples:
        >>> isinstance(data_doctags(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "DocTagsFiles")


def data_chunks(root: Optional[Path] = None) -> Path:
    """Return the chunk directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the chunk directory.

    Examples:
        >>> isinstance(data_chunks(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "ChunkedDocTagFiles")


def data_vectors(root: Optional[Path] = None) -> Path:
    """Return the vectors directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the vectors directory.

    Examples:
        >>> isinstance(data_vectors(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "Embeddings")


def data_manifests(root: Optional[Path] = None) -> Path:
    """Return the manifests directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the manifests directory.

    Examples:
        >>> isinstance(data_manifests(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "Manifests")


def prepare_data_root(data_root_arg: Optional[Path], default_root: Path) -> Path:
    """Resolve and prepare the DocsToKG data root for a pipeline invocation."""

    resolved = detect_data_root(data_root_arg) if data_root_arg is not None else default_root
    if data_root_arg is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved)
    data_manifests(resolved)
    return resolved


def resolve_pipeline_path(
    *,
    cli_value: Optional[Path],
    default_path: Path,
    resolved_data_root: Path,
    data_root_overridden: bool,
    resolver: Callable[[Path], Path],
) -> Path:
    """Derive a pipeline directory path respecting data-root overrides."""

    if data_root_overridden and (cli_value is None or cli_value == default_path):
        return resolver(resolved_data_root)
    if cli_value is None:
        return default_path
    return cli_value


def data_pdfs(root: Optional[Path] = None) -> Path:
    """Return the PDFs directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the PDFs directory.

    Examples:
        >>> isinstance(data_pdfs(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "PDFs")


def data_html(root: Optional[Path] = None) -> Path:
    """Return the HTML directory and ensure it exists.

    Args:
        root: Optional override for the starting directory used when
            resolving the DocsToKG data root.

    Returns:
        Absolute path to the HTML directory.

    Examples:
        >>> isinstance(data_html(), Path)
        True
    """

    return _ensure_dir(detect_data_root(root) / "HTML")


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
    """Return manifest doc identifier and vectors output path for ``chunk_file``.

    Args:
        chunk_file: Path to the chunk JSONL artefact.
        chunks_root: Root directory containing chunk artefacts.
        vectors_root: Root directory where vector outputs should be written.

    Returns:
        Tuple containing the manifest ``doc_id`` and the full vectors output path.
    """

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
    """Return POSIX-style relative identifier for a document path.

    Args:
        path: Absolute path to the document on disk.
        root: Root directory that anchors relative identifiers.

    Returns:
        str: POSIX-style relative path suitable for manifest IDs.
    """

    return path.relative_to(root).as_posix()


def compute_stable_shard(identifier: str, shard_count: int) -> int:
    """Deterministically map ``identifier`` to a shard in ``[0, shard_count)``."""

    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def should_skip_output(
    output_path: Path,
    manifest_entry: Optional[Mapping[str, object]],
    input_hash: str,
    resume: bool,
    force: bool,
) -> bool:
    """Return ``True`` when resume/skip conditions indicate work can be skipped."""

    if not resume or force:
        return False
    if not output_path.exists():
        return False
    if not manifest_entry:
        return False
    stored_hash = manifest_entry.get("input_hash") if isinstance(manifest_entry, Mapping) else None
    return stored_hash == input_hash


def _stringify_path(value: Path | str | None) -> str | None:
    """Return a string representation for path-like values used in manifests."""

    if value is None:
        return None
    return str(value)


def manifest_log_skip(
    *,
    stage: str,
    doc_id: str,
    input_path: Path | str,
    input_hash: str,
    output_path: Path | str,
    duration_s: float = 0.0,
    schema_version: Optional[str] = None,
    hash_alg: Optional[str] = None,
    **extra: object,
) -> None:
    """Record a manifest entry indicating the pipeline skipped work.

    Args:
        stage: Logical pipeline phase originating the log entry.
        doc_id: Identifier of the document being processed.
        input_path: Source artefact that would have been processed.
        input_hash: Content hash associated with ``input_path``.
        output_path: Destination artefact that remained unchanged.
        duration_s: Elapsed seconds for the short-circuited step.
        schema_version: Manifest schema version for downstream readers.
        hash_alg: Hash algorithm used to compute ``input_hash``.
        **extra: Additional metadata to merge into the manifest row.
    """
    payload: Dict[str, object] = {
        "stage": stage,
        "doc_id": doc_id,
        "status": "skip",
        "duration_s": float(duration_s),
        "schema_version": schema_version,
        "input_path": _stringify_path(input_path),
        "input_hash": input_hash,
        "hash_alg": hash_alg or resolve_hash_algorithm(),
        "output_path": _stringify_path(output_path),
    }
    payload.update(extra)
    manifest_append(**payload)


def manifest_log_success(
    *,
    stage: str,
    doc_id: str,
    duration_s: float,
    schema_version: str,
    input_path: Path | str,
    input_hash: str,
    output_path: Path | str,
    hash_alg: Optional[str] = None,
    **extra: object,
) -> None:
    """Record a manifest entry marking successful pipeline output.

    Args:
        stage: Logical pipeline phase originating the log entry.
        doc_id: Identifier of the document being processed.
        duration_s: Elapsed seconds for the successful step.
        schema_version: Manifest schema version for downstream readers.
        input_path: Source artefact that produced ``output_path``.
        input_hash: Content hash associated with ``input_path``.
        output_path: Destination artefact written by the pipeline.
        hash_alg: Hash algorithm used to compute ``input_hash``.
        **extra: Additional metadata to merge into the manifest row.
    """
    payload: Dict[str, object] = {
        "stage": stage,
        "doc_id": doc_id,
        "status": "success",
        "duration_s": float(duration_s),
        "schema_version": schema_version,
        "input_path": _stringify_path(input_path),
        "input_hash": input_hash,
        "hash_alg": hash_alg or resolve_hash_algorithm(),
        "output_path": _stringify_path(output_path),
    }
    payload.update(extra)
    manifest_append(**payload)


def manifest_log_failure(
    *,
    stage: str,
    doc_id: str,
    duration_s: float,
    schema_version: str,
    input_path: Path | str,
    input_hash: str,
    output_path: Path | str,
    error: str,
    hash_alg: Optional[str] = None,
    **extra: object,
) -> None:
    """Record a manifest entry describing a failed pipeline attempt.

    Args:
        stage: Logical pipeline phase originating the log entry.
        doc_id: Identifier of the document being processed.
        duration_s: Elapsed seconds before the failure occurred.
        schema_version: Manifest schema version for downstream readers.
        input_path: Source artefact that triggered the failure.
        input_hash: Content hash associated with ``input_path``.
        output_path: Destination artefact that may be incomplete.
        error: Human-readable description of the failure condition.
        hash_alg: Hash algorithm used to compute ``input_hash``.
        **extra: Additional metadata to merge into the manifest row.
    """
    payload: Dict[str, object] = {
        "stage": stage,
        "doc_id": doc_id,
        "status": "failure",
        "duration_s": float(duration_s),
        "schema_version": schema_version,
        "input_path": _stringify_path(input_path),
        "input_hash": input_hash,
        "hash_alg": hash_alg or resolve_hash_algorithm(),
        "output_path": _stringify_path(output_path),
        "error": error,
    }
    payload.update(extra)
    manifest_append(**payload)


# --- Logging and I/O Utilities ---


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that enriches structured logs with shared context."""

    def __init__(self, logger: logging.Logger, base_fields: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(logger, {})
        self.base_fields: Dict[str, Any] = dict(base_fields or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        fields = dict(self.base_fields)
        extra_fields = extra.get("extra_fields")
        if isinstance(extra_fields, dict):
            fields.update(extra_fields)
        extra["extra_fields"] = fields
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **fields: object) -> "StructuredLogger":
        filtered = {k: v for k, v in fields.items() if v is not None}
        self.base_fields.update(filtered)
        return self

    def child(self, **fields: object) -> "StructuredLogger":
        merged = dict(self.base_fields)
        merged.update({k: v for k, v in fields.items() if v is not None})
        return StructuredLogger(self.logger, merged)


def get_logger(name: str, level: str = "INFO", *, base_fields: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """Get a structured JSON logger configured for console output.

    Args:
        name: Name of the logger to create or retrieve.
        level: Logging level (case insensitive). Defaults to ``"INFO"``.

    Returns:
        Configured :class:`logging.Logger` instance.

    Examples:
        >>> logger = get_logger("docparse")
        >>> logger.level == logging.INFO
        True
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()

        class JSONFormatter(logging.Formatter):
            """Emit structured JSON log messages for DocParsing utilities.

            Attributes:
                default_time_format: Timestamp template applied to log records.

            Examples:
                >>> formatter = JSONFormatter()
                >>> hasattr(formatter, "format")
                True
            """

            def format(self, record: logging.LogRecord) -> str:
                """Render a log record as a JSON string.

                Args:
                    record: Logging record produced by the DocParsing pipeline.

                Returns:
                    JSON-formatted string containing canonical log fields and optional extras.
                """
                ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                    timespec="milliseconds"
                )
                if ts.endswith("+00:00"):
                    ts = ts[:-6] + "Z"
                payload = {
                    "timestamp": ts,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                extra_fields = getattr(record, "extra_fields", None)
                if isinstance(extra_fields, dict):
                    payload.update(extra_fields)
                return json.dumps(payload, ensure_ascii=False)

        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return StructuredLogger(logger, base_fields)


def log_event(logger: logging.Logger, level: str, message: str, **fields: object) -> None:
    """Emit a structured log record using the ``extra_fields`` convention."""

    emitter = getattr(logger, level, None)
    if not callable(emitter):
        raise AttributeError(f"Logger has no level '{level}'")
    emitter(message, extra={"extra_fields": fields})


def find_free_port(start: int = 8000, span: int = 32) -> int:
    """Locate an available TCP port on localhost within a range.

    Args:
        start: Starting port for the scan. Defaults to ``8000``.
        span: Number of sequential ports to check. Defaults to ``32``.

    Returns:
        The first free port number. Falls back to an OS-assigned ephemeral port
        if the requested range is exhausted.

    Examples:
        >>> port = find_free_port(8500, 1)
        >>> isinstance(port, int)
        True
    """

    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port

    logger = get_logger(__name__)
    logger.warning(
        "Port scan exhausted",
        extra={"extra_fields": {"start": start, "span": span, "action": "ephemeral_port"}},
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def atomic_write(path: Path) -> Iterator[TextIO]:
    """Write to a temporary file and atomically replace the destination.

    Pattern: open a sibling ``*.tmp`` file, write the payload, flush and
    ``fsync`` the descriptor, then ``rename`` it over the original path. This
    guarantees that readers never observe a partially written file even if the
    process crashes mid-write.

    Args:
        path: Target path to write.

    Returns:
        Context manager yielding a writable text handle.

    Yields:
        Writable text file handle. Caller must write data before context exit.

    Raises:
        Any exception raised while writing or replacing the file is propagated
        after the temporary file is cleaned up.

    Examples:
        >>> target = Path("/tmp/example.txt")
        >>> with atomic_write(target) as handle:
        ...     _ = handle.write("hello")
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# --- Dataset Iterators ---


def iter_doctags(directory: Path) -> Iterator[Path]:
    """Yield DocTags files within ``directory`` and subdirectories.

    Args:
        directory: Root directory to scan for DocTags artifacts.

    Returns:
        Iterator over absolute ``Path`` objects.

    Yields:
        Absolute paths to discovered ``.doctags`` or ``.doctag`` files sorted
        lexicographically.

    Examples:
        >>> next(iter_doctags(Path(".")), None) is None
        True
    """

    extensions = ("*.doctags", "*.doctag")
    seen = set()
    for pattern in extensions:
        for candidate in directory.rglob(pattern):
            if candidate.is_file() and not candidate.name.startswith("."):
                seen.add(candidate.resolve())
    for path in sorted(seen):
        yield path


def iter_chunks(directory: Path) -> Iterator[Path]:
    """Yield chunk JSONL files from ``directory`` and all descendants.

    Args:
        directory: Directory containing chunk artifacts.

    Returns:
        Iterator over absolute ``Path`` objects.

    Yields:
        Absolute paths to files matching ``*.chunks.jsonl`` sorted
        lexicographically.

    Examples:
        >>> next(iter_chunks(Path(".")), None) is None
        True
    """

    seen = set()
    for candidate in directory.rglob("*.chunks.jsonl"):
        if candidate.is_file() and not candidate.name.startswith("."):
            seen.add(candidate.resolve())
    for path in sorted(seen):
        yield path


# --- JSONL Helpers ---


def jsonl_load(path: Path, skip_invalid: bool = False, max_errors: int = 10) -> List[dict]:
    """Load a JSONL file into memory with optional error tolerance."""

    return list(
        _iter_jsonl_records(
            path,
            start=None,
            end=None,
            skip_invalid=skip_invalid,
            max_errors=max_errors,
        )
    )


def jsonl_save(
    path: Path, rows: List[dict], validate: Optional[Callable[[dict], None]] = None
) -> None:
    """Persist dictionaries to a JSONL file atomically.

    Args:
        path: Destination JSONL file.
        rows: Sequence of dictionaries to serialize.
        validate: Optional callback invoked per row before serialization.

    Returns:
        None: This function performs I/O side effects only.

    Raises:
        ValueError: If ``validate`` raises an exception for any row.

    Examples:
        >>> tmp = Path("/tmp/example.jsonl")
        >>> jsonl_save(tmp, [{"a": 1}])
        >>> tmp.read_text(encoding="utf-8").strip()
        '{"a": 1}'
    """

    with atomic_write(path) as handle:
        for index, row in enumerate(rows):
            if validate is not None:
                try:
                    validate(row)
                except Exception as exc:  # pragma: no cover - error path exercised
                    raise ValueError(f"Validation failed for row {index}: {exc}") from exc
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_append_iter(
    target: Path | TextIO,
    rows: Iterable[Mapping],
    *,
    atomic: bool = True,
) -> int:
    """Append JSON-serialisable rows to a JSONL file.

    Args:
        target: Destination path or writable handle for the JSONL file.
        rows: Iterable of JSON-serialisable mappings.
        atomic: When True, writes occur via :func:`atomic_write` (ignored when
            ``target`` is already an open handle).

    Returns:
        The number of rows written.
    """

    if hasattr(target, "write"):
        handle = target  # type: ignore[assignment]
        count = 0
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")  # type: ignore[arg-type]
            count += 1
        return count

    path = Path(target)
    if not atomic:
        count = 0
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count

    count = 0
    with atomic_write(path) as handle:
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as src:
                shutil.copyfileobj(src, handle)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_jsonl_split_map(
    path: Path,
    *,
    chunk_bytes: int = 32 * 1024 * 1024,
    min_chunk_bytes: int = 1 * 1024 * 1024,
) -> List[Tuple[int, int]]:
    """Return newline-aligned byte ranges that partition ``path``."""

    size = path.stat().st_size
    if size == 0:
        return [(0, 0)]

    chunk_bytes = max(chunk_bytes, min_chunk_bytes)
    offsets: List[Tuple[int, int]] = []
    with path.open("rb") as handle:
        start = 0
        while start < size:
            target = start + chunk_bytes
            if target >= size:
                end = size
            else:
                handle.seek(target)
                handle.readline()
                end = handle.tell()
                if end <= start:
                    end = min(size, start + chunk_bytes)
            offsets.append((start, end))
            start = end
    return offsets


def _iter_jsonl_records(
    path: Path,
    *,
    start: Optional[int],
    end: Optional[int],
    skip_invalid: bool,
    max_errors: int,
) -> Iterator[dict]:
    logger = get_logger(__name__)
    errors = 0
    with path.open("rb") as handle:
        if start:
            handle.seek(start)
            if start != 0:
                handle.readline()
        while True:
            pos = handle.tell()
            if end is not None and pos >= end:
                break
            raw = handle.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    errors += 1
                    if errors >= max_errors:
                        log_event(
                            logger,
                            "error",
                            "Too many JSON errors",
                            path=str(path),
                            errors=errors,
                            start=start,
                            end=end,
                        )
                        break
                    continue
                raise ValueError(
                    f"Invalid JSON in {path} at byte offset {pos}: {exc}"
                ) from exc
    if errors:
        log_event(
            logger,
            "warning",
            "Skipped invalid JSON lines",
            path=str(path),
            skipped=errors,
            start=start,
            end=end,
        )


# --- Collection Utilities ---


class Batcher(Iterable[List[T]]):
    """Yield fixed-size batches from an iterable with optional policies.

    Args:
        iterable: Source iterable providing items to batch.
        batch_size: Maximum number of elements per yielded batch.
        policy: Optional batching policy. When ``"length"`` the iterable is
            bucketed by ``lengths`` before batching.
        lengths: Sequence of integer lengths aligned with ``iterable`` used for
            length-aware batching policies.

    Examples:
        >>> list(Batcher([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """

    def __init__(
        self,
        iterable: Iterable[T],
        batch_size: int,
        *,
        policy: Optional[str] = None,
        lengths: Optional[Sequence[int]] = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._items: List[T] = list(iterable)
        self._batch_size = batch_size
        self._policy = (policy or "").lower() or None
        if self._policy:
            if self._policy not in {"length"}:
                raise ValueError(f"Unsupported batching policy: {policy}")
            if lengths is None:
                raise ValueError("lengths must be provided when using a policy")
            if len(lengths) != len(self._items):
                raise ValueError("lengths must align with iterable length")
            self._lengths = [int(max(0, length)) for length in lengths]
        else:
            self._lengths = None

    @staticmethod
    def _length_bucket(length: int) -> int:
        """Return the power-of-two bucket for ``length``."""

        if length <= 0:
            return 0
        return 1 << (int(math.log2(length - 1)) + 1)

    def _ordered_indices(self) -> List[int]:
        if not self._lengths:
            return list(range(len(self._items)))
        pairs = [
            (idx, self._length_bucket(self._lengths[idx]))
            for idx in range(len(self._items))
        ]
        pairs.sort(key=lambda pair: (pair[1], pair[0]))
        return [idx for idx, _ in pairs]

    def __iter__(self) -> Iterator[List[T]]:
        if not self._policy:
            for i in range(0, len(self._items), self._batch_size):
                yield self._items[i : i + self._batch_size]
            return

        ordered_indices = self._ordered_indices()
        for i in range(0, len(ordered_indices), self._batch_size):
            batch_indices = ordered_indices[i : i + self._batch_size]
            yield [self._items[idx] for idx in batch_indices]


# --- Manifest Utilities ---


def _manifest_filename(stage: str) -> str:
    """Return manifest filename for a given stage."""

    safe = stage.strip() or "all"
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "-" for c in safe)
    return f"docparse.{safe}.manifest.jsonl"


def manifest_append(
    stage: str,
    doc_id: str,
    status: str,
    *,
    duration_s: float = 0.0,
    warnings: Optional[List[str]] = None,
    error: Optional[str] = None,
    schema_version: str = "",
    **metadata,
) -> None:
    """Append a structured entry to the processing manifest.

    Args:
        stage: Pipeline stage emitting the entry.
        doc_id: Identifier of the document being processed.
        status: Outcome status (``success``, ``failure``, or ``skip``).
        duration_s: Optional duration in seconds.
        warnings: Optional list of warning labels.
        error: Optional error description.
        schema_version: Schema identifier recorded for the output.
        **metadata: Arbitrary additional fields to include.

    Returns:
        ``None``.

    Raises:
        ValueError: If ``status`` is not recognised.

    Examples:
        >>> manifest_append("chunk", "doc1", "success")
        >>> (data_manifests() / "docparse.chunk.manifest.jsonl").exists()
        True
    """

    allowed_status = {"success", "failure", "skip"}
    if status not in allowed_status:
        raise ValueError(f"status must be one of {sorted(allowed_status)}")

    manifest_path = data_manifests() / _manifest_filename(stage)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "doc_id": doc_id,
        "status": status,
        "duration_s": round(duration_s, 3),
        "warnings": warnings or [],
        "schema_version": schema_version,
    }
    if error is not None:
        entry["error"] = str(error)
    entry.update(metadata)

    jsonl_append_iter(manifest_path, [entry])


def resolve_hash_algorithm(default: str = "sha1") -> str:
    """Return the active content hash algorithm, honoring env overrides.

    Args:
        default: Fallback algorithm name to use when no override is present.

    Returns:
        Hash algorithm identifier resolved from ``DOCSTOKG_HASH_ALG`` or ``default``.
    """

    env_override = os.getenv("DOCSTOKG_HASH_ALG")
    return env_override.strip() if env_override else default


def compute_chunk_uuid(
    doc_id: str,
    start_offset: int,
    text: str,
    *,
    algorithm: str = "sha1",
) -> str:
    """Derive a deterministic UUID for a chunk using doc ID, offset, and text content.

    Args:
        doc_id: Identifier for the source document (used as a namespace component).
        start_offset: Character offset of the chunk text within the document.
        text: Chunk text used for content-based stability.
        algorithm: Hash algorithm name; defaults to ``sha1`` but honours
            :envvar:`DOCSTOKG_HASH_ALG` overrides.

    Returns:
        UUID string derived from the hash digest while enforcing RFC4122 metadata bits.
    """

    safe_doc_id = str(doc_id)
    try:
        safe_offset = int(start_offset)
    except (TypeError, ValueError):
        safe_offset = 0
    normalised_text = unicodedata.normalize("NFKC", str(text or ""))

    selected_algorithm = resolve_hash_algorithm(algorithm)
    hasher = hashlib.new(selected_algorithm)
    hasher.update(safe_doc_id.encode("utf-8"))
    hasher.update(bytes((0x1F,)))
    hasher.update(str(safe_offset).encode("utf-8"))
    hasher.update(bytes((0x1F,)))
    hasher.update(normalised_text.encode("utf-8"))
    digest = hasher.digest()
    if len(digest) < 16:
        digest = (digest * ((16 // len(digest)) + 1))[:16]
    raw = bytearray(digest[:16])
    raw[6] = (raw[6] & 0x0F) | 0x50  # set UUID version bits to 5
    raw[8] = (raw[8] & 0x3F) | 0x80  # set UUID variant bits to RFC 4122
    return str(uuid.UUID(bytes=bytes(raw)))


def relative_path(path: Path | str, root: Optional[Path]) -> str:
    """Return ``path`` rendered relative to ``root`` when feasible."""

    candidate = Path(path)
    if root is None:
        return str(candidate)
    try:
        root_path = Path(root).resolve()
        return str(candidate.resolve().relative_to(root_path))
    except Exception:
        return str(candidate)


def quarantine_artifact(
    path: Path,
    reason: str,
    *,
    logger: Optional[logging.Logger] = None,
    create_placeholder: bool = False,
) -> Path:
    """Move ``path`` to a ``.quarantine`` sibling for operator review.

    Args:
        path: Artefact to quarantine.
        reason: Explanation describing why the artefact was quarantined.
        logger: Optional logger used to emit structured diagnostics.
        create_placeholder: When ``True`` a placeholder file is created even if
            ``path`` does not presently exist (useful for failed writes).

    Returns:
        Path to the quarantined artefact.
    """

    original = Path(path)
    parent = original.parent
    parent.mkdir(parents=True, exist_ok=True)
    suffix = ".quarantine"
    candidate = parent / f"{original.name}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = parent / f"{original.name}{suffix}{counter}"
        counter += 1

    try:
        if original.exists():
            original.rename(candidate)
        elif create_placeholder:
            candidate.write_text(reason + "\n", encoding="utf-8")
    except Exception:
        if logger:
            logger.exception(
                "Failed to quarantine artifact",
                extra={
                    "extra_fields": {
                        "path": str(original),
                        "attempted_quarantine": str(candidate),
                        "reason": reason,
                    }
                },
            )
        raise

    if logger:
        logger.warning(
            "Quarantined artifact",
            extra={
                "extra_fields": {
                    "path": str(original),
                    "quarantine_path": str(candidate),
                    "reason": reason,
                }
            },
        )
    return candidate


def compute_content_hash(path: Path, algorithm: str = "sha1") -> str:
    """Compute a content hash for ``path`` using the requested algorithm.

    Args:
        path: File whose contents should be hashed.
        algorithm: Hash algorithm name supported by :mod:`hashlib`.

    Notes:
        The ``DOCSTOKG_HASH_ALG`` environment variable overrides ``algorithm``
        when set, enabling fleet-wide hash changes without code edits.

    Returns:
        Hex digest string.

    Examples:
        >>> tmp = Path("/tmp/hash.txt")
        >>> _ = tmp.write_text("hello", encoding="utf-8")
        >>> compute_content_hash(tmp) == hashlib.sha1(b"hello").hexdigest()
        True
    """

    selected_algorithm = resolve_hash_algorithm(algorithm)
    hasher = hashlib.new(selected_algorithm)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    normalised = unicodedata.normalize("NFKC", text)
    hasher.update(normalised.encode("utf-8"))
    return hasher.hexdigest()


def load_manifest_index(stage: str, root: Optional[Path] = None) -> Dict[str, dict]:
    """Load the latest manifest entries for a specific pipeline stage.

    Args:
        stage: Manifest stage identifier to filter entries by.
        root: Optional DocsToKG data root used to resolve the manifest path.

    Returns:
        Mapping of ``doc_id`` to the most recent manifest entry for that stage.

    Raises:
        None: Manifest rows that fail to parse are skipped to keep processing resilient.

    Examples:
        >>> index = load_manifest_index("embeddings")  # doctest: +SKIP
        >>> isinstance(index, dict)
        True
    """

    manifest_dir = data_manifests(root)
    stage_path = manifest_dir / _manifest_filename(stage)
    index: Dict[str, dict] = {}
    if not stage_path.exists():
        return index

    with stage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("stage") != stage:
                continue
            doc_id = entry.get("doc_id")
            if not doc_id:
                continue
            index[doc_id] = entry
    return index


# --- Concurrency Utilities ---


@contextlib.contextmanager
def acquire_lock(path: Path, timeout: float = 60.0) -> Iterator[bool]:
    """Acquire an advisory lock using ``.lock`` sentinel files.

    Args:
        path: Target file path whose lock should be acquired.
        timeout: Maximum time in seconds to wait for the lock.

    Returns:
        Iterator yielding a boolean when the lock is acquired.

    Yields:
        ``True`` once the lock is acquired.

    Raises:
        TimeoutError: If the lock cannot be obtained within ``timeout``.

    Examples:
        >>> target = Path("/tmp/lock.txt")
        >>> with acquire_lock(target):
        ...     pass
    """

    lock_path = path.with_suffix(path.suffix + ".lock")
    start = time.time()
    while lock_path.exists():
        try:
            pid_text = lock_path.read_text(encoding="utf-8").strip()
            existing_pid = int(pid_text) if pid_text else None
        except (OSError, ValueError):
            existing_pid = None

        if existing_pid and not _pid_is_running(existing_pid):
            lock_path.unlink(missing_ok=True)
            continue

        if time.time() - start > timeout:
            raise TimeoutError(f"Could not acquire lock on {path} after {timeout}s")
        time.sleep(0.1)

    try:
        lock_path.write_text(str(os.getpid()), encoding="utf-8")
        yield True
    finally:
        lock_path.unlink(missing_ok=True)


def _pid_is_running(pid: int) -> bool:
    """Return ``True`` if a process with the given PID appears to be alive."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - platform specific
        return True
    except OSError:  # pragma: no cover - defensive guard
        return False
    return True


def set_spawn_or_warn(logger: Optional[logging.Logger] = None) -> None:
    """Ensure the multiprocessing start method is set to ``spawn``.

    Args:
        logger: Optional logger that receives diagnostic messages about the start
            method configuration.

    Returns:
        None: The function mutates global multiprocessing state and logs warnings.

    This helper attempts to set the start method to ``spawn`` with ``force=True``.
    If a ``RuntimeError`` occurs (meaning the method was already set), it checks
    if the current method is ``spawn``. If not, it emits a warning about the
    potential CUDA safety risk, logging the current method so callers understand
    the degraded safety state.
    """

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
        if logger is not None:
            logger.debug("Multiprocessing start method set to 'spawn'")
        return
    except RuntimeError:
        current = mp.get_start_method(allow_none=True)
        if current == "spawn":
            if logger is not None:
                logger.debug("Multiprocessing start method already 'spawn'")
            return
        message = "Multiprocessing start method is %s; CUDA workloads require 'spawn'." % (
            current or "unset"
        )
        if logger is not None:
            logger.warning(message)
        else:
            logging.getLogger(__name__).warning(message)


# --- Unified CLI ---


CommandHandler = Callable[[Sequence[str]], int]

CLI_DESCRIPTION = """\
Unified DocParsing CLI

Examples:
  python -m DocsToKG.DocParsing.cli all --resume
  python -m DocsToKG.DocParsing.cli chunk --data-root Data
  python -m DocsToKG.DocParsing.cli embed --resume
  python -m DocsToKG.DocParsing.cli doctags --mode pdf --workers 2
  python -m DocsToKG.DocParsing.cli token-profiles --doctags-dir Data/DocTagsFiles
"""

_PDF_SUFFIXES: tuple[str, ...] = (".pdf",)
_HTML_SUFFIXES: tuple[str, ...] = (".html", ".htm")


def _run_chunk(argv: Sequence[str]) -> int:
    """Execute the Docling chunker subcommand."""

    from DocsToKG.DocParsing import chunking as chunk_module

    parser = chunk_module.build_parser()
    parser.prog = "docparse chunk"
    args = parser.parse_args(argv)
    return chunk_module.main(args)


def _run_embed(argv: Sequence[str]) -> int:
    """Execute the embedding pipeline subcommand."""

    from DocsToKG.DocParsing import embedding as embedding_module

    parser = embedding_module.build_parser()
    parser.prog = "docparse embed"
    args = parser.parse_args(argv)
    return embedding_module.main(args)


def _run_token_profiles(argv: Sequence[str]) -> int:
    """Execute the tokenizer profiling subcommand."""

    from DocsToKG.DocParsing import token_profiles as token_profiles_module

    parser = token_profiles_module.build_parser()
    parser.prog = "docparse token-profiles"
    args = parser.parse_args(argv)
    return token_profiles_module.main(args)


def _build_doctags_parser(prog: str = "docparse doctags") -> argparse.ArgumentParser:
    """Create an :mod:`argparse` parser configured for DocTags conversion."""

    from DocsToKG.DocParsing import doctags as doctags_module

    examples = """Examples:
  docparse doctags --input Data/HTML
  docparse doctags --mode pdf --workers 4
  docparse doctags --mode html --overwrite
"""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Convert HTML or PDF corpora to DocTags using Docling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "html", "pdf"],
        default="auto",
        help="Select conversion backend; auto infers from input directory",
    )
    doctags_module.add_data_root_option(parser)
    parser.add_argument(
        "--in-dir",
        "--input",
        dest="in_dir",
        type=Path,
        default=None,
        help="Directory containing HTML or PDF sources (defaults vary by mode)",
    )
    parser.add_argument(
        "--out-dir",
        "--output",
        dest="out_dir",
        type=Path,
        default=None,
        help="Destination for generated .doctags files (defaults vary by mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes to launch; backend defaults used when omitted",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override vLLM model path or identifier for PDF conversion",
    )
    parser.add_argument(
        "--served-model-name",
        dest="served_model_names",
        action="append",
        nargs="+",
        default=None,
        help="Model alias to expose from vLLM (repeatable)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory allocated to the vLLM server",
    )
    doctags_module.add_resume_force_options(
        parser,
        resume_help="Skip documents whose outputs already exist with matching content hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DocTags files (HTML mode only)",
    )
    return parser


def _scan_pdf_html(input_dir: Path) -> tuple[bool, bool]:
    """Return booleans indicating whether PDFs or HTML files exist beneath ``input_dir``."""

    has_pdf = False
    has_html = False

    if not input_dir.exists():
        return has_pdf, has_html

    for root, _dirs, files in os.walk(input_dir):
        if not files:
            continue
        for name in files:
            lower = name.lower()
            if not has_pdf and lower.endswith(_PDF_SUFFIXES):
                has_pdf = True
            elif not has_html and lower.endswith(_HTML_SUFFIXES):
                has_html = True
            if has_pdf and has_html:
                return has_pdf, has_html
    return has_pdf, has_html


def _directory_contains_suffixes(directory: Path, suffixes: tuple[str, ...]) -> bool:
    """Return True when ``directory`` contains at least one file ending with ``suffixes``."""

    if not directory.exists():
        return False
    suffixes_lower = tuple(s.lower() for s in suffixes)
    for root, _dirs, files in os.walk(directory):
        if not files:
            continue
        for name in files:
            if name.lower().endswith(suffixes_lower):
                return True
    return False


def _detect_mode(input_dir: Path) -> str:
    """Infer conversion mode based on the contents of ``input_dir``."""

    if not input_dir.exists():
        raise ValueError(f"Cannot auto-detect mode in {input_dir}: directory not found")

    has_pdf, has_html = _scan_pdf_html(input_dir)
    if has_pdf and not has_html:
        return "pdf"
    if has_html and not has_pdf:
        return "html"
    if has_pdf and has_html:
        raise ValueError(f"Cannot auto-detect mode in {input_dir}: found both PDF and HTML files")
    raise ValueError(f"Cannot auto-detect mode in {input_dir}: no PDF or HTML files found")


def _merge_args(parser: argparse.ArgumentParser, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Merge override values into the default parser namespace."""

    base = parser.parse_args([])
    for key, value in overrides.items():
        if value is not None:
            setattr(base, key, value)
    return base


def _run_doctags(argv: Sequence[str]) -> int:
    """Execute the DocTags conversion subcommand."""

    from DocsToKG.DocParsing import doctags as doctags_module

    parser = _build_doctags_parser()
    args = parser.parse_args(argv)
    logger = get_logger(__name__)

    resolved_root = (
        detect_data_root(args.data_root) if args.data_root is not None else detect_data_root()
    )

    html_default_in = data_html(resolved_root)
    pdf_default_in = data_pdfs(resolved_root)
    doctags_default_out = data_doctags(resolved_root)

    mode = args.mode
    if args.in_dir is not None:
        input_dir = args.in_dir.resolve()
        if mode == "auto":
            mode = _detect_mode(input_dir)
    else:
        if mode == "auto":
            html_present = _directory_contains_suffixes(html_default_in, _HTML_SUFFIXES)
            pdf_present = _directory_contains_suffixes(pdf_default_in, _PDF_SUFFIXES)
            if html_present and not pdf_present:
                mode = "html"
            elif pdf_present and not html_present:
                mode = "pdf"
            else:
                raise ValueError("Cannot auto-detect mode: specify --mode or --input explicitly")
        input_dir = html_default_in if mode == "html" else pdf_default_in

    output_dir = args.out_dir.resolve() if args.out_dir is not None else doctags_default_out

    args.in_dir = input_dir
    args.out_dir = output_dir

    logger.info(
        "Unified DocTags conversion",
        extra={
            "extra_fields": {
                "mode": mode,
                "data_root": str(resolved_root),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "workers": args.workers,
                "resume": args.resume,
                "force": args.force,
                "overwrite": args.overwrite,
                "model": args.model,
                "served_model_names": args.served_model_names,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        },
    )

    base_overrides = {
        "data_root": args.data_root,
        "input": input_dir,
        "output": output_dir,
        "workers": args.workers,
        "resume": args.resume,
        "force": args.force,
    }

    if mode == "html":
        html_overrides = {
            **base_overrides,
            "overwrite": args.overwrite,
        }
        html_args = _merge_args(doctags_module.html_build_parser(), html_overrides)
        return doctags_module.html_main(html_args)

    overrides = {
        **base_overrides,
        "model": args.model,
        "served_model_names": args.served_model_names,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    pdf_args = _merge_args(doctags_module.pdf_build_parser(), overrides)
    return doctags_module.pdf_main(pdf_args)


def _preview_list(items: List[str], limit: int = 5) -> List[str]:
    """Return a truncated preview list with remainder hint."""

    if len(items) <= limit:
        return list(items)
    preview = list(items[:limit])
    preview.append(f"... (+{len(items) - limit} more)")
    return preview


def _plan_doctags(argv: Sequence[str]) -> Dict[str, Any]:
    """Compute which DocTags inputs would be processed."""

    from DocsToKG.DocParsing import doctags as doctags_module

    parser = _build_doctags_parser()
    args = parser.parse_args(argv)
    resolved_root = (
        detect_data_root(args.data_root) if args.data_root is not None else detect_data_root()
    )

    html_default_in = data_html(resolved_root)
    pdf_default_in = data_pdfs(resolved_root)
    doctags_default_out = data_doctags(resolved_root)

    mode = args.mode
    if args.in_dir is not None:
        input_dir = args.in_dir.resolve()
        if mode == "auto":
            mode = _detect_mode(input_dir)
    else:
        if mode == "auto":
            html_present = _directory_contains_suffixes(html_default_in, _HTML_SUFFIXES)
            pdf_present = _directory_contains_suffixes(pdf_default_in, _PDF_SUFFIXES)
            if html_present and not pdf_present:
                mode = "html"
            elif pdf_present and not html_present:
                mode = "pdf"
            else:
                raise ValueError("Cannot auto-detect mode: specify --mode or --input explicitly")
        input_dir = html_default_in if mode == "html" else pdf_default_in

    output_dir = args.out_dir.resolve() if args.out_dir is not None else doctags_default_out

    if not input_dir.exists():
        return {
            "stage": "doctags",
            "mode": mode,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "process": [],
            "skip": [],
            "notes": ["Input directory missing"],
        }

    if mode == "html":
        files = doctags_module.list_htmls(input_dir)
        manifest_stage = getattr(doctags_module, "HTML_MANIFEST_STAGE", "doctags-html")
        overwrite = bool(getattr(args, "overwrite", False))
    else:
        files = doctags_module.list_pdfs(input_dir)
        manifest_stage = doctags_module.MANIFEST_STAGE
        overwrite = False

    manifest_index = load_manifest_index(manifest_stage, resolved_root) if args.resume else {}
    planned: List[str] = []
    skipped: List[str] = []

    for path in files:
        doc_id, out_path = derive_doc_id_and_doctags_path(path, input_dir, output_dir)
        input_hash = compute_content_hash(path)
        manifest_entry = manifest_index.get(doc_id)
        skip = should_skip_output(out_path, manifest_entry, input_hash, args.resume, args.force)
        if mode == "html" and overwrite:
            skip = False
        if skip:
            skipped.append(doc_id)
        else:
            planned.append(doc_id)

    return {
        "stage": "doctags",
        "mode": mode,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "process": planned,
        "skip": skipped,
        "notes": [],
    }


def _plan_chunk(argv: Sequence[str]) -> Dict[str, Any]:
    """Compute which DocTags files the chunk stage would touch."""

    from DocsToKG.DocParsing import chunking as chunk_module
    from DocsToKG.DocParsing import doctags as doctags_module

    parser = chunk_module.build_parser()
    args = parser.parse_args(argv)
    resolved_root = doctags_module.prepare_data_root(args.data_root, chunk_module.DEFAULT_DATA_ROOT)
    data_root_overridden = args.data_root is not None

    in_dir = doctags_module.resolve_pipeline_path(
        cli_value=args.in_dir,
        default_path=chunk_module.DEFAULT_IN_DIR,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=data_doctags,
    ).resolve()

    out_dir = doctags_module.resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=chunk_module.DEFAULT_OUT_DIR,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=data_chunks,
    ).resolve()

    if not in_dir.exists():
        return {
            "stage": "chunk",
            "input_dir": str(in_dir),
            "output_dir": str(out_dir),
            "process": [],
            "skip": [],
            "notes": ["DocTags directory missing"],
        }

    files = list(iter_doctags(in_dir))
    manifest_index = (
        load_manifest_index(chunk_module.MANIFEST_STAGE, resolved_root) if args.resume else {}
    )
    planned: List[str] = []
    skipped: List[str] = []

    for path in files:
        rel_id, out_path = derive_doc_id_and_chunks_path(path, in_dir, out_dir)
        input_hash = compute_content_hash(path)
        manifest_entry = manifest_index.get(rel_id)
        if should_skip_output(out_path, manifest_entry, input_hash, args.resume, args.force):
            skipped.append(rel_id)
        else:
            planned.append(rel_id)

    return {
        "stage": "chunk",
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "process": planned,
        "skip": skipped,
        "notes": [],
    }


def _plan_embed(argv: Sequence[str]) -> Dict[str, Any]:
    """Compute which chunk files the embed stage would process or validate."""

    from DocsToKG.DocParsing import doctags as doctags_module
    from DocsToKG.DocParsing import embedding as embedding_module

    parser = embedding_module.build_parser()
    args = parser.parse_args(argv)
    resolved_root = doctags_module.prepare_data_root(
        args.data_root, embedding_module.DEFAULT_DATA_ROOT
    )
    data_root_overridden = args.data_root is not None

    chunks_dir = doctags_module.resolve_pipeline_path(
        cli_value=args.chunks_dir,
        default_path=embedding_module.DEFAULT_CHUNKS_DIR,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=data_chunks,
    ).resolve()

    vectors_dir = doctags_module.resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=embedding_module.DEFAULT_VECTORS_DIR,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=data_vectors,
    ).resolve()

    if args.validate_only:
        validate: List[str] = []
        missing: List[str] = []
        for chunk_path in iter_chunks(chunks_dir):
            doc_id, vector_path = derive_doc_id_and_vectors_path(
                chunk_path, chunks_dir, vectors_dir
            )
            if vector_path.exists():
                validate.append(doc_id)
            else:
                missing.append(doc_id)
        return {
            "stage": "embed",
            "action": "validate",
            "chunks_dir": str(chunks_dir),
            "vectors_dir": str(vectors_dir),
            "validate": validate,
            "missing": missing,
            "notes": [],
        }

    files = list(iter_chunks(chunks_dir))
    manifest_index = (
        load_manifest_index(embedding_module.MANIFEST_STAGE, resolved_root) if args.resume else {}
    )
    planned: List[str] = []
    skipped: List[str] = []

    for chunk_path in files:
        doc_id, vector_path = derive_doc_id_and_vectors_path(chunk_path, chunks_dir, vectors_dir)
        input_hash = compute_content_hash(chunk_path)
        manifest_entry = manifest_index.get(doc_id)
        if should_skip_output(vector_path, manifest_entry, input_hash, args.resume, args.force):
            skipped.append(doc_id)
        else:
            planned.append(doc_id)

    return {
        "stage": "embed",
        "action": "generate",
        "chunks_dir": str(chunks_dir),
        "vectors_dir": str(vectors_dir),
        "process": planned,
        "skip": skipped,
        "notes": [],
    }


def _display_plan(plans: Sequence[Dict[str, Any]]) -> None:
    """Pretty-print plan summaries to stdout."""

    print("docparse all plan")
    for entry in plans:
        stage = entry.get("stage", "unknown")
        notes = entry.get("notes", [])
        if stage == "doctags":
            desc = f"doctags (mode={entry.get('mode')})"
            process = entry.get("process", [])
            skip = entry.get("skip", [])
            print(f"- {desc}: process {len(process)}, skip {len(skip)}")
            print(f"  input:  {entry.get('input_dir')}")
            print(f"  output: {entry.get('output_dir')}")
            if process:
                print("  process preview:", ", ".join(_preview_list(process)))
            if skip:
                print("  skip preview:", ", ".join(_preview_list(skip)))
        elif stage == "chunk":
            process = entry.get("process", [])
            skip = entry.get("skip", [])
            print(f"- chunk: process {len(process)}, skip {len(skip)}")
            print(f"  input:  {entry.get('input_dir')}")
            print(f"  output: {entry.get('output_dir')}")
            if process:
                print("  process preview:", ", ".join(_preview_list(process)))
            if skip:
                print("  skip preview:", ", ".join(_preview_list(skip)))
        elif stage == "embed" and entry.get("action") == "validate":
            validate = entry.get("validate", [])
            missing = entry.get("missing", [])
            print(
                f"- embed (validate-only): validate {len(validate)}, missing vectors {len(missing)}"
            )
            print(f"  chunks:  {entry.get('chunks_dir')}")
            print(f"  vectors: {entry.get('vectors_dir')}")
            if validate:
                print("  validate preview:", ", ".join(_preview_list(validate)))
            if missing:
                print("  missing preview:", ", ".join(_preview_list(missing)))
        elif stage == "embed":
            process = entry.get("process", [])
            skip = entry.get("skip", [])
            print(f"- embed: process {len(process)}, skip {len(skip)}")
            print(f"  chunks:  {entry.get('chunks_dir')}")
            print(f"  vectors: {entry.get('vectors_dir')}")
            if process:
                print("  process preview:", ", ".join(_preview_list(process)))
            if skip:
                print("  skip preview:", ", ".join(_preview_list(skip)))
        else:
            print(f"- {stage}: no actionable items")
        if notes:
            print("  notes:", "; ".join(notes))
    print()


def _run_all(argv: Sequence[str]) -> int:
    """Execute DocTags conversion, chunking, and embedding sequentially."""

    parser = argparse.ArgumentParser(
        description="Run doctags → chunk → embed in sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="DocsToKG data root override passed to all stages",
    )
    parser.add_argument(
        "--log-level",
        type=lambda value: str(value).upper(),
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity applied to all stages",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each stage by skipping outputs with matching manifests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration in each stage even when outputs exist",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "html", "pdf"],
        default="auto",
        help="DocTags conversion mode",
    )
    parser.add_argument(
        "--doctags-in-dir",
        type=Path,
        default=None,
        help="Override DocTags input directory",
    )
    parser.add_argument(
        "--doctags-out-dir",
        type=Path,
        default=None,
        help="Override DocTags output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rewriting DocTags outputs (HTML mode only)",
    )
    parser.add_argument(
        "--vllm-wait-timeout",
        type=int,
        default=None,
        help="Seconds to wait for vLLM readiness during the DocTags stage",
    )
    parser.add_argument(
        "--chunk-out-dir",
        type=Path,
        default=None,
        help="Output directory override for chunk JSONL files",
    )
    parser.add_argument(
        "--chunk-workers",
        type=int,
        default=None,
        help="Worker processes for the chunk stage",
    )
    parser.add_argument(
        "--chunk-min-tokens",
        type=int,
        default=None,
        help="Minimum tokens per chunk passed to the chunk stage",
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per chunk passed to the chunk stage",
    )
    parser.add_argument(
        "--structural-markers",
        type=Path,
        default=None,
        help="Structural marker configuration forwarded to the chunk stage",
    )
    parser.add_argument(
        "--chunk-shard-count",
        type=int,
        default=None,
        help="Total number of shards for the chunk stage",
    )
    parser.add_argument(
        "--chunk-shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for the chunk stage",
    )
    parser.add_argument(
        "--embed-out-dir",
        type=Path,
        default=None,
        help="Output directory override for embedding JSONL files",
    )
    parser.add_argument(
        "--embed-offline",
        action="store_true",
        help="Run the embedding stage with TRANSFORMERS_OFFLINE=1",
    )
    parser.add_argument(
        "--embed-validate-only",
        action="store_true",
        help="Skip embedding generation and only validate existing vectors",
    )
    parser.add_argument(
        "--splade-sparsity-warn-pct",
        dest="splade_sparsity_warn_pct",
        type=float,
        default=None,
        help="Override SPLADE sparsity warning threshold for the embed stage",
    )
    parser.add_argument(
        "--splade-zero-pct-warn-threshold",
        dest="splade_sparsity_warn_pct",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embed-shard-count",
        type=int,
        default=None,
        help="Total number of shards for the embed stage (defaults to chunk shard count)",
    )
    parser.add_argument(
        "--embed-shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for the embed stage (defaults to chunk shard index)",
    )
    parser.add_argument(
        "--embed-format",
        choices=["jsonl", "parquet"],
        default=None,
        help="Vector output format for the embed stage",
    )
    parser.add_argument(
        "--embed-no-cache",
        action="store_true",
        help="Disable Qwen cache reuse during the embed stage",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show a plan of the files each stage would touch instead of running",
    )

    args = parser.parse_args(argv)
    logger = get_logger(__name__, level=args.log_level)

    chunk_shard_count = args.chunk_shard_count
    chunk_shard_index = args.chunk_shard_index
    embed_shard_count = args.embed_shard_count
    if embed_shard_count is None and chunk_shard_count is not None:
        embed_shard_count = chunk_shard_count
    embed_shard_index = args.embed_shard_index
    if embed_shard_index is None and chunk_shard_index is not None:
        embed_shard_index = chunk_shard_index

    extra = {
        "resume": bool(args.resume),
        "force": bool(args.force),
        "mode": args.mode,
        "log_level": args.log_level,
    }
    if args.data_root:
        extra["data_root"] = str(args.data_root)
    if chunk_shard_count is not None:
        extra["chunk_shard_count"] = chunk_shard_count
    if chunk_shard_index is not None:
        extra["chunk_shard_index"] = chunk_shard_index
    if embed_shard_count is not None:
        extra["embed_shard_count"] = embed_shard_count
    if embed_shard_index is not None:
        extra["embed_shard_index"] = embed_shard_index
    if args.embed_format:
        extra["embed_format"] = args.embed_format
    logger.info("docparse all starting", extra={"extra_fields": extra})

    doctags_args: List[str] = []
    doctags_args.extend(["--log-level", args.log_level])
    if args.data_root:
        doctags_args.extend(["--data-root", str(args.data_root)])
    if args.resume:
        doctags_args.append("--resume")
    if args.force:
        doctags_args.append("--force")
    if args.mode != "auto":
        doctags_args.extend(["--mode", args.mode])
    if args.doctags_in_dir:
        doctags_args.extend(["--in-dir", str(args.doctags_in_dir)])
    if args.doctags_out_dir:
        doctags_args.extend(["--out-dir", str(args.doctags_out_dir)])
    if args.overwrite:
        doctags_args.append("--overwrite")
    if args.vllm_wait_timeout is not None:
        doctags_args.extend(["--vllm-wait-timeout", str(args.vllm_wait_timeout)])

    chunk_args: List[str] = []
    chunk_args.extend(["--log-level", args.log_level])
    if args.data_root:
        chunk_args.extend(["--data-root", str(args.data_root)])
    if args.resume:
        chunk_args.append("--resume")
    if args.force:
        chunk_args.append("--force")
    if args.doctags_out_dir:
        chunk_args.extend(["--in-dir", str(args.doctags_out_dir)])
    if args.chunk_out_dir:
        chunk_args.extend(["--out-dir", str(args.chunk_out_dir)])
    if args.chunk_workers:
        chunk_args.extend(["--workers", str(args.chunk_workers)])
    if args.chunk_min_tokens:
        chunk_args.extend(["--min-tokens", str(args.chunk_min_tokens)])
    if args.chunk_max_tokens:
        chunk_args.extend(["--max-tokens", str(args.chunk_max_tokens)])
    if args.structural_markers:
        chunk_args.extend(["--structural-markers", str(args.structural_markers)])
    if chunk_shard_count is not None:
        chunk_args.extend(["--shard-count", str(chunk_shard_count)])
    if chunk_shard_index is not None:
        chunk_args.extend(["--shard-index", str(chunk_shard_index)])

    embed_args: List[str] = []
    embed_args.extend(["--log-level", args.log_level])
    if args.data_root:
        embed_args.extend(["--data-root", str(args.data_root)])
    if args.resume:
        embed_args.append("--resume")
    if args.force:
        embed_args.append("--force")
    if args.chunk_out_dir:
        embed_args.extend(["--chunks-dir", str(args.chunk_out_dir)])
    if args.embed_out_dir:
        embed_args.extend(["--out-dir", str(args.embed_out_dir)])
    if args.embed_offline:
        embed_args.append("--offline")
    if args.embed_validate_only:
        embed_args.append("--validate-only")
    if args.embed_format:
        embed_args.extend(["--format", args.embed_format])
    if args.embed_no_cache:
        embed_args.append("--no-cache")
    if embed_shard_count is not None:
        embed_args.extend(["--shard-count", str(embed_shard_count)])
    if embed_shard_index is not None:
        embed_args.extend(["--shard-index", str(embed_shard_index)])
    if args.splade_sparsity_warn_pct is not None:
        embed_args.extend(["--splade-sparsity-warn-pct", str(args.splade_sparsity_warn_pct)])

    if args.plan:
        plans: List[Dict[str, Any]] = []
        try:
            plans.append(_plan_doctags(doctags_args))
        except Exception as exc:  # pragma: no cover - plan path should handle gracefully
            plans.append(
                {
                    "stage": "doctags",
                    "mode": args.mode,
                    "input_dir": None,
                    "output_dir": None,
                    "total": 0,
                    "process": [],
                    "skip": [],
                    "notes": [f"DocTags plan unavailable ({exc})"],
                }
            )
        try:
            plans.append(_plan_chunk(chunk_args))
        except Exception as exc:  # pragma: no cover
            plans.append(
                {
                    "stage": "chunk",
                    "input_dir": None,
                    "output_dir": None,
                    "total": 0,
                    "process": [],
                    "skip": [],
                    "notes": [f"Chunk plan unavailable ({exc})"],
                }
            )
        try:
            plans.append(_plan_embed(embed_args))
        except Exception as exc:  # pragma: no cover
            plans.append(
                {
                    "stage": "embed",
                    "operation": "unknown",
                    "chunks_dir": None,
                    "vectors_dir": None,
                    "total": 0,
                    "process": [],
                    "skip": [],
                    "notes": [f"Embed plan unavailable ({exc})"],
                }
            )
        _display_plan(plans)
        return 0

    exit_code = _run_doctags(doctags_args)
    if exit_code != 0:
        logger.error(
            "DocTags stage failed",
            extra={"extra_fields": {"exit_code": exit_code}},
        )
        return exit_code

    exit_code = _run_chunk(chunk_args)
    if exit_code != 0:
        logger.error(
            "Chunk stage failed",
            extra={"extra_fields": {"exit_code": exit_code}},
        )
        return exit_code

    exit_code = _run_embed(embed_args)
    if exit_code != 0:
        logger.error(
            "Embedding stage failed",
            extra={"extra_fields": {"exit_code": exit_code}},
        )
        return exit_code

    logger.info("docparse all completed", extra={"extra_fields": {"status": "success"}})
    return 0


class _Command:
    """Callable wrapper storing handler metadata for subcommands."""

    __slots__ = ("handler", "help")

    def __init__(self, handler: CommandHandler, help: str) -> None:
        """Capture the callable ``handler`` and its CLI help text."""
        self.handler = handler
        self.help = help


COMMANDS: Dict[str, _Command] = {
    "all": _Command(_run_all, "Run doctags → chunk → embed sequentially"),
    "chunk": _Command(_run_chunk, "Run the Docling hybrid chunker"),
    "embed": _Command(_run_embed, "Generate BM25/SPLADE/dense vectors"),
    "doctags": _Command(_run_doctags, "Convert HTML/PDF corpora into DocTags"),
    "token-profiles": _Command(
        _run_token_profiles,
        "Print token count ratios for DocTags samples across tokenizers",
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to one of the DocParsing subcommands."""

    parser = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS.keys(), help="CLI to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the command")
    parsed = parser.parse_args(argv)
    command = COMMANDS[parsed.command]
    return command.handler(parsed.args)


def run_all(argv: Sequence[str] | None = None) -> int:
    """Public wrapper for the ``all`` subcommand."""

    return _run_all([] if argv is None else list(argv))


def chunk(argv: Sequence[str] | None = None) -> int:
    """Public wrapper for the ``chunk`` subcommand."""

    return _run_chunk([] if argv is None else list(argv))


def embed(argv: Sequence[str] | None = None) -> int:
    """Public wrapper for the ``embed`` subcommand."""

    return _run_embed([] if argv is None else list(argv))


def doctags(argv: Sequence[str] | None = None) -> int:
    """Public wrapper for the ``doctags`` subcommand."""

    return _run_doctags([] if argv is None else list(argv))


def token_profiles(argv: Sequence[str] | None = None) -> int:
    """Public wrapper for the ``token-profiles`` subcommand."""

    return _run_token_profiles([] if argv is None else list(argv))
