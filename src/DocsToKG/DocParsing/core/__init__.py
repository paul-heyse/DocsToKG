# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.__init__",
#   "purpose": "Core namespace aggregating shared DocParsing orchestration helpers.",
#   "sections": []
# }
# === /NAVMAP ===

"""Core namespace aggregating shared DocParsing orchestration helpers.

This package surface brings together reusable CLI builders, batching and planning
utilities, manifest writers, filesystem helpers, and environment bootstrap logic
that power every DocParsing stage.

Downstream code can import from ``DocsToKG.DocParsing.core`` to access:
- Process-safe file writes via safe_write()
- Opinionated defaults and resume-safe JSONL writers
- Portable multiprocessing coordination (set_spawn_or_warn, find_free_port)
- Tokenizer and embedding initialization routines
- Manifest and filesystem helpers

All features are available from this top-level namespace without needing to know
which submodule provides each feature.

Example:
    from DocsToKG.DocParsing.core import safe_write, set_spawn_or_warn
    from pathlib import Path

    # Atomically write output with process safety
    safe_write(Path("results.json"), lambda: save_results())

    # Configure subprocess spawning strategy
    set_spawn_or_warn()
"""

from __future__ import annotations

from DocsToKG.DocParsing.env import (
    data_chunks,
    data_doctags,
    data_html,
    data_manifests,
    data_pdfs,
    data_vectors,
    detect_data_root,
    ensure_model_environment,
    ensure_qwen_dependencies,
    ensure_qwen_environment,
    ensure_splade_dependencies,
    ensure_splade_environment,
    expand_path,
    init_hf_env,
    looks_like_filesystem_path,
    prepare_data_root,
    resolve_hf_home,
    resolve_model_root,
    resolve_pdf_model_path,
    resolve_pipeline_path,
)
from DocsToKG.DocParsing.io import (
    atomic_write,
    build_jsonl_split_map,
    compute_chunk_uuid,
    compute_content_hash,
    dedupe_preserve_order,
    iter_doctags,
    iter_jsonl,
    iter_jsonl_batches,
    iter_manifest_entries,
    jsonl_append_iter,
    jsonl_load,
    jsonl_save,
    load_manifest_index,
    make_hasher,
    manifest_append,
    quarantine_artifact,
    relative_path,
    resolve_hash_algorithm,
)
from DocsToKG.DocParsing.logging import (
    StructuredLogger,
    get_logger,
    log_event,
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
    summarize_manifest,
)

from .batching import Batcher
from .cli_utils import CLIOption, build_subcommand, detect_mode
from .concurrency import ReservedPort, find_free_port, safe_write, set_spawn_or_warn
from .discovery import (
    DEFAULT_CAPTION_MARKERS,
    DEFAULT_DISCOVERY_IGNORE,
    DEFAULT_HEADING_MARKERS,
    ChunkDiscovery,
    configure_discovery_ignore,
    compute_relative_doc_id,
    compute_stable_shard,
    derive_doc_id_and_chunks_path,
    derive_doc_id_and_doctags_path,
    derive_doc_id_and_vectors_path,
    get_discovery_ignore_patterns,
    iter_chunks,
    load_structural_marker_config,
    load_structural_marker_profile,
    parse_discovery_ignore,
    split_discovery_ignore,
    vector_artifact_name,
)
from .http import DEFAULT_HTTP_TIMEOUT, get_http_session, normalize_http_timeout
from .manifest import ResumeController, should_skip_output
from .models import (
    DEFAULT_SERIALIZER_PROVIDER,
    DEFAULT_TOKENIZER,
    UUID_NAMESPACE,
    BM25Stats,
    ChunkResult,
    ChunkTask,
    ChunkWorkerConfig,
    QwenCfg,
    SpladeCfg,
)
from .runner import (
    ItemFingerprint,
    ItemOutcome,
    StageContext,
    StageError,
    StageHooks,
    StageOptions,
    StageOutcome,
    StagePlan,
    WorkItem,
    run_stage,
)

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
    "ReservedPort",
    "find_free_port",
    "atomic_write",
    "safe_write",
    "ChunkDiscovery",
    "iter_doctags",
    "iter_chunks",
    "jsonl_load",
    "jsonl_save",
    "jsonl_append_iter",
    "iter_jsonl",
    "iter_jsonl_batches",
    "build_jsonl_split_map",
    "get_logger",
    "log_event",
    "Batcher",
    "compute_chunk_uuid",
    "quarantine_artifact",
    "manifest_append",
    "StructuredLogger",
    "manifest_log_failure",
    "manifest_log_skip",
    "manifest_log_success",
    "compute_content_hash",
    "make_hasher",
    "resolve_hash_algorithm",
    "load_manifest_index",
    "set_spawn_or_warn",
    "derive_doc_id_and_vectors_path",
    "derive_doc_id_and_doctags_path",
    "derive_doc_id_and_chunks_path",
    "compute_relative_doc_id",
    "compute_stable_shard",
    "vector_artifact_name",
    "should_skip_output",
    "relative_path",
    "init_hf_env",
    "ensure_model_environment",
    "ensure_splade_dependencies",
    "ensure_qwen_dependencies",
    "ensure_splade_environment",
    "ensure_qwen_environment",
    "DEFAULT_HTTP_TIMEOUT",
    "normalize_http_timeout",
    "get_http_session",
    "iter_manifest_entries",
    "summarize_manifest",
    "ResumeController",
    "UUID_NAMESPACE",
    "BM25Stats",
    "SpladeCfg",
    "QwenCfg",
    "ChunkWorkerConfig",
    "ChunkTask",
    "ChunkResult",
    "DEFAULT_HEADING_MARKERS",
    "DEFAULT_CAPTION_MARKERS",
    "DEFAULT_DISCOVERY_IGNORE",
    "configure_discovery_ignore",
    "get_discovery_ignore_patterns",
    "parse_discovery_ignore",
    "split_discovery_ignore",
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
    "detect_mode",
    "StagePlan",
    "WorkItem",
    "ItemOutcome",
    "StageOutcome",
    "StageOptions",
    "StageHooks",
    "StageError",
    "StageContext",
    "ItemFingerprint",
    "run_stage",
]
