"""CLI parser utilities for configuring the DocParsing embedding stage.

The embedding CLI mirrors the chunking interface but introduces additional
flags for dense/sparse backends, shard coordination, and cache management. This
module defines the reusable option tuples and parser constructors consumed by
``docparse embed`` and orchestration scripts so vector generation runs are
configured consistently across environments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from DocsToKG.DocParsing.config import parse_args_with_overrides
from DocsToKG.DocParsing.core import CLIOption, build_subcommand
from DocsToKG.DocParsing.doctags import add_data_root_option, add_resume_force_options

from .config import EMBED_PROFILE_PRESETS, SPLADE_SPARSITY_WARN_THRESHOLD_PCT

EMBED_CLI_OPTIONS: Tuple[CLIOption, ...] = (
    CLIOption(
        ("--config",),
        {"type": Path, "default": None, "help": "Path to stage config file (JSON/YAML/TOML)."},
    ),
    CLIOption(
        ("--profile",),
        {
            "type": str,
            "default": None,
            "choices": sorted(EMBED_PROFILE_PRESETS),
            "help": "Preset controlling batch sizes, SPLADE backend, and Qwen settings.",
        },
    ),
    CLIOption(
        ("--log-level",),
        {
            "type": lambda value: str(value).upper(),
            "default": "INFO",
            "choices": ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
            "help": "Logging verbosity for console output (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--no-cache",),
        {"action": "store_true", "help": "Disable Qwen LLM caching between batches (debug)."},
    ),
    CLIOption(
        ("--shard-count",),
        {
            "type": int,
            "default": 1,
            "help": "Total number of shards for distributed runs (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--shard-index",),
        {
            "type": int,
            "default": 0,
            "help": "Zero-based shard index to process (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--chunks-dir",),
        {
            "type": Path,
            "default": None,
            "help": "Override path to chunk files (auto-detected relative to data root).",
        },
    ),
    CLIOption(
        ("--out-dir", "--vectors-dir"),
        {
            "type": Path,
            "default": None,
            "help": "Directory where vector outputs will be written (auto-detected).",
        },
    ),
    CLIOption(
        ("--vector-format", "--format"),
        {
            "type": str,
            "default": "jsonl",
            "choices": ["jsonl", "parquet"],
            "help": (
                "Vector output format (default: %(default)s). "
                "Parquet requires the pyarrow dependency."
            ),
        },
    ),
    CLIOption(("--bm25-k1",), {"type": float, "default": 1.5}),
    CLIOption(("--bm25-b",), {"type": float, "default": 0.75}),
    CLIOption(("--batch-size-splade",), {"type": int, "default": 32}),
    CLIOption(("--batch-size-qwen",), {"type": int, "default": 64}),
    CLIOption(("--splade-max-active-dims",), {"type": int, "default": None}),
    CLIOption(
        ("--splade-model-dir",),
        {
            "type": Path,
            "default": None,
            "help": "Explicit path to the SPLADE model directory (defaults to DocsToKG cache).",
        },
    ),
    CLIOption(
        ("--splade-attn",),
        {
            "type": str,
            "default": "auto",
            "choices": ["auto", "flash", "sdpa", "eager"],
            "help": (
                "SPLADE attention backend preference order (default: %(default)s). "
                "Auto first attempts FlashAttention 2, then scaled dot-product attention (SDPA) "
                "before falling back to eager/standard attention."
            ),
        },
    ),
    CLIOption(
        ("--qwen-dtype",),
        {
            "type": str,
            "default": "bfloat16",
            "choices": ["float32", "bfloat16", "float16", "int8"],
            "help": "Qwen inference dtype (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--qwen-quant",),
        {
            "type": str,
            "default": None,
            "help": "Optional Qwen quantisation preset (nf4, awq, ...).",
        },
    ),
    CLIOption(
        ("--qwen-model-dir",),
        {
            "type": Path,
            "default": None,
            "help": "Explicit path to the Qwen model directory (defaults to DocsToKG cache).",
        },
    ),
    CLIOption(("--qwen-dim",), {"type": int, "default": 2560}),
    CLIOption(("--tp", "--tensor-parallel"), {"type": int, "default": 1}),
    CLIOption(
        ("--sparsity-warn-threshold-pct",),
        {
            "type": float,
            "default": SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
            "help": (
                "Warn when SPLADE sparsity exceeds this percent of active dims "
                f"(default: {SPLADE_SPARSITY_WARN_THRESHOLD_PCT})."
            ),
        },
    ),
    CLIOption(("--sparsity-report-top-n",), {"type": int, "default": 10}),
    CLIOption(
        ("--files-parallel",),
        {
            "type": int,
            "default": 1,
            "help": "Process up to N chunk files concurrently during embedding (default: 1 for serial runs).",
        },
    ),
    CLIOption(
        ("--validate-only",),
        {"action": "store_true", "help": "Validate existing vectors in --out-dir and exit."},
    ),
    CLIOption(
        ("--plan-only",),
        {
            "action": "store_true",
            "help": "Show resume/skip plan and exit without generating embeddings.",
        },
    ),
    CLIOption(
        ("--offline",),
        {
            "action": "store_true",
            "help": (
                "Disable network access by setting TRANSFORMERS_OFFLINE=1. "
                "All models must already exist in local caches."
            ),
        },
    ),
    CLIOption(
        ("--embedding-device",),
        {"type": str, "default": None, "dest": "embedding_device", "help": "Preferred device for providers (auto, cpu, cuda, cuda:N)."},
    ),
    CLIOption(
        ("--embedding-dtype",),
        {"type": str, "default": None, "dest": "embedding_dtype", "help": "Preferred dtype for dense providers (auto, float32, float16, bfloat16)."},
    ),
    CLIOption(
        ("--embedding-batch-size",),
        {"type": int, "default": None, "dest": "embedding_batch_size", "help": "Global batch size hint shared by providers."},
    ),
    CLIOption(
        ("--embedding-max-concurrency",),
        {"type": int, "default": None, "dest": "embedding_max_concurrency", "help": "Maximum in-flight batches per provider (defaults to files-parallel)."},
    ),
    CLIOption(
        ("--embedding-normalize-l2",),
        {"type": str, "default": None, "dest": "embedding_normalize_l2", "help": "Enable/disable L2 normalisation for dense outputs (true/false)."},
    ),
    CLIOption(
        ("--embedding-cache-dir",),
        {"type": Path, "default": None, "dest": "embedding_cache_dir", "help": "Shared cache directory for provider artifacts."},
    ),
    CLIOption(
        ("--embedding-telemetry-tag",),
        {
            "action": "append",
            "default": None,
            "dest": "embedding_telemetry_tags",
            "metavar": "KEY=VALUE",
            "help": "Attach custom telemetry tags (repeatable).",
        },
    ),
    CLIOption(
        ("--dense-backend",),
        {
            "type": str,
            "default": None,
            "dest": "dense_backend",
            "choices": ["qwen_vllm", "sentence_transformers", "tei", "none"],
            "help": "Dense embedding backend (default: qwen_vllm).",
        },
    ),
    CLIOption(
        ("--dense-fallback-backend",),
        {
            "type": str,
            "default": None,
            "dest": "dense_fallback_backend",
            "help": "Optional dense backend fallback to try when the primary fails.",
        },
    ),
    CLIOption(
        ("--dense-qwen-model-id",),
        {
            "type": str,
            "default": None,
            "dest": "dense_qwen_vllm_model_id",
            "help": "Qwen model identifier (defaults to Qwen/Qwen3-Embedding-4B).",
        },
    ),
    CLIOption(
        ("--dense-qwen-download-dir",),
        {
            "type": Path,
            "default": None,
            "dest": "dense_qwen_vllm_download_dir",
            "help": "Directory containing the Qwen weights (defaults to DocsToKG cache).",
        },
    ),
    CLIOption(
        ("--dense-qwen-batch-size",),
        {"type": int, "default": None, "dest": "dense_qwen_vllm_batch_size", "help": "Batch size override for Qwen embeddings."},
    ),
    CLIOption(
        ("--dense-qwen-queue-depth",),
        {"type": int, "default": None, "dest": "dense_qwen_vllm_queue_depth", "help": "Max queued batches for Qwen embedding worker."},
    ),
    CLIOption(
        ("--dense-qwen-quantization",),
        {"type": str, "default": None, "dest": "dense_qwen_vllm_quantization", "help": "Quantization preset for Qwen (e.g., nf4, awq)."},
    ),
    CLIOption(
        ("--dense-qwen-dimension",),
        {"type": int, "default": None, "dest": "dense_qwen_vllm_dimension", "help": "Expected dense vector dimension (defaults to 2560)."},
    ),
    CLIOption(
        ("--dense-tei-url",),
        {"type": str, "default": None, "dest": "dense_tei_url", "help": "Base URL for TEI embedding service."},
    ),
    CLIOption(
        ("--dense-tei-timeout",),
        {"type": float, "default": None, "dest": "dense_tei_timeout_seconds", "help": "Per-request timeout when using TEI backend (seconds)."},
    ),
    CLIOption(
        ("--dense-tei-max-inflight",),
        {"type": int, "default": None, "dest": "dense_tei_max_inflight", "help": "Maximum concurrent TEI requests."},
    ),
    CLIOption(
        ("--dense-sentence-transformers-model-id",),
        {
            "type": str,
            "default": None,
            "dest": "dense_sentence_transformers_model_id",
            "help": "SentenceTransformer model identifier for dense backend.",
        },
    ),
    CLIOption(
        ("--dense-sentence-transformers-batch-size",),
        {
            "type": int,
            "default": None,
            "dest": "dense_sentence_transformers_batch_size",
            "help": "Batch size override for sentence-transformers backend.",
        },
    ),
    CLIOption(
        ("--dense-sentence-transformers-normalize-l2",),
        {
            "type": str,
            "default": None,
            "dest": "dense_sentence_transformers_normalize_l2",
            "help": "Enable/disable L2 normalization for sentence-transformers backend (true/false).",
        },
    ),
    CLIOption(
        ("--sparse-backend",),
        {
            "type": str,
            "default": None,
            "dest": "sparse_backend",
            "choices": ["splade_st", "none"],
            "help": "Sparse embedding backend (default: splade_st).",
        },
    ),
    CLIOption(
        ("--sparse-splade-model-dir",),
        {
            "type": Path,
            "default": None,
            "dest": "sparse_splade_st_model_dir",
            "help": "Path to SPLADE model directory (defaults to DocsToKG cache).",
        },
    ),
    CLIOption(
        ("--sparse-splade-batch-size",),
        {
            "type": int,
            "default": None,
            "dest": "sparse_splade_st_batch_size",
            "help": "Batch size override for SPLADE backend.",
        },
    ),
    CLIOption(
        ("--sparse-splade-attn-backend",),
        {
            "type": str,
            "default": None,
            "dest": "sparse_splade_st_attn_backend",
            "help": "Attention backend override for SPLADE (auto/flash/sdpa/eager).",
        },
    ),
    CLIOption(
        ("--sparse-splade-max-active-dims",),
        {
            "type": int,
            "default": None,
            "dest": "sparse_splade_st_max_active_dims",
            "help": "Override SPLADE max active dimensions.",
        },
    ),
    CLIOption(
        ("--lexical-backend",),
        {
            "type": str,
            "default": None,
            "dest": "lexical_backend",
            "choices": ["local_bm25", "none"],
            "help": "Lexical provider (default: local_bm25).",
        },
    ),
    CLIOption(
        ("--lexical-local-bm25-k1",),
        {"type": float, "default": None, "dest": "lexical_local_bm25_k1", "help": "Override BM25 k1 parameter."},
    ),
    CLIOption(
        ("--lexical-local-bm25-b",),
        {"type": float, "default": None, "dest": "lexical_local_bm25_b", "help": "Override BM25 b parameter."},
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the embedding pipeline."""

    parser = argparse.ArgumentParser()
    add_data_root_option(parser)
    build_subcommand(parser, EMBED_CLI_OPTIONS)
    add_resume_force_options(
        parser,
        resume_help="Skip chunk files whose vector outputs already exist with matching hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone embedding execution."""

    parser = build_parser()
    return parse_args_with_overrides(parser, argv)


__all__ = ["EMBED_CLI_OPTIONS", "build_parser", "parse_args"]
