Here’s a **Typer signature skeleton** you can paste into `src/DocsToKG/DocParsing/cli.py` (or split across modules and register via `add_typer`). It wires **root/global options**, **shared runner knobs**, and **stage-specific panels** (Doctags, Chunk, Embed), plus the **config show/diff** utilities and **inspect**. It’s intentionally “all signatures, zero business logic” with **TODOs** where your agents should call the settings builder, unified runner, and providers.

```python
# cli.py — Typer signature skeleton (signatures + help panels; no business logic)

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, List

import typer

# ---------------------------
# Typer app & sub-apps
# ---------------------------
app = typer.Typer(
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
    help="Unified DocParsing CLI (profiles + runner + providers).",
)

doctags_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")
chunk_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")
embed_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")
config_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")
inspect_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")

app.add_typer(doctags_app, name="doctags", help="Convert PDF/HTML into DocTags JSONL.")
app.add_typer(chunk_app, name="chunk", help="Chunk DocTags into token-aware units.")
app.add_typer(embed_app, name="embed", help="Embed chunks into vectors (dense/sparse/lexical).")
app.add_typer(config_app, name="config", help="Show/diff effective configuration.")
app.add_typer(inspect_app, name="inspect", help="Inspect parquet datasets quickly.")


# ---------------------------
# Enums for validated choices
# ---------------------------
class LogFormat(str, Enum):
    console = "console"
    json = "json"


class Policy(str, Enum):
    io = "io"
    cpu = "cpu"
    gpu = "gpu"


class Schedule(str, Enum):
    fifo = "fifo"
    sjf = "sjf"


class Adaptive(str, Enum):
    off = "off"
    conservative = "conservative"
    aggressive = "aggressive"


class Format(str, Enum):
    parquet = "parquet"
    jsonl = "jsonl"


class DoctagsMode(str, Enum):
    auto = "auto"
    pdf = "pdf"
    html = "html"


class DenseBackend(str, Enum):
    qwen_vllm = "qwen_vllm"
    tei = "tei"
    sentence_transformers = "sentence_transformers"
    none = "none"


class TeiCompression(str, Enum):
    auto = "auto"
    gzip = "gzip"
    none = "none"


class AttnBackend(str, Enum):
    auto = "auto"
    sdpa = "sdpa"
    eager = "eager"
    flash_attention_2 = "flash_attention_2"


class ConfigStage(str, Enum):
    app = "app"
    runner = "runner"
    doctags = "doctags"
    chunk = "chunk"
    embed = "embed"
    all = "all"


class ConfigFmt(str, Enum):
    yaml = "yaml"
    json = "json"
    toml = "toml"
    env = "env"


class DiffFmt(str, Enum):
    unified = "unified"
    json = "json"
    yaml = "yaml"
    table = "table"


class DatasetKind(str, Enum):
    chunks = "chunks"
    vectors_dense = "vectors-dense"
    vectors_sparse = "vectors-sparse"
    vectors_lexical = "vectors-lexical"


# --------------------------------
# Root callback (GLOBAL options)
# --------------------------------
@app.callback()
def _root(
    ctx: typer.Context,
    profile: Annotated[
        Optional[str],
        typer.Option("--profile", help="Profile name from docstokg.toml/.yaml to layer.")
    ] = None,
    data_root: Annotated[
        Path,
        typer.Option("--data-root", help="Base data directory (Doctags/Chunks/Vectors/Manifests).", rich_help_panel="Global")
    ] = Path("Data"),
    manifests_root: Annotated[
        Optional[Path],
        typer.Option("--manifests-root", help="Override Manifests directory (default: Data/Manifests).", rich_help_panel="Global")
    ] = None,
    models_root: Annotated[
        Optional[Path],
        typer.Option("--models-root", help="Local model cache dir.", rich_help_panel="Global")
    ] = None,
    log_level: Annotated[
        Optional[str],
        typer.Option("--log-level", help="Root log level (DEBUG|INFO|WARNING|ERROR).", rich_help_panel="Global")
    ] = None,
    log_format: Annotated[
        LogFormat,
        typer.Option("--log-format", help="Console or structured JSON.", rich_help_panel="Global")
    ] = LogFormat.console,
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase verbosity (-v=DEBUG, -vv=TRACE).",
            rich_help_panel="Global",
        ),
    ] = 0,
    metrics: Annotated[
        bool,
        typer.Option("--metrics/--no-metrics", help="Expose Prometheus metrics HTTP endpoint.", rich_help_panel="Global")
    ] = False,
    metrics_port: Annotated[
        int,
        typer.Option("--metrics-port", help="Prometheus metrics port.", rich_help_panel="Global")
    ] = 9108,
    tracing: Annotated[
        bool,
        typer.Option("--tracing/--no-tracing", help="Enable OpenTelemetry export.", rich_help_panel="Global")
    ] = False,
    strict_config: Annotated[
        bool,
        typer.Option("--strict-config/--no-strict-config", help="Unknown/deprecated keys are errors (unset to warn).", rich_help_panel="Global")
    ] = True,
):
    """
    Build AppContext (profiles → ENV → CLI), configure logging/telemetry, and stash in ctx.obj.
    """
    # TODO: materialize Settings via your builder:
    # ctx.obj = build_app_context(profile=profile, data_root=data_root, ..., strict_config=strict_config)
    # TODO: configure logging (structlog) + metrics/tracing once here.


# --------------------------------
# Shared runner knobs (helper)
# --------------------------------
def _runner_common_kwargs() -> dict:
    """Only for docstring reference; actual options are declared per command."""
    return {}  # placeholder so IDEs show this section


# ---------------------------
# doctags subcommand
# ---------------------------
@doctags_app.command("run")
def doctags_run(
    ctx: typer.Context,

    # I/O
    input_dir: Annotated[
        Path,
        typer.Option("--input-dir", help="Directory to scan for PDF/HTML.", rich_help_panel="I/O")
    ] = Path("Data/Raw"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Destination for Doctags/{yyyy}/{mm}/*.jsonl", rich_help_panel="I/O")
    ] = Path("Data/Doctags"),
    mode: Annotated[
        DoctagsMode,
        typer.Option("--mode", help="auto=by extension; force pdf|html.", rich_help_panel="I/O")
    ] = DoctagsMode.auto,

    # Engine/model
    model_id: Annotated[
        Optional[str],
        typer.Option("--model-id", help="Docling/Granite model id (PDF path).", rich_help_panel="Engine")
    ] = None,
    vllm_wait_timeout_s: Annotated[
        float,
        typer.Option("--vllm-wait-timeout-s", help="Wait timeout for served VLM (seconds).", rich_help_panel="Engine")
    ] = 60.0,

    # Workflow
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip when outputs+fingerprint match.", rich_help_panel="Workflow")
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Recompute and atomically replace outputs.", rich_help_panel="Workflow")
    ] = False,

    # Runner (per-stage override)
    policy: Annotated[
        Optional[Policy],
        typer.Option("--policy", help="Runner policy (default=io).", rich_help_panel="Runner (this command)")
    ] = None,
    workers: Annotated[
        Optional[int],
        typer.Option("--workers", help="Max parallel files.", rich_help_panel="Runner (this command)")
    ] = None,
    schedule: Annotated[
        Optional[Schedule],
        typer.Option("--schedule", help="fifo | sjf (shortest-job-first).", rich_help_panel="Runner (this command)")
    ] = None,
    retries: Annotated[
        Optional[int],
        typer.Option("--retries", help="Retry attempts for retryable errors.", rich_help_panel="Runner (this command)")
    ] = None,
    retry_backoff_s: Annotated[
        Optional[float],
        typer.Option("--retry-backoff-s", help="Exponential backoff base (sec).", rich_help_panel="Runner (this command)")
    ] = None,
    timeout_s: Annotated[
        Optional[float],
        typer.Option("--timeout-s", help="Per-item timeout (0=disabled).", rich_help_panel="Runner (this command)")
    ] = None,
    error_budget: Annotated[
        Optional[int],
        typer.Option("--error-budget", help="Stop after N failures (0=stop on first).", rich_help_panel="Runner (this command)")
    ] = None,
    max_queue: Annotated[
        Optional[int],
        typer.Option("--max-queue", help="Submission backpressure.", rich_help_panel="Runner (this command)")
    ] = None,
    adaptive: Annotated[
        Optional[Adaptive],
        typer.Option("--adaptive", help="Auto-tune workers (off|conservative|aggressive).", rich_help_panel="Runner (this command)")
    ] = None,
    fingerprinting: Annotated[
        Optional[bool],
        typer.Option("--fingerprinting/--no-fingerprinting", help="Exact resume via *.fp.json.", rich_help_panel="Runner (this command)")
    ] = None,
):
    """Convert sources to DocTags via the unified runner."""
    # TODO: validate inputs; call run_stage(plan=DoctagsPlan(...), worker=..., options=..., hooks=...)


# ---------------------------
# chunk subcommand
# ---------------------------
@chunk_app.command("run")
def chunk_run(
    ctx: typer.Context,

    # I/O
    in_dir: Annotated[
        Path,
        typer.Option("--in-dir", help="DocTags input directory.", rich_help_panel="I/O")
    ] = Path("Data/Doctags"),
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Chunks output dataset.", rich_help_panel="I/O")
    ] = Path("Data/Chunks"),
    fmt: Annotated[
        Format,
        typer.Option("--format", help="parquet (default) | jsonl.", rich_help_panel="I/O")
    ] = Format.parquet,

    # Chunking
    min_tokens: Annotated[
        int,
        typer.Option("--min-tokens", help="Minimum tokens per chunk.", rich_help_panel="Chunking")
    ] = 120,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Maximum tokens per chunk.", rich_help_panel="Chunking")
    ] = 800,
    tokenizer_model: Annotated[
        Optional[str],
        typer.Option("--tokenizer-model", help="Tokenizer id to align with dense embedder.", rich_help_panel="Chunking")
    ] = None,

    # Workflow
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip when outputs+fingerprint match.", rich_help_panel="Workflow")
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Recompute and atomically replace outputs.", rich_help_panel="Workflow")
    ] = False,

    # Runner (per-stage override)
    policy: Annotated[
        Optional[Policy],
        typer.Option("--policy", help="Runner policy (default=cpu).", rich_help_panel="Runner (this command)")
    ] = None,
    workers: Annotated[
        Optional[int],
        typer.Option("--workers", help="Max parallel files.", rich_help_panel="Runner (this command)")
    ] = None,
    schedule: Annotated[
        Optional[Schedule],
        typer.Option("--schedule", help="fifo | sjf.", rich_help_panel="Runner (this command)")
    ] = None,
    retries: Annotated[
        Optional[int],
        typer.Option("--retries", help="Retry attempts.", rich_help_panel="Runner (this command)")
    ] = None,
    retry_backoff_s: Annotated[
        Optional[float],
        typer.Option("--retry-backoff-s", help="Backoff base (sec).", rich_help_panel="Runner (this command)")
    ] = None,
    timeout_s: Annotated[
        Optional[float],
        typer.Option("--timeout-s", help="Per-item timeout.", rich_help_panel="Runner (this command)")
    ] = None,
    error_budget: Annotated[
        Optional[int],
        typer.Option("--error-budget", help="Stop after N failures.", rich_help_panel="Runner (this command)")
    ] = None,
    max_queue: Annotated[
        Optional[int],
        typer.Option("--max-queue", help="Submission backpressure.", rich_help_panel="Runner (this command)")
    ] = None,
    adaptive: Annotated[
        Optional[Adaptive],
        typer.Option("--adaptive", help="Auto-tune workers.", rich_help_panel="Runner (this command)")
    ] = None,
    fingerprinting: Annotated[
        Optional[bool],
        typer.Option("--fingerprinting/--no-fingerprinting", help="Exact resume via *.fp.json.", rich_help_panel="Runner (this command)")
    ] = None,
):
    """Chunk DocTags to token-aware units using the unified runner."""
    # TODO: run_stage(plan=ChunkPlan(...), worker=..., options=..., hooks=None)


# ---------------------------
# embed subcommand
# ---------------------------
@embed_app.command("run")
def embed_run(
    ctx: typer.Context,

    # I/O & families
    chunks_dir: Annotated[
        Path,
        typer.Option("--chunks-dir", help="Chunks dataset root.", rich_help_panel="I/O & Families")
    ] = Path("Data/Chunks"),
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Vectors dataset root.", rich_help_panel="I/O & Families")
    ] = Path("Data/Vectors"),
    vector_format: Annotated[
        Format,
        typer.Option("--vector-format", help="parquet (default) | jsonl.", rich_help_panel="I/O & Families")
    ] = Format.parquet,
    enable_dense: Annotated[
        bool,
        typer.Option("--enable-dense/--no-dense", help="Toggle dense vectors.", rich_help_panel="I/O & Families")
    ] = True,
    enable_sparse: Annotated[
        bool,
        typer.Option("--enable-sparse/--no-sparse", help="Toggle SPLADE vectors.", rich_help_panel="I/O & Families")
    ] = True,
    enable_lexical: Annotated[
        bool,
        typer.Option("--enable-lexical/--no-lexical", help="Toggle BM25 lexical vectors.", rich_help_panel="I/O & Families")
    ] = True,
    plan_only: Annotated[
        bool,
        typer.Option("--plan-only", help="Plan and show summary; do not write vectors.", rich_help_panel="I/O & Families")
    ] = False,

    # Dense provider selection
    dense_backend: Annotated[
        DenseBackend,
        typer.Option("--dense-backend", help="qwen_vllm | tei | sentence_transformers | none.", rich_help_panel="Dense Providers")
    ] = DenseBackend.qwen_vllm,

    # Dense: Qwen vLLM
    qwen_model_id: Annotated[
        Optional[str],
        typer.Option("--qwen-model-id", help="Qwen embedding model id.", rich_help_panel="Dense • Qwen vLLM")
    ] = None,
    qwen_batch_size: Annotated[
        Optional[int],
        typer.Option("--qwen-batch-size", help="Batch size.", rich_help_panel="Dense • Qwen vLLM")
    ] = None,
    qwen_dtype: Annotated[
        Optional[str],
        typer.Option("--qwen-dtype", help="auto|float16|bfloat16.", rich_help_panel="Dense • Qwen vLLM")
    ] = None,
    qwen_device: Annotated[
        Optional[str],
        typer.Option("--qwen-device", help="cpu|cuda|cuda:N.", rich_help_panel="Dense • Qwen vLLM")
    ] = None,
    qwen_tp: Annotated[
        Optional[int],
        typer.Option("--qwen-tp", help="Tensor parallelism.", rich_help_panel="Dense • Qwen vLLM")
    ] = None,
    qwen_max_model_len: Annotated[
        Optional[int],
        typer.Option("--qwen-max-model-len", help="Context length.", rich_help_panel="Dense • Qwen vLLM")
    ] = None,
    qwen_gpu_mem_util: Annotated[
        Optional[float],
        typer.Option("--qwen-gpu-mem-util", help="GPU memory utilization (0.1–0.99).", rich_help_panel="Dense • Qwen vLLM")
    ] = None,
    qwen_download_dir: Annotated[
        Optional[Path],
        typer.Option("--qwen-download-dir", help="Model cache dir.", rich_help_panel="Dense • Qwen vLLM")
    ] = None,
    qwen_warmup: Annotated[
        Optional[bool],
        typer.Option("--qwen-warmup/--no-qwen-warmup", help="Run warmup batch on open().", rich_help_panel="Dense • Qwen vLLM")
    ] = None,

    # Dense: TEI
    tei_url: Annotated[
        Optional[str],
        typer.Option("--tei-url", help="TEI base URL (required when backend=tei).", rich_help_panel="Dense • TEI")
    ] = None,
    tei_api_key: Annotated[
        Optional[str],
        typer.Option("--tei-api-key", help="Auth token (redacted).", rich_help_panel="Dense • TEI")
    ] = None,
    tei_timeout_s: Annotated[
        Optional[float],
        typer.Option("--tei-timeout-s", help="HTTP timeout seconds.", rich_help_panel="Dense • TEI")
    ] = None,
    tei_verify_tls: Annotated[
        Optional[bool],
        typer.Option("--tei-verify-tls/--no-tei-verify-tls", help="Verify TLS.", rich_help_panel="Dense • TEI")
    ] = None,
    tei_compression: Annotated[
        Optional[TeiCompression],
        typer.Option("--tei-compression", help="auto|gzip|none.", rich_help_panel="Dense • TEI")
    ] = None,
    tei_max_inflight: Annotated[
        Optional[int],
        typer.Option("--tei-max-inflight", help="Client-side concurrency cap.", rich_help_panel="Dense • TEI")
    ] = None,
    tei_batch_size: Annotated[
        Optional[int],
        typer.Option("--tei-batch-size", help="Batch size per request.", rich_help_panel="Dense • TEI")
    ] = None,

    # Dense: Sentence-Transformers
    st_model_id: Annotated[
        Optional[str],
        typer.Option("--st-model-id", help="Sentence-Transformers model id.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,
    st_revision: Annotated[
        Optional[str],
        typer.Option("--st-revision", help="HF revision/tag.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,
    st_device: Annotated[
        Optional[str],
        typer.Option("--st-device", help="auto|cpu|cuda|cuda:N.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,
    st_dtype: Annotated[
        Optional[str],
        typer.Option("--st-dtype", help="auto|float32|float16|bfloat16.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,
    st_batch_size: Annotated[
        Optional[int],
        typer.Option("--st-batch-size", help="Batch size.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,
    st_max_seq_len: Annotated[
        Optional[int],
        typer.Option("--st-max-seq-len", help="Max sequence length.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,
    st_trust_remote_code: Annotated[
        Optional[bool],
        typer.Option("--st-trust-remote-code/--no-st-trust-remote-code", help="HF trust_remote_code.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,
    st_use_mmap: Annotated[
        Optional[bool],
        typer.Option("--st-use-mmap/--no-st-use-mmap", help="Memory-map weights.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,
    st_threads: Annotated[
        Optional[int],
        typer.Option("--st-threads", help="CPU intra-op threads.", rich_help_panel="Dense • Sentence-Transformers")
    ] = None,

    # Sparse: SPLADE
    splade_model_id: Annotated[
        Optional[str],
        typer.Option("--splade-model-id", help="SPLADE model id.", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_revision: Annotated[
        Optional[str],
        typer.Option("--splade-revision", help="HF revision/tag.", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_device: Annotated[
        Optional[str],
        typer.Option("--splade-device", help="auto|cpu|cuda|cuda:N.", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_dtype: Annotated[
        Optional[str],
        typer.Option("--splade-dtype", help="auto|float32|float16|bfloat16.", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_batch_size: Annotated[
        Optional[int],
        typer.Option("--splade-batch-size", help="Batch size.", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_topk: Annotated[
        Optional[int],
        typer.Option("--splade-topk", help="Keep top-k per doc (0=all).", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_prune_below: Annotated[
        Optional[float],
        typer.Option("--splade-prune-below", help="Drop weights below threshold.", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_norm: Annotated[
        Optional[str],
        typer.Option("--splade-norm", help="none|l2.", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_tokenizer: Annotated[
        Optional[str],
        typer.Option("--splade-tokenizer", help="Tokenizer id override.", rich_help_panel="Sparse • SPLADE")
    ] = None,
    splade_attn: Annotated[
        Optional[AttnBackend],
        typer.Option("--splade-attn", help="auto|sdpa|eager|flash_attention_2.", rich_help_panel="Sparse • SPLADE")
    ] = None,

    # Lexical: BM25
    bm25_k1: Annotated[
        Optional[float],
        typer.Option("--bm25-k1", help="> 0.", rich_help_panel="Lexical • BM25")
    ] = None,
    bm25_b: Annotated[
        Optional[float],
        typer.Option("--bm25-b", help="0..1.", rich_help_panel="Lexical • BM25")
    ] = None,
    bm25_stopwords: Annotated[
        Optional[str],
        typer.Option("--bm25-stopwords", help="none|english|PATH.", rich_help_panel="Lexical • BM25")
    ] = None,
    bm25_tokenizer: Annotated[
        Optional[str],
        typer.Option("--bm25-tokenizer", help="simple|spacy_en|regexp:...", rich_help_panel="Lexical • BM25")
    ] = None,
    bm25_min_df: Annotated[
        Optional[int],
        typer.Option("--bm25-min-df", help="≥ 0.", rich_help_panel="Lexical • BM25")
    ] = None,
    bm25_max_df_ratio: Annotated[
        Optional[float],
        typer.Option("--bm25-max-df-ratio", help="(0,1].", rich_help_panel="Lexical • BM25")
    ] = None,

    # Workflow
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip when outputs+fingerprint match.", rich_help_panel="Workflow")
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Recompute and atomically replace outputs.", rich_help_panel="Workflow")
    ] = False,

    # Runner (per-stage override)
    policy: Annotated[
        Optional[Policy],
        typer.Option("--policy", help="Runner policy (default=gpu).", rich_help_panel="Runner (this command)")
    ] = None,
    workers: Annotated[
        Optional[int],
        typer.Option("--workers", help="Max parallel files.", rich_help_panel="Runner (this command)")
    ] = None,
    schedule: Annotated[
        Optional[Schedule],
        typer.Option("--schedule", help="fifo | sjf.", rich_help_panel="Runner (this command)")
    ] = None,
    retries: Annotated[
        Optional[int],
        typer.Option("--retries", help="Retry attempts.", rich_help_panel="Runner (this command)")
    ] = None,
    retry_backoff_s: Annotated[
        Optional[float],
        typer.Option("--retry-backoff-s", help="Backoff base (sec).", rich_help_panel="Runner (this command)")
    ] = None,
    timeout_s: Annotated[
        Optional[float],
        typer.Option("--timeout-s", help="Per-item timeout.", rich_help_panel="Runner (this command)")
    ] = None,
    error_budget: Annotated[
        Optional[int],
        typer.Option("--error-budget", help="Stop after N failures.", rich_help_panel="Runner (this command)")
    ] = None,
    max_queue: Annotated[
        Optional[int],
        typer.Option("--max-queue", help="Submission backpressure.", rich_help_panel="Runner (this command)")
    ] = None,
    adaptive: Annotated[
        Optional[Adaptive],
        typer.Option("--adaptive", help="Auto-tune workers.", rich_help_panel="Runner (this command)")
    ] = None,
    fingerprinting: Annotated[
        Optional[bool],
        typer.Option("--fingerprinting/--no-fingerprinting", help="Exact resume via *.fp.json.", rich_help_panel="Runner (this command)")
    ] = None,
):
    """Embed chunks into vectors via providers behind a unified runner."""
    # TODO: build EmbedCfg from ctx.obj, merge CLI overrides, run_stage(plan=EmbedPlan(...), ...)


# ---------------------------
# all subcommand
# ---------------------------
@app.command("all")
def run_all(
    ctx: typer.Context,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Apply resume to all stages.", rich_help_panel="Workflow")
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Apply force to all stages.", rich_help_panel="Workflow")
    ] = False,
    stop_on_fail: Annotated[
        bool,
        typer.Option("--stop-on-fail/--keep-going", help="Cancel downstream stages if one fails.", rich_help_panel="Workflow")
    ] = True,

    # Optional: a minimal subset of runner knobs to apply to all stages
    workers: Annotated[
        Optional[int],
        typer.Option("--workers", help="Apply to all stages unless overridden.", rich_help_panel="Runner (all)")
    ] = None,
    schedule: Annotated[
        Optional[Schedule],
        typer.Option("--schedule", help="Apply to all stages unless overridden.", rich_help_panel="Runner (all)")
    ] = None,
):
    """Run Doctags → Chunk → Embed sequentially using the unified runner."""
    # TODO: orchestrate stage invocations; propagate ctx.obj configs; short-circuit on failure if stop_on_fail.


# ---------------------------
# config group: show / diff
# ---------------------------
@config_app.command("show")
def config_show(
    ctx: typer.Context,
    profile: Annotated[
        Optional[str],
        typer.Option("--profile", help="Profile to apply while showing.", rich_help_panel="Profiles")
    ] = None,
    stage: Annotated[
        ConfigStage,
        typer.Option("--stage", help="Restrict to a section.", rich_help_panel="Output")
    ] = ConfigStage.all,
    fmt: Annotated[
        ConfigFmt,
        typer.Option("--format", help="Output format.", rich_help_panel="Output")
    ] = ConfigFmt.yaml,
    annotate_source: Annotated[
        bool,
        typer.Option("--annotate-source/--no-annotate-source", help="Show per-key origin (default off).", rich_help_panel="Output")
    ] = False,
    redact: Annotated[
        bool,
        typer.Option("--redact/--no-redact", help="Redact tokens/keys.", rich_help_panel="Output")
    ] = True,
):
    """Print the effective configuration (after profile + ENV + CLI)."""
    # TODO: rebuild effective config the same way root callback does; print in chosen format.


@config_app.command("diff")
def config_diff(
    ctx: typer.Context,

    # LHS
    lhs_profile: Annotated[
        str,
        typer.Option("--lhs-profile", help="Left profile name or 'none'.", rich_help_panel="LHS")
    ] = "none",
    lhs_file: Annotated[
        Optional[Path],
        typer.Option("--lhs-file", help="Optional profile file for LHS.", rich_help_panel="LHS")
    ] = None,
    lhs_env_file: Annotated[
        Optional[Path],
        typer.Option("--lhs-env-file", help="Optional .env to simulate ENV layer for LHS.", rich_help_panel="LHS")
    ] = None,
    lhs_override: Annotated[
        List[str],
        typer.Option("--lhs-override", help="Dot-path KEY=VALUE overrides (repeatable).", rich_help_panel="LHS")
    ] = [],

    # RHS
    rhs_profile: Annotated[
        str,
        typer.Option("--rhs-profile", help="Right profile name or 'none'.", rich_help_panel="RHS")
    ] = "gpu",
    rhs_file: Annotated[
        Optional[Path],
        typer.Option("--rhs-file", help="Optional profile file for RHS.", rich_help_panel="RHS")
    ] = None,
    rhs_env_file: Annotated[
        Optional[Path],
        typer.Option("--rhs-env-file", help="Optional .env to simulate ENV layer for RHS.", rich_help_panel="RHS")
    ] = None,
    rhs_override: Annotated[
        List[str],
        typer.Option("--rhs-override", help="Dot-path KEY=VALUE overrides (repeatable).", rich_help_panel="RHS")
    ] = [],

    # Output
    stage: Annotated[
        ConfigStage,
        typer.Option("--stage", help="Restrict diff to section.", rich_help_panel="Output")
    ] = ConfigStage.all,
    fmt: Annotated[
        DiffFmt,
        typer.Option("--format", help="Diff format (unified|json|yaml|table).", rich_help_panel="Output")
    ] = DiffFmt.unified,
    show_hash: Annotated[
        bool,
        typer.Option("--show-hash/--no-show-hash", help="Print per-stage cfg_hash.", rich_help_panel="Output")
    ] = True,
):
    """Diff two effective configurations (defaults → profile → env → overrides) without running."""
    # TODO: materialize both configs independently, compute changes + cfg_hash deltas, render in chosen format.


# ---------------------------
# inspect subcommand
# ---------------------------
@inspect_app.command("dataset")
def inspect_dataset(
    ctx: typer.Context,
    dataset: Annotated[
        DatasetKind,
        typer.Option("--dataset", help="Which dataset to scan.", rich_help_panel="Inspect")
    ] = DatasetKind.chunks,
    root: Annotated[
        Path,
        typer.Option("--root", help="Dataset base (defaults to app.data_root).", rich_help_panel="Inspect")
    ] = Path("Data"),
    columns: Annotated[
        Optional[List[str]],
        typer.Option("--columns", help="Projection columns (comma-repeat).", rich_help_panel="Inspect")
    ] = None,
    filters: Annotated[
        Optional[str],
        typer.Option("--filters", help="Optional pushdown expression.", rich_help_panel="Inspect")
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Row limit (0=all).", rich_help_panel="Inspect")
    ] = 0,
    stats: Annotated[
        bool,
        typer.Option("--stats/--no-stats", help="Include row/file/size stats.", rich_help_panel="Inspect")
    ] = True,
):
    """Quickly print schema, partitions, and counts for a dataset (parquet)."""
    # TODO: call DatasetView.open_* and render a concise summary.


# ---------------------------
# Entry point
# ---------------------------
def main():
    app()


if __name__ == "__main__":
    main()
```

### Notes for your agents

* **Implementation hooks (TODOs):**

  * Root callback → call your unified **Settings builder** (profile → ENV → CLI), configure structlog, and stash an `AppContext` in `ctx.obj`.
  * Stage commands → merge CLI overrides onto the `ctx.obj` slices, then call the **unified runner** with each stage’s **plan/worker/hooks** (PR-5), using providers (PR-4).
  * `config show|diff` → reuse the same builder to materialize the effective config(s) and print/diff; add redaction and per-key origin if you enabled source annotations.
  * `inspect` → call your PR-8 **DatasetView** helpers.

* **Help grouping:** controlled via `rich_help_panel`. Panels included: **Global**, **Runner (this command)**, **I/O**, **Chunking**, **Engine**, **I/O & Families**, **Dense • Qwen vLLM**, **Dense • TEI**, **Dense • Sentence-Transformers**, **Sparse • SPLADE**, **Lexical • BM25**, **Workflow**, **Inspect**.

* **Legacy aliases:** if you want hidden compatibility flags (e.g., legacy `--bm25-k1`), add an extra parameter name to the same option (Click supports multiple option strings) and set `hidden=True` on the alias; inside the handler, print a one-line deprecation. Example:

  ```python
  bm25_k1: Annotated[
      Optional[float],
      typer.Option("--bm25-k1", "--bm25_k1", hidden=False, help="> 0.", rich_help_panel="Lexical • BM25")
  ] = None
  ```

  (Set `hidden=True` after a transition period.)

If you want, I can also generate a **minimal “builder glue”** (how to map these option values into your Pydantic Settings models and runner `StageOptions`) so you can plug it in with virtually no hand wiring.
