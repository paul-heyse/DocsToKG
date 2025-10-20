"""Top-level command-line harness for DocsToKG content downloads.

Responsibilities
----------------
- Wire the argument parsing layer in :mod:`DocsToKG.ContentDownload.args` to the
  :class:`~DocsToKG.ContentDownload.runner.DownloadRun` execution engine.
- Configure resolver pipelines, telemetry sinks, and manifest destinations
  before handing control off to the threaded runner.
- Provide a testable :func:`main` entry point that accepts an ``argv`` sequence
  and returns a :class:`~DocsToKG.ContentDownload.summary.RunResult`, making it
  simple for automation and unit tests to capture outcomes.
- Bridge polite networking defaults (OpenAlex mailto, resolver retries) and
  resume semantics so the CLI mirrors production runs.

Key Interactions
----------------
- Delegates configuration synthesis to :func:`resolve_config` and
  :func:`bootstrap_run_environment`.
- Builds resolver pipelines via :mod:`DocsToKG.ContentDownload.pipeline` and
  feeds progress/metrics into :mod:`DocsToKG.ContentDownload.telemetry` sinks.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from DocsToKG.ContentDownload import pipeline as resolvers
from DocsToKG.ContentDownload.args import (
    bootstrap_run_environment,
    build_parser,
    build_query,
    parse_args,
    resolve_config,
    resolve_topic_id_if_needed,
)
from DocsToKG.ContentDownload.core import (
    DEFAULT_MIN_PDF_BYTES,
    DEFAULT_SNIFF_BYTES,
    DEFAULT_TAIL_CHECK_BYTES,
    WorkArtifact,
    classify_payload,
    slugify,
)
from DocsToKG.ContentDownload.download import (
    DownloadConfig,
    DownloadOptions,
    DownloadState,
    _collect_location_urls,
    build_download_outcome,
    create_artifact,
    download_candidate,
    ensure_dir,
    process_one_work,
)
from DocsToKG.ContentDownload.pipeline import (
    ResolverPipeline,
    apply_config_overrides,
    default_resolvers,
    load_resolver_config,
    read_resolver_config,
)
from DocsToKG.ContentDownload.providers import OpenAlexWorkProvider, WorkProvider
from DocsToKG.ContentDownload.pyalex_shim import ConfigProxy
from DocsToKG.ContentDownload.runner import DownloadRun, iterate_openalex
from DocsToKG.ContentDownload.summary import RunResult, emit_console_summary
from DocsToKG.ContentDownload.telemetry import (
    MANIFEST_SCHEMA_VERSION,
    AttemptSink,
    CsvSink,
    JsonlSink,
    LastAttemptCsvSink,
    ManifestEntry,
    ManifestIndexSink,
    MultiSink,
    RotatingJsonlSink,
    RunTelemetry,
    SqliteSink,
    SummarySink,
    load_previous_manifest,
)

__all__ = (
    "AttemptSink",
    "CsvSink",
    "DEFAULT_MIN_PDF_BYTES",
    "DEFAULT_SNIFF_BYTES",
    "DEFAULT_TAIL_CHECK_BYTES",
    "DownloadState",
    "JsonlSink",
    "LastAttemptCsvSink",
    "ManifestEntry",
    "RunTelemetry",
    "ManifestIndexSink",
    "MultiSink",
    "RotatingJsonlSink",
    "SqliteSink",
    "SummarySink",
    "MANIFEST_SCHEMA_VERSION",
    "WorkArtifact",
    "WorkProvider",
    "OpenAlexWorkProvider",
    "ResolverPipeline",
    "resolvers",
    "apply_config_overrides",
    "default_resolvers",
    "build_query",
    "classify_payload",
    "create_artifact",
    "download_candidate",
    "ensure_dir",
    "DownloadRun",
    "iterate_openalex",
    "build_download_outcome",
    "_collect_location_urls",
    "load_previous_manifest",
    "load_resolver_config",
    "main",
    "process_one_work",
    "read_resolver_config",
    "resolve_topic_id_if_needed",
    "slugify",
    "DownloadConfig",
    "DownloadOptions",
    "oa_config",
)

LOGGER = logging.getLogger("DocsToKG.ContentDownload")

oa_config = ConfigProxy()


def main(argv: Optional[Sequence[str]] = None) -> RunResult:
    """CLI entry point that orchestrates parsing, execution, and reporting."""

    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parse_args(parser, argv)
    resolved = resolve_config(args, parser, resolver_factory=default_resolvers)
    bootstrap_run_environment(resolved)
    download_run = DownloadRun(resolved)
    download_run.iterate_openalex_func = iterate_openalex
    download_run.download_candidate_func = download_candidate
    download_run.process_one_work_func = process_one_work
    # Allow tests and callers to monkeypatch sink implementations via cli module.
    download_run.jsonl_sink_factory = JsonlSink
    download_run.rotating_jsonl_sink_factory = RotatingJsonlSink
    download_run.manifest_index_sink_factory = ManifestIndexSink
    download_run.last_attempt_sink_factory = LastAttemptCsvSink
    download_run.sqlite_sink_factory = SqliteSink
    download_run.summary_sink_factory = SummarySink
    download_run.csv_sink_factory = CsvSink
    download_run.multi_sink_factory = MultiSink
    download_run.run_telemetry_factory = RunTelemetry
    result = download_run.run()
    emit_console_summary(result, dry_run=args.dry_run)
    return result


if __name__ == "__main__":
    main()
