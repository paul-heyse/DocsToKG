"""Command-line entry point for DocsToKG content downloads."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from pyalex import config as oa_config

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
    DownloadOptions,
    DownloadState,
    _build_download_outcome,
    build_download_outcome,
    create_artifact,
    download_candidate,
    ensure_dir,
    process_one_work,
)
from DocsToKG.ContentDownload.pipeline import (
    apply_config_overrides,
    default_resolvers,
    load_resolver_config,
    read_resolver_config,
)
from DocsToKG.ContentDownload.providers import OpenAlexWorkProvider, WorkProvider
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
    RunTelemetry,
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
    "MANIFEST_SCHEMA_VERSION",
    "WorkArtifact",
    "WorkProvider",
    "OpenAlexWorkProvider",
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
    "_build_download_outcome",
    "load_previous_manifest",
    "load_resolver_config",
    "main",
    "process_one_work",
    "read_resolver_config",
    "resolve_topic_id_if_needed",
    "slugify",
    "DownloadOptions",
    "oa_config",
)

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


def main(argv: Optional[Sequence[str]] = None) -> RunResult:
    """CLI entry point that orchestrates parsing, execution, and reporting."""

    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parse_args(parser, argv)
    resolved = resolve_config(args, parser)
    bootstrap_run_environment(resolved)
    download_run = DownloadRun(resolved)
    result = download_run.run()
    emit_console_summary(result, dry_run=args.dry_run)
    return result


if __name__ == "__main__":
    main()
