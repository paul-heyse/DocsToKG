# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.ontology_download",
#   "purpose": "Implements DocsToKG.OntologyDownload.ontology_download behaviors and helpers",
#   "sections": [
#     {
#       "id": "retry-with-backoff",
#       "name": "retry_with_backoff",
#       "anchor": "function-retry-with-backoff",
#       "kind": "function"
#     },
#     {
#       "id": "sanitize-filename",
#       "name": "sanitize_filename",
#       "anchor": "function-sanitize-filename",
#       "kind": "function"
#     },
#     {
#       "id": "generate-correlation-id",
#       "name": "generate_correlation_id",
#       "anchor": "function-generate-correlation-id",
#       "kind": "function"
#     },
#     {
#       "id": "mask-sensitive-data",
#       "name": "mask_sensitive_data",
#       "anchor": "function-mask-sensitive-data",
#       "kind": "function"
#     },
#     {
#       "id": "configerror",
#       "name": "ConfigError",
#       "anchor": "class-configerror",
#       "kind": "class"
#     },
#     {
#       "id": "ensure-python-version",
#       "name": "ensure_python_version",
#       "anchor": "function-ensure-python-version",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-sequence",
#       "name": "_coerce_sequence",
#       "anchor": "function-coerce-sequence",
#       "kind": "function"
#     },
#     {
#       "id": "loggingconfiguration",
#       "name": "LoggingConfiguration",
#       "anchor": "class-loggingconfiguration",
#       "kind": "class"
#     },
#     {
#       "id": "validationconfig",
#       "name": "ValidationConfig",
#       "anchor": "class-validationconfig",
#       "kind": "class"
#     },
#     {
#       "id": "parse-rate-limit-to-rps",
#       "name": "parse_rate_limit_to_rps",
#       "anchor": "function-parse-rate-limit-to-rps",
#       "kind": "function"
#     },
#     {
#       "id": "downloadconfiguration",
#       "name": "DownloadConfiguration",
#       "anchor": "class-downloadconfiguration",
#       "kind": "class"
#     },
#     {
#       "id": "defaultsconfig",
#       "name": "DefaultsConfig",
#       "anchor": "class-defaultsconfig",
#       "kind": "class"
#     },
#     {
#       "id": "resolvedconfig",
#       "name": "ResolvedConfig",
#       "anchor": "class-resolvedconfig",
#       "kind": "class"
#     },
#     {
#       "id": "environmentoverrides",
#       "name": "EnvironmentOverrides",
#       "anchor": "class-environmentoverrides",
#       "kind": "class"
#     },
#     {
#       "id": "get-env-overrides",
#       "name": "get_env_overrides",
#       "anchor": "function-get-env-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "apply-env-overrides",
#       "name": "_apply_env_overrides",
#       "anchor": "function-apply-env-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "make-fetch-spec",
#       "name": "_make_fetch_spec",
#       "anchor": "function-make-fetch-spec",
#       "kind": "function"
#     },
#     {
#       "id": "merge-defaults",
#       "name": "merge_defaults",
#       "anchor": "function-merge-defaults",
#       "kind": "function"
#     },
#     {
#       "id": "build-resolved-config",
#       "name": "build_resolved_config",
#       "anchor": "function-build-resolved-config",
#       "kind": "function"
#     },
#     {
#       "id": "validate-schema",
#       "name": "_validate_schema",
#       "anchor": "function-validate-schema",
#       "kind": "function"
#     },
#     {
#       "id": "load-raw-yaml",
#       "name": "load_raw_yaml",
#       "anchor": "function-load-raw-yaml",
#       "kind": "function"
#     },
#     {
#       "id": "load-config",
#       "name": "load_config",
#       "anchor": "function-load-config",
#       "kind": "function"
#     },
#     {
#       "id": "validate-config",
#       "name": "validate_config",
#       "anchor": "function-validate-config",
#       "kind": "function"
#     },
#     {
#       "id": "jsonformatter",
#       "name": "JSONFormatter",
#       "anchor": "class-jsonformatter",
#       "kind": "class"
#     },
#     {
#       "id": "compress-old-log",
#       "name": "_compress_old_log",
#       "anchor": "function-compress-old-log",
#       "kind": "function"
#     },
#     {
#       "id": "cleanup-logs",
#       "name": "_cleanup_logs",
#       "anchor": "function-cleanup-logs",
#       "kind": "function"
#     },
#     {
#       "id": "setup-logging",
#       "name": "setup_logging",
#       "anchor": "function-setup-logging",
#       "kind": "function"
#     },
#     {
#       "id": "create-stub-module",
#       "name": "_create_stub_module",
#       "anchor": "function-create-stub-module",
#       "kind": "function"
#     },
#     {
#       "id": "create-stub-bnode",
#       "name": "_create_stub_bnode",
#       "anchor": "function-create-stub-bnode",
#       "kind": "function"
#     },
#     {
#       "id": "create-stub-literal",
#       "name": "_create_stub_literal",
#       "anchor": "function-create-stub-literal",
#       "kind": "function"
#     },
#     {
#       "id": "create-stub-uri",
#       "name": "_create_stub_uri",
#       "anchor": "function-create-stub-uri",
#       "kind": "function"
#     },
#     {
#       "id": "stubnamespace",
#       "name": "_StubNamespace",
#       "anchor": "class-stubnamespace",
#       "kind": "class"
#     },
#     {
#       "id": "stubnamespacemanager",
#       "name": "_StubNamespaceManager",
#       "anchor": "class-stubnamespacemanager",
#       "kind": "class"
#     },
#     {
#       "id": "import-module",
#       "name": "_import_module",
#       "anchor": "function-import-module",
#       "kind": "function"
#     },
#     {
#       "id": "create-pystow-stub",
#       "name": "_create_pystow_stub",
#       "anchor": "function-create-pystow-stub",
#       "kind": "function"
#     },
#     {
#       "id": "stubgraph",
#       "name": "_StubGraph",
#       "anchor": "class-stubgraph",
#       "kind": "class"
#     },
#     {
#       "id": "create-rdflib-stub",
#       "name": "_create_rdflib_stub",
#       "anchor": "function-create-rdflib-stub",
#       "kind": "function"
#     },
#     {
#       "id": "create-pronto-stub",
#       "name": "_create_pronto_stub",
#       "anchor": "function-create-pronto-stub",
#       "kind": "function"
#     },
#     {
#       "id": "create-owlready-stub",
#       "name": "_create_owlready_stub",
#       "anchor": "function-create-owlready-stub",
#       "kind": "function"
#     },
#     {
#       "id": "get-pystow",
#       "name": "get_pystow",
#       "anchor": "function-get-pystow",
#       "kind": "function"
#     },
#     {
#       "id": "get-rdflib",
#       "name": "get_rdflib",
#       "anchor": "function-get-rdflib",
#       "kind": "function"
#     },
#     {
#       "id": "get-pronto",
#       "name": "get_pronto",
#       "anchor": "function-get-pronto",
#       "kind": "function"
#     },
#     {
#       "id": "get-owlready2",
#       "name": "get_owlready2",
#       "anchor": "function-get-owlready2",
#       "kind": "function"
#     },
#     {
#       "id": "storagebackend",
#       "name": "StorageBackend",
#       "anchor": "class-storagebackend",
#       "kind": "class"
#     },
#     {
#       "id": "safe-identifiers",
#       "name": "_safe_identifiers",
#       "anchor": "function-safe-identifiers",
#       "kind": "function"
#     },
#     {
#       "id": "directory-size",
#       "name": "_directory_size",
#       "anchor": "function-directory-size",
#       "kind": "function"
#     },
#     {
#       "id": "localstoragebackend",
#       "name": "LocalStorageBackend",
#       "anchor": "class-localstoragebackend",
#       "kind": "class"
#     },
#     {
#       "id": "fsspecstoragebackend",
#       "name": "FsspecStorageBackend",
#       "anchor": "class-fsspecstoragebackend",
#       "kind": "class"
#     },
#     {
#       "id": "get-storage-backend",
#       "name": "get_storage_backend",
#       "anchor": "function-get-storage-backend",
#       "kind": "function"
#     },
#     {
#       "id": "log-download-memory",
#       "name": "_log_download_memory",
#       "anchor": "function-log-download-memory",
#       "kind": "function"
#     },
#     {
#       "id": "downloadresult",
#       "name": "DownloadResult",
#       "anchor": "class-downloadresult",
#       "kind": "class"
#     },
#     {
#       "id": "downloadfailure",
#       "name": "DownloadFailure",
#       "anchor": "class-downloadfailure",
#       "kind": "class"
#     },
#     {
#       "id": "tokenbucket",
#       "name": "TokenBucket",
#       "anchor": "class-tokenbucket",
#       "kind": "class"
#     },
#     {
#       "id": "is-retryable-status",
#       "name": "_is_retryable_status",
#       "anchor": "function-is-retryable-status",
#       "kind": "function"
#     },
#     {
#       "id": "enforce-idn-safety",
#       "name": "_enforce_idn_safety",
#       "anchor": "function-enforce-idn-safety",
#       "kind": "function"
#     },
#     {
#       "id": "rebuild-netloc",
#       "name": "_rebuild_netloc",
#       "anchor": "function-rebuild-netloc",
#       "kind": "function"
#     },
#     {
#       "id": "validate-url-security",
#       "name": "validate_url_security",
#       "anchor": "function-validate-url-security",
#       "kind": "function"
#     },
#     {
#       "id": "sha256-file",
#       "name": "sha256_file",
#       "anchor": "function-sha256-file",
#       "kind": "function"
#     },
#     {
#       "id": "validate-member-path",
#       "name": "_validate_member_path",
#       "anchor": "function-validate-member-path",
#       "kind": "function"
#     },
#     {
#       "id": "check-compression-ratio",
#       "name": "_check_compression_ratio",
#       "anchor": "function-check-compression-ratio",
#       "kind": "function"
#     },
#     {
#       "id": "extract-zip-safe",
#       "name": "extract_zip_safe",
#       "anchor": "function-extract-zip-safe",
#       "kind": "function"
#     },
#     {
#       "id": "extract-tar-safe",
#       "name": "extract_tar_safe",
#       "anchor": "function-extract-tar-safe",
#       "kind": "function"
#     },
#     {
#       "id": "extract-archive-safe",
#       "name": "extract_archive_safe",
#       "anchor": "function-extract-archive-safe",
#       "kind": "function"
#     },
#     {
#       "id": "streamingdownloader",
#       "name": "StreamingDownloader",
#       "anchor": "class-streamingdownloader",
#       "kind": "class"
#     },
#     {
#       "id": "get-bucket",
#       "name": "_get_bucket",
#       "anchor": "function-get-bucket",
#       "kind": "function"
#     },
#     {
#       "id": "download-stream",
#       "name": "download_stream",
#       "anchor": "function-download-stream",
#       "kind": "function"
#     },
#     {
#       "id": "validationrequest",
#       "name": "ValidationRequest",
#       "anchor": "class-validationrequest",
#       "kind": "class"
#     },
#     {
#       "id": "validationresult",
#       "name": "ValidationResult",
#       "anchor": "class-validationresult",
#       "kind": "class"
#     },
#     {
#       "id": "validationtimeout",
#       "name": "ValidationTimeout",
#       "anchor": "class-validationtimeout",
#       "kind": "class"
#     },
#     {
#       "id": "log-validation-memory",
#       "name": "_log_validation_memory",
#       "anchor": "function-log-validation-memory",
#       "kind": "function"
#     },
#     {
#       "id": "write-validation-json",
#       "name": "_write_validation_json",
#       "anchor": "function-write-validation-json",
#       "kind": "function"
#     },
#     {
#       "id": "python-merge-sort",
#       "name": "_python_merge_sort",
#       "anchor": "function-python-merge-sort",
#       "kind": "function"
#     },
#     {
#       "id": "term-to-string",
#       "name": "_term_to_string",
#       "anchor": "function-term-to-string",
#       "kind": "function"
#     },
#     {
#       "id": "canonicalize-turtle",
#       "name": "_canonicalize_turtle",
#       "anchor": "function-canonicalize-turtle",
#       "kind": "function"
#     },
#     {
#       "id": "canonicalize-blank-nodes-line",
#       "name": "_canonicalize_blank_nodes_line",
#       "anchor": "function-canonicalize-blank-nodes-line",
#       "kind": "function"
#     },
#     {
#       "id": "sort-triple-file",
#       "name": "_sort_triple_file",
#       "anchor": "function-sort-triple-file",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-streaming",
#       "name": "normalize_streaming",
#       "anchor": "function-normalize-streaming",
#       "kind": "function"
#     },
#     {
#       "id": "validatorsubprocesserror",
#       "name": "ValidatorSubprocessError",
#       "anchor": "class-validatorsubprocesserror",
#       "kind": "class"
#     },
#     {
#       "id": "worker-pronto",
#       "name": "_worker_pronto",
#       "anchor": "function-worker-pronto",
#       "kind": "function"
#     },
#     {
#       "id": "worker-owlready2",
#       "name": "_worker_owlready2",
#       "anchor": "function-worker-owlready2",
#       "kind": "function"
#     },
#     {
#       "id": "run-validator-subprocess",
#       "name": "_run_validator_subprocess",
#       "anchor": "function-run-validator-subprocess",
#       "kind": "function"
#     },
#     {
#       "id": "run-with-timeout",
#       "name": "_run_with_timeout",
#       "anchor": "function-run-with-timeout",
#       "kind": "function"
#     },
#     {
#       "id": "prepare-xbrl-package",
#       "name": "_prepare_xbrl_package",
#       "anchor": "function-prepare-xbrl-package",
#       "kind": "function"
#     },
#     {
#       "id": "validate-rdflib",
#       "name": "validate_rdflib",
#       "anchor": "function-validate-rdflib",
#       "kind": "function"
#     },
#     {
#       "id": "validate-pronto",
#       "name": "validate_pronto",
#       "anchor": "function-validate-pronto",
#       "kind": "function"
#     },
#     {
#       "id": "validate-owlready2",
#       "name": "validate_owlready2",
#       "anchor": "function-validate-owlready2",
#       "kind": "function"
#     },
#     {
#       "id": "validate-robot",
#       "name": "validate_robot",
#       "anchor": "function-validate-robot",
#       "kind": "function"
#     },
#     {
#       "id": "validate-arelle",
#       "name": "validate_arelle",
#       "anchor": "function-validate-arelle",
#       "kind": "function"
#     },
#     {
#       "id": "load-validator-plugins",
#       "name": "_load_validator_plugins",
#       "anchor": "function-load-validator-plugins",
#       "kind": "function"
#     },
#     {
#       "id": "run-validator-task",
#       "name": "_run_validator_task",
#       "anchor": "function-run-validator-task",
#       "kind": "function"
#     },
#     {
#       "id": "run-validators",
#       "name": "run_validators",
#       "anchor": "function-run-validators",
#       "kind": "function"
#     },
#     {
#       "id": "run-worker-cli",
#       "name": "_run_worker_cli",
#       "anchor": "function-run-worker-cli",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     },
#     {
#       "id": "get-manifest-schema",
#       "name": "get_manifest_schema",
#       "anchor": "function-get-manifest-schema",
#       "kind": "function"
#     },
#     {
#       "id": "validate-manifest-dict",
#       "name": "validate_manifest_dict",
#       "anchor": "function-validate-manifest-dict",
#       "kind": "function"
#     },
#     {
#       "id": "ontologydownloaderror",
#       "name": "OntologyDownloadError",
#       "anchor": "class-ontologydownloaderror",
#       "kind": "class"
#     },
#     {
#       "id": "resolvererror",
#       "name": "ResolverError",
#       "anchor": "class-resolvererror",
#       "kind": "class"
#     },
#     {
#       "id": "validationerror",
#       "name": "ValidationError",
#       "anchor": "class-validationerror",
#       "kind": "class"
#     },
#     {
#       "id": "configurationerror",
#       "name": "ConfigurationError",
#       "anchor": "class-configurationerror",
#       "kind": "class"
#     },
#     {
#       "id": "fetchspec",
#       "name": "FetchSpec",
#       "anchor": "class-fetchspec",
#       "kind": "class"
#     },
#     {
#       "id": "fetchresult",
#       "name": "FetchResult",
#       "anchor": "class-fetchresult",
#       "kind": "class"
#     },
#     {
#       "id": "manifest",
#       "name": "Manifest",
#       "anchor": "class-manifest",
#       "kind": "class"
#     },
#     {
#       "id": "resolver",
#       "name": "Resolver",
#       "anchor": "class-resolver",
#       "kind": "class"
#     },
#     {
#       "id": "resolvercandidate",
#       "name": "ResolverCandidate",
#       "anchor": "class-resolvercandidate",
#       "kind": "class"
#     },
#     {
#       "id": "plannedfetch",
#       "name": "PlannedFetch",
#       "anchor": "class-plannedfetch",
#       "kind": "class"
#     },
#     {
#       "id": "parse-http-datetime",
#       "name": "parse_http_datetime",
#       "anchor": "function-parse-http-datetime",
#       "kind": "function"
#     },
#     {
#       "id": "parse-iso-datetime",
#       "name": "parse_iso_datetime",
#       "anchor": "function-parse-iso-datetime",
#       "kind": "function"
#     },
#     {
#       "id": "parse-version-timestamp",
#       "name": "parse_version_timestamp",
#       "anchor": "function-parse-version-timestamp",
#       "kind": "function"
#     },
#     {
#       "id": "infer-version-timestamp",
#       "name": "infer_version_timestamp",
#       "anchor": "function-infer-version-timestamp",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-datetime",
#       "name": "_coerce_datetime",
#       "anchor": "function-coerce-datetime",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-timestamp",
#       "name": "_normalize_timestamp",
#       "anchor": "function-normalize-timestamp",
#       "kind": "function"
#     },
#     {
#       "id": "populate-plan-metadata",
#       "name": "_populate_plan_metadata",
#       "anchor": "function-populate-plan-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "migrate-manifest-inplace",
#       "name": "_migrate_manifest_inplace",
#       "anchor": "function-migrate-manifest-inplace",
#       "kind": "function"
#     },
#     {
#       "id": "read-manifest",
#       "name": "_read_manifest",
#       "anchor": "function-read-manifest",
#       "kind": "function"
#     },
#     {
#       "id": "validate-manifest",
#       "name": "_validate_manifest",
#       "anchor": "function-validate-manifest",
#       "kind": "function"
#     },
#     {
#       "id": "parse-last-modified",
#       "name": "_parse_last_modified",
#       "anchor": "function-parse-last-modified",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-last-modified",
#       "name": "_fetch_last_modified",
#       "anchor": "function-fetch-last-modified",
#       "kind": "function"
#     },
#     {
#       "id": "write-manifest",
#       "name": "_write_manifest",
#       "anchor": "function-write-manifest",
#       "kind": "function"
#     },
#     {
#       "id": "build-destination",
#       "name": "_build_destination",
#       "anchor": "function-build-destination",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-license-allowed",
#       "name": "_ensure_license_allowed",
#       "anchor": "function-ensure-license-allowed",
#       "kind": "function"
#     },
#     {
#       "id": "resolver-candidates",
#       "name": "_resolver_candidates",
#       "anchor": "function-resolver-candidates",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-plan-with-fallback",
#       "name": "_resolve_plan_with_fallback",
#       "anchor": "function-resolve-plan-with-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-one",
#       "name": "fetch_one",
#       "anchor": "function-fetch-one",
#       "kind": "function"
#     },
#     {
#       "id": "plan-one",
#       "name": "plan_one",
#       "anchor": "function-plan-one",
#       "kind": "function"
#     },
#     {
#       "id": "plan-all",
#       "name": "plan_all",
#       "anchor": "function-plan-all",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-all",
#       "name": "fetch_all",
#       "anchor": "function-fetch-all",
#       "kind": "function"
#     },
#     {
#       "id": "safe-lock-component",
#       "name": "_safe_lock_component",
#       "anchor": "function-safe-lock-component",
#       "kind": "function"
#     },
#     {
#       "id": "version-lock",
#       "name": "_version_lock",
#       "anchor": "function-version-lock",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Unified ontology downloader orchestrating settings, retries, and logging."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import gzip
import heapq

# --- Foundation utilities ---
import importlib
import json
import logging
import os
import platform
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

try:  # pragma: no cover - platform specific availability
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - platform specific availability
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-windows
    msvcrt = None  # type: ignore[assignment]
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from importlib import metadata
from itertools import islice
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    retryable: Callable[[BaseException], bool],
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    jitter: float = 0.5,
    callback: Optional[Callable[[int, BaseException, float], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Execute ``func`` with exponential backoff until it succeeds.

    Args:
        func: Zero-argument callable to invoke.
        retryable: Predicate returning ``True`` when the raised exception should
            trigger another attempt.
        max_attempts: Maximum number of attempts including the initial call.
        backoff_base: Base delay in seconds used for the exponential schedule.
        jitter: Maximum random jitter (uniform) added to each delay.
        callback: Optional hook invoked before sleeping with
            ``(attempt_number, error, delay_seconds)``.
        sleep: Sleep function, overridable for deterministic tests.

    Returns:
        The result produced by ``func`` when it succeeds.

    Raises:
        ValueError: If ``max_attempts`` is less than one.
        BaseException: Re-raises the last exception from ``func`` when retries
            are exhausted or the predicate indicates it is not retryable.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except BaseException as exc:  # pragma: no cover - behaviour verified via callers
            if attempt >= max_attempts or not retryable(exc):
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            if jitter > 0:
                delay += random.uniform(0.0, jitter)
            if callback is not None:
                try:
                    callback(attempt, exc, delay)
                except Exception:  # pragma: no cover - defensive against callbacks
                    pass
            sleep(max(delay, 0.0))


from .config import (  # noqa: WPS347 - public re-export surface
    ConfigError,
    DefaultsConfig,
    DownloadConfiguration,
    LoggingConfiguration,
    ResolvedConfig,
    ValidationConfig,
    build_resolved_config,
    ensure_python_version,
    get_env_overrides,
    load_config,
    load_raw_yaml,
    parse_rate_limit_to_rps,
    validate_config,
)
from .pipeline import merge_defaults
from .io_safe import generate_correlation_id, mask_sensitive_data, sanitize_filename




class JSONFormatter(logging.Formatter):
    """Formatter emitting JSON structured logs.

    Attributes:
        default_msec_format: Fractional second format applied to timestamps.

    Examples:
        >>> formatter = JSONFormatter()
        >>> isinstance(formatter.format(logging.LogRecord(\"test\", 20, __file__, 1, \"msg\", (), None)), str)  # doctest: +SKIP
        True
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a logging record into a JSON line.

        Args:
            record: Logging record produced by the underlying logger.

        Returns:
            JSON string containing logging metadata and contextual extras.
        """

        now = datetime.now(timezone.utc)
        payload = {
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "ontology_id": getattr(record, "ontology_id", None),
            "stage": getattr(record, "stage", None),
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(mask_sensitive_data(payload))


def _compress_old_log(path: Path) -> None:
    """Compress a log file in-place using gzip to reclaim disk space.

    Args:
        path: Filesystem location of the log file to compress.
    """

    compressed_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(compressed_path, "wb") as target:
        target.write(source.read())
    path.unlink(missing_ok=True)


def _cleanup_logs(log_dir: Path, retention_days: int) -> None:
    """Apply rotation and retention policy to the log directory.

    Args:
        log_dir: Directory that stores structured log files.
        retention_days: Number of days to keep uncompressed log files.
    """

    now = datetime.now(timezone.utc)
    retention_delta = timedelta(days=retention_days)
    for file in log_dir.glob("*.jsonl"):
        mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        if now - mtime > retention_delta:
            _compress_old_log(file)
    for file in log_dir.glob("*.jsonl.gz"):
        mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        if now - mtime > retention_delta:
            file.unlink(missing_ok=True)


def setup_logging(
    *,
    level: str = "INFO",
    retention_days: int = 30,
    max_log_size_mb: int = 100,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Configure structured logging handlers for ontology downloads.

    Args:
        level: Logging level applied to the ontology downloader logger.
        retention_days: Number of days to retain uncompressed log files before archival.
        max_log_size_mb: Maximum size of each log file before rotation occurs.
        log_dir: Optional override for the log directory; falls back to defaults when omitted.

    Returns:
        Logger instance configured with console and rotating JSON handlers.
    """

    resolved_dir = log_dir or Path(os.environ.get("ONTOFETCH_LOG_DIR", ""))
    if not resolved_dir:
        resolved_dir = LOG_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_logs(resolved_dir, retention_days)

    logger = logging.getLogger("DocsToKG.OntologyDownload")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(logger.handlers):
        if getattr(handler, "_ontofetch_managed", False):
            logger.removeHandler(handler)
            handler.close()

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    stream_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    file_name = sanitize_filename(f"ontofetch-{today}.jsonl")
    file_handler = RotatingFileHandler(
        resolved_dir / file_name,
        maxBytes=int(max_log_size_mb * 1024 * 1024),
        backupCount=5,
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    logger.propagate = True

    return logger


DefaultsConfiguration = DefaultsConfig
LoggingConfig = LoggingConfiguration
ValidationConfiguration = ValidationConfig


# --- Storage infrastructure ---

try:  # pragma: no cover - optional dependency
    import fsspec  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - allow local-only mode
    fsspec = None  # type: ignore

_pystow: Optional[Any] = None
_rdflib: Optional[Any] = None
_pronto: Optional[Any] = None
_owlready2: Optional[Any] = None

_STUB_ATTR = "_ontofetch_stub"
_BNODE_COUNTER = 0


def _create_stub_module(name: str, attrs: Dict[str, Any]) -> ModuleType:
    """Create a stub module populated with the provided attributes.

    Args:
        name: Dotted module path that should appear in :data:`sys.modules`.
        attrs: Mapping of attribute names to objects exposed by the stub.

    Returns:
        Module instance that mimics the requested package for test isolation.
    """

    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    setattr(module, _STUB_ATTR, True)
    return module


def _create_stub_bnode(value: Optional[str] = None) -> str:
    """Create a deterministic blank node identifier for rdflib stubs.

    Args:
        value: Optional explicit identifier to reuse instead of auto-incrementing.

    Returns:
        RDF blank node identifier anchored by the ``_:`` prefix.
    """

    global _BNODE_COUNTER
    if value is not None:
        return value
    _BNODE_COUNTER += 1
    return f"_:b{_BNODE_COUNTER}"


def _create_stub_literal(value: Any = None) -> str:
    """Represent literals as simple string values for stub graphs.

    Args:
        value: Python value to coerce into an rdflib-style literal.

    Returns:
        String literal representation suitable for Turtle serialization.
    """

    if value is None:
        return '""'
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _create_stub_uri(value: Optional[str] = None) -> str:
    """Create a URI reference that matches rdflib serialization expectations.

    Args:
        value: URI string, optionally already wrapped in angle brackets.

    Returns:
        URI reference wrapped in angle brackets for Turtle compatibility.
    """

    if value is None:
        return "<>"
    if value.startswith("<") and value.endswith(">"):  # pragma: no cover - defensive
        return value
    return f"<{value}>"


class _StubNamespace:
    """Minimal replacement mimicking rdflib Namespace behaviour.

    Attributes:
        _base: Base IRI string used to expand namespace members.

    Examples:
        >>> ns = _StubNamespace("http://example.org/")
        >>> ns["Term"]
        'http://example.org/Term'
    """

    def __init__(self, base: str):
        """Initialise the namespace with a base IRI string.

        Args:
            base: Base IRI used to expand namespace members.

        Returns:
            None.
        """

        self._base = base

    def __getitem__(self, key: str) -> str:
        """Return the expanded URI for the provided key.

        Args:
            key: Local name appended to the base namespace.

        Returns:
            Fully qualified IRI for the requested key.
        """

        return f"{self._base}{key}"


class _StubNamespaceManager:
    """Provide a namespaces() method compatible with rdflib.

    Attributes:
        _bindings: Mapping of prefixes to namespace IRIs.

    Examples:
        >>> manager = _StubNamespaceManager()
        >>> manager.bind("ex", "http://example.org/")
        >>> list(manager.namespaces())
        [('ex', 'http://example.org/')]
    """

    def __init__(self) -> None:
        """Create the namespace manager with empty prefix bindings.

        Args:
            None.

        Returns:
            None.
        """

        self._bindings: Dict[str, str] = {}

    def bind(self, prefix: str, namespace: str) -> None:
        """Associate a namespace prefix with a full URI.

        Args:
            prefix: Namespace shorthand used in Turtle output.
            namespace: Fully qualified namespace IRI.

        Returns:
            None.
        """

        self._bindings[prefix] = namespace

    def namespaces(self) -> Iterable[Tuple[str, str]]:
        """Yield bound namespaces as ``(prefix, namespace)`` tuples.

        Args:
            None.

        Returns:
            Iterable of namespace bindings recorded by the manager.
        """

        return self._bindings.items()


def _import_module(name: str) -> Any:
    """Import a module by name using :mod:`importlib`.

    The indirection makes it trivial to monkeypatch the import logic in unit
    tests without modifying global interpreter state.

    Args:
        name: Fully qualified module name to load.

    Returns:
        Imported module, falling back to the real implementation when present.

    Raises:
        ModuleNotFoundError: If the module cannot be located.
    """

    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _STUB_ATTR, False):
        sys.modules.pop(name, None)
    return importlib.import_module(name)


def _create_pystow_stub(root: Path) -> ModuleType:
    """Return a stub module implementing ``join`` similar to pystow.

    Args:
        root: Filesystem directory that acts as the backing storage root.

    Returns:
        Module object exposing a ``join`` helper compatible with pystow usage.
    """

    base = root

    def join(*segments: str) -> Path:
        """Join path segments beneath the stub root directory.

        Args:
            *segments: Arbitrary path components appended to the root.

        Returns:
            Path anchored at ``root`` with the provided segments appended.
        """

        return base.joinpath(*segments)

    module = _create_stub_module(
        "pystow",
        {
            "join": join,
        },
    )
    module._root = base  # type: ignore[attr-defined]
    return module


class _StubGraph:
    """Lightweight graph implementation mirroring rdflib essentials.

    Attributes:
        _triples: In-memory list of Turtle triple strings.
        _last_text: Raw Turtle content captured during parsing.
        namespace_manager: Stub namespace manager for rdflib compatibility.

    Examples:
        >>> graph = _StubGraph()
        >>> graph.parse("tests/data/example.ttl")  # doctest: +SKIP
        >>> len(graph)
        0
    """

    _ontofetch_stub = True

    def __init__(self) -> None:
        """Initialise an empty in-memory graph representation.

        Args:
            None.

        Returns:
            None.
        """

        self._triples: List[Tuple[str, str, str]] = []
        self._last_text = "# Stub TTL output\n"
        self.namespace_manager = _StubNamespaceManager()

    def parse(self, source: str, format: Optional[str] = None, **_kwargs: object) -> "_StubGraph":
        """Parse a Turtle file into the stub graph.

        Args:
            source: Local filesystem path to a Turtle document.
            format: Optional rdflib format hint that is ignored by the stub.
            _kwargs: Additional keyword arguments unused by the stub.

        Returns:
            The current graph instance populated with parsed triple strings.

        Raises:
            FileNotFoundError: If the Turtle file cannot be read.
        """

        text = Path(source).read_text()
        self._last_text = text
        triples: List[Tuple[str, str, str]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@prefix"):
                try:
                    _, remainder = line.split(None, 1)
                except ValueError:
                    continue
                parts = [segment.strip() for segment in remainder.split(None, 2)]
                if len(parts) >= 2:
                    prefix = parts[0].rstrip(":")
                    namespace = parts[1].strip("<>")
                    namespace = namespace.rstrip(".")
                    self.namespace_manager.bind(prefix, namespace)
                remaining = parts[2] if len(parts) == 3 else ""
                line = remaining.strip()
                if not line:
                    continue
            if line.endswith("."):
                line = line[:-1].strip()
            if not line:
                continue
            pieces = line.split(None, 2)
            if len(pieces) < 3:
                continue
            subject, predicate, obj = pieces
            triples.append((subject, predicate, obj))
        self._triples = triples
        return self

    def serialize(
        self, destination: Optional[Any] = None, format: Optional[str] = None, **_kwargs: object
    ):
        """Serialize the stub graph back to Turtle text.

        Args:
            destination: Optional output target path or file-like object.
            format: Optional rdflib format that is ignored by the stub.
            _kwargs: Additional keyword arguments unused by the stub.

        Returns:
            Destination handle or Turtle text when serializing to a string.

        Raises:
            OSError: If writing to the provided destination fails.
        """

        if destination is None:
            return self._last_text
        if isinstance(destination, (str, Path)):
            Path(destination).write_text(self._last_text)
            return destination
        # File-like object (e.g., BytesIO)
        destination.write(b"# Stub TTL output\n")
        return destination

    def bind(self, prefix: str, namespace: str) -> None:
        """Attach a namespace binding to the internal namespace manager.

        Args:
            prefix: Namespace shorthand used in serialization.
            namespace: Fully qualified namespace IRI.

        Returns:
            None.
        """

        self.namespace_manager.bind(prefix, namespace)

    def namespaces(self) -> Iterable[Tuple[str, str]]:
        """Return the stored namespace bindings for serialization.

        Args:
            None.

        Returns:
            Iterable containing prefix-to-namespace mappings.
        """

        return self.namespace_manager.namespaces()

    def __len__(self) -> int:
        """Return the number of parsed triples retained by the stub.

        Args:
            None.

        Returns:
            Integer count of Turtle triples captured during parsing.
        """

        return len(self._triples)

    def __iter__(self):
        """Iterate over the stored Turtle triple representations.

        Args:
            None.

        Returns:
            Iterator yielding Turtle triple strings from the stub graph.
        """

        return iter(self._triples)


def _create_rdflib_stub() -> ModuleType:
    """Create a stub implementation compatible with rdflib usage in tests.

    Returns:
        Module object that mirrors the small subset of rdflib used by validators.
    """

    namespace_module = _create_stub_module(
        "namespace",
        {
            "NamespaceManager": _StubNamespaceManager,
        },
    )
    return _create_stub_module(
        "rdflib",
        {
            "Graph": _StubGraph,
            "Namespace": _StubNamespace,
            "BNode": lambda value=None: _create_stub_bnode(value),
            "Literal": lambda value=None: _create_stub_literal(value),
            "URIRef": lambda value=None: _create_stub_uri(value),
            "namespace": namespace_module,
        },
    )


def _create_pronto_stub() -> ModuleType:
    """Create a stub module mimicking pronto interfaces.

    Returns:
        Module object exposing a lightweight :class:`Ontology` implementation.
    """

    class _StubOntology:
        _ontofetch_stub = True

        def __init__(self, _path: Optional[str] = None) -> None:
            """Initialise the stub ontology with an optional source path.

            Args:
                _path: Optional location of the source ontology file.

            Returns:
                None.
            """

            self.path = _path

        def terms(self) -> Iterable[str]:
            """Return a static list of ontology term identifiers.

            Args:
                None.

            Returns:
                Iterable containing deterministic ontology term IDs.
            """

            return ["TERM:0000001", "TERM:0000002"]

        def dump(self, destination: str, format: str = "obojson") -> None:
            """Write an empty ontology JSON payload to ``destination``.

            Args:
                destination: Filesystem path where the JSON stub is stored.
                format: Serialization format, kept for API equivalence.

            Returns:
                None.
            """

            Path(destination).write_text('{"graphs": []}')

    return _create_stub_module("pronto", {"Ontology": _StubOntology})


def _create_owlready_stub() -> ModuleType:
    """Create a stub module mimicking owlready2 key behaviour.

    Returns:
        Module object providing ``get_ontology`` compatible with owlready2.
    """

    class _StubOntology:
        _ontofetch_stub = True

        def __init__(self, iri: str) -> None:
            """Store the ontology IRI for later inspection.

            Args:
                iri: Ontology IRI compatible with owlready2 usage.

            Returns:
                None.
            """

            self.iri = iri

        def load(self) -> "_StubOntology":
            """Provide a fluent API returning the ontology itself.

            Args:
                None.

            Returns:
                The same stub ontology instance, mimicking owlready2.

            Raises:
                None.
            """

            return self

        def classes(self) -> List[str]:
            """Return a deterministic set of class identifiers.

            Args:
                None.

            Returns:
                List of representative ontology class names.
            """

            return ["Class1", "Class2", "Class3"]

    def get_ontology(iri: str) -> _StubOntology:
        """Return a stub ontology bound to the provided IRI.

        Args:
            iri: Ontology IRI used to instantiate the stub.

        Returns:
            `_StubOntology` instance wrapping the provided IRI.
        """

        return _StubOntology(iri)

    return _create_stub_module("owlready2", {"get_ontology": get_ontology})


def get_pystow() -> Any:
    """Return the real :mod:`pystow` module or a fallback stub.

    Args:
        None.

    Returns:
        Real pystow module when installed, otherwise a deterministic stub.
    """

    global _pystow
    if _pystow is not None:
        return _pystow
    try:
        _pystow = _import_module("pystow")
    except ModuleNotFoundError:  # pragma: no cover - allow minimal envs
        root = Path(os.environ.get("PYSTOW_HOME", "") or (Path.home() / ".data"))
        _pystow = _create_pystow_stub(root)
        sys.modules.setdefault("pystow", _pystow)
    return _pystow


def get_rdflib() -> Any:
    """Return :mod:`rdflib` or a stub supporting limited graph operations.

    Args:
        None.

    Returns:
        Real rdflib module when available, else a stub graph implementation.
    """

    global _rdflib
    if _rdflib is not None:
        return _rdflib
    try:
        _rdflib = _import_module("rdflib")
    except ModuleNotFoundError:  # pragma: no cover - exercised in tests
        _rdflib = _create_rdflib_stub()
        sys.modules.setdefault("rdflib", _rdflib)
    return _rdflib


def get_pronto() -> Any:
    """Return :mod:`pronto` or a stub with minimal ontology behaviour.

    Args:
        None.

    Returns:
        Real pronto module when installed, else a stub ontology wrapper.
    """

    global _pronto
    if _pronto is not None:
        return _pronto
    try:
        _pronto = _import_module("pronto")
    except ModuleNotFoundError:  # pragma: no cover - test fallback
        _pronto = _create_pronto_stub()
        sys.modules.setdefault("pronto", _pronto)
    return _pronto


def get_owlready2() -> Any:
    """Return :mod:`owlready2` or a stub matching the API used in validators.

    Args:
        None.

    Returns:
        Real owlready2 module when available, else a limited stub replacement.
    """

    global _owlready2
    if _owlready2 is not None:
        return _owlready2
    try:
        _owlready2 = _import_module("owlready2")
    except Exception:  # pragma: no cover - fallback when import fails
        _owlready2 = _create_owlready_stub()
        sys.modules.setdefault("owlready2", _owlready2)
    return _owlready2


pystow = get_pystow()

# Root directory where ontology fetch artefacts are persisted between runs.
DATA_ROOT = pystow.join("ontology-fetcher")
# Directory storing configuration manifests used by the downloader.
CONFIG_DIR = DATA_ROOT / "configs"
# Cache directory holding transient download data (e.g., tarballs, manifests).
CACHE_DIR = DATA_ROOT / "cache"
# Directory capturing structured logs emitted during ontology operations.
LOG_DIR = DATA_ROOT / "logs"
# Definitive storage area where processed ontology versions are kept.
LOCAL_ONTOLOGY_DIR = DATA_ROOT / "ontologies"

for directory in (CONFIG_DIR, CACHE_DIR, LOG_DIR, LOCAL_ONTOLOGY_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class StorageBackend(Protocol):
    """Protocol describing the operations required by the downloader pipeline.

    Attributes:
        root_path: Canonical base path that implementations expose for disk
            storage.  Remote-only backends can synthesize this attribute for
            instrumentation purposes.

    Examples:
        >>> class MemoryBackend(StorageBackend):
        ...     root_path = Path(\"/tmp\")  # pragma: no cover - illustrative stub
        ...     def prepare_version(self, ontology_id: str, version: str) -> Path:
        ...         ...
    """

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Return a working directory prepared for the given ontology version.

        Args:
            ontology_id: Identifier of the ontology being downloaded.
            version: Version string representing the in-flight download.

        Returns:
            Path to a freshly prepared directory tree ready for population.
        """

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure the requested version is present locally and return its path.

        Args:
            ontology_id: Identifier whose version must be present.
            version: Version string that should exist on local storage.

        Returns:
            Path to the local directory containing the requested version.
        """

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return sorted version identifiers currently stored for an ontology.

        Args:
            ontology_id: Identifier whose known versions are requested.

        Returns:
            Sorted list of version strings recognised by the backend.
        """

    def available_ontologies(self) -> List[str]:
        """Return sorted ontology identifiers known to the backend.

        Args:
            None.

        Returns:
            Alphabetically sorted list of ontology identifiers the backend can
            service.
        """

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Persist the working directory after validation succeeds.

        Args:
            ontology_id: Identifier of the ontology that completed processing.
            version: Version string associated with the finalized artifacts.
            local_dir: Directory containing the validated ontology payload.

        Returns:
            None.
        """

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the canonical storage path for ``ontology_id``/``version``.

        Args:
            ontology_id: Identifier of the ontology being queried.
            version: Version string for which a canonical path is needed.

        Returns:
            Path pointing to the storage location for the requested version.
        """

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete a stored version returning the number of bytes reclaimed.

        Args:
            ontology_id: Identifier whose version should be removed.
            version: Version string targeted for deletion.

        Returns:
            Number of bytes reclaimed by removing the stored version.

        Raises:
            OSError: If the underlying storage provider fails to delete data.
        """

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Update the latest version marker for operators and CLI tooling.

        Args:
            ontology_id: Identifier whose latest marker requires updating.
            version: Version string to record as the active release.

        Returns:
            None.
        """


def _safe_identifiers(ontology_id: str, version: str) -> Tuple[str, str]:
    """Return identifiers sanitized for filesystem usage.

    Args:
        ontology_id: Raw ontology identifier that may contain unsafe characters.
        version: Version label that should be filesystem-friendly.

    Returns:
        Tuple ``(safe_id, safe_version)`` containing sanitised values.
    """

    safe_id = sanitize_filename(ontology_id)
    safe_version = sanitize_filename(version)
    return safe_id, safe_version


def _directory_size(path: Path) -> int:
    """Return the total size in bytes for all regular files under ``path``.

    Args:
        path: Root directory whose files should be measured.

    Returns:
        Cumulative size in bytes of every regular file within ``path``.
    """

    total = 0
    for entry in path.rglob("*"):
        try:
            info = entry.stat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


class LocalStorageBackend:
    """Storage backend that keeps ontology artifacts on the local filesystem.

    Attributes:
        root: Base directory that stores ontology versions grouped by identifier.

    Examples:
        >>> backend = LocalStorageBackend(LOCAL_ONTOLOGY_DIR)
        >>> backend.available_ontologies()
        []
    """

    def __init__(self, root: Path) -> None:
        """Initialise the backend with a given storage root.

        Args:
            root: Directory used to persist ontology artifacts.

        Returns:
            None
        """

        self.root = root

    def _version_dir(self, ontology_id: str, version: str) -> Path:
        """Return the directory where a given ontology version is stored.

        Args:
            ontology_id: Identifier whose storage directory is required.
            version: Version string combined with the identifier.

        Returns:
            Path pointing to ``root/<ontology_id>/<version>``.
        """

        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return self.root / safe_id / safe_version

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Create the working directory structure for a download run.

        Args:
            ontology_id: Identifier of the ontology being processed.
            version: Canonical version string for the ontology.

        Returns:
            Path to the prepared base directory containing ``original``,
            ``normalized``, and ``validation`` subdirectories.
        """

        base = self.ensure_local_version(ontology_id, version)
        for subdir in ("original", "normalized", "validation"):
            (base / subdir).mkdir(parents=True, exist_ok=True)
        return base

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure a local workspace exists for ``ontology_id``/``version``.

        Args:
            ontology_id: Identifier whose workspace must exist.
            version: Version string that should map to a directory.

        Returns:
            Path to the local directory for the ontology version.
        """

        base = self._version_dir(ontology_id, version)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return sorted versions already present for an ontology.

        Args:
            ontology_id: Identifier whose stored versions should be listed.

        Returns:
            Sorted list of version strings found under the storage root.
        """

        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        base = self.root / safe_id
        if not base.exists():
            return []
        versions = [entry.name for entry in base.iterdir() if entry.is_dir()]
        return sorted(versions)

    def available_ontologies(self) -> List[str]:
        """Return ontology identifiers discovered under ``root``.

        Args:
            None.

        Returns:
            Sorted list of ontology identifiers available locally.
        """

        if not self.root.exists():
            return []
        return sorted([entry.name for entry in self.root.iterdir() if entry.is_dir()])

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Finalize a local version directory (no-op for purely local storage).

        Args:
            ontology_id: Identifier that finished processing.
            version: Version string associated with the processed ontology.
            local_dir: Directory containing the ready-to-serve ontology.

        Returns:
            None.
        """

        _ = (ontology_id, version, local_dir)  # pragma: no cover - intentional no-op

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the local storage directory for the requested version.

        Args:
            ontology_id: Identifier being queried.
            version: Version string whose storage path is needed.

        Returns:
            Path pointing to the stored ontology version.
        """

        return self._version_dir(ontology_id, version)

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete a stored ontology version returning reclaimed bytes.

        Args:
            ontology_id: Identifier whose stored version should be removed.
            version: Version string targeted for deletion.

        Returns:
            Number of bytes reclaimed by removing the version directory.

        Raises:
            OSError: Propagated if filesystem deletion fails.
        """

        path = self._version_dir(ontology_id, version)
        if not path.exists():
            return 0
        reclaimed = _directory_size(path)
        shutil.rmtree(path)
        return reclaimed

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Update symlink and marker file indicating the latest version.

        Args:
            ontology_id: Identifier whose latest marker should be updated.
            version: Version string to record as the latest processed build.

        Returns:
            None.
        """

        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        base = self.root / safe_id
        base.mkdir(parents=True, exist_ok=True)
        link = base / "latest"
        marker = base / "latest.txt"
        target = Path(version)

        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if marker.exists():
                marker.unlink()
            marker.write_text(version)
        else:
            if marker.exists():
                marker.unlink()


class FsspecStorageBackend(LocalStorageBackend):
    """Hybrid storage backend that mirrors artifacts to an fsspec location.

    Attributes:
        fs: ``fsspec`` filesystem instance used for remote operations.
        base_path: Root path within the remote filesystem where artefacts live.

    Examples:
        >>> backend = FsspecStorageBackend("memory://ontologies")  # doctest: +SKIP
        >>> backend.available_ontologies()  # doctest: +SKIP
        []
    """

    def __init__(self, url: str) -> None:
        """Create a hybrid storage backend backed by an fsspec URL.

        Args:
            url: Remote ``fsspec`` URL (for example ``s3://bucket/prefix``).

        Raises:
            ConfigError: If :mod:`fsspec` is not installed or the URL is invalid.

        Returns:
            None
        """

        if fsspec is None:  # pragma: no cover - exercised when dependency missing
            raise ConfigError(
                "fsspec required for remote storage. Install it via 'pip install fsspec'."
            )
        fs, path = fsspec.core.url_to_fs(url)  # type: ignore[attr-defined]
        self.fs = fs
        self.base_path = PurePosixPath(path)
        super().__init__(LOCAL_ONTOLOGY_DIR)

    def _remote_version_path(self, ontology_id: str, version: str) -> PurePosixPath:
        """Return the remote filesystem path for the specified ontology version.

        Args:
            ontology_id: Identifier whose remote storage path is required.
            version: Version string associated with the ontology release.

        Returns:
            Posix-style path referencing the remote storage location.
        """

        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return (self.base_path / safe_id / safe_version).with_suffix("")

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return versions aggregated from local cache and remote storage.

        Args:
            ontology_id: Identifier whose version catalogue is required.

        Returns:
            Sorted list combining local and remote version identifiers.
        """

        local_versions = super().available_versions(ontology_id)
        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        remote_dir = self.base_path / safe_id
        try:
            entries = self.fs.ls(str(remote_dir), detail=False)
        except FileNotFoundError:
            entries = []
        remote_versions = [
            PurePosixPath(entry).name for entry in entries if entry and not entry.endswith(".tmp")
        ]
        return sorted({*local_versions, *remote_versions})

    def available_ontologies(self) -> List[str]:
        """Return ontology identifiers available locally or remotely.

        Args:
            None.

        Returns:
            Sorted set union of local and remote ontology identifiers.
        """

        local_ids = super().available_ontologies()
        try:
            entries = self.fs.ls(str(self.base_path), detail=False)
        except FileNotFoundError:
            entries = []
        remote_ids = [
            PurePosixPath(entry).name for entry in entries if entry and not entry.endswith(".tmp")
        ]
        return sorted({*local_ids, *remote_ids})

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Mirror a remote ontology version into the local cache when absent.

        Args:
            ontology_id: Identifier whose version should exist locally.
            version: Version string to ensure within the local cache.

        Returns:
            Path to the local directory containing the requested version.
        """

        base = super().ensure_local_version(ontology_id, version)
        manifest_path = base / "manifest.json"
        if manifest_path.exists():
            return base

        remote_dir = self._remote_version_path(ontology_id, version)
        if not self.fs.exists(str(remote_dir)):
            return base

        try:
            remote_files = self.fs.find(str(remote_dir))
        except FileNotFoundError:
            remote_files = []
        for remote_file in remote_files:
            remote_path = PurePosixPath(remote_file)
            relative = remote_path.relative_to(remote_dir)
            local_path = base / Path(str(relative))
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.fs.get_file(str(remote_path), str(local_path))
        return base

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Upload the finalized local directory to the remote store.

        Args:
            ontology_id: Identifier of the ontology that has completed processing.
            version: Version string associated with the finalised ontology.
            local_dir: Directory containing the validated ontology payload.

        Returns:
            None.
        """

        remote_dir = self._remote_version_path(ontology_id, version)
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(local_dir)
            remote_path = remote_dir / PurePosixPath(str(relative).replace("\\", "/"))
            self.fs.makedirs(str(remote_path.parent), exist_ok=True)
            self.fs.put_file(str(path), str(remote_path))

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete both local and remote copies of a stored version.

        Args:
            ontology_id: Identifier whose stored version should be deleted.
            version: Version string targeted for deletion.

        Returns:
            Total bytes reclaimed across local and remote storage.

        Raises:
            OSError: Propagated if remote deletion fails irrecoverably.
        """

        reclaimed = super().delete_version(ontology_id, version)
        remote_dir = self._remote_version_path(ontology_id, version)
        if not self.fs.exists(str(remote_dir)):
            return reclaimed

        try:
            remote_files = self.fs.find(str(remote_dir))
        except FileNotFoundError:
            remote_files = []
        for remote_file in remote_files:
            try:
                info = self.fs.info(remote_file)
            except FileNotFoundError:
                continue
            size = info.get("size") if isinstance(info, dict) else None
            if isinstance(size, (int, float)):
                reclaimed += int(size)
        self.fs.rm(str(remote_dir), recursive=True)
        return reclaimed


def get_storage_backend() -> StorageBackend:
    """Instantiate the storage backend based on environment configuration.

    Args:
        None.

    Returns:
        Storage backend instance selected according to ``ONTOFETCH_STORAGE_URL``.
    """

    storage_url = os.getenv("ONTOFETCH_STORAGE_URL")
    if storage_url:
        return FsspecStorageBackend(storage_url)
    return LocalStorageBackend(LOCAL_ONTOLOGY_DIR)


STORAGE: StorageBackend = get_storage_backend()


# --- Network utilities ---
import hashlib
import ipaddress
import logging
import socket
import tarfile
import threading
import time
import unicodedata
import zipfile
from urllib.parse import ParseResult, urlparse, urlunparse

import pooch
import psutil
import requests


def _log_download_memory(logger: logging.Logger, event: str) -> None:
    """Emit debug-level memory usage snapshots when enabled.

    Args:
        logger: Logger instance controlling verbosity for download telemetry.
        event: Short label describing the lifecycle point (e.g., ``before``).

    Returns:
        None
    """
    is_enabled = getattr(logger, "isEnabledFor", None)
    if callable(is_enabled):
        enabled = is_enabled(logging.DEBUG)
    else:  # pragma: no cover - fallback for stub loggers
        enabled = False
    if not enabled:
        return
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024**2)
    logger.debug(
        "memory usage",
        extra={"stage": "download", "event": event, "memory_mb": round(memory_mb, 2)},
    )


@dataclass(slots=True)
class DownloadResult:
    """Result metadata for a completed download operation.

    Attributes:
        path: Final file path where the ontology document was stored.
        status: Download status (`fresh`, `updated`, or `cached`).
        sha256: SHA-256 checksum of the downloaded artifact.
        etag: HTTP ETag returned by the upstream server, when available.
        last_modified: Upstream last-modified header value if provided.

    Examples:
        >>> result = DownloadResult(Path("ontology.owl"), "fresh", "deadbeef", None, None)
        >>> result.status
        'fresh'
    """

    path: Path
    status: str
    sha256: str
    etag: Optional[str]
    last_modified: Optional[str]


class DownloadFailure(ConfigError):
    """Raised when an HTTP download attempt fails.

    Attributes:
        status_code: Optional HTTP status code returned by the upstream service.
        retryable: Whether the failure is safe to retry with an alternate resolver.

    Examples:
        >>> raise DownloadFailure("Unavailable", status_code=503, retryable=True)
        Traceback (most recent call last):
        DownloadFailure: Unavailable
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class TokenBucket:
    """Token bucket used to enforce per-host and per-service rate limits.

    Each unique combination of host and logical service identifier receives
    its own bucket so resolvers can honour provider-specific throttling
    guidance without starving other endpoints.

    Attributes:
        rate: Token replenishment rate per second.
        capacity: Maximum number of tokens the bucket may hold.
        tokens: Current token balance available for consumption.
        timestamp: Monotonic timestamp of the last refill.
        lock: Threading lock protecting bucket state.

    Examples:
        >>> bucket = TokenBucket(rate_per_sec=2.0, capacity=4.0)
        >>> bucket.consume(1.0)  # consumes immediately
        >>> isinstance(bucket.tokens, float)
        True
    """

    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity or rate_per_sec
        self.tokens = self.capacity
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> None:
        """Consume tokens from the bucket, sleeping until capacity is available.

        Args:
            tokens: Number of tokens required for the current download request.

        Returns:
            None
        """
        while True:
            with self.lock:
                now = time.monotonic()
                delta = now - self.timestamp
                self.timestamp = now
                self.tokens = min(self.capacity, self.tokens + delta * self.rate)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                needed = tokens - self.tokens
            time.sleep(max(needed / self.rate, 0.0))


_TOKEN_BUCKETS: Dict[str, TokenBucket] = {}


_RETRYABLE_HTTP_STATUSES = {403, 408, 425, 429, 500, 502, 503, 504}


def _is_retryable_status(status_code: Optional[int]) -> bool:
    if status_code is None:
        return True
    if status_code >= 500:
        return True
    return status_code in _RETRYABLE_HTTP_STATUSES


_RDF_FORMAT_LABELS = {
    "application/rdf+xml": "RDF/XML",
    "text/turtle": "Turtle",
    "application/n-triples": "N-Triples",
    "application/trig": "TriG",
    "application/ld+json": "JSON-LD",
}
_RDF_ALIAS_GROUPS = {
    "application/rdf+xml": {"application/rdf+xml", "application/xml", "text/xml"},
    "text/turtle": {"text/turtle", "application/x-turtle"},
    "application/n-triples": {"application/n-triples", "text/plain"},
    "application/trig": {"application/trig"},
    "application/ld+json": {"application/ld+json"},
}
RDF_MIME_ALIASES: Set[str] = set()
RDF_MIME_FORMAT_LABELS: Dict[str, str] = {}
for canonical, aliases in _RDF_ALIAS_GROUPS.items():
    label = _RDF_FORMAT_LABELS[canonical]
    for alias in aliases:
        RDF_MIME_ALIASES.add(alias)
        RDF_MIME_FORMAT_LABELS[alias] = label


def _enforce_idn_safety(host: str) -> None:
    """Validate internationalized hostnames and reject suspicious patterns.

    Args:
        host: Hostname component extracted from the download URL.

    Returns:
        None

    Raises:
        ConfigError: If the hostname mixes multiple scripts or contains invisible characters.
    """

    if all(ord(char) < 128 for char in host):
        return

    scripts = set()
    for char in host:
        if ord(char) < 128:
            if char.isalpha():
                scripts.add("LATIN")
            continue

        category = unicodedata.category(char)
        if category in {"Mn", "Me", "Cf"}:
            raise ConfigError("Internationalized host contains invisible characters")

        try:
            name = unicodedata.name(char)
        except ValueError as exc:
            raise ConfigError("Internationalized host contains unknown characters") from exc

        for script in ("LATIN", "CYRILLIC", "GREEK"):
            if script in name:
                scripts.add(script)
                break

    if len(scripts) > 1:
        raise ConfigError("Internationalized host mixes multiple scripts")


def _rebuild_netloc(parsed: ParseResult, ascii_host: str) -> str:
    """Reconstruct URL netloc with a normalized hostname.

    Args:
        parsed: Parsed URL components produced by :func:`urllib.parse.urlparse`.
        ascii_host: ASCII-normalized hostname (potentially IPv6).

    Returns:
        String suitable for use as the netloc portion of a URL.
    """

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        auth = f"{auth}@"

    host_component = ascii_host
    if ":" in host_component and not host_component.startswith("["):
        host_component = f"[{host_component}]"

    port = f":{parsed.port}" if parsed.port else ""
    return f"{auth}{host_component}{port}"


def validate_url_security(url: str, http_config: Optional[DownloadConfiguration] = None) -> str:
    """Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists.

    Hostnames are converted to punycode before resolution, and both direct IP
    addresses and DNS results are rejected when they target private or loopback
    ranges to prevent server-side request forgery.

    Args:
        url: URL returned by a resolver for ontology download.
        http_config: Download configuration providing optional host allowlist.

    Returns:
        HTTPS URL safe for downstream download operations.

    Raises:
        ConfigError: If the URL violates security requirements or allowlists.
    """

    parsed = urlparse(url)
    logger = logging.getLogger("DocsToKG.OntologyDownload")
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ConfigError("Only HTTP(S) URLs are allowed for ontology downloads")

    host = parsed.hostname
    if not host:
        raise ConfigError("URL must include hostname")

    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False

    ascii_host = host.lower()
    if not is_ip:
        _enforce_idn_safety(host)
        try:
            ascii_host = host.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ConfigError(f"Invalid internationalized hostname: {host}") from exc

    parsed = parsed._replace(netloc=_rebuild_netloc(parsed, ascii_host))

    allowed = http_config.normalized_allowed_hosts() if http_config else None
    allow_private = False
    if allowed:
        exact, suffixes = allowed
        if ascii_host in exact or any(
            ascii_host == suffix or ascii_host.endswith(f".{suffix}") for suffix in suffixes
        ):
            allow_private = True
        else:
            raise ConfigError(f"Host {host} not in allowlist")

    if scheme == "http":
        if allow_private:
            logger.warning(
                "allowing http url for explicit allowlist host",
                extra={"stage": "download", "original_url": url},
            )
        else:
            logger.warning(
                "upgrading http url to https",
                extra={"stage": "download", "original_url": url},
            )
            parsed = parsed._replace(scheme="https")
            scheme = "https"

    if scheme != "https" and not allow_private:
        raise ConfigError("Only HTTPS URLs are allowed for ontology downloads")

    if is_ip:
        address = ipaddress.ip_address(ascii_host)
        if not allow_private and (
            address.is_private or address.is_loopback or address.is_reserved or address.is_multicast
        ):
            raise ConfigError(f"Refusing to download from private address {host}")
        return urlunparse(parsed)

    try:
        infos = socket.getaddrinfo(ascii_host, None)
    except socket.gaierror as exc:
        logger.warning(
            "dns resolution failed",
            extra={"stage": "download", "hostname": host, "error": str(exc)},
        )
        return urlunparse(parsed)

    for info in infos:
        candidate_ip = ipaddress.ip_address(info[4][0])
        if not allow_private and (
            candidate_ip.is_private
            or candidate_ip.is_loopback
            or candidate_ip.is_reserved
            or candidate_ip.is_multicast
        ):
            raise ConfigError(f"Refusing to download from private address resolved for {host}")

    return urlunparse(parsed)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for the provided file.

    Args:
        path: Path to the file whose digest should be calculated.

    Returns:
        Hexadecimal SHA-256 checksum string.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_MAX_COMPRESSION_RATIO = 10.0


def _validate_member_path(member_name: str) -> Path:
    """Validate archive member paths to prevent traversal attacks.

    Args:
        member_name: Path declared within the archive.

    Returns:
        Sanitised relative path safe for extraction on the local filesystem.

    Raises:
        ConfigError: If the member path is absolute or contains traversal segments.
    """

    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute():
        raise ConfigError(f"Unsafe absolute path detected in archive: {member_name}")
    if not relative.parts:
        raise ConfigError(f"Empty path detected in archive: {member_name}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ConfigError(f"Unsafe path detected in archive: {member_name}")
    return Path(*relative.parts)


def _check_compression_ratio(
    *,
    total_uncompressed: int,
    compressed_size: int,
    archive: Path,
    logger: Optional[logging.Logger],
    archive_type: str,
) -> None:
    """Ensure compressed archives do not expand beyond the permitted ratio.

    Args:
        total_uncompressed: Sum of file sizes within the archive.
        compressed_size: Archive file size on disk (or sum of compressed entries).
        archive: Path to the archive on disk.
        logger: Optional logger for emitting diagnostic messages.
        archive_type: Human readable label for error messages (ZIP/TAR).

    Raises:
        ConfigError: If the archive exceeds the allowed expansion ratio.
    """

    if compressed_size <= 0:
        return
    ratio = total_uncompressed / float(compressed_size)
    if ratio > _MAX_COMPRESSION_RATIO:
        if logger:
            logger.error(
                "archive compression ratio too high",
                extra={
                    "stage": "extract",
                    "archive": str(archive),
                    "ratio": round(ratio, 2),
                    "compressed_bytes": compressed_size,
                    "uncompressed_bytes": total_uncompressed,
                    "limit": _MAX_COMPRESSION_RATIO,
                },
            )
        raise ConfigError(
            f"{archive_type} archive {archive} expands to {total_uncompressed} bytes, "
            f"exceeding {_MAX_COMPRESSION_RATIO}:1 compression ratio"
        )


def extract_zip_safe(
    zip_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Extract a ZIP archive while preventing traversal and compression bombs.

    Args:
        zip_path: Path to the ZIP file to extract.
        destination: Directory where extracted files should be stored.
        logger: Optional logger for emitting extraction telemetry.

    Returns:
        List of extracted file paths.

    Raises:
        ConfigError: If the archive contains unsafe paths, compression bombs, or is missing.
    """

    if not zip_path.exists():
        raise ConfigError(f"ZIP archive not found: {zip_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        safe_members: List[tuple[zipfile.ZipInfo, Path]] = []
        total_uncompressed = 0
        for member in members:
            member_path = _validate_member_path(member.filename)
            if member.is_dir():
                safe_members.append((member, member_path))
                continue
            total_uncompressed += int(member.file_size)
            safe_members.append((member, member_path))
        compressed_size = max(
            zip_path.stat().st_size,
            sum(int(member.compress_size) for member in members) or 0,
        )
        _check_compression_ratio(
            total_uncompressed=total_uncompressed,
            compressed_size=compressed_size,
            archive=zip_path,
            logger=logger,
            archive_type="ZIP",
        )
        for member, member_path in safe_members:
            target_path = destination / member_path
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(target_path)
    if logger:
        logger.info(
            "extracted zip archive",
            extra={"stage": "extract", "archive": str(zip_path), "files": len(extracted)},
        )
    return extracted


def extract_tar_safe(
    tar_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks.

    Args:
        tar_path: Path to the tar archive (tar, tar.gz, tar.xz).
        destination: Directory where extracted files should be stored.
        logger: Optional logger for emitting extraction telemetry.

    Returns:
        List of extracted file paths.

    Raises:
        ConfigError: If the archive is missing, unsafe, or exceeds compression limits.
    """

    if not tar_path.exists():
        raise ConfigError(f"TAR archive not found: {tar_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    try:
        with tarfile.open(tar_path, mode="r:*") as archive:
            members = archive.getmembers()
            safe_members: List[tuple[tarfile.TarInfo, Path]] = []
            total_uncompressed = 0
            for member in members:
                member_path = _validate_member_path(member.name)
                if member.isdir():
                    safe_members.append((member, member_path))
                    continue
                if member.islnk() or member.issym():
                    raise ConfigError(f"Unsafe link detected in archive: {member.name}")
                if member.isdev():
                    raise ConfigError(
                        f"Unsupported special file detected in archive: {member.name}"
                    )
                if not member.isfile():
                    raise ConfigError(f"Unsupported tar member type encountered: {member.name}")
                total_uncompressed += int(member.size)
                safe_members.append((member, member_path))
            compressed_size = tar_path.stat().st_size
            _check_compression_ratio(
                total_uncompressed=total_uncompressed,
                compressed_size=compressed_size,
                archive=tar_path,
                logger=logger,
                archive_type="TAR",
            )
            for member, member_path in safe_members:
                if member.isdir():
                    (destination / member_path).mkdir(parents=True, exist_ok=True)
                    continue
                target_path = destination / member_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_file = archive.extractfile(member)
                if extracted_file is None:
                    raise ConfigError(f"Failed to extract member: {member.name}")
                with extracted_file as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted.append(target_path)
    except tarfile.TarError as exc:
        raise ConfigError(f"Failed to extract tar archive {tar_path}: {exc}") from exc
    if logger:
        logger.info(
            "extracted tar archive",
            extra={"stage": "extract", "archive": str(tar_path), "files": len(extracted)},
        )
    return extracted


_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2")


def extract_archive_safe(
    archive_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Extract archives by dispatching to the appropriate safe handler.

    Args:
        archive_path: Path to the archive on disk.
        destination: Directory where files should be extracted.
        logger: Optional logger receiving structured extraction telemetry.

    Returns:
        List of paths extracted from the archive in the order processed.

    Raises:
        ConfigError: If the archive format is unsupported or extraction fails.
    """

    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        return extract_zip_safe(archive_path, destination, logger=logger)
    if any(lower_name.endswith(suffix) for suffix in _TAR_SUFFIXES):
        return extract_tar_safe(archive_path, destination, logger=logger)
    raise ConfigError(f"Unsupported archive format: {archive_path}")


class StreamingDownloader(pooch.HTTPDownloader):
    """Custom downloader supporting HEAD validation, conditional requests, resume, and caching.

    The downloader shares a :mod:`requests` session so it can issue a HEAD probe
    prior to streaming content, verifies Content-Type and Content-Length against
    expectations, and persists ETag/Last-Modified headers for cache-friendly
    revalidation.

    Attributes:
        destination: Final location where the ontology will be stored.
        custom_headers: HTTP headers supplied by the resolver.
        http_config: Download configuration governing retries and limits.
        previous_manifest: Manifest from prior runs used for caching.
        logger: Logger used for structured telemetry.
        status: Final download status (`fresh`, `updated`, or `cached`).
        response_etag: ETag returned by the upstream server, if present.
        response_last_modified: Last-modified timestamp provided by the server.
        expected_media_type: MIME type provided by the resolver for validation.

    Examples:
        >>> from pathlib import Path
        >>> from DocsToKG.OntologyDownload import DownloadConfiguration
        >>> downloader = StreamingDownloader(
        ...     destination=Path("/tmp/ontology.owl"),
        ...     headers={},
        ...     http_config=DownloadConfiguration(),
        ...     previous_manifest={},
        ...     logger=logging.getLogger("test"),
        ... )
        >>> downloader.status
        'fresh'
    """

    def __init__(
        self,
        *,
        destination: Path,
        headers: Dict[str, str],
        http_config: DownloadConfiguration,
        previous_manifest: Optional[Dict[str, object]],
        logger: logging.Logger,
        expected_media_type: Optional[str] = None,
    ) -> None:
        super().__init__(headers={}, progressbar=False, timeout=http_config.timeout_sec)
        self.destination = destination
        self.custom_headers = headers
        self.http_config = http_config
        self.previous_manifest = previous_manifest or {}
        self.logger = logger
        self.status = "fresh"
        self.response_etag: Optional[str] = None
        self.response_last_modified: Optional[str] = None
        self.expected_media_type = expected_media_type

    def _preliminary_head_check(
        self, url: str, session: requests.Session
    ) -> tuple[Optional[str], Optional[int]]:
        """Probe the origin with HEAD to audit media type and size before downloading.

        The HEAD probe allows the pipeline to abort before streaming large
        payloads that exceed configured limits and to log early warnings for
        mismatched Content-Type headers reported by the origin.

        Args:
            url: Fully qualified download URL resolved by the planner.
            session: Prepared requests session used for outbound calls.

        Returns:
            Tuple ``(content_type, content_length)`` extracted from response
            headers. Each element is ``None`` when the origin omits it.

        Raises:
            ConfigError: If the origin reports a payload larger than the
                configured ``max_download_size_gb`` limit.
        """

        try:
            response = session.head(
                url,
                headers=self.custom_headers,
                timeout=self.http_config.timeout_sec,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            self.logger.debug(
                "HEAD request exception, proceeding with GET",
                extra={"stage": "download", "error": str(exc), "url": url},
            )
            return None, None

        if response.status_code >= 400:
            self.logger.debug(
                "HEAD request failed, proceeding with GET",
                extra={
                    "stage": "download",
                    "method": "HEAD",
                    "status_code": response.status_code,
                    "url": url,
                },
            )
            return None, None

        content_type = response.headers.get("Content-Type")
        content_length_header = response.headers.get("Content-Length")
        content_length = int(content_length_header) if content_length_header else None

        if content_length:
            max_bytes = self.http_config.max_download_size_gb * (1024**3)
            if content_length > max_bytes:
                self.logger.error(
                    "file exceeds size limit (HEAD check)",
                    extra={
                        "stage": "download",
                        "content_length": content_length,
                        "limit_bytes": max_bytes,
                        "url": url,
                    },
                )
                raise ConfigError(
                    "File size {size} bytes exceeds limit of {limit} GB (detected via HEAD)".format(
                        size=content_length,
                        limit=self.http_config.max_download_size_gb,
                    )
                )

        return content_type, content_length

    def _validate_media_type(
        self,
        actual_content_type: Optional[str],
        expected_media_type: Optional[str],
        url: str,
    ) -> None:
        """Validate that the received ``Content-Type`` header is acceptable, tolerating aliases.

        RDF endpoints often return generic XML or Turtle aliases, so the
        validator accepts a small set of known MIME variants while still
        surfacing actionable warnings for unexpected types.

        Args:
            actual_content_type: Raw header value reported by the origin server.
            expected_media_type: MIME type declared by resolver metadata.
            url: Download URL logged when mismatches occur.

        Returns:
            None
        """

        if not self.http_config.validate_media_type:
            return
        if not expected_media_type:
            return
        if not actual_content_type:
            self.logger.warning(
                "server did not provide Content-Type header",
                extra={
                    "stage": "download",
                    "expected_media_type": expected_media_type,
                    "url": url,
                },
            )
            return

        actual_mime = actual_content_type.split(";")[0].strip().lower()
        expected_mime = expected_media_type.strip().lower()
        if actual_mime == expected_mime:
            return

        expected_label = RDF_MIME_FORMAT_LABELS.get(expected_mime)
        actual_label = RDF_MIME_FORMAT_LABELS.get(actual_mime)
        if expected_label and actual_label:
            if expected_label == actual_label:
                if actual_mime != expected_mime:
                    self.logger.info(
                        "acceptable media type variation",
                        extra={
                            "stage": "download",
                            "expected": expected_mime,
                            "actual": actual_mime,
                            "label": expected_label,
                            "url": url,
                        },
                    )
                return
            variation_hint = {
                "stage": "download",
                "expected_media_type": expected_mime,
                "expected_label": expected_label,
                "actual_media_type": actual_mime,
                "actual_label": actual_label,
                "url": url,
            }
            self.logger.warning(
                "media type mismatch detected",
                extra={
                    **variation_hint,
                    "action": "proceeding with download",
                    "override_hint": "Set defaults.http.validate_media_type: false to disable validation",
                },
            )
            return

        self.logger.warning(
            "media type mismatch detected",
            extra={
                "stage": "download",
                "expected_media_type": expected_mime,
                "actual_media_type": actual_mime,
                "url": url,
                "action": "proceeding with download",
                "override_hint": "Set defaults.http.validate_media_type: false to disable validation",
            },
        )

    def __call__(self, url: str, output_file: str, pooch_logger: logging.Logger) -> None:  # type: ignore[override]
        """Stream ontology content to disk while enforcing download policies.

        Args:
            url: Secure download URL resolved by the planner.
            output_file: Temporary filename managed by pooch during download.
            pooch_logger: Logger instance supplied by pooch (unused).

        Raises:
            ConfigError: If download limits are exceeded or filesystem errors occur.
            requests.HTTPError: Propagated when HTTP status codes indicate failure.

        Returns:
            None
        """
        manifest_headers: Dict[str, str] = {}
        if "etag" in self.previous_manifest:
            manifest_headers["If-None-Match"] = self.previous_manifest["etag"]
        if "last_modified" in self.previous_manifest:
            manifest_headers["If-Modified-Since"] = self.previous_manifest["last_modified"]
        request_headers = {**self.custom_headers, **manifest_headers}
        part_path = Path(output_file + ".part")
        destination_part_path = Path(str(self.destination) + ".part")
        if not part_path.exists() and destination_part_path.exists():
            part_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination_part_path, part_path)
        resume_position = part_path.stat().st_size if part_path.exists() else 0
        if resume_position:
            request_headers["Range"] = f"bytes={resume_position}-"
        attempt = 0
        session = requests.Session()
        head_content_type, _ = self._preliminary_head_check(url, session)
        if head_content_type:
            self._validate_media_type(head_content_type, self.expected_media_type, url)
        while True:
            attempt += 1
            try:
                with session.get(
                    url,
                    headers=request_headers,
                    stream=True,
                    timeout=self.http_config.download_timeout_sec,
                    allow_redirects=True,
                ) as response:
                    if response.status_code == 304 and Path(self.destination).exists():
                        self.status = "cached"
                        self.response_etag = response.headers.get(
                            "ETag"
                        ) or self.previous_manifest.get("etag")
                        self.response_last_modified = response.headers.get(
                            "Last-Modified"
                        ) or self.previous_manifest.get("last_modified")
                        part_path.unlink(missing_ok=True)
                        return
                    if response.status_code == 206:
                        self.status = "updated"
                    response.raise_for_status()
                    self._validate_media_type(
                        response.headers.get("Content-Type"),
                        self.expected_media_type,
                        url,
                    )
                    length_header = response.headers.get("Content-Length")
                    total_bytes: Optional[int] = None
                    next_progress: Optional[float] = 0.1
                    if length_header:
                        try:
                            total_bytes = int(length_header)
                        except ValueError:
                            total_bytes = None
                        max_bytes = self.http_config.max_download_size_gb * (1024**3)
                        if total_bytes is not None and total_bytes > max_bytes:
                            self.logger.error(
                                "file exceeds size limit",
                                extra={
                                    "stage": "download",
                                    "size": total_bytes,
                                    "limit": max_bytes,
                                },
                            )
                            raise ConfigError(
                                f"File size {total_bytes} exceeds configured limit of {self.http_config.max_download_size_gb} GB"
                            )
                        if total_bytes:
                            completed_fraction = resume_position / total_bytes
                            if completed_fraction >= 1:
                                next_progress = None
                            else:
                                next_progress = ((int(completed_fraction * 10)) + 1) / 10
                    self.response_etag = response.headers.get("ETag")
                    self.response_last_modified = response.headers.get("Last-Modified")
                    mode = "ab" if resume_position else "wb"
                    bytes_downloaded = resume_position
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with part_path.open(mode) as fh:
                            for chunk in response.iter_content(chunk_size=1 << 20):
                                if not chunk:
                                    continue
                                fh.write(chunk)
                                bytes_downloaded += len(chunk)
                                if total_bytes and next_progress:
                                    progress = bytes_downloaded / total_bytes
                                    while next_progress and progress >= next_progress:
                                        self.logger.info(
                                            "download progress",
                                            extra={
                                                "stage": "download",
                                                "status": "in-progress",
                                                "progress": {
                                                    "percent": round(min(progress, 1.0) * 100, 1)
                                                },
                                            },
                                        )
                                        next_progress += 0.1
                                        if next_progress > 1:
                                            next_progress = None
                                            break
                                if bytes_downloaded > self.http_config.max_download_size_gb * (
                                    1024**3
                                ):
                                    self.logger.error(
                                        "download exceeded size limit",
                                        extra={
                                            "stage": "download",
                                            "size": bytes_downloaded,
                                            "limit": self.http_config.max_download_size_gb
                                            * (1024**3),
                                        },
                                    )
                                    raise ConfigError(
                                        "Download exceeded maximum configured size while streaming"
                                    )
                    except OSError as exc:
                        part_path.unlink(missing_ok=True)
                        self.logger.error(
                            "filesystem error during download",
                            extra={"stage": "download", "error": str(exc)},
                        )
                        if "No space left" in str(exc):
                            raise ConfigError(
                                "No space left on device while writing download"
                            ) from exc
                        raise ConfigError(f"Failed to write download: {exc}") from exc
                    break
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
                requests.exceptions.SSLError,
            ) as exc:
                if attempt > self.http_config.max_retries:
                    raise
                sleep_time = self.http_config.backoff_factor * (2 ** (attempt - 1))
                self.logger.warning(
                    "download retry",
                    extra={
                        "stage": "download",
                        "attempt": attempt,
                        "sleep_sec": sleep_time,
                        "error": str(exc),
                    },
                )
                time.sleep(sleep_time)
        part_path.replace(Path(output_file))
        destination_part_path.unlink(missing_ok=True)


def _get_bucket(
    host: str, http_config: DownloadConfiguration, service: Optional[str] = None
) -> TokenBucket:
    """Return a token bucket keyed by host and optional service name.

    Args:
        host: Hostname extracted from the download URL.
        http_config: Download configuration providing base rate limits.
        service: Logical service identifier enabling per-service overrides.

    Returns:
        TokenBucket instance shared across downloads for throttling, seeded
        with either per-host defaults or service-specific overrides.
    """
    key = f"{service}:{host}" if service else host
    bucket = _TOKEN_BUCKETS.get(key)
    if bucket is None:
        rate = http_config.rate_limit_per_second()
        if service:
            service_rate = http_config.parse_service_rate_limit(service)
            if service_rate:
                rate = service_rate
        bucket = TokenBucket(rate_per_sec=rate, capacity=rate)
        _TOKEN_BUCKETS[key] = bucket
    return bucket


def download_stream(
    *,
    url: str,
    destination: Path,
    headers: Dict[str, str],
    previous_manifest: Optional[Dict[str, object]],
    http_config: DownloadConfiguration,
    cache_dir: Path,
    logger: logging.Logger,
    expected_media_type: Optional[str] = None,
    service: Optional[str] = None,
) -> DownloadResult:
    """Download ontology content with HEAD validation, rate limiting, caching, retries, and hash checks.

    Args:
        url: URL of the ontology document to download.
        destination: Target file path for the downloaded content.
        headers: HTTP headers forwarded to the download request.
        previous_manifest: Manifest metadata from a prior run, used for caching.
        http_config: Download configuration containing timeouts, limits, and rate controls.
        cache_dir: Directory where intermediary cached files are stored.
        logger: Logger adapter for structured download telemetry.
        expected_media_type: Expected Content-Type for validation, if known.
        service: Logical service identifier for per-service rate limiting.

    Returns:
        DownloadResult describing the final artifact and metadata.

    Raises:
        ConfigError: If validation fails, limits are exceeded, or HTTP errors occur.
    """
    secure_url = validate_url_security(url, http_config)
    parsed = urlparse(secure_url)
    bucket = _get_bucket(parsed.hostname or "default", http_config, service)
    bucket.consume()

    start_time = time.monotonic()
    _log_download_memory(logger, "before")
    downloader = StreamingDownloader(
        destination=destination,
        headers=headers,
        http_config=http_config,
        previous_manifest=previous_manifest,
        logger=logger,
        expected_media_type=expected_media_type,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(destination.name)
    try:
        cached_path = Path(
            pooch.retrieve(
                secure_url,
                path=cache_dir,
                fname=safe_name,
                known_hash=None,
                downloader=downloader,
                progressbar=False,
            )
        )
    except requests.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        message = f"HTTP error while downloading {secure_url}: {exc}"
        retryable = _is_retryable_status(status_code)
        logger.error(
            "download request failed",
            extra={
                "stage": "download",
                "url": secure_url,
                "error": str(exc),
                "status_code": status_code,
            },
        )
        raise DownloadFailure(message, status_code=status_code, retryable=retryable) from exc
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.exceptions.SSLError,
    ) as exc:
        logger.error(
            "download request failed",
            extra={"stage": "download", "url": secure_url, "error": str(exc)},
        )
        raise DownloadFailure(
            f"HTTP error while downloading {secure_url}: {exc}", retryable=True
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive catch for pooch errors
        logger.error(
            "pooch download error",
            extra={"stage": "download", "url": secure_url, "error": str(exc)},
        )
        raise ConfigError(f"Download failed for {secure_url}: {exc}") from exc
    if downloader.status == "cached":
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(
            "cache hit",
            extra={"stage": "download", "status": "cached", "elapsed_ms": round(elapsed, 2)},
        )
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, destination)
        sha256 = sha256_file(destination)
        _log_download_memory(logger, "after")
        return DownloadResult(
            path=destination,
            status="cached",
            sha256=sha256,
            etag=downloader.response_etag,
            last_modified=downloader.response_last_modified,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination)
    sha256 = sha256_file(destination)
    expected_hash = previous_manifest.get("sha256") if previous_manifest else None
    if expected_hash and expected_hash != sha256:
        logger.error(
            "sha256 mismatch detected",
            extra={
                "stage": "download",
                "expected": expected_hash,
                "actual": sha256,
                "url": secure_url,
            },
        )
        destination.unlink(missing_ok=True)
        cached_path.unlink(missing_ok=True)
        return download_stream(
            url=url,
            destination=destination,
            headers=headers,
            previous_manifest=None,
            http_config=http_config,
            cache_dir=cache_dir,
            logger=logger,
            expected_media_type=expected_media_type,
            service=service,
        )
    elapsed = (time.monotonic() - start_time) * 1000
    logger.info(
        "download complete",
        extra={
            "stage": "download",
            "status": downloader.status,
            "elapsed_ms": round(elapsed, 2),
            "sha256": sha256,
        },
    )
    _log_download_memory(logger, "after")
    return DownloadResult(
        path=destination,
        status=downloader.status,
        sha256=sha256,
        etag=downloader.response_etag,
        last_modified=downloader.response_last_modified,
    )


# --- Validation utilities ---
import logging
import re
from dataclasses import dataclass

rdflib = get_rdflib()
pronto = get_pronto()
owlready2 = get_owlready2()


@dataclass(slots=True)
class ValidationRequest:
    """Parameters describing a single validation task.

    Attributes:
        name: Identifier of the validator to execute.
        file_path: Path to the ontology document to inspect.
        normalized_dir: Directory used to write normalized artifacts.
        validation_dir: Directory for validator reports and logs.
        config: Resolved configuration that supplies timeout thresholds.

    Examples:
        >>> from pathlib import Path
        >>> from DocsToKG.OntologyDownload import ResolvedConfig
        >>> req = ValidationRequest(
        ...     name="rdflib",
        ...     file_path=Path("ontology.owl"),
        ...     normalized_dir=Path("normalized"),
        ...     validation_dir=Path("validation"),
        ...     config=ResolvedConfig.from_defaults(),
        ... )
        >>> req.name
        'rdflib'
    """

    name: str
    file_path: Path
    normalized_dir: Path
    validation_dir: Path
    config: ResolvedConfig


@dataclass(slots=True)
class ValidationResult:
    """Outcome produced by a validator.

    Attributes:
        ok: Indicates whether the validator succeeded.
        details: Arbitrary metadata describing validator output.
        output_files: Generated files for downstream processing.

    Examples:
        >>> result = ValidationResult(ok=True, details={"triples": 10}, output_files=["ontology.ttl"])
        >>> result.ok
        True
    """

    ok: bool
    details: Dict[str, object]
    output_files: List[str]

    def to_dict(self) -> Dict[str, object]:
        """Represent the validation result as a JSON-serializable dict.

        Args:
            None.

        Returns:
            Dictionary with boolean status, detail payload, and output paths.
        """
        return {
            "ok": self.ok,
            "details": self.details,
            "output_files": self.output_files,
        }


class ValidationTimeout(Exception):
    """Raised when a validation task exceeds the configured timeout.

    Args:
        message: Optional description of the timeout condition.

    Examples:
        >>> raise ValidationTimeout("rdflib exceeded 60s")
        Traceback (most recent call last):
        ...
        ValidationTimeout: rdflib exceeded 60s
    """


def _log_validation_memory(logger: logging.Logger, validator: str, event: str) -> None:
    """Emit memory usage diagnostics for a validator when debug logging is enabled.

    Args:
        logger: Logger responsible for validator telemetry.
        validator: Name of the validator emitting the event.
        event: Lifecycle label describing when the measurement is captured.
    """
    is_enabled = getattr(logger, "isEnabledFor", None)
    if callable(is_enabled):
        enabled = is_enabled(logging.DEBUG)
    else:  # pragma: no cover - fallback for stub loggers
        enabled = False
    if not enabled:
        return
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024**2)
    logger.debug(
        "memory usage",
        extra={
            "stage": "validate",
            "validator": validator,
            "event": event,
            "memory_mb": round(memory_mb, 2),
        },
    )


def _write_validation_json(path: Path, payload: MutableMapping[str, object]) -> None:
    """Persist structured validation metadata to disk as JSON.

    Args:
        path: Destination path for the JSON payload.
        payload: Mapping containing validation results.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _python_merge_sort(source: Path, destination: Path, *, chunk_size: int = 100_000) -> None:
    """Sort an N-Triples file using a disk-backed merge strategy.

    Args:
        source: Path to the unsorted triple file.
        destination: Output path that receives sorted triples.
        chunk_size: Number of lines loaded into memory per chunk before flushing.
    """

    with tempfile.TemporaryDirectory(prefix="ontology-sort-") as tmp_dir:
        chunk_paths: List[Path] = []
        with source.open("r", encoding="utf-8") as reader:
            while True:
                lines = list(islice(reader, chunk_size))
                if not lines:
                    break
                lines.sort()
                chunk_path = Path(tmp_dir) / f"chunk-{len(chunk_paths)}.nt"
                chunk_path.write_text("".join(lines), encoding="utf-8")
                chunk_paths.append(chunk_path)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if not chunk_paths:
            destination.write_text("", encoding="utf-8")
            return

        with contextlib.ExitStack() as stack:
            iterators: List[Iterator[str]] = []
            for chunk_path in chunk_paths:
                handle = stack.enter_context(chunk_path.open("r", encoding="utf-8"))
                iterators.append(iter(handle))
            with destination.open("w", encoding="utf-8") as writer:
                for line in heapq.merge(*iterators):
                    writer.write(line)


def _term_to_string(term, namespace_manager) -> str:
    """Render an RDF term using the provided namespace manager.

    Args:
        term: RDF term such as a URIRef, BNode, or Literal.
        namespace_manager: Namespace manager responsible for prefix resolution.

    Returns:
        Term rendered in N3 form, falling back to :func:`str` when unavailable.
    """
    formatter = getattr(term, "n3", None)
    if callable(formatter):
        return formatter(namespace_manager)
    return str(term)


def _canonicalize_turtle(graph) -> str:
    """Return canonical Turtle output with sorted prefixes and triples.

    The canonical form mirrors the ontology downloader specification by sorting
    prefixes lexicographically and emitting triples ordered by subject,
    predicate, and object so downstream hashing yields deterministic values.

    Args:
        graph: RDF graph containing triples to canonicalize.

    Returns:
        Canonical Turtle serialization as a string.
    """

    namespace_manager = getattr(graph, "namespace_manager", None)
    if namespace_manager is None or not hasattr(namespace_manager, "namespaces"):
        raise AttributeError("graph lacks namespace manager support")

    try:
        namespace_items = list(namespace_manager.namespaces())
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise AttributeError("unable to iterate namespaces") from exc

    prefix_map: Dict[str, str] = {}
    for prefix, namespace in namespace_items:
        key = prefix or ""
        prefix_map[key] = str(namespace)

    try:
        triples = list(graph)
    except Exception as exc:  # pragma: no cover - stub graphs are not iterable
        raise AttributeError("graph is not iterable") from exc

    triple_lines = [
        f"{_term_to_string(subject, namespace_manager)} {_term_to_string(predicate, namespace_manager)} {_term_to_string(obj, namespace_manager)} ."
        for subject, predicate, obj in sorted(
            ((s, p, o) for s, p, o in triples),
            key=lambda item: (
                _term_to_string(item[0], namespace_manager),
                _term_to_string(item[1], namespace_manager),
                _term_to_string(item[2], namespace_manager),
            ),
        )
    ]

    bnode_map: Dict[str, str] = {}
    triple_lines = [_canonicalize_blank_nodes_line(line, bnode_map) for line in triple_lines]

    prefix_lines = []
    for key in sorted(prefix_map):
        label = f"{key}:" if key else ":"
        prefix_lines.append(f"@prefix {label} <{prefix_map[key]}> .")

    lines: List[str] = []
    lines.extend(prefix_lines)
    if prefix_lines and triple_lines:
        lines.append("")
    lines.extend(triple_lines)
    return "\n".join(lines) + ("\n" if lines else "")


_BNODE_PATTERN = re.compile(r"_:[A-Za-z0-9]+")


def _canonicalize_blank_nodes_line(line: str, mapping: Dict[str, str]) -> str:
    """Replace blank node identifiers with deterministic sequential labels.

    Args:
        line: Serialized triple line containing blank node identifiers.
        mapping: Mutable mapping preserving deterministic blank node assignments.

    Returns:
        Triple line with normalized blank node identifiers.
    """

    def _replace(match: re.Match[str]) -> str:
        key = match.group(0)
        mapped = mapping.get(key)
        if mapped is None:
            mapped = f"_:b{len(mapping)}"
            mapping[key] = mapped
        return mapped

    return _BNODE_PATTERN.sub(_replace, line)


def _sort_triple_file(source: Path, destination: Path) -> None:
    """Sort serialized triple lines using platform sort when available.

    Args:
        source: Path to the unsorted triple file.
        destination: Output path that receives sorted triples.
    """

    sort_binary = shutil.which("sort")
    if sort_binary:
        try:
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                subprocess.run(  # noqa: PLW1510 - intentional check handling
                    [sort_binary, source.as_posix()],
                    check=True,
                    stdout=handle,
                    text=True,
                )
            return
        except (subprocess.SubprocessError, OSError):
            # Fall back to pure Python sorting when the external command fails.
            pass

    _python_merge_sort(source, destination)


def normalize_streaming(
    source: Path,
    output_path: Optional[Path] = None,
    *,
    graph=None,
    chunk_bytes: int = 1 << 20,
) -> str:
    """Normalize ontologies using streaming canonical Turtle serialization.

    The streaming path serializes triples to a temporary file, leverages the
    platform ``sort`` command (when available) to order triples lexicographically,
    and streams the canonical Turtle output while computing a SHA-256 digest.
    When ``output_path`` is provided the canonical form is persisted without
    retaining the entire content in memory.

    Args:
        source: Path to the ontology document providing triples.
        output_path: Optional destination for the normalized Turtle document.
        graph: Optional pre-loaded RDF graph re-used instead of reparsing.
        chunk_bytes: Threshold controlling how frequently buffered bytes are flushed.

    Returns:
        SHA-256 hex digest of the canonical Turtle content.
    """

    graph_obj = graph if graph is not None else rdflib.Graph()
    if graph is None:
        graph_obj.parse(source.as_posix())

    namespace_manager = getattr(graph_obj, "namespace_manager", None)
    if namespace_manager is None or not hasattr(namespace_manager, "namespaces"):
        raise AttributeError("graph lacks namespace manager support")

    try:
        namespace_items = list(namespace_manager.namespaces())
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise AttributeError("unable to iterate namespaces") from exc

    prefix_map: Dict[str, str] = {}
    for prefix, namespace in namespace_items:
        key = prefix or ""
        prefix_map[key] = str(namespace)

    prefix_lines = []
    for key in sorted(prefix_map):
        label = f"{key}:" if key else ":"
        prefix_lines.append(f"@prefix {label} <{prefix_map[key]}> .\n")

    chunk_limit = max(1, int(chunk_bytes))
    buffer = bytearray()
    sha256 = hashlib.sha256()

    def _flush(writer: Optional[BinaryIO]) -> None:
        if not buffer:
            return
        sha256.update(buffer)
        if writer is not None:
            writer.write(buffer)
        buffer.clear()

    with tempfile.TemporaryDirectory(prefix="ontology-stream-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        unsorted_path = tmp_path / "triples.unsorted"
        with unsorted_path.open("w", encoding="utf-8", newline="\n") as handle:
            for subject, predicate, obj in graph_obj:
                line = (
                    f"{_term_to_string(subject, namespace_manager)} "
                    f"{_term_to_string(predicate, namespace_manager)} "
                    f"{_term_to_string(obj, namespace_manager)} ."
                )
                handle.write(line + "\n")

        sorted_path = tmp_path / "triples.sorted"
        _sort_triple_file(unsorted_path, sorted_path)

        with contextlib.ExitStack() as stack:
            writer: Optional[BinaryIO] = None
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = stack.enter_context(output_path.open("wb"))

            def _emit(text: str) -> None:
                buffer.extend(text.encode("utf-8"))
                if len(buffer) >= chunk_limit:
                    _flush(writer)

            wrote_any = False
            for line in prefix_lines:
                _emit(line)
                wrote_any = True

            bnode_map: Dict[str, str] = {}
            blank_line_pending = bool(prefix_lines)

            with sorted_path.open("r", encoding="utf-8") as reader:
                for raw_line in reader:
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    if blank_line_pending:
                        _emit("\n")
                        blank_line_pending = False
                    canonical_line = _canonicalize_blank_nodes_line(line, bnode_map) + "\n"
                    _emit(canonical_line)
                    wrote_any = True

            if not wrote_any and writer is not None:
                writer.truncate(0)
                writer.flush()

            _flush(writer)

    return sha256.hexdigest()


class ValidatorSubprocessError(RuntimeError):
    """Raised when a validator subprocess exits unsuccessfully.

    Attributes:
        message: Human-readable description of the underlying subprocess failure.

    Examples:
        >>> raise ValidatorSubprocessError("rdflib validator crashed")
        Traceback (most recent call last):
        ...
        ValidatorSubprocessError: rdflib validator crashed
    """


def _worker_pronto(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Pronto validation logic and emit JSON-friendly results."""

    file_path = Path(payload["file_path"])
    ontology = pronto.Ontology(file_path.as_posix())
    terms = len(list(ontology.terms()))
    result: Dict[str, Any] = {"ok": True, "terms": terms}

    normalized_path = payload.get("normalized_path")
    if normalized_path:
        destination = Path(normalized_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ontology.dump(destination.as_posix(), format="obojson")
        result["normalized_written"] = True

    return result


def _worker_owlready2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Owlready2 validation logic and emit JSON-friendly results."""

    file_path = Path(payload["file_path"])
    ontology = owlready2.get_ontology(file_path.resolve().as_uri()).load()
    entities = len(list(ontology.classes()))
    return {"ok": True, "entities": entities}


_WORKER_DISPATCH = {
    "pronto": _worker_pronto,
    "owlready2": _worker_owlready2,
}


def _run_validator_subprocess(
    name: str, payload: Dict[str, object], *, timeout: int
) -> Dict[str, object]:
    """Execute a validator worker module within a subprocess.

    The subprocess workflow enforces parser timeouts, returns JSON payloads,
    and helps release memory held by heavy libraries such as Pronto and
    Owlready2 after each validation completes.
    """

    command = [sys.executable, "-m", "DocsToKG.OntologyDownload.validation", "worker", name]
    env = os.environ.copy()

    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationTimeout(f"{name} validator exceeded {timeout}s") from exc
    except OSError as exc:
        raise ValidatorSubprocessError(f"Failed to launch {name} validator: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        message = stderr or (f"{name} validator subprocess failed with code {completed.returncode}")
        raise ValidatorSubprocessError(message)

    stdout = completed.stdout.decode("utf-8", errors="ignore").strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValidatorSubprocessError(f"{name} validator returned invalid JSON output") from exc


def _run_with_timeout(func, timeout_sec: int) -> None:
    """Execute a callable and raise :class:`ValidationTimeout` on deadline expiry.

    Args:
        func: Callable invoked without arguments.
        timeout_sec: Number of seconds allowed for execution.

    Returns:
        None

    Raises:
        ValidationTimeout: When the callable exceeds the allotted runtime.
    """
    if platform.system() in ("Linux", "Darwin"):
        import signal

        class _Alarm(Exception):
            """Sentinel exception raised when the alarm signal fires.

            Args:
                message: Optional description associated with the exception.

            Attributes:
                message: Optional description associated with the exception.

            Examples:
                >>> try:
                ...     raise _Alarm()
                ... except _Alarm:
                ...     pass
            """

        def _handler(signum, frame):  # pragma: no cover - platform dependent
            """Signal handler converting SIGALRM into :class:`ValidationTimeout`.

            Args:
                signum: Received signal number.
                frame: Current stack frame (unused).
            """
            raise ValidationTimeout()  # pragma: no cover - bridges to outer scope

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_sec)
        try:
            func()
        finally:
            signal.alarm(0)
    else:  # Windows
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                future.result(timeout=timeout_sec)
            except FuturesTimeoutError as exc:  # pragma: no cover - platform specific
                raise ValidationTimeout() from exc


def _prepare_xbrl_package(
    request: ValidationRequest, logger: logging.Logger
) -> tuple[Path, List[str]]:
    """Extract XBRL taxonomy ZIP archives for downstream validation.

    Args:
        request: Validation request describing the ontology package under test.
        logger: Logger used to record extraction telemetry.

    Returns:
        Tuple containing the entrypoint path passed to Arelle and a list of artifacts.

    Raises:
        ValueError: If the archive is malformed or contains unsafe paths.
    """
    package_path = request.file_path
    if package_path.suffix.lower() != ".zip":
        return package_path, []
    if not zipfile.is_zipfile(package_path):
        raise ValueError("XBRL package is not a valid ZIP archive")

    with zipfile.ZipFile(package_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path detected in archive: {member.filename}")
            if member.compress_size == 0 and member.file_size > 0:
                raise ValueError(f"Zip entry {member.filename} has invalid compression size")
            ratio = member.file_size / max(member.compress_size, 1)
            if ratio > 10:
                raise ValueError(
                    f"Zip entry {member.filename} exceeds compression ratio limit (ratio={ratio:.1f})"
                )

    temp_dir = Path(tempfile.mkdtemp(prefix="ontofetch-xbrl-"))
    try:
        with zipfile.ZipFile(package_path) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                target_path = temp_dir / member_path
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    final_dir = request.validation_dir / "arelle" / package_path.stem
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_dir), final_dir)
    logger.info(
        "extracted xbrl package",
        extra={"stage": "validate", "validator": "arelle", "destination": str(final_dir)},
    )

    entrypoint_candidates = sorted(final_dir.rglob("*.xsd")) or sorted(final_dir.rglob("*.xml"))
    entrypoint = entrypoint_candidates[0] if entrypoint_candidates else package_path
    artifacts = [str(path) for path in final_dir.rglob("*") if path.is_file()]
    return entrypoint, artifacts


def validate_rdflib(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Parse ontologies with rdflib, canonicalize Turtle output, and emit hashes.

    Args:
        request: Validation request describing the target ontology and output directories.
        logger: Logger adapter used for structured validation events.

    Returns:
        ValidationResult capturing success state, metadata, canonical hash,
        and generated files.

    Raises:
        ValidationTimeout: Propagated when parsing exceeds configured timeout.
    """
    graph = rdflib.Graph()
    payload: Dict[str, object] = {"ok": False}
    timeout = request.config.defaults.validation.parser_timeout_sec

    def _parse() -> None:
        """Parse the ontology with rdflib to populate the graph object."""
        graph.parse(request.file_path.as_posix())

    try:
        _log_validation_memory(logger, "rdflib", "before")
        _run_with_timeout(_parse, timeout)
        _log_validation_memory(logger, "rdflib", "after")
        triple_count = len(graph)
        payload = {"ok": True, "triples": triple_count}
        output_files: List[str] = []
        normalization_mode = "in-memory"

        if "ttl" in request.config.defaults.normalize_to:
            request.normalized_dir.mkdir(parents=True, exist_ok=True)
            stem = request.file_path.stem
            normalized_ttl = request.normalized_dir / f"{stem}.ttl"
            threshold_bytes = (
                request.config.defaults.validation.streaming_normalization_threshold_mb
                * 1024
                * 1024
            )
            file_size = request.file_path.stat().st_size
            streaming_hash: Optional[str] = None
            normalized_sha: Optional[str] = None
            if file_size >= threshold_bytes:
                normalization_mode = "streaming"
                try:
                    streaming_hash = normalize_streaming(
                        request.file_path,
                        output_path=normalized_ttl,
                        graph=graph,
                    )
                    normalized_sha = streaming_hash
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "streaming normalization failed, falling back to in-memory",
                        extra={
                            "stage": "validate",
                            "validator": "rdflib",
                            "error": str(exc),
                        },
                    )
                    normalization_mode = "in-memory"
                    streaming_hash = None

            if normalized_sha is None:
                try:
                    canonical_ttl = _canonicalize_turtle(graph)
                except AttributeError:
                    graph.serialize(destination=normalized_ttl, format="turtle")
                    canonical_ttl = normalized_ttl.read_text(encoding="utf-8")
                else:
                    normalized_ttl.write_text(canonical_ttl, encoding="utf-8")
                normalized_sha = hashlib.sha256(canonical_ttl.encode("utf-8")).hexdigest()

            payload["normalized_sha256"] = normalized_sha
            payload["normalization_mode"] = normalization_mode
            output_files.append(str(normalized_ttl))
            if streaming_hash is not None:
                payload["streaming_nt_sha256"] = streaming_hash
        if (
            "ttl" in request.config.defaults.normalize_to
            and payload.get("streaming_nt_sha256") is None
        ):
            try:
                payload["streaming_nt_sha256"] = normalize_streaming(
                    request.file_path,
                    graph=graph,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "failed to compute streaming normalization hash",
                    extra={
                        "stage": "validate",
                        "validator": "rdflib",
                        "error": str(exc),
                    },
                )

        _write_validation_json(request.validation_dir / "rdflib_parse.json", payload)
        return ValidationResult(ok=True, details=payload, output_files=output_files)
    except ValidationTimeout:
        message = f"Parser timeout after {timeout}s"
        payload = {"ok": False, "error": message}
    except MemoryError as exc:
        payload = {"ok": False, "error": "rdflib memory limit exceeded"}
        logger.warning(
            "rdflib memory error",
            extra={"stage": "validate", "validator": "rdflib", "error": str(exc)},
        )
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "rdflib_parse.json", payload)
    logger.warning(
        "rdflib validation failed",
        extra={"stage": "validate", "validator": "rdflib", "error": payload.get("error")},
    )
    return ValidationResult(ok=False, details=payload, output_files=[])


def validate_pronto(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Execute Pronto validation in an isolated subprocess and emit OBO Graphs when requested.

    Args:
        request: Validation request describing ontology inputs and output directories.
        logger: Structured logger for recording warnings and failures.

    Returns:
        ValidationResult with parsed ontology statistics, subprocess output,
        and any generated artifacts.

    Raises:
        ValidationTimeout: Propagated when Pronto takes longer than allowed.
    """

    try:
        timeout = request.config.defaults.validation.parser_timeout_sec
        payload: Dict[str, object] = {"file_path": str(request.file_path)}
        normalized_path: Optional[Path] = None
        if "obographs" in request.config.defaults.normalize_to:
            request.normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = request.normalized_dir / (request.file_path.stem + ".json")
            payload["normalized_path"] = str(normalized_path)

        _log_validation_memory(logger, "pronto", "before")
        result_payload = _run_validator_subprocess("pronto", payload, timeout=timeout)
        _log_validation_memory(logger, "pronto", "after")
        result_payload.setdefault("ok", True)
        output_files: List[str] = []
        if normalized_path and result_payload.get("normalized_written"):
            output_files.append(str(normalized_path))
        _write_validation_json(request.validation_dir / "pronto_parse.json", result_payload)
        return ValidationResult(
            ok=bool(result_payload.get("ok")),
            details=result_payload,
            output_files=output_files,
        )
    except ValidationTimeout:
        payload = {"ok": False, "error": f"Parser timeout after {timeout}s"}
    except ValidatorSubprocessError as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive catch
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "pronto_parse.json", payload)
    logger.warning(
        "pronto validation failed",
        extra={"stage": "validate", "validator": "pronto", "error": payload.get("error")},
    )
    return ValidationResult(ok=False, details=payload, output_files=[])


def validate_owlready2(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Inspect ontologies with Owlready2 in a subprocess to count entities and catch parsing errors.

    Args:
        request: Validation request referencing the ontology to parse.
        logger: Logger for reporting failures or memory warnings.

    Returns:
        ValidationResult summarizing entity counts or failure details.

    Raises:
        None
    """
    try:
        size_mb = request.file_path.stat().st_size / (1024**2)
        limit = request.config.defaults.validation.skip_reasoning_if_size_mb
        if size_mb > limit:
            reason = f"Skipping reasoning for large file (> {limit} MB)"
            payload = {"ok": True, "skipped": True, "reason": reason}
            _write_validation_json(request.validation_dir / "owlready2_parse.json", payload)
            logger.info(
                "owlready2 reasoning skipped",
                extra={
                    "stage": "validate",
                    "validator": "owlready2",
                    "file_size_mb": round(size_mb, 2),
                    "limit_mb": limit,
                },
            )
            return ValidationResult(ok=True, details=payload, output_files=[])
        timeout = request.config.defaults.validation.parser_timeout_sec
        payload = {"file_path": str(request.file_path)}
        _log_validation_memory(logger, "owlready2", "before")
        result_payload = _run_validator_subprocess("owlready2", payload, timeout=timeout)
        _log_validation_memory(logger, "owlready2", "after")
        result_payload.setdefault("ok", True)
        _write_validation_json(request.validation_dir / "owlready2_parse.json", result_payload)
        return ValidationResult(
            ok=bool(result_payload.get("ok")), details=result_payload, output_files=[]
        )
    except ValidationTimeout:
        message = f"Parser timeout after {request.config.defaults.validation.parser_timeout_sec}s"
        payload = {"ok": False, "error": message}
    except ValidatorSubprocessError as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive catch
        payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "owlready2_parse.json", payload)
    logger.warning(
        "owlready2 validation failed",
        extra={"stage": "validate", "validator": "owlready2", "error": payload.get("error")},
    )
    return ValidationResult(ok=False, details=payload, output_files=[])


def validate_robot(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Run ROBOT CLI validation and conversion workflows when available.

    Args:
        request: Validation request detailing ontology paths and output locations.
        logger: Logger adapter for reporting warnings and CLI errors.

    Returns:
        ValidationResult describing generated outputs or encountered issues.

    Raises:
        None
    """
    robot_path = shutil.which("robot")
    result_payload: Dict[str, object]
    output_files: List[str] = []
    if not robot_path:
        result_payload = {"ok": True, "skipped": True, "reason": "robot binary not found"}
        _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
        logger.info(
            "robot not installed; skipping",
            extra={"stage": "validate", "validator": "robot", "skip": True},
        )
        return ValidationResult(ok=True, details=result_payload, output_files=[])

    normalized_path = request.normalized_dir / (request.file_path.stem + ".ttl")
    request.normalized_dir.mkdir(parents=True, exist_ok=True)
    report_path = request.validation_dir / "robot_report.tsv"
    try:
        _log_validation_memory(logger, "robot", "before")
        convert_cmd = [
            robot_path,
            "convert",
            "-i",
            str(request.file_path),
            "-o",
            str(normalized_path),
        ]
        report_cmd = [robot_path, "report", "-i", str(request.file_path), "-o", str(report_path)]
        subprocess.run(convert_cmd, check=True, capture_output=True)
        subprocess.run(report_cmd, check=True, capture_output=True)
        _log_validation_memory(logger, "robot", "after")
        output_files = [str(normalized_path), str(report_path)]
        result_payload = {"ok": True, "outputs": output_files}
        _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
        return ValidationResult(ok=True, details=result_payload, output_files=output_files)
    except subprocess.CalledProcessError as exc:
        result_payload = {"ok": False, "error": exc.stderr.decode("utf-8", errors="ignore")}
    except MemoryError as exc:
        result_payload = {"ok": False, "error": "robot memory limit exceeded"}
        _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
        logger.warning(
            "robot memory error",
            extra={"stage": "validate", "validator": "robot", "error": str(exc)},
        )
        return ValidationResult(ok=False, details=result_payload, output_files=output_files)
    except Exception as exc:  # pylint: disable=broad-except
        result_payload = {"ok": False, "error": str(exc)}
    _write_validation_json(request.validation_dir / "robot_report.json", result_payload)
    logger.warning(
        "robot validation failed",
        extra={"stage": "validate", "validator": "robot", "error": result_payload.get("error")},
    )
    return ValidationResult(ok=False, details=result_payload, output_files=output_files)


def validate_arelle(request: ValidationRequest, logger: logging.Logger) -> ValidationResult:
    """Validate XBRL ontologies with Arelle CLI if installed.

    Args:
        request: Validation request referencing the ontology under test.
        logger: Logger used to communicate validation progress and failures.

    Returns:
        ValidationResult indicating whether the validation completed and
        referencing any produced log files.

    Raises:
        None
    """
    try:
        from arelle import Cntlr  # type: ignore

        entrypoint, artifacts = _prepare_xbrl_package(request, logger)
        controller = Cntlr.Cntlr(
            logFile=str(request.validation_dir / "arelle.log"), logToBuffer=True
        )
        _log_validation_memory(logger, "arelle", "before")
        controller.run(["--file", str(entrypoint)])
        _log_validation_memory(logger, "arelle", "after")
        payload = {
            "ok": True,
            "log": str(request.validation_dir / "arelle.log"),
            "entrypoint": str(entrypoint),
        }
        if artifacts:
            payload["artifacts"] = artifacts
        _write_validation_json(request.validation_dir / "arelle_validation.json", payload)
        outputs = [payload["log"], *(artifacts or [])]
        return ValidationResult(ok=True, details=payload, output_files=outputs)
    except ValueError as exc:
        payload = {"ok": False, "error": str(exc)}
        _write_validation_json(request.validation_dir / "arelle_validation.json", payload)
        logger.warning(
            "arelle package validation failed",
            extra={"stage": "validate", "validator": "arelle", "error": str(exc)},
        )
        return ValidationResult(ok=False, details=payload, output_files=[])
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
        _write_validation_json(request.validation_dir / "arelle_validation.json", payload)
        logger.warning(
            "arelle validation failed",
            extra={"stage": "validate", "validator": "arelle", "error": payload.get("error")},
        )
        return ValidationResult(ok=False, details=payload, output_files=[])


VALIDATORS = {
    "rdflib": validate_rdflib,
    "pronto": validate_pronto,
    "owlready2": validate_owlready2,
    "robot": validate_robot,
    "arelle": validate_arelle,
}


def _load_validator_plugins(logger: Optional[logging.Logger] = None) -> None:
    """Discover validator plugins registered via entry points."""

    logger = logger or logging.getLogger(__name__)
    try:
        entry_points = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "validator plugin discovery failed",
            extra={"stage": "init", "error": str(exc)},
        )
        return

    for entry in entry_points.select(group="docstokg.ontofetch.validator"):
        try:
            handler = entry.load()
            if not callable(handler):
                raise TypeError("validator plugin must be callable")
            VALIDATORS[entry.name] = handler
            logger.info(
                "validator plugin registered",
                extra={"stage": "init", "validator": entry.name},
            )
        except Exception as exc:  # pragma: no cover - plugin faults
            logger.warning(
                "validator plugin failed",
                extra={"stage": "init", "validator": entry.name, "error": str(exc)},
            )


_load_validator_plugins()


def _run_validator_task(
    validator: Callable[[ValidationRequest, logging.Logger], ValidationResult],
    request: ValidationRequest,
    logger: logging.Logger,
) -> ValidationResult:
    """Execute a single validator with exception guards."""

    try:
        return validator(request, logger)
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"ok": False, "error": str(exc)}
        _write_validation_json(request.validation_dir / f"{request.name}_parse.json", payload)
        logger.error(
            "validator crashed",
            extra={
                "stage": "validate",
                "validator": request.name,
                "error": payload.get("error"),
            },
        )
        return ValidationResult(ok=False, details=payload, output_files=[])


def run_validators(
    requests: Iterable[ValidationRequest], logger: logging.Logger
) -> Dict[str, ValidationResult]:
    """Execute registered validators and aggregate their results.

    Args:
        requests: Iterable of validation requests that specify validators to run.
        logger: Logger adapter shared across validation executions.

    Returns:
        Mapping from validator name to the corresponding ValidationResult.
    """

    request_list = list(requests)
    if not request_list:
        return {}

    def _determine_max_workers() -> int:
        for request in request_list:
            validation_config = getattr(request.config.defaults, "validation", None)
            if validation_config is not None and hasattr(
                validation_config, "max_concurrent_validators"
            ):
                value = int(validation_config.max_concurrent_validators)
                return max(1, min(8, value))
        return 2

    max_workers = _determine_max_workers()
    results: Dict[str, ValidationResult] = {}
    futures: Dict[Any, ValidationRequest] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for request in request_list:
            validator = VALIDATORS.get(request.name)
            if not validator:
                continue
            future = executor.submit(_run_validator_task, validator, request, logger)
            futures[future] = request

        for future in as_completed(futures):
            request = futures[future]
            try:
                results[request.name] = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                payload = {"ok": False, "error": str(exc)}
                _write_validation_json(
                    request.validation_dir / f"{request.name}_parse.json", payload
                )
                logger.error(
                    "validator crashed",
                    extra={
                        "stage": "validate",
                        "validator": request.name,
                        "error": payload.get("error"),
                    },
                )
                results[request.name] = ValidationResult(ok=False, details=payload, output_files=[])

    return results


def _run_worker_cli(name: str, stdin_payload: str) -> None:
    """Execute a validator worker handler and emit JSON to stdout."""

    handler = _WORKER_DISPATCH.get(name)
    if handler is None:
        raise SystemExit(f"Unknown validator worker '{name}'")
    payload = json.loads(stdin_payload or "{}")
    result = handler(payload)
    sys.stdout.write(json.dumps(result))


def main() -> None:
    """Entry point for module execution providing validator worker dispatch.

    Args:
        None.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser(description="Ontology validator worker runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker", help="Run a validator worker")
    worker_parser.add_argument("name", choices=sorted(_WORKER_DISPATCH))
    args = parser.parse_args()

    if args.command == "worker":
        payload = sys.stdin.read()
        _run_worker_cli(args.name, payload)
    else:  # pragma: no cover - argparse enforces choices
        parser.error("Unknown command")


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess dispatch
    main()
# --- Download pipeline ---

from DocsToKG.OntologyDownload import resolvers as _ontology_resolvers

RESOLVERS = _ontology_resolvers.RESOLVERS
FetchPlan = _ontology_resolvers.FetchPlan
normalize_license_to_spdx = _ontology_resolvers.normalize_license_to_spdx

ONTOLOGY_DIR = LOCAL_ONTOLOGY_DIR

MANIFEST_SCHEMA_VERSION = "1.0"

MANIFEST_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DocsToKG Ontology Manifest",
    "type": "object",
    "required": [
        "schema_version",
        "id",
        "resolver",
        "url",
        "filename",
        "status",
        "sha256",
        "downloaded_at",
        "target_formats",
        "validation",
        "artifacts",
        "resolver_attempts",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "id": {"type": "string", "minLength": 1},
        "resolver": {"type": "string", "minLength": 1},
        "url": {
            "type": "string",
            "format": "uri",
            "pattern": r"^https?://",
        },
        "filename": {"type": "string", "minLength": 1},
        "version": {"type": ["string", "null"]},
        "license": {"type": ["string", "null"]},
        "status": {"type": "string", "minLength": 1},
        "sha256": {"type": "string", "minLength": 1},
        "normalized_sha256": {"type": ["string", "null"]},
        "fingerprint": {"type": ["string", "null"]},
        "etag": {"type": ["string", "null"]},
        "last_modified": {"type": ["string", "null"]},
        "downloaded_at": {"type": "string", "format": "date-time"},
        "target_formats": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "validation": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "details": {"type": "object"},
                    "output_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["ok", "details", "output_files"],
            },
        },
        "artifacts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "resolver_attempts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "resolver": {"type": "string"},
                    "url": {"type": "string"},
                    "attempt": {"type": "integer", "minimum": 1},
                    "status": {"type": "string"},
                    "error": {"type": "string"},
                },
                "required": ["resolver"],
            },
        },
    },
    "additionalProperties": True,
}

Draft202012Validator.check_schema(MANIFEST_JSON_SCHEMA)
_MANIFEST_VALIDATOR = Draft202012Validator(MANIFEST_JSON_SCHEMA)


def get_manifest_schema() -> Dict[str, Any]:
    """Return a deep copy of the manifest JSON Schema definition.

    Args:
        None

    Returns:
        Dictionary describing the manifest JSON Schema.
    """

    return deepcopy(MANIFEST_JSON_SCHEMA)


def validate_manifest_dict(payload: Mapping[str, Any], *, source: Optional[Path] = None) -> None:
    """Validate manifest payload against the JSON Schema definition.

    Args:
        payload: Manifest dictionary loaded from JSON.
        source: Optional filesystem path for contextual error reporting.

    Returns:
        None

    Raises:
        ConfigurationError: If validation fails.
    """

    try:
        _MANIFEST_VALIDATOR.validate(payload)
    except JSONSchemaValidationError as exc:
        location = " -> ".join(str(part) for part in exc.path)
        message = exc.message
        if location:
            message = f"{location}: {message}"
        context = f" for {source}" if source else ""
        raise ConfigurationError(f"Manifest validation failed{context}: {message}") from exc


class OntologyDownloadError(RuntimeError):
    """Base exception for ontology download failures.

    Args:
        message: Description of the failure encountered.

    Examples:
        >>> raise OntologyDownloadError("unexpected error")
        Traceback (most recent call last):
        ...
        OntologyDownloadError: unexpected error
    """


class ResolverError(OntologyDownloadError):
    """Raised when resolver planning fails.

    Args:
        message: Description of the resolver failure.

    Examples:
        >>> raise ResolverError("resolver unavailable")
        Traceback (most recent call last):
        ...
        ResolverError: resolver unavailable
    """


class ValidationError(OntologyDownloadError):
    """Raised when validation encounters unrecoverable issues.

    Args:
        message: Human-readable description of the validation failure.

    Examples:
        >>> raise ValidationError("robot validator crashed")
        Traceback (most recent call last):
        ...
        ValidationError: robot validator crashed
    """


class ConfigurationError(OntologyDownloadError):
    """Raised when configuration or manifest validation fails.

    Args:
        message: Details about the configuration inconsistency.

    Examples:
        >>> raise ConfigurationError("manifest missing sha256")
        Traceback (most recent call last):
        ...
        ConfigurationError: manifest missing sha256
    """


@dataclass(slots=True)
class FetchSpec:
    """Specification describing a single ontology download.

    Attributes:
        id: Stable identifier for the ontology to fetch.
        resolver: Name of the resolver strategy used to locate resources.
        extras: Resolver-specific configuration overrides.
        target_formats: Normalized ontology formats that should be produced.

    Examples:
        >>> spec = FetchSpec(id="CHEBI", resolver="obo", extras={}, target_formats=("owl",))
        >>> spec.resolver
        'obo'
    """

    id: str
    resolver: str
    extras: Dict[str, object]
    target_formats: Sequence[str]


@dataclass(slots=True)
class FetchResult:
    """Outcome of a single ontology fetch operation.

    Attributes:
        spec: Fetch specification that initiated the download.
        local_path: Path to the downloaded ontology document.
        status: Final download status (e.g., `success`, `skipped`).
        sha256: SHA-256 digest of the downloaded file.
        manifest_path: Path to the generated manifest JSON file.
        artifacts: Ancillary files produced during extraction or validation.

    Examples:
        >>> from pathlib import Path
        >>> spec = FetchSpec(id="CHEBI", resolver="obo", extras={}, target_formats=("owl",))
        >>> result = FetchResult(
        ...     spec=spec,
        ...     local_path=Path("CHEBI.owl"),
        ...     status="success",
        ...     sha256="deadbeef",
        ...     manifest_path=Path("manifest.json"),
        ...     artifacts=(),
        ... )
        >>> result.status
        'success'
    """

    spec: FetchSpec
    local_path: Path
    status: str
    sha256: str
    manifest_path: Path
    artifacts: Sequence[str]


ResolvedConfig.model_rebuild()


@dataclass(slots=True)
class Manifest:
    """Provenance information for a downloaded ontology artifact.

    Attributes:
        schema_version: Manifest schema version identifier.
        id: Ontology identifier recorded in the manifest.
        resolver: Resolver used to retrieve the ontology.
        url: Final URL from which the ontology was fetched.
        filename: Local filename of the downloaded artifact.
        version: Resolver-reported ontology version, if available.
        license: License identifier associated with the ontology.
        status: Result status reported by the downloader.
        sha256: Hash of the downloaded artifact for integrity checking.
        normalized_sha256: Hash of the canonical normalized TTL output.
        fingerprint: Composite fingerprint combining key provenance values.
        etag: HTTP ETag returned by the upstream server, when provided.
        last_modified: Upstream last-modified timestamp, if supplied.
        downloaded_at: UTC timestamp of the completed download.
        target_formats: Desired conversion targets for normalization.
        validation: Mapping of validator names to their results.
        artifacts: Additional file paths generated during processing.
        resolver_attempts: Ordered record of resolver attempts during download.

    Examples:
        >>> manifest = Manifest(
        ...     schema_version="1.0",
        ...     id="CHEBI",
        ...     resolver="obo",
        ...     url="https://example.org/chebi.owl",
        ...     filename="chebi.owl",
        ...     version=None,
        ...     license="CC-BY",
        ...     status="success",
        ...     sha256="deadbeef",
        ...     normalized_sha256=None,
        ...     fingerprint=None,
        ...     etag=None,
        ...     last_modified=None,
        ...     downloaded_at="2024-01-01T00:00:00Z",
        ...     target_formats=("owl",),
        ...     validation={},
        ...     artifacts=(),
        ...     resolver_attempts=(),
        ... )
        >>> manifest.resolver
        'obo'
    """

    schema_version: str
    id: str
    resolver: str
    url: str
    filename: str
    version: Optional[str]
    license: Optional[str]
    status: str
    sha256: str
    normalized_sha256: Optional[str]
    fingerprint: Optional[str]
    etag: Optional[str]
    last_modified: Optional[str]
    downloaded_at: str
    target_formats: Sequence[str]
    validation: Dict[str, ValidationResult]
    artifacts: Sequence[str]
    resolver_attempts: Sequence[Dict[str, object]]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary for the manifest.

        Args:
            None

        Returns:
            Dictionary representing the manifest payload.
        """

        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "resolver": self.resolver,
            "url": self.url,
            "filename": self.filename,
            "version": self.version,
            "license": self.license,
            "status": self.status,
            "sha256": self.sha256,
            "normalized_sha256": self.normalized_sha256,
            "fingerprint": self.fingerprint,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "downloaded_at": self.downloaded_at,
            "target_formats": list(self.target_formats),
            "validation": {name: result.to_dict() for name, result in self.validation.items()},
            "artifacts": list(self.artifacts),
            "resolver_attempts": [dict(entry) for entry in self.resolver_attempts],
        }

    def to_json(self) -> str:
        """Serialize the manifest to a stable, human-readable JSON string.

        Args:
            None

        Returns:
            JSON document encoding the manifest metadata.
        """

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class Resolver(Protocol):
    """Protocol describing resolver planning behaviour.

    Attributes:
        None

    Examples:
        >>> import logging
        >>> spec = FetchSpec(id="CHEBI", resolver="dummy", extras={}, target_formats=("owl",))
        >>> class DummyResolver:
        ...     def plan(self, spec, config, logger):
        ...         return FetchPlan(
        ...             url="https://example.org/chebi.owl",
        ...             headers={},
        ...             filename_hint="chebi.owl",
        ...             version="v1",
        ...             license="CC-BY",
        ...             media_type="application/rdf+xml",
        ...         )
        ...
        >>> plan = DummyResolver().plan(spec, ResolvedConfig.from_defaults(), logging.getLogger("test"))
        >>> plan.url
        'https://example.org/chebi.owl'
    """

    def plan(self, spec: FetchSpec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Return a FetchPlan describing how to obtain the ontology.

        Args:
            spec: Ontology fetch specification under consideration.
            config: Fully resolved configuration containing defaults.
            logger: Logger adapter scoped to the current fetch request.

        Returns:
            Concrete plan containing download URL, headers, and metadata.
        """
        ...


@dataclass(slots=True)
class ResolverCandidate:
    """Resolver plan captured for download-time fallback.

    Attributes:
        resolver: Name of the resolver that produced the plan.
        plan: Concrete :class:`FetchPlan` describing how to fetch the ontology.

    Examples:
        >>> candidate = ResolverCandidate(
        ...     resolver="obo",
        ...     plan=FetchPlan(
        ...         url="https://example.org/hp.owl",
        ...         headers={},
        ...         filename_hint=None,
        ...         version="2024-01-01",
        ...         license="CC-BY",
        ...         media_type="application/rdf+xml",
        ...         service="obo",
        ...     ),
        ... )
        >>> candidate.resolver
        'obo'
    """

    resolver: str
    plan: FetchPlan


@dataclass(slots=True)
class PlannedFetch:
    """Plan describing how an ontology would be fetched without side effects.

    Attributes:
        spec: Original fetch specification provided by the caller.
        resolver: Name of the resolver selected to satisfy the plan.
        plan: Concrete :class:`FetchPlan` generated by the resolver.
        candidates: Ordered list of resolver candidates available for fallback.

    Examples:
        >>> fetch_plan = PlannedFetch(
        ...     spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=("owl",)),
        ...     resolver="obo",
        ...     plan=FetchPlan(
        ...         url="https://example.org/hp.owl",
        ...         headers={},
        ...         filename_hint="hp.owl",
        ...         version="2024-01-01",
        ...         license="CC-BY-4.0",
        ...         media_type="application/rdf+xml",
        ...     ),
        ...     candidates=(
        ...         ResolverCandidate(
        ...             resolver="obo",
        ...             plan=FetchPlan(
        ...                 url="https://example.org/hp.owl",
        ...                 headers={},
        ...                 filename_hint="hp.owl",
        ...                 version="2024-01-01",
        ...                 license="CC-BY-4.0",
        ...                 media_type="application/rdf+xml",
        ...             ),
        ...         ),
        ...     ),
        ... )
        >>> fetch_plan.resolver
        'obo'
    """

    spec: FetchSpec
    resolver: str
    plan: FetchPlan
    candidates: Sequence[ResolverCandidate]
    metadata: Dict[str, object] = field(default_factory=dict)
    last_modified: Optional[str] = None
    last_modified_at: Optional[datetime] = None
    size: Optional[int] = None


def parse_http_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse HTTP ``Last-Modified`` style timestamps into UTC datetimes.

    Args:
        value: Timestamp string from HTTP headers such as ``Last-Modified``.

    Returns:
        Optional[datetime]: Normalized UTC datetime when parsing succeeds.

    Raises:
        None: Parsing failures are converted into a ``None`` return value.
    """

    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamps into timezone-aware UTC datetimes.

    Args:
        value: ISO-8601 formatted timestamp string.

    Returns:
        Optional[datetime]: Normalized UTC datetime when parsing succeeds.

    Raises:
        None: Invalid values return ``None`` instead of raising.
    """

    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_version_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse version strings or manifest timestamps into UTC datetimes.

    Args:
        value: Version identifier or timestamp string to normalize.

    Returns:
        Optional[datetime]: Parsed UTC datetime when the input matches supported formats.

    Raises:
        None: All parsing failures result in ``None``.
    """

    if not value or not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    parsed = parse_iso_datetime(text)
    if parsed is not None:
        return parsed

    candidates: List[str] = []

    def _add_candidate(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    _add_candidate(text)
    _add_candidate(text.replace("_", "-"))
    _add_candidate(text.replace("/", "-"))
    _add_candidate(text.replace("_", ""))
    _add_candidate(text.replace("-", ""))

    patterns = (
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%d-%H-%M-%S",
    )

    for candidate in candidates:
        parsed = parse_iso_datetime(candidate)
        if parsed is not None:
            return parsed
        for fmt in patterns:
            try:
                naive = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            return naive.replace(tzinfo=timezone.utc)
    return None


def infer_version_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Infer a timestamp from resolver version identifiers.

    Args:
        value: Resolver version string containing date-like fragments.

    Returns:
        Optional[datetime]: Parsed UTC datetime when the value contains recoverable dates.

    Raises:
        None: Returns ``None`` instead of raising on unparseable inputs.
    """

    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    parsed = parse_version_timestamp(text)
    if parsed is not None:
        return parsed

    # Attempt to recover from composite strings like "2024-01-01-release" by
    # extracting contiguous digit blocks that resemble dates.
    matches = re.findall(r"(\d{4}[\d-]{4,})", text)
    for match in matches:
        parsed = parse_version_timestamp(match)
        if parsed is not None:
            return parsed

    digits_only = re.sub(r"\D", "", text)
    if len(digits_only) >= 8:
        parsed = parse_version_timestamp(digits_only[:14])
        if parsed is not None:
            return parsed
    parsed = parse_iso_datetime(value)
    if parsed is not None:
        return parsed
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            naive = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=timezone.utc)
    return None


def _coerce_datetime(value: Optional[str]) -> Optional[datetime]:
    """Return timezone-aware datetime parsed from HTTP or ISO timestamp."""

    parsed = parse_http_datetime(value)
    if parsed is not None:
        return parsed
    return parse_iso_datetime(value)


def _normalize_timestamp(value: Optional[str]) -> Optional[str]:
    """Return canonical ISO8601 string for HTTP timestamp headers."""

    parsed = _coerce_datetime(value)
    if parsed is None:
        return value
    return parsed.isoformat().replace("+00:00", "Z")


def _populate_plan_metadata(
    planned: PlannedFetch,
    config: ResolvedConfig,
    adapter: logging.LoggerAdapter,
) -> PlannedFetch:
    """Augment planned fetch with HTTP metadata when available."""

    if not isinstance(planned.metadata, dict):
        planned.metadata = dict(planned.metadata)
    metadata = planned.metadata

    if planned.plan.content_length is not None and planned.size is None:
        planned.size = planned.plan.content_length
    if planned.plan.content_length is not None:
        metadata.setdefault("content_length", planned.plan.content_length)
    if planned.plan.last_modified and not planned.last_modified:
        normalized = _normalize_timestamp(planned.plan.last_modified)
        planned.last_modified = normalized
        planned.last_modified_at = _coerce_datetime(normalized)
        planned.plan.last_modified = normalized
    elif planned.last_modified:
        normalized = _normalize_timestamp(planned.last_modified)
        planned.last_modified = normalized
        planned.last_modified_at = _coerce_datetime(normalized)
        if normalized:
            planned.plan.last_modified = normalized
            metadata.setdefault("last_modified", normalized)

    needs_size = planned.size is None
    needs_last_modified = planned.last_modified is None
    if not (needs_size or needs_last_modified):
        return planned

    try:
        validate_url_security(planned.plan.url, config.defaults.http)
    except ConfigError as exc:
        adapter.error(
            "metadata probe blocked by URL policy",
            extra={
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": planned.plan.url,
                "error": str(exc),
            },
        )
        raise

    timeout = getattr(config.defaults.http, "timeout_sec", 30)
    headers = dict(planned.plan.headers or {})

    try:
        head_response = requests.head(
            planned.plan.url,
            headers=headers,
            allow_redirects=True,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        adapter.warning(
            "metadata probe failed",
            extra={
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": planned.plan.url,
                "error": str(exc),
            },
        )
        return planned

    headers_map = head_response.headers
    status = head_response.status_code
    ok = head_response.ok
    head_response.close()

    if status == 405:
        try:
            get_response = requests.get(
                planned.plan.url,
                headers=headers,
                allow_redirects=True,
                timeout=timeout,
                stream=True,
            )
        except requests.RequestException as exc:
            adapter.warning(
                "metadata probe failed",
                extra={
                    "stage": "plan",
                    "ontology_id": planned.spec.id,
                    "url": planned.plan.url,
                    "error": str(exc),
                },
            )
            return planned
        headers_map = get_response.headers
        ok = get_response.ok
        status = get_response.status_code
        get_response.close()

    if not ok:
        adapter.warning(
            "metadata probe rejected",
            extra={
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": planned.plan.url,
                "status": status,
            },
        )
        return planned

    last_modified_value = headers_map.get("Last-Modified") or headers_map.get("last-modified")
    if last_modified_value:
        normalized = _normalize_timestamp(last_modified_value)
        planned.last_modified = normalized or last_modified_value
        planned.last_modified_at = _coerce_datetime(normalized or last_modified_value)
        planned.plan.last_modified = normalized or last_modified_value
        metadata["last_modified"] = planned.plan.last_modified

    if planned.size is None:
        content_length_value = headers_map.get("Content-Length") or headers_map.get(
            "content-length"
        )
        if content_length_value:
            try:
                parsed_length = int(content_length_value)
            except ValueError:
                parsed_length = None
            if parsed_length is not None:
                planned.size = parsed_length
                planned.plan.content_length = parsed_length
                metadata["content_length"] = parsed_length

    etag = headers_map.get("ETag") or headers_map.get("etag")
    if etag:
        metadata["etag"] = etag

    return planned


def _migrate_manifest_inplace(payload: dict) -> None:
    """Upgrade manifests created with older schema versions in place."""

    version = str(payload.get("schema_version", "") or "")
    if version in {"", "1.0"}:
        payload.setdefault("schema_version", "1.0")
        return
    if version == "0.9":
        payload["schema_version"] = "1.0"
        payload.setdefault("resolver_attempts", [])
        return
    logging.getLogger(__name__).warning(
        "unknown manifest schema version",
        extra={"stage": "manifest", "schema_version": version},
    )


def _read_manifest(manifest_path: Path) -> Optional[dict]:
    """Return previously recorded manifest data if a valid JSON file exists.

    Args:
        manifest_path: Filesystem path where the manifest is stored.

    Returns:
        Parsed manifest dictionary when available and valid, otherwise ``None``.
    """
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    _migrate_manifest_inplace(payload)
    validate_manifest_dict(payload, source=manifest_path)
    return payload


def _validate_manifest(manifest: Manifest) -> None:
    """Check that a manifest instance satisfies structural and type requirements.

    Args:
        manifest: Manifest produced after a download completes.

    Raises:
        ConfigurationError: If required fields are missing or contain invalid types.
    """
    validate_manifest_dict(manifest.to_dict())

    required_fields = [
        "id",
        "resolver",
        "url",
        "filename",
        "status",
        "sha256",
        "downloaded_at",
    ]
    for field_name in required_fields:
        value = getattr(manifest, field_name)
        if value in {None, ""}:
            raise ConfigurationError(f"Manifest field '{field_name}' must be populated")
    if not manifest.url.startswith(("https://", "http://")):
        raise ConfigurationError("Manifest URL must use http or https scheme")
    if not isinstance(manifest.schema_version, str):
        raise ConfigurationError("Manifest schema_version must be a string")
    if not isinstance(manifest.validation, dict):
        raise ConfigurationError("Manifest validation payload must be a dictionary")
    if not isinstance(manifest.artifacts, Sequence):
        raise ConfigurationError("Manifest artifacts must be a sequence of paths")
    for item in manifest.artifacts:
        if not isinstance(item, str):
            raise ConfigurationError("Manifest artifacts must contain only string paths")
    if not isinstance(manifest.resolver_attempts, Sequence):
        raise ConfigurationError("Manifest resolver_attempts must be a sequence")
    for entry in manifest.resolver_attempts:
        if not isinstance(entry, dict):
            raise ConfigurationError("Manifest resolver_attempts must contain dictionaries")
    if manifest.normalized_sha256 is not None and not isinstance(manifest.normalized_sha256, str):
        raise ConfigurationError("Manifest normalized_sha256 must be a string when provided")
    if manifest.fingerprint is not None and not isinstance(manifest.fingerprint, str):
        raise ConfigurationError("Manifest fingerprint must be a string when provided")


def _parse_last_modified(value: Optional[str]) -> Optional[datetime]:
    """Return a timezone-aware datetime parsed from HTTP date headers."""

    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fetch_last_modified(
    plan: FetchPlan, config: ResolvedConfig, logger: logging.Logger
) -> Optional[str]:
    """Probe the upstream plan URL for a Last-Modified header."""

    timeout = max(1, getattr(config.defaults.http, "timeout_sec", 30) or 30)
    headers = dict(plan.headers or {})
    try:
        response = requests.head(
            plan.url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        if response.status_code == 405:
            response.close()
            response = requests.get(
                plan.url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )
        header = response.headers.get("Last-Modified")
        response.close()
        return header
    except requests.RequestException as exc:  # pragma: no cover - depends on network
        logger.warning(
            "last-modified probe failed",
            extra={"stage": "plan", "resolver": plan.service or plan.url, "error": str(exc)},
        )
        return None


def _write_manifest(manifest_path: Path, manifest: Manifest) -> None:
    """Persist a validated manifest to disk as JSON.

    Args:
        manifest_path: Destination path for the manifest file.
        manifest: Manifest describing the downloaded ontology artifact.
    """
    _validate_manifest(manifest)
    manifest_path.write_text(manifest.to_json())


def _build_destination(
    spec: FetchSpec, plan: FetchPlan, config: ResolvedConfig
) -> Tuple[Path, str, Path]:
    """Determine the output directory and filename for a download.

    Args:
        spec: Fetch specification identifying the ontology.
        plan: Resolver plan containing URL metadata and optional hints.
        config: Resolved configuration with storage layout parameters.

    Returns:
        Tuple containing the target file path, resolved version, and base directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = plan.version or timestamp
    base_dir = ONTOLOGY_DIR / sanitize_filename(spec.id) / sanitize_filename(version)
    for subdir in ("original", "normalized", "validation"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    parsed = urlparse(plan.url)
    candidate = Path(parsed.path).name if parsed.path else f"{spec.id}.owl"
    filename = plan.filename_hint or sanitize_filename(candidate)
    destination = base_dir / "original" / filename
    return destination, version, base_dir


def _ensure_license_allowed(plan: FetchPlan, config: ResolvedConfig, spec: FetchSpec) -> None:
    """Confirm the ontology license is present in the configured allow list.

    Args:
        plan: Resolver plan returned for the ontology.
        config: Resolved configuration containing accepted licenses.
        spec: Fetch specification for contextual error reporting.

    Raises:
        ConfigurationError: If the plan's license is not permitted.
    """
    allowed = {
        normalize_license_to_spdx(entry) or entry for entry in config.defaults.accept_licenses
    }
    plan_license = normalize_license_to_spdx(plan.license)
    if not allowed or plan_license is None:
        return
    if plan_license not in allowed:
        raise ConfigurationError(
            f"License '{plan.license}' for ontology '{spec.id}' is not in the allowlist: {sorted(allowed)}"
        )


def _resolver_candidates(spec: FetchSpec, config: ResolvedConfig) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def _add(name: Optional[str]) -> None:
        if not name or name in seen:
            return
        candidates.append(name)
        seen.add(name)

    _add(spec.resolver)
    if config.defaults.resolver_fallback_enabled:
        for name in config.defaults.prefer_source:
            _add(name)
    return candidates


def _resolve_plan_with_fallback(
    spec: FetchSpec, config: ResolvedConfig, adapter: logging.LoggerAdapter
) -> Tuple[ResolverCandidate, Sequence[ResolverCandidate]]:
    attempts: List[str] = []
    candidates: List[ResolverCandidate] = []
    primary: Optional[ResolverCandidate] = None
    for attempt_number, resolver_name in enumerate(_resolver_candidates(spec, config), start=1):
        resolver = RESOLVERS.get(resolver_name)
        if resolver is None:
            message = "resolver not registered"
            attempts.append(f"{resolver_name}: {message}")
            adapter.warning(
                "resolver missing",
                extra={
                    "stage": "plan",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                    "error": message,
                },
            )
            continue
        adapter.info(
            "resolver attempt",
            extra={
                "stage": "plan",
                "resolver": resolver_name,
                "attempt": attempt_number,
            },
        )
        try:
            plan = resolver.plan(spec, config, adapter)
        except ConfigError as exc:
            message = str(exc)
            attempts.append(f"{resolver_name}: {message}")
            adapter.warning(
                "resolver failed",
                extra={
                    "stage": "plan",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                    "error": message,
                },
            )
            continue
        except Exception as exc:  # pylint: disable=broad-except
            message = str(exc)
            attempts.append(f"{resolver_name}: {message}")
            adapter.warning(
                "resolver failed",
                extra={
                    "stage": "plan",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                    "error": message,
                },
            )
            continue

        candidate = ResolverCandidate(resolver=resolver_name, plan=plan)
        candidates.append(candidate)
        if primary is None:
            primary = candidate
            if resolver_name != spec.resolver:
                adapter.info(
                    "resolver fallback success",
                    extra={
                        "stage": "plan",
                        "resolver": resolver_name,
                        "attempt": attempt_number,
                    },
                )
        else:
            adapter.info(
                "resolver fallback candidate",
                extra={
                    "stage": "plan",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                },
            )
    if primary is None:
        details = "; ".join(attempts) if attempts else "no resolvers attempted"
        raise ResolverError(f"All resolvers exhausted for ontology '{spec.id}': {details}")
    return primary, candidates


def fetch_one(
    spec: FetchSpec,
    *,
    config: Optional[ResolvedConfig] = None,
    correlation_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> FetchResult:
    """Fetch, validate, and persist a single ontology described by *spec*.

    Args:
        spec: Ontology fetch specification describing sources and formats.
        config: Optional resolved configuration overriding global defaults.
        correlation_id: Correlation identifier for structured logging.
        logger: Optional logger to reuse instead of configuring a new one.
        force: When ``True``, bypass local cache checks and redownload artifacts.

    Returns:
        FetchResult: Structured result containing manifest metadata and resolver attempts.

    Raises:
        ResolverError: If all resolver candidates fail to retrieve the ontology.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    logging_config = active_config.defaults.logging
    log = logger or setup_logging(
        level=logging_config.level,
        retention_days=logging_config.retention_days,
        max_log_size_mb=logging_config.max_log_size_mb,
    )
    correlation = correlation_id or generate_correlation_id()
    adapter = logging.LoggerAdapter(
        log, extra={"correlation_id": correlation, "ontology_id": spec.id}
    )
    adapter.info("planning fetch", extra={"stage": "plan"})

    primary, candidates = _resolve_plan_with_fallback(spec, active_config, adapter)
    download_config = active_config.defaults.http
    candidate_list = list(candidates) or [primary]

    resolver_attempts: List[Dict[str, object]] = []
    last_error: Optional[Exception] = None

    for attempt_number, candidate in enumerate(candidate_list, start=1):
        attempt_record: Dict[str, object] = {
            "resolver": candidate.resolver,
            "url": candidate.plan.url,
            "attempt": attempt_number,
        }

        pending_spec = FetchSpec(
            id=spec.id,
            resolver=candidate.resolver,
            extras=spec.extras,
            target_formats=spec.target_formats,
        )
        try:
            _ensure_license_allowed(candidate.plan, active_config, pending_spec)
        except ConfigurationError as exc:
            adapter.warning(
                "resolver license rejected",
                extra={
                    "stage": "plan",
                    "resolver": candidate.resolver,
                    "attempt": attempt_number,
                    "error": str(exc),
                },
            )
            attempt_record.update({"status": "rejected", "error": str(exc)})
            resolver_attempts.append(dict(attempt_record))
            last_error = exc
            continue

        if candidate.plan.service:
            adapter.extra["service"] = candidate.plan.service
        else:
            adapter.extra.pop("service", None)

        def _execute_candidate() -> FetchResult:
            pending_destination, pending_version, pending_base_dir = _build_destination(
                pending_spec, candidate.plan, active_config
            )
            pending_manifest_path = pending_base_dir / "manifest.json"

            with _version_lock(pending_spec.id, pending_version):
                previous_manifest = None
                if not force:
                    STORAGE.ensure_local_version(pending_spec.id, pending_version)
                    previous_manifest = _read_manifest(pending_manifest_path)

                adapter.info(
                    "downloading",
                    extra={
                        "stage": "download",
                        "url": candidate.plan.url,
                        "destination": str(pending_destination),
                        "version": pending_version,
                        "resolver": candidate.resolver,
                        "attempt": attempt_number,
                    },
                )

                pending_secure_url = validate_url_security(candidate.plan.url, download_config)
                result = download_stream(
                    url=pending_secure_url,
                    destination=pending_destination,
                    headers=candidate.plan.headers,
                    previous_manifest=previous_manifest,
                    http_config=download_config,
                    cache_dir=CACHE_DIR,
                    logger=adapter,
                    expected_media_type=candidate.plan.media_type,
                    service=candidate.plan.service,
                )

                effective_spec = pending_spec
                destination = pending_destination
                version = pending_version
                base_dir = pending_base_dir
                manifest_path = pending_manifest_path
                secure_url = pending_secure_url
                plan = candidate.plan

                normalized_dir = base_dir / "normalized"
                validation_dir = base_dir / "validation"
                validation_requests: List[ValidationRequest] = [
                    ValidationRequest(
                        name="rdflib",
                        file_path=destination,
                        normalized_dir=normalized_dir,
                        validation_dir=validation_dir,
                        config=active_config,
                    ),
                    ValidationRequest(
                        name="pronto",
                        file_path=destination,
                        normalized_dir=normalized_dir,
                        validation_dir=validation_dir,
                        config=active_config,
                    ),
                    ValidationRequest(
                        name="owlready2",
                        file_path=destination,
                        normalized_dir=normalized_dir,
                        validation_dir=validation_dir,
                        config=active_config,
                    ),
                    ValidationRequest(
                        name="robot",
                        file_path=destination,
                        normalized_dir=normalized_dir,
                        validation_dir=validation_dir,
                        config=active_config,
                    ),
                    ValidationRequest(
                        name="arelle",
                        file_path=destination,
                        normalized_dir=normalized_dir,
                        validation_dir=validation_dir,
                        config=active_config,
                    ),
                ]

                media_type = (plan.media_type or "").strip().lower()
                if media_type and media_type not in RDF_MIME_ALIASES:
                    validation_requests = [
                        request
                        for request in validation_requests
                        if request.name not in {"rdflib", "robot"}
                    ]
                    adapter.info(
                        "skipping rdf validators",
                        extra={
                            "stage": "validate",
                            "media_type": media_type,
                            "validator": "rdf",
                        },
                    )

                artifacts = [str(destination)]
                if plan.media_type == "application/zip" or destination.suffix.lower() == ".zip":
                    extraction_dir = destination.parent / f"{destination.stem}_extracted"
                    try:
                        extracted_paths = extract_archive_safe(
                            destination, extraction_dir, logger=adapter
                        )
                        artifacts.extend(str(path) for path in extracted_paths)
                    except ConfigError as exc:
                        adapter.error(
                            "zip extraction failed",
                            extra={"stage": "extract", "error": str(exc)},
                        )
                        if not active_config.defaults.continue_on_error:
                            raise OntologyDownloadError(
                                f"Extraction failed for '{effective_spec.id}': {exc}"
                            ) from exc

                validation_results = run_validators(validation_requests, adapter)

                normalized_hash = None
                normalization_mode = "none"
                rdflib_result = validation_results.get("rdflib")
                if rdflib_result and isinstance(rdflib_result.details, dict):
                    maybe_hash = rdflib_result.details.get("normalized_sha256")
                    if isinstance(maybe_hash, str):
                        normalized_hash = maybe_hash
                    maybe_mode = rdflib_result.details.get("normalization_mode")
                    if isinstance(maybe_mode, str):
                        normalization_mode = maybe_mode

                target_formats_sorted = ",".join(sorted(effective_spec.target_formats))

                fingerprint_components = [
                    MANIFEST_SCHEMA_VERSION,
                    effective_spec.id,
                    effective_spec.resolver,
                    version,
                    result.sha256,
                    normalized_hash or "",
                    secure_url,
                    target_formats_sorted,
                    normalization_mode,
                ]
                fingerprint = hashlib.sha256(
                    "|".join(fingerprint_components).encode("utf-8")
                ).hexdigest()

                attempt_record["status"] = "success"
                resolver_attempts.append(dict(attempt_record))

                manifest = Manifest(
                    schema_version=MANIFEST_SCHEMA_VERSION,
                    id=effective_spec.id,
                    resolver=effective_spec.resolver,
                    url=secure_url,
                    filename=destination.name,
                    version=version,
                    license=plan.license,
                    status=result.status,
                    sha256=result.sha256,
                    normalized_sha256=normalized_hash,
                    fingerprint=fingerprint,
                    etag=result.etag,
                    last_modified=result.last_modified,
                    downloaded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    target_formats=effective_spec.target_formats,
                    validation=validation_results,
                    artifacts=artifacts,
                    resolver_attempts=resolver_attempts,
                )
                _write_manifest(manifest_path, manifest)
                STORAGE.finalize_version(effective_spec.id, version, base_dir)

                adapter.info(
                    "fetch complete",
                    extra={
                        "stage": "complete",
                        "status": result.status,
                        "sha256": result.sha256,
                        "manifest": str(manifest_path),
                    },
                )

                return FetchResult(
                    spec=effective_spec,
                    local_path=destination,
                    status=result.status,
                    sha256=result.sha256,
                    manifest_path=manifest_path,
                    artifacts=artifacts,
                )

        try:
            return _execute_candidate()
        except ConfigError as exc:
            attempt_record.update({"status": "failed", "error": str(exc)})
            resolver_attempts.append(dict(attempt_record))
            adapter.warning(
                "download attempt failed",
                extra={
                    "stage": "download",
                    "resolver": candidate.resolver,
                    "attempt": attempt_number,
                    "error": str(exc),
                },
            )
            last_error = exc
            retryable = getattr(exc, "retryable", False)
            if retryable:
                adapter.info(
                    "trying fallback resolver",
                    extra={
                        "stage": "download",
                        "resolver": candidate.resolver,
                        "attempt": attempt_number,
                    },
                )
                continue
            raise OntologyDownloadError(f"Download failed for '{pending_spec.id}': {exc}") from exc
        except Exception as exc:
            last_error = exc
            attempt_record.update({"status": "error", "error": str(exc)})
            resolver_attempts.append(dict(attempt_record))
            raise

    if last_error is None:
        raise OntologyDownloadError(f"All resolver candidates failed for '{spec.id}'")
    if isinstance(last_error, ConfigurationError):
        raise last_error
    raise OntologyDownloadError(f"Download failed for '{spec.id}': {last_error}") from last_error


def plan_one(
    spec: FetchSpec,
    *,
    config: Optional[ResolvedConfig] = None,
    correlation_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> PlannedFetch:
    """Return a resolver plan for a single ontology without performing downloads.

    Args:
        spec: Fetch specification describing the ontology to plan.
        config: Optional resolved configuration providing defaults and limits.
        correlation_id: Correlation identifier reused for logging context.
        logger: Logger instance used to emit resolver telemetry.

    Returns:
        PlannedFetch containing the normalized spec, resolver name, and plan.

    Raises:
        ResolverError: If all resolvers fail to produce a plan for ``spec``.
        ConfigurationError: If licence checks reject the planned ontology.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    logging_config = active_config.defaults.logging
    log = logger or setup_logging(
        level=logging_config.level,
        retention_days=logging_config.retention_days,
        max_log_size_mb=logging_config.max_log_size_mb,
    )
    correlation = correlation_id or generate_correlation_id()
    adapter = logging.LoggerAdapter(
        log, extra={"correlation_id": correlation, "ontology_id": spec.id}
    )
    adapter.info("planning fetch", extra={"stage": "plan"})

    primary, candidates = _resolve_plan_with_fallback(spec, active_config, adapter)
    effective_spec = FetchSpec(
        id=spec.id,
        resolver=primary.resolver,
        extras=spec.extras,
        target_formats=spec.target_formats,
    )
    _ensure_license_allowed(primary.plan, active_config, effective_spec)
    planned = PlannedFetch(
        spec=effective_spec,
        resolver=primary.resolver,
        plan=primary.plan,
        candidates=tuple(candidates),
        last_modified=primary.plan.last_modified,
        size=primary.plan.content_length,
    )
    return _populate_plan_metadata(planned, active_config, adapter)


def plan_all(
    specs: Iterable[FetchSpec],
    *,
    config: Optional[ResolvedConfig] = None,
    logger: Optional[logging.Logger] = None,
    since: Optional[datetime] = None,
) -> List[PlannedFetch]:
    """Return resolver plans for a collection of ontologies.

    Args:
        specs: Iterable of fetch specifications to resolve.
        config: Optional resolved configuration reused across plans.
        logger: Logger instance used for annotation-aware logging.
        since: Optional cutoff date; plans older than this timestamp are filtered out.

    Returns:
        List of PlannedFetch entries describing each ontology plan.

    Raises:
        ResolverError: Propagated when fallback planning fails for any spec.
        ConfigurationError: When licence enforcement rejects a planned ontology.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    logging_config = active_config.defaults.logging
    log = logger or setup_logging(
        level=logging_config.level,
        retention_days=logging_config.retention_days,
        max_log_size_mb=logging_config.max_log_size_mb,
    )
    correlation = generate_correlation_id()
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation})

    spec_list = list(specs)
    if not spec_list:
        return []

    max_workers = max(1, active_config.defaults.http.concurrent_plans)
    adapter.info(
        "planning batch",
        extra={
            "stage": "plan",
            "progress": {"total": len(spec_list)},
            "workers": max_workers,
        },
    )

    results: Dict[int, PlannedFetch] = {}
    futures: Dict[object, tuple[int, FetchSpec]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, spec in enumerate(spec_list):
            future = executor.submit(
                plan_one,
                spec,
                config=active_config,
                correlation_id=correlation,
                logger=log,
            )
            futures[future] = (index, spec)

        for future in as_completed(futures):
            index, spec = futures[future]
            try:
                planned = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                adapter.error(
                    "planning failed",
                    extra={
                        "stage": "plan",
                        "ontology_id": spec.id,
                        "error": str(exc),
                    },
                )
                if isinstance(exc, (ConfigError, ConfigurationError)):
                    for pending in futures:
                        pending.cancel()
                    raise
                if not active_config.defaults.continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    raise
            else:
                results[index] = planned

    ordered_indices = sorted(results)
    ordered_plans = [results[i] for i in ordered_indices]

    if since is None:
        return ordered_plans

    filtered: List[PlannedFetch] = []
    for plan in ordered_plans:
        last_modified = plan.last_modified_at or _coerce_datetime(plan.last_modified)
        if last_modified is None:
            header = _fetch_last_modified(plan.plan, active_config, log)
            if header:
                plan.plan.last_modified = header
                plan.last_modified = header
                plan.last_modified_at = _coerce_datetime(header)
                last_modified = plan.last_modified_at
        if last_modified and last_modified < since:
            adapter.info(
                "plan filtered by since",
                extra={
                    "stage": "plan",
                    "ontology_id": plan.spec.id,
                    "last_modified": plan.last_modified,
                    "since": since.isoformat().replace("+00:00", "Z"),
                },
            )
            continue
        filtered.append(plan)
    return filtered


def fetch_all(
    specs: Iterable[FetchSpec],
    *,
    config: Optional[ResolvedConfig] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> List[FetchResult]:
    """Fetch a sequence of ontologies sequentially.

    Args:
        specs: Iterable of fetch specifications to process.
        config: Optional resolved configuration shared across downloads.
        logger: Logger used to emit progress and error events.
        force: When True, skip manifest reuse and download everything again.

    Returns:
        List of FetchResult entries corresponding to completed downloads.

    Raises:
        OntologyDownloadError: Propagated when downloads fail and the pipeline
            is configured to stop on error.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    if logger is not None:
        log = logger
    else:
        candidate = logging.getLogger("DocsToKG.OntologyDownload")
        if candidate.handlers:
            log = candidate
        else:
            logging_config = active_config.defaults.logging
            log = setup_logging(
                level=logging_config.level,
                retention_days=logging_config.retention_days,
                max_log_size_mb=logging_config.max_log_size_mb,
            )
    correlation = generate_correlation_id()
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation})

    spec_list = list(specs)
    total = len(spec_list)
    if not spec_list:
        return []

    max_workers = max(1, active_config.defaults.http.concurrent_downloads)
    adapter.info(
        "starting batch",
        extra={"stage": "batch", "progress": {"total": total}, "workers": max_workers},
    )

    results_map: Dict[int, FetchResult] = {}
    futures: Dict[object, tuple[int, FetchSpec]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, spec in enumerate(spec_list, start=1):
            adapter.info(
                "starting ontology fetch",
                extra={
                    "stage": "start",
                    "ontology_id": spec.id,
                    "progress": {"current": index, "total": total},
                },
            )
            future = executor.submit(
                fetch_one,
                spec,
                config=active_config,
                correlation_id=correlation,
                logger=log,
                force=force,
            )
            futures[future] = (index, spec)

        for future in as_completed(futures):
            index, spec = futures[future]
            try:
                result = future.result()
                results_map[index] = result
                adapter.info(
                    "progress update",
                    extra={
                        "stage": "progress",
                        "ontology_id": spec.id,
                        "progress": {"current": len(results_map), "total": total},
                    },
                )
            except Exception as exc:  # pylint: disable=broad-except
                adapter.error(
                    "ontology fetch failed",
                    extra={"stage": "error", "ontology_id": spec.id, "error": str(exc)},
                )
                if not active_config.defaults.continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    raise

    ordered_results = [results_map[i] for i in sorted(results_map)]
    return ordered_results


# --- Globals ---

__all__ = [
    # Foundation
    "retry_with_backoff",
    "sanitize_filename",
    "generate_correlation_id",
    "mask_sensitive_data",
    # Configuration
    "ConfigError",
    "ConfigurationError",
    "DefaultsConfig",
    "DownloadConfiguration",
    "LoggingConfiguration",
    "ValidationConfig",
    "ResolvedConfig",
    "build_resolved_config",
    "ensure_python_version",
    "setup_logging",
    "parse_rate_limit_to_rps",
    "load_config",
    "load_raw_yaml",
    "validate_config",
    "get_env_overrides",
    # Storage
    "CACHE_DIR",
    "CONFIG_DIR",
    "LOG_DIR",
    "LOCAL_ONTOLOGY_DIR",
    "StorageBackend",
    "LocalStorageBackend",
    "FsspecStorageBackend",
    "get_storage_backend",
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
    "STORAGE",
    # Network
    "TokenBucket",
    "DownloadFailure",
    "DownloadResult",
    "RDF_MIME_ALIASES",
    "RDF_MIME_FORMAT_LABELS",
    "download_stream",
    "extract_archive_safe",
    "validate_url_security",
    "sha256_file",
    # Validation
    "ValidationRequest",
    "ValidationResult",
    "run_validators",
    # Pipeline
    "merge_defaults",
    "FetchSpec",
    "FetchResult",
    "Manifest",
    "ResolverCandidate",
    "PlannedFetch",
    "OntologyDownloadError",
    "ResolverError",
    "ValidationError",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_JSON_SCHEMA",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
    "get_manifest_schema",
    "validate_manifest_dict",
]


def _safe_lock_component(value: str) -> str:
    """Return a filesystem-safe token for lock filenames."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    sanitized = sanitized.strip("._") or "lock"
    return sanitized


@contextmanager
def _version_lock(ontology_id: str, version: str) -> Iterator[None]:
    """Acquire an inter-process lock for a specific ontology version."""

    lock_dir = CACHE_DIR / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = (
        lock_dir / f"{_safe_lock_component(ontology_id)}__{_safe_lock_component(version)}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()

        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - fallback when no locking backend available
            yield
            return

        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
