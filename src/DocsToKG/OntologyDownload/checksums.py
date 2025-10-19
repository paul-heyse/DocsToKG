"""Checksum parsing, normalisation, and verification helpers.

Configuration files and resolver payloads may describe expected digests in a
variety of formats.  This module normalises those declarations, validates the
algorithms that callers request, and exposes streaming utilities that compute
hashes without materialising entire ontology archives in memory.  The helpers
are used by the planner when constructing manifests and by the downloader to
enforce tamper-detection policies prior to ingestion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Type
from urllib.parse import urlparse

import requests

from .errors import ConfigError, OntologyDownloadError
from .io import (
    SESSION_POOL,
    get_bucket,
    is_retryable_error,
    retry_with_backoff,
    validate_url_security,
)
from .settings import DownloadConfiguration

ErrorType = Type[Exception]

_DIGEST_PATTERN = re.compile(r"(?i)\b([0-9a-f]{32,128})\b")
_CHECKSUM_STREAM_CHUNK_SIZE = 8192
_CHECKSUM_STREAM_TAIL_BYTES = 128


@dataclass(slots=True, frozen=True)
class ExpectedChecksum:
    """Expected checksum derived from configuration or resolver metadata."""

    algorithm: str
    value: str

    def to_known_hash(self) -> str:
        """Return ``algorithm:value`` string suitable for pooch known_hash."""

        return f"{self.algorithm}:{self.value}"

    def to_mapping(self) -> dict:
        """Return mapping representation for manifest and index serialization."""

        return {"algorithm": self.algorithm, "value": self.value}


def _normalize_algorithm(algorithm: Optional[str], *, context: str, error_cls: ErrorType) -> str:
    candidate = (algorithm or "sha256").strip().lower()
    if candidate not in {"md5", "sha1", "sha256", "sha512"}:
        raise error_cls(f"{context}: unsupported checksum algorithm '{candidate}'")
    return candidate


def _normalize_checksum(
    algorithm: str,
    value: str,
    *,
    context: str,
    error_cls: ErrorType,
) -> Tuple[str, str]:
    normalized_algorithm = _normalize_algorithm(algorithm, context=context, error_cls=error_cls)
    if not isinstance(value, str):
        raise error_cls(f"{context}: checksum value must be a string")
    checksum = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32,128}", checksum):
        raise error_cls(f"{context}: checksum value must be a hexadecimal digest")
    return normalized_algorithm, checksum


def parse_checksum_extra(
    value: object,
    *,
    context: str,
    error_cls: ErrorType = ConfigError,
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize checksum extras to ``(algorithm, value)`` tuples."""

    if value is None:
        return None, None
    if isinstance(value, str):
        return _normalize_checksum("sha256", value, context=context, error_cls=error_cls)
    if isinstance(value, Mapping):
        algorithm_raw = value.get("algorithm", "sha256")
        checksum_raw = value.get("value")
        if not isinstance(algorithm_raw, str):
            raise error_cls(f"{context}: checksum algorithm must be a string")
        if not isinstance(checksum_raw, str):
            raise error_cls(f"{context}: checksum value must be a string")
        return _normalize_checksum(
            algorithm_raw,
            checksum_raw,
            context=context,
            error_cls=error_cls,
        )
    raise error_cls(f"{context}: checksum must be provided as a string or mapping")


def parse_checksum_url_extra(
    value: object,
    *,
    context: str,
    error_cls: ErrorType = ConfigError,
) -> Tuple[Optional[str], Optional[str]]:
    """Normalise checksum URL extras to ``(url, algorithm)`` tuples."""

    if value is None:
        return None, None
    if isinstance(value, str):
        url = value.strip()
        if not url:
            raise error_cls(f"{context}: checksum_url must not be empty")
        return url, None
    if isinstance(value, Mapping):
        url_value = value.get("url")
        algorithm_value = value.get("algorithm")
        if not isinstance(url_value, str) or not url_value.strip():
            raise error_cls(f"{context}: checksum_url must include a non-empty 'url'")
        algorithm = None
        if algorithm_value is not None:
            if not isinstance(algorithm_value, str):
                raise error_cls(f"{context}: checksum_url algorithm must be a string when provided")
            algorithm = _normalize_algorithm(
                algorithm_value,
                context=context,
                error_cls=error_cls,
            )
        return url_value.strip(), algorithm
    raise error_cls(f"{context}: checksum_url must be provided as a string or mapping")


def _extract_checksum_from_text(text: str, *, context: str) -> str:
    match = _DIGEST_PATTERN.search(text)
    if not match:
        raise OntologyDownloadError(f"Unable to parse checksum from {context}")
    return match.group(0).lower()


def _fetch_checksum_from_url(
    *,
    url: str,
    algorithm: str,
    http_config: DownloadConfiguration,
    logger: logging.Logger,
) -> str:
    secure_url = validate_url_security(url, http_config)
    parsed = urlparse(secure_url)
    host = parsed.hostname
    bucket = get_bucket(http_config=http_config, service=None, host=host)
    polite_headers = http_config.polite_http_headers()
    max_bytes = http_config.max_checksum_bytes()

    def _fetch_once() -> str:
        bucket.consume()
        tail = b""
        total_bytes = 0
        with SESSION_POOL.lease(
            service=None,
            host=host,
            http_config=http_config,
        ) as session:
            try:
                response = session.get(
                    secure_url,
                    timeout=http_config.timeout_sec,
                    stream=True,
                    headers=polite_headers,
                )
            except requests.RequestException:
                raise
            with response:
                response.raise_for_status()
                for chunk in response.iter_content(_CHECKSUM_STREAM_CHUNK_SIZE):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        logger.error(
                            "checksum response exceeded limit",
                            extra={
                                "stage": "download",
                                "checksum_url": secure_url,
                                "algorithm": algorithm,
                                "received_bytes": total_bytes,
                                "limit_bytes": max_bytes,
                            },
                        )
                        raise OntologyDownloadError(
                            f"Checksum response from {secure_url} exceeded {max_bytes} bytes"
                        )
                    buffer = tail + chunk
                    match = _DIGEST_PATTERN.search(buffer.decode("utf-8", errors="ignore"))
                    if match:
                        digest = match.group(1).lower()
                        logger.info(
                            "fetched checksum",
                            extra={
                                "stage": "download",
                                "checksum_url": secure_url,
                                "algorithm": algorithm,
                            },
                        )
                        return digest
                    tail = buffer[-_CHECKSUM_STREAM_TAIL_BYTES:]
        raise OntologyDownloadError(f"Unable to parse checksum from {secure_url}")

    def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
        logger.warning(
            "checksum fetch retry",
            extra={
                "stage": "download",
                "checksum_url": secure_url,
                "attempt": attempt,
                "sleep_sec": round(delay, 2),
                "error": str(exc),
            },
        )

    try:
        return retry_with_backoff(
            _fetch_once,
            retryable=is_retryable_error,
            max_attempts=max(1, http_config.max_retries),
            backoff_base=http_config.backoff_factor,
            jitter=http_config.backoff_factor,
            callback=_on_retry,
        )
    except OntologyDownloadError:
        raise
    except requests.RequestException as exc:  # pragma: no cover - exercised via retry logic
        raise OntologyDownloadError(f"Failed to fetch checksum from {secure_url}: {exc}") from exc


def resolve_expected_checksum(
    *,
    spec: Any,
    plan: Any,
    download_config: DownloadConfiguration,
    logger: logging.Logger,
    error_cls: ErrorType = ConfigError,
) -> Optional[ExpectedChecksum]:
    """Determine the expected checksum metadata for downstream enforcement."""

    context = f"ontology '{getattr(spec, 'id', 'unknown')}'"

    plan_checksum: Optional[Tuple[str, str]] = None
    if getattr(plan, "checksum", None):
        algorithm = getattr(plan, "checksum_algorithm", None) or "sha256"
        plan_checksum = _normalize_checksum(
            algorithm, plan.checksum, context=context, error_cls=error_cls
        )

    spec_checksum = (None, None)
    extras = getattr(spec, "extras", {})
    if isinstance(extras, Mapping):
        spec_checksum = parse_checksum_extra(
            extras.get("checksum"), context=context, error_cls=error_cls
        )
    if (
        spec_checksum[1] is not None
        and plan_checksum is not None
        and spec_checksum[1] != plan_checksum[1]
    ):
        raise error_cls(
            f"{context}: conflicting checksum values between resolver and specification extras"
        )

    algorithm: Optional[str] = None
    value: Optional[str] = None
    if plan_checksum is not None:
        algorithm, value = plan_checksum
    if spec_checksum[1] is not None:
        algorithm, value = spec_checksum

    checksum_url_source: Optional[Tuple[str, Optional[str]]] = None
    plan_checksum_url = getattr(plan, "checksum_url", None)
    if plan_checksum_url:
        checksum_url_source = (plan_checksum_url, getattr(plan, "checksum_algorithm", None))
    else:
        if isinstance(extras, Mapping):
            url_tuple = parse_checksum_url_extra(
                extras.get("checksum_url"),
                context=context,
                error_cls=error_cls,
            )
            if url_tuple[0]:
                checksum_url_source = url_tuple

    if value is None and checksum_url_source:
        raw_url, url_algorithm = checksum_url_source
        normalized_algorithm = _normalize_algorithm(
            url_algorithm or algorithm or "sha256",
            context=context,
            error_cls=error_cls,
        )
        value = _fetch_checksum_from_url(
            url=raw_url,
            algorithm=normalized_algorithm,
            http_config=download_config,
            logger=logger,
        )
        algorithm = normalized_algorithm

    if value is None or algorithm is None:
        return None

    normalized_algorithm, normalized_value = _normalize_checksum(
        algorithm,
        value,
        context=context,
        error_cls=error_cls,
    )
    checksum = ExpectedChecksum(
        algorithm=normalized_algorithm,
        value=normalized_value,
    )
    logger.info(
        "expected checksum resolved",
        extra={
            "stage": "download",
            "checksum_algorithm": normalized_algorithm,
        },
    )
    return checksum
