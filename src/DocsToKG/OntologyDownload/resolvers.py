# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.resolvers",
#   "purpose": "Implements DocsToKG.OntologyDownload.resolvers behaviors and helpers",
#   "sections": [
#     {"id": "normalize_license_to_spdx", "name": "normalize_license_to_spdx", "anchor": "function-normalize_license_to_spdx", "kind": "function"},
#     {"id": "_get_service_bucket", "name": "_get_service_bucket", "anchor": "function-_get_service_bucket", "kind": "function"},
#     {"id": "FetchPlan", "name": "FetchPlan", "anchor": "class-FetchPlan", "kind": "class"},
#     {"id": "BaseResolver", "name": "BaseResolver", "anchor": "class-BaseResolver", "kind": "class"},
#     {"id": "OBOResolver", "name": "OBOResolver", "anchor": "class-OBOResolver", "kind": "class"},
#     {"id": "OLSResolver", "name": "OLSResolver", "anchor": "class-OLSResolver", "kind": "class"},
#     {"id": "BioPortalResolver", "name": "BioPortalResolver", "anchor": "class-BioPortalResolver", "kind": "class"},
#     {"id": "LOVResolver", "name": "LOVResolver", "anchor": "class-LOVResolver", "kind": "class"},
#     {"id": "SKOSResolver", "name": "SKOSResolver", "anchor": "class-SKOSResolver", "kind": "class"},
#     {"id": "XBRLResolver", "name": "XBRLResolver", "anchor": "class-XBRLResolver", "kind": "class"},
#     {"id": "OntobeeResolver", "name": "OntobeeResolver", "anchor": "class-OntobeeResolver", "kind": "class"},
#     {"id": "_load_resolver_plugins", "name": "_load_resolver_plugins", "anchor": "function-_load_resolver_plugins", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""Ontology resolver implementations.

This module defines the strategies that translate planner specifications into
actionable fetch plans. Each resolver applies polite headers, unified retry
logic, SPDX-normalized licensing, and service-specific rate limits while
participating in the automatic fallback chains described in the ontology
download refactor. New resolvers can be registered through the ``RESOLVERS`` map.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Dict, Iterable, Optional

import requests

try:  # pragma: no cover - optional dependency guidance
    from bioregistry import get_obo_download, get_owl_download, get_rdf_download
except ModuleNotFoundError:  # pragma: no cover - provide actionable error for runtime use
    get_obo_download = None  # type: ignore[assignment]
    get_owl_download = None  # type: ignore[assignment]
    get_rdf_download = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency shim
    from ols_client import OlsClient as _OlsClient
except ImportError:
    try:
        from ols_client import Client as _OlsClient
    except ImportError:  # ols-client not installed
        _OlsClient = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guidance
    from ontoportal_client import BioPortalClient
except ModuleNotFoundError:  # pragma: no cover - provide actionable error later
    BioPortalClient = None  # type: ignore[assignment]

from .ontology_download import (
    ConfigError,
    ResolvedConfig,
    TokenBucket,
    get_pystow,
    retry_with_backoff,
)
# --- Globals ---

OlsClient = _OlsClient
pystow = get_pystow()


_SERVICE_BUCKETS: Dict[str, TokenBucket] = {}
_SERVICE_BUCKET_LOCK = threading.Lock()
# --- Public Functions ---


def normalize_license_to_spdx(value: Optional[str]) -> Optional[str]:
    """Normalize common license strings to canonical SPDX identifiers.

    Resolver metadata frequently reports informal variants such as ``CC BY 4.0``;
    converting to SPDX ensures allowlist comparisons remain consistent.

    Args:
        value: Raw license string returned by a resolver (may be ``None``).

    Returns:
        Canonical SPDX identifier when a mapping is known, otherwise the
        cleaned original value or ``None`` when the input is empty.
    """

    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    lower = cleaned.lower()
    replacements = {
        "cc-by": "CC-BY-4.0",
        "cc by": "CC-BY-4.0",
        "cc-by-4.0": "CC-BY-4.0",
        "cc by 4.0": "CC-BY-4.0",
        "creative commons attribution": "CC-BY-4.0",
        "creative commons attribution 4.0": "CC-BY-4.0",
        "cc0": "CC0-1.0",
        "cc-0": "CC0-1.0",
        "cc0-1.0": "CC0-1.0",
        "creative commons zero": "CC0-1.0",
        "public domain": "CC0-1.0",
        "apache": "Apache-2.0",
        "apache 2": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "apache license": "Apache-2.0",
        "apache license 2.0": "Apache-2.0",
    }
    if lower in replacements:
        return replacements[lower]
    if lower.startswith("cc-by") and "4" in lower:
        return "CC-BY-4.0"
    if lower.startswith("cc0") or "public domain" in lower or "cc-0" in lower:
        return "CC0-1.0"
    if "apache" in lower and "2" in lower:
        return "Apache-2.0"
    return cleaned
# --- Private Helpers ---


def _get_service_bucket(service: str, config: ResolvedConfig) -> TokenBucket:
    """Return a token bucket for resolver API requests respecting rate limits."""

    http_config = config.defaults.http
    rate = http_config.parse_service_rate_limit(service) or http_config.rate_limit_per_second()
    rate = max(rate, 0.1)
    capacity = max(rate, 1.0)
    with _SERVICE_BUCKET_LOCK:
        bucket = _SERVICE_BUCKETS.get(service)
        if bucket is None or bucket.rate != rate or bucket.capacity != capacity:
            bucket = TokenBucket(rate_per_sec=rate, capacity=capacity)
            _SERVICE_BUCKETS[service] = bucket
        return bucket


@dataclass(slots=True)
# --- Public Classes ---

class FetchPlan:
    """Concrete plan output from a resolver.

    Attributes:
        url: Final URL from which to download the ontology document.
        headers: HTTP headers required by the upstream service.
        filename_hint: Optional filename recommended by the resolver.
        version: Version identifier derived from resolver metadata.
        license: License reported for the ontology.
        media_type: MIME type of the artifact when known.
        service: Logical service identifier used for rate limiting.

    Examples:
        >>> plan = FetchPlan(
        ...     url="https://example.org/ontology.owl",
        ...     headers={"Accept": "application/rdf+xml"},
        ...     filename_hint="ontology.owl",
        ...     version="2024-01-01",
        ...     license="CC-BY",
        ...     media_type="application/rdf+xml",
        ...     service="ols",
        ... )
        >>> plan.service
        'ols'
    """

    url: str
    headers: Dict[str, str]
    filename_hint: Optional[str]
    version: Optional[str]
    license: Optional[str]
    media_type: Optional[str]
    service: Optional[str] = None
    last_modified: Optional[str] = None
    content_length: Optional[int] = None


class BaseResolver:
    """Shared helpers for resolver implementations.

    Provides polite header construction, retry orchestration, and metadata
    normalization utilities shared across concrete resolver classes.

    Attributes:
        None

    Examples:
        >>> class DemoResolver(BaseResolver):
        ...     def plan(self, spec, config, logger):
        ...         return self._build_plan(url="https://example.org/demo.owl")
        ...
        >>> demo = DemoResolver()
        >>> isinstance(demo._build_plan(url="https://example.org").url, str)
        True
    """

    def _execute_with_retry(
        self,
        func,
        *,
        config: ResolvedConfig,
        logger: logging.Logger,
        name: str,
        service: Optional[str] = None,
    ):
        """Run a callable with retry semantics tailored for resolver APIs.

        Args:
            func: Callable performing the API request.
            config: Resolved configuration containing retry and timeout settings.
            logger: Logger adapter used to record retry attempts.
            name: Human-friendly resolver name used in log messages.

        Returns:
            Result returned by the supplied callable.

        Raises:
            ConfigError: When retry limits are exceeded or HTTP errors occur.
        """
        max_attempts = max(1, config.defaults.http.max_retries)
        backoff_base = config.defaults.http.backoff_factor

        def _retryable(exc: Exception) -> bool:
            return isinstance(exc, (requests.Timeout, requests.ConnectionError))

        def _on_retry(attempt: int, exc: Exception, sleep_time: float) -> None:
            logger.warning(
                "resolver timeout",
                extra={
                    "stage": "plan",
                    "resolver": name,
                    "attempt": attempt,
                    "sleep_sec": sleep_time,
                },
            )

        bucket = _get_service_bucket(service, config) if service else None

        def _invoke():
            if bucket is not None:
                bucket.consume()
            return func()

        try:
            return retry_with_backoff(
                _invoke,
                retryable=_retryable,
                max_attempts=max_attempts,
                backoff_base=backoff_base,
                jitter=backoff_base,
                callback=_on_retry,
            )
        except requests.Timeout as exc:
            raise ConfigError(
                f"{name} API timeout after {config.defaults.http.timeout_sec}s"
            ) from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                raise
            raise ConfigError(f"{name} API error {status}: {exc}") from exc
        except requests.ConnectionError as exc:
            raise ConfigError(f"{name} API connection error: {exc}") from exc

    def _extract_correlation_id(self, logger: logging.Logger) -> Optional[str]:
        """Return the correlation id from a logger adapter when available.

        Args:
            logger: Logger or adapter potentially carrying an ``extra`` dictionary.

        Returns:
            Correlation identifier string when present, otherwise ``None``.
        """

        extra = getattr(logger, "extra", None)
        if isinstance(extra, dict):
            value = extra.get("correlation_id")
            if isinstance(value, str):
                return value
        return None

    def _build_polite_headers(
        self, config: ResolvedConfig, logger: logging.Logger
    ) -> Dict[str, str]:
        """Create polite headers derived from configuration and logger context.

        Args:
            config: Resolved configuration providing HTTP header defaults.
            logger: Logger adapter whose correlation id is propagated to headers.

        Returns:
            Dictionary of polite header values ready to attach to HTTP sessions.
        """

        http_config = config.defaults.http
        return http_config.polite_http_headers(correlation_id=self._extract_correlation_id(logger))

    @staticmethod
    def _apply_headers_to_session(session: Any, headers: Dict[str, str]) -> None:
        """Apply polite headers to a client session when supported.

        Args:
            session: HTTP client or session object whose ``headers`` may be updated.
            headers: Mapping of header names to polite values to merge.

        Returns:
            None
        """

        if session is None:
            return
        mapping = getattr(session, "headers", None)
        if mapping is None:
            return
        updater = getattr(mapping, "update", None)
        if callable(updater):
            updater(headers)
        elif isinstance(mapping, dict):  # pragma: no cover - defensive branch
            mapping.update(headers)

    def _build_plan(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        filename_hint: Optional[str] = None,
        version: Optional[str] = None,
        license: Optional[str] = None,
        media_type: Optional[str] = None,
        service: Optional[str] = None,
        last_modified: Optional[str] = None,
        content_length: Optional[int] = None,
    ) -> FetchPlan:
        """Construct a ``FetchPlan`` from resolver components.

        Args:
            url: Canonical download URL for the ontology.
            headers: HTTP headers required when issuing the download.
            filename_hint: Suggested filename derived from resolver metadata.
            version: Version string reported by the resolver.
            license: License identifier reported by the resolver.
            media_type: MIME type associated with the ontology.
            service: Logical service identifier used for rate limiting.

        Returns:
            FetchPlan capturing resolver metadata.
        """
        normalized_license = normalize_license_to_spdx(license)
        return FetchPlan(
            url=url,
            headers=headers or {},
            filename_hint=filename_hint,
            version=version,
            license=normalized_license,
            media_type=media_type,
            service=service,
            last_modified=last_modified,
            content_length=content_length,
        )


class OBOResolver(BaseResolver):
    """Resolve ontologies hosted on the OBO Library using Bioregistry helpers.

    Attributes:
        None

    Examples:
        >>> resolver = OBOResolver()
        >>> callable(getattr(resolver, "plan"))
        True
    """

    def plan(self, spec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Resolve download URLs using Bioregistry-provided endpoints.

        Args:
            spec: Fetch specification describing the ontology to download.
            config: Global configuration with retry and timeout settings.
            logger: Logger adapter used to emit planning telemetry.

        Returns:
            FetchPlan pointing to the preferred download URL.

        Raises:
            ConfigError: If no download URL can be derived.
        """
        if get_obo_download is None or get_owl_download is None or get_rdf_download is None:
            raise ConfigError(
                "bioregistry is required for the OBO resolver. Install it with: pip install bioregistry"
            )

        preferred_formats = list(spec.target_formats) or ["owl", "obo", "rdf"]
        for fmt in preferred_formats:
            if fmt == "owl":
                url = get_owl_download(spec.id)
            elif fmt == "obo":
                url = get_obo_download(spec.id)
            else:
                url = get_rdf_download(spec.id)
            if url:
                logger.info(
                    "resolved download url",
                    extra={
                        "stage": "plan",
                        "resolver": "obo",
                        "ontology_id": spec.id,
                        "format": fmt,
                        "url": url,
                    },
                )
                return self._build_plan(
                    url=url,
                    media_type="application/rdf+xml",
                    service="obo",
                )
        raise ConfigError(f"No download link found via Bioregistry for {spec.id}")


class OLSResolver(BaseResolver):
    """Resolve ontologies from the Ontology Lookup Service (OLS4).

    Attributes:
        client: OLS client instance used to perform API calls.
        credentials_path: Path where the API token is expected.

    Examples:
        >>> resolver = OLSResolver()
        >>> resolver.credentials_path.name.endswith(".txt")
        True
    """

    def __init__(self) -> None:
        client_factory = OlsClient
        if client_factory is None:
            raise ConfigError("ols-client package is required for the OLS resolver")
        try:
            self.client = client_factory()
        except TypeError:
            try:
                self.client = client_factory("https://www.ebi.ac.uk/ols4")
            except (
                TypeError
            ):  # pragma: no cover - newer ols-client versions require keyword argument
                self.client = client_factory(base_url="https://www.ebi.ac.uk/ols4")
        self.credentials_path = pystow.join("ontology-fetcher", "configs") / "ols_api_token.txt"

    def plan(self, spec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Discover download locations via the OLS API.

        Args:
            spec: Fetch specification containing ontology identifiers and extras.
            config: Resolved configuration that provides retry policies.
            logger: Logger adapter used for planner progress messages.

        Returns:
            FetchPlan describing the download URL, headers, and metadata.

        Raises:
            ConfigError: When the API rejects credentials or yields no URLs.
        """
        ontology_id = spec.id.lower()
        headers = self._build_polite_headers(config, logger)
        try:
            session = getattr(self.client, "session", None)
        except RuntimeError:  # placeholder clients used in tests may raise
            session = None
        self._apply_headers_to_session(session, headers)

        try:
            record = self._execute_with_retry(
                lambda: self.client.get_ontology(ontology_id),
                config=config,
                logger=logger,
                name="ols",
                service="ols",
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                raise ConfigError(
                    f"OLS authentication failed with status {status}. Configure API key at {self.credentials_path}"
                ) from exc
            raise
        download_url = None
        version = None
        license_value = None
        filename = None
        if hasattr(record, "download"):
            download_url = getattr(record, "download")
        if not download_url and isinstance(record, dict):
            download_url = (
                record.get("config", {}).get("downloadLocation")
                or record.get("download")
                or record.get("links", {}).get("download")
            )
            version = record.get("version")
            license_value = record.get("license")
            filename = record.get("preferredPrefix")
        elif download_url:
            version = getattr(record, "version", None)
            license_value = getattr(record, "license", None)
        if not download_url:
            submissions = self._execute_with_retry(
                lambda: self.client.get_ontology_versions(ontology_id),
                config=config,
                logger=logger,
                name="ols",
                service="ols",
            )
            for submission in submissions:
                candidate = submission.get("downloadLocation") or submission.get("links", {}).get(
                    "download"
                )
                if candidate:
                    download_url = candidate
                    version = submission.get("version") or submission.get("shortForm")
                    license_value = submission.get("license")
                    break
        if not download_url:
            raise ConfigError(f"Unable to resolve OLS download URL for {ontology_id}")
        logger.info(
            "resolved download url",
            extra={"stage": "plan", "resolver": "ols", "ontology_id": spec.id, "url": download_url},
        )
        return self._build_plan(
            url=download_url,
            filename_hint=filename,
            version=version,
            license=license_value,
            service="ols",
        )


class BioPortalResolver(BaseResolver):
    """Resolve ontologies using the BioPortal (OntoPortal) API.

    Attributes:
        client: BioPortal client used to query ontology metadata.
        api_key_path: Path on disk containing the API key.

    Examples:
        >>> resolver = BioPortalResolver()
        >>> resolver.api_key_path.suffix
        '.txt'
    """

    def __init__(self) -> None:
        if BioPortalClient is None:
            raise ConfigError(
                "ontoportal-client is required for the BioPortal resolver. Install it with: pip install ontoportal-client"
            )
        self.client = BioPortalClient()
        config_dir = pystow.join("ontology-fetcher", "configs")
        self.api_key_path = config_dir / "bioportal_api_key.txt"

    def _load_api_key(self) -> Optional[str]:
        """Load the BioPortal API key from disk when available.

        Args:
            self: Resolver instance requesting the API key.

        Returns:
            Optional[str]: API key string stripped of whitespace, or ``None`` when missing.
        """
        if self.api_key_path.exists():
            return self.api_key_path.read_text().strip() or None
        return None

    def plan(self, spec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Resolve BioPortal download URLs and authorization headers.

        Args:
            spec: Fetch specification with optional API extras like acronyms.
            config: Resolved configuration that governs HTTP retry behaviour.
            logger: Logger adapter for structured telemetry.

        Returns:
            FetchPlan containing the resolved download URL and headers.

        Raises:
            ConfigError: If authentication fails or no download link is available.
        """
        acronym = spec.extras.get("acronym", spec.id.upper())
        headers = self._build_polite_headers(config, logger)
        self._apply_headers_to_session(getattr(self.client, "session", None), headers)

        try:
            ontology = self._execute_with_retry(
                lambda: self.client.get_ontology(acronym),
                config=config,
                logger=logger,
                name="bioportal",
                service="bioportal",
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                raise ConfigError(
                    f"BioPortal authentication failed with status {status}. Configure API key at {self.api_key_path}"
                ) from exc
            raise
        version = None
        license_value = None
        if isinstance(ontology, dict):
            license_value = ontology.get("license")
        latest_submission = self._execute_with_retry(
            lambda: self.client.get_latest_submission(acronym),
            config=config,
            logger=logger,
            name="bioportal",
            service="bioportal",
        )
        if isinstance(latest_submission, dict):
            download_url = (
                latest_submission.get("download")
                or latest_submission.get("links", {}).get("download")
                or latest_submission.get("ontologyPurl")
            )
            version = latest_submission.get("version") or latest_submission.get("submissionId")
        else:
            download_url = getattr(latest_submission, "download", None)
            if not download_url:
                links = getattr(latest_submission, "links", {})
                download_url = links.get("download") if isinstance(links, dict) else None
            version = getattr(latest_submission, "version", None)
            license_value = license_value or getattr(latest_submission, "license", None)
        if not download_url:
            raise ConfigError(f"No BioPortal submission with download URL for {acronym}")
        headers: Dict[str, str] = {}
        api_key = self._load_api_key()
        if api_key:
            headers["Authorization"] = f"apikey {api_key}"
        logger.info(
            "resolved download url",
            extra={
                "stage": "plan",
                "resolver": "bioportal",
                "ontology_id": spec.id,
                "url": download_url,
            },
        )
        return self._build_plan(
            url=download_url,
            headers=headers,
            version=version,
            license=license_value,
            service="bioportal",
        )


class LOVResolver(BaseResolver):
    """Resolve vocabularies from Linked Open Vocabularies (LOV).

    Queries the LOV API, normalises metadata fields, and returns Turtle
    download plans enriched with service identifiers for rate limiting.

    Attributes:
        API_ROOT: Base URL for the LOV API endpoints.
        session: Requests session used to execute API calls.

    Examples:
        >>> resolver = LOVResolver()
        >>> isinstance(resolver.session, requests.Session)
        True
    """

    API_ROOT = "https://lov.linkeddata.es/dataset/lov/api/v2"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def _fetch_metadata(self, uri: str, timeout: int) -> Any:
        response = self.session.get(
            f"{self.API_ROOT}/vocabulary/info",
            params={"uri": uri},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _iter_dicts(payload: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(payload, dict):
            yield payload
            for value in payload.values():
                yield from LOVResolver._iter_dicts(value)
        elif isinstance(payload, list):
            for item in payload:
                yield from LOVResolver._iter_dicts(item)

    def plan(self, spec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Discover download metadata from the LOV API.

        Args:
            spec: Fetch specification providing ontology identifier and extras.
            config: Resolved configuration supplying timeout and header defaults.
            logger: Logger adapter used to emit planning telemetry.

        Returns:
            FetchPlan describing the resolved download URL and metadata.

        Raises:
            ConfigError: If required metadata is missing or the LOV API fails.
        """

        uri = spec.extras.get("uri")
        if not uri:
            raise ConfigError("LOV resolver requires 'extras.uri'")

        headers = self._build_polite_headers(config, logger)
        self._apply_headers_to_session(self.session, headers)

        timeout = config.defaults.http.timeout_sec
        metadata = self._execute_with_retry(
            lambda: self._fetch_metadata(uri, timeout),
            config=config,
            logger=logger,
            name="lov",
            service="lov",
        )

        download_url: Optional[str] = None
        license_value: Optional[str] = None
        version: Optional[str] = None
        media_type: Optional[str] = None

        for candidate in self._iter_dicts(metadata):
            if download_url is None:
                for key in (
                    "downloadURL",
                    "downloadUrl",
                    "download",
                    "accessURL",
                    "accessUrl",
                    "url",
                ):
                    value = candidate.get(key)
                    if isinstance(value, str) and value.strip():
                        download_url = value.strip()
                        break
                    if isinstance(value, dict):
                        nested = value.get("url") or value.get("downloadURL")
                        if isinstance(nested, str) and nested.strip():
                            download_url = nested.strip()
                            break
            license_value = license_value or candidate.get("license")
            if isinstance(license_value, dict):
                label = license_value.get("title") or license_value.get("label")
                if isinstance(label, str):
                    license_value = label
            if version is None:
                for key in ("version", "issued", "release", "releaseDate"):
                    value = candidate.get(key)
                    if isinstance(value, str) and value.strip():
                        version = value.strip()
                        break
            if media_type is None:
                candidate_type = candidate.get("mediaType") or candidate.get("format")
                if isinstance(candidate_type, str) and candidate_type.strip():
                    media_type = candidate_type.strip()

        if not download_url:
            raise ConfigError("LOV metadata did not include a download URL")

        logger.info(
            "resolved download url",
            extra={
                "stage": "plan",
                "resolver": "lov",
                "ontology_id": spec.id,
                "url": download_url,
            },
        )

        return self._build_plan(
            url=download_url,
            license=license_value,
            version=version,
            media_type=media_type or "text/turtle",
            service="lov",
        )


class SKOSResolver(BaseResolver):
    """Resolver for direct SKOS/RDF URLs.

    Attributes:
        None

    Examples:
        >>> resolver = SKOSResolver()
        >>> callable(getattr(resolver, "plan"))
        True
    """

    def plan(self, spec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Return a fetch plan for explicitly provided SKOS URLs.

        Args:
            spec: Fetch specification containing the `extras.url` field.
            config: Resolved configuration (unused, included for API symmetry).
            logger: Logger adapter used to report resolved URL information.

        Returns:
            FetchPlan with the provided URL and appropriate media type.

        Raises:
            ConfigError: If the specification omits the required URL.
        """
        url = spec.extras.get("url")
        if not url:
            raise ConfigError("SKOS resolver requires 'extras.url'")
        logger.info(
            "resolved download url",
            extra={"stage": "plan", "resolver": "skos", "ontology_id": spec.id, "url": url},
        )
        return self._build_plan(
            url=url,
            media_type="text/turtle",
            service="skos",
        )


class DirectResolver(BaseResolver):
    """Resolver that consumes explicit URLs supplied via ``spec.extras``."""

    def plan(self, spec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Return a fetch plan using the direct URL provided in ``spec.extras``.

        Args:
            spec: Fetch specification containing the upstream download details.
            config: Resolved configuration (unused, provided for interface parity).
            logger: Logger adapter used to record telemetry.

        Returns:
            FetchPlan referencing the explicit URL.

        Raises:
            ConfigError: If the specification omits the required URL or provides invalid extras.
        """

        extras = getattr(spec, "extras", {}) or {}
        if not isinstance(extras, dict):
            raise ConfigError("direct resolver expects spec.extras to be a mapping")

        url = extras.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ConfigError("direct resolver requires 'extras.url'")
        url = url.strip()

        headers = extras.get("headers")
        if headers is not None and not isinstance(headers, dict):
            raise ConfigError("direct resolver expects 'extras.headers' to be a mapping")

        filename_hint = extras.get("filename_hint")
        if filename_hint is not None and not isinstance(filename_hint, str):
            raise ConfigError("direct resolver expects 'extras.filename_hint' to be a string")

        license_hint = extras.get("license")
        if license_hint is not None and not isinstance(license_hint, str):
            raise ConfigError("direct resolver expects 'extras.license' to be a string")

        version = extras.get("version")
        if version is not None and not isinstance(version, str):
            raise ConfigError("direct resolver expects 'extras.version' to be a string")

        logger.info(
            "resolved download url",
            extra={
                "stage": "plan",
                "resolver": "direct",
                "ontology_id": spec.id,
                "url": url,
            },
        )
        return self._build_plan(
            url=url,
            headers=headers,
            filename_hint=filename_hint,
            version=version,
            license=license_hint,
            service="direct",
        )


class XBRLResolver(BaseResolver):
    """Resolver for XBRL taxonomy packages.

    Attributes:
        None

    Examples:
        >>> resolver = XBRLResolver()
        >>> callable(getattr(resolver, "plan"))
        True
    """

    def plan(self, spec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Return a fetch plan for XBRL ZIP archives provided via extras.

        Args:
            spec: Fetch specification containing the upstream download URL.
            config: Resolved configuration (unused, included for API compatibility).
            logger: Logger adapter for structured observability.

        Returns:
            FetchPlan referencing the specified ZIP archive.

        Raises:
            ConfigError: If the specification omits the required URL.
        """
        url = spec.extras.get("url")
        if not url:
            raise ConfigError("XBRL resolver requires 'extras.url'")
        logger.info(
            "resolved download url",
            extra={"stage": "plan", "resolver": "xbrl", "ontology_id": spec.id, "url": url},
        )
        return self._build_plan(
            url=url,
            media_type="application/zip",
            service="xbrl",
        )


class OntobeeResolver(BaseResolver):
    """Resolver that constructs Ontobee-backed PURLs for OBO ontologies.

    Provides a lightweight fallback resolver that constructs deterministic
    PURLs for OBO prefixes when primary resolvers fail.

    Attributes:
        _FORMAT_MAP: Mapping of preferred formats to extensions and media types.

    Examples:
        >>> resolver = OntobeeResolver()
        >>> resolver._FORMAT_MAP['owl'][0]
        'owl'
    """

    _FORMAT_MAP = {
        "owl": ("owl", "application/rdf+xml"),
        "obo": ("obo", "text/plain"),
        "ttl": ("ttl", "text/turtle"),
    }

    def plan(self, spec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Return a fetch plan pointing to Ontobee-managed PURLs.

        Args:
            spec: Fetch specification describing the ontology identifier and preferred formats.
            config: Resolved configuration (unused beyond interface compatibility).
            logger: Logger adapter for structured telemetry.

        Returns:
            FetchPlan pointing to an Ontobee-hosted download URL.

        Raises:
            ConfigError: If the ontology identifier is invalid.
        """

        prefix = spec.id.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]+", prefix):
            raise ConfigError("Ontobee resolver requires alphanumeric ontology id")

        preferred = [fmt.lower() for fmt in (spec.target_formats or []) if isinstance(fmt, str)]
        if not preferred:
            preferred = ["owl", "obo"]

        for fmt in preferred:
            mapping = self._FORMAT_MAP.get(fmt)
            if mapping:
                extension, media_type = mapping
                url = f"https://purl.obolibrary.org/obo/{prefix.lower()}.{extension}"
                logger.info(
                    "resolved download url",
                    extra={
                        "stage": "plan",
                        "resolver": "ontobee",
                        "ontology_id": spec.id,
                        "url": url,
                    },
                )
                return self._build_plan(
                    url=url,
                    media_type=media_type,
                    service="ontobee",
                )

        # Fall back to OWL representation if no preferred format matched.
        url = f"https://purl.obolibrary.org/obo/{prefix.lower()}.owl"
        logger.info(
            "resolved download url",
            extra={
                "stage": "plan",
                "resolver": "ontobee",
                "ontology_id": spec.id,
                "url": url,
            },
        )
        return self._build_plan(url=url, media_type="application/rdf+xml", service="ontobee")


def _load_resolver_plugins(logger: Optional[logging.Logger] = None) -> None:
    """Discover resolver plugins registered via Python entry points."""

    logger = logger or logging.getLogger(__name__)
    try:
        entry_points = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "resolver plugin discovery failed",
            extra={"stage": "init", "error": str(exc)},
        )
        return

    for entry in entry_points.select(group="docstokg.ontofetch.resolver"):
        try:
            loaded = entry.load()
            resolver = loaded() if isinstance(loaded, type) else loaded
            if not hasattr(resolver, "plan"):
                raise TypeError("resolver plugin must define a plan method")
            name = getattr(resolver, "NAME", entry.name)
            RESOLVERS[name] = resolver
            logger.info(
                "resolver plugin registered",
                extra={"stage": "init", "resolver": name},
            )
        except Exception as exc:  # pragma: no cover - plugin faults
            logger.warning(
                "resolver plugin failed",
                extra={"stage": "init", "resolver": entry.name, "error": str(exc)},
            )


_PLUGINS_LOADED = False


def _ensure_plugins_loaded(logger: Optional[logging.Logger] = None) -> None:
    """Ensure resolver plugins are loaded at most once per interpreter."""

    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _load_resolver_plugins(logger)
    _PLUGINS_LOADED = True


_LOGGER = logging.getLogger(__name__)

RESOLVERS: Dict[str, BaseResolver] = {
    "obo": OBOResolver(),
    "bioregistry": OBOResolver(),
    "lov": LOVResolver(),
    "skos": SKOSResolver(),
    "direct": DirectResolver(),
    "xbrl": XBRLResolver(),
    "ontobee": OntobeeResolver(),
}

if _OlsClient is not None:
    try:
        RESOLVERS["ols"] = OLSResolver()
    except Exception as exc:  # pragma: no cover - depends on local credentials
        _LOGGER.debug("OLS resolver disabled: %s", exc)
else:  # pragma: no cover - depends on optional dependency presence
    _LOGGER.debug("OLS resolver disabled because ols-client is not installed")

if BioPortalClient is not None:
    try:
        RESOLVERS["bioportal"] = BioPortalResolver()
    except Exception as exc:  # pragma: no cover - depends on API key availability
        _LOGGER.debug("BioPortal resolver disabled: %s", exc)
else:  # pragma: no cover - depends on optional dependency presence
    _LOGGER.debug("BioPortal resolver disabled because ontoportal-client is not installed")

_ensure_plugins_loaded()


# --- Globals ---

__all__ = [
    "FetchPlan",
    "OBOResolver",
    "OLSResolver",
    "BioPortalResolver",
    "LOVResolver",
    "SKOSResolver",
    "DirectResolver",
    "XBRLResolver",
    "OntobeeResolver",
    "RESOLVERS",
    "normalize_license_to_spdx",
]
