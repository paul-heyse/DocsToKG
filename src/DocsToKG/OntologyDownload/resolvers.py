# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.resolvers",
#   "purpose": "Resolver implementations for DocsToKG ontology downloads",
#   "sections": [
#     {
#       "id": "normalize-license-to-spdx",
#       "name": "normalize_license_to_spdx",
#       "anchor": "function-normalize-license-to-spdx",
#       "kind": "function"
#     },
#     {
#       "id": "fetchplan",
#       "name": "FetchPlan",
#       "anchor": "class-fetchplan",
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
#       "id": "baseresolver",
#       "name": "BaseResolver",
#       "anchor": "class-baseresolver",
#       "kind": "class"
#     },
#     {
#       "id": "oboresolver",
#       "name": "OBOResolver",
#       "anchor": "class-oboresolver",
#       "kind": "class"
#     },
#     {
#       "id": "olsresolver",
#       "name": "OLSResolver",
#       "anchor": "class-olsresolver",
#       "kind": "class"
#     },
#     {
#       "id": "bioportalresolver",
#       "name": "BioPortalResolver",
#       "anchor": "class-bioportalresolver",
#       "kind": "class"
#     },
#     {
#       "id": "lovresolver",
#       "name": "LOVResolver",
#       "anchor": "class-lovresolver",
#       "kind": "class"
#     },
#     {
#       "id": "skosresolver",
#       "name": "SKOSResolver",
#       "anchor": "class-skosresolver",
#       "kind": "class"
#     },
#     {
#       "id": "directresolver",
#       "name": "DirectResolver",
#       "anchor": "class-directresolver",
#       "kind": "class"
#     },
#     {
#       "id": "xbrlresolver",
#       "name": "XBRLResolver",
#       "anchor": "class-xbrlresolver",
#       "kind": "class"
#     },
#     {
#       "id": "ontobeeresolver",
#       "name": "OntobeeResolver",
#       "anchor": "class-ontobeeresolver",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Resolver implementations and registry wiring for ontology downloads.

The resolver layer turns a logical ontology specification into concrete fetch
plans.  This module defines:

- the resolver registry and plugin integration used to discover third-party
  resolvers,
- shared mixins and the :class:`BaseResolver` contract that implement retry,
  license, and checksum enforcement,
- concrete resolvers for OBO, OLS, BioPortal, LOV, SKOS, Ontobee, XBRL, and
  direct-download sources,
- utilities such as :func:`normalize_license_to_spdx` that harmonise metadata
  before manifests are written.

Resolvers are invoked by :mod:`DocsToKG.OntologyDownload.planning` to produce
robust fallback chains and by the validation pipeline to understand the
provenance of downloaded artefacts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple
from urllib.parse import urlparse

import requests

from . import io
from .cancellation import CancellationToken
from .errors import ResolverError, UserConfigError
from .io import (
    get_bucket,
    is_retryable_error,
    retry_with_backoff,
    validate_url_security,
)
from .plugins import ensure_resolver_plugins, register_plugin_registry
from .settings import DownloadConfiguration, ResolvedConfig, get_pystow

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .planning import FetchSpec

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

OlsClient = _OlsClient
pystow = get_pystow()

_FORMAT_TO_MEDIA = {
    "owl": "application/rdf+xml",
    "rdf": "application/rdf+xml",
    "rdf/xml": "application/rdf+xml",
    "rdfxml": "application/rdf+xml",
    "obo": "text/plain",
    "ttl": "text/turtle",
    "turtle": "text/turtle",
    "jsonld": "application/ld+json",
    "json-ld": "application/ld+json",
    "ntriples": "application/n-triples",
    "nt": "application/n-triples",
    "trig": "application/trig",
}

LOGGER = logging.getLogger(__name__)


# --- Normalization Helpers ---


def normalize_license_to_spdx(value: Optional[str]) -> Optional[str]:
    """Normalize common license strings to canonical SPDX identifiers."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("title", "prefLabel", "name", "label"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                value = candidate
                break
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
    if cleaned.upper().startswith("CC-BY-"):
        return cleaned.upper()
    return cleaned


# --- Resolver Data Structures ---


@dataclass(slots=True)
class FetchPlan:
    """Concrete plan output from a resolver."""

    url: str
    headers: Dict[str, str]
    filename_hint: Optional[str]
    version: Optional[str]
    license: Optional[str]
    media_type: Optional[str]
    service: Optional[str] = None
    last_modified: Optional[str] = None
    content_length: Optional[int] = None
    checksum: Optional[str] = None
    checksum_algorithm: Optional[str] = None
    checksum_url: Optional[str] = None


class Resolver(Protocol):
    """Protocol describing resolver planning behaviour."""

    def plan(
        self,
        spec: "FetchSpec",
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Produce a :class:`FetchPlan` describing how to retrieve ``spec``.

        Args:
            spec: The fetch specification to plan for.
            config: The resolved configuration.
            logger: Logger for this operation.
            cancellation_token: Optional token for cooperative cancellation.

        Returns:
            A fetch plan describing how to retrieve the specification.
        """
        ...


@dataclass(slots=True)
class ResolverCandidate:
    """Resolver plan captured for download-time fallback."""

    resolver: str
    plan: FetchPlan


class BaseResolver:
    """Shared helpers for resolver implementations."""

    def _normalize_media_type(self, media_type: Optional[str]) -> Optional[str]:
        if not media_type:
            return None
        canonical = media_type.split(";")[0].strip().lower()
        if not canonical:
            return None
        if canonical in io.RDF_MIME_FORMAT_LABELS:
            return canonical
        return canonical

    def _preferred_media_type(
        self,
        spec: "FetchSpec",
        *,
        default: Optional[str],
    ) -> Optional[str]:
        formats = getattr(spec, "target_formats", ()) or ()
        for value in formats:
            if not isinstance(value, str):
                continue
            candidate = _FORMAT_TO_MEDIA.get(value.strip().lower())
            if candidate:
                return candidate
        return default

    def _negotiate_media_type(
        self,
        *,
        spec: "FetchSpec",
        default: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[str], Dict[str, str]]:
        resolved_headers: Dict[str, str] = dict(headers or {})
        accept_key: Optional[str] = None
        for key in resolved_headers:
            if key.lower() == "accept":
                accept_key = key
                break
        if accept_key is not None:
            media = self._normalize_media_type(resolved_headers[accept_key])
            if media:
                resolved_headers[accept_key] = media
                return media, resolved_headers
        preferred = self._preferred_media_type(spec, default=default)
        media_type = self._normalize_media_type(preferred)
        if media_type:
            resolved_headers["Accept"] = media_type
        return media_type, resolved_headers

    def _execute_with_retry(
        self,
        func,
        *,
        config: ResolvedConfig,
        logger: logging.Logger,
        name: str,
        service: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """Run a callable with retry semantics tailored for resolver APIs."""

        max_attempts = max(1, config.defaults.http.max_retries)
        backoff_base = config.defaults.http.backoff_factor

        def _retryable(exc: Exception) -> bool:
            return is_retryable_error(exc)

        def _on_retry(attempt: int, exc: Exception, sleep_time: float) -> None:
            logger.warning(
                "resolver timeout",
                extra={
                    "stage": "plan",
                    "resolver": name,
                    "attempt": attempt,
                    "sleep_sec": sleep_time,
                    "error": str(exc),
                },
            )

        def _invoke():
            bucket = (
                get_bucket(
                    http_config=config.defaults.http,
                    service=service,
                    host=host,
                )
                if service
                else None
            )
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
            raise ResolverError(
                f"{name} API timeout after {config.defaults.http.timeout_sec}s"
            ) from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                raise UserConfigError(f"{name} API rejected credentials ({status})") from exc
            raise ResolverError(f"{name} API error {status}: {exc}") from exc
        except requests.ConnectionError as exc:
            raise ResolverError(f"{name} API connection error: {exc}") from exc

    def _extract_correlation_id(self, logger: logging.Logger) -> Optional[str]:
        """Return the correlation id from a logger adapter when available."""

        extra = getattr(logger, "extra", None)
        if isinstance(extra, dict):
            value = extra.get("correlation_id")
            if isinstance(value, str):
                return value
        return None

    def _build_polite_headers(
        self, config: ResolvedConfig, logger: logging.Logger
    ) -> Dict[str, str]:
        """Create polite headers derived from configuration and logger context."""

        http_config = config.defaults.http
        return http_config.polite_http_headers(correlation_id=self._extract_correlation_id(logger))

    @staticmethod
    def _apply_headers_to_session(session: Any, headers: Dict[str, str]) -> None:
        """Apply polite headers to a client session when supported."""

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

    def _request_with_retry(
        self,
        *,
        method: str,
        url: str,
        config: ResolvedConfig,
        logger: logging.Logger,
        service: Optional[str],
        session: Optional[requests.Session] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> requests.Response:
        """Issue an HTTP request with polite headers and retry semantics."""

        final_headers: Dict[str, str] = dict(self._build_polite_headers(config, logger))
        if headers:
            final_headers.update({str(k): str(v) for k, v in dict(headers).items()})

        timeout_value = timeout if timeout is not None else config.defaults.http.timeout_sec
        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("headers", final_headers)
        request_kwargs.setdefault("timeout", timeout_value)

        method_name = method.lower()
        method_upper = method.upper()

        def _perform() -> requests.Response:
            if session is not None:
                requester = getattr(session, "request", None)
                if callable(requester):
                    response = requester(method=method_upper, url=url, **request_kwargs)
                else:
                    method_func = getattr(session, method_name, None)
                    if not callable(method_func):
                        raise AttributeError(f"session lacks '{method}' request method")
                    response = method_func(url, **request_kwargs)
            else:
                response = requests.request(method_upper, url, **request_kwargs)
            response.raise_for_status()
            return response

        parsed = urlparse(url)
        host = parsed.hostname

        return self._execute_with_retry(
            _perform,
            config=config,
            logger=logger,
            name=service or "resolver",
            service=service,
            host=host,
        )

    def _build_plan(
        self,
        *,
        url: str,
        http_config: Optional[DownloadConfiguration] = None,
        headers: Optional[Dict[str, str]] = None,
        filename_hint: Optional[str] = None,
        version: Optional[str] = None,
        license: Optional[str] = None,
        media_type: Optional[str] = None,
        service: Optional[str] = None,
        last_modified: Optional[str] = None,
        content_length: Optional[int] = None,
        checksum: Optional[str] = None,
        checksum_algorithm: Optional[str] = None,
        checksum_url: Optional[str] = None,
    ) -> FetchPlan:
        """Construct a ``FetchPlan`` from resolver components."""

        secure_url = validate_url_security(url, http_config)
        normalized_license = normalize_license_to_spdx(license)
        return FetchPlan(
            url=secure_url,
            headers=headers or {},
            filename_hint=filename_hint,
            version=version,
            license=normalized_license,
            media_type=media_type,
            service=service,
            last_modified=last_modified,
            content_length=content_length,
            checksum=checksum,
            checksum_algorithm=checksum_algorithm,
            checksum_url=checksum_url,
        )


# --- Resolver Implementations ---


class OBOResolver(BaseResolver):
    """Resolve ontologies hosted on the OBO Library using Bioregistry helpers."""

    def plan(
        self,
        spec,
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Build a :class:`FetchPlan` by resolving OBO-hosted download URLs."""

        if cancellation_token and cancellation_token.is_cancelled():
            raise ResolverError("Operation cancelled")

        if get_obo_download is None or get_owl_download is None or get_rdf_download is None:
            raise UserConfigError(
                "bioregistry is required for the OBO resolver. Install it with: pip install bioregistry"
            )

        preferred_formats = list(spec.target_formats) or ["owl", "obo", "rdf"]
        for fmt in preferred_formats:
            if cancellation_token and cancellation_token.is_cancelled():
                raise ResolverError("Operation cancelled")
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
                media_type, headers = self._negotiate_media_type(
                    spec=spec,
                    default=_FORMAT_TO_MEDIA.get(fmt, "application/rdf+xml"),
                )
                return self._build_plan(
                    url=url,
                    http_config=config.defaults.http,
                    headers=headers,
                    media_type=media_type or _FORMAT_TO_MEDIA.get(fmt, "application/rdf+xml"),
                    service="obo",
                )
        raise ResolverError(f"No download link found via Bioregistry for {spec.id}")


class OLSResolver(BaseResolver):
    """Resolve ontologies from the Ontology Lookup Service (OLS4)."""

    def __init__(self) -> None:
        client_factory = OlsClient
        if client_factory is None:
            raise UserConfigError("ols-client package is required for the OLS resolver")
        try:
            self.client = client_factory()
        except TypeError:
            try:
                self.client = client_factory("https://www.ebi.ac.uk/ols4")
            except TypeError:  # pragma: no cover - keyword-only versions
                self.client = client_factory(base_url="https://www.ebi.ac.uk/ols4")
        self.credentials_path = pystow.join("ontology-fetcher", "configs") / "ols_api_token.txt"

    def plan(
        self,
        spec,
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Plan an OLS download by negotiating media type and authentication."""

        if cancellation_token and cancellation_token.is_cancelled():
            raise ResolverError("Operation cancelled")

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
                host=urlparse(getattr(self.client, "base_url", "https://www.ebi.ac.uk")).hostname,
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                raise UserConfigError(
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
            filename = record.get("id")
        if not download_url:
            raise ResolverError(f"No OLS download URL found for {spec.id}")
        logger.info(
            "resolved download url",
            extra={
                "stage": "plan",
                "resolver": "ols",
                "ontology_id": spec.id,
                "url": download_url,
            },
        )
        media_type, download_headers = self._negotiate_media_type(
            spec=spec,
            default="application/rdf+xml",
            headers=headers,
        )
        return self._build_plan(
            url=download_url,
            http_config=config.defaults.http,
            headers=download_headers,
            filename_hint=filename,
            version=version,
            license=normalize_license_to_spdx(license_value),
            media_type=media_type,
            service="ols",
        )


class BioPortalResolver(BaseResolver):
    """Resolve ontologies using the BioPortal (OntoPortal) API."""

    def __init__(self) -> None:
        if BioPortalClient is None:
            raise UserConfigError(
                "ontoportal-client is required for the BioPortal resolver. Install it with: pip install ontoportal-client"
            )
        self.client = BioPortalClient()
        config_dir = pystow.join("ontology-fetcher", "configs")
        self.api_key_path = config_dir / "bioportal_api_key.txt"

    def _load_api_key(self) -> Optional[str]:
        if self.api_key_path.exists():
            return self.api_key_path.read_text().strip() or None
        return None

    def plan(
        self,
        spec,
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Plan a BioPortal download by combining ontology and submission metadata."""

        if cancellation_token and cancellation_token.is_cancelled():
            raise ResolverError("Operation cancelled")

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
                host=urlparse(
                    getattr(self.client, "base_url", "https://data.bioontology.org")
                ).hostname,
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                raise UserConfigError(
                    f"BioPortal authentication failed with status {status}. Configure API key at {self.api_key_path}"
                ) from exc
            raise
        version = None
        license_value = None
        if isinstance(ontology, dict):
            license_value = normalize_license_to_spdx(ontology.get("license"))
        latest_submission = self._execute_with_retry(
            lambda: self.client.get_latest_submission(acronym),
            config=config,
            logger=logger,
            name="bioportal",
            service="bioportal",
            host=urlparse(
                getattr(self.client, "base_url", "https://data.bioontology.org")
            ).hostname,
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
            license_value = license_value or normalize_license_to_spdx(
                getattr(latest_submission, "license", None)
            )
        if not download_url:
            raise ResolverError(f"No BioPortal submission with download URL for {acronym}")
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
        media_type, download_headers = self._negotiate_media_type(
            spec=spec,
            default="application/rdf+xml",
            headers=headers,
        )
        return self._build_plan(
            url=download_url,
            http_config=config.defaults.http,
            headers=download_headers,
            version=version,
            license=license_value,
            media_type=media_type,
            service="bioportal",
        )


class LOVResolver(BaseResolver):
    """Resolve vocabularies from Linked Open Vocabularies (LOV)."""

    API_ROOT = "https://lov.linkeddata.es/dataset/lov/api/v2"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    @staticmethod
    def _iter_dicts(payload: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(payload, dict):
            yield payload
            for value in payload.values():
                yield from LOVResolver._iter_dicts(value)
        elif isinstance(payload, list):
            for item in payload:
                yield from LOVResolver._iter_dicts(item)

    def plan(
        self,
        spec,
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Plan a LOV download by inspecting linked metadata for URLs and licenses."""

        uri = spec.extras.get("uri")
        if not uri:
            raise UserConfigError("LOV resolver requires 'extras.uri'")

        headers = self._build_polite_headers(config, logger)
        self._apply_headers_to_session(self.session, headers)

        timeout = max(1, config.defaults.http.timeout_sec)
        response = self._request_with_retry(
            method="GET",
            url=f"{self.API_ROOT}/vocabulary/info",
            config=config,
            logger=logger,
            service="lov",
            session=self.session,
            headers=headers,
            timeout=timeout,
            params={"uri": uri},
        )
        metadata = response.json()
        download_url = None
        license_value = None
        version_value = None
        preferred_media = spec.extras.get("media_type")
        for entry in LOVResolver._iter_dicts(metadata):
            if not isinstance(entry, dict):
                continue
            if download_url is None:
                candidate = entry.get("download") or entry.get("downloadURL")
                if candidate:
                    download_url = candidate
            if license_value is None:
                license_value = normalize_license_to_spdx(entry.get("license"))
            if version_value is None:
                candidate_version = entry.get("version") or entry.get("latestVersion")
                if isinstance(candidate_version, str):
                    version_value = candidate_version
            if preferred_media is None:
                preferred_media = entry.get("format") or entry.get("mediaType")
        if not download_url:
            raise ResolverError(f"LOV resolver could not locate download URL for {uri}")
        logger.info(
            "resolved download url",
            extra={
                "stage": "plan",
                "resolver": "lov",
                "ontology_id": spec.id,
                "url": download_url,
            },
        )
        media_type, download_headers = self._negotiate_media_type(
            spec=spec,
            default=preferred_media,
        )
        return self._build_plan(
            url=download_url,
            http_config=config.defaults.http,
            headers=download_headers,
            media_type=media_type or preferred_media,
            service="lov",
            license=license_value,
            version=version_value,
        )


class SKOSResolver(BaseResolver):
    """Resolve SKOS vocabularies specified directly via configuration."""

    def plan(
        self,
        spec,
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Plan a SKOS vocabulary download from explicit configuration metadata."""

        url = spec.extras.get("url")
        if not url:
            raise UserConfigError("SKOS resolver requires 'extras.url'")
        headers = self._build_polite_headers(config, logger)
        logger.info(
            "resolved download url",
            extra={"stage": "plan", "resolver": "skos", "ontology_id": spec.id, "url": url},
        )
        default_media = (
            spec.extras.get("media_type")
            if isinstance(spec.extras.get("media_type"), str)
            else "application/rdf+xml"
        )
        media_type, download_headers = self._negotiate_media_type(
            spec=spec,
            default=default_media,
            headers=headers,
        )
        return self._build_plan(
            url=url,
            http_config=config.defaults.http,
            headers=download_headers,
            media_type=media_type or default_media,
            service="skos",
            license=spec.extras.get("license"),
        )


class DirectResolver(BaseResolver):
    """Resolve direct download links specified in configuration extras."""

    def plan(
        self,
        spec,
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Plan a direct download from declarative extras without discovery."""

        if cancellation_token and cancellation_token.is_cancelled():
            raise ResolverError("Operation cancelled")

        extras = spec.extras
        if not isinstance(extras, Mapping):
            raise UserConfigError("direct resolver expects spec.extras to be a mapping")
        url = extras.get("url")
        if not isinstance(url, str):
            raise UserConfigError("direct resolver requires 'extras.url'")
        headers_extra = extras.get("headers")
        if headers_extra is not None and not isinstance(headers_extra, Mapping):
            raise UserConfigError("direct resolver expects 'extras.headers' to be a mapping")
        filename_hint = extras.get("filename_hint")
        if filename_hint is not None and not isinstance(filename_hint, str):
            raise UserConfigError("direct resolver expects 'extras.filename_hint' to be a string")
        license_value = extras.get("license")
        if license_value is not None and not isinstance(license_value, str):
            raise UserConfigError("direct resolver expects 'extras.license' to be a string")
        version = extras.get("version")
        if version is not None and not isinstance(version, str):
            raise UserConfigError("direct resolver expects 'extras.version' to be a string")

        headers = self._build_polite_headers(config, logger)
        headers.update({str(k): str(v) for k, v in dict(headers_extra or {}).items()})
        logger.info(
            "resolved download url",
            extra={"stage": "plan", "resolver": "direct", "ontology_id": spec.id, "url": url},
        )
        default_media = (
            extras.get("media_type") if isinstance(extras.get("media_type"), str) else None
        )
        media_type, download_headers = self._negotiate_media_type(
            spec=spec,
            default=default_media,
            headers=headers,
        )
        return self._build_plan(
            url=url,
            http_config=config.defaults.http,
            headers=download_headers,
            filename_hint=filename_hint,
            version=version,
            license=normalize_license_to_spdx(license_value),
            media_type=media_type or default_media,
            service=extras.get("service"),
        )


class XBRLResolver(BaseResolver):
    """Resolve XBRL taxonomy downloads from regulator endpoints."""

    def plan(
        self,
        spec,
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Plan an XBRL taxonomy download using declarative resolver extras."""

        url = spec.extras.get("url")
        if not url:
            raise UserConfigError("XBRL resolver requires 'extras.url'")
        headers = self._build_polite_headers(config, logger)
        logger.info(
            "resolved download url",
            extra={"stage": "plan", "resolver": "xbrl", "ontology_id": spec.id, "url": url},
        )
        default_media = spec.extras.get("media_type", "application/zip")
        media_type, download_headers = self._negotiate_media_type(
            spec=spec,
            default=default_media,
            headers=headers,
        )
        return self._build_plan(
            url=url,
            http_config=config.defaults.http,
            headers=download_headers,
            filename_hint=spec.extras.get("filename_hint"),
            media_type=media_type or default_media,
            service="xbrl",
        )


class OntobeeResolver(BaseResolver):
    """Resolve Ontobee-hosted ontologies via canonical PURLs."""

    _FORMAT_MAP = {
        "owl": ("owl", "application/rdf+xml"),
        "obo": ("obo", "text/plain"),
        "ttl": ("ttl", "text/turtle"),
    }

    def plan(
        self,
        spec,
        config: ResolvedConfig,
        logger: logging.Logger,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> FetchPlan:
        """Plan an Ontobee download by composing the appropriate REST endpoint."""

        prefix = spec.id.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]+", prefix):
            raise UserConfigError("Ontobee resolver requires alphanumeric ontology id")

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
                negotiated_media, download_headers = self._negotiate_media_type(
                    spec=spec,
                    default=media_type,
                )
                return self._build_plan(
                    url=url,
                    http_config=config.defaults.http,
                    headers=download_headers,
                    media_type=negotiated_media or media_type,
                    service="ontobee",
                )

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


RESOLVERS: Dict[str, BaseResolver] = {
    "obo": OBOResolver(),
    "bioregistry": OBOResolver(),
    "lov": LOVResolver(),
    "skos": SKOSResolver(),
    "direct": DirectResolver(),
    "xbrl": XBRLResolver(),
    "ontobee": OntobeeResolver(),
}

if OlsClient is not None:
    try:
        RESOLVERS["ols"] = OLSResolver()
    except Exception as exc:  # pragma: no cover - depends on local credentials
        LOGGER.debug("OLS resolver disabled: %s", exc)
else:  # pragma: no cover
    LOGGER.debug("OLS resolver disabled because ols-client is not installed")

if BioPortalClient is not None:
    try:
        RESOLVERS["bioportal"] = BioPortalResolver()
    except Exception as exc:  # pragma: no cover - depends on API key availability
        LOGGER.debug("BioPortal resolver disabled: %s", exc)
else:  # pragma: no cover
    LOGGER.debug("BioPortal resolver disabled because ontoportal-client is not installed")

register_plugin_registry("resolver", RESOLVERS)
ensure_resolver_plugins(RESOLVERS, logger=LOGGER)


__all__ = [
    "FetchPlan",
    "Resolver",
    "ResolverCandidate",
    "BaseResolver",
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
