"""
Ontology Resolver Implementations

This module defines the resolver strategies that convert download
specifications into actionable fetch plans. Each resolver encapsulates the
API integration, retry logic, and metadata extraction necessary to interact
with external services such as the OBO Library, OLS, BioPortal, and SKOS/XBRL
endpoints.

Key Features:
- Shared retry/backoff helpers for consistent API resilience
- Resolver-specific metadata extraction (version, license, media type)
- Support for additional services through the pluggable ``RESOLVERS`` map

Usage:
    from DocsToKG.OntologyDownload.resolvers import RESOLVERS

    resolver = RESOLVERS[\"obo\"]
    plan = resolver.plan(spec, config, logger)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from bioregistry import get_obo_download, get_owl_download, get_rdf_download
from ols_client import OlsClient
from ontoportal_client import BioPortalClient

from .config import ConfigError, ResolvedConfig
from .optdeps import get_pystow

pystow = get_pystow()


@dataclass(slots=True)
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


class BaseResolver:
    """Shared helpers for resolver implementations.

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
        self, func, *, config: ResolvedConfig, logger: logging.Logger, name: str
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
        attempts = 0
        max_attempts = max(1, config.defaults.http.max_retries)
        while True:
            attempts += 1
            try:
                return func()
            except requests.Timeout as exc:
                if attempts >= max_attempts:
                    raise ConfigError(
                        f"{name} API timeout after {config.defaults.http.timeout_sec}s"
                    ) from exc
                sleep_time = config.defaults.http.backoff_factor * (2 ** (attempts - 1))
                logger.warning(
                    "resolver timeout",
                    extra={
                        "stage": "plan",
                        "resolver": name,
                        "attempt": attempts,
                        "sleep_sec": sleep_time,
                    },
                )
                time.sleep(sleep_time)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in {401, 403}:
                    raise
                raise ConfigError(f"{name} API error {status}: {exc}") from exc

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
        return FetchPlan(
            url=url,
            headers=headers or {},
            filename_hint=filename_hint,
            version=version,
            license=license,
            media_type=media_type,
            service=service,
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
                    service=spec.resolver,
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
        self.client = OlsClient()
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
        try:
            record = self._execute_with_retry(
                lambda: self.client.get_ontology(ontology_id),
                config=config,
                logger=logger,
                name="ols",
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
            service=spec.resolver,
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
        try:
            ontology = self._execute_with_retry(
                lambda: self.client.get_ontology(acronym),
                config=config,
                logger=logger,
                name="bioportal",
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
            service=spec.resolver,
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
            service=spec.resolver,
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
            service=spec.resolver,
        )


RESOLVERS = {
    "obo": OBOResolver(),
    "bioregistry": OBOResolver(),
    "ols": OLSResolver(),
    "bioportal": BioPortalResolver(),
    "skos": SKOSResolver(),
    "xbrl": XBRLResolver(),
}


__all__ = [
    "FetchPlan",
    "OBOResolver",
    "OLSResolver",
    "BioPortalResolver",
    "SKOSResolver",
    "XBRLResolver",
    "RESOLVERS",
]
