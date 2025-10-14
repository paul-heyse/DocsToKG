"""Resolver pipeline and provider implementations for the OpenAlex downloader.

The pipeline is intentionally lightweight so it can be reused by both the
command-line entrypoint and tests.  Resolvers yield candidate URLs (and
associated metadata) which are attempted in priority order until a confirmed
PDF is downloaded.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence
from urllib.parse import quote, urljoin, urlparse

import requests

try:  # Optional dependency; landing-page resolver guards at runtime.
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - handled downstream
    BeautifulSoup = None


DEFAULT_RESOLVER_ORDER: List[str] = [
    "unpaywall",
    "crossref",
    "landing_page",
    "arxiv",
    "pmc",
    "europe_pmc",
    "core",
    "doaj",
    "semantic_scholar",
    "wayback",
]


@dataclass
class ResolverResult:
    """Represents either a candidate URL or an informational event."""

    url: Optional[str]
    referer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event: Optional[str] = None
    event_reason: Optional[str] = None
    http_status: Optional[int] = None

    @property
    def is_event(self) -> bool:
        return self.url is None


@dataclass
class ResolverConfig:
    resolver_order: List[str] = field(
        default_factory=lambda: list(DEFAULT_RESOLVER_ORDER)
    )
    resolver_toggles: Dict[str, bool] = field(
        default_factory=lambda: {name: True for name in DEFAULT_RESOLVER_ORDER}
    )
    max_attempts_per_work: int = 25
    timeout: float = 30.0
    sleep_jitter: float = 0.35
    polite_headers: Dict[str, str] = field(default_factory=dict)
    unpaywall_email: Optional[str] = None
    core_api_key: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    doaj_api_key: Optional[str] = None
    resolver_timeouts: Dict[str, float] = field(default_factory=dict)
    resolver_rate_limits: Dict[str, float] = field(default_factory=dict)

    def get_timeout(self, resolver_name: str) -> float:
        return self.resolver_timeouts.get(resolver_name, self.timeout)

    def is_enabled(self, resolver_name: str) -> bool:
        return self.resolver_toggles.get(resolver_name, True)


@dataclass
class AttemptRecord:
    work_id: str
    resolver_name: str
    resolver_order: Optional[int]
    url: Optional[str]
    status: str
    http_status: Optional[int]
    content_type: Optional[str]
    elapsed_ms: Optional[float]
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttemptLogger(Protocol):
    def log(self, record: AttemptRecord) -> None:
        ...


@dataclass
class DownloadOutcome:
    classification: str
    path: Optional[str]
    http_status: Optional[int]
    content_type: Optional[str]
    elapsed_ms: Optional[float]
    error: Optional[str] = None

    @property
    def is_pdf(self) -> bool:
        return self.classification in {"pdf", "pdf_unknown"}


@dataclass
class PipelineResult:
    success: bool
    resolver_name: Optional[str] = None
    url: Optional[str] = None
    outcome: Optional[DownloadOutcome] = None
    html_paths: List[str] = field(default_factory=list)
    reason: Optional[str] = None


class Resolver(Protocol):
    name: str

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        ...

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        ...


@dataclass
class ResolverMetrics:
    attempts: Counter = field(default_factory=Counter)
    successes: Counter = field(default_factory=Counter)
    html: Counter = field(default_factory=Counter)
    skips: Counter = field(default_factory=Counter)

    def record_attempt(self, resolver_name: str, outcome: DownloadOutcome) -> None:
        self.attempts[resolver_name] += 1
        if outcome.classification == "html":
            self.html[resolver_name] += 1
        if outcome.is_pdf:
            self.successes[resolver_name] += 1

    def record_skip(self, resolver_name: str, reason: str) -> None:
        key = f"{resolver_name}:{reason}"
        self.skips[key] += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "attempts": dict(self.attempts),
            "successes": dict(self.successes),
            "html": dict(self.html),
            "skips": dict(self.skips),
        }


DownloadFunc = Callable[[requests.Session, "WorkArtifact", str, Optional[str], float], DownloadOutcome]


class ResolverPipeline:
    """Executes resolvers in priority order until a PDF download succeeds."""

    def __init__(
        self,
        resolvers: Sequence[Resolver],
        config: ResolverConfig,
        download_func: DownloadFunc,
        logger: AttemptLogger,
        metrics: Optional[ResolverMetrics] = None,
    ) -> None:
        self._resolver_map = {resolver.name: resolver for resolver in resolvers}
        self.config = config
        self.download_func = download_func
        self.logger = logger
        self.metrics = metrics or ResolverMetrics()
        self._last_invocation: Dict[str, float] = defaultdict(lambda: 0.0)

    def _respect_rate_limit(self, resolver_name: str) -> None:
        limit = self.config.resolver_rate_limits.get(resolver_name)
        if not limit:
            return
        last = self._last_invocation[resolver_name]
        now = time.monotonic()
        delta = now - last
        if delta < limit:
            time.sleep(limit - delta)

    def _jitter_sleep(self) -> None:
        if self.config.sleep_jitter <= 0:
            return
        time.sleep(self.config.sleep_jitter + random.random() * 0.1)

    def run(
        self,
        session: requests.Session,
        artifact: "WorkArtifact",
    ) -> PipelineResult:
        seen_urls: set[str] = set()
        html_paths: List[str] = []
        attempt_counter = 0

        for order_index, resolver_name in enumerate(self.config.resolver_order, start=1):
            resolver = self._resolver_map.get(resolver_name)
            if resolver is None:
                self.logger.log(
                    AttemptRecord(
                        work_id=artifact.work_id,
                        resolver_name=resolver_name,
                        resolver_order=order_index,
                        url=None,
                        status="skipped",
                        http_status=None,
                        content_type=None,
                        elapsed_ms=None,
                        reason="resolver-missing",
                    )
                )
                self.metrics.record_skip(resolver_name, "missing")
                continue

            if not self.config.is_enabled(resolver_name):
                self.logger.log(
                    AttemptRecord(
                        work_id=artifact.work_id,
                        resolver_name=resolver_name,
                        resolver_order=order_index,
                        url=None,
                        status="skipped",
                        http_status=None,
                        content_type=None,
                        elapsed_ms=None,
                        reason="resolver-disabled",
                    )
                )
                self.metrics.record_skip(resolver_name, "disabled")
                continue

            if not resolver.is_enabled(self.config, artifact):
                self.logger.log(
                    AttemptRecord(
                        work_id=artifact.work_id,
                        resolver_name=resolver_name,
                        resolver_order=order_index,
                        url=None,
                        status="skipped",
                        http_status=None,
                        content_type=None,
                        elapsed_ms=None,
                        reason="resolver-not-applicable",
                    )
                )
                self.metrics.record_skip(resolver_name, "not-applicable")
                continue

            self._respect_rate_limit(resolver_name)

            for result in resolver.iter_urls(session, self.config, artifact):
                if result.is_event:
                    self.logger.log(
                        AttemptRecord(
                            work_id=artifact.work_id,
                            resolver_name=resolver_name,
                            resolver_order=order_index,
                            url=None,
                            status=result.event or "event",
                            http_status=result.http_status,
                            content_type=None,
                            elapsed_ms=None,
                            reason=result.event_reason,
                            metadata=result.metadata,
                        )
                    )
                    if result.event_reason:
                        self.metrics.record_skip(resolver_name, result.event_reason)
                    continue

                url = result.url
                if not url:
                    continue
                if url in seen_urls:
                    self.logger.log(
                        AttemptRecord(
                            work_id=artifact.work_id,
                            resolver_name=resolver_name,
                            resolver_order=order_index,
                            url=url,
                            status="skipped",
                            http_status=None,
                            content_type=None,
                            elapsed_ms=None,
                            reason="duplicate-url",
                            metadata=result.metadata,
                        )
                    )
                    self.metrics.record_skip(resolver_name, "duplicate-url")
                    continue

                seen_urls.add(url)
                attempt_counter += 1
                outcome = self.download_func(
                    session,
                    artifact,
                    url,
                    result.referer,
                    self.config.get_timeout(resolver_name),
                )
                self._last_invocation[resolver_name] = time.monotonic()
                self.logger.log(
                    AttemptRecord(
                        work_id=artifact.work_id,
                        resolver_name=resolver_name,
                        resolver_order=order_index,
                        url=url,
                        status=outcome.classification,
                        http_status=outcome.http_status,
                        content_type=outcome.content_type,
                        elapsed_ms=outcome.elapsed_ms,
                        reason=outcome.error,
                        metadata=result.metadata,
                    )
                )
                self.metrics.record_attempt(resolver_name, outcome)

                if outcome.classification == "html" and outcome.path:
                    html_paths.append(outcome.path)

                if outcome.is_pdf:
                    return PipelineResult(
                        success=True,
                        resolver_name=resolver_name,
                        url=url,
                        outcome=outcome,
                        html_paths=html_paths,
                    )

                if attempt_counter >= self.config.max_attempts_per_work:
                    return PipelineResult(
                        success=False,
                        resolver_name=resolver_name,
                        url=url,
                        outcome=outcome,
                        html_paths=html_paths,
                        reason="max-attempts-reached",
                    )

                self._jitter_sleep()

        return PipelineResult(success=False, html_paths=html_paths)


# --- Resolver Implementations -------------------------------------------------


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[16:]
    return doi.strip()


def _normalize_pmcid(pmcid: Optional[str]) -> Optional[str]:
    if not pmcid:
        return None
    pmcid = pmcid.strip()
    if pmcid.lower().startswith("pmc"):
        pmcid = pmcid[3:]
    return f"PMC{pmcid}" if pmcid else None


def _strip_prefix(value: Optional[str], prefix: str) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if value.lower().startswith(prefix.lower()):
        return value[len(prefix) :]
    return value


def _absolute_url(base: str, href: str) -> str:
    parsed = urlparse(href)
    if parsed.scheme and parsed.netloc:
        return href
    return urljoin(base, href)


class UnpaywallResolver:
    name = "unpaywall"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return bool(config.unpaywall_email and artifact.doi)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        doi = _normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-doi",
            )
            return
        try:
            resp = session.get(
                f"https://api.unpaywall.org/v2/{quote(doi)}",
                params={"email": config.unpaywall_email},
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException as exc:  # pragma: no cover - network errors
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"message": str(exc)},
            )
            return

        if resp.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=resp.status_code,
            )
            return

        try:
            data = resp.json()
        except ValueError:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
            )
            return

        seen = set()
        best = (data or {}).get("best_oa_location") or {}
        url = best.get("url_for_pdf")
        if url:
            seen.add(url)
            yield ResolverResult(
                url=url,
                metadata={"source": "best_oa_location"},
            )

        for loc in (data or {}).get("oa_locations", []) or []:
            if not isinstance(loc, dict):
                continue
            url = loc.get("url_for_pdf")
            if url and url not in seen:
                seen.add(url)
                yield ResolverResult(
                    url=url,
                    metadata={"source": "oa_location"},
                )


class CrossrefResolver:
    name = "crossref"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        doi = _normalize_doi(artifact.doi)
        if not doi:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-doi",
            )
            return
        try:
            resp = session.get(
                f"https://api.crossref.org/works/{quote(doi)}",
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException as exc:  # pragma: no cover - network errors
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"message": str(exc)},
            )
            return

        if resp.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=resp.status_code,
            )
            return

        try:
            data = resp.json()
        except ValueError:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="json-error",
            )
            return

        message = (data or {}).get("message") or {}
        links = message.get("link") or []
        seen = set()
        priority_urls: List[ResolverResult] = []
        secondary_urls: List[ResolverResult] = []
        for link in links:
            if not isinstance(link, dict):
                continue
            url = link.get("URL")
            if not url or url in seen:
                continue
            seen.add(url)
            meta = {
                "content_type": link.get("content-type"),
                "content_version": link.get("content-version"),
                "application": link.get("intended-application"),
            }
            ctype = (link.get("content-type") or "").lower()
            result = ResolverResult(url=url, metadata=meta)
            if "application/pdf" in ctype:
                priority_urls.append(result)
            else:
                secondary_urls.append(result)

        for result in chain(priority_urls, secondary_urls):  # type: ignore[name-defined]
            yield result


class LandingPageResolver:
    name = "landing_page"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return bool(artifact.landing_urls)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        if BeautifulSoup is None:
            yield ResolverResult(
                url=None,
                event="skipped",
                event_reason="no-beautifulsoup",
            )
            return
        for landing in artifact.landing_urls:
            try:
                resp = session.get(
                    landing,
                    headers=config.polite_headers,
                    timeout=config.get_timeout(self.name),
                )
            except requests.RequestException as exc:  # pragma: no cover - network errors
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="request-error",
                    metadata={"landing": landing, "message": str(exc)},
                )
                continue

            if resp.status_code != 200:
                yield ResolverResult(
                    url=None,
                    event="error",
                    event_reason="http-error",
                    http_status=resp.status_code,
                    metadata={"landing": landing},
                )
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if meta and meta.get("content"):
                url = _absolute_url(landing, meta["content"].strip())
                yield ResolverResult(url=url, referer=landing, metadata={"pattern": "meta"})
                continue

            for link in soup.find_all("link"):
                rel = " ".join(link.get("rel") or []).lower()
                typ = (link.get("type") or "").lower()
                href = link.get("href") or ""
                if "alternate" in rel and "application/pdf" in typ and href:
                    url = _absolute_url(landing, href.strip())
                    yield ResolverResult(url=url, referer=landing, metadata={"pattern": "link"})
                    break


class ArxivResolver:
    name = "arxiv"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return bool(artifact.arxiv_id)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        arxiv_id = artifact.arxiv_id
        if not arxiv_id:
            return []
        arxiv_id = _strip_prefix(arxiv_id, "arxiv:")
        return [
            ResolverResult(
                url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                metadata={"identifier": arxiv_id},
            )
        ]


class PmcResolver:
    name = "pmc"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return bool(artifact.pmcid or artifact.pmid or artifact.doi)

    def _lookup_pmcids(
        self, session: requests.Session, identifiers: List[str], config: ResolverConfig
    ) -> List[str]:
        if not identifiers:
            return []
        try:
            resp = session.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={
                    "ids": ",".join(identifiers),
                    "format": "json",
                    "tool": "docs-to-kg",
                    "email": config.unpaywall_email or "",
                },
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        results: List[str] = []
        for record in data.get("records", []) or []:
            pmcid = record.get("pmcid")
            if pmcid:
                results.append(_normalize_pmcid(pmcid))
        return [pmc for pmc in results if pmc]

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        pmcids: List[str] = []
        if artifact.pmcid:
            pmcids.append(_normalize_pmcid(artifact.pmcid))
        identifiers = []
        if artifact.doi:
            identifiers.append(_normalize_doi(artifact.doi))
        if artifact.pmid:
            identifiers.append(artifact.pmid)
        if not pmcids:
            pmcids.extend(self._lookup_pmcids(session, identifiers, config))

        seen = set()
        for pmcid in pmcids:
            if not pmcid or pmcid in seen:
                continue
            seen.add(pmcid)
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
            try:
                resp = session.get(
                    oa_url,
                    timeout=config.get_timeout(self.name),
                )
            except requests.RequestException:
                continue
            if resp.status_code != 200:
                continue
            for match in re.finditer(r'href="([^"]+\.pdf)"', resp.text, flags=re.I):
                href = match.group(1)
                url = _absolute_url(oa_url, href)
                yield ResolverResult(
                    url=url,
                    metadata={"pmcid": pmcid, "source": "oa"},
                )
            yield ResolverResult(
                url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
                metadata={"pmcid": pmcid, "source": "pdf-fallback"},
            )


class EuropePmcResolver:
    name = "europe_pmc"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        doi = _normalize_doi(artifact.doi)
        if not doi:
            return []
        try:
            resp = session.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": f'DOI:"{doi}"', "format": "json", "pageSize": 3},
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        for result in (data.get("resultList", {}) or {}).get("result", []) or []:
            ft_list = (result or {}).get("fullTextUrlList", {}).get("fullTextUrl", [])
            for entry in ft_list:
                if not isinstance(entry, dict):
                    continue
                if (entry.get("documentStyle") or "").lower() == "pdf" and entry.get("url"):
                    yield ResolverResult(url=entry["url"], metadata={"source": "europepmc"})


class CoreResolver:
    name = "core"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return bool(config.core_api_key and artifact.doi)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        doi = _normalize_doi(artifact.doi)
        if not doi:
            return []
        headers = dict(config.polite_headers)
        headers["Authorization"] = f"Bearer {config.core_api_key}"
        try:
            resp = session.get(
                "https://api.core.ac.uk/v3/search/works",
                params={"q": f'doi:"{doi}"', "page": 1, "pageSize": 3},
                headers=headers,
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        for hit in data.get("results", []) or []:
            if not isinstance(hit, dict):
                continue
            url = hit.get("downloadUrl") or hit.get("pdfDownloadLink")
            if url:
                yield ResolverResult(url=url, metadata={"source": "core"})
            for entry in hit.get("fullTextLinks") or []:
                if isinstance(entry, dict):
                    href = entry.get("url") or entry.get("link")
                    if href and href.lower().endswith(".pdf"):
                        yield ResolverResult(url=href, metadata={"source": "core"})


class DoajResolver:
    name = "doaj"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        doi = _normalize_doi(artifact.doi)
        if not doi:
            return []
        headers = dict(config.polite_headers)
        if config.doaj_api_key:
            headers["X-API-KEY"] = config.doaj_api_key
        try:
            resp = session.get(
                "https://doaj.org/api/v2/search/articles/",
                params={"pageSize": 3, "q": f'doi:"{doi}"'},
                headers=headers,
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        for result in data.get("results", []) or []:
            bibjson = result.get("bibjson") or {}
            for link in bibjson.get("link", []) or []:
                if not isinstance(link, dict):
                    continue
                href = link.get("url")
                if not href:
                    continue
                if (link.get("type") or "").lower() == "fulltext" and href.lower().endswith(".pdf"):
                    yield ResolverResult(url=href, metadata={"source": "doaj"})


class SemanticScholarResolver:
    name = "semantic_scholar"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return artifact.doi is not None

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        doi = _normalize_doi(artifact.doi)
        if not doi:
            return []
        headers = dict(config.polite_headers)
        if config.semantic_scholar_api_key:
            headers["x-api-key"] = config.semantic_scholar_api_key
        try:
            resp = session.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}",
                params={"fields": "title,openAccessPdf"},
                headers=headers,
                timeout=config.get_timeout(self.name),
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        pdf = (data.get("openAccessPdf") or {}).get("url")
        if pdf:
            return [ResolverResult(url=pdf, metadata={"source": "semantic-scholar"})]
        return []


class WaybackResolver:
    name = "wayback"

    def is_enabled(self, config: ResolverConfig, artifact: "WorkArtifact") -> bool:
        return bool(artifact.failed_pdf_urls)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        for original in artifact.failed_pdf_urls:
            try:
                resp = session.get(
                    "https://archive.org/wayback/available",
                    params={"url": original},
                    timeout=config.get_timeout(self.name),
                )
            except requests.RequestException:
                continue
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except ValueError:
                continue
            closest = (data.get("archived_snapshots") or {}).get("closest") or {}
            if closest.get("available") and closest.get("url"):
                metadata = {"original": original}
                if closest.get("timestamp"):
                    metadata["timestamp"] = closest["timestamp"]
                yield ResolverResult(url=closest["url"], metadata=metadata)


def default_resolvers() -> List[Resolver]:
    return [
        UnpaywallResolver(),
        CrossrefResolver(),
        LandingPageResolver(),
        ArxivResolver(),
        PmcResolver(),
        EuropePmcResolver(),
        CoreResolver(),
        DoajResolver(),
        SemanticScholarResolver(),
        WaybackResolver(),
    ]


__all__ = [
    "AttemptRecord",
    "AttemptLogger",
    "DownloadOutcome",
    "PipelineResult",
    "Resolver",
    "ResolverConfig",
    "ResolverPipeline",
    "ResolverResult",
    "ResolverMetrics",
    "default_resolvers",
    "DEFAULT_RESOLVER_ORDER",
]

