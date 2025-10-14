#!/usr/bin/env python3
"""Download PDFs for OpenAlex works with a configurable resolver stack."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

import pyalex
from pyalex import Topics, Works, config as oa_config

from DocsToKG.ContentDownload.resolvers import (
    AttemptRecord,
    DownloadOutcome,
    PipelineResult,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    default_resolvers,
)


MAX_SNIFF_BYTES = 64 * 1024
SUCCESS_STATUSES = {"pdf", "pdf_unknown"}

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


def slugify(text: str, keep: int = 80) -> str:
    text = re.sub(r"[^\w\s\-]+", "", text or "")
    text = re.sub(r"\s+", "_", text.strip())
    return text[:keep] or "untitled"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def classify_payload(head_bytes: bytes, content_type: str, url: str) -> Optional[str]:
    """Return 'pdf', 'html', or None if undecided."""

    ctype = (content_type or "").lower()
    stripped = head_bytes.lstrip() if head_bytes else b""
    prefix = stripped[:64].lower()

    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        return "html"
    if prefix.startswith(b"<head") or prefix.startswith(b"<body"):
        return "html"

    if stripped.startswith(b"%PDF") or b"%PDF" in head_bytes[:2048]:
        return "pdf"

    if "html" in ctype:
        return "html"
    if "pdf" in ctype:
        return "pdf"

    if url.lower().endswith(".pdf"):
        return "pdf"

    return None


@dataclass
class WorkArtifact:
    work_id: str
    title: str
    publication_year: Optional[int]
    doi: Optional[str]
    pmid: Optional[str]
    pmcid: Optional[str]
    arxiv_id: Optional[str]
    landing_urls: List[str]
    pdf_urls: List[str]
    open_access_url: Optional[str]
    source_display_names: List[str]
    base_stem: str
    pdf_dir: Path
    html_dir: Path
    failed_pdf_urls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.namespaces: Dict[str, Path] = {"pdf": self.pdf_dir, "html": self.html_dir}


def _strip_prefix(value: Optional[str], prefix: str) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if value.lower().startswith(prefix.lower()):
        return value[len(prefix) :]
    return value


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[16:]
    return doi.strip()


def _normalize_pmid(pmid: Optional[str]) -> Optional[str]:
    if not pmid:
        return None
    pmid = pmid.strip()
    match = re.search(r"(\d+)", pmid)
    return match.group(1) if match else None


def _normalize_pmcid(pmcid: Optional[str]) -> Optional[str]:
    if not pmcid:
        return None
    pmcid = pmcid.strip()
    match = re.search(r"PMC?(\d+)", pmcid, flags=re.I)
    if match:
        return f"PMC{match.group(1)}"
    return None


def _normalize_arxiv(arxiv_id: Optional[str]) -> Optional[str]:
    if not arxiv_id:
        return None
    arxiv_id = _strip_prefix(arxiv_id, "arxiv:") or arxiv_id
    arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "")
    return arxiv_id.strip()


def _collect_location_urls(work: Dict[str, Any]) -> Dict[str, List[str]]:
    landing_urls: List[str] = []
    pdf_urls: List[str] = []
    sources: List[str] = []

    def append_location(loc: Optional[Dict[str, Any]]) -> None:
        if not isinstance(loc, dict):
            return
        landing = loc.get("landing_page_url")
        pdf = loc.get("pdf_url")
        source = (loc.get("source") or {}).get("display_name")
        if landing:
            landing_urls.append(landing)
        if pdf:
            pdf_urls.append(pdf)
        if source:
            sources.append(source)

    append_location(work.get("best_oa_location"))
    append_location(work.get("primary_location"))
    for loc in work.get("locations", []) or []:
        append_location(loc)

    oa_url = ((work.get("open_access") or {}).get("oa_url") or None)
    if oa_url:
        pdf_urls.append(oa_url)

    # De-duplicate while preserving order
    def dedupe(items: List[str]) -> List[str]:
        seen = set()
        uniq = []
        for item in items:
            if item and item not in seen:
                uniq.append(item)
                seen.add(item)
        return uniq

    return {
        "landing": dedupe(landing_urls),
        "pdf": dedupe(pdf_urls),
        "sources": dedupe(sources),
    }


def build_query(args: argparse.Namespace) -> Works:
    query = Works()
    if args.topic_id:
        query = query.filter(topics={"id": args.topic_id})
    else:
        query = query.search(args.topic)

    query = query.filter(publication_year=f"{args.year_start}-{args.year_end}")
    if args.oa_only:
        query = query.filter(open_access={"is_oa": True})

    query = query.select(
        [
            "id",
            "title",
            "publication_year",
            "ids",
            "open_access",
            "best_oa_location",
            "primary_location",
            "locations",
        ]
    )

    query = query.sort(publication_date="desc")
    return query


def resolve_topic_id_if_needed(topic_text: Optional[str]) -> Optional[str]:
    if not topic_text:
        return None
    hits = Topics().search(topic_text).get()
    if not hits:
        return None
    return hits[0]["id"]


def create_artifact(work: Dict[str, Any], pdf_dir: Path, html_dir: Path) -> WorkArtifact:
    work_id = (work.get("id") or "W").rsplit("/", 1)[-1]
    title = work.get("title") or work.get("display_name") or ""
    year = work.get("publication_year")
    ids = work.get("ids") or {}
    doi = _normalize_doi(ids.get("doi"))
    pmid = _normalize_pmid(ids.get("pmid"))
    pmcid = _normalize_pmcid(ids.get("pmcid"))
    arxiv_id = _normalize_arxiv(ids.get("arxiv"))

    locations = _collect_location_urls(work)
    landing_urls = locations["landing"]
    pdf_urls = locations["pdf"]
    sources = locations["sources"]
    oa_url = (work.get("open_access") or {}).get("oa_url")

    base = f"{year or 'noyear'}__{slugify(title)}__{work_id}".strip("_")

    artifact = WorkArtifact(
        work_id=work_id,
        title=title,
        publication_year=year,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        arxiv_id=arxiv_id,
        landing_urls=landing_urls,
        pdf_urls=pdf_urls,
        open_access_url=oa_url,
        source_display_names=sources,
        base_stem=base,
        pdf_dir=pdf_dir,
        html_dir=html_dir,
        metadata={"openalex_id": work.get("id")},
    )
    return artifact


def download_candidate(
    session: requests.Session,
    artifact: WorkArtifact,
    url: str,
    referer: Optional[str],
    timeout: float,
) -> DownloadOutcome:
    headers: Dict[str, str] = {}
    if referer:
        headers["Referer"] = referer
    start = time.monotonic()
    content_type_hint = ""
    try:
        try:
            head = session.head(url, allow_redirects=True, timeout=timeout, headers=headers)
            content_type_hint = head.headers.get("Content-Type", "")
        except requests.RequestException:
            pass

        with session.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as response:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if response.status_code != 200:
                return DownloadOutcome(
                    classification="http_error",
                    path=None,
                    http_status=response.status_code,
                    content_type=response.headers.get("Content-Type"),
                    elapsed_ms=elapsed_ms,
                )

            content_type = response.headers.get("Content-Type") or content_type_hint
            sniff_buffer = bytearray()
            detected: Optional[str] = None
            flagged_unknown = False
            dest_path: Optional[Path] = None
            handle = None

            try:
                for chunk in response.iter_content(chunk_size=1 << 15):
                    if not chunk:
                        continue
                    if detected is None:
                        sniff_buffer.extend(chunk)
                        detected = classify_payload(bytes(sniff_buffer), content_type, url)
                        if detected is None and len(sniff_buffer) >= MAX_SNIFF_BYTES:
                            detected = "pdf"
                            flagged_unknown = True

                        if detected is not None:
                            if detected == "html":
                                dest_path = artifact.html_dir / f"{artifact.base_stem}.html"
                                ensure_dir(dest_path.parent)
                            else:
                                dest_path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
                                ensure_dir(dest_path.parent)
                            handle = open(dest_path, "wb")
                            handle.write(sniff_buffer)
                            sniff_buffer.clear()
                            continue
                    else:
                        assert handle is not None
                        handle.write(chunk)

                if detected is None:
                    return DownloadOutcome(
                        classification="empty",
                        path=None,
                        http_status=response.status_code,
                        content_type=content_type,
                        elapsed_ms=elapsed_ms,
                    )
            finally:
                if handle is not None:
                    handle.close()

            path_str = str(dest_path) if dest_path else None
            classification = detected or "pdf"
            if flagged_unknown and classification == "pdf":
                classification = "pdf_unknown"
            return DownloadOutcome(
                classification=classification,
                path=path_str,
                http_status=response.status_code,
                content_type=content_type,
                elapsed_ms=elapsed_ms,
            )
    except requests.RequestException as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return DownloadOutcome(
            classification="request_error",
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=elapsed_ms,
            error=str(exc),
        )


class CsvAttemptLogger:
    HEADER = [
        "timestamp",
        "work_id",
        "resolver_source",
        "resolver_order",
        "attempt_url",
        "status",
        "http_status",
        "content_type",
        "elapsed_ms",
        "reason",
        "metadata",
    ]

    def __init__(self, path: Optional[Path]) -> None:
        self._path = path
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        if path:
            ensure_dir(path.parent)
            exists = path.exists()
            self._file = path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
            if not exists:
                self._writer.writeheader()

    def log(self, record: AttemptRecord) -> None:
        if not self._writer:
            return
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "work_id": record.work_id,
            "resolver_source": record.resolver_name,
            "resolver_order": record.resolver_order,
            "attempt_url": record.url,
            "status": record.status,
            "http_status": record.http_status,
            "content_type": record.content_type,
            "elapsed_ms": record.elapsed_ms,
            "reason": record.reason,
            "metadata": json.dumps(record.metadata, sort_keys=True) if record.metadata else "",
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


def read_resolver_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return json.loads(text)
    else:
        data = yaml.safe_load(text)
        return data or {}


def apply_config_overrides(
    config: ResolverConfig,
    data: Dict[str, Any],
    resolver_names: Sequence[str],
) -> None:
    for field_name in (
        "resolver_order",
        "resolver_toggles",
        "max_attempts_per_work",
        "timeout",
        "sleep_jitter",
        "polite_headers",
        "unpaywall_email",
        "core_api_key",
        "semantic_scholar_api_key",
        "doaj_api_key",
        "resolver_timeouts",
        "resolver_rate_limits",
    ):
        if field_name in data and data[field_name] is not None:
            setattr(config, field_name, data[field_name])

    for name in resolver_names:
        config.resolver_toggles.setdefault(name, True)


def load_resolver_config(args: argparse.Namespace, resolver_names: Sequence[str]) -> ResolverConfig:
    config = ResolverConfig()
    if args.resolver_config:
        config_data = read_resolver_config(Path(args.resolver_config))
        apply_config_overrides(config, config_data, resolver_names)

    # Environment fallbacks
    config.unpaywall_email = (
        args.unpaywall_email
        or config.unpaywall_email
        or os.getenv("UNPAYWALL_EMAIL")
        or args.mailto
    )
    config.core_api_key = args.core_api_key or config.core_api_key or os.getenv("CORE_API_KEY")
    config.semantic_scholar_api_key = (
        args.semantic_scholar_api_key or config.semantic_scholar_api_key or os.getenv("S2_API_KEY")
    )
    config.doaj_api_key = args.doaj_api_key or config.doaj_api_key or os.getenv("DOAJ_API_KEY")

    if args.max_resolver_attempts:
        config.max_attempts_per_work = args.max_resolver_attempts
    if args.resolver_timeout:
        config.timeout = args.resolver_timeout

    for name in resolver_names:
        config.resolver_toggles.setdefault(name, True)

    for disabled in args.disable_resolver or []:
        config.resolver_toggles[disabled] = False

    # Polite headers include mailto when available
    headers = dict(config.polite_headers)
    if args.mailto:
        headers.setdefault("mailto", args.mailto)
    headers.setdefault("User-Agent", "DocsToKG-OpenAlexDownloader/1.0")
    config.polite_headers = headers

    # Apply resolver rate defaults (Unpaywall recommended 1 QPS)
    config.resolver_rate_limits.setdefault("unpaywall", 1.0)

    return config


def iterate_openalex(query: Works, per_page: int, max_results: Optional[int]) -> Iterable[Dict[str, Any]]:
    pager = query.paginate(per_page=per_page, n_max=None)
    retrieved = 0
    for page in pager:
        for work in page:
            yield work
            retrieved += 1
            if max_results and retrieved >= max_results:
                return


def attempt_openalex_candidates(
    session: requests.Session,
    artifact: WorkArtifact,
    logger: CsvAttemptLogger,
    metrics: ResolverMetrics,
) -> Optional[DownloadOutcome]:
    candidates = list(artifact.pdf_urls)
    if artifact.open_access_url:
        candidates.append(artifact.open_access_url)

    seen = set()
    html_paths: List[str] = []
    for url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        outcome = download_candidate(session, artifact, url, referer=None, timeout=30.0)
        logger.log(
            AttemptRecord(
                work_id=artifact.work_id,
                resolver_name="openalex",
                resolver_order=0,
                url=url,
                status=outcome.classification,
                http_status=outcome.http_status,
                content_type=outcome.content_type,
                elapsed_ms=outcome.elapsed_ms,
                reason=outcome.error,
                metadata={"source": "openalex"},
            )
        )
        metrics.record_attempt("openalex", outcome)
        if outcome.classification == "html" and outcome.path:
            html_paths.append(outcome.path)
        if outcome.is_pdf:
            return outcome
        if outcome.classification not in SUCCESS_STATUSES and url:
            artifact.failed_pdf_urls.append(url)
    if not seen:
        metrics.record_skip("openalex", "no-candidates")
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Download OpenAlex PDFs for a topic and year range with resolvers.",
    )
    parser.add_argument("--topic", type=str, help="Free-text topic search.")
    parser.add_argument(
        "--topic-id",
        type=str,
        help="OpenAlex Topic ID (e.g., https://openalex.org/T12345). Overrides --topic.",
    )
    parser.add_argument("--year-start", type=int, required=True, help="Start year (inclusive).")
    parser.add_argument("--year-end", type=int, required=True, help="End year (inclusive).")
    parser.add_argument("--out", type=Path, default=Path("./pdfs"), help="Output folder for PDFs.")
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Folder for HTML responses (default: sibling 'HTML').",
    )
    parser.add_argument("--mailto", type=str, default=None, help="Email for the OpenAlex polite pool.")
    parser.add_argument("--per-page", type=int, default=200, help="Results per page (1-200).")
    parser.add_argument("--max", type=int, default=None, help="Maximum works to process.")
    parser.add_argument("--oa-only", action="store_true", help="Only consider open-access works.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between works.")

    # Resolver configuration
    parser.add_argument("--resolver-config", type=str, default=None, help="Path to resolver config (YAML/JSON).")
    parser.add_argument("--unpaywall-email", type=str, default=None, help="Override Unpaywall email credential.")
    parser.add_argument("--core-api-key", type=str, default=None, help="CORE API key override.")
    parser.add_argument(
        "--semantic-scholar-api-key",
        type=str,
        default=None,
        help="Semantic Scholar Graph API key override.",
    )
    parser.add_argument("--doaj-api-key", type=str, default=None, help="DOAJ API key override.")
    parser.add_argument(
        "--disable-resolver",
        action="append",
        default=[],
        help="Disable a resolver by name (can be repeated).",
    )
    parser.add_argument(
        "--max-resolver-attempts",
        type=int,
        default=None,
        help="Maximum resolver attempts per work.",
    )
    parser.add_argument(
        "--resolver-timeout",
        type=float,
        default=None,
        help="Default timeout (seconds) for resolver HTTP requests.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV log file capturing resolver attempts.",
    )

    args = parser.parse_args()

    if not args.topic and not args.topic_id:
        parser.error("Provide --topic or --topic-id.")

    if args.mailto:
        oa_config.email = args.mailto

    topic_id = args.topic_id
    if not topic_id and args.topic:
        topic_id = None  # placeholder for optional topic resolution

    query = build_query(
        argparse.Namespace(
            topic=args.topic,
            topic_id=topic_id,
            year_start=args.year_start,
            year_end=args.year_end,
            oa_only=args.oa_only,
        )
    )

    pdf_dir = args.out
    html_dir = args.html_out or (pdf_dir.parent / "HTML")

    ensure_dir(pdf_dir)
    ensure_dir(html_dir)

    # Resolver setup
    resolvers = default_resolvers()
    resolver_names = [resolver.name for resolver in resolvers]
    config = load_resolver_config(args, resolver_names)

    logger = CsvAttemptLogger(args.log_csv)
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline(
        resolvers=resolvers,
        config=config,
        download_func=download_candidate,
        logger=logger,
        metrics=metrics,
    )

    session = requests.Session()
    session.headers.update(config.polite_headers)

    processed = 0
    saved = 0
    html_only = 0
    try:
        for work in iterate_openalex(query, per_page=args.per_page, max_results=args.max):
            processed += 1
            artifact = create_artifact(work, pdf_dir=pdf_dir, html_dir=html_dir)

            if (artifact.pdf_dir / f"{artifact.base_stem}.pdf").exists():
                print(f"[skip] {artifact.base_stem}.pdf (exists)")
                continue

            result = attempt_openalex_candidates(session, artifact, logger, metrics)
            if result and result.is_pdf:
                saved += 1
                print(f"[ok]   {artifact.base_stem}.pdf ← openalex")
                time.sleep(args.sleep)
                continue

            pipeline_result: PipelineResult = pipeline.run(session, artifact)
            if pipeline_result.success and pipeline_result.outcome:
                saved += 1
                print(
                    f"[ok]   {artifact.base_stem}.pdf ← {pipeline_result.resolver_name}"
                )
            else:
                if pipeline_result.html_paths:
                    html_only += 1
                    print(
                        f"[miss] {artifact.work_id} — HTML saved ({Path(pipeline_result.html_paths[-1]).name})"
                    )
                else:
                    reason = pipeline_result.reason or "no-resolver-success"
                    print(f"[miss] {artifact.work_id} — {reason}")
                logger.log(
                    AttemptRecord(
                        work_id=artifact.work_id,
                        resolver_name="final",
                        resolver_order=None,
                        url=None,
                        status="miss",
                        http_status=None,
                        content_type=None,
                        elapsed_ms=None,
                        reason=pipeline_result.reason or "no-resolver-success",
                    )
                )

            time.sleep(args.sleep)
    finally:
        logger.close()

    summary = metrics.summary()
    print(
        f"\nDone. Processed {processed} works, saved {saved} PDFs, HTML-only {html_only}."
    )
    print("Resolver summary:")
    for key, values in summary.items():
        print(f"  {key}: {values}")

    LOGGER.info(
        "resolver_run_summary %s",
        json.dumps(
            {
                "processed": processed,
                "saved": saved,
                "html_only": html_only,
                "summary": summary,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()

