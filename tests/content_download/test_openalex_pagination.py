"""Unit tests covering OpenAlex pagination helpers."""

from __future__ import annotations

import typing
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import TracebackType as _TypesTracebackType
from typing import Dict, Iterable, List, Optional

import pytest
import requests

if not hasattr(typing, "TracebackType"):
    typing.TracebackType = _TypesTracebackType  # type: ignore[attr-defined]

from DocsToKG.ContentDownload.providers import OpenAlexWorkProvider
from DocsToKG.ContentDownload.runner import iterate_openalex


class RecordingWorks:
    """Stub Works query capturing pagination parameters and page consumption."""

    def __init__(self, pages: Iterable[Iterable[Dict[str, object]]]):
        self._pages: List[List[Dict[str, object]]] = [list(page) for page in pages]
        self.paginate_calls: List[Dict[str, Optional[int]]] = []
        self.pages_iterated = 0

    def paginate(
        self, per_page: int, n_max: Optional[int] = None
    ) -> Iterable[Iterable[Dict[str, object]]]:
        self.paginate_calls.append({"per_page": per_page, "n_max": n_max})

        pages = [list(page) for page in self._pages]
        if n_max is not None:
            remaining = n_max
            limited_pages: List[List[Dict[str, object]]] = []
            for page in pages:
                if remaining <= 0:
                    break
                limited_pages.append(page)
                remaining -= len(page)
            pages = limited_pages

        def _iterator() -> Iterable[Iterable[Dict[str, object]]]:
            for page in pages:
                self.pages_iterated += 1
                yield list(page)

        return _iterator()


class FlakyRecordingWorks(RecordingWorks):
    """Recording works stub that raises HTTP errors before yielding pages."""

    def __init__(self, pages: Iterable[Iterable[Dict[str, object]]], failures: int):
        super().__init__(pages)
        self.failures_remaining = failures
        self.failures_triggered = 0
        self.next_calls = 0

    def paginate(
        self, per_page: int, n_max: Optional[int] = None
    ) -> Iterable[Iterable[Dict[str, object]]]:
        base_iterable = super().paginate(per_page, n_max)
        iterator = iter(base_iterable)
        works = self

        class _FlakyIterator:
            def __iter__(self) -> "_FlakyIterator":
                return self

            def __next__(self) -> Iterable[Dict[str, object]]:
                works.next_calls += 1
                if works.failures_remaining > 0:
                    works.failures_remaining -= 1
                    works.failures_triggered += 1
                    error = requests.HTTPError("boom")
                    raise error
                return next(iterator)

        return _FlakyIterator()


def _sample_pages() -> List[List[Dict[str, object]]]:
    return [
        [{"id": "W1"}, {"id": "W2"}],
        [{"id": "W3"}, {"id": "W4"}],
        [{"id": "W5"}],
    ]


def test_iterate_openalex_passes_max_results_to_pagination() -> None:
    works = RecordingWorks(_sample_pages())

    results = list(iterate_openalex(works, per_page=2, max_results=3))

    assert [item["id"] for item in results] == ["W1", "W2", "W3"]
    assert works.paginate_calls == [{"per_page": 2, "n_max": 3}]
    assert works.pages_iterated == 2


def test_provider_iterates_minimal_pages_with_max(tmp_path) -> None:
    works = RecordingWorks(_sample_pages())

    provider = OpenAlexWorkProvider(
        query=works,
        artifact_factory=lambda work, *_: work["id"],
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
        per_page=2,
        max_results=3,
    )

    artifacts = list(provider.iter_artifacts())

    assert artifacts == ["W1", "W2", "W3"]
    assert works.paginate_calls == [{"per_page": 2, "n_max": 3}]
    assert works.pages_iterated == 2


def test_provider_with_query_only_iterates_all_results(tmp_path) -> None:
    works = RecordingWorks(_sample_pages())

    provider = OpenAlexWorkProvider(
        query=works,
        artifact_factory=lambda work, *_: work["id"],
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
    )

    artifacts = list(provider.iter_artifacts())

    assert artifacts == ["W1", "W2", "W3", "W4", "W5"]
    assert works.paginate_calls == [{"per_page": 200, "n_max": None}]
    assert works.pages_iterated == 3


def test_provider_query_only_retries_failed_pages(tmp_path) -> None:
    works = FlakyRecordingWorks(_sample_pages(), failures=1)

    provider = OpenAlexWorkProvider(
        query=works,
        artifact_factory=lambda work, *_: work["id"],
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
        per_page=2,
        max_results=3,
        retry_attempts=2,
        retry_backoff=0.0,
        retry_max_delay=0.0,
    )

    artifacts = list(provider.iter_artifacts())

    assert artifacts == ["W1", "W2", "W3"]
    assert works.failures_triggered == 1
    assert works.paginate_calls == [{"per_page": 2, "n_max": 3}]
    assert works.pages_iterated == 2
    assert works.next_calls == works.pages_iterated + works.failures_triggered


def test_provider_rejects_per_page_above_openalex_limit(tmp_path) -> None:
    works = RecordingWorks(_sample_pages())

    with pytest.raises(ValueError, match="per_page must be between 1 and 200"):
        OpenAlexWorkProvider(
            query=works,
            artifact_factory=lambda work, *_: work["id"],
            pdf_dir=tmp_path / "pdf",
            html_dir=tmp_path / "html",
            xml_dir=tmp_path / "xml",
            per_page=500,
        )


def test_provider_allows_retry_max_delay_none(tmp_path) -> None:
    works = RecordingWorks(_sample_pages())
    captured_kwargs: Dict[str, object] = {}

    def _iter_openalex(
        query: RecordingWorks,
        *,
        per_page: int,
        max_results: Optional[int],
        retry_attempts: int,
        retry_backoff: float,
        retry_max_delay: Optional[float],
        retry_after_cap: Optional[float],
    ) -> Iterable[Dict[str, object]]:
        captured_kwargs.update(
            {
                "query": query,
                "per_page": per_page,
                "max_results": max_results,
                "retry_attempts": retry_attempts,
                "retry_backoff": retry_backoff,
                "retry_max_delay": retry_max_delay,
                "retry_after_cap": retry_after_cap,
            }
        )
        pager = query.paginate(per_page=per_page, n_max=max_results)
        yield from pager

    provider = OpenAlexWorkProvider(
        query=works,
        artifact_factory=lambda work, *_: work["id"],
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
        per_page=2,
        max_results=3,
        retry_max_delay=None,
        iterate_openalex_func=_iter_openalex,
    )

    artifacts = list(provider.iter_artifacts())

    assert artifacts == ["W1", "W2", "W3"]
    assert captured_kwargs["query"] is works
    assert captured_kwargs["retry_max_delay"] is None
    assert captured_kwargs["per_page"] == 2
    assert captured_kwargs["max_results"] == 3
    assert works.paginate_calls == [{"per_page": 2, "n_max": 3}]


@pytest.mark.parametrize("retry_after_format", ["numeric", "http-date"])
def test_iterate_openalex_retries_with_retry_after_headers(
    monkeypatch: pytest.MonkeyPatch, retry_after_format: str
) -> None:
    class _RetryAfterPager:
        def __init__(self, pages: Iterable[Iterable[Dict[str, object]]], header: str) -> None:
            self._pages = [list(page) for page in pages]
            self._header = header
            self._index = 0
            self.calls = 0
            self.failures = 0

        def __iter__(self) -> "_RetryAfterPager":
            return self

        def __next__(self) -> Iterable[Dict[str, object]]:
            self.calls += 1
            if self.failures == 0:
                self.failures += 1
                response = requests.Response()
                response.status_code = 429
                response.headers["Retry-After"] = self._header
                error = requests.HTTPError("rate limited")
                error.response = response
                raise error
            if self._index >= len(self._pages):
                raise StopIteration
            page = self._pages[self._index]
            self._index += 1
            return list(page)

    class _RetryAfterWorks:
        def __init__(self, header: str) -> None:
            self._header = header
            self.paginate_calls: List[Dict[str, Optional[int]]] = []
            self.pages_iterated = 0
            self.pager: Optional[_RetryAfterPager] = None

        def paginate(
            self, per_page: int, n_max: Optional[int] = None
        ) -> Iterable[Iterable[Dict[str, object]]]:
            self.paginate_calls.append({"per_page": per_page, "n_max": n_max})
            pager = _RetryAfterPager([[{"id": "W1"}]], self._header)
            self.pager = pager
            return pager

    if retry_after_format == "numeric":
        header_value = "120"
    else:
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        header_value = format_datetime(future)

    works = _RetryAfterWorks(header_value)
    sleeps: List[float] = []

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.runner.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.runner.random.uniform",
        lambda _low, _high: 0.0,
    )
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.runner.base_backoff",
        2.0,
        raising=False,
    )

    results = list(
        iterate_openalex(
            works,
            per_page=1,
            max_results=None,
            retry_attempts=3,
            retry_backoff=2.0,
            retry_max_delay=3.0,
            retry_after_cap=3.0,
        )
    )

    assert [item["id"] for item in results] == ["W1"]
    assert works.paginate_calls == [{"per_page": 1, "n_max": None}]
    assert works.pager is not None
    assert works.pager.failures == 1
    assert len(sleeps) == works.pager.failures == 1
    assert sleeps == [pytest.approx(3.0)]
