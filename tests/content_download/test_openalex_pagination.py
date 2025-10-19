"""Unit tests covering OpenAlex pagination helpers."""

from __future__ import annotations

import typing
from types import TracebackType as _TypesTracebackType
from typing import Dict, Iterable, List, Optional

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
