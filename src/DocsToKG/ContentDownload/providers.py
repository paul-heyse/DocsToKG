# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.providers",
#   "purpose": "Source adapter interfaces for the DocsToKG content download pipeline.",
#   "sections": [
#     {
#       "id": "workprovider",
#       "name": "WorkProvider",
#       "anchor": "class-workprovider",
#       "kind": "class"
#     },
#     {
#       "id": "openalexworkprovider",
#       "name": "OpenAlexWorkProvider",
#       "anchor": "class-openalexworkprovider",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Source adapter interfaces for the DocsToKG content download pipeline.

Responsibilities
----------------
- Define the :class:`WorkProvider` protocol that abstracts over upstream
  catalogues and yields normalised :class:`~DocsToKG.ContentDownload.core.WorkArtifact`
  objects for the resolver pipeline.
- Ship a production-ready :class:`OpenAlexWorkProvider` implementation that
  wraps pyalex pagination, converts responses into artefact directories, and
  honours CLI limits (per-page, ``--max``) alongside retry/backoff settings.
- Provide extension points via ``ArtifactFactory`` so tests and future data
  sources (Crossref, arXiv, institutional APIs) can reuse the same orchestration
  without touching the download logic.

Design Notes
------------
- Providers remain streaming friendly—the iterator yields artifacts lazily so
  large result sets do not load into memory.
- The OpenAlex provider accepts either a live pyalex query or an iterable of
  pre-fetched works, simplifying unit tests and dry-run tooling.
- Pagination delegates to :func:`iterate_openalex`, inheriting equal-jitter
  retry behaviour and CLI-provided ``retry_after_cap`` settings.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Protocol, runtime_checkable

from pyalex import Works

from DocsToKG.ContentDownload.core import WorkArtifact

ArtifactFactory = Callable[[Dict[str, Any], Path, Path, Path], WorkArtifact]


@runtime_checkable
class WorkProvider(Protocol):
    """Protocol describing a source of :class:`WorkArtifact` instances."""

    name: str

    def iter_artifacts(self) -> Iterator[WorkArtifact]:
        """Yield normalized work artifacts ready for download processing."""

    def __iter__(self) -> Iterator[WorkArtifact]:
        return self.iter_artifacts()


class OpenAlexWorkProvider:
    """Produce :class:`WorkArtifact` instances from an OpenAlex Works query."""

    name = "openalex"

    def __init__(
        self,
        *,
        query: Optional[Works] = None,
        works_iterable: Optional[Iterable[Dict[str, Any]]] = None,
        artifact_factory: ArtifactFactory,
        pdf_dir: Path,
        html_dir: Path,
        xml_dir: Path,
        per_page: int = 200,
        max_results: Optional[int] = None,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
        retry_max_delay: Optional[float] = 75.0,
        retry_after_cap: Optional[float] = None,
        iterate_openalex_func: Optional[Callable[..., Iterable[Dict[str, Any]]]] = None,
    ) -> None:
        if query is None and works_iterable is None:
            raise ValueError("Provide an OpenAlex query or an iterable of works.")
        self._query = query
        self._works_iterable = works_iterable
        self._artifact_factory = artifact_factory
        self._pdf_dir = pdf_dir
        self._html_dir = html_dir
        self._xml_dir = xml_dir
        normalized_per_page = int(per_page)
        if not 1 <= normalized_per_page <= 200:
            raise ValueError("OpenAlex per_page must be between 1 and 200")
        self._per_page = normalized_per_page
        if max_results is None:
            self._max_results = None
        elif max_results < 0:
            self._max_results = None
        else:
            self._max_results = max_results
        self._retry_attempts = max(0, int(retry_attempts))
        self._retry_backoff = max(0.0, float(retry_backoff))
        self._retry_max_delay = float(retry_max_delay) if retry_max_delay is not None else None
        self._retry_after_cap = float(retry_after_cap) if retry_after_cap is not None else None
        if iterate_openalex_func is None:
            from DocsToKG.ContentDownload.runner import iterate_openalex

            iterate_openalex_func = iterate_openalex
        self._iterate_openalex_func = iterate_openalex_func

    def iter_artifacts(self) -> Iterator[WorkArtifact]:
        """Iterate over OpenAlex works and yield normalized :class:`WorkArtifact` objects."""

        if self._max_results == 0:
            return

        yielded = 0
        for item in self._iter_source():
            if isinstance(item, Mapping):
                works_iterable = (item,)
            elif isinstance(item, IterableABC):
                works_iterable = item
            else:
                raise TypeError(
                    f"OpenAlexWorkProvider expected a mapping or iterable of mappings, got {type(item)!r}"
                )

            for work in works_iterable:
                if not isinstance(work, Mapping):
                    raise TypeError(
                        "OpenAlexWorkProvider expected works to be mapping-like objects."
                    )
                artifact = self._artifact_factory(
                    work, self._pdf_dir, self._html_dir, self._xml_dir
                )
                yield artifact
                yielded += 1
                if self._max_results is not None and yielded >= self._max_results:
                    return

    def __iter__(self) -> Iterator[WorkArtifact]:
        return self.iter_artifacts()

    def _iterate_openalex(self) -> Iterable[Dict[str, Any]]:
        assert self._query is not None  # For mypy; guarded by caller.
        yield from self._iterate_openalex_func(
            self._query,
            per_page=self._per_page,
            max_results=self._max_results,
            retry_attempts=self._retry_attempts,
            retry_backoff=self._retry_backoff,
            retry_max_delay=self._retry_max_delay,
            retry_after_cap=self._retry_after_cap,
        )

    def _iter_source(self) -> Iterator[Dict[str, Any]]:
        if self._works_iterable is not None:
            yield from iter(self._works_iterable)
            return
        if self._query is None:
            raise RuntimeError(
                "OpenAlexWorkProvider requires a query when no iterable is supplied."
            )
        yield from self._iterate_openalex()


__all__ = ("WorkProvider", "OpenAlexWorkProvider")
