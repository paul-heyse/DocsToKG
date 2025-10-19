"""
Source adapter interfaces for the DocsToKG content download pipeline.

This module formalises the boundary between the CLI and upstream catalogues by
exposing a :class:`WorkProvider` protocol. Providers are responsible for
yielding :class:`~DocsToKG.ContentDownload.core.WorkArtifact` instances that
downstream resolver components can process without needing to know which
catalogue supplied the metadata. Today we ship an OpenAlex implementation, but
the protocol enables future adapters (e.g., Crossref, arXiv) to plug in without
modifying the pipeline.
"""

from __future__ import annotations

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
    ) -> None:
        if query is None and works_iterable is None:
            raise ValueError("Provide an OpenAlex query or an iterable of works.")
        self._query = query
        self._works_iterable = works_iterable
        self._artifact_factory = artifact_factory
        self._pdf_dir = pdf_dir
        self._html_dir = html_dir
        self._xml_dir = xml_dir
        self._per_page = max(1, per_page)
        if max_results is None:
            self._max_results = None
        elif max_results < 0:
            self._max_results = None
        else:
            self._max_results = max_results

    def iter_artifacts(self) -> Iterator[WorkArtifact]:
        """Iterate over OpenAlex works and yield normalized :class:`WorkArtifact` objects."""

        if self._max_results == 0:
            return

        yielded = 0
        for work in self._iter_source():
            artifact = self._artifact_factory(work, self._pdf_dir, self._html_dir, self._xml_dir)
            yield artifact
            yielded += 1
            if self._max_results is not None and yielded >= self._max_results:
                break

    def __iter__(self) -> Iterator[WorkArtifact]:
        return self.iter_artifacts()

    def _iterate_openalex(self) -> Iterable[Dict[str, Any]]:
        pager = self._query.paginate(
            per_page=self._per_page,
            n_max=self._max_results if self._max_results is not None else None,
        )
        retrieved = 0
        for page in pager:
            for work in page:
                if self._max_results is not None and retrieved >= self._max_results:
                    return
                yield work
                retrieved += 1

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
