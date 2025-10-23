"""In-memory analogue of the OpenSearch integration used by hybrid search.

The real deployment writes chunks to OpenSearch for BM25/SPLADE retrieval. Tests
and local development environments cannot depend on an actual cluster, so this
module provides feature-complete stand-ins:

- ``OpenSearchIndexTemplate`` and ``OpenSearchSchemaManager`` describe and
  validate namespace-specific index templates. The schema manager mirrors the
  subset of the OpenSearch API that the README’s “Index provisioning” section
  references, ensuring token-count and SPLADE mappings exist before ingesting
  data.
- ``OpenSearchSimulator`` implements the
  :class:`~DocsToKG.HybridSearch.interfaces.LexicalIndex` protocol entirely in
  memory. It maintains DF statistics for Okapi BM25 (``search_bm25_true``),
  exposes heuristic BM25/SPLADE scorers that align with ingestion’s synthetic
  feature generation, supports cursor-based pagination, and provides deterministic
  highlighting. The scoring routines intentionally mirror the mathematics that
  production OpenSearch applies so that fusion tests exercise realistic values.
- ``matches_filters`` performs the metadata filtering logic shared between the
  simulator and ``HybridSearchService`` when narrowing searches by namespace,
  ACLs, or arbitrary document tags.

Together these helpers let the ingestion pipeline, fusion logic, and diagnostics
run without external services while still exercising the same interfaces,
metadata layouts, and scoring curves expected in production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from ..config import ChunkingConfig
from ..interfaces import LexicalIndex
from ..types import ChunkPayload

__all__ = (
    "OpenSearchIndexTemplate",
    "OpenSearchSchemaManager",
    "OpenSearchSimulator",
    "matches_filters",
)


@dataclass(slots=True)
class OpenSearchIndexTemplate:
    """Representation of a namespace-specific OpenSearch template.

    Attributes:
        name: Human-readable name of the template (e.g. ``"hybrid-chunks-demo"``).
        namespace: DocsToKG namespace that owns the template.
        body: Raw OpenSearch template payload including settings and mappings.
        chunking: Chunking configuration applied to documents within the namespace.

    Examples:
        >>> template = OpenSearchIndexTemplate(
        ...     name="hybrid-chunks-demo",
        ...     namespace="demo",
        ...     body={"settings": {}, "mappings": {}},
        ...     chunking=ChunkingConfig(),
        ... )
        >>> template.asdict()["namespace"]
        'demo'
    """

    name: str
    namespace: str
    body: Mapping[str, object]
    chunking: ChunkingConfig

    def asdict(self) -> Dict[str, object]:
        """Return a dictionary representation of the template.

        Args:
            None

        Returns:
            Dictionary containing serializable template fields.
        """
        return {
            "name": self.name,
            "namespace": self.namespace,
            "body": dict(self.body),
            "chunking": {
                "max_tokens": self.chunking.max_tokens,
                "overlap": self.chunking.overlap,
            },
        }


class OpenSearchSchemaManager:
    """Manage simulated OpenSearch index templates for tests.

    The manager mirrors the minimal subset of the OpenSearch API required by the
    validation harness. It stores templates keyed by namespace and exposes helper
    methods to create or retrieve definitions.

    Attributes:
        _templates: Internal mapping of namespace names to template instances.

    Examples:
        >>> manager = OpenSearchSchemaManager()
        >>> template = manager.bootstrap_template("demo")
        >>> manager.get_template("demo") is template
        True
    """

    def __init__(self) -> None:
        self._templates: MutableMapping[str, OpenSearchIndexTemplate] = {}

    def bootstrap_template(
        self,
        namespace: str,
        chunking: Optional[ChunkingConfig] = None,
    ) -> OpenSearchIndexTemplate:
        """Create and cache a template for ``namespace``.

        Args:
            namespace: Namespace identifier that will own the template.
            chunking: Optional chunking configuration. Defaults to ``ChunkingConfig()``.

        Returns:
            Registered template instance for ``namespace``.
        """
        chunking_config = chunking or ChunkingConfig()
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                },
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                        }
                    }
                },
            },
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "namespace": {"type": "keyword"},
                    "vector_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "splade": {"type": "rank_features"},
                    "token_count": {"type": "integer"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "author": {"type": "keyword"},
                            "tags": {"type": "keyword"},
                            "published_at": {"type": "date"},
                            "acl": {"type": "keyword"},
                        },
                    },
                },
            },
        }
        template = OpenSearchIndexTemplate(
            name=f"hybrid-chunks-{namespace}",
            namespace=namespace,
            body=body,
            chunking=chunking_config,
        )
        self._templates[namespace] = template
        self.validate_namespace_schema(namespace)
        return template

    def get_template(self, namespace: str) -> Optional[OpenSearchIndexTemplate]:
        """Return the template registered for ``namespace`` if it exists.

        Args:
            namespace: Namespace identifier used during registration.

        Returns:
            Template for ``namespace`` or ``None`` when the namespace is unknown.
        """

        return self._templates.get(namespace)

    def list_templates(self) -> Mapping[str, OpenSearchIndexTemplate]:
        """Return a shallow copy of the namespace → template mapping.

        Args:
            None

        Returns:
            Mapping of namespace names to template definitions.
        """

        return dict(self._templates)

    def validate_namespace_schema(self, namespace: str) -> None:
        """Ensure required OpenSearch fields are present for ``namespace``."""

        template = self.get_template(namespace)
        if template is None:
            raise ValueError(f"no template registered for namespace={namespace!r}")
        try:
            mappings = template.body["mappings"]  # type: ignore[index]
            properties = mappings["properties"]  # type: ignore[index]
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                f"invalid template for namespace={namespace!r}: missing mappings/properties"
            ) from exc
        if not isinstance(properties, Mapping):
            raise ValueError(
                f"invalid template for namespace={namespace!r}: properties must be a mapping"
            )

        required_types = {
            "doc_id": "keyword",
            "chunk_id": "keyword",
            "namespace": "keyword",
            "vector_id": "keyword",
            "text": "text",
            "token_count": "integer",
        }
        for field, expected_type in required_types.items():
            field_config = properties.get(field)
            if not isinstance(field_config, Mapping) or field_config.get("type") != expected_type:
                raise ValueError(
                    f"template {namespace!r}: field {field!r} must declare type={expected_type}"
                )

        text_config = properties.get("text")
        if not isinstance(text_config, Mapping) or text_config.get("type") != "text":
            raise ValueError(
                f"template {namespace!r}: field 'text' must declare type=text with a valid analyzer"
            )
        analyzer = text_config.get("analyzer")
        if analyzer is not None and str(analyzer).lower() == "keyword":
            raise ValueError(f"template {namespace!r}: field 'text' must not use analyzer=keyword")

        splade_config = properties.get("splade")
        if not isinstance(splade_config, Mapping) or splade_config.get("type") != "rank_features":
            raise ValueError(
                f"template {namespace!r}: field 'splade' must declare type=rank_features"
            )


@dataclass(slots=True)
class _StoredChunk:
    """Wrapper used to keep chunk payloads inside the simulator.

    Attributes:
        payload: Chunk payload stored in memory.

    Examples:
        >>> from DocsToKG.HybridSearch.types import ChunkFeatures, ChunkPayload
        >>> features = ChunkFeatures(bm25_terms={}, splade_weights={}, embedding=[])  # doctest: +SKIP
        >>> chunk = ChunkPayload(  # doctest: +SKIP
        ...     doc_id="d1",
        ...     chunk_id="c1",
        ...     vector_id="v1",
        ...     namespace="demo",
        ...     text="example",
        ...     token_count=1,
        ...     metadata={},
        ...     features=features,
        ... )
        >>> _StoredChunk(chunk).payload is chunk  # doctest: +SKIP
        True
    """

    payload: ChunkPayload


class OpenSearchSimulator(LexicalIndex):
    """Simplified OpenSearch-like index used for development and tests.

    The simulator implements the :class:`~DocsToKG.HybridSearch.interfaces.LexicalIndex`
    protocol and mimics the behaviour of the production OpenSearch integration.

    Attributes:
        _chunks: Mapping of vector identifiers to stored chunk payloads.
        _avg_length: Average token length across indexed chunks (used for BM25).
        _templates: Registry of namespace templates registered via ``register_template``.

    Examples:
        >>> simulator = OpenSearchSimulator()
        >>> simulator.bulk_upsert([])
        >>> simulator.stats()["document_count"]
        0.0
    """

    def __init__(self) -> None:
        self._chunks: Dict[str, _StoredChunk] = {}
        self._avg_length: float = 0.0
        self._templates: Dict[str, OpenSearchIndexTemplate] = {}
        self._df: Dict[str, int] = {}

    def bulk_upsert(self, chunks: Sequence[ChunkPayload]) -> None:
        """Insert or update ``chunks`` within the simulator.

        Args:
            chunks: Iterable of chunk payloads to store or replace.

        Returns:
            None

        Raises:
            None
        """
        for chunk in chunks:
            previous = self._chunks.get(chunk.vector_id)
            if previous is not None:
                self._update_df_on_remove(set(previous.payload.features.bm25_terms.keys()))
            self._chunks[chunk.vector_id] = _StoredChunk(chunk)
            self._update_df_on_add(set(chunk.features.bm25_terms.keys()))
        self._recompute_avg_length()

    def bulk_delete(self, vector_ids: Sequence[str]) -> None:
        """Remove chunk payloads whose vector identifiers are present in ``vector_ids``.

        Args:
            vector_ids: Vector identifiers associated with chunks to remove.

        Returns:
            None

        Raises:
            None
        """
        for vector_id in vector_ids:
            existing = self._chunks.pop(vector_id, None)
            if existing is not None:
                self._update_df_on_remove(set(existing.payload.features.bm25_terms.keys()))
        self._recompute_avg_length()

    def _update_df_on_add(self, terms: set[str]) -> None:
        for token in terms:
            self._df[token] = self._df.get(token, 0) + 1

    def _update_df_on_remove(self, terms: set[str]) -> None:
        for token in terms:
            current = self._df.get(token, 0)
            if current <= 1:
                self._df.pop(token, None)
            else:
                self._df[token] = current - 1

    def fetch(self, vector_ids: Sequence[str]) -> List[ChunkPayload]:
        """Return the stored payloads matching ``vector_ids``.

        Args:
            vector_ids: Identifiers of the desired chunks.

        Returns:
            List of chunk payloads that were previously stored.
        """
        return [self._chunks[vid].payload for vid in vector_ids if vid in self._chunks]

    def vector_ids(self) -> List[str]:
        """Return all vector identifiers currently stored.

        Args:
            None

        Returns:
            Vector identifiers known to the simulator.
        """
        return list(self._chunks.keys())

    def register_template(self, template: OpenSearchIndexTemplate) -> None:
        """Associate ``template`` with its namespace for later lookups.

        Args:
            template: Template definition to register.

        Returns:
            None
        """
        self._templates[template.namespace] = template

    def template_for(self, namespace: str) -> Optional[OpenSearchIndexTemplate]:
        """Return the registered template for ``namespace`` if present.

        Args:
            namespace: Namespace identifier to look up.

        Returns:
            Registered template or ``None`` when no template exists.
        """
        return self._templates.get(namespace)

    def search_bm25(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
    ) -> Tuple[List[Tuple[ChunkPayload, float]], Optional[int]]:
        """Execute a BM25-like search using ``query_weights``.

        Args:
            query_weights: Sparse query representation with token weights.
            filters: Metadata filters applied to stored chunks.
            top_k: Maximum number of results to return in the current page.
            cursor: Optional cursor returned from a previous call.

        Returns:
            Tuple containing hits and the next pagination cursor (or ``None``).
        """
        return self._search_sparse(
            lambda stored: self._bm25_score(stored, query_weights),
            filters,
            top_k,
            cursor,
        )

    def search_splade(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
    ) -> Tuple[List[Tuple[ChunkPayload, float]], Optional[int]]:
        """Execute a SPLADE-style sparse search using ``query_weights``.

        Args:
            query_weights: Sparse query with SPLADE activations.
            filters: Metadata filters applied to stored chunks.
            top_k: Maximum number of results to return in the current page.
            cursor: Optional cursor returned from a previous call.

        Returns:
            Tuple containing hits and the next pagination cursor (or ``None``).
        """
        return self._search_sparse(
            lambda stored: sum(
                weight * stored.payload.features.splade_weights.get(token, 0.0)
                for token, weight in query_weights.items()
            ),
            filters,
            top_k,
            cursor,
        )

    def highlight(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> List[str]:
        """Return naive highlight tokens that appear in ``chunk``.

        Args:
            chunk: Chunk text that should be scanned for highlights.
            query_tokens: Tokens extracted from the query phrase.

        Returns:
            List of tokens present in both ``chunk`` and ``query_tokens``.
        """
        lowered = chunk.text.lower()
        highlights: List[str] = []
        for token in query_tokens:
            token_lower = token.lower()
            if token_lower in lowered:
                highlights.append(token)
        return highlights

    def stats(self) -> Mapping[str, float]:
        """Return summary statistics about the indexed corpus.

        Args:
            None

        Returns:
            Mapping containing document counts and average chunk length.
        """
        return {
            "document_count": float(len(self._chunks)),
            "avg_token_length": float(self._avg_length),
        }

    def _filtered_chunks(self, filters: Mapping[str, object]) -> List[_StoredChunk]:
        """Return stored chunks that satisfy ``filters``.

        Args:
            filters: Metadata filters applied to stored chunks.

        Returns:
            Stored chunks matching the provided filters.
        """
        return [chunk for chunk in self._chunks.values() if matches_filters(chunk.payload, filters)]

    def _bm25_score(
        self,
        stored: _StoredChunk,
        query_weights: Mapping[str, float],
    ) -> float:
        """Compute a BM25-inspired score for ``stored`` given ``query_weights``.

        Args:
            stored: Stored chunk candidate being scored.
            query_weights: Sparse query representation with token weights.

        Returns:
            BM25-inspired similarity score for the stored chunk.
        """
        score = 0.0
        for token, weight in query_weights.items():
            chunk_weight = stored.payload.features.bm25_terms.get(token)
            if chunk_weight is None:
                continue
            score += weight * chunk_weight
        return float(score)

    def _paginate(
        self,
        results: List[Tuple[ChunkPayload, float]],
        top_k: int,
        cursor: Optional[int],
    ) -> Tuple[List[Tuple[ChunkPayload, float]], Optional[int]]:
        """Paginate ``results`` according to ``top_k`` and ``cursor``.

        Args:
            results: Ranked search hits produced by a sparse search.
            top_k: Maximum number of results to include in the response.
            cursor: Optional offset cursor used for pagination.

        Returns:
            Tuple containing the sliced page of results and the next cursor.
        """
        offset = cursor or 0
        end = offset + top_k
        page = results[offset:end]
        next_cursor = end if end < len(results) else None
        return page, next_cursor

    def _search_sparse(
        self,
        scoring_fn: Callable[[_StoredChunk], float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int],
    ) -> Tuple[List[Tuple[ChunkPayload, float]], Optional[int]]:
        """Shared sparse search implementation used by BM25 and SPLADE search.

        Args:
            scoring_fn: Callable that scores stored chunks.
            filters: Metadata filters applied to the search corpus.
            top_k: Maximum number of results requested by the caller.
            cursor: Optional pagination cursor.

        Returns:
            Tuple containing search hits and a possible pagination cursor.
        """
        candidates = self._filtered_chunks(filters)
        scored: List[Tuple[ChunkPayload, float]] = []
        for stored in candidates:
            score = scoring_fn(stored)
            if score > 0.0:
                scored.append((stored.payload, float(score)))
        # Stable ordering on ties to keep cursors deterministic.
        scored.sort(key=lambda item: (-item[1], item[0].vector_id))
        return self._paginate(scored, top_k, cursor)

    def search_bm25_true(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> Tuple[List[Tuple[ChunkPayload, float]], Optional[int]]:
        """Execute Okapi BM25 using stored DF statistics."""

        candidates = self._filtered_chunks(filters)
        if not candidates or not query_weights:
            return ([], None)

        N = max(1, len(self._chunks))
        avgdl = self._avg_length if self._avg_length > 0.0 else 1.0
        terms = list(query_weights.keys())

        def bm25_score(stored: _StoredChunk) -> float:
            """Compute BM25 score for ``stored`` using cached statistics."""

            dl = max(1.0, float(stored.payload.token_count))
            score = 0.0
            for token in terms:
                tf = float(stored.payload.features.bm25_terms.get(token, 0.0))
                if tf <= 0.0:
                    continue
                df = self._df.get(token, 0)
                if df <= 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                if denom <= 0.0:
                    continue
                score += idf * (tf * (k1 + 1.0)) / denom
            return float(score)

        scored = [(stored.payload, bm25_score(stored)) for stored in candidates]
        scored = [(payload, score) for payload, score in scored if score > 0.0]
        scored.sort(key=lambda item: (-item[1], item[0].vector_id))
        return self._paginate(scored, top_k, cursor)

    def _recompute_avg_length(self) -> None:
        """Update the cached average chunk length metric.

        Args:
            None

        Returns:
            None
        """
        if not self._chunks:
            self._avg_length = 0.0
            return
        total = sum(chunk.payload.token_count for chunk in self._chunks.values())
        self._avg_length = total / len(self._chunks)


def matches_filters(chunk: ChunkPayload, filters: Mapping[str, object]) -> bool:
    """Return ``True`` when ``chunk`` satisfies the provided ``filters``."""

    def _stringify_for_match(value: object) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return None

    def _numeric_for_match(value: object) -> Optional[float]:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        if isinstance(value, bytes):
            try:
                return float(value.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                return None
        return None

    def _value_matches(lhs: object, rhs: object) -> bool:
        if lhs == rhs:
            return True
        lhs_str = _stringify_for_match(lhs)
        rhs_str = _stringify_for_match(rhs)
        if lhs_str is not None and rhs_str is not None:
            if lhs_str == rhs_str:
                return True
            if lhs_str.lower() in ("true", "false") and rhs_str.lower() in ("true", "false"):
                return lhs_str.lower() == rhs_str.lower()
        lhs_num = _numeric_for_match(lhs)
        rhs_num = _numeric_for_match(rhs)
        if lhs_num is not None and rhs_num is not None:
            if not (math.isnan(lhs_num) or math.isnan(rhs_num)):
                if math.isclose(lhs_num, rhs_num, rel_tol=1e-9, abs_tol=1e-9):
                    return True
        return False

    def _sequence_values(value: object) -> Optional[Sequence[object]]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
            return value
        return None

    for key, expected in filters.items():
        if key == "namespace":
            if chunk.namespace != expected:
                return False
            continue
        value = chunk.metadata.get(key)
        expected_seq = _sequence_values(expected)
        value_seq = _sequence_values(value)
        if expected_seq is not None:
            candidates = tuple(expected_seq)
            if value_seq is not None:
                value_candidates = tuple(value_seq)
                if not any(
                    _value_matches(candidate_value, candidate_expected)
                    for candidate_value in value_candidates
                    for candidate_expected in candidates
                ):
                    return False
            else:
                if not any(_value_matches(value, candidate) for candidate in candidates):
                    return False
        else:
            if value_seq is not None:
                if not any(
                    _value_matches(candidate_value, expected) for candidate_value in value_seq
                ):
                    return False
            else:
                if not _value_matches(value, expected):
                    return False
    return True
