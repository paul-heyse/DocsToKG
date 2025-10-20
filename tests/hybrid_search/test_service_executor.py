"""Tests for HybridSearchService executor configuration."""

from types import MethodType

from DocsToKG.HybridSearch.config import HybridSearchConfig, RetrievalConfig
from DocsToKG.HybridSearch.features import FeatureGenerator
from DocsToKG.HybridSearch.service import ChannelResults, HybridSearchService
from DocsToKG.HybridSearch.types import HybridSearchRequest


class _StubConfigManager:
    def __init__(self, config: HybridSearchConfig) -> None:
        self._config = config

    def get(self) -> HybridSearchConfig:
        return self._config

    def set_config(self, config: HybridSearchConfig) -> None:
        self._config = config


class _StubDenseStore:
    def __init__(self) -> None:
        self._resolver = None
        self.adapter_stats = None
        self.device = 0
        self.config = type("Config", (), {"flat_use_fp16": False})()

    def get_gpu_resources(self):
        return None

    def set_id_resolver(self, resolver) -> None:
        self._resolver = resolver

    def stats(self):  # pragma: no cover - unused in test
        return {}


class _StubFaissRouter:
    def __init__(self, store: _StubDenseStore) -> None:
        self.default_store = store
        self._resolver = None

    def set_resolver(self, resolver) -> None:
        self._resolver = resolver

    def get(self, namespace):  # pragma: no cover - unused in test
        return self.default_store


class _StubRegistry:
    def resolve_faiss_id(self, vector_id: int):  # pragma: no cover - unused in test
        return None

    def resolve_embedding(self, vector_id: int, *, cache):  # pragma: no cover - unused in test
        raise AssertionError("resolve_embedding should not be called in this test")

    def resolve_embeddings(self, vector_ids):  # pragma: no cover - unused in test
        raise AssertionError("resolve_embeddings should not be called in this test")

    def bulk_get(self, vector_ids):  # pragma: no cover - unused in test
        return []

    def count(self) -> int:  # pragma: no cover - unused in test
        return 0


class _StubLexicalIndex:
    def validate_namespace_schema(self, namespace: str) -> None:  # pragma: no cover - unused in test
        return None

    def stats(self):  # pragma: no cover - unused in test
        return {}


def test_service_uses_configured_executor_workers() -> None:
    """HybridSearchService should honour executor_max_workers when configured."""

    config = HybridSearchConfig(
        retrieval=RetrievalConfig(executor_max_workers=5)
    )
    manager = _StubConfigManager(config)
    feature_generator = FeatureGenerator()
    dense_store = _StubDenseStore()
    router = _StubFaissRouter(dense_store)

    service = HybridSearchService(
        config_manager=manager,
        feature_generator=feature_generator,
        faiss_index=dense_store,
        opensearch=_StubLexicalIndex(),
        registry=_StubRegistry(),
        faiss_router=router,
    )

    try:
        assert service._executor._max_workers == 5
    finally:
        service.close()


def test_service_refreshes_executor_on_config_reload() -> None:
    """Executor pool is refreshed when the config changes at runtime."""

    config = HybridSearchConfig(retrieval=RetrievalConfig(executor_max_workers=2))
    manager = _StubConfigManager(config)
    feature_generator = FeatureGenerator()
    dense_store = _StubDenseStore()
    router = _StubFaissRouter(dense_store)

    service = HybridSearchService(
        config_manager=manager,
        feature_generator=feature_generator,
        faiss_index=dense_store,
        opensearch=_StubLexicalIndex(),
        registry=_StubRegistry(),
        faiss_router=router,
    )

    try:
        assert service._executor_max_workers == 2

        def _stub_execute(self, *args, **kwargs):
            return ChannelResults(candidates=[], scores={})

        service._execute_bm25 = MethodType(_stub_execute, service)
        service._execute_splade = MethodType(_stub_execute, service)
        service._execute_dense = MethodType(_stub_execute, service)

        manager.set_config(
            HybridSearchConfig(retrieval=RetrievalConfig(executor_max_workers=4))
        )

        request = HybridSearchRequest(
            query="example",
            namespace=None,
            filters={},
            page_size=5,
            diagnostics=False,
        )

        response = service.search(request)

        assert response.results == []
        assert service._executor_max_workers == 4
        assert service._executor._max_workers == 4
    finally:
        service.close()
