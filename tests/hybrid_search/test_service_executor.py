"""Tests for HybridSearchService executor configuration."""

from DocsToKG.HybridSearch.config import HybridSearchConfig, RetrievalConfig
from DocsToKG.HybridSearch.features import FeatureGenerator
from DocsToKG.HybridSearch.service import HybridSearchService


class _StubConfigManager:
    def __init__(self, config: HybridSearchConfig) -> None:
        self._config = config

    def get(self) -> HybridSearchConfig:
        return self._config


class _StubDenseStore:
    def __init__(self) -> None:
        self._resolver = None

    def get_gpu_resources(self):
        return None

    def set_id_resolver(self, resolver) -> None:
        self._resolver = resolver


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


class _StubLexicalIndex:
    def validate_namespace_schema(self, namespace: str) -> None:  # pragma: no cover - unused in test
        return None


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
