"""Tests for HybridSearchService executor configuration."""

import threading

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
        self.device = 0
        self.adapter_stats = None

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


def test_executor_rebuilds_on_config_change() -> None:
    """Executor pool should rebuild when the config's worker count changes."""

    initial_config = HybridSearchConfig(
        retrieval=RetrievalConfig(executor_max_workers=1)
    )
    manager = _StubConfigManager(initial_config)
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

    release_gate = threading.Event()
    continue_gate = threading.Event()

    try:
        def _blocking_task() -> str:
            release_gate.set()
            continue_gate.wait(timeout=5)
            return "finished"

        # Submit work to the original executor so we can ensure it finishes.
        original_future = service._executor.submit(_blocking_task)
        assert release_gate.wait(timeout=1)
        original_executor = service._executor

        # Simulate a config reload with a higher worker budget.
        manager._config = HybridSearchConfig(
            retrieval=RetrievalConfig(executor_max_workers=2)
        )
        service._ensure_executor_capacity(manager.get())

        assert service._executor is not original_executor
        assert service._executor._max_workers == 2

        # New submissions should run on the rebuilt executor.
        future = service._executor.submit(lambda: "new-executor")
        assert future.result(timeout=1) == "new-executor"

        # Unblock the original task and ensure it completed successfully.
        continue_gate.set()
        assert original_future.result(timeout=5) == "finished"
        assert original_executor._shutdown is True
    finally:
        continue_gate.set()
        service.close()
