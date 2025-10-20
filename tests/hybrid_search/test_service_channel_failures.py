"""Tests covering failure handling within :mod:`DocsToKG.HybridSearch.service`."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import pytest

from DocsToKG.HybridSearch.config import HybridSearchConfig
from DocsToKG.HybridSearch.service import HybridSearchService, RequestValidationError
from DocsToKG.HybridSearch.types import HybridSearchRequest


class _StubFuture:
    """Minimal future implementation used to simulate executor behaviour."""

    def __init__(self, *, result: Any = None, exception: Optional[BaseException] = None) -> None:
        self._result = result
        self._exception = exception
        self._done = False
        self.cancelled = False

    def result(self) -> Any:
        self._done = True
        if self._exception is not None:
            raise self._exception
        return self._result

    def cancel(self) -> bool:
        self.cancelled = True
        self._done = True
        return True

    def done(self) -> bool:
        return self._done


class _StubExecutor:
    """Executor that returns pre-seeded futures for deterministic ordering."""

    def __init__(self, futures: Iterable[_StubFuture]) -> None:
        self._futures = list(futures)

    def submit(self, *args: object, **kwargs: object) -> _StubFuture:  # noqa: D401 - signature parity
        if not self._futures:
            raise AssertionError("No futures remaining in stub executor")
        return self._futures.pop(0)


class _StubMetrics:
    def set_gauge(self, *args: object, **kwargs: object) -> None:
        return None

    def observe(self, *args: object, **kwargs: object) -> None:
        return None

    def increment(self, *args: object, **kwargs: object) -> None:
        return None


class _StubLogger:
    def __init__(self) -> None:
        self.exceptions: list[Dict[str, Any]] = []

    def debug(self, *args: object, **kwargs: object) -> None:
        return None

    def exception(self, *args: object, **kwargs: object) -> None:
        self.exceptions.append({"args": args, "kwargs": kwargs})


class _StubObservability:
    def __init__(self) -> None:
        self.metrics = _StubMetrics()
        self.logger = _StubLogger()

    @contextmanager
    def trace(self, *_: object, **__: object) -> Iterable[None]:
        yield


class _StubFeatureGenerator:
    def compute_features(self, _: str) -> object:
        return object()


class _StubDenseStore:
    device = 0
    config = SimpleNamespace(flat_use_fp16=False)

    def get_gpu_resources(self) -> None:
        return None


def _stub_dense_store(_: Any) -> _StubDenseStore:
    return _StubDenseStore()


def test_search_cancels_pending_channels_on_failure() -> None:
    """A failing channel should cancel remaining futures and raise uniformly."""

    service = object.__new__(HybridSearchService)
    service._config_manager = SimpleNamespace(get=lambda: HybridSearchConfig())
    service._feature_generator = _StubFeatureGenerator()
    service._observability = _StubObservability()
    service._registry = SimpleNamespace()
    service._dense_store = MethodType(lambda self, namespace: _stub_dense_store(namespace), service)

    successful_channel = SimpleNamespace(candidates=[], embeddings=None, scores={})
    bm25_future = _StubFuture(result=successful_channel)
    splade_future = _StubFuture(exception=RuntimeError("splade backend unavailable"))
    dense_future = _StubFuture(result=successful_channel)
    service._executor = _StubExecutor([bm25_future, splade_future, dense_future])

    request = HybridSearchRequest(
        query="example query",
        namespace="demo",
        filters={},
        page_size=5,
    )

    with pytest.raises(RequestValidationError) as excinfo:
        service.search(request)

    message = str(excinfo.value)
    assert "splade" in message
    assert "demo" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)

    assert bm25_future.cancelled is False
    assert dense_future.cancelled is True

    logged = service._observability.logger.exceptions
    assert logged, "expected failure to be logged"
    log_event = logged[0]["kwargs"].get("extra", {}).get("event", {})
    assert log_event.get("channel") == "splade"
    assert log_event.get("namespace") == "demo"
    assert log_event.get("query") == "example query"
*** End of File
