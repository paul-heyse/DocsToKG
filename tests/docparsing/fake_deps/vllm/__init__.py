"""Minimal vLLM stub used in DocParsing tests."""

from __future__ import annotations

from typing import Iterable, List

DEFAULT_DENSE_DIM = 2560


class _Embedding:
    """Simplified embedding container mimicking vLLM outputs."""

    def __init__(self, values: Iterable[float]) -> None:
        self.embedding = list(values)


class _EmbeddingResult:
    """Wrapper providing the ``outputs`` attribute from vLLM."""

    def __init__(self, values: Iterable[float]) -> None:
        self.outputs = _Embedding(values)


class LLM:
    """Very small subset of :class:`vllm.LLM` for testing."""

    def __init__(self, *_, dense_dim: int = DEFAULT_DENSE_DIM, **__) -> None:
        self.dense_dim = dense_dim

    def embed(self, prompts: Iterable[str]) -> List[_EmbeddingResult]:
        vector = [float(index) for index in range(self.dense_dim)]
        return [_EmbeddingResult(vector) for _ in prompts]


class PoolingParams:  # pragma: no cover - simple attribute holder
    def __init__(self, *_, **__):
        pass

