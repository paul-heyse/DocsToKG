"""Base interfaces and supporting types for embedding providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

ProviderTelemetryEmitter = Callable[["ProviderTelemetryEvent"], None]


@dataclass(slots=True)
class ProviderIdentity:
    """Metadata describing a provider implementation."""

    name: str
    version: Optional[str] = None
    extra: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderTelemetryEvent:
    """Structured telemetry payload emitted by providers."""

    provider: ProviderIdentity
    phase: str
    data: Mapping[str, object]


@dataclass(slots=True)
class ProviderContext:
    """Common configuration shared with providers at runtime."""

    device: str = "auto"
    dtype: str = "auto"
    batch_hint: Optional[int] = None
    max_concurrency: Optional[int] = None
    normalize_l2: bool = True
    offline: bool = False
    cache_dir: Optional[Path] = None
    telemetry_tags: Mapping[str, str] = field(default_factory=dict)
    telemetry_emitter: Optional[ProviderTelemetryEmitter] = None

    def emit(self, provider: ProviderIdentity, *, phase: str, data: Mapping[str, object]) -> None:
        """Emit a telemetry event if an emitter has been configured."""

        if self.telemetry_emitter is None:
            return
        payload = dict(self.telemetry_tags)
        payload.update(data)
        event = ProviderTelemetryEvent(provider=provider, phase=phase, data=payload)
        self.telemetry_emitter(event)


@dataclass(slots=True)
class ProviderError(RuntimeError):
    """Raised by providers when initialization or inference fails."""

    provider: str
    category: str
    detail: str
    retryable: bool = False
    wrapped: Optional[BaseException] = None

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        base = f"[{self.provider}] {self.category}: {self.detail}"
        if self.retryable:
            base += " (retryable)"
        return base


@runtime_checkable
class DenseEmbeddingBackend(Protocol):
    """Interface for dense embedding providers."""

    identity: ProviderIdentity

    def open(self, context: ProviderContext) -> None:
        ...

    def close(self) -> None:
        ...

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_hint: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        ...


@runtime_checkable
class SparseEmbeddingBackend(Protocol):
    """Interface for sparse embedding providers (e.g., SPLADE)."""

    identity: ProviderIdentity

    def open(self, context: ProviderContext) -> None:
        ...

    def close(self) -> None:
        ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[Tuple[str, float]]]:
        ...


@runtime_checkable
class LexicalEmbeddingBackend(Protocol):
    """Interface for lexical providers (e.g., BM25)."""

    identity: ProviderIdentity

    def open(self, context: ProviderContext) -> None:
        ...

    def close(self) -> None:
        ...

    def accumulate_stats(self, texts: Sequence[str]) -> object:
        ...

    def vector(
        self,
        text: str,
        stats: object,
    ) -> Tuple[Sequence[str], Sequence[float]]:
        ...
