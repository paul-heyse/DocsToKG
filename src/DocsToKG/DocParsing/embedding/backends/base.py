# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.embedding.backends.base",
#   "purpose": "Base interfaces and supporting types for embedding providers.",
#   "sections": [
#     {
#       "id": "provideridentity",
#       "name": "ProviderIdentity",
#       "anchor": "class-provideridentity",
#       "kind": "class"
#     },
#     {
#       "id": "providertelemetryevent",
#       "name": "ProviderTelemetryEvent",
#       "anchor": "class-providertelemetryevent",
#       "kind": "class"
#     },
#     {
#       "id": "providercontext",
#       "name": "ProviderContext",
#       "anchor": "class-providercontext",
#       "kind": "class"
#     },
#     {
#       "id": "providererror",
#       "name": "ProviderError",
#       "anchor": "class-providererror",
#       "kind": "class"
#     },
#     {
#       "id": "denseembeddingbackend",
#       "name": "DenseEmbeddingBackend",
#       "anchor": "class-denseembeddingbackend",
#       "kind": "class"
#     },
#     {
#       "id": "sparseembeddingbackend",
#       "name": "SparseEmbeddingBackend",
#       "anchor": "class-sparseembeddingbackend",
#       "kind": "class"
#     },
#     {
#       "id": "lexicalembeddingbackend",
#       "name": "LexicalEmbeddingBackend",
#       "anchor": "class-lexicalembeddingbackend",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Base interfaces and supporting types for embedding providers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

ProviderTelemetryEmitter = Callable[["ProviderTelemetryEvent"], None]


@dataclass(slots=True)
class ProviderIdentity:
    """Metadata describing a provider implementation."""

    name: str
    version: str | None = None
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
    batch_hint: int | None = None
    max_concurrency: int | None = None
    normalize_l2: bool = True
    offline: bool = False
    cache_dir: Path | None = None
    telemetry_tags: Mapping[str, str] = field(default_factory=dict)
    telemetry_emitter: ProviderTelemetryEmitter | None = None

    def emit(self, provider: ProviderIdentity, *, phase: str, data: Mapping[str, object]) -> None:
        """Emit a telemetry event if an emitter has been configured."""

        if self.telemetry_emitter is None:
            return
        payload = {
            "device": self.device,
            "dtype": self.dtype,
            "batch_hint": self.batch_hint,
            "max_concurrency": self.max_concurrency,
            "normalize_l2": self.normalize_l2,
            "offline": self.offline,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
        payload.update(self.telemetry_tags)
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
    wrapped: BaseException | None = None

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        base = f"[{self.provider}] {self.category}: {self.detail}"
        if self.retryable:
            base += " (retryable)"
        return base


@runtime_checkable
class DenseEmbeddingBackend(Protocol):
    """Interface for dense embedding providers."""

    identity: ProviderIdentity

    def open(self, context: ProviderContext) -> None: ...

    def close(self) -> None: ...

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_hint: int | None = None,
    ) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class SparseEmbeddingBackend(Protocol):
    """Interface for sparse embedding providers (e.g., SPLADE)."""

    identity: ProviderIdentity

    def open(self, context: ProviderContext) -> None: ...

    def close(self) -> None: ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[tuple[str, float]]]: ...


@runtime_checkable
class LexicalEmbeddingBackend(Protocol):
    """Interface for lexical providers (e.g., BM25)."""

    identity: ProviderIdentity

    def open(self, context: ProviderContext) -> None: ...

    def close(self) -> None: ...

    def accumulate_stats(self, texts: Sequence[str]) -> object: ...

    def vector(
        self,
        text: str,
        stats: object,
    ) -> tuple[Sequence[str], Sequence[float]]: ...
