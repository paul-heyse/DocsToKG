"""Dense embedding provider for Text Embeddings Inference (TEI) services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import httpx

from ..base import DenseEmbeddingBackend, ProviderContext, ProviderError, ProviderIdentity


@dataclass(slots=True)
class TEIConfig:
    url: str
    timeout_seconds: float = 30.0
    max_inflight: int = 4
    headers: Dict[str, str] = field(default_factory=dict)
    verify: bool = True


class TEIProvider(DenseEmbeddingBackend):
    """HTTP-based dense embedding provider compatible with TEI deployments."""

    identity = ProviderIdentity(name="dense.tei", version="1.0.0")

    def __init__(self, config: TEIConfig) -> None:
        self._cfg = config
        self._ctx: ProviderContext | None = None
        self._client: httpx.Client | None = None

    def open(self, context: ProviderContext) -> None:
        self._ctx = context
        if not self._cfg.url:
            raise ProviderError(
                provider=self.identity.name,
                category="validation",
                detail="dense.tei.url must be provided.",
                retryable=False,
            )
        limits = httpx.Limits(
            max_connections=max(1, self._cfg.max_inflight),
            max_keepalive_connections=max(1, self._cfg.max_inflight),
        )
        timeout = httpx.Timeout(self._cfg.timeout_seconds)
        self._client = httpx.Client(
            base_url=self._cfg.url,
            timeout=timeout,
            limits=limits,
            headers=self._cfg.headers,
            verify=self._cfg.verify,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._ctx = None

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_hint: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        if self._client is None:
            raise ProviderError(
                provider=self.identity.name,
                category="runtime",
                detail="Provider has not been opened before use.",
                retryable=False,
            )
        payload = {"inputs": list(texts)}
        try:
            response = self._client.post("", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(
                provider=self.identity.name,
                category="network",
                detail=f"TEI request failed: {exc}",
                retryable=True,
                wrapped=exc,
            ) from exc
        if response.status_code >= 500:
            raise ProviderError(
                provider=self.identity.name,
                category="network",
                detail=f"TEI returned {response.status_code}",
                retryable=True,
            )
        if response.status_code >= 400:
            raise ProviderError(
                provider=self.identity.name,
                category="validation",
                detail=f"TEI returned {response.status_code}: {response.text}",
                retryable=False,
            )
        try:
            vectors_raw = response.json()
        except ValueError as exc:
            raise ProviderError(
                provider=self.identity.name,
                category="runtime",
                detail="Failed to decode TEI response as JSON.",
                retryable=False,
                wrapped=exc,
            ) from exc
        if not isinstance(vectors_raw, list):
            raise ProviderError(
                provider=self.identity.name,
                category="validation",
                detail="TEI response did not contain a list of vectors.",
                retryable=False,
            )
        vectors: List[List[float]] = []
        for vector in vectors_raw:
            if not isinstance(vector, (list, tuple)):
                raise ProviderError(
                    provider=self.identity.name,
                    category="validation",
                    detail="TEI response vectors must be lists.",
                    retryable=False,
                )
            vectors.append([float(x) for x in vector])
        if self._ctx:
            self._ctx.emit(
                self.identity,
                phase="embed",
                data={"vector_count": len(vectors), "batch_size_effective": len(texts)},
            )
        return vectors


__all__ = ["TEIConfig", "TEIProvider"]
