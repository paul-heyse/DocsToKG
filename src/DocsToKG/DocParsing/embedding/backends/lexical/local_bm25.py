# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.embedding.backends.lexical.local_bm25",
#   "purpose": "Local BM25 lexical embedding provider.",
#   "sections": [
#     {
#       "id": "tokenize",
#       "name": "_tokenize",
#       "anchor": "function-tokenize",
#       "kind": "function"
#     },
#     {
#       "id": "bm25accumulator",
#       "name": "BM25Accumulator",
#       "anchor": "class-bm25accumulator",
#       "kind": "class"
#     },
#     {
#       "id": "localbm25config",
#       "name": "LocalBM25Config",
#       "anchor": "class-localbm25config",
#       "kind": "class"
#     },
#     {
#       "id": "localbm25provider",
#       "name": "LocalBM25Provider",
#       "anchor": "class-localbm25provider",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Local BM25 lexical embedding provider."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence, Tuple

from DocsToKG.DocParsing.core.models import BM25Stats

from ..base import LexicalEmbeddingBackend, ProviderContext, ProviderError, ProviderIdentity

try:  # pragma: no cover - regex compilation is deterministic
    import regex as re  # type: ignore
except Exception:  # pragma: no cover - fallback when regex unavailable
    import re

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> Sequence[str]:
    if not text:
        return ()
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [match.group(0) for match in TOKEN_RE.finditer(normalized)]


@dataclass(slots=True)
class BM25Accumulator:
    N: int = 0
    total_tokens: int = 0
    df: Counter = field(default_factory=Counter)

    def add_document(self, text: str) -> None:
        tokens = _tokenize(text)
        self.N += 1
        self.total_tokens += len(tokens)
        self.df.update(set(tokens))

    def finalize(self) -> BM25Stats:
        avgdl = self.total_tokens / max(self.N, 1)
        return BM25Stats(N=self.N, avgdl=avgdl, df=dict(self.df))


@dataclass(slots=True)
class LocalBM25Config:
    k1: float = 1.5
    b: float = 0.75


class LocalBM25Provider(LexicalEmbeddingBackend):
    identity = ProviderIdentity(name="lexical.local_bm25", version="1.0.0")

    def __init__(self, config: LocalBM25Config) -> None:
        self._cfg = config
        self._ctx: ProviderContext | None = None

    def open(self, context: ProviderContext) -> None:
        self._ctx = context

    def close(self) -> None:
        self._ctx = None

    def accumulate_stats(self, texts: Sequence[str]) -> BM25Stats:
        acc = BM25Accumulator()
        for text in texts:
            acc.add_document(text)
        stats = acc.finalize()
        if self._ctx:
            self._ctx.emit(
                self.identity,
                phase="bm25_stats",
                data={
                    "documents": stats.N,
                    "avgdl": round(stats.avgdl, 4),
                    "unique_terms": len(stats.df),
                },
            )
        return stats

    def vector(self, text: str, stats: object) -> Tuple[Sequence[str], Sequence[float]]:
        if not isinstance(stats, BM25Stats):
            raise ProviderError(
                provider=self.identity.name,
                category="validation",
                detail="BM25 statistics must be provided as a BM25Stats instance.",
                retryable=False,
            )

        tokens = _tokenize(text)
        if not tokens:
            return (), ()

        freqs = Counter(tokens)
        k1 = float(self._cfg.k1)
        b = float(self._cfg.b)
        avgdl = float(stats.avgdl or 0.0)
        N = int(stats.N or 0)
        dl = len(tokens)

        terms = []
        weights = []
        for term, tf in freqs.items():
            df = stats.df.get(term, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / max(avgdl, 1e-9)))
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            weight = idf * numerator / denominator
            terms.append(term)
            weights.append(float(weight))

        order = sorted(range(len(weights)), key=weights.__getitem__, reverse=True)
        ordered_terms = [terms[idx] for idx in order]
        ordered_weights = [weights[idx] for idx in order]

        if self._ctx:
            self._ctx.emit(
                self.identity,
                phase="bm25_vector",
                data={"token_count": dl, "unique_terms": len(ordered_terms)},
            )

        return ordered_terms, ordered_weights


__all__ = [
    "BM25Accumulator",
    "LocalBM25Config",
    "LocalBM25Provider",
]
