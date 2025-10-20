"""Qwen3/vLLM dense embedding provider."""

from __future__ import annotations

import atexit
import os
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from DocsToKG.DocParsing.core.models import QwenCfg
from DocsToKG.DocParsing.env import (
    ensure_model_environment,
    ensure_qwen_dependencies,
    ensure_qwen_environment,
)

from ..base import DenseEmbeddingBackend, ProviderContext, ProviderError, ProviderIdentity
from ..utils import bounded_batch_size, resolve_device

HF_HOME, MODEL_ROOT = ensure_model_environment()


@dataclass(slots=True)
class QwenVLLMConfig:
    """Configuration for the Qwen/vLLM provider."""

    model_dir: Path
    model_id: Optional[str] = None
    dtype: str = "bfloat16"
    tensor_parallelism: int = 1
    gpu_memory_utilization: float = 0.60
    batch_size: int = 32
    quantization: Optional[str] = None
    dimension: int = 2560
    cache_enabled: bool = True
    queue_depth: int = 8


def _shutdown_llm_instance(llm) -> None:
    """Best-effort shutdown for a cached Qwen LLM instance."""

    try:
        engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        if engine and hasattr(engine, "shutdown"):
            engine.shutdown()
    except Exception:  # pragma: no cover - defensive cleanup
        pass
    try:
        if hasattr(llm, "shutdown"):
            llm.shutdown()  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - defensive cleanup
        pass


class _LRUCache:
    """Simple LRU cache that automatically closes evicted entries."""

    def __init__(self, maxsize: int = 2) -> None:
        self.maxsize = max(1, maxsize)
        self._store: "OrderedDict[Tuple[str, str, int, float, Optional[str]], object]" = (
            OrderedDict()
        )

    def get(self, key):
        try:
            value = self._store[key]
        except KeyError:
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key, value) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        self._evict_if_needed()

    def clear(self) -> None:
        for _, value in list(self._store.items()):
            _shutdown_llm_instance(value)
        self._store.clear()

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.maxsize:
            _, value = self._store.popitem(last=False)
            _shutdown_llm_instance(value)


try:  # pragma: no cover - optional dependency
    from collections import OrderedDict
except ImportError:  # pragma: no cover - fallback
    OrderedDict = dict  # type: ignore

_QWEN_LLM_CACHE = _LRUCache(maxsize=2)


def _qwen_cache_key(cfg: QwenCfg) -> Tuple[str, str, int, float, Optional[str]]:
    quant = cfg.quantization if cfg.quantization else None
    return (
        str(cfg.model_dir),
        cfg.dtype,
        int(cfg.tp),
        float(cfg.gpu_mem_util),
        quant,
    )


def _get_vllm_components():
    ensure_qwen_dependencies()
    try:
        from vllm import LLM as llm_cls  # type: ignore
        from vllm import PoolingParams as pooling_cls
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise ProviderError(
            provider="dense.qwen_vllm",
            category="init",
            detail="vLLM is required for the Qwen embedding backend.",
            retryable=False,
            wrapped=exc,
        ) from exc
    return llm_cls, pooling_cls


def _qwen_embed_direct(cfg: QwenCfg, texts: Sequence[str], batch_size: Optional[int]) -> List[List[float]]:
    effective_batch = batch_size or cfg.batch_size
    use_cache = bool(getattr(cfg, "cache_enabled", True))
    cache_key = _qwen_cache_key(cfg)
    llm_cls, pooling_cls = _get_vllm_components()
    llm = _QWEN_LLM_CACHE.get(cache_key) if use_cache else None
    if llm is None:
        llm = llm_cls(
            model=str(cfg.model_dir),
            task="embed",
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tp,
            gpu_memory_utilization=cfg.gpu_mem_util,
            quantization=cfg.quantization,
            download_dir=str(HF_HOME),
        )
        if use_cache:
            _QWEN_LLM_CACHE.put(cache_key, llm)
    pool = pooling_cls(normalize=True, dimensions=int(cfg.dim))
    out: List[List[float]] = []
    try:
        for i in range(0, len(texts), effective_batch):
            batch = list(texts[i : i + effective_batch])
            try:
                res = llm.embed(batch, pooling_params=pool)
            except TypeError:
                res = llm.embed(batch)
            for r in res:
                embedding = getattr(r, "outputs", None)
                if embedding is not None:
                    embedding = getattr(embedding, "embedding", embedding)
                else:
                    embedding = r
                out.append([float(x) for x in embedding])
    finally:
        if not use_cache:
            _shutdown_llm_instance(llm)
    return out


class _QwenQueue:
    """Serialize Qwen embedding requests across worker threads."""

    def __init__(self, cfg: QwenCfg, maxsize: int) -> None:
        self._cfg = cfg
        self._queue: "queue.Queue[Tuple[List[str], int, Future[List[List[float]]]] | None]" = (
            queue.Queue(maxsize=max(1, maxsize))
        )
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="QwenEmbeddingQueue", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            texts, batch_size, future = item
            try:
                result = _qwen_embed_direct(self._cfg, texts, batch_size=batch_size)
            except Exception as exc:  # pragma: no cover - propagate to caller
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._queue.task_done()

    def embed(self, texts: Sequence[str], batch_size: int) -> List[List[float]]:
        if self._closed:
            raise RuntimeError("Qwen embedding queue has been closed")
        future: Future[List[List[float]]] = Future()
        self._queue.put((list(texts), int(batch_size), future))
        return future.result()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._queue.join()
        self._thread.join()


class QwenVLLMProvider(DenseEmbeddingBackend):
    """Dense embedding provider that wraps Qwen3 via vLLM."""

    identity = ProviderIdentity(name="dense.qwen_vllm", version="1.0.0")

    def __init__(self, config: QwenVLLMConfig) -> None:
        self._cfg_spec = config
        self._ctx: ProviderContext | None = None
        self._queue: _QwenQueue | None = None
        self._qwen_cfg: QwenCfg | None = None

    def open(self, context: ProviderContext) -> None:
        self._ctx = context
        model_dir = self._cfg_spec.model_dir.expanduser().resolve()
        if self._cfg_spec.model_id:
            os.environ["DOCSTOKG_QWEN_MODEL_ID"] = self._cfg_spec.model_id
        device_hint = resolve_device(context.device, default="auto")
        env = ensure_qwen_environment(
            device=None if device_hint == "auto" else device_hint,
            dtype=context.dtype if context.dtype != "auto" else None,
            model_dir=model_dir,
        )
        self._qwen_cfg = QwenCfg(
            model_dir=model_dir,
            dtype=self._cfg_spec.dtype if self._cfg_spec.dtype else env["dtype"],
            tp=int(self._cfg_spec.tensor_parallelism),
            gpu_mem_util=float(self._cfg_spec.gpu_memory_utilization),
            batch_size=int(self._cfg_spec.batch_size),
            quantization=self._cfg_spec.quantization,
            dim=int(self._cfg_spec.dimension),
            cache_enabled=bool(self._cfg_spec.cache_enabled),
        )
        depth = self._cfg_spec.queue_depth or 1
        if context.max_concurrency:
            depth = max(depth, int(context.max_concurrency))
        self._queue = _QwenQueue(self._qwen_cfg, maxsize=depth)

    def close(self) -> None:
        if self._queue is not None:
            self._queue.shutdown()
            self._queue = None
        self._qwen_cfg = None
        if self._ctx is not None:
            self._ctx.emit(
                self.identity,
                phase="close",
                data={"status": "closed"},
            )
        self._ctx = None

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_hint: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        if self._queue is None or self._qwen_cfg is None:
            raise ProviderError(
                provider=self.identity.name,
                category="runtime",
                detail="Provider has not been opened before use.",
                retryable=False,
            )
        batch_size = bounded_batch_size(
            preferred=batch_hint or self._qwen_cfg.batch_size,
            fallback=self._qwen_cfg.batch_size,
        )
        vectors = self._queue.embed(texts, batch_size=batch_size)
        if self._ctx:
            self._ctx.emit(
                self.identity,
                phase="embed",
                data={
                    "batch_size_effective": batch_size,
                    "vector_count": len(vectors),
                },
            )
        return vectors


def flush_llm_cache() -> None:
    """Expose cache flush for legacy callers/tests."""

    _QWEN_LLM_CACHE.clear()


atexit.register(flush_llm_cache)


__all__ = ["QwenVLLMConfig", "QwenVLLMProvider", "flush_llm_cache"]
