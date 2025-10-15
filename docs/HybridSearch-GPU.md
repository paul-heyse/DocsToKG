# HybridSearch GPU Guide

This document summarises the GPU-only workflow now built into
`DocsToKG.HybridSearch`. The dense FAISS stack can build and query indexes
directly on GPU without touching CPU memory, and similarity utilities expose
GPU-backed cosine helpers for deduplication and fusion.

## Key capabilities

- `FaissIndexManager` constructs `GpuIndexFlat`, `GpuIndexIVFFlat`, and
  `GpuIndexIVFPQ` instances without CPU fallbacks.
- Training samples are drawn on GPU with configurable oversampling to keep
  IVFFlat/PQ builds fast.
- Strict CPU→GPU cloning (used during restore) prohibits CPU coarse quantisers.
- Runtime device selection via `DenseIndexConfig.device` or the
  `HYBRIDSEARCH_FAISS_DEVICE` environment variable.
- `DocsToKG.HybridSearch.similarity` provides GPU cosine helpers:
  `cosine_against_corpus_gpu`, `pairwise_inner_products`, and fast L2
  normalisation.

## Requirements

- `faiss-gpu` wheel installed (the build must expose `GpuIndex*` classes).
- CUDA runtime compatible with the FAISS wheel.
- At least one visible CUDA device:

```python
import faiss
assert faiss.get_num_gpus() >= 1
```

## Config knobs

`DenseIndexConfig` gained additional fields consumed on demand:

```python
DenseIndexConfig(
    index_type="ivf_pq",
    nlist=4096,
    nprobe=32,
    pq_m=32,
    pq_bits=8,
    oversample=2,
    device=0,
    ivfpq_use_precomputed=True,
    ivfpq_float16_lut=True,
)
```

- `device` selects the GPU ordinal. Override at runtime with
  `HYBRIDSEARCH_FAISS_DEVICE`.
- `ivfpq_use_precomputed` / `ivfpq_float16_lut` toggle FAISS IVFPQ tuning.
- `oversample` controls training sample size (`nlist * oversample`).

## Quick start

```python
import numpy as np
from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.dense import FaissIndexManager
from DocsToKG.HybridSearch.ids import vector_uuid_to_faiss_int

cfg = DenseIndexConfig(index_type="ivf_flat", nlist=1024, nprobe=16, device=0)
manager = FaissIndexManager(dim=768, config=cfg)

xb = np.random.rand(10_000, 768).astype("float32")
ids = [f"vec-{i}" for i in range(len(xb))]
manager.set_id_resolver({vector_uuid_to_faiss_int(vid): vid for vid in ids}.get)
manager.add([row for row in xb], ids)

query = np.random.rand(768).astype("float32")
results = manager.search(query, top_k=10)
```

## Similarity helpers

`DocsToKG.HybridSearch.similarity` now exposes GPU utilities that reuse FAISS
resources. They normalise vectors and compute cosine similarity without
roundtripping through NumPy:

```python
from DocsToKG.HybridSearch.similarity import cosine_against_corpus_gpu

scores = cosine_against_corpus_gpu(query, xb)
```

`pairwise_inner_products` powers both fusion and deduplication paths, and
`max_inner_product` provides a convenience wrapper.

## Validation

GPU-only tests cover flat, IVFFlat, IVFPQ, and the cosine helpers. Run them with:

```bash
pytest -q tests/test_hybridsearch_gpu_only.py
```

Tests skip automatically if FAISS cannot see a GPU.
