# 1. HybridSearch GPU Guide

This document summarises the GPU-only workflow now built into
`DocsToKG.HybridSearch`. The dense FAISS stack can build and query indexes
directly on GPU without touching CPU memory, and similarity utilities expose
GPU-backed cosine helpers for deduplication and fusion.

## 2. Key Capabilities

- `FaissIndexManager` constructs `GpuIndexFlat`, `GpuIndexIVFFlat`, and
  `GpuIndexIVFPQ` instances without CPU fallbacks.
- Training samples are drawn on GPU with configurable oversampling to keep
  IVFFlat/PQ builds fast.
- Strict CPU→GPU cloning (used during restore) prohibits CPU coarse quantisers.
- Runtime device selection via `DenseIndexConfig.device` or the
  `HYBRIDSEARCH_FAISS_DEVICE` environment variable.
- `DocsToKG.HybridSearch.vectorstore` provides GPU cosine helpers:
  `cosine_against_corpus_gpu`, `pairwise_inner_products`, and fast L2
  normalisation.

## 3. Requirements

- `faiss-gpu` wheel installed (the build must expose `GpuIndex*` classes).
- CUDA runtime compatible with the FAISS wheel.
- At least one visible CUDA device:

```python
import faiss
assert faiss.get_num_gpus() >= 1
```

## 4. Config Knobs

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
    multi_gpu_mode="single",
    gpu_temp_memory_bytes=None,
    gpu_indices_32_bit=True,
    expected_ntotal=1_000_000,
    rebuild_delete_threshold=10000,
    force_64bit_ids=False,
    interleaved_layout=True,
    flat_use_fp16=False,
)
```

- `device` selects the GPU ordinal. Override at runtime with
  `HYBRIDSEARCH_FAISS_DEVICE`.
- `ivfpq_use_precomputed` / `ivfpq_float16_lut` toggle FAISS IVFPQ tuning.
- `oversample` controls training sample size (`nlist * oversample`).
- `multi_gpu_mode="replicate"` fans the index out to every visible GPU using
  `GpuMultipleClonerOptions`; leave as `"single"` to pin the index to one device.
- `gpu_temp_memory_bytes` configures FAISS' temporary scratch allocator (set it
  when you want deterministic peak VRAM usage).
- `gpu_indices_32_bit=True` stores FAISS indices in 32-bit format, reducing VRAM
  pressure when you know `ntotal < 2**31`.
- `expected_ntotal` hints the expected vector count so FAISS can reserve GPU memory up front.
- `rebuild_delete_threshold` batches delete-heavy workloads before forcing a full rebuild (set it
  above `0` to defer rebuilds; the default of `0` rebuilds immediately for correctness).
- `force_64bit_ids=True` disables the 32-bit ID optimisation when the full FAISS ID space is required.
- `interleaved_layout` toggles GPU IVF interleaving optimisations (leave enabled unless debugging).
- `flat_use_fp16` enables float16 compute for flat indexes when your workload tolerates the precision trade-off.

Training samples for IVF indexes draw from a deterministic NumPy RNG, so repeated
runs over the same corpus will reuse the exact calibration set. This makes it
easier to compare recall/latency deltas between experiments.

## 5. Quick Start

```python
import numpy as np
from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.vectorstore import FaissIndexManager
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

## 6. Similarity Helpers

`DocsToKG.HybridSearch.vectorstore` now exposes GPU utilities that reuse FAISS
resources. They normalise vectors and compute cosine similarity without
roundtripping through NumPy:

```python
from DocsToKG.HybridSearch.vectorstore import cosine_against_corpus_gpu

scores = cosine_against_corpus_gpu(query, xb)
```

`pairwise_inner_products` powers both fusion and deduplication paths, and
`max_inner_product` provides a convenience wrapper.

## 7. Validation

GPU-only tests cover flat, IVFFlat, IVFPQ, and the cosine helpers. Run them with:

```bash
pytest -q tests/test_hybridsearch_gpu_only.py
```

Tests skip automatically if FAISS cannot see a GPU.
