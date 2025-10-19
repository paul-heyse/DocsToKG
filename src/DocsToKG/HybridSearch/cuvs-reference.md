# cuVS GPU Toolkit — Reference for DocsToKG Agents

## Table of Contents
- [1) Runtime prerequisites](#1-runtime-prerequisites)
- [2) Package layout](#2-package-layout)
- [3) Common runtime building blocks](#3-common-runtime-building-blocks)
  - [3.1 Resources & streams](#31-resources--streams)
  - [3.2 Device vs host arrays](#32-device-vs-host-arrays)
  - [3.3 Error handling](#33-error-handling)
- [4) Nearest-neighbor algorithms](#4-nearest-neighbor-algorithms)
- [5) Clustering (KMeans)](#5-clustering-kmeans)
- [6) Distance utilities](#6-distance-utilities)
- [7) Common module utilities](#7-common-module-utilities)
- [8) Interoperability & external tooling](#8-interoperability--external-tooling)
- [9) Integration with DocsToKG HybridSearch](#9-integration-with-docstokg-hybridsearch)
- [10) Testing & troubleshooting](#10-testing--troubleshooting)
- [11) Minimal smoke test](#11-minimal-smoke-test)
- [12) Useful references](#12-useful-references)
**Package**: `cuvs` (Python)  
**Version**: `25.10.00` (see `cuvs/VERSION`)  
**Companion libs**: `libcuvs` (CUDA shared objects), `librmm`, `rapids_logger`, CUDA 12 toolchain

This guide documents the NVIDIA cuVS components shipped in the DocsToKG environment and how they relate to HybridSearch. Read it when you need to understand the GPU primitives that FAISS can dispatch to, or when you want to experiment with cuVS directly (either within DocsToKG or in external notebooks/scripts).

> **Reminder**: the current custom FAISS wheel was built **without** cuVS kernels. Attempting to call `faiss.knn_gpu(..., use_cuvs=True)` raises `RuntimeError: cuVS has not been compiled into the current version so it cannot be used.` HybridSearch therefore treats cuVS as *optional* and reports the disabled state via `AdapterStats`. This document still covers the cuVS surface so the team can assess future wheel upgrades or use cuVS as a standalone ANN toolkit.

---

## 1) Runtime prerequisites

cuVS Python extensions (`cuvs/neighbors/*.so`, `cuvs/cluster/kmeans/*.so`, etc.) depend on the following shared libraries:

- `.venv/lib/python3.13/site-packages/libcuvs/lib64/libcuvs.so`
- `.venv/lib/python3.13/site-packages/libcuvs/lib64/libcuvs_c.so`
- `.venv/lib/python3.13/site-packages/librmm/lib64/librmm.so`
- `.venv/lib/python3.13/site-packages/rapids_logger/lib64/librapids_logger.so`
- CUDA runtime/toolchain: `libcudart.so.12`, `libcublas(.so/.so.12)`, `libcublasLt.so.12`, `libcusolver.so.11`, `libcusparse.so.12`, `libcurand.so.10`, `libnvJitLink.so.12`
- NCCL: `.venv/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2`

The cuVS package’s `__init__.py` calls `libcuvs.load_library()` when the wheel is installed, but the RAPIDS libraries above are not on the default loader path. DocsToKG therefore preloads them in `DocsToKG.HybridSearch.store._ensure_cuvs_loader_path()` and amends `LD_LIBRARY_PATH` before importing FAISS/cuVS. If you access cuVS outside HybridSearch, replicate the same directory additions.

---

## 2) Package layout

Top-level modules under `.venv/lib/python3.13/site-packages/cuvs`:

- `cuvs.common` — Runtime helpers (CUDA stream/resource wrappers, DLPack bridging, error checking).
- `cuvs.distance` — Pairwise distance computations and enumerations for distance types.
- `cuvs.cluster` — KMeans clustering.
- `cuvs.neighbors` — Approximate and exact ANN algorithms.
- `cuvs.preprocessing` — (Currently empty placeholder in this wheel; reserved for future featurizers).
- `cuvs.tests` — Embedded pytest suite; useful as runnable examples.

Each `neighbors` submodule exposes a `build()/search()/save()/load()` surface tailored to the ANN method.

---

## 3) Common runtime building blocks

### 3.1 Resources & streams

Most cuVS compute entrypoints accept an optional `resources` argument:

```python
from cuvs.common import Resources

handle = Resources()        # wraps RAPIDS RAFT resources/stream
# pass `resources=handle` to cuVS functions for stream reuse
handle.sync()               # explicit synchronization if you supplied the handle
```

If you omit `resources`, cuVS allocates one internally and synchronizes before returning (see `cuvs.common.auto_sync_resources`). Supplying your own handle lets you integrate with other CUDA frameworks (CuPy, PyTorch) by borrowing their stream pointer.

### 3.2 Device vs host arrays

cuVS accepts:

- Host NumPy arrays (automatically transferred to device; slower).
- Device arrays implementing the RAPIDS `device_ndarray` protocol (`pylibraft.common.device_ndarray`), CuPy arrays, or other DLPack-compatible buffers.

Most ANN examples in `cuvs.tests` create device arrays with RAPIDS’ `device_ndarray`, e.g.:

```python
from pylibraft.common import device_ndarray
dataset_device = device_ndarray(np.random.rand(10000, 64).astype(np.float32))
```

When outputs are provided as device buffers, cuVS writes results in place; otherwise it returns host copies.

### 3.3 Error handling

All C++ → Python boundaries call `cuvs.common.exceptions.check_cuvs`, raising Python exceptions with informative messages when the underlying RAFT/cuVS code fails (e.g., invalid parameters, CUDA errors). Catch these exceptions around build/search calls for telemetry.

---

## 4) Nearest-neighbor algorithms

The cuVS wheel ships several GPU ANN implementations. Each submodule typically exports `Index`, parameter classes, and core functions. Highlights:

- **Brute force** (`cuvs.neighbors.brute_force`) — Exact kNN search; useful as a baseline or for small datasets. Functions: `build`, `search`, `extend`, `save`, `load`.
- **CAGRA** (`cuvs.neighbors.cagra`) — Graph-based ANN (Continuous Approximate Graph Representation). Supports `IndexParams`, `SearchParams`, incremental `extend`, and serialization.
- **HNSW** (`cuvs.neighbors.hnsw`) — Hierarchical Navigable Small World graph ANN (`build`, `extend`, `search`).
- **IVF-Flat / IVF-PQ** (`cuvs.neighbors.ivf_flat`, `ivf_pq`) — Inverted file structures with optional product quantization. Parameter knobs include number of lists (`n_lists`), probes, PQ bits, LUT dtype.
- **NN-Descent** (`cuvs.neighbors.nn_descent`) — Graph refinement algorithm for ANN graphs.
- **Vamana** (`cuvs.neighbors.vamana`) — Disk-friendly graph ANN with `Index`, `IndexParams`, `build`, `save`.
- **All neighbors** (`cuvs.neighbors.all_neighbors`) — End-to-end pipelines that blend brute force with filtering.
- **Filters** (`cuvs.neighbors.filters`) — Post-processing utilities (e.g., `apply_filter`).
- **Refine** (`cuvs.neighbors.refine`) — Improve ANN results using refinement passes.
- **Tiered index** (`cuvs.neighbors.tiered_index`) — Multi-stage ANN pipeline for balanced performance/recall.
- **Scalar quantizer / binary quantizer** — packaged under `cuvs.neighbors` with dedicated tests (see `test_scalar_quantizer.py`, `test_binary_quantizer.py`).
- **Multi-GPU variants** (`cuvs.neighbors.mg`) — Distributed versions of CAGRA, IVF-Flat, IVF-PQ via `mg.cagra`, `mg.ivf_flat`, `mg.ivf_pq`. These require NCCL for collective ops.

All algorithms follow a similar usage pattern:

```python
from cuvs.neighbors import ivf_pq
from cuvs.common import Resources
from pylibraft.common import device_ndarray

handle = Resources()
dataset = device_ndarray(np.random.rand(10000, 64).astype(np.float32))

index = ivf_pq.build(
    ivf_pq.IndexParams(n_lists=1024, metric="sqeuclidean", pq_bits=8),
    dataset,
    resources=handle,
)

queries = device_ndarray(np.random.rand(1000, 64).astype(np.float32))
distances = device_ndarray(np.empty((1000, 10), np.float32))
neighbors = device_ndarray(np.empty((1000, 10), np.int64))

ivf_pq.search(
    ivf_pq.SearchParams(n_probes=128),
    index,
    queries,
    k=10,
    distances=distances,
    neighbors=neighbors,
    resources=handle,
)

handle.sync()
```

See `cuvs/tests/test_*.py` for exhaustive, runnable scenarios (build/search, serialization, precision toggles).

---

## 5) Clustering (KMeans)

`cuvs.cluster.kmeans` provides GPU-accelerated KMeans:

- `fit(params, dataset[, init_centers])` — train cluster centers; returns `(centers, inertia, n_iters)`.
- `predict(dataset, centers)` — assign points to nearest centers.
- `cluster_cost(dataset, centers)` — compute sum of squared distances / inertia.
- `KMeansParams` — configuration (number of clusters, initialization, iterations).

Supports both host NumPy arrays and device arrays; tests cover accuracy against scikit-learn (see `cuvs/tests/test_kmeans.py`).

---

## 6) Distance utilities

`cuvs.distance` exports:

- `pairwise_distance(x, y, metric='sqeuclidean', output=None, resources=None)`
- `DISTANCE_NAMES` / `DISTANCE_TYPES` — enumerations aligning with RAFT/FAISS identifiers.

Distances run entirely on GPU with CUDA kernels. You can supply host or device arrays; output defaults to a new device array but can be injected.

---

## 7) Common module utilities

- `cuvs.common.device_tensor_view` — Helpers to create device views over Python buffers.
- `cuvs.common.cydlpack` — DLPack conversions for interoperability with CuPy/PyTorch.
- `cuvs.common.exceptions` — Wraps cuVS return codes (`check_cuvs`, `CuvsError`).
- `cuvs.common.resources` — As described earlier, wraps RAFT handles and offers `auto_sync_resources`.

These utilities ensure zero-copy data transfers across GPU-aware libraries.

---

## 8) Interoperability & external tooling

- **RAPIDS/RAFT**: cuVS is a thin layer atop RAPIDS RAFT; expect compatibility with RAFT’s resource handles and RAFT memory allocators.
- **CuPy / PyTorch**: use DLPack or raw pointer support (`cupy.cuda.Stream.ptr`) when coordinating streams.
- **pylibraft**: included in tests for convenience (`pylibraft.common.device_ndarray`); install it in environments where you plan to run cuVS tests or examples.

---

## 9) Integration with DocsToKG HybridSearch

- **Loader path**: `DocsToKG.HybridSearch.store._ensure_cuvs_loader_path()` preloads `libcuvs`, `librmm`, and `rapids_logger` and adjusts `LD_LIBRARY_PATH` so FAISS/cuVS extensions can import without runtime errors.
- **Capability detection**: `resolve_cuvs_state(requested)` calls `faiss.should_use_cuvs(...)`. Given the current FAISS build, the call returns `False` or raises an assertion when `use_cuvs=True`; HybridSearch therefore records `cuvs_enabled=False` and falls back to FAISS implementations. When a cuVS-enabled FAISS wheel becomes available, the same hooks will enable cuVS automatically.
- **Telemetry**: `AdapterStats` includes `cuvs_enabled`, `cuvs_available`, `cuvs_requested`, `cuvs_applied`, and HybridSearch logs decision points so operators know whether cuVS was used.

---

## 10) Testing & troubleshooting

- Run `python -m pytest cuvs/tests` inside the virtualenv to exercise all algorithms (requires GPU with CUDA 12 + sufficient VRAM).
- If you see `OSError: libcuvs_c.so: cannot open shared object file`, ensure the loader path fixes are applied (importing `DocsToKG.HybridSearch.store` once per process is sufficient) or manually prepend the required directories to `LD_LIBRARY_PATH`.
- `RuntimeError: cuVS has not been compiled into the current version` indicates the FAISS wheel lacks cuVS kernels. HybridSearch already guards this path; standalone scripts should catch the error and retry with cuVS disabled.
- Verify CUDA compatibility via `nvidia-smi` and ensure the CUDA toolkit on the host matches the versions required by cuVS (12.x).

---

## 11) Minimal smoke test

```python
import numpy as np
from cuvs.common import Resources
from cuvs.neighbors import brute_force
from pylibraft.common import device_ndarray

handle = Resources()
dataset = device_ndarray(np.random.rand(1000, 64).astype(np.float32))
queries = device_ndarray(np.random.rand(10, 64).astype(np.float32))

index = brute_force.build(brute_force.IndexParams(metric="sqeuclidean"), dataset, resources=handle)
distances = np.empty((10, 5), dtype=np.float32)
neighbors = np.empty((10, 5), dtype=np.int64)

brute_force.search(
    brute_force.SearchParams(metric="sqeuclidean"),
    index,
    queries,
    k=5,
    distances=distances,
    neighbors=neighbors,
    resources=handle,
)
handle.sync()

print("Neighbors shape:", neighbors.shape)
```

This confirms the loader path is correct, CUDA kernels execute, and resources synchronize as expected.

---

## 12) Useful references

- RAPIDS cuVS docs: <https://docs.rapids.ai/api/cuvs/stable/>
- RAFT resources API: <https://docs.rapids.ai/api/raft/stable/>
- cuVS GitHub repository: <https://github.com/rapidsai/cuvs>
- DocsToKG HybridSearch cuVS hooks: `src/DocsToKG/HybridSearch/store.py` (`resolve_cuvs_state`, `_ensure_cuvs_loader_path`)

Update this document whenever the cuVS wheel or loader behaviour changes (e.g., when FAISS is rebuilt with cuVS enabled).
