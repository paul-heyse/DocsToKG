# libcuvs Loader & Integration Reference

## Table of Contents
- [1) Loader workflow](#1-loader-workflow)
- [2) Integration with DocsToKG HybridSearch](#2-integration-with-docstokg-hybridsearch)
- [3) FAISS interactions](#3-faiss-interactions)
- [4) Loader path and environment tuning](#4-loader-path-and-environment-tuning)
  - [4.1 Environment variables](#41-environment-variables)
  - [4.2 Manual preload (standalone scripts)](#42-manual-preload-standalone-scripts)
- [5) Dependency chain summary](#5-dependency-chain-summary)
- [6) Validation steps](#6-validation-steps)
- [7) When upgrading FAISS/cuVS](#7-when-upgrading-faisscuvs)
- [8) Reference links](#8-reference-links)
**Package**: `libcuvs` (Python helper around cuVS shared libraries)  
**Version**: `25.10.00` (see `libcuvs/VERSION`)  
**Companion artifacts**:
- `.venv/lib/python3.13/site-packages/libcuvs/lib64/libcuvs.so`
- `.venv/lib/python3.13/site-packages/libcuvs/lib64/libcuvs_c.so`
- `.venv/lib/python3.13/site-packages/librmm/lib64/librmm.so`
- `.venv/lib/python3.13/site-packages/rapids_logger/lib64/librapids_logger.so`
- CUDA 12 runtime/toolchain (`libcudart.so.12`, `libcublas*.so.12`, `libcusolver.so.11`, `libcusparse.so.12`, `libcurand.so.10`, `libnvJitLink.so.12`)
- NCCL (`.venv/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2`)

This guide documents how the `libcuvs` loader cooperates with cuVS and FAISS, how DocsToKG ensures the shared objects are available at runtime, and what you must validate when upgrading wheels or deploying on new systems.

---

## 1) Loader workflow

`libcuvs` is a lightweight Python package whose sole job is to load the cuVS CUDA libraries into the current process. Its API surface is small:

```python
from libcuvs import load_library

handles = load_library()  # returns list of ctypes.CDLL objects
```

`libcuvs.load.load_library()`:

1. **Loads dependencies**:
   - Tries to import `libraft` and `librmm` (both RAPIDS wheels). If found, calls their respective `load_library()` helpers so `libraft.so`, `librmm.so`, and `librapids_logger.so` (plus the RAPIDS memory manager’s initialization routines) are resident. This bootstraps the RAPIDS Memory Manager (`rmm`) which cuVS (and FAISS when cuVS is enabled) relies on for device allocations.
   - If those modules are missing (conda/system installs), it assumes the libraries are on the system loader path.
2. **Honours `RAPIDS_LIBCUVS_PREFER_SYSTEM_LIBRARY`**:
   - If set to anything other than `false`, it prefers `ctypes.CDLL('libcuvs.so')` before falling back to the wheel’s copy at `libcuvs/lib64/libcuvs.so`.
   - Default behaviour loads the wheel’s bundled `.so` files first; if they’re absent (e.g., prebuilt install), it falls back to system copies.
3. **Loads the pair**: `libcuvs.so` (C++ API) and `libcuvs_c.so` (C shim used by Python bindings). Both libraries are linked against RMM, the RAPIDS logging stack, CUDA, and NCCL, so keeping those dependencies preloaded is essential.
4. Returns the `ctypes.CDLL` handles so callers can inspect `._name` or other metadata if desired.

Importing `cuvs` automatically calls `libcuvs.load_library()` (see `cuvs/__init__.py`), so `libcuvs` is the underlying mechanism that makes cuVS usable in Python environments.

---

## 2) Integration with DocsToKG HybridSearch

HybridSearch includes its own loader shim (`_ensure_cuvs_loader_path()` in `store.py`) because:

- The wheel-provided libraries (`libcuvs.so`, `libcuvs_c.so`, `librmm.so`, `librapids_logger.so`) are not on the default `LD_LIBRARY_PATH`, so `ctypes.CDLL` would fail.
- cuVS extension modules (e.g., `cuvs.neighbors.ivf_pq`) depend on those libraries *and* the CUDA toolchain. Without preloading, `ImportError: libcuvs_c.so: cannot open shared object file` would occur.

The helper performs four steps at import time:

1. Discover the package roots for `libcuvs`, `librmm`, and `rapids_logger` via `importlib.util.find_spec`.
2. Attempt to load each `.so` using `ctypes.CDLL` with `RTLD_GLOBAL`, logging (at DEBUG) any failures.
3. Merge their directories into `LD_LIBRARY_PATH` so child processes and subsequent imports inherit the search path.
4. Toggle a module-level flag to avoid repeated work.

After this hook runs, FAISS/cuVS modules can import safely, and direct calls to `libcuvs.load_library()` succeed because the C++ shared objects are already resident.

---

## 3) FAISS interactions

FAISS integrates with cuVS via optional GPU kernels (cuVS “fast distance” routines). With the current custom FAISS wheel:

- `faiss.should_use_cuvs(GpuDistanceParams)` returns `True` or `False` depending on dtype/layout and whether cuVS kernels are compiled in; `should_use_cuvs(GpuIndexConfig)` returns `False`.
- Calling `faiss.knn_gpu(..., use_cuvs=True)` raises `RuntimeError: cuVS has not been compiled into the current version so it cannot be used.` No cuVS kernels are linked into `_swigfaiss.so`.
- HybridSearch’s `resolve_cuvs_state()` uses `faiss.should_use_cuvs` to decide whether cuVS is available and records the outcome in `AdapterStats` (`cuvs_enabled`, `cuvs_available`, `cuvs_applied`, etc.). Keeping `librmm.so` preloaded ensures the RAPIDS memory pool is ready if future FAISS wheels route allocations through RMM when cuVS kernels become available.
- Despite the disabled kernels, FAISS still links against `libcuvs` for symbol resolution (the interface exists even if the implementation is stubbed). That is why ensuring `libcuvs` is loadable matters even when cuVS execution is disabled.

When FAISS is rebuilt with cuVS kernels:

- `knn_gpu(..., use_cuvs=True)` succeeds and dispatches to cuVS routines, leveraging `GpuDistanceParams.use_cuvs`.
- Additional telemetry should be captured in `AdapterStats` to confirm applied state.
- Update both the FAISS and cuVS reference docs to reflect new capabilities and testing requirements.

---

## 4) Loader path and environment tuning

### 4.1 Environment variables

- `RAPIDS_LIBCUVS_PREFER_SYSTEM_LIBRARY`: If truthy (`true`, `1`, etc.), `libcuvs.load_library()` attempts to use system-installed libraries before the wheel’s bundled copies. Default (`false`) uses wheel copies.
- `RAPIDS_LIBRAFT_PREFER_SYSTEM_LIBRARY`, `RAPIDS_LIBRMM_PREFER_SYSTEM_LIBRARY`, etc., work similarly for other RAPIDS components.
- `LD_LIBRARY_PATH`: After `_ensure_cuvs_loader_path()` runs, it contains the directories for libcuvs, librmm, and rapids_logger so child processes inherit the loader settings.

### 4.2 Manual preload (standalone scripts)

If you interact with cuVS outside DocsToKG modules, replicate the preload sequence:

```python
import ctypes
from pathlib import Path

root = Path(".venv/lib/python3.13/site-packages")

ctypes.CDLL(str(root / "rapids_logger/lib64/librapids_logger.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(root / "librmm/lib64/librmm.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(root / "libcuvs/lib64/libcuvs.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(root / "libcuvs/lib64/libcuvs_c.so"), mode=ctypes.RTLD_GLOBAL)
```

Then add parent directories to `LD_LIBRARY_PATH` or rely on `ctypes` to keep handles alive.

---

## 5) Dependency chain summary

`libcuvs` relies on:

- `libraft.so` (which depends on `librmm.so`, `librapids_logger.so`, CUDA libs, NCCL).
- `librmm.so` (also requires `librapids_logger.so` and CUDA).
- `libcuvs_c.so` & `libcuvs.so` (export cuVS C/C++ APIs).

`cuvs` Python modules rely on:

- `libcuvs.load_library()` to preload the above.
- Additional algorithm-specific kernels compiled as `.cpython-*.so` (e.g., `cuvs/neighbors/ivf_pq/ivf_pq.cpython-313-*.so`). Each of these `.so` files is dynamically linked against `libcuvs_c.so` and thus inherits its dependencies.

FAISS (custom wheel) links against:

- Standard CUDA/BLAS libraries.
- cuVS interface (for `should_use_cuvs`/`GpuDistanceParams`), though actual kernels are absent in the current build.

---

## 6) Validation steps

Run these checks when upgrading wheels or preparing deployments:

1. **ctypes smoke test**:

```python
from libcuvs import load_library
handles = load_library()
print("libcuvs libraries loaded from:", [h._name for h in handles])
```

2. **CuVS import**:

```python
import cuvs
from cuvs.neighbors import ivf_pq
```

3. **FAISS cuVS toggle**:

```python
import faiss
cfg = faiss.GpuIndexFlatConfig()
print("should_use_cuvs(config):", faiss.should_use_cuvs(cfg))
```

Expected output (current environment): `False` with a FAISS runtime error if you attempt to call `knn_gpu(..., use_cuvs=True)`.

4. **LD_LIBRARY_PATH inspection** (after importing HybridSearch):

```python
from DocsToKG.HybridSearch import store  # triggers preload
import os
print(os.environ["LD_LIBRARY_PATH"])
```

Should include the `libcuvs/lib64`, `librmm/lib64`, and `rapids_logger/lib64` directories.

---

## 7) When upgrading FAISS/cuVS

- Ensure new wheels include compatible versions of `libcuvs`, `libraft`, `librmm`, and `rapids_logger`.
- Verify CUDA toolkit compatibility (currently targeting CUDA 12.9).
- Re-run FAISS GPU smoke tests with cuVS toggled on (`DenseIndexConfig.use_cuvs=True`).
- Update HybridSearch docs (FAISS reference, cuVS reference, this doc) to reflect new behaviours, especially if `use_cuvs` becomes functional.
- Adjust `_ensure_cuvs_loader_path()` if directory structures change.

---

## 8) Reference links

- RAPIDS libcuvs source: <https://github.com/rapidsai/cuvs/tree/main/python/libcuvs>
- cuVS Python reference: <https://docs.rapids.ai/api/cuvs/stable/>
- RAFT loader docs: <https://docs.rapids.ai/api/raft/stable/>
- DocsToKG integration: `src/DocsToKG/HybridSearch/store.py`

Keep this document synchronized with wheel updates and any changes to loader behaviour or FAISS/cuVS integration. A future update enabling cuVS inside FAISS will require revisions here and in `faiss-gpu-wheel-reference.md`.
