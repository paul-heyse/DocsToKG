## 1. Analyse Current Dynamic Stub Coverage
- [x] 1.1 Extract from `tests/docparsing/stubs.py` the full set of dependency names currently registered in `sys.modules` (sentence_transformers, vllm, tqdm, pydantic, transformers, docling_core and its nested packages).
- [x] 1.2 For each dependency, list every attribute or class that is assigned (e.g. `SparseEncoder`, `LLM`, `PoolingParams`, `BaseModel`, `MarkdownParams`, `ChunkingDocSerializer`, etc.) in a table saved to `tests/docparsing/fake_deps/MIGRATION_NOTES.md` for reference.
- [x] 1.3 Note any helper behaviours (e.g. `_StubSparseEncoder.decode`, `_StubDoclingDocument.load_from_doctags`, serializer factories) that must be preserved verbatim so tests remain deterministic.

## 2. Scaffold Static Fake Dependency Package
- [x] 2.1 Create `tests/docparsing/fake_deps/__init__.py` explaining that this package mirrors optional production dependencies for tests and MyPy.
- [x] 2.2 Add a sentinel module path helper (e.g. define `PACKAGE_ROOT = Path(__file__).parent`) to simplify runtime path insertion from the stubs helper.
- [x] 2.3 Create subpackages mirroring each namespace currently injected:
  - `tests/docparsing/fake_deps/sentence_transformers/__init__.py` exporting `SparseEncoder`, `_StubSparseRow`, `_StubSparseBatch`, `_StubSparseValues`.
  - `tests/docparsing/fake_deps/vllm/__init__.py` exporting `_StubEmbedding`, `LLM`, `PoolingParams`.
  - `tests/docparsing/fake_deps/tqdm/__init__.py` exporting `tqdm`.
  - `tests/docparsing/fake_deps/pydantic/__init__.py` exporting `BaseModel`, `Field`, `ConfigDict`, `field_validator`, `model_validator`.
  - `tests/docparsing/fake_deps/transformers/__init__.py` exporting `AutoTokenizer`.
  - `tests/docparsing/fake_deps/docling_core/` replicating the nested layout:
    - `__init__.py` attaching `transforms` and `types`.
    - `transforms/__init__.py`, `transforms/chunker/__init__.py`, `transforms/chunker/base.py`, `transforms/chunker/hybrid_chunker.py`, `transforms/chunker/hierarchical_chunker.py`, `transforms/chunker/tokenizer/__init__.py`, `transforms/chunker/tokenizer/huggingface.py`.
    - `transforms/serializer/__init__.py`, `transforms/serializer/base.py`, `transforms/serializer/common.py`, `transforms/serializer/markdown.py`.
    - `types/__init__.py`, `types/doc/__init__.py`, `types/doc/document.py`.
- [x] 2.4 Move the corresponding class/function implementations from `dependency_stubs()` into their new modules, keeping logic identical but adjusting imports (`from __future__ import annotations`, `from typing import List, Sequence`, etc.) and defining `__all__` for clarity.
- [x] 2.5 Ensure parent packages re-export their child modules (e.g. `docling_core.transforms.chunker.hybrid_chunker` defines `HybridChunker` and `docling_core.transforms.chunker.__all__` includes `hybrid_chunker`).

## 3. Rework Test Harness Loader
- [x] 3.1 Update `tests/docparsing/stubs.py` so `dependency_stubs()` inserts `tests/docparsing` (or the absolute `fake_deps` path) onto `sys.path` if not already present.
- [x] 3.2 Replace the `_install` helper with logic that imports each fake module via `importlib.import_module("tests.docparsing.fake_deps.<module>")` and registers it under the production module name in `sys.modules` only when the real dependency is absent.
- [x] 3.3 Add a guard so that if a real dependency is installed, the helper leaves it untouched unless a `force=True` flag is passed (matching existing semantics).
- [x] 3.4 Remove the inline class definitions from `dependency_stubs()` once they are provided by the static package, leaving only the loader orchestration and optional debug logging.

## 4. Contributor Guidance
- [x] 4.1 Populate `tests/docparsing/fake_deps/README.md` (or expand `MIGRATION_NOTES.md`) with instructions for adding new fake modules, emphasising matching production namespaces and keeping functions deterministic for tests.
- [x] 4.2 Add a module-level docstring to `tests/docparsing/stubs.py` pointing developers to the static package and the documentation file for future updates.

## 5. Validation & Regression Safety
- [x] 5.1 Run `pre-commit run mypy --files tests/docparsing/stubs.py tests/docparsing/fake_deps` and confirm no `attr-defined` warnings remain.
- [x] 5.2 Execute targeted tests that rely on the stubs (e.g. `pytest tests/docparsing -k dependency_stubs`) to verify runtime parity with the prior dynamic approach.
- [x] 5.3 Add or adjust unit tests (if feasible) that explicitly import representative fake modules to guard against missing exports.
- [x] 5.4 Capture a summary of verification results in `openspec/changes/refactor-docparsing-stub-modules/tasks.md` beneath the checklist once completed.

---

### Verification Summary
- `pre-commit run mypy --files tests/docparsing/stubs.py tests/docparsing/fake_deps` (pass)
- `pytest tests/docparsing -k dependency_stubs` (3 selected tests, all pass)
