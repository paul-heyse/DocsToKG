# Migration Notes

The table below records the modules that `dependency_stubs()` previously created
via `ModuleType` along with the attributes we must continue to expose. Use this
list as the source of truth when verifying new fake implementations.

| Real module name | Fake module path | Exposed attributes / behaviour |
| ---------------- | ---------------- | ------------------------------ |
| `sentence_transformers` | `tests.docparsing.fake_deps.sentence_transformers` | `SparseEncoder` class supporting `.encode()` returning `_StubSparseBatch`, `.decode()` truncating token-weight pairs, and the internal `_StubSparseRow` / `_StubSparseValues` helpers used by the encoder. |
| `vllm` | `tests.docparsing.fake_deps.vllm` | `LLM` class with `.embed()` producing deterministic vectors, `PoolingParams` data holder, internal `_StubEmbedding` wrapper exposing `.outputs.embedding`. |
| `tqdm` | `tests.docparsing.fake_deps.tqdm` | `tqdm()` function returning the iterable unchanged (or an empty list) to mimic optional progress bars. |
| `pydantic` | _removed in DocsToKG 0.3.0_ | Hard dependency; tests now import the real package so no fake module is provided. |
| `transformers` | `tests.docparsing.fake_deps.transformers` | `AutoTokenizer` class with `.from_pretrained()` constructor and callable behaviour returning token lists. |
| `docling_core` + nested packages | `tests.docparsing.fake_deps.docling_core.*` | `transforms` namespace with `chunker`, `serializer` subpackages; `BaseChunk`, `HybridChunker`, `ChunkingDocSerializer`, `ChunkingSerializerProvider`, tokenizer helpers, markdown serializers, and `create_ser_result` factory. |
| `docling_core.types.doc.document` | `tests.docparsing.fake_deps.docling_core.types.doc.document` | `DoclingDocument`, `DocTagsDocument`, simple picture data classes (`PictureClassificationData`, `PictureDescriptionData`, `PictureMoleculeData`), and `PictureItem` stub exposing `.annotations` and `.caption_text`. |

All behaviours should stay deterministic and side-effect free to keep tests
predictable. When new fake functionality is added, append a new row describing
the module and expected exports.
