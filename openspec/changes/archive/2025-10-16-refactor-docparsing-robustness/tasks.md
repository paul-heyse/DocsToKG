# Implementation Tasks

## 1. Make doc_id unambiguous across stages (use relative path)

- [x] 1.1 **Update DoclingHybridChunkerPipelineWithMin.py**: Change `ChunkRow.doc_id = name` to `ChunkRow.doc_id = rel_id` in main() function
- [x] 1.2 **Verify**: Ensure `rel_id = path.relative_to(in_dir).as_posix()` is computed correctly
- [x] 1.3 **Preserve**: Keep `name = path.stem` for local filename generation and logging contexts only
- [x] 1.4 **Test**: Verify chunk rows now match manifest `doc_id`s perfectly (no basename collisions)

## 2. Prevent PDF output collisions by mirroring input layout

- [x] 2.1 **Update pipelines.py pdf_main()**: Change `out_path = output_dir / (pdf_path.stem + ".doctags")` to `out_path = (output_dir / rel).with_suffix(".doctags")`
- [x] 2.2 **Update pdf_convert_one()**: Add `task.output_path.parent.mkdir(parents=True, exist_ok=True)` before writing outputs
- [x] 2.3 **Verify**: Ensure nested directory structure is created for subdirectory PDFs
- [x] 2.4 **Test**: Confirm PDF outputs mirror input directory structure (matches HTML pipeline behavior)

## 3. Remove duplicate fields in schemas.ChunkRow (Pydantic footgun)

- [x] 3.1 **Update schemas.py**: Remove the second triplet of `has_image_captions`, `has_image_classification`, `num_images` fields from ChunkRow class
- [x] 3.2 **Keep**: Retain the first triplet defined earlier in the file
- [x] 3.3 **Verify**: Ensure Pydantic model validation produces single field instances
- [x] 3.4 **Test**: Confirm no field override warnings during model construction

## 4. Cache the Qwen LLM (big perf win)

- [x] 4.1 **Update EmbeddingV2.py**: Add `_QWEN_LLM_CACHE: Dict[Tuple[str, str, int, float, Optional[str]], LLM] = {}` module-level cache
- [x] 4.2 **Update qwen_embed()**: Replace direct LLM instantiation with cache lookup using key `(str(cfg.model_dir), cfg.dtype, int(cfg.tp), float(cfg.gpu_mem_util), cfg.quantization)`
- [x] 4.3 **Implement**: Add cache miss logic to instantiate and store LLM instance
- [x] 4.4 **Verify**: Ensure identical output vectors regardless of cache state
- [x] 4.5 **Test**: Confirm performance improvement from eliminating repeated model initialization

## 5. Kill tiny duplication & share iterators in EmbeddingV2

- [x] 5.1 **Update EmbeddingV2.py**: Remove dead first assignment of `uuids` and `texts` lists in `process_chunk_file_vectors`
- [x] 5.2 **Remove**: Delete `iter_chunk_files()` function definition from EmbeddingV2
- [x] 5.3 **Import**: Add `from DocsToKG.DocParsing._common import iter_chunks`
- [x] 5.4 **Replace**: Change `files = iter_chunk_files(chunks_dir)` to `files = list(iter_chunks(chunks_dir))`
- [x] 5.5 **Verify**: Ensure consistent file discovery behavior across pipeline stages

## 6. Resolve PDF model default from env/cache instead of user path

- [x] 6.1 **Update pipelines.py**: Remove hard-coded `DEFAULT_MODEL_PATH = "/home/paul/hf-cache/granite-docling-258M"`
- [x] 6.2 **Add**: Implement `_hf_home()` function checking `HF_HOME` environment variable
- [x] 6.3 **Add**: Implement `_model_root(hf)` function checking `DOCSTOKG_MODEL_ROOT` with fallback to HF home
- [x] 6.4 **Add**: Implement `_default_pdf_model(root)` function checking `DOCLING_PDF_MODEL` with conventional path fallback
- [x] 6.5 **Update**: Set `DEFAULT_MODEL_PATH = str(_default_pdf_model(_model_root(_hf_home())))`
- [x] 6.6 **Preserve**: Keep CLI `--model` override mechanism intact
- [x] 6.7 **Test**: Verify pipeline starts successfully with various environment configurations

## 7. Remove stray print() in HTML pipeline (structured logs only)

- [x] 7.1 **Update pipelines.py html_main()**: Remove `print("Force mode: reprocessing all documents")` statement
- [x] 7.2 **Remove**: Delete `print("Resume mode enabled: unchanged outputs will be skipped")` statement
- [x] 7.3 **Verify**: Ensure structured logs already cover these messages
- [x] 7.4 **Test**: Confirm no unstructured output appears on stdout

## 8. Deprecate legacy shims and then remove them

- [x] 8.1 **Update **init**.py**: Add `import warnings` and deprecation warning for legacy submodule imports
- [x] 8.2 **Add**: Implement warning message: "DocsToKG.DocParsing.pdf_pipeline and .html_pipeline are deprecated. Import DocsToKG.DocParsing.pipelines instead. These shims will be removed in the next release."
- [x] 8.3 **Mark**: Add comments indicating shims will be removed in next release
- [x] 8.4 **Test**: Verify deprecation warnings appear when importing legacy modules
- [x] 8.5 **Plan**: Remove `pdf_pipeline.py` lightweight module once tests migrate to direct pipelines usage

## 9. Centralize spawn setup (avoid drift)

- [x] 9.1 **Add to _common.py**: Implement `set_spawn_or_warn(logger: Optional[logging.Logger] = None)` helper function
- [x] 9.2 **Implement**: Add `mp.set_start_method("spawn", force=True)` with RuntimeError handling
- [x] 9.3 **Add**: Include warning logic for incompatible start methods
- [x] 9.4 **Update pipelines.py**: Import `set_spawn_or_warn` from _common
- [x] 9.5 **Replace**: Change inline spawn setup in `pdf_main()` to `set_spawn_or_warn(logger)`
- [x] 9.6 **Replace**: Change inline spawn setup in `html_main()` to `set_spawn_or_warn(_LOGGER)`
- [x] 9.7 **Test**: Verify CUDA safety guarantees maintained across both pipelines

## 10. (Optional) Streaming Pass-B vector writing to reduce peak RAM

- [x] 10.1 **Add to EmbeddingV2.py**: Implement `iter_rows_in_batches(path: Path, batch_size: int)` helper function
- [x] 10.2 **Refactor**: Update `process_chunk_file_vectors()` to stream in batches instead of loading full file
- [x] 10.3 **Implement**: Use `atomic_write()` context manager for streaming output
- [x] 10.4 **Process**: Handle SPLADE and Qwen embedding generation in batches
- [x] 10.5 **Write**: Stream vector rows immediately instead of accumulating in memory
- [x] 10.6 **Test**: Verify identical output schema with reduced memory usage

## 11. Parameterize VLM prompt/stop tokens for PDFs

- [x] 11.1 **Update pipelines.py pdf_build_parser()**: Add `--vlm-prompt` argument with default "Convert this page to docling."
- [x] 11.2 **Add**: Implement `--vlm-stop` argument with default `["</doctag>", "<|end_of_text|>"]`
- [x] 11.3 **Update PdfTask dataclass**: Add `vlm_prompt: str` and `vlm_stop: Tuple[str, ...]` fields
- [x] 11.4 **Update pdf_main()**: Pass prompt and stop tokens to PdfTask constructor
- [x] 11.5 **Update pdf_convert_one()**: Use `task.vlm_prompt` and `task.vlm_stop` in ApiVlmOptions
- [x] 11.6 **Test**: Verify VLM parameters can be customized via CLI arguments

## 12. Align Vectors vs Embeddings directory naming

- [x] 12.1 **Choose approach**: Either update docs to "Vectors" or change code to "Embeddings"
- [x] 12.2 **Option A**: Update README documentation to reference "Vectors" directory
- [x] 12.3 **Option B**: Change `data_vectors()` function to return "Embeddings" instead of "Vectors"
- [x] 12.4 **Update**: Ensure consistency between code implementation and documentation
- [x] 12.5 **Test**: Verify directory naming alignment across all references

## 13. Create comprehensive tests

- [x] 13.1 **Add test_pdf_output_paths.py**: Test PDF basename collision prevention with mirrored outputs
- [x] 13.2 **Add test_chunk_doc_id_consistency.py**: Test chunk doc_id matches relative path
- [x] 13.3 **Add test_llm_caching.py**: Test Qwen LLM cache effectiveness and output consistency
- [x] 13.4 **Add test_configuration_resolution.py**: Test environment variable precedence for model paths
- [x] 13.5 **Add test_deprecation_warnings.py**: Test legacy module import warnings
- [x] 13.6 **Add test_spawn_configuration.py**: Test multiprocessing spawn setup
- [x] 13.7 **Execute**: Run full pipeline smoke test with new behavior
- [x] 13.8 **Verify**: Confirm manifest entries maintain backward compatibility
- [x] 13.9 **Validate**: Test resume functionality with new identifier scheme

## 14. Legacy shim module cleanup

- [x] 14.1 **Update pdf_pipeline.py**: Make `list_pdfs` recursive using `rglob("*.pdf")`
- [x] 14.2 **Implement**: Write mirrored outputs while forwarding to `manifest_append`
- [x] 14.3 **Plan**: Remove `pdf_pipeline.py` after tests switch to production entrypoint
- [x] 14.4 **Migrate**: Update test fixtures to monkeypatch `pipelines.pdf_convert_one` directly
- [x] 14.5 **Cleanup**: Remove legacy shim module in next release
