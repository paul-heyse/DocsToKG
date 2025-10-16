# Implementation Tasks

## 1. Document Identifier Canonicalization

- [x] 1.1 Update chunker to compute relative path identifier from input directory
- [x] 1.2 Assign ChunkRow doc_id field to relative path instead of stem
- [x] 1.3 Preserve stem variable for local filename generation and logging contexts
- [x] 1.4 Verify manifest doc_id alignment with chunk row doc_id values
- [x] 1.5 Update manifest indexing logic to handle relative path identifiers

## 2. Output Path Collision Prevention

- [x] 2.1 Modify PDF task construction to compute relative path from input directory
- [x] 2.2 Update output path calculation to append relative path to output directory with suffix replacement
- [x] 2.3 Add parent directory creation logic in PDF conversion worker before writing outputs
- [x] 2.4 Verify output path mirroring matches HTML pipeline behavior
- [x] 2.5 Update manifest entries to record mirrored output paths

## 3. Schema Integrity Enforcement

- [x] 3.1 Identify duplicate field declarations in ChunkRow class definition
- [x] 3.2 Remove second occurrence of has_image_captions field definition
- [x] 3.3 Remove second occurrence of has_image_classification field definition
- [x] 3.4 Remove second occurrence of num_images field definition
- [x] 3.5 Verify field validators and constraints remain intact on retained definitions
- [x] 3.6 Confirm Pydantic model validation produces single field instances

## 4. Performance Optimization Through Caching

- [x] 4.1 Import LLM and PoolingParams types in EmbeddingV2 module header
- [x] 4.2 Declare module-level cache dictionary with tuple key of configuration parameters
- [x] 4.3 Implement cache key computation from QwenCfg attributes
- [x] 4.4 Add cache lookup logic at qwen_embed function entry
- [x] 4.5 Instantiate and cache LLM on cache miss
- [x] 4.6 Return cached LLM instance on cache hit
- [x] 4.7 Verify vector output consistency with and without caching

## 5. Code Deduplication and Standardization

- [x] 5.1 Remove first assignment of uuids list in process_chunk_file_vectors
- [x] 5.2 Remove first assignment of texts list in process_chunk_file_vectors
- [x] 5.3 Delete iter_chunk_files function definition from EmbeddingV2
- [x] 5.4 Import iter_chunks from_common module
- [x] 5.5 Replace iter_chunk_files calls with iter_chunks
- [x] 5.6 Wrap iter_chunks result with list() where materialized list required

## 6. Legacy Module Deprecation

- [x] 6.1 Add deprecation warning to pdf_pipeline submodule registration in __init__.py
- [x] 6.2 Add deprecation warning to html_pipeline submodule registration in __init__.py
- [x] 6.3 Update test fixtures to monkeypatch pipelines module directly
- [x] 6.4 Remove or mark pdf_pipeline.py for removal in subsequent release
- [x] 6.5 Update documentation to reference unified CLI only
- [x] 6.6 Verify no internal code imports deprecated modules

## 7. Configuration Externalization

- [x] 7.1 Create HuggingFace home resolution function checking HF_HOME environment variable
- [x] 7.2 Create model root resolution function checking DOCSTOKG_MODEL_ROOT with fallback to HF home
- [x] 7.3 Create PDF model resolution function checking DOCLING_PDF_MODEL with conventional path fallback
- [x] 7.4 Replace DEFAULT_MODEL_PATH constant with resolved path from environment
- [x] 7.5 Preserve --model CLI flag override mechanism
- [x] 7.6 Add logging of resolved model paths for observability
- [x] 7.7 Update documentation with environment variable precedence rules

## 8. Logging Standardization

- [x] 8.1 Identify all print() statements in pipelines.py
- [x] 8.2 Replace force mode print statement in html_main with logger.info
- [x] 8.3 Replace resume mode print statement in html_main with logger.info
- [x] 8.4 Verify JSON structured logging captures equivalent information
- [x] 8.5 Confirm no unstructured output remains in stdout

## 9. Directory Naming Alignment

- [x] 9.1 Audit README documentation for embedding output directory references
- [x] 9.2 Audit code for data_vectors function directory name
- [x] 9.3 Choose canonical name (Vectors or Embeddings) based on project convention
- [x] 9.4 Update either code or documentation to achieve alignment
- [x] 9.5 Update CLI examples and manifest path references for consistency
- [x] 9.6 Add migration note if directory name changes in code

## 10. Optional Enhancement - Multiprocessing Standardization

- [x] 10.1 Create set_spawn_or_warn helper function in_common module
- [x] 10.2 Implement spawn start method configuration with RuntimeError handling
- [x] 10.3 Add optional logger parameter for diagnostic output
- [x] 10.4 Replace inline spawn setup in pdf_main with helper call
- [x] 10.5 Replace inline spawn setup in html_main with helper call
- [x] 10.6 Verify CUDA safety guarantees maintained across both pipelines

## 11. Validation and Testing

- [x] 11.1 Create integration test for PDF basename collision scenario with mirrored outputs
- [x] 11.2 Create unit test for chunk doc_id relative path consistency
- [x] 11.3 Create unit test for ChunkRow schema field uniqueness
- [x] 11.4 Create performance test confirming LLM cache effectiveness
- [x] 11.5 Create deprecation warning test for legacy module imports
- [x] 11.6 Create configuration resolution test with environment variable overrides
- [x] 11.7 Create logging format test confirming JSON structure
- [x] 11.8 Execute full pipeline smoke test with new behavior
- [x] 11.9 Verify manifest entries maintain backward compatibility
- [x] 11.10 Validate resume functionality with new identifier scheme
