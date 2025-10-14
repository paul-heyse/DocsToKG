# Glossary

| Term | Definition |
|------|------------|
| **DocsToKG** | Project that converts unstructured documents into knowledge-graph representations backed by hybrid search. |
| **Chunk** | A contiguous span of document text produced by the DocParsing pipeline for downstream embedding. |
| **Hybrid Search** | Retrieval strategy combining lexical (BM25), sparse (SPLADE), and dense (FAISS) signals. |
| **Namespace** | Logical grouping of documents and embeddings, allowing segmented search behaviour. |
| **FAISS Snapshot** | Serialized dense index artifact created by `DocsToKG.HybridSearch.storage.serialize_state`. |
| **Ontology Fetcher** | CLI tool (`python -m DocsToKG.OntologyDownload.cli`) that downloads and validates ontology sources. |
| **Self-hit Accuracy** | Percentage of benchmark queries that retrieve their own ground-truth chunk in evaluation datasets. |
| **Validation Harness** | Automated checks (`DocsToKG.HybridSearch.validation`) that ensure hybrid search outputs remain within guardrails. |
| **Docling** | Document parsing toolkit used by DocsToKG to transform PDFs and HTML into structured chunks. |
| **vLLM** | Optimised inference runtime used by DocParsing accelerators for language models. |

Need another term? Submit a documentation issue with the definition and canonical references.
