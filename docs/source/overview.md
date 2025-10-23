# Overview

DocsToKG orchestrates end-to-end ingestion of technical documentation into a knowledge graph. The repository is organised into four major domains:

- **ContentDownload** – resilient acquisition of source documents.
- **DocParsing** – DocTags conversion, chunking, and embeddings.
- **HybridSearch** – GPU-accelerated retrieval pipelines.
- **OntologyDownload** – ontology synchronisation and validation.

Each domain exposes a Python API as well as CLI entry points. Build the HTML documentation with:

```bash
make -C docs dirhtml
```

For incremental authoring with live reload:

```bash
sphinx-autobuild docs/source docs/build/dirhtml
```

