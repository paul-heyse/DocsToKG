Overview
========

The DocsToKG project orchestrates end-to-end ingestion of technical
documentation into knowledge graph form. The repository is organised into four
major domains:

* **ContentDownload** – resilient acquisition of source documents.
* **DocParsing** – DocTags conversion, chunking, and embeddings.
* **HybridSearch** – GPU-accelerated retrieval pipelines.
* **OntologyDownload** – ontology synchronisation and validation.

Each domain exposes a Python API as well as CLI entry points. The Sphinx
configuration in this directory imports modules from ``src/DocsToKG`` and uses
``sphinx.ext.autosummary`` to generate API documentation from inline
docstrings. To rebuild the HTML site:

.. code-block:: bash

   sphinx-build -b html "docs/sphinx documentation" "docs/sphinx documentation/_build/html"

For incremental authoring, enable live reload with ``sphinx-autobuild``:

.. code-block:: bash

   sphinx-autobuild "docs/sphinx documentation" "docs/sphinx documentation/_build/html"
