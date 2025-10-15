# Module: __init__

DocsToKG Main Package

This package provides the core functionality for DocsToKG (Document to Knowledge Graph),
a comprehensive system for transforming documents into structured knowledge graphs using
vector search, machine learning, and AI technologies.

The package includes modules for:
- Document processing and parsing
- Hybrid search capabilities (BM25 + dense retrieval)
- Content ingestion and indexing
- API interfaces and configuration management

Key Features:
- Multi-format document processing (PDF, DOCX, TXT, HTML)
- AI-powered content analysis and entity extraction
- Scalable vector similarity search using FAISS
- RESTful API for integration with external systems
- Comprehensive observability and monitoring

Usage:
    from docstokg import ping

    # Check if the package is working
    response = ping()
    print(response)  # "pong"

For detailed API documentation, see the docs/04-api/ directory.

## Functions

### `ping()`

Check if the DocsToKG package is working correctly.

This function provides a simple health check for the package.
It returns "pong" when called, indicating the package is
properly installed and importable.

Args:
None

Returns:
String indicating the package is working

Examples:
>>> from docstokg import ping
>>> response = ping()
>>> print(response)
pong
