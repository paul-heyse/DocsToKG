# 1. Code Annotation Standards for Auto-Generation

This document establishes standards for code annotations (comments and docstrings) that enable automated documentation generation. Following these standards ensures that the `generate_api_docs.py` script and Sphinx autodoc can extract meaningful, consistent documentation from your code.

## 2. Philosophy

**Code annotations should:**

- **Enable AI agents** to understand code functionality and requirements
- **Support automated documentation** generation without manual intervention
- **Provide clear context** for human developers and reviewers
- **Follow consistent patterns** across the entire codebase

## 3. General Principles

### 1. Every Public Interface Needs Documentation

All public classes, functions, methods, and modules must have docstrings that explain:

- **Purpose**: What this code does and why it exists
- **Usage**: How to use it correctly
- **Parameters**: What inputs are expected
- **Returns**: What outputs are provided
- **Side Effects**: Any external state changes or important behaviors

### 2. Documentation-First Development

Write documentation before implementing functionality:

```python
def process_documents(document_ids: List[str]) -> Dict[str, ProcessingResult]:
    """Process multiple documents through the knowledge graph pipeline.

    This function handles batch document processing, ensuring efficient
    resource utilization and proper error handling for large document sets.

    Args:
        document_ids: List of document identifiers to process

    Returns:
        Dictionary mapping document IDs to their processing results

    Raises:
        DocumentProcessingError: If any document fails processing
        ValidationError: If input parameters are invalid
    """
    # Implementation follows...
    pass
```

### 3. Consistent Terminology

Use consistent terminology across all annotations:

- Use "document" consistently (not "file", "record", etc.)
- Use "process" for transformation operations
- Use "search" for retrieval operations
- Use "index" for vector storage structures

## 4. Module-Level Documentation

Every Python module must start with a comprehensive docstring:

```python
"""
Document Processing Service

This module provides comprehensive document processing capabilities for the
DocsToKG system, including content extraction, metadata analysis, and
knowledge graph construction.

The service handles various document formats and integrates with the
vector search and knowledge graph components to provide end-to-end
document understanding.

Key Features:
- Multi-format document parsing (PDF, DOCX, TXT, HTML)
- AI-powered content classification and entity extraction
- Integration with Faiss vector indexing
- Asynchronous processing for scalability

Dependencies:
- faiss: Vector similarity search
- transformers: AI model inference
- PyPDF2: PDF document parsing
- python-docx: Word document parsing

Usage:
    from src.processing.document_service import DocumentProcessor

    processor = DocumentProcessor()
    result = await processor.process_document("doc_123")
"""
```

## 5. Class Documentation

Classes must document their purpose, responsibilities, and key methods:

```python
class DocumentProcessor:
    """Handles document processing pipeline from ingestion to knowledge graph.

    This class orchestrates the complete document processing workflow:
    1. Document validation and format detection
    2. Content extraction and preprocessing
    3. Entity recognition and relationship extraction
    4. Vector embedding generation
    5. Knowledge graph construction

    The processor is designed to handle both individual documents and
    batch processing scenarios, with built-in error handling and
    progress tracking.

    Attributes:
        config (ProcessingConfig): Configuration settings for processing
        embedding_model (SentenceTransformer): Model for text embeddings
        vector_index (FaissIndex): Vector similarity search index

    Examples:
        >>> processor = DocumentProcessor()
        >>> result = await processor.process_document("doc_123")
        >>> print(f"Processed {result.document_count} documents")
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the document processor.

        Args:
            config: Processing configuration. Uses defaults if None.
        """
        self.config = config or ProcessingConfig()
        self.embedding_model = None  # Lazy initialization
        self.vector_index = None     # Lazy initialization
```

## 6. Function and Method Documentation

Functions must document parameters, return values, and behavior:

```python
def extract_entities(
    text: str,
    model: str = "default",
    confidence_threshold: float = 0.8
) -> List[Entity]:
    """Extract named entities from document text using AI models.

    This function uses pre-trained language models to identify and
    classify named entities (people, organizations, locations, etc.)
    within document content. The extracted entities are used for
    knowledge graph construction and semantic search enhancement.

    Args:
        text: The document text to analyze
        model: Entity recognition model to use ("default", "spacy", "transformers")
        confidence_threshold: Minimum confidence score for entity extraction

    Returns:
        List of extracted entities with their types and positions

    Raises:
        ModelLoadError: If the specified model cannot be loaded
        ProcessingError: If entity extraction fails

    Examples:
        >>> entities = extract_entities("Apple Inc. was founded by Steve Jobs")
        >>> print(f"Found {len(entities)} entities")
        >>> for entity in entities:
        ...     print(f"- {entity.text}: {entity.type}")
    """
```

## 7. Parameter Documentation

Document all parameters with their types, purposes, and constraints:

```python
def search_similar_documents(
    query: str,
    limit: int = 10,
    threshold: float = 0.7,
    filters: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True
) -> List[SearchResult]:
    """Search for documents similar to the given query.

    Args:
        query: Natural language search query
        limit: Maximum number of results to return (1-100)
        threshold: Similarity threshold for results (0.0-1.0)
        filters: Optional filters to apply to search results
        include_metadata: Whether to include document metadata in results

    Returns:
        List of search results ordered by similarity score
    """
```

## 8. Complex Type Documentation

For complex types, provide detailed explanations:

```python
def build_knowledge_graph(
    entities: List[Entity],
    relationships: List[Relationship],
    config: KnowledgeGraphConfig
) -> KnowledgeGraph:
    """Build a knowledge graph from extracted entities and relationships.

    Args:
        entities: List of extracted named entities from documents
        relationships: List of relationships between entities
        config: Configuration for knowledge graph construction including:
            - max_depth: Maximum relationship depth to include
            - relationship_types: Types of relationships to include
            - deduplication_strategy: How to handle duplicate entities

    Returns:
        Constructed knowledge graph ready for querying
    """
```

## 9. Exception Documentation

Document all exceptions that can be raised:

```python
def authenticate_user(credentials: UserCredentials) -> UserSession:
    """Authenticate a user and create a session.

    Raises:
        AuthenticationError: If credentials are invalid
        AccountLockedError: If the user account is locked
        SystemUnavailableError: If authentication service is down
    """
```

## 10. Class Method Documentation

Document all public methods, especially those that change object state:

```python
class VectorIndex:
    """Manages vector similarity search operations using Faiss."""

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the vector index.

        This method processes documents, generates embeddings, and
        adds them to the Faiss index for similarity search.

        Args:
            documents: Documents to add to the index

        Returns:
            Number of documents successfully added

        Raises:
            IndexError: If document format is incompatible
            MemoryError: If insufficient memory for index expansion
        """

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        """Search for similar vectors in the index.

        Args:
            query_vector: Query vector for similarity search
            k: Number of nearest neighbors to return

        Returns:
            List of search results with similarity scores
        """
```

## 11. Property Documentation

Document properties that expose important state:

```python
class DocumentProcessor:

    @property
    def processing_status(self) -> ProcessingStatus:
        """Current processing status of the document pipeline.

        Returns:
            Current status including active operations and queue length
        """
        return self._status

    @property
    def supported_formats(self) -> List[str]:
        """List of document formats supported by this processor.

        Returns:
            List of supported MIME types and file extensions
        """
        return self._supported_formats
```

## 12. Magic Method Documentation

Document important magic methods:

```python
def __init__(self, config: ProcessingConfig):
    """Initialize document processor with configuration.

    Args:
        config: Processing configuration containing model settings,
               performance parameters, and operational flags
    """

def __repr__(self) -> str:
    """String representation for debugging.

    Returns:
        String showing processor status and configuration summary
    """
```

## 13. Inline Comments

Use inline comments sparingly for complex logic:

```python
def process_documents_batch(self, documents: List[Document]) -> BatchResult:
    """Process a batch of documents efficiently.

    This method optimizes processing by:
    1. Grouping similar documents together
    2. Reusing model instances across documents
    3. Parallelizing independent operations
    """

    # Group documents by processing requirements to optimize resource usage
    document_groups = self._group_documents_by_requirements(documents)

    results = []
    for group in document_groups:
        # Process each group with appropriate strategy
        if len(group) == 1:
            # Single document - use standard processing
            result = self._process_single_document(group[0])
        else:
            # Multiple documents - use batch processing
            result = self._process_document_batch(group)

        results.append(result)

    return BatchResult.combine(results)
```

## 14. Constants and Configuration Documentation

Document important constants and configuration values:

```python
# Maximum number of documents that can be processed in a single batch
# This limit prevents memory issues with very large document sets
MAX_BATCH_SIZE = 100

# Default similarity threshold for search results
# Values above this threshold are considered relevant matches
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# Supported document formats with their MIME types
SUPPORTED_FORMATS = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain',
    '.html': 'text/html'
}
```

## 15. Error Handling Documentation

Document error handling strategies:

```python
def safe_document_processing(document_id: str) -> Optional[ProcessingResult]:
    """Safely process a document with comprehensive error handling.

    This function wraps document processing with error handling for:
    - File access errors (permissions, missing files)
    - Format parsing errors (corrupted files, unsupported formats)
    - Model inference errors (GPU memory, model loading failures)
    - Network errors (external service dependencies)

    Args:
        document_id: Unique identifier for the document

    Returns:
        Processing result if successful, None if processing failed

    Note:
        Failed documents are logged but do not stop batch processing
    """
```

## 16. Integration with Auto-Generation

These annotation standards are designed to work seamlessly with:

### `generate_api_docs.py`

- Extracts module, class, and function documentation
- Generates Markdown API reference from docstrings
- Preserves parameter and return value information

### Sphinx Autodoc

- Uses Google/NumPy style docstrings for HTML generation
- Links between related classes and functions
- Includes type hints in generated documentation

### AI Agent Understanding

- Provides clear context for code comprehension
- Enables requirement extraction from docstrings
- Supports automated testing and validation

## 17. Validation

All new code must pass documentation validation:

```bash
# Check documentation completeness
python docs/scripts/validate_docs.py

# Generate API documentation to verify annotations work
python docs/scripts/generate_api_docs.py

# Build Sphinx docs to ensure compatibility
python docs/scripts/build_docs.py --format html
```

## 18. Examples

See the `docs/examples/` directory for complete code examples showing proper annotation patterns for different scenarios.
