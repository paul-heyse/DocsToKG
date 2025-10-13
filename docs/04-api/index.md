# API Reference

This section provides comprehensive documentation for the DocsToKG REST API. All endpoints follow RESTful conventions and return JSON responses.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Most endpoints require authentication via Bearer token in the Authorization header:

```
Authorization: Bearer <your-token>
```

## Common Response Formats

### Success Response

```json
{
  "status": "success",
  "data": {
    // Response data
  },
  "message": "Operation completed successfully"
}
```

### Error Response

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      // Additional error context
    }
  }
}
```

## Document Management

### Upload Document

**POST** `/documents`

Upload a new document for processing into the knowledge graph.

**Headers**:

- `Content-Type: multipart/form-data` or `application/json`

**Parameters** (multipart/form-data):

- `file` (file): Document file (PDF, DOCX, TXT, etc.)
- `metadata` (JSON string, optional): Additional document metadata

**Parameters** (application/json):

```json
{
  "title": "Document Title",
  "content": "Document content as text",
  "metadata": {
    "author": "Author Name",
    "source": "Source URL",
    "tags": ["tag1", "tag2"]
  }
}
```

**Response**:

```json
{
  "status": "success",
  "data": {
    "id": "doc_123",
    "title": "Document Title",
    "status": "processing",
    "created_at": "2025-01-13T10:30:00Z"
  }
}
```

### Get Document

**GET** `/documents/{document_id}`

Retrieve document information and metadata.

**Response**:

```json
{
  "status": "success",
  "data": {
    "id": "doc_123",
    "title": "Document Title",
    "content": "Document content...",
    "metadata": {
      "author": "Author Name",
      "word_count": 1500,
      "processing_status": "completed"
    },
    "created_at": "2025-01-13T10:30:00Z",
    "updated_at": "2025-01-13T10:35:00Z"
  }
}
```

### List Documents

**GET** `/documents`

List documents with optional filtering and pagination.

**Query Parameters**:

- `limit` (int, optional): Number of results per page (default: 20)
- `offset` (int, optional): Pagination offset (default: 0)
- `status` (string, optional): Filter by processing status
- `tags` (string, optional): Filter by tags (comma-separated)
- `search` (string, optional): Search in title and content

**Response**:

```json
{
  "status": "success",
  "data": {
    "documents": [
      {
        "id": "doc_123",
        "title": "Document Title",
        "status": "completed",
        "created_at": "2025-01-13T10:30:00Z"
      }
    ],
    "pagination": {
      "total": 150,
      "limit": 20,
      "offset": 0,
      "has_next": true
    }
  }
}
```

### Update Document

**PUT** `/documents/{document_id}`

Update document metadata or content.

**Request Body**:

```json
{
  "title": "Updated Title",
  "metadata": {
    "tags": ["updated", "important"]
  }
}
```

### Delete Document

**DELETE** `/documents/{document_id}`

Remove a document from the system.

**Response**:

```json
{
  "status": "success",
  "message": "Document deleted successfully"
}
```

## Search and Query

### Semantic Search

**POST** `/search`

Perform semantic search across documents using natural language queries.

**Request Body**:

```json
{
  "query": "machine learning algorithms for document classification",
  "limit": 10,
  "threshold": 0.7,
  "filters": {
    "tags": ["ml", "classification"],
    "date_from": "2024-01-01",
    "date_to": "2025-12-31"
  }
}
```

**Response**:

```json
{
  "status": "success",
  "data": {
    "query": "machine learning algorithms for document classification",
    "results": [
      {
        "document_id": "doc_123",
        "title": "ML Classification Techniques",
        "score": 0.89,
        "snippet": "This document discusses various machine learning algorithms...",
        "metadata": {
          "page": 5,
          "section": "Classification Methods"
        }
      }
    ],
    "total_results": 1,
    "execution_time_ms": 245
  }
}
```

### Vector Similarity Search

**POST** `/search/vector`

Search using vector embeddings for exact similarity matching.

**Request Body**:

```json
{
  "vector": [0.1, 0.2, 0.3, ...], // Embedding vector
  "limit": 10,
  "index_type": "document_chunks"
}
```

### Get Search Suggestions

**GET** `/search/suggestions?q={query}`

Get autocomplete suggestions for search queries.

## Knowledge Graph Operations

### Get Document Entities

**GET** `/documents/{document_id}/entities`

Extract and return entities (people, organizations, concepts) from a document.

**Response**:

```json
{
  "status": "success",
  "data": {
    "entities": [
      {
        "text": "Machine Learning",
        "type": "CONCEPT",
        "confidence": 0.95,
        "position": {
          "start": 120,
          "end": 135
        }
      }
    ]
  }
}
```

### Get Document Relationships

**GET** `/documents/{document_id}/relationships`

Get relationships between entities within a document.

## Analytics and Insights

### Document Statistics

**GET** `/analytics/documents`

Get overall statistics about the document collection.

**Response**:

```json
{
  "status": "success",
  "data": {
    "total_documents": 1250,
    "total_chunks": 15000,
    "avg_processing_time_ms": 2500,
    "top_tags": ["research", "ml", "api"],
    "storage_used_mb": 850
  }
}
```

### Search Analytics

**GET** `/analytics/searches`

Get insights about search patterns and popular queries.

## System Administration

### Health Check

**GET** `/health`

Check system health and status.

**Response**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "vector_search": "healthy",
    "embedding_service": "healthy"
  },
  "uptime_seconds": 3600
}
```

### System Information

**GET** `/system/info`

Get detailed system information and configuration.

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Malformed or missing required parameters |
| `UNAUTHORIZED` | Authentication required or invalid |
| `NOT_FOUND` | Requested resource does not exist |
| `RATE_LIMITED` | Too many requests, try again later |
| `PROCESSING_FAILED` | Document processing encountered an error |
| `SERVICE_UNAVAILABLE` | Required service is temporarily unavailable |

## Rate Limits

- **Search endpoints**: 100 requests per minute
- **Document upload**: 10 uploads per minute
- **General API**: 1000 requests per hour

Rate limits are applied per API key/user.

## SDKs and Libraries

While you can use the API directly, we provide SDKs for popular languages:

- **Python SDK**: `pip install docstokg-python`
- **JavaScript SDK**: `npm install docstokg-js`
- **Go SDK**: `go get github.com/yourorg/docstokg-go`

For examples and more information, visit the [GitHub repository](https://github.com/yourorg/docstokg).
