# 1. Documentation Style Guide

This style guide establishes standards for DocsToKG documentation to ensure consistency, clarity, and accessibility for both human readers and AI agents.

## 2. Writing Principles

### 1. Audience-First Approach

- **Know your audience**: Write for both technical and non-technical readers
- **AI-friendly**: Structure content so AI agents can easily parse and understand requirements
- **Progressive disclosure**: Start with high-level concepts, then dive into details

### 2. Clarity and Simplicity

- **Use simple language**: Avoid jargon or explain it clearly
- **Be concise**: Remove unnecessary words while maintaining completeness
- **One concept per paragraph**: Each paragraph should focus on a single idea

### 3. Consistency

- **Terminology**: Use consistent terms throughout (e.g., always use "knowledge graph" not "KG")
- **Voice**: Use active voice and present tense where possible
- **Formatting**: Follow consistent Markdown patterns

## 3. Content Structure

### Headings

- Use `#` for main title, `##` for sections, `###` for subsections
- Make headings descriptive and scannable
- Use sentence case (capitalize first word only)

```markdown
# Main Title
## Section Title
### Subsection Title
```

### Lists

- Use `-` for unordered lists
- Use `1.` for ordered lists
- Keep list items parallel in structure
- Use bullet points for options, numbered steps for procedures

### Code and Commands

- Use inline `code` for file names, commands, and short code snippets
- Use code blocks for longer examples
- Specify language for syntax highlighting:

```python
def example_function(param: str) -> bool:
    """Example function with type hints."""
    return len(param) > 0
```

### Links and References

- Use descriptive link text: `[OpenSpec Instructions](https://github.com/paul-heyse/DocsToKG/blob/main/openspec/AGENTS.md)`
- Create anchors for internal links: `## Section Title {#section-title}`
- Reference requirements by ID when possible

## 4. Documentation Types

### API Documentation

```markdown
# API Endpoint: /api/v1/documents

**Method**: POST

**Description**: Upload a document for processing into the knowledge graph.

**Parameters**:
- `file` (file): The document file to upload
- `metadata` (object, optional): Additional metadata for the document

**Responses**:
- `200`: Success with document ID
- `400`: Invalid file format
- `413`: File too large

**Example**:
```bash
curl -X POST /api/v1/documents \
  -F "file=@document.pdf" \
  -F "metadata={\"title\":\"Research Paper\"}"
```

```

### Architecture Documentation
```markdown
# Component: Vector Search Service

## Overview
The Vector Search Service handles similarity search operations using Faiss.

## Responsibilities
- Index document embeddings for fast retrieval
- Execute similarity searches with configurable parameters
- Maintain index performance and memory efficiency

## Dependencies
- **Faiss**: Vector similarity search library
- **Redis**: Caching layer for search results
- **Document Processor**: Provides embeddings for indexing

## Data Flow
1. Receives document embeddings from Document Processor
2. Updates Faiss index with new vectors
3. Handles search requests and returns results
4. Caches frequent queries in Redis
```

### Setup Guides

```markdown
# Development Environment Setup

## Prerequisites
Before starting, ensure you have:
- Python 3.12 or higher
- Git for version control
- 16GB RAM recommended for parsing workloads

## Installation Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/paul-heyse/DocsToKG.git
   cd DocsToKG
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -e .
   pip install -r requirements.in                      # optional GPU stack
   pip install -r docs/build/sphinx/requirements.txt   # documentation tooling
   ```

## Verification

Run tests to verify installation:

```bash
pytest -q
```

```

## 5. Language and Tone

### Word Choice
- Use **must/shall** for requirements (not should/may)
- Use **can** for capabilities, **will** for future actions
- Avoid contractions in formal documentation
- Use inclusive language (they/them, not he/she)

### Technical Terms
- Define acronyms on first use: "Faiss (Facebook AI Similarity Search)"
- Use domain-specific terms consistently
- Link to glossary definitions when available

### Code Comments
- File headers: Purpose, dependencies, key functions
- Function comments: Parameters, return values, side effects
- Class comments: Responsibilities and key methods
- Use triple quotes for docstrings in Python

```python
"""
Vector Search Service

This module provides vector similarity search capabilities using Faiss.
Handles document indexing, search operations, and result caching.

Key Features:
- High-performance similarity search
- Automatic index optimization
- Redis-backed result caching
"""
```

## 6. Visual Elements

### Diagrams

- Use Mermaid for flowcharts and sequence diagrams
- Keep diagrams simple and focused
- Provide text alternatives for accessibility

### Screenshots

- Include only when necessary to illustrate complex UIs
- Provide descriptive captions
- Use consistent styling and annotations

### Tables

- Use for comparing options or showing structured data
- Include clear headers
- Keep tables narrow (avoid horizontal scrolling)

## 7. Maintenance Guidelines

### Regular Updates

- Review documentation monthly for accuracy
- Update examples when APIs change
- Remove outdated information promptly

### Review Process

- All changes require review before merging
- Technical accuracy verified by subject matter experts
- Style consistency checked by documentation team

### Automation

- Automated checks for broken links
- Style guide enforcement via linting tools
- Auto-generation of API documentation from code

## 8. Tools and Workflow

### Required Tools

- **Markdown editor**: VS Code, Typora, or similar
- **Mermaid**: For diagrams in Markdown
- **Vale**: For style guide enforcement
- **LinkChecker**: For broken link detection

### Workflow

1. Create content following this style guide
2. Run automated checks (`vale docs/`)
3. Check for broken links (`linkchecker docs/`)
4. Submit for review via pull request
5. Merge after approval

## 9. Examples

See the `/templates/` directory for complete examples of each documentation type.

For questions about this style guide, open an issue or reach out to the documentation team.
