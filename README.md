# DocsToKG

**Document to Knowledge Graph** - A comprehensive system for transforming documents into structured knowledge graphs using vector search, machine learning, and AI technologies.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.docstokg.dev)

## 🚀 Overview

DocsToKG transforms documents into intelligent, searchable knowledge graphs that enable:

- **Intelligent Document Processing**: Extract and structure content from various document formats
- **Knowledge Graph Construction**: Build interconnected representations of document content
- **Semantic Search**: Natural language queries across document collections
- **AI-Powered Analysis**: Modern language models for content understanding and classification

### Key Features

- **Multi-format Support**: Process PDF, DOCX, TXT, HTML, and other document types
- **Scalable Architecture**: Handle large document collections efficiently
- **AI Integration**: Leverage modern language models for content understanding
- **Vector Search**: High-performance similarity search using FAISS
- **RESTful API**: Complete API for integration with other systems
- **Comprehensive Documentation**: Extensive guides and references

## 📚 Documentation

Our documentation is organized for easy navigation and comprehensive coverage:

### 📋 **[Getting Started](./docs/01-overview/)**

- Project overview and architecture
- Key capabilities and use cases
- Technology stack overview

### ⚙️ **[Setup Guide](./docs/02-setup/)**

- Installation and configuration
- Development environment setup
- Quick start examples

### 🏗️ **[Architecture](./docs/03-architecture/)**

- System design and components
- Data flow and integration points
- Performance considerations

### 🔌 **[API Reference](./docs/04-api/)**

- Complete REST API documentation
- Authentication and usage examples
- Integration guides

### 👥 **[Development](./docs/05-development/)**

- Contributing guidelines
- Development workflow
- Code standards and practices

### 📖 **[Technical Reference](./docs/07-reference/)**

- FAISS integration guide
- External dependencies
- Performance optimization

### 📋 **[Documentation Framework](./docs/)**

- Documentation standards and processes
- Review and maintenance procedures
- Contribution guidelines

## 🛠️ Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Git for version control

### Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/docstokg.git
cd docstokg

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/ -v
```

### Basic Usage

```python
from docstokg import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process a document
result = await processor.process_document("path/to/document.pdf")
print(f"Extracted {len(result.entities)} entities")

# Search for similar documents
similar_docs = processor.search_similar("machine learning algorithms", k=10)
```

## Parallel Execution

Use the ``--workers`` flag to enable bounded parallelism when downloading content via
the OpenAlex pipeline:

```bash
# Sequential (default, safest)
python -m DocsToKG.ContentDownload.download_pyalex_pdfs --workers 1 --topic "oncology" --year-start 2020 --year-end 2024

# Parallel (2-5x throughput)
python -m DocsToKG.ContentDownload.download_pyalex_pdfs --workers 3 --topic "oncology" --year-start 2020 --year-end 2024
```

**Recommendations:**

- Start with ``--workers=3`` for production workloads.
- Monitor rate limit compliance with resolver APIs while scaling.
- Higher values (>5) may overwhelm resolver providers despite per-resolver rate limiting.
- Each worker maintains its own HTTP session with retry logic.

### Additional CLI Flags

- ``--dry-run``: compute resolver coverage without writing files.
- ``--resume-from <manifest.jsonl>``: skip works already recorded as successful.
- ``--extract-html-text``: save plaintext alongside HTML fallbacks (requires ``trafilatura``).
- ``--enable-resolver openaire`` (and ``hal``/``osf``): opt into additional EU/preprint resolvers.

### Troubleshooting Content Downloads

- **Partial files remain (``*.part``)** – rerun with fewer workers or check network
  stability before retrying.
- **Resolver rate limit warnings** – lower ``--workers`` or increase per-resolver
  ``resolver_min_interval_s``.
- **High memory usage** – reduce ``--workers`` to limit in-flight downloads.

### Logging and Exports

- Attempts log to JSONL by default. Convert to CSV with
  ``python scripts/export_attempts_csv.py attempts.jsonl attempts.csv``.
- Alternatively, use ``jq``:
  ``jq -r '[.timestamp,.work_id,.status,.url] | @csv' attempts.jsonl > attempts.csv``.

## 🔧 Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run tests
pytest tests/ -v

# Check code quality
flake8 src/ tests/
mypy src/ --ignore-missing-imports
```

### Running Documentation

```bash
# Generate all documentation
python docs/scripts/generate_all_docs.py

# Quick documentation build (skip validation)
python docs/scripts/generate_all_docs.py --quick

# Validate documentation only
python docs/scripts/generate_all_docs.py --validate-only

# Build HTML documentation
python docs/scripts/build_docs.py --format html
```

### Contributing

We welcome contributions! Please see our [Development Guide](./docs/05-development/) for detailed contribution guidelines.

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes following our standards
4. **Test** your changes thoroughly
5. **Submit** a pull request with a clear description

## 📊 Project Structure

```
docstokg/
├── src/                    # Source code
│   ├── docstokg/          # Main package
│   ├── processing/        # Document processing modules
│   └── search/            # Search and indexing modules
├── docs/                  # Documentation
│   ├── 01-overview/       # Project overview
│   ├── 02-setup/          # Installation guide
│   ├── 03-architecture/   # System architecture
│   ├── 04-api/            # API reference
│   ├── 05-development/    # Development guide
│   ├── 07-reference/      # Technical references
│   ├── scripts/           # Documentation automation
│   └── templates/         # Documentation templates
├── tests/                 # Test suite
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── pyproject.toml        # Project configuration
```

## 🎯 Use Cases

### Research & Academia

- Literature review automation
- Cross-document concept linking
- Research trend analysis

### Enterprise Knowledge Management

- Document discovery and retrieval
- Knowledge base construction
- Compliance and audit support

### Content Management

- Article and blog organization
- Content recommendation systems
- Automated tagging and categorization

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_api.py -v
pytest tests/test_processing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Performance testing
pytest tests/test_performance.py -v
```

## 📈 Performance

### Benchmarks

- **Document Processing**: ~2-5 seconds per document (depending on size)
- **Search Latency**: <100ms for similarity search queries
- **Index Construction**: Scales to millions of documents
- **Memory Usage**: Optimized for various hardware configurations

### Scalability

- **Horizontal Scaling**: Support for multiple processing nodes
- **GPU Acceleration**: FAISS GPU support for high-performance indexing
- **Batch Processing**: Efficient handling of large document collections

## 🔒 Security

- Input validation and sanitization
- Secure API authentication
- Safe document processing
- Regular security audits

## 📞 Support

### Getting Help

- **📖 Documentation**: Comprehensive guides and references
- **🐛 Issues**: [GitHub Issues](https://github.com/yourorg/docstokg/issues) for bug reports
- **💬 Discussions**: [GitHub Discussions](https://github.com/yourorg/docstokg/discussions) for questions
- **📧 Email**: Contact the development team

### Community

- **Contributors**: Join our community of contributors
- **Discussions**: Participate in technical discussions
- **Feedback**: Help improve the project

## 📋 Roadmap

### Current Release (v1.0.0)

- ✅ Core document processing pipeline
- ✅ Vector similarity search with FAISS
- ✅ RESTful API for integration
- ✅ Comprehensive documentation framework
- ✅ Automated testing and validation

### Upcoming Features

- 🔄 Enhanced AI model integration
- 🔄 Advanced knowledge graph features
- 🔄 Multi-language support
- 🔄 Enterprise security features

## 🙏 Acknowledgments

- **FAISS Team** for the excellent vector search library
- **Hugging Face** for transformer models and tools
- **Open Source Community** for various libraries and tools
- **Contributors** for their valuable input and improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the developer community**

*For detailed technical information, see our [Documentation](./docs/) and [API Reference](./docs/04-api/).*
