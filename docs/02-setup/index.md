# Setup Guide

This guide will help you get DocsToKG up and running on your local development environment.

## Prerequisites

Before starting the installation, ensure you have:

### System Requirements

- **Operating System**: Linux, macOS, or Windows (WSL2 recommended for Windows)
- **Memory**: 8GB RAM minimum (16GB recommended for large document processing)
- **Storage**: 10GB free disk space minimum
- **Network**: Internet connection for downloading dependencies

### Software Prerequisites

#### Required Software

- **Python 3.8+**: [Download from python.org](https://python.org/downloads/)
- **Git**: For version control [Download from git-scm.com](https://git-scm.com/downloads)
- **Docker** (optional): For containerized deployment [Download from docker.com](https://docker.com/get-started)

#### Python Packages

The following packages will be installed automatically:

- **Faiss**: Vector similarity search library
- **FastAPI**: Web framework for APIs
- **SQLAlchemy**: Database ORM
- **Transformers**: Hugging Face models
- **PyTorch**: Deep learning framework

## Quick Start Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourorg/docstokg.git
cd docstokg
```

### 2. Set Up Virtual Environment

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages:

```bash
# Install from requirements file
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Set Up Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your specific configuration:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/docstokg

# API Configuration
API_HOST=localhost
API_PORT=8000
DEBUG=True

# Vector Search Configuration
FAISS_INDEX_PATH=./data/faiss_index
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Security
SECRET_KEY=your-secret-key-here
```

### 5. Initialize Database

Set up the database schema:

```bash
# Run database migrations
python -m scripts.db_init

# Verify database connection
python -c "from app.database import engine; print('Database connected successfully')"
```

### 6. Start the Application

Launch the DocsToKG services:

```bash
# Start the API server
python -m app.main

# Or use uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

## Verification

### Test the Installation

1. **Check API Health**:

   ```bash
   curl http://localhost:8000/health
   ```

   Expected response:

   ```json
   {"status": "healthy", "version": "1.0.0"}
   ```

2. **Test Document Upload** (if sample data is available):

   ```bash
   curl -X POST http://localhost:8000/api/v1/documents \
     -H "Content-Type: application/json" \
     -d '{"title": "Test Document", "content": "This is a test document for verification."}'
   ```

### Run Tests

Execute the test suite to verify everything works:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_api.py -v
pytest tests/test_processing.py -v
```

## Development Setup

For contributors who want to modify the codebase:

### Additional Development Tools

```bash
# Install development tools
pip install black isort mypy pytest-cov pre-commit

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/
```

### IDE Configuration

**VS Code**:

- Install Python extension
- Configure format on save with Black
- Set up mypy for type checking

**PyCharm**:

- Enable Django support (even though we're using FastAPI)
- Configure Black as code formatter
- Set up pytest as test runner

## Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError` for dependencies
**Solution**: Ensure virtual environment is activated and dependencies installed

#### Database Connection Issues

**Problem**: Connection refused or authentication errors
**Solution**:

1. Verify database is running
2. Check DATABASE_URL in `.env`
3. Ensure database user has proper permissions

#### Memory Issues

**Problem**: Out of memory during document processing
**Solution**:

1. Increase system memory allocation
2. Process documents in smaller batches
3. Use more efficient embedding models

### Getting Help

If you encounter issues:

1. Check the logs in `logs/` directory
2. Review this setup guide again
3. Search existing [GitHub issues](https://github.com/yourorg/docstokg/issues)
4. Ask in [GitHub discussions](https://github.com/yourorg/docstokg/discussions)

## Next Steps

With DocsToKG installed and running:

1. **Explore the API**: See [API Reference](../04-api/) for available endpoints
2. **Process Documents**: Upload and process your first documents
3. **Customize Configuration**: Adjust settings for your use case
4. **Set Up Production**: Follow [Operations Guide](../06-operations/) for production deployment

Happy exploring! 🚀
