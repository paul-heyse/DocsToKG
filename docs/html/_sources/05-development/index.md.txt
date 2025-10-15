# 1. Development Guide

Welcome to the DocsToKG development community! This guide will help you contribute to the project effectively.

## 2. Getting Started as a Contributor

### 1. Set Up Development Environment

Follow the [Setup Guide](../02-setup/index.md) to get your local environment running.

### 2. Understand the Codebase

- **Start with the overview**: Read [Architecture Guide](../03-architecture/index.md) to understand system components
- **Explore the API**: Check [API Reference](../04-api/index.md) for integration points
- **Study key modules**: Focus on `src/DocsToKG/ContentDownload`, `src/DocsToKG/DocParsing`, `src/DocsToKG/HybridSearch`, and `src/DocsToKG/OntologyDownload`

### 3. Find Your First Issue

- **Good First Issues**: Look for issues labeled `good first issue` or `help wanted`
- **Documentation**: Help improve docs, add examples, or clarify explanations
- **Bug Fixes**: Start with straightforward bug reports
- **Feature Requests**: Pick features that align with your interests

## 3. Development Workflow

### 3.1 Branch Strategy

We keep the branching model lightweight:

```
main                         # Default branch
└── feature/<description>    # Short-lived branches for changes
```

- Branch off `main` for every change (features, bug fixes, docs updates).
- Rebase or merge `main` regularly when working on longer efforts.
- Delete feature branches after merging to keep the repository tidy.

### 3.2 Documentation-First Development

DocsToKG follows a **documentation-first development** approach:

1. **Document requirements** before implementation
2. **Write API specifications** for new endpoints
3. **Create usage examples** alongside code
4. **Update documentation** as features evolve

### 3.3 Documentation Integration

#### Pre-Development Phase

**Before starting implementation:**

- [ ] Document requirements in appropriate section
- [ ] Create API specifications for new endpoints
- [ ] Define data models and relationships
- [ ] Plan examples and use cases

#### During Development

**As you implement features:**

- [ ] Write comprehensive docstrings for all public interfaces
- [ ] Update documentation as functionality evolves
- [ ] Test documentation examples to ensure they work
- [ ] Validate changes using automated tools

#### Pre-Submission Phase

**Before submitting pull requests:**

- [ ] Run `python docs/scripts/validate_code_annotations.py src/`
- [ ] Run `python docs/scripts/generate_api_docs.py` to update API docs
- [ ] Run `python docs/scripts/validate_docs.py` for content validation
- [ ] Review changes using documentation checklist

## 4. Commit Conventions

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code restructuring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

**Examples**:

```
feat(api): add document upload endpoint
fix(search): resolve vector index corruption issue
docs: update API reference with new endpoints
docs(api): add comprehensive search endpoint documentation
docs(architecture): update component diagram for new service
```

## 5. Pull Request Process

1. **Create a branch** from `main` with descriptive name (e.g., `feature/hybrid-metrics`)
2. **Make your changes** following coding standards
3. **Write tests** for new functionality
4. **Update documentation** for all user-facing changes
5. **Validate documentation** using automated tools
6. **Run tests locally** to ensure everything works
7. **Submit PR** with clear description and link to related issues

### 5.1 Documentation Requirements for PRs

**For all code changes:**

- [ ] Update docstrings for modified public interfaces
- [ ] Add usage examples for new functionality
- [ ] Update API documentation for endpoint changes
- [ ] Run `python docs/scripts/validate_code_annotations.py src/`
- [ ] Run `python docs/scripts/generate_api_docs.py`

**For new features:**

- [ ] Document requirements in appropriate section
- [ ] Create API specifications for new endpoints
- [ ] Add comprehensive usage examples
- [ ] Update architecture documentation if needed

**For bug fixes:**

- [ ] Update troubleshooting documentation
- [ ] Add test cases for the fix
- [ ] Document workaround if applicable

### 5.2 Code Review Process

- **Reviewers**: At least one maintainer review required
- **CI Checks**: All automated checks must pass
- **Testing**: New features need test coverage
- **Documentation**: Update docs for user-facing changes

## 6. Coding Standards

### 6.1 Python Standards

**Formatting**:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint and type-check
ruff check src/ tests/
mypy src/ --strict
```

**Documentation**:

- Use Google-style docstrings
- Include type hints for all function parameters and return values
- Document classes, methods, and modules comprehensively

**Example**:

```python
def process_document(
    document_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """Process a document through the ingestion pipeline.

    This function handles the complete document processing workflow
    including validation, content extraction, and knowledge graph
    construction.

    Args:
        document_id: Unique identifier for the document
        metadata: Optional metadata to associate with document

    Returns:
        ProcessingResult with status and any extracted entities

    Raises:
        DocumentProcessingError: If processing fails
        ValidationError: If document format is invalid
    """
```

### 6.2 Testing Standards

**Test Structure**:

- Hybrid search suites: `tests/test_hybrid_search.py`, `tests/test_hybrid_search_real_vectors.py`, `tests/test_hybrid_search_scale.py`
- Ontology tooling: `tests/ontology_download/` and `tests/test_resolver_pipeline.py`
- Parsing and ingestion: `tests/test_pipeline_behaviour.py`, `tests/test_content_download_utils.py`
- CLI coverage: `tests/test_cli.py`, `tests/test_download_utils.py`

**Test Coverage**:

- Aim for >80% coverage for new code
- Test both success and failure scenarios
- Use descriptive test names

**Example Test**:

```python
def test_hybrid_search_returns_results(hybrid_service, seeded_chunk_fixture):
    """Hybrid search should return seeded document when query matches."""
    request = HybridSearchRequest(query="knowledge graph", page_size=3)
    response = hybrid_service.search(request)
    assert response.results, "Expected at least one result"
    assert response.results[0].doc_id == seeded_chunk_fixture.doc_id
```

## 7. Documentation Contributions

### 7.1 Improving Existing Documentation

1. **Identify gaps**: Look for unclear explanations or missing information
2. **Check style**: Follow the [Style Guide](../STYLE_GUIDE.md)
3. **Test examples**: Verify code examples work correctly
4. **Update cross-references**: Ensure links between documents work

### 7.2 Adding New Documentation

1. **Use templates**: Check `docs/templates/` for appropriate templates
2. **Follow structure**: Maintain consistent organization
3. **Include examples**: Provide practical examples when possible
4. **Consider AI agents**: Structure content for easy parsing

## 8. Release Process

### 8.1 Version Management

We follow [Semantic Versioning](https://semver.org/):

- **Major** (x.0.0): Breaking changes
- **Minor** (x.y.0): New features, backward compatible
- **Patch** (x.y.z): Bug fixes, no new features

### 8.2 Release Checklist

- [ ] All tests pass (`pytest -q`, plus optional markers)
- [ ] Documentation regenerated and validated
- [ ] Version numbers bumped in `pyproject.toml`
- [ ] Release notes drafted (GitHub release or docs entry)
- [ ] FAISS and ontology snapshots archived for rollback

## 9. Communication

### 9.1 Development Discussions

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and community discussions

### 9.2 Getting Help

1. **Search existing issues** and discussions first
2. **Check documentation** - your question might already be answered
3. **Create a new issue** with clear description and reproduction steps
4. **Ask in discussions** for general questions

### 9.3 Contributing to This Guide

This development guide itself can be improved! If you find:

- Unclear instructions
- Missing information
- Outdated processes
- Better examples

Please contribute improvements through the same PR process.

## 10. Recognition

Contributors are recognized through:

- **GitHub Contributors** list
- **Release notes** mentioning major contributors
- **Special badges** for significant contributions
- **Community spotlight** features

## 11. Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating, you agree to:

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative
- Focus on what is best for the community

Thank you for contributing to DocsToKG! 🚀
