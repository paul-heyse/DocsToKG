# Development Guide

Welcome to the DocsToKG development community! This guide will help you contribute to the project effectively.

## Getting Started as a Contributor

### 1. Set Up Development Environment

Follow the [Setup Guide](../02-setup/) to get your local environment running.

### 2. Understand the Codebase

- **Start with the overview**: Read [Architecture Guide](../03-architecture/) to understand system components
- **Explore the API**: Check [API Reference](../04-api/) for integration points
- **Study key modules**: Focus on `src/app/`, `src/processing/`, and `src/search/`

### 3. Find Your First Issue

- **Good First Issues**: Look for issues labeled `good first issue` or `help wanted`
- **Documentation**: Help improve docs, add examples, or clarify explanations
- **Bug Fixes**: Start with straightforward bug reports
- **Feature Requests**: Pick features that align with your interests

## Development Workflow

### Branch Strategy

We use a structured branching strategy:

```
main                    # Production-ready code
├── develop            # Integration branch for features
    ├── feature/xyz    # New features (branch from develop)
    ├── bugfix/xyz     # Bug fixes (branch from develop)
    └── hotfix/xyz     # Critical fixes (branch from main)
```

### Documentation-First Development

DocsToKG follows a **documentation-first development** approach:

1. **Document requirements** before implementation
2. **Write API specifications** for new endpoints
3. **Create usage examples** alongside code
4. **Update documentation** as features evolve

### Documentation Integration

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

### Commit Conventions

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

### Pull Request Process

1. **Create a branch** from `develop` with descriptive name
2. **Make your changes** following coding standards
3. **Write tests** for new functionality
4. **Update documentation** for all user-facing changes
5. **Validate documentation** using automated tools
6. **Run tests locally** to ensure everything works
7. **Submit PR** with clear description and link to related issues

#### Documentation Requirements for PRs

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

### Code Review Process

- **Reviewers**: At least one maintainer review required
- **CI Checks**: All automated checks must pass
- **Testing**: New features need test coverage
- **Documentation**: Update docs for user-facing changes

## Coding Standards

### Python Standards

**Formatting**:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
mypy src/ --ignore-missing-imports
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

### Testing Standards

**Test Structure**:

- Unit tests in `tests/test_unit/`
- Integration tests in `tests/test_integration/`
- API tests in `tests/test_api/`

**Test Coverage**:

- Aim for >80% coverage for new code
- Test both success and failure scenarios
- Use descriptive test names

**Example Test**:

```python
def test_document_processing_success():
    """Test successful document processing workflow."""
    # Arrange
    document = create_test_document()

    # Act
    result = process_document(document.id)

    # Assert
    assert result.status == "completed"
    assert len(result.entities) > 0
    assert result.processing_time < 5000  # ms
```

## Documentation Contributions

### Improving Existing Documentation

1. **Identify gaps**: Look for unclear explanations or missing information
2. **Check style**: Follow the [Style Guide](../STYLE_GUIDE.md)
3. **Test examples**: Verify code examples work correctly
4. **Update cross-references**: Ensure links between documents work

### Adding New Documentation

1. **Use templates**: Check `docs/templates/` for appropriate templates
2. **Follow structure**: Maintain consistent organization
3. **Include examples**: Provide practical examples when possible
4. **Consider AI agents**: Structure content for easy parsing

## Release Process

### Version Management

We follow [Semantic Versioning](https://semver.org/):

- **Major** (x.0.0): Breaking changes
- **Minor** (x.y.0): New features, backward compatible
- **Patch** (x.y.z): Bug fixes, no new features

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers bumped
- [ ] Release notes drafted
- [ ] Docker images built and tested

## Communication

### Development Discussions

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and community discussions
- **Discord/Slack**: For real-time chat (links in README)

### Getting Help

1. **Search existing issues** and discussions first
2. **Check documentation** - your question might already be answered
3. **Create a new issue** with clear description and reproduction steps
4. **Ask in discussions** for general questions

### Contributing to This Guide

This development guide itself can be improved! If you find:

- Unclear instructions
- Missing information
- Outdated processes
- Better examples

Please contribute improvements through the same PR process.

## Recognition

Contributors are recognized through:

- **GitHub Contributors** list
- **Release notes** mentioning major contributors
- **Special badges** for significant contributions
- **Community spotlight** features

## Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating, you agree to:

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative
- Focus on what is best for the community

Thank you for contributing to DocsToKG! 🚀
