# Contributing to polyID

Thank you for your interest in contributing to polyID! We welcome contributions from the community.

## Ways to Contribute

- Report bugs and issues
- Suggest new features
- Submit code changes
- Improve documentation
- Help with testing

## Development Setup

1. Follow the [installation guide](Installation.md)
2. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We follow these coding standards:

- **Python**: PEP 8 with Black formatting
- **Docstrings**: Google style
- **Type hints**: Required for new code
- **Testing**: 100% coverage for new features

### Code Formatting

```bash
# Format code
black polyid/

# Check style
flake8 polyid/

# Sort imports
isort polyid/
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=polyid --cov-report=html

# Specific test file
pytest tests/test_specific.py
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation if needed
7. Commit with clear messages
8. Push to your fork
9. Create a Pull Request

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No breaking changes without discussion

## Issue Reporting

When reporting bugs:

1. Use the [bug report template](../ISSUE_TEMPLATE/bug_report.md)
2. Include Python version, OS, and polyID version
3. Provide minimal reproducible example
4. Include error messages and stack traces

## Feature Requests

For new features:

1. Check existing issues first
2. Use the [feature request template](../ISSUE_TEMPLATE/feature_request.md)
3. Describe the problem and proposed solution
4. Consider implementation complexity

## Code of Conduct

Please be respectful and constructive in all interactions. We follow a code of conduct to ensure a positive community.

## Getting Help

- [GitHub Discussions](https://github.com/your-repo/polyID/discussions)
- [Documentation](../docs/README.md)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/polyid) (tag: polyid)

Thank you for contributing to polyID!