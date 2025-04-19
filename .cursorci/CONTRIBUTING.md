# Contributing to Google Ads Agent

Thank you for your interest in contributing to the Google Ads Agent project! This guide will help you understand our development workflow and quality standards.

## Development Process

1. **Fork the Repository**: Start by forking the repository to your own GitHub account
2. **Create a Branch**: Create a feature branch for your changes
3. **Make Changes**: Implement your changes following the coding standards
4. **Test Locally**: Run tests locally to ensure everything works as expected
5. **Submit a Pull Request**: Push your changes and create a PR against the main branch

## Code Quality Standards

All code contributions should meet the following standards:

- **Linting**: Code must pass Flake8 and Pylint checks
- **Type Annotations**: All functions and methods should include type annotations
- **Test Coverage**: New code should include tests and maintain >90% coverage
- **Documentation**: Code should be well-documented with docstrings

## Running Tests Locally

Before submitting a PR, please run the following checks locally:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 services/ tests/
pylint services/ tests/

# Run type checking
mypy services/

# Run tests with coverage
pytest --cov=services tests/
```

## CI Pipeline

Our CI pipeline will automatically run on all PRs and includes:

1. Linting with Flake8 and Pylint
2. Type checking with MyPy
3. Running tests with pytest and checking coverage
4. Build verification

## Commit Message Guidelines

- Use clear, descriptive commit messages
- Begin with a short summary line (50 chars or less)
- If needed, provide more detailed explanations after a blank line
- Reference issue numbers when relevant (e.g., "Fixes #123")

## Pull Request Process

1. Update the README.md or documentation with details of changes if needed
2. Make sure your code passes all CI checks
3. Request a review from at least one maintainer
4. The PR will be merged once it receives approval

Thank you for contributing to Google Ads Agent! 