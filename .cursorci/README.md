# Continuous Integration Setup for Google Ads Agent

This directory contains CI configuration for the Google Ads Agent project, enabling automated testing, linting, and type checking on every push and pull request.

## CI Pipeline Overview

The CI pipeline consists of the following jobs:

1. **Lint**: Analyzes code quality using Flake8 and Pylint to ensure coding standards are maintained
2. **Type Check**: Validates type annotations using MyPy to catch type-related errors
3. **Test**: Runs the test suite with pytest and generates code coverage reports
4. **Build**: Verifies that the project can be successfully built and imported

## Requirements

The CI pipeline automatically installs all necessary dependencies from the `requirements.txt` file.

## Local Development

To run the same checks locally before committing:

```bash
# Install development dependencies
pip install flake8 pylint mypy pytest pytest-cov

# Run linting
flake8 --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 --count --max-complexity=10 --max-line-length=127 --statistics
pylint --disable=C0111,C0103,C0303,C0330 services/ tests/

# Run type checking
mypy --ignore-missing-imports services/

# Run tests with coverage
pytest --cov=services tests/
```

## CI Configuration

The CI workflow is defined in the `ci.yml` file, which uses GitHub Actions to automate the testing process. The workflow is triggered on pushes and pull requests to the main, master, and develop branches.

## Code Coverage Requirements

The pipeline enforces a minimum code coverage of 90% for all tests. If coverage falls below this threshold, the CI build will fail. 