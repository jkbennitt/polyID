# GitHub Configuration for polyID

This directory contains the GitHub configuration for the polyID project, a machine learning tool for polymer property prediction.

## Directory Structure

```
.github/
├── ISSUE_TEMPLATE/
│   ├── bug_report.md          # Bug report template
│   └── feature_request.md     # Feature request template
├── wiki/
│   ├── Home.md                # Wiki home page
│   ├── Installation.md        # Installation guide
│   ├── Quick-Start.md         # Quick start guide
│   └── Contributing.md        # Contributing guidelines
└── workflows/
    ├── ci.yml                 # Continuous integration workflow
    ├── pypi_publish.yml       # PyPI publishing workflow
    └── pypi_test_publish.yml  # TestPyPI publishing workflow
```

## Workflows

### CI Workflow (ci.yml)
- **Triggers**: Push and PR to main/master/develop branches
- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **Steps**:
  - Environment setup with conda
  - Code linting (flake8, black, isort)
  - Test execution with coverage
  - Coverage reporting to Codecov
  - Build artifact upload

### Publishing Workflows
- **pypi_publish.yml**: Publishes to PyPI on tags from master branch
- **pypi_test_publish.yml**: Publishes to TestPyPI on tags from develop branch
- Both include error monitoring and log artifact uploads

## Issue Templates

Two issue templates are provided:
- **Bug Report**: Structured template for reporting bugs with environment details
- **Feature Request**: Template for suggesting new features with use cases

## Wiki Documentation

Initial wiki content files are provided in the `wiki/` directory. These can be used to populate the GitHub Wiki or as reference documentation.

## Secrets Required

For publishing workflows, the following secrets need to be configured in repository settings:
- `PYPI_TOKEN`: Token for PyPI publishing
- `PYPI_TEST_TOKEN`: Token for TestPyPI publishing

## Error Monitoring

Workflows include comprehensive error handling:
- Artifact uploads for logs on both success and failure
- Failure-specific logs with diagnostic information
- Retention policies for artifacts (30 days for success, 7 days for failures)

## Maintenance

- Keep workflow actions updated to latest versions
- Review and update Python versions in CI matrix as needed
- Monitor workflow run times and optimize as project grows
- Update wiki content as features are added