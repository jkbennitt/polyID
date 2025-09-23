# Installation

This guide will help you install polyID on your system.

## Prerequisites

- Python 3.8 or higher
- Conda or Miniconda (recommended for environment management)
- Git

## Quick Install

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/polyID.git
   cd polyID
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate polyid
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## Detailed Installation

### Environment Setup

polyID uses a conda environment to manage dependencies. The `environment.yml` file contains all required packages:

```yaml
name: polyid
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - numpy
  - pandas
  - scikit-learn
  - rdkit
  # ... other dependencies
```

### Alternative Installation Methods

#### Using pip only:
```bash
pip install polyid
```

#### Development Installation:
```bash
pip install -e .[dev]
```

This installs additional development dependencies including testing and documentation tools.

## Verification

After installation, verify polyID is working:

```python
import polyid
print(polyid.__version__)
```

## Troubleshooting

### Common Issues

1. **RDKit installation fails**: Ensure you have the correct conda channel:
   ```bash
   conda install -c conda-forge rdkit
   ```

2. **Import errors**: Make sure the conda environment is activated.

3. **GPU support**: For GPU acceleration, install CUDA-compatible versions of dependencies.

### Getting Help

If you encounter issues, please check:
- [GitHub Issues](https://github.com/your-repo/polyID/issues)
- [Documentation](../docs/README.md)
- [Deployment Guide](../docs/deployment/DEPLOYMENT_GUIDE.md)