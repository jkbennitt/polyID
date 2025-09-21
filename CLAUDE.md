# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolyID is a Python package for polymer property prediction using graph neural networks. It leverages message-passing neural networks built with TensorFlow/Keras to predict polymer properties from molecular structures. The framework integrates RDKit for molecular processing, NFP for neural fingerprints, and m2p for polymer structure generation.

## Development Commands

### Environment Setup
```bash
# Create conda environment with required dependencies
conda env create -f environment.yml
conda activate polyID

# Development installation
pip install -e .
```

### Testing
```bash
# Run original PolyID unit tests
pytest tests/

# Run specific unit test file
pytest tests/test_singlemodel.py

# Run HF Spaces deployment tests
pytest hf-spaces-testing/

# Run specific HF Spaces test
pytest hf-spaces-testing/test_polyid_comprehensive.py

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Run all linting and formatting checks
tox

# Individual tools
tox -e black     # Code formatting check
tox -e isort     # Import sorting check
tox -e flake8    # Linting check

# Auto-format code
black polyid/ tests/
isort --profile black polyid/ tests/
```

### Package Building
```bash
# Build distribution packages
python setup.py sdist bdist_wheel

# Check package
twine check dist/*
```

## Core Architecture

### Main Classes

**SingleModel** (`polyid/polyid.py:61-465`)
- Manages training and prediction for a single neural network model
- Handles data preprocessing, scaling, and TensorFlow dataset generation
- Key methods: `train()`, `predict()`, `generate_preprocessor()`, `generate_data_scaler()`

**MultiModel** (`polyid/polyid.py:468-1030`)
- Manages ensemble of SingleModel instances for k-fold cross-validation
- Handles data splitting, aggregate predictions, and model persistence
- Key methods: `split_data()`, `train_models()`, `make_aggregate_predictions()`

### Neural Network Architecture

**Base Models** (`polyid/models/base_models.py`)
- `global100()`: Core message-passing architecture using NFP layers
- Uses atom/bond embeddings, GlobalUpdate, EdgeUpdate, NodeUpdate layers
- Supports multi-task prediction with shared representations

**Model Components**:
- **Preprocessors** (`polyid/preprocessors/`): Convert SMILES to graph representations
- **Features** (`polyid/preprocessors/features.py`): Atom and bond feature extraction
- **Parameters** (`polyid/parameters.py`): Training hyperparameter management

### Key Dependencies & Compatibility

**Chemistry Stack**:
- `rdkit`: Molecular structure processing and manipulation
- `nfp`: Neural fingerprint library for graph neural networks
- `m2p`: Polymer structure generation from monomers
- `shortuuid`: Unique identifier generation for data integrity

**ML/DL Stack**:
- `tensorflow>=2.14`: Core deep learning framework
- `scikit-learn`: Data preprocessing and model evaluation
- `pandas`: Data manipulation and analysis

**Important**: This project requires the full chemistry stack (RDKit, NFP, m2p) which has specific system dependencies. For Hugging Face Spaces deployment, use Standard GPU Spaces rather than ZeroGPU for full compatibility.

## Typical Workflows

### Training New Models
1. **Data Preparation**: Load polymer dataset with `smiles_polymer` column and target properties
2. **MultiModel Setup**: `load_dataset()` → `split_data()` with k-fold cross-validation
3. **Preprocessing**: `generate_preprocessors()` and `generate_data_scaler()`
4. **Training**: `train_models()` with neural network architecture function
5. **Evaluation**: Check validation results and domain of validity

### Making Predictions
1. **Model Loading**: Use `MultiModel.load_models()` or `SingleModel.load_model()`
2. **Input Preparation**: Ensure DataFrame has `smiles_polymer` column
3. **Prediction**: `make_predictions()` or `make_aggregate_predictions()`
4. **Results**: Get predicted values with confidence estimates

### Model Persistence
- Models save as `.h5` (TensorFlow model) + `.pk` (preprocessor/scaler/metadata)
- MultiModel saves ensemble state to `parameters.pk` + individual model folders
- Use `save_folder` parameter in training methods for automatic persistence

## File Organization Patterns

### Core Project Structure
- **Core Logic**: Single file `polyid/polyid.py` contains main SingleModel/MultiModel classes
- **Modular Architecture**: Neural networks in `models/`, preprocessing in `preprocessors/`
- **Configuration**: Parameters class provides centralized hyperparameter management
- **Examples**: Jupyter notebooks demonstrate complete workflows

### Testing Structure
- **`tests/`**: Original PolyID unit tests with pytest fixtures for model setup and validation
- **`hf-spaces-testing/`**: Hugging Face Spaces deployment testing suite with comprehensive validation tools

### Documentation and Reports
- **`docs/`**: Organized documentation with subcategories:
  - `docs/deployment/`: Deployment guides and implementation instructions
  - `docs/analysis/`: Technical analysis and compatibility studies
  - `docs/guides/`: Development workflow and branch management guides
  - `docs/reviews/`: Detailed technical reviews and system assessments
- **`reports/`**: Generated analysis reports, performance metrics, and validation results

## Current Development Context

**Active Branch**: `standard-gpu-deployment`
**Deployment Target**: Hugging Face Spaces with Standard GPU for chemistry package compatibility
**Python Version**: 3.10/3.11 (as specified in environment.yml/documentation)

## Key Development Notes

- Always test with the chemistry stack dependencies when making changes to core functionality
- Use the provided Parameters class for consistent hyperparameter management
- Follow the established pattern of preprocessor → scaler → model training
- When adding new neural architectures, follow the pattern in `models/base_models.py`
- Model IDs and hash generation are critical for reproducibility and data integrity