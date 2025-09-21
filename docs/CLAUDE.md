# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

### New Folder Organization
```
polyID/
├── src/polyid/              # Main package source code
│   ├── models/              # Model implementations
│   └── preprocessors/       # Data preprocessing utilities
├── app.py                   # Hugging Face Spaces app (root requirement)
├── packages.txt             # HF Spaces system dependencies (root requirement)
├── requirements.txt         # Python dependencies (root requirement)
├── scripts/deployment/      # Deployment and test scripts
├── examples/               # Usage examples
│   ├── notebooks/          # Jupyter notebooks
│   └── scripts/           # Example scripts
├── docs/                  # Documentation files
├── tests/                 # Unit and integration tests
└── data/                  # Data files
```

## Development Commands

### Running the Application
- **Normal mode**: `python app.py`
- **Development mode** (auto-mock if models unavailable): `python app.py --dev`
- **Force mock mode** for testing: `python app.py --mock`

### Testing
- **Local structure validation**: `python app.py` (validates polymer structures without GPU)
- **Simple API test**: `python scripts/deployment/test_simple.py`
- **Mock mode test**: `python scripts/deployment/test_mock.py`
- **API endpoint test**: `python scripts/deployment/test_api_endpoint.py`
- **API test**: `python scripts/deployment/test_api.py`

### Dependencies
- **Install requirements**: `pip install -r requirements.txt`
- **Python version**: 3.10.13 (ZeroGPU requirement - fixed version for GPU acceleration)
- **Key packages**: `rdkit` (modern package for molecular fingerprinting), `gradio`, `spaces`, `torch`

## Architecture Overview

### Core Structure
This is a **PolyID fork** adapted for Hugging Face Spaces deployment with ZeroGPU acceleration. The original PolyID provides polymer property prediction using graph neural networks.

**Important**: ZeroGPU requires Python 3.10.13 exactly - this is a hard constraint for GPU access on HF Spaces.

### Key Components

#### Main Application (`app.py`)
- **Gradio interface** for polymer property prediction in root directory (HF Spaces requirement)
- **ZeroGPU acceleration** via `@GPU` decorator
- **Dual mode operation**: real model predictions vs mock mode
- **PaleoBond-compatible response format** with `polymer_id`, `smiles`, and `properties`
- **Import path adjustment** to use `src/polyid/` package structure
- **ZeroGPU acceleration** via `@GPU` decorator
- **Dual mode operation**: real model predictions vs mock mode
- **PaleoBond-compatible response format** with `polymer_id`, `smiles`, and `properties`

#### PolyID Package (`src/polyid/`)
- `MultiModel`, `SingleModel`: Core model classes for predictions
- `PmPreprocessor`: Preprocessing utilities in `preprocessors/`
- `DoV`: Domain of validity checking
- `Parameters`: Model parameter management
- Model storage in `src/polyid/models/` (base_models.py, tacticity_models.py, callbacks.py)

#### Model Loading Strategy
- **Safe loading**: `load_models_safe()` attempts to load from `src/polyid/models/`
- **Automatic fallback**: Falls back to mock mode if model loading fails
- **Mock predictions**: Generate deterministic predictions using SMILES hash for testing

#### Data Flow
1. SMILES input → DataFrame with `smiles_polymer`, `pm`, `distribution`, `smiles_monomer`
2. Model predictions via `mm.make_aggregate_predictions(df)`
3. Extract `{col}_pred_mean` columns for final properties
4. Return JSON with polymer_id, smiles, and properties

### Deployment Context

#### Hugging Face Spaces
- **ZeroGPU hardware**: Efficient GPU acceleration for model inference
- **Auto-sync**: GitHub main branch syncs to HF Spaces via `.github/workflows/hf_spaces_sync.yml`
- **API endpoint**: Available at `/run/predict` for programmatic access

#### Testing Strategy
- **Local validation**: Structure validation without GPU requirements
- **Mock mode**: Simulated predictions for development/testing
- **API testing**: Multiple test files validate different aspects of functionality

### Important Notes
- **Python 3.13 compatibility**: Packages incompatible with 3.13 have been removed
- **Mock mode determinism**: Uses SMILES hash as random seed for consistent test results
- **Model availability**: Application gracefully handles missing models by falling back to mock predictions
- **PaleoBond integration**: Response format designed for compatibility with PaleoBond systems

### File Organization
- **Package source**: All package code is now in `src/polyid/`
- **Deployment**: Hugging Face app is in `deployment/huggingface/app.py`
- **Test scripts**: Located in `scripts/deployment/`
- **Examples**: Organized in `examples/notebooks/` and `examples/scripts/`
- **Documentation**: Centralized in `docs/` directory