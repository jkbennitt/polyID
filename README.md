---
title: PolyID
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.46.0"
app_file: app.py
python_version: "3.11"
hardware: standard-gpu
pinned: false
license: bsd-3-clause
short_description: Polymer property prediction using graph neural networks
---

# PolyID - Polymer Property Prediction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE.md)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Standard GPU](https://img.shields.io/badge/Hardware-Standard%20GPU-green.svg)](https://huggingface.co/docs/hub/spaces-gpus)

<p align="center">
  <img src="https://raw.githubusercontent.com/NREL/polyID/master/images/polyID-logo_color-full.svg" alt="PolyID Logo" width="400"/>
</p>

## Overview

PolyID™ provides a framework for building, training, and predicting polymer properties using graph neural networks. This Hugging Face Spaces deployment offers an interactive interface for predicting polymer properties from molecular structures using state-of-the-art machine learning models.

**Current Status**: Production deployment on Standard GPU Spaces with full chemistry stack compatibility

## Quick Start

### 🚀 Deploy Your Own PolyID Space
This is an open-source project designed for Hugging Face Spaces deployment. To use PolyID:

1. **Fork this repository** to your GitHub account
2. **Create a new Hugging Face Space** and connect it to your fork
3. **Select Standard GPU hardware** for full chemistry stack compatibility
4. **Deploy and use** your own PolyID interface for polymer property prediction

### 🔧 Development & Customization
For researchers wanting to modify or extend PolyID:

```bash
# Clone repository (Standard GPU deployment branch)
git clone -b standard-gpu-deployment https://github.com/jkbennitt/polyID
cd polyID

# Setup development environment
conda env create -f environment.yml
conda activate polyID
pip install -e .

# Run core PolyID functionality
pytest tests/

# Test HF Spaces deployment locally
python app.py
```

**Note**: PolyID is optimized for HF Spaces deployment. Local development is primarily for customization and research.

## Project Structure

```
polyID/
├── 📁 polyid/                    # Core package source code
├── 📁 docs/                      # Comprehensive documentation
│   ├── 📁 deployment/           # HF Spaces deployment guides
│   ├── 📁 analysis/             # Technical analysis reports
│   ├── 📁 guides/               # Development guides
│   └── 📁 reviews/              # System architecture reviews
├── 📁 hf-spaces-testing/        # HF Spaces specific testing suite
├── 📁 tests/                    # Original unit tests
├── 📁 reports/                  # Analysis reports and results
├── 📁 examples/                 # Usage examples and notebooks
└── 📁 data/                     # Sample datasets
```

## Features

- **🧪 Real-time Polymer Property Prediction**: Predict glass transition temperature (Tg), melting temperature (Tm), and other properties
- **🧠 Graph Neural Networks**: Advanced message-passing neural networks for molecular representation
- **📊 Domain of Validity Analysis**: Assess prediction reliability and confidence intervals
- **🖥️ Interactive Interface**: User-friendly Gradio interface for easy polymer analysis
- **⚡ Standard GPU Performance**: Optimized for HF Spaces Standard GPU deployment
- **🔬 Full Chemistry Stack**: Complete RDKit, NFP, and m2p integration

## Technology Stack

This deployment leverages **Standard GPU Spaces** for optimal performance and compatibility:

**Chemistry Processing**:
- **RDKit 2023.09+**: Molecular fingerprinting and structure analysis
- **NFP**: Neural fingerprint layers for graph neural networks
- **m2p**: Polymer structure generation and processing

**Machine Learning**:
- **TensorFlow 2.16+**: Core deep learning framework
- **scikit-learn**: Data preprocessing and model evaluation
- **NumPy/Pandas**: Data manipulation and analysis

**Interface & Deployment**:
- **Gradio 5.46+**: Interactive web interface
- **Python 3.11**: Optimal performance and compatibility
- **Standard GPU**: Full chemistry stack support

## Documentation Navigation

### 📚 [Complete Documentation](docs/README.md)
- **[Deployment Guides](docs/deployment/)**: HF Spaces setup and configuration
- **[Technical Analysis](docs/analysis/)**: Performance and compatibility studies
- **[Development Guides](docs/guides/)**: Contributing and development workflows
- **[System Reviews](docs/reviews/)**: Architecture and scientific validation

### 🧪 [Testing Suites](hf-spaces-testing/README.md)
- **HF Spaces Testing**: Specialized tests for cloud deployment
- **Chemistry Stack Validation**: Comprehensive package compatibility tests
- **Performance Monitoring**: Deployment performance analysis
- **Original Unit Tests**: Core package functionality tests in `tests/`

### 📊 [Analysis Reports](reports/README.md)
- **Chemistry Stack Verification**: Package compatibility reports
- **Performance Analysis**: Deployment performance metrics
- **Validation Reports**: Model accuracy and reliability studies

## Development vs Deployment

### 🔬 Development Environment
- **Purpose**: Research, model customization, algorithm development
- **Location**: `tests/` directory
- **Focus**: Core PolyID functionality, training new models
- **Setup**: Full conda environment for research and development

### ☁️ HF Spaces Deployment (Primary Use Case)
- **Purpose**: Production polymer property prediction interface
- **Location**: `hf-spaces-testing/` directory
- **Focus**: User-friendly interface, real-time predictions, reliability
- **Setup**: Optimized for Standard GPU Spaces with full chemistry stack

## Model Architecture

**SingleModel** (`polyid/polyid.py`):
- Manages training and prediction for individual neural network models
- Handles data preprocessing, scaling, and TensorFlow dataset generation

**MultiModel** (`polyid/polyid.py`):
- Manages ensemble of SingleModel instances for k-fold cross-validation
- Provides aggregate predictions with improved reliability

**Neural Networks** (`polyid/models/base_models.py`):
- Message-passing architecture using NFP layers
- Global/Edge/Node update mechanisms for molecular graphs

## Citation

If you use PolyID in your work, please cite:

```bibtex
@article{wilson2023polyid,
  title={PolyID: Artificial Intelligence for Discovering Performance-Advantaged and Sustainable Polymers},
  author={Wilson, A Nolan and St John, Peter C and Marin, Daniela H and Hoyt, Caroline B and Rognerud, Erik G and Nimlos, Mark R and Cywar, Robin M and Rorrer, Nicholas A and Shebek, Kevin M and Broadbelt, Linda J and Beckham, Gregg T and Crowley, Michael F},
  journal={Macromolecules},
  volume={56},
  number={21},
  pages={8547--8557},
  year={2023},
  publisher={ACS Publications}
}
```

## Contributing

1. **Review Documentation**: Start with [docs/guides/](docs/guides/) for development setup
2. **Choose Test Suite**: Use appropriate testing based on your contribution
   - Core functionality → `tests/`
   - HF Spaces features → `hf-spaces-testing/`
3. **Follow Standards**: Adhere to code quality standards (black, isort, flake8)
4. **Update Documentation**: Keep docs synchronized with changes

## Standard GPU Deployment

This application is specifically designed for **Hugging Face Standard GPU Spaces** to ensure:

- ✅ Full compatibility with complex chemistry packages (RDKit, NFP, m2p)
- ✅ Advanced molecular processing capabilities
- ✅ Complete TensorFlow/neural network functionality
- ✅ Reliable prediction performance
- ✅ Optimal resource utilization

## Support & Resources

- **📖 Documentation**: [docs/](docs/)
- **🐛 Issues**: [GitHub Issues](https://github.com/NREL/polyID/issues)
- **📧 Contact**: [NREL Research Team](mailto:contact@nrel.gov)
- **📚 Paper**: [Macromolecules 2023](https://pubs.acs.org/doi/10.1021/acs.macromol.3c01089)

## License

PolyID is licensed under the BSD 3-Clause License. See [LICENSE.md](LICENSE.md) for details.