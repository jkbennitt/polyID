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

<p align="center">
  <img src="https://raw.githubusercontent.com/NREL/polyID/master/images/polyID-logo_color-full.svg" alt="PolyID Logo" width="400"/>
</p>

## Overview

PolyID™ provides a framework for building, training, and predicting polymer properties using graph neural networks. This Hugging Face Spaces deployment offers an interactive interface for predicting polymer properties from molecular structures.

## Features

- **Real-time Polymer Property Prediction**: Predict glass transition temperature (Tg), melting temperature (Tm), and other properties
- **Graph Neural Networks**: Leverages message-passing neural networks for molecular representation
- **Domain of Validity Analysis**: Assess prediction reliability and confidence
- **Interactive Interface**: User-friendly Gradio interface for easy polymer analysis

## Technology Stack

This deployment uses **Standard GPU Spaces** for full compatibility with the complete chemistry software stack:

- **Chemistry Processing**: RDKit for molecular fingerprinting and structure analysis
- **Neural Networks**: TensorFlow 2.16+ with NFP (Neural Fingerprint) layers
- **Polymer Structures**: m2p for polymer structure generation and processing
- **Interface**: Gradio 5.48+ for interactive web interface
- **Python**: 3.11 for optimal performance and compatibility

## Usage

1. **Input Polymer Structure**: Enter a SMILES string representing your polymer structure
2. **Select Properties**: Choose which properties to predict (Tg, Tm, etc.)
3. **Get Predictions**: Receive predictions with confidence intervals
4. **Analyze Results**: Review domain of validity analysis for prediction reliability

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

## Standard GPU Deployment

This application is deployed on **Hugging Face Standard GPU Spaces** to ensure full compatibility with:

- Complex chemistry packages (RDKit, NFP, m2p)
- Advanced molecular processing capabilities
- Complete TensorFlow/neural network functionality
- Reliable prediction performance

For more information about PolyID, visit the [GitHub repository](https://github.com/NREL/polyID).

## License

PolyID is licensed under the BSD 3-Clause License.