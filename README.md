---
title: PolyID ZeroGPU
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: bsd-3-clause
short_description: PolyID polymer property prediction with ZeroGPU acceleration
---


<p align="center">
  <img src="https://raw.githubusercontent.com/NREL/polyID/master/images/polyID-logo_color-full.svg" alt="PolyID Logo" width="400"/>
</p>

## Attribution
This repository is a fork of the original [PolyID](https://github.com/NREL/polyID) project, adapted for deployment on Hugging Face Spaces with ZeroGPU acceleration. The original PolyID provides a framework for building, training, and predicting polymer properties using graph neural networks.

PolyID<sup>TM</sup> provides a framework for building, training, and predicting polymer properities using graph neural networks. The codes leverages [nfp](https://pypi.org/project/nfp/), for building tensorflow-based message-passing neural networ, and [m2p](https://pypi.org/project/m2p/), for building polymer structures.  The notebooks have been provided that demonstrate how to: (1) build polymer structures from a polymer database and split into a training/validation and test set, (2) train a message passing neural network from using the trainining/validation set, and (3) evaluate the trained network on the test set. These three notebooks follow the methodology used in the forthcoming publication.

1. [Building polymer structures](https://github.com/NREL/polyID/blob/master/examples/1_generate_polymer_structures.ipynb): `examples/1_generate_polymer_structures.ipynb`
2. [Training a message passing neural network](https://github.com/NREL/polyID/blob/master/examples/2_generate_and_train_models.ipynb): `examples/2_generate_and_train_models.ipynb`
3. [Predicting and evaluating a trained network](https://github.com/NREL/polyID/blob/master/examples/3_evaluate_model_performance_and_DoV.ipynb): `examples/3_evaluate_model_performance_and_DoV.ipynb`

Additional notebooks have been provided to provide more examples and capabilities of the PolyID code base.

4. [Checking domain of validity](https://github.com/NREL/polyID/blob/master/examples/example_determine_domain-of-validity.ipynb): `examples/example_determine_domain-of-validity.ipynb`
5. [Generating hierarchical fingerprints for performance comparison](https://github.com/NREL/polyID/blob/master/examples/example_hierarchical_fingerprints.ipynb): `examples/example_hierarchical_fingerprints.ipynb`
6. [Predicting with the trained model](https://github.com/NREL/polyID/blob/master/examples/example_predict_with_trained_models.ipynb): `examples/example_predict_with_trained_models.ipynb`

For more details, see the manuscript [PolyID: Artificial Intelligence for Discovering Performance-Advantaged and Sustainable Polymers](https://doi.org/10.1021/acs.macromol.3c00994), _Macromolecules_ 2023.

## Testing Strategy

1. **Local testing (structure validation without GPU)**: Run `python app.py` to validate polymer structures locally without requiring GPU resources.
2. **Space deployment with ZeroGPU hardware selection**: Deploy the application on Hugging Face Spaces using ZeroGPU for efficient hardware acceleration.
3. **Interface testing via web UI**: Test the user interface through the web-based Gradio interface to ensure proper functionality and user experience.
4. **API testing via auto-generated /run/predict endpoint**: Use the following curl command to test the API endpoint:
   ```
   curl -X POST "https://your-space.hf.space/run/predict" -H "Content-Type: application/json" -d '{"data": ["CC(C)C(=O)OCC"]}'
   ```

### Additional Notes on Key Success Criteria
- Successful structure validation without errors during local testing.
- Accurate polymer property predictions with minimal deviation from expected results.
- Proper API responses with correct JSON formatting and timely execution.
- Seamless integration and performance on ZeroGPU hardware during deployment.

## Model Requirements

### Quick Start Instructions

1. **Install Dependencies**: Ensure Python 3.13 is installed, then run:
   ```
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   - Normal mode: `python app.py`
   - Development mode (auto-mock if models unavailable): `python app.py --dev`
   - Force mock mode for testing: `python app.py --mock`

3. **Access the Interface**: Open the provided URL in your browser to interact with the Gradio interface.

### Dependency Notes

- **Python 3.13 Compatibility**: This version has been updated to remove packages incompatible with Python 3.13 (`rdkit-pypi`, `tensorflow-addons`, `mordred`).
- **Mock Mode**: When models are unavailable or mock mode is forced, the app generates realistic simulated predictions based on SMILES hash for testing purposes.
- **Model Loading**: The app safely attempts to load trained models from `polyid/models/`. If loading fails, it automatically falls back to mock mode.
- **Hardware Acceleration**: Utilizes ZeroGPU for efficient deployment on Hugging Face Spaces.

## Cite
If you use PolyID in your work, please cite
```
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