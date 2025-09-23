# Quick Start

Get started with polyID in minutes with this quick start guide.

## Basic Usage

### 1. Import and Initialize

```python
import polyid
from polyid import PolyIDPredictor

# Initialize predictor
predictor = PolyIDPredictor()
```

### 2. Load or Generate Polymer Data

```python
# Example polymer SMILES
polymer_smiles = "CC(C)C(=O)OCCOC(=O)C(C)C"

# Predict properties
results = predictor.predict_properties(polymer_smiles)
print(results)
```

### 3. Advanced Prediction

```python
# Predict multiple properties
properties = ['molecular_weight', 'tacticity', 'glass_transition_temp']
predictions = predictor.predict_multiple(polymer_smiles, properties)
print(predictions)
```

## Example Workflow

```python
import pandas as pd
from polyid import generate_polymer_structures, train_model

# 1. Generate polymer structures
structures = generate_polymer_structures(n_polymers=100)

# 2. Train a model
model = train_model(structures, target_property='molecular_weight')

# 3. Make predictions
test_polymer = "CC(C)C(=O)OCCOC(=O)C(C)C"
prediction = model.predict(test_polymer)
print(f"Predicted molecular weight: {prediction}")
```

## Next Steps

- Learn about [model training](Model-Training.md)
- Explore [usage examples](../examples/)
- Check [API reference](API-Reference.md)
- Review [performance reports](../docs/reports/)

## Need Help?

- [Installation issues](Installation.md)
- [GitHub Issues](https://github.com/your-repo/polyID/issues)
- [Documentation](../docs/README.md)