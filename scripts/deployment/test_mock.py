import sys
import os
# Add the deployment/huggingface directory to path to import app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deployment', 'huggingface'))

# Import the necessary parts
from app import predict_polymer_properties, MOCK_MODE, MODEL_AVAILABLE

# Set mock mode
MOCK_MODE = True

# Test with a sample SMILES
sample_smiles = "CC(C)C(=O)OCC"  # Example SMILES

result = predict_polymer_properties(sample_smiles)
print("Result:", result)

# Check if it's JSON and has expected keys
import json
try:
    parsed = json.loads(result)
    print("Parsed JSON:", parsed)
    expected_keys = [
        'glass_transition_temp', 'melting_temp', 'decomposition_temp', 'thermal_stability_score',
        'tensile_strength', 'elongation_at_break', 'youngs_modulus', 'flexibility_score',
        'water_resistance', 'acid_resistance', 'base_resistance', 'solvent_resistance',
        'uv_stability', 'oxygen_permeability', 'moisture_vapor_transmission', 'biodegradability',
        'hydrophane_opal_compatibility', 'pyrite_compatibility', 'fossil_compatibility', 'meteorite_compatibility',
        'analysis_time', 'confidence_score'
    ]
    if all(key in parsed for key in expected_keys):
        print("Response format matches expected JSON structure.")
    else:
        missing_keys = [key for key in expected_keys if key not in parsed]
        print(f"Missing keys in response: {missing_keys}")
except json.JSONDecodeError:
    print("Result is not valid JSON.")