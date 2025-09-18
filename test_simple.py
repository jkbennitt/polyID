#!/usr/bin/env python3

import json
import hashlib
import random

# Mock the prediction function for testing
def generate_mock_predictions(smiles):
    hash_val = int(hashlib.md5(smiles.encode()).hexdigest(), 16)
    random.seed(hash_val)

    prediction_columns = [
        'glass_transition_temp', 'melting_temp', 'decomposition_temp', 'thermal_stability_score',
        'tensile_strength', 'elongation_at_break', 'youngs_modulus', 'flexibility_score',
        'water_resistance', 'acid_resistance', 'base_resistance', 'solvent_resistance',
        'uv_stability', 'oxygen_permeability', 'moisture_vapor_transmission', 'biodegradability',
        'hydrophane_opal_compatibility', 'pyrite_compatibility', 'fossil_compatibility', 'meteorite_compatibility',
        'analysis_time', 'confidence_score'
    ]

    result = {}
    for col in prediction_columns:
        if col == 'glass_transition_temp':
            result[col] = round(random.uniform(-50, 200), 2)
        elif col == 'melting_temp':
            result[col] = round(random.uniform(0, 300), 2)
        elif col == 'decomposition_temp':
            result[col] = round(random.uniform(200, 500), 2)
        elif col == 'thermal_stability_score':
            result[col] = round(random.uniform(0, 10), 2)
        elif col == 'tensile_strength':
            result[col] = round(random.uniform(10, 100), 2)
        elif col == 'elongation_at_break':
            result[col] = round(random.uniform(1, 500), 2)
        elif col == 'youngs_modulus':
            result[col] = round(random.uniform(0.1, 10), 2)
        elif col == 'flexibility_score':
            result[col] = round(random.uniform(0, 10), 2)
        elif col in ['water_resistance', 'acid_resistance', 'base_resistance', 'solvent_resistance']:
            result[col] = round(random.uniform(0, 10), 2)
        elif col == 'uv_stability':
            result[col] = round(random.uniform(0, 10), 2)
        elif col == 'oxygen_permeability':
            result[col] = round(random.uniform(0, 1000), 2)
        elif col == 'moisture_vapor_transmission':
            result[col] = round(random.uniform(0, 100), 2)
        elif col == 'biodegradability':
            result[col] = round(random.uniform(0, 10), 2)
        elif col in ['hydrophane_opal_compatibility', 'pyrite_compatibility', 'fossil_compatibility', 'meteorite_compatibility']:
            result[col] = round(random.uniform(0, 10), 2)
        elif col == 'analysis_time':
            result[col] = round(random.uniform(0.1, 10), 2)
        elif col == 'confidence_score':
            result[col] = round(random.uniform(0, 1), 3)
        else:
            result[col] = round(random.uniform(0, 100), 2)
    return result

def predict_polymer_properties(smiles):
    try:
        if not smiles:
            return json.dumps({"error": "Please enter a SMILES string"}, indent=2)

        # Generate polymer ID from SMILES hash
        polymer_id = f"POLY-{hashlib.md5(smiles.encode()).hexdigest()[:8].upper()}"

        # Mock mode - generate simulated predictions
        properties = generate_mock_predictions(smiles)

        # PaleoBond-compatible response format
        response = {
            "polymer_id": polymer_id,
            "smiles": smiles,
            "properties": properties
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {str(e)}"}, indent=2)

if __name__ == "__main__":
    # Test with sample SMILES
    test_smiles = "CC(C)C(=O)OCC"
    print("Testing PolyID API with SMILES:", test_smiles)
    print()

    result = predict_polymer_properties(test_smiles)
    print("Response:")
    print(result)
    print()

    # Verify format
    try:
        parsed = json.loads(result)
        if "polymer_id" in parsed and "smiles" in parsed and "properties" in parsed:
            print("✓ Response format matches PaleoBond-compatible structure")
            print(f"  - polymer_id: {parsed['polymer_id']}")
            print(f"  - smiles: {parsed['smiles']}")
            print(f"  - properties count: {len(parsed['properties'])}")
        else:
            print("✗ Response format does not match expected structure")
    except:
        print("✗ Response is not valid JSON")