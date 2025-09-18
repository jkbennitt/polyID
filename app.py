import gradio as gr
from spaces import GPU
import pandas as pd
from polyid import MultiModel
from polyid.preprocessors import PmPreprocessor
import json
import argparse
import hashlib
import random

# Global flags
MODEL_AVAILABLE = False
MOCK_MODE = False
MODEL_FOLDER = "polyid/models"

# Default prediction columns for mock mode - updated to match specification
DEFAULT_PREDICTION_COLUMNS = [
    'glass_transition_temp', 'melting_temp', 'decomposition_temp', 'thermal_stability_score',
    'tensile_strength', 'elongation_at_break', 'youngs_modulus', 'flexibility_score',
    'water_resistance', 'acid_resistance', 'base_resistance', 'solvent_resistance',
    'uv_stability', 'oxygen_permeability', 'moisture_vapor_transmission', 'biodegradability',
    'hydrophane_opal_compatibility', 'pyrite_compatibility', 'fossil_compatibility', 'meteorite_compatibility',
    'analysis_time', 'confidence_score'
]
prediction_columns = DEFAULT_PREDICTION_COLUMNS

mm = None

def load_models_safe():
    global MODEL_AVAILABLE, prediction_columns, mm
    try:
        mm = MultiModel.load_models(MODEL_FOLDER)
        MODEL_AVAILABLE = True
        prediction_columns = mm.prediction_columns
        return mm
    except Exception as e:
        print(f"Model loading failed: {e}")
        MODEL_AVAILABLE = False
        mm = None
        return None

def generate_mock_predictions(smiles):
    hash_val = int(hashlib.md5(smiles.encode()).hexdigest(), 16)
    random.seed(hash_val)
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

@GPU
def predict_polymer_properties(smiles):
    try:
        if not smiles:
            return json.dumps({"error": "Please enter a SMILES string"}, indent=2)

        # Generate polymer ID from SMILES hash
        polymer_id = f"POLY-{hashlib.md5(smiles.encode()).hexdigest()[:8].upper()}"

        # Create dataframe with SMILES
        df = pd.DataFrame({
            'smiles_polymer': [smiles],
            'pm': [0.5],  # Default values, adjust as needed
            'distribution': ['random'],
            'smiles_monomer': [smiles]  # Assuming single monomer for simplicity
        })

        if MODEL_AVAILABLE and not MOCK_MODE:
            # Make predictions
            predictions = mm.make_aggregate_predictions(df)

            # Extract prediction values (assuming mean aggregation)
            properties = {}
            for col in prediction_columns:
                pred_col = f"{col}_pred_mean"
                if pred_col in predictions.columns:
                    properties[col] = float(predictions[pred_col].iloc[0])
                else:
                    properties[col] = None  # Use None instead of "N/A" for JSON compatibility
        else:
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

# Gradio interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolyID Polymer Property Predictor")
    parser.add_argument('--mock', action='store_true', help='Force mock mode for testing')
    parser.add_argument('--dev', action='store_true', help='Development mode (enables mock if models unavailable)')
    args = parser.parse_args()

    # Load models first to check availability
    load_models_safe()

    MOCK_MODE = args.mock or (args.dev and not MODEL_AVAILABLE)

    description = "Predict polymer properties using PolyID models."
    if not MODEL_AVAILABLE or MOCK_MODE:
        description += " (Currently in mock mode - using simulated predictions)"

    with gr.Blocks(title="PolyID Polymer Property Predictor") as iface:
        gr.Markdown(f"# PolyID Polymer Property Predictor\n{description}")
        with gr.Row():
            smiles_input = gr.Textbox(label="SMILES String", placeholder="Enter polymer SMILES...")
            predict_btn = gr.Button("Predict")
        output = gr.JSON(label="Predicted Properties")

        predict_btn.click(
            fn=predict_polymer_properties,
            inputs=smiles_input,
            outputs=output,
            api_name="predict"
        )

    iface.launch(show_api=True)