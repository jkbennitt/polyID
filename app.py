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

# Default prediction columns for mock mode
DEFAULT_PREDICTION_COLUMNS = ['Tg', 'Tm', 'density', 'E', 'G', 'nu']
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
        if col == 'Tg':
            result[col] = round(random.uniform(-50, 200), 2)
        elif col == 'Tm':
            result[col] = round(random.uniform(0, 300), 2)
        elif col == 'density':
            result[col] = round(random.uniform(0.8, 1.5), 3)
        elif col == 'E':
            result[col] = round(random.uniform(0.1, 10), 2)
        elif col == 'G':
            result[col] = round(random.uniform(0.01, 5), 2)
        elif col == 'nu':
            result[col] = round(random.uniform(0.2, 0.5), 3)
        else:
            result[col] = round(random.uniform(0, 100), 2)
    return result

@GPU
def predict_polymer_properties(smiles):
    try:
        if not smiles:
            return json.dumps({"error": "Please enter a SMILES string"}, indent=2)
        
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
            result = {}
            for col in prediction_columns:
                pred_col = f"{col}_pred_mean"
                if pred_col in predictions.columns:
                    result[col] = float(predictions[pred_col].iloc[0])
                else:
                    result[col] = "N/A"
        else:
            result = generate_mock_predictions(smiles)
        
        return json.dumps(result, indent=2)
    
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
    
    iface = gr.Interface(
        fn=predict_polymer_properties,
        inputs=gr.Textbox(label="SMILES String", placeholder="Enter polymer SMILES..."),
        outputs=gr.JSON(label="Predicted Properties"),
        title="PolyID Polymer Property Predictor",
        description=description
    )
    
    iface.launch()