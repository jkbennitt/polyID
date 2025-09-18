#!/usr/bin/env python3

import json
import hashlib
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

# Mock the prediction function
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
            return {"error": "Please enter a SMILES string"}

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

        return response

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/run/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            # Extract SMILES from the data
            if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                smiles = data['data'][0]
            elif 'smiles' in data:
                smiles = data['smiles']
            else:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid data format"}).encode())
                return

            # Make prediction
            result = predict_polymer_properties(smiles)

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not Found"}).encode())

    def log_message(self, format, *args):
        # Suppress default logging
        pass

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print("Starting test API server on port 8000...")
    print("Test with: curl -X POST 'http://localhost:8000/run/predict' -H 'Content-Type: application/json' -d '{\"data\": [\"CC(C)C(=O)OCC\"]}'")
    httpd.serve_forever()