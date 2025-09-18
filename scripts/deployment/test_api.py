import requests
import json

# Test the Gradio API endpoint
url = "http://localhost:7860/gradio_api/predict"
data = {"smiles": "CC(C)C(=O)OCC"}

print("Testing API endpoint:", url)
print("Data:", data)

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)

# Try alternative endpoints
alternative_urls = [
    "http://localhost:7860/api/predict",
    "http://localhost:7860/predict",
    "http://localhost:7860/run/predict"
]

for alt_url in alternative_urls:
    print(f"\nTesting alternative endpoint: {alt_url}")
    try:
        response = requests.post(alt_url, json=data)
        print("Status Code:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        print("Error:", e)