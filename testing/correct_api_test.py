#!/usr/bin/env python3
"""
PolyID Correct API Testing with Gradio 5.46 format
"""

import requests
import json
import time

SPACE_URL = "https://jkbennitt-polyid-private.hf.space"
GRADIO_API_URL = f"{SPACE_URL}/gradio_api"

def get_gradio_config():
    """Get Gradio configuration to understand the interface"""
    try:
        response = requests.get(f"{SPACE_URL}/config", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Failed to get config: {e}")
    return None

def test_gradio_predict():
    """Test prediction using Gradio API format"""
    print("Testing Gradio API predictions...")

    # Test polymers
    test_polymers = ["CC", "CCCC", "CCO", "c1ccccc1"]

    for smiles in test_polymers:
        print(f"Testing polymer: {smiles}")
        try:
            # Gradio 5.x format typically uses /call/<function_name>
            prediction_url = f"{GRADIO_API_URL}/call/predict_properties"

            # Try different payload formats
            payloads = [
                {"data": [smiles]},
                {"data": [smiles, True, True, True, True]},  # With all properties selected
                {"inputs": [smiles]},
                [smiles]
            ]

            for i, payload in enumerate(payloads):
                try:
                    start_time = time.time()
                    response = requests.post(
                        prediction_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=60
                    )
                    response_time = time.time() - start_time

                    print(f"  Payload {i+1}: Status {response.status_code} ({response_time:.2f}s)")

                    if response.status_code == 200:
                        try:
                            data = response.json()
                            print(f"    Response: {str(data)[:200]}...")
                            return True
                        except:
                            print(f"    Response text: {response.text[:200]}...")

                except Exception as e:
                    print(f"  Payload {i+1} failed: {e}")

        except Exception as e:
            print(f"  Error: {e}")

    return False

def test_queue_system():
    """Test Gradio queue system"""
    print("\nTesting Gradio queue system...")

    try:
        # Try queue approach
        queue_url = f"{GRADIO_API_URL}/queue/push"
        payload = {
            "data": ["CC"],
            "event_data": None,
            "fn_index": 0,
            "trigger_id": 1
        }

        response = requests.post(queue_url, json=payload, timeout=30)
        print(f"Queue push: {response.status_code}")

        if response.status_code == 200:
            queue_data = response.json()
            print(f"Queue response: {queue_data}")

            # Check for event_id to poll for results
            if "event_id" in queue_data:
                event_id = queue_data["event_id"]
                print(f"Polling for event_id: {event_id}")

                # Poll for results
                for i in range(10):  # Poll up to 10 times
                    time.sleep(2)
                    poll_url = f"{GRADIO_API_URL}/queue/data"
                    poll_response = requests.get(f"{poll_url}?event_id={event_id}", timeout=10)
                    print(f"Poll {i+1}: {poll_response.status_code}")

                    if poll_response.status_code == 200:
                        poll_data = poll_response.json()
                        print(f"Poll result: {poll_data}")
                        if poll_data.get("msg") == "process_completed":
                            return True

    except Exception as e:
        print(f"Queue test failed: {e}")

    return False

def analyze_interface_structure():
    """Analyze the interface to understand component structure"""
    print("\nAnalyzing interface structure...")

    config = get_gradio_config()
    if config:
        print("Interface components:")
        components = config.get("components", [])
        for comp in components[:10]:  # Show first 10 components
            comp_type = comp.get("type", "unknown")
            comp_id = comp.get("id", "unknown")
            print(f"  ID {comp_id}: {comp_type}")

        # Look for function names or endpoints
        dependencies = config.get("dependencies", [])
        print(f"\nFound {len(dependencies)} dependencies/functions")
        for dep in dependencies[:5]:  # Show first 5
            fn_index = dep.get("id", "unknown")
            targets = dep.get("targets", [])
            print(f"  Function {fn_index}: targets {targets}")

def main():
    print("PolyID Correct API Testing")
    print("=" * 30)

    # Analyze structure first
    analyze_interface_structure()

    # Test prediction APIs
    success = test_gradio_predict()

    if not success:
        print("\nTrying queue system...")
        success = test_queue_system()

    if success:
        print("\nAPI testing successful!")
    else:
        print("\nAPI testing needs further investigation")

if __name__ == "__main__":
    main()