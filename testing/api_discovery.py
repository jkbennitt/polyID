#!/usr/bin/env python3
"""
PolyID API Endpoint Discovery
"""

import requests
import json

SPACE_URL = "https://jkbennitt-polyid-private.hf.space"

def discover_gradio_api():
    """Discover Gradio API endpoints"""
    print("Discovering Gradio API endpoints...")

    # Common Gradio API endpoints to try
    endpoints = [
        "/api/predict",
        "/call/predict",
        "/queue/push",
        "/queue/data",
        "/info",
        "/config",
        "/api"
    ]

    for endpoint in endpoints:
        url = f"{SPACE_URL}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            print(f"{endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"  Content length: {len(response.text)}")
                if len(response.text) < 1000:  # Print short responses
                    print(f"  Content: {response.text[:500]}")
        except Exception as e:
            print(f"{endpoint}: Error - {e}")

def try_gradio_call():
    """Try Gradio /call/* endpoints"""
    print("\nTrying Gradio call endpoints...")

    # Common function names in Gradio
    functions = ["predict", "inference", "run", "process"]

    for func in functions:
        url = f"{SPACE_URL}/call/{func}"
        try:
            # Try GET first
            response = requests.get(url, timeout=10)
            print(f"/call/{func} GET: {response.status_code}")

            # Try POST with empty data
            response = requests.post(url, json={"data": []}, timeout=10)
            print(f"/call/{func} POST: {response.status_code}")

        except Exception as e:
            print(f"/call/{func}: Error - {e}")

def check_space_config():
    """Check space configuration"""
    print("\nChecking space configuration...")

    try:
        response = requests.get(f"{SPACE_URL}/config", timeout=10)
        if response.status_code == 200:
            config = response.json()
            print("Space config found:")
            print(json.dumps(config, indent=2)[:1000])
            return config
    except Exception as e:
        print(f"Config check failed: {e}")

    return None

def main():
    print("PolyID API Discovery")
    print("=" * 30)

    discover_gradio_api()
    try_gradio_call()
    config = check_space_config()

    print("\nDiscovery completed")

if __name__ == "__main__":
    main()