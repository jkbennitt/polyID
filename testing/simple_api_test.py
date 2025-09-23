#!/usr/bin/env python3
"""
Simple PolyID Live Space API Testing (no Unicode)
"""

import requests
import json
import time

# Test configuration
SPACE_URL = "https://jkbennitt-polyid-private.hf.space"

def test_space_status():
    """Test basic space availability"""
    print("Testing space availability...")
    try:
        response = requests.get(SPACE_URL, timeout=30)
        print(f"Space accessible: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Space access failed: {e}")
        return False

def test_polymer_predictions():
    """Test polymer property predictions"""
    print("Testing polymer predictions...")

    test_polymers = ["CC", "CCCC", "CCO"]
    results = []

    for smiles in test_polymers:
        print(f"Testing polymer: {smiles}")
        try:
            payload = {"data": [smiles]}
            start_time = time.time()
            response = requests.post(
                f"{SPACE_URL}/api/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print(f"Success ({response_time:.2f}s): {len(str(data))} chars response")
                results.append({
                    "smiles": smiles,
                    "success": True,
                    "response_time": response_time
                })
            else:
                print(f"Failed: {response.status_code}")
                results.append({
                    "smiles": smiles,
                    "success": False,
                    "error": response.status_code
                })

        except Exception as e:
            print(f"Exception: {e}")
            results.append({
                "smiles": smiles,
                "success": False,
                "error": str(e)
            })

    return results

def main():
    """Run tests"""
    print("PolyID Live Space Testing")
    print("=" * 30)

    # Test connectivity
    if not test_space_status():
        print("Space not accessible")
        return

    # Test predictions
    results = test_polymer_predictions()

    # Summary
    successful = sum(1 for r in results if r.get("success"))
    total = len(results)
    print(f"\nSummary: {successful}/{total} successful predictions")

    if successful > 0:
        avg_time = sum(r["response_time"] for r in results if r.get("success")) / successful
        print(f"Average response time: {avg_time:.2f}s")

if __name__ == "__main__":
    main()