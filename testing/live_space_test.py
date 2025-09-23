#!/usr/bin/env python3
"""
PolyID Live Space Direct API Testing
===================================

Test the live PolyID Hugging Face Space functionality directly
URL: https://jkbennitt-polyid-private.hf.space
"""

import requests
import json
import time
from typing import Dict, List, Any

# Test configuration
SPACE_URL = "https://jkbennitt-polyid-private.hf.space"
API_URL = f"{SPACE_URL}/api/predict"

# Test polymer SMILES strings
TEST_POLYMERS = [
    "CC",  # Simple test case
    "CCCC",  # Butane
    "c1ccccc1",  # Benzene ring
    "CC(C)(C)C",  # Branched structure
    "CC(=O)OC",  # Ester group
    "CCN",  # Amine group
    "CCO",  # Alcohol group
    "CCOCC",  # Ether
]

def test_space_status():
    """Test basic space availability"""
    print("🌐 Testing space availability...")
    try:
        response = requests.get(SPACE_URL, timeout=30)
        print(f"   ✅ Space accessible: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Space access failed: {e}")
        return False

def test_gradio_api():
    """Test Gradio API endpoint"""
    print("🔧 Testing Gradio API...")
    try:
        # Try to get the API info
        api_info_url = f"{SPACE_URL}/info"
        response = requests.get(api_info_url, timeout=30)
        if response.status_code == 200:
            print(f"   ✅ API info accessible")
            return True
        else:
            print(f"   ⚠️  API info returned {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False

def test_polymer_predictions():
    """Test polymer property predictions"""
    print("🧪 Testing polymer predictions...")
    results = []

    for i, smiles in enumerate(TEST_POLYMERS):
        print(f"   Testing polymer {i+1}: {smiles}")
        try:
            # Simulate form data for Gradio
            payload = {
                "data": [smiles]
            }

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
                print(f"      ✅ Success ({response_time:.2f}s): {data}")
                results.append({
                    "smiles": smiles,
                    "success": True,
                    "response_time": response_time,
                    "data": data
                })
            else:
                print(f"      ❌ Failed: {response.status_code} - {response.text}")
                results.append({
                    "smiles": smiles,
                    "success": False,
                    "error": f"{response.status_code}: {response.text}"
                })

        except Exception as e:
            print(f"      ❌ Exception: {e}")
            results.append({
                "smiles": smiles,
                "success": False,
                "error": str(e)
            })

    return results

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("⚠️  Testing error handling...")

    invalid_inputs = [
        "",  # Empty string
        "INVALID_SMILES_123",  # Invalid SMILES
        "C" * 1000,  # Very long string
        None,  # None value
    ]

    for test_input in invalid_inputs:
        print(f"   Testing invalid input: {repr(test_input)}")
        try:
            payload = {"data": [test_input]}
            response = requests.post(
                f"{SPACE_URL}/api/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code in [400, 422, 500]:
                print(f"      ✅ Proper error handling: {response.status_code}")
            else:
                print(f"      ⚠️  Unexpected response: {response.status_code}")

        except Exception as e:
            print(f"      ✅ Exception caught (expected): {e}")

def performance_benchmark():
    """Basic performance benchmark"""
    print("⚡ Performance benchmark...")

    test_smiles = "CC"  # Simple test case
    num_requests = 5
    times = []

    for i in range(num_requests):
        try:
            payload = {"data": [test_smiles]}
            start_time = time.time()
            response = requests.post(
                f"{SPACE_URL}/api/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response_time = time.time() - start_time
            times.append(response_time)
            print(f"   Request {i+1}: {response_time:.2f}s")

        except Exception as e:
            print(f"   Request {i+1} failed: {e}")

    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"   📊 Performance stats:")
        print(f"      Average: {avg_time:.2f}s")
        print(f"      Min: {min_time:.2f}s")
        print(f"      Max: {max_time:.2f}s")

def main():
    """Run all tests"""
    print("🧬 PolyID Live Space Testing Suite")
    print("=" * 50)

    # Basic connectivity
    if not test_space_status():
        print("❌ Space not accessible, aborting tests")
        return

    # API tests
    test_gradio_api()

    # Functional tests
    results = test_polymer_predictions()

    # Error handling
    test_error_handling()

    # Performance
    performance_benchmark()

    # Summary
    print("\n📋 Test Summary:")
    successful_predictions = sum(1 for r in results if r.get("success"))
    total_predictions = len(results)
    print(f"   Successful predictions: {successful_predictions}/{total_predictions}")

    if successful_predictions > 0:
        avg_response_time = sum(r["response_time"] for r in results if r.get("success")) / successful_predictions
        print(f"   Average response time: {avg_response_time:.2f}s")

    print("✅ Testing completed!")

if __name__ == "__main__":
    main()