#!/usr/bin/env python3
"""
Comprehensive API Testing Script for PolyID PaleoBond Integration

Tests all API endpoints with the specified SMILES molecules:
- /health: System health check
- /run/predict: Single SMILES prediction
- /batch_predict: Batch SMILES predictions
- /metrics: Performance metrics
- Error handling for invalid SMILES

Validates response formats and 22 PaleoBond properties.
"""

import json
import time
import requests
from typing import Dict, List, Any
from datetime import datetime

# Test configuration
API_BASE_URL = "http://localhost:7861"
TIMEOUT = 30  # seconds

# Test SMILES molecules from integration context
TEST_SMILES = [
    "CCO",                                    # Ethanol (simple)
    "CC(C)(C)OC(=O)C=C",                     # Poly(tert-butyl acrylate)
    "CC=C(C)C(=O)OC",                        # PMMA
    "CC(c1ccccc1)",                          # Polystyrene (corrected)
    "CC(=O)OC1=CC=CC=C1C(=O)O"              # Aspirin (complex)
]

# Invalid SMILES for error testing
INVALID_SMILES = [
    "",                                      # Empty string
    "INVALID",                               # Invalid SMILES
    "C1CC1<>",                               # Invalid characters
    None,                                    # None value
    123                                      # Wrong type
]

# Expected PaleoBond properties (22 total)
EXPECTED_PROPERTIES = [
    "glass_transition_temp",
    "melting_temp",
    "decomposition_temp",
    "thermal_stability_score",
    "tensile_strength",
    "elongation_at_break",
    "youngs_modulus",
    "flexibility_score",
    "water_resistance",
    "acid_resistance",
    "base_resistance",
    "solvent_resistance",
    "uv_stability",
    "oxygen_permeability",
    "moisture_vapor_transmission",
    "biodegradability",
    "hydrophane_opal_compatibility",
    "pyrite_compatibility",
    "fossil_compatibility",
    "meteorite_compatibility",
    "analysis_time",
    "confidence_score"
]

class APITester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }

    def log_test_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log individual test results"""
        self.results["tests"][test_name] = {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            **details
        }
        print(f"{'PASS' if success else 'FAIL'} {test_name}: {'PASSED' if success else 'FAILED'}")

    def test_health_endpoint(self) -> bool:
        """Test /health endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            response_time = time.time() - start_time

            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "response_size": len(response.content)
            }

            if success:
                data = response.json()
                details["health_data"] = data
                # Validate health response structure
                required_fields = ["status", "timestamp", "components"]
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    success = False
                    details["error"] = f"Missing required fields: {missing_fields}"
                else:
                    details["components_status"] = data["components"]
            else:
                details["error"] = f"HTTP {response.status_code}: {response.text}"

            self.log_test_result("health_endpoint", success, details)
            return success

        except Exception as e:
            self.log_test_result("health_endpoint", False, {"error": str(e)})
            return False

    def test_single_predict_endpoint(self, smiles: str, test_name: str) -> bool:
        """Test /run/predict endpoint with single SMILES"""
        try:
            payload = {"smiles": smiles}
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/run/predict",
                json=payload,
                timeout=TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response_time = time.time() - start_time

            details = {
                "smiles": smiles,
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "response_size": len(response.content)
            }

            if response.status_code == 200:
                data = response.json()
                details["response_data"] = data

                # Validate PaleoBond format
                required_fields = ["polymer_id", "smiles", "properties", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    success = False
                    details["error"] = f"Missing required fields: {missing_fields}"
                else:
                    # Check properties
                    properties = data.get("properties", {})
                    missing_props = [prop for prop in EXPECTED_PROPERTIES if prop not in properties]
                    if missing_props:
                        success = False
                        details["error"] = f"Missing properties: {missing_props}"
                        details["found_properties"] = list(properties.keys())
                    else:
                        success = True
                        details["properties_count"] = len(properties)
                        details["processing_time"] = data.get("processing_time_seconds")
            elif response.status_code == 400:
                # Expected for invalid SMILES
                data = response.json()
                if "error" in data and "INVALID_SMILES" in data.get("error_code", ""):
                    success = True  # This is expected behavior
                    details["expected_error"] = True
                else:
                    success = False
                    details["error"] = f"Unexpected 400 response: {data}"
            else:
                success = False
                details["error"] = f"HTTP {response.status_code}: {response.text}"

            self.log_test_result(f"single_predict_{test_name}", success, details)
            return success

        except Exception as e:
            self.log_test_result(f"single_predict_{test_name}", False, {"error": str(e), "smiles": smiles})
            return False

    def test_batch_predict_endpoint(self, smiles_list: List[str]) -> bool:
        """Test /batch_predict endpoint with multiple SMILES"""
        try:
            payload = {"smiles_list": smiles_list}
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/batch_predict",
                json=payload,
                timeout=TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response_time = time.time() - start_time

            details = {
                "batch_size": len(smiles_list),
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "response_size": len(response.content)
            }

            if response.status_code in [200, 207]:  # 207 for partial success
                data = response.json()
                details["response_data"] = data

                # Validate batch response structure
                required_fields = ["results", "summary", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    success = False
                    details["error"] = f"Missing required fields: {missing_fields}"
                else:
                    results = data.get("results", [])
                    summary = data.get("summary", {})

                    if len(results) != len(smiles_list):
                        success = False
                        details["error"] = f"Results count mismatch: expected {len(smiles_list)}, got {len(results)}"
                    else:
                        # Check each result
                        successful_results = 0
                        failed_results = 0

                        for i, result in enumerate(results):
                            if "error" not in result:
                                # Validate properties
                                properties = result.get("properties", {})
                                if len(properties) == len(EXPECTED_PROPERTIES):
                                    successful_results += 1
                                else:
                                    failed_results += 1
                                    details[f"result_{i}_missing_props"] = len(EXPECTED_PROPERTIES) - len(properties)
                            else:
                                failed_results += 1

                        success = successful_results > 0  # At least some should succeed
                        details["successful_results"] = successful_results
                        details["failed_results"] = failed_results
                        details["summary_match"] = (summary.get("successful") == successful_results and
                                                   summary.get("failed") == failed_results)
            else:
                success = False
                details["error"] = f"HTTP {response.status_code}: {response.text}"

            self.log_test_result("batch_predict", success, details)
            return success

        except Exception as e:
            self.log_test_result("batch_predict", False, {"error": str(e), "batch_size": len(smiles_list)})
            return False

    def test_metrics_endpoint(self) -> bool:
        """Test /metrics endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/metrics", timeout=TIMEOUT)
            response_time = time.time() - start_time

            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "response_size": len(response.content)
            }

            if success:
                data = response.json()
                details["metrics_data"] = data
                # Validate metrics structure
                expected_fields = ["predictions_total", "predictions_success", "predictions_failed",
                                 "average_response_time", "uptime_seconds", "memory_usage_mb"]
                missing_fields = [field for field in expected_fields if field not in data]
                if missing_fields:
                    success = False
                    details["error"] = f"Missing required fields: {missing_fields}"
            else:
                details["error"] = f"HTTP {response.status_code}: {response.text}"

            self.log_test_result("metrics_endpoint", success, details)
            return success

        except Exception as e:
            self.log_test_result("metrics_endpoint", False, {"error": str(e)})
            return False

    def test_error_handling(self, invalid_input: Any, test_name: str) -> bool:
        """Test error handling with invalid inputs"""
        try:
            payload = {"smiles": invalid_input}
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/run/predict",
                json=payload,
                timeout=TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response_time = time.time() - start_time

            details = {
                "invalid_input": str(invalid_input),
                "status_code": response.status_code,
                "response_time": round(response_time, 3)
            }

            # Should return 400 for invalid inputs
            if response.status_code == 400:
                data = response.json()
                if "error" in data and "error_code" in data:
                    success = True
                    details["error_response"] = data
                else:
                    success = False
                    details["error"] = "Missing error fields in 400 response"
            else:
                success = False
                details["error"] = f"Expected 400, got {response.status_code}: {response.text}"

            self.log_test_result(f"error_handling_{test_name}", success, details)
            return success

        except Exception as e:
            self.log_test_result(f"error_handling_{test_name}", False, {"error": str(e), "invalid_input": str(invalid_input)})
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        print("Starting Comprehensive API Testing")
        print("=" * 50)

        # Test health endpoint
        print("\n1. Testing /health endpoint...")
        self.test_health_endpoint()

        # Test single predictions
        print("\n2. Testing /run/predict endpoint with individual SMILES...")
        for i, smiles in enumerate(TEST_SMILES):
            test_name = f"smiles_{i+1}_{smiles.replace('(', '').replace(')', '').replace('=', '_')[:20]}"
            self.test_single_predict_endpoint(smiles, test_name)

        # Test batch prediction
        print("\n3. Testing /batch_predict endpoint...")
        self.test_batch_predict_endpoint(TEST_SMILES)

        # Test error handling
        print("\n4. Testing error handling...")
        for i, invalid in enumerate(INVALID_SMILES):
            test_name = f"invalid_{i+1}_{str(invalid)[:10]}"
            self.test_error_handling(invalid, test_name)

        # Test metrics endpoint
        print("\n5. Testing /metrics endpoint...")
        self.test_metrics_endpoint()

        # Generate summary
        self.generate_summary()

        print("\n" + "=" * 50)
        print("Testing Complete")
        return self.results

    def generate_summary(self):
        """Generate test summary"""
        tests = self.results["tests"]
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test["success"])
        failed_tests = total_tests - passed_tests

        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": round(passed_tests / total_tests * 100, 2) if total_tests > 0 else 0,
            "test_duration": None  # Could calculate from timestamps
        }

        print("\nTest Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {self.results['summary']['success_rate']:.2f}%")
        # List failed tests
        if failed_tests > 0:
            print(f"\nFailed Tests:")
            for test_name, result in tests.items():
                if not result["success"]:
                    error = result.get("error", "Unknown error")
                    print(f"   - {test_name}: {error}")

def main():
    """Main test execution"""
    tester = APITester()

    try:
        results = tester.run_all_tests()

        # Save results to file
        output_file = "comprehensive_api_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {output_file}")

        # Print summary report
        print("\n" + "=" * 80)
        print("COMPREHENSIVE API TEST REPORT")
        print("=" * 80)

        summary = results["summary"]
        print(f"Test Timestamp: {results['test_timestamp']}")
        print(f"Total Tests Run: {summary['total_tests']}")
        print(f"Tests Passed: {summary['passed_tests']}")
        print(f"Tests Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        if summary['success_rate'] >= 90:
            print("🎉 Overall Result: EXCELLENT")
        elif summary['success_rate'] >= 75:
            print("👍 Overall Result: GOOD")
        elif summary['success_rate'] >= 50:
            print("⚠️  Overall Result: FAIR")
        else:
            print("❌ Overall Result: POOR")

        # Detailed results
        print("\n📋 Detailed Test Results:")
        for test_name, result in results["tests"].items():
            status = "PASS" if result["success"] else "FAIL"
            response_time = result.get("response_time", "N/A")
            print(f"   {test_name}: {status} ({response_time}s)")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())