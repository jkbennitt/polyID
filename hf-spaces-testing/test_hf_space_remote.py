#!/usr/bin/env python
"""
Remote verification of PolyID HF Space chemistry stack
Tests the deployed space at: https://huggingface.co/spaces/jkbennitt/polyid-private
"""

import requests
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime


class HFSpaceVerifier:
    """Verify deployed HF Space functionality"""

    def __init__(self, space_url: str):
        self.space_url = space_url.rstrip('/')
        self.api_url = f"{self.space_url}/api/predict"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "space_url": self.space_url,
            "tests": []
        }

    def test_space_availability(self) -> Dict:
        """Test if the HF Space is accessible"""
        print("\n" + "="*60)
        print("Testing HF Space Availability")
        print("-"*60)

        try:
            response = requests.get(self.space_url, timeout=30)

            if response.status_code == 200:
                print(f"✓ Space is accessible (Status: {response.status_code})")

                # Check for key indicators in the response
                content = response.text.lower()

                indicators = {
                    "gradio": "gradio" in content,
                    "polyid": "polyid" in content,
                    "polymer": "polymer" in content,
                    "rdkit": "rdkit" in content,
                    "tensorflow": "tensorflow" in content
                }

                print("\nContent indicators found:")
                for key, found in indicators.items():
                    status = "✓" if found else "✗"
                    print(f"  {status} {key.capitalize()}")

                return {
                    "status": "success",
                    "http_status": response.status_code,
                    "indicators": indicators
                }
            else:
                print(f"✗ Space returned status: {response.status_code}")
                return {
                    "status": "error",
                    "http_status": response.status_code,
                    "error": f"HTTP {response.status_code}"
                }

        except requests.RequestException as e:
            print(f"✗ Failed to connect: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def test_gradio_interface(self) -> Dict:
        """Test if Gradio interface is properly loaded"""
        print("\n" + "="*60)
        print("Testing Gradio Interface")
        print("-"*60)

        try:
            # Try to get Gradio config
            config_url = f"{self.space_url}/config"
            response = requests.get(config_url, timeout=30)

            if response.status_code == 200:
                try:
                    config = response.json()
                    print("✓ Gradio config retrieved")

                    # Check for expected components
                    if "components" in config:
                        print(f"  • Components found: {len(config['components'])}")

                    return {
                        "status": "success",
                        "config_available": True,
                        "components": len(config.get("components", []))
                    }
                except json.JSONDecodeError:
                    print("✗ Could not parse Gradio config")
                    return {
                        "status": "partial",
                        "config_available": False
                    }
            else:
                print(f"✗ Config endpoint returned: {response.status_code}")
                return {
                    "status": "error",
                    "http_status": response.status_code
                }

        except requests.RequestException as e:
            print(f"✗ Failed to test Gradio: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def test_api_endpoint(self) -> Dict:
        """Test the API prediction endpoint"""
        print("\n" + "="*60)
        print("Testing API Endpoint")
        print("-"*60)

        test_cases = [
            {
                "name": "Polyethylene",
                "smiles": "CC",
                "properties": ["Glass Transition Temperature (Tg)", "Density"]
            },
            {
                "name": "Polystyrene",
                "smiles": "CC(c1ccccc1)",
                "properties": ["Glass Transition Temperature (Tg)", "Melting Temperature (Tm)"]
            },
            {
                "name": "Complex polymer",
                "smiles": "CC(C)(C(=O)OC)",
                "properties": ["Density", "Elastic Modulus"]
            }
        ]

        results = []

        for test_case in test_cases:
            print(f"\nTesting: {test_case['name']}")
            print(f"  SMILES: {test_case['smiles']}")

            try:
                # Prepare API request
                payload = {
                    "data": [
                        test_case["smiles"],
                        test_case["properties"]
                    ]
                }

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"  ✓ Prediction successful")

                    # Check response structure
                    if "data" in result:
                        print(f"    • Response contains {len(result['data'])} items")

                    results.append({
                        "test": test_case["name"],
                        "status": "success",
                        "response": result
                    })
                else:
                    print(f"  ✗ API returned: {response.status_code}")
                    results.append({
                        "test": test_case["name"],
                        "status": "error",
                        "http_status": response.status_code
                    })

            except requests.RequestException as e:
                print(f"  ✗ Request failed: {e}")
                results.append({
                    "test": test_case["name"],
                    "status": "error",
                    "error": str(e)
                })

            time.sleep(2)  # Rate limiting

        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"\n✓ Successful predictions: {success_count}/{len(test_cases)}")

        return {
            "status": "success" if success_count > 0 else "error",
            "tests_passed": success_count,
            "tests_total": len(test_cases),
            "results": results
        }

    def check_chemistry_components(self) -> Dict:
        """Check if chemistry components are mentioned in the space"""
        print("\n" + "="*60)
        print("Checking Chemistry Components")
        print("-"*60)

        try:
            response = requests.get(self.space_url, timeout=30)

            if response.status_code == 200:
                content = response.text

                # Look for component status indicators
                components = {
                    "RDKit": ["rdkit", "RDKit"],
                    "NFP": ["nfp", "NFP", "neural fingerprint"],
                    "m2p": ["m2p", "monomer", "polymer"],
                    "TensorFlow": ["tensorflow", "TensorFlow", "tf."],
                    "PolyID": ["polyid", "PolyID", "SingleModel", "MultiModel"]
                }

                found_components = {}

                for component, keywords in components.items():
                    found = any(keyword in content for keyword in keywords)
                    found_components[component] = found
                    status = "✓" if found else "✗"
                    print(f"  {status} {component}")

                # Check for status indicators in the UI
                if "System Status" in content:
                    print("\n✓ System status section found")

                    # Look for specific status messages
                    status_messages = {
                        "RDKit: Available": "RDKit operational",
                        "NFP: Available": "NFP operational",
                        "PolyID: Available": "PolyID operational",
                        "tensorflow": "TensorFlow mentioned"
                    }

                    for message, description in status_messages.items():
                        if message in content:
                            print(f"    ✓ {description}")

                return {
                    "status": "success",
                    "components_found": found_components,
                    "system_status_section": "System Status" in content
                }
            else:
                print(f"✗ Could not retrieve space content")
                return {
                    "status": "error",
                    "http_status": response.status_code
                }

        except requests.RequestException as e:
            print(f"✗ Failed to check components: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def generate_report(self) -> str:
        """Generate comprehensive report"""
        report = []
        report.append("="*70)
        report.append("HF SPACE CHEMISTRY STACK VERIFICATION REPORT")
        report.append("="*70)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Space URL: {self.results['space_url']}")
        report.append("")

        # Calculate summary
        total_tests = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"]
                    if t.get("status") == "success")

        report.append("SUMMARY")
        report.append("-"*70)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed}/{total_tests}")
        report.append(f"Success Rate: {(passed/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
        report.append("")

        # Test results
        report.append("TEST RESULTS")
        report.append("-"*70)

        for test in self.results["tests"]:
            status_symbol = "✓" if test.get("status") == "success" else "✗"
            report.append(f"\n{status_symbol} {test['name']}")
            report.append(f"   Status: {test.get('status', 'unknown')}")

            if "details" in test:
                for key, value in test["details"].items():
                    if key != "status":
                        report.append(f"   {key}: {value}")

        report.append("")
        report.append("="*70)

        return "\n".join(report)

    def run_all_tests(self):
        """Run all verification tests"""
        tests = [
            ("Space Availability", self.test_space_availability),
            ("Gradio Interface", self.test_gradio_interface),
            ("Chemistry Components", self.check_chemistry_components),
            ("API Endpoint", self.test_api_endpoint)
        ]

        for test_name, test_func in tests:
            result = test_func()
            self.results["tests"].append({
                "name": test_name,
                "status": result.get("status"),
                "details": result
            })

        # Generate and print report
        report = self.generate_report()
        print("\n" + report)

        # Save results
        with open("hf_space_verification.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: hf_space_verification.json")


def main():
    """Main verification workflow"""
    print("Starting HF Space Chemistry Stack Verification")
    print("Target: https://huggingface.co/spaces/jkbennitt/polyid-private")

    # Initialize verifier
    verifier = HFSpaceVerifier("https://huggingface.co/spaces/jkbennitt/polyid-private")

    # Run all tests
    verifier.run_all_tests()

    # Check if all critical tests passed
    critical_tests = ["Space Availability", "Chemistry Components"]
    critical_passed = all(
        t.get("status") == "success"
        for t in verifier.results["tests"]
        if t["name"] in critical_tests
    )

    if critical_passed:
        print("\n✓ HF Space chemistry stack verification completed!")
        return 0
    else:
        print("\n✗ Critical issues detected in HF Space deployment")
        return 1


if __name__ == "__main__":
    exit(main())