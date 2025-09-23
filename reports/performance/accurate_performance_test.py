#!/usr/bin/env python3
"""
Accurate PolyID HF Space Performance Analysis
Using correct Gradio API endpoints
"""

import requests
import time
import json
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import statistics

class AccuratePolyIDTester:
    """Accurate performance testing for PolyID HF Space using correct Gradio API"""

    def __init__(self, space_url: str = "https://jkbennitt-polyid-private.hf.space"):
        self.space_url = space_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolyID-Performance-Test/1.0'
        })

        # Test polymer dataset
        self.test_polymers = {
            "simple": {
                "polyethylene": "CC",
                "polypropylene": "CC(C)",
                "polystyrene": "CC(c1ccccc1)"
            },
            "medium": {
                "pmma": "CC(C)(C(=O)OC)",
                "pla": "CC(O)C(=O)",
                "pvc": "CC(Cl)"
            },
            "complex": {
                "pet": "COC(=O)c1ccc(C(=O)O)cc1.OCCO",
                "polycarbonate": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1.O=C(Cl)Cl",
                "polyimide": "c1ccc2c(c1)C(=O)N(c1ccc(C(=O)c3ccc(N4C(=O)c5ccccc5C4=O)cc3)cc1)C2=O"
            }
        }

        self.properties = [
            "Glass Transition Temperature (Tg)",
            "Melting Temperature (Tm)",
            "Density",
            "Elastic Modulus"
        ]

        # Get function hash for API calls
        self.api_info = self._get_api_info()

    def _get_api_info(self) -> Dict:
        """Get Gradio API information"""
        try:
            config_response = self.session.get(f"{self.space_url}/config", timeout=30)
            if config_response.status_code == 200:
                config = config_response.json()

                # Find the main prediction function
                dependencies = config.get("dependencies", [])
                for dep in dependencies:
                    if dep.get("trigger") == "click" and "predict" in str(dep).lower():
                        return {
                            "fn_index": dep.get("targets", [[]])[0][0] if dep.get("targets") else 0,
                            "session_hash": f"session_{int(time.time())}"
                        }

                # Fallback
                return {"fn_index": 1, "session_hash": f"session_{int(time.time())}"}

        except Exception as e:
            print(f"Could not get API info: {e}")

        return {"fn_index": 1, "session_hash": f"session_{int(time.time())}"}

    def _make_gradio_request(self, smiles: str, properties: List[str], timeout: int = 60) -> Dict:
        """Make a request to the Gradio API"""
        try:
            # Gradio API endpoint
            api_url = f"{self.space_url}/gradio_api/call/{self.api_info['fn_index']}"

            # Request payload for main_prediction_interface function
            payload = {
                "data": [smiles, properties],
                "session_hash": self.api_info["session_hash"]
            }

            start_time = time.time()

            # Make the API call
            response = self.session.post(api_url, json=payload, timeout=timeout)

            if response.status_code == 200:
                # Get the event ID for streaming response
                result = response.json()
                event_id = result.get("event_id")

                if event_id:
                    # Stream the result
                    stream_url = f"{self.space_url}/gradio_api/call/{self.api_info['fn_index']}/{event_id}"
                    stream_response = self.session.get(stream_url, timeout=timeout)

                    latency = time.time() - start_time

                    if stream_response.status_code == 200:
                        # Parse the streaming response
                        lines = stream_response.text.strip().split('\n')
                        for line in lines:
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    if data.get("msg") == "process_completed":
                                        return {
                                            "success": True,
                                            "latency_s": latency,
                                            "response_data": data.get("output", {})
                                        }
                                except json.JSONDecodeError:
                                    continue

                return {
                    "success": False,
                    "latency_s": time.time() - start_time,
                    "error": "No valid response data"
                }

            return {
                "success": False,
                "latency_s": time.time() - start_time,
                "error": f"HTTP {response.status_code}",
                "http_status": response.status_code
            }

        except requests.Timeout:
            return {
                "success": False,
                "latency_s": timeout,
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "latency_s": time.time() - start_time if 'start_time' in locals() else 0,
                "error": str(e)
            }

    def test_interface_response_time(self) -> Dict:
        """Test the basic interface loading time"""
        print("Testing Interface Response Time...")

        start_time = time.time()
        try:
            response = self.session.get(self.space_url, timeout=30)
            load_time = time.time() - start_time

            result = {
                "status": "success" if response.status_code == 200 else "error",
                "load_time_s": load_time,
                "http_status": response.status_code,
                "content_size_bytes": len(response.content)
            }

            if response.status_code == 200:
                content = response.text.lower()
                result["components_detected"] = {
                    "gradio": "gradio" in content,
                    "polyid": "polyid" in content,
                    "rdkit": "rdkit" in content,
                    "tensorflow": "tensorflow" in content or "tf." in content,
                    "system_status": "system status" in content
                }

            print(f"  Interface load time: {load_time:.3f}s")
            return result

        except Exception as e:
            return {
                "status": "error",
                "load_time_s": time.time() - start_time,
                "error": str(e)
            }

    def test_prediction_latency(self) -> Dict:
        """Test prediction latency for different polymer complexities"""
        print("Testing Prediction Latency...")

        results = []

        for complexity, polymers in self.test_polymers.items():
            print(f"  Testing {complexity} polymers...")

            for name, smiles in polymers.items():
                print(f"    Testing {name}...")

                # Test with single property first
                test_results = []
                for i in range(3):  # 3 runs for consistency
                    result = self._make_gradio_request(smiles, ["Glass Transition Temperature (Tg)"])
                    test_results.append(result)

                    if result["success"]:
                        print(f"      Run {i+1}: {result['latency_s']:.3f}s")
                    else:
                        print(f"      Run {i+1}: Failed - {result.get('error', 'Unknown error')}")

                    time.sleep(1)  # Rate limiting

                # Calculate statistics
                successful_tests = [r for r in test_results if r["success"]]

                if successful_tests:
                    latencies = [r["latency_s"] for r in successful_tests]

                    polymer_result = {
                        "polymer": name,
                        "complexity": complexity,
                        "smiles": smiles,
                        "smiles_length": len(smiles),
                        "successful_runs": len(successful_tests),
                        "total_runs": len(test_results),
                        "success_rate": len(successful_tests) / len(test_results) * 100,
                        "avg_latency_s": statistics.mean(latencies),
                        "min_latency_s": min(latencies),
                        "max_latency_s": max(latencies),
                        "std_latency_s": statistics.stdev(latencies) if len(latencies) > 1 else 0
                    }

                    results.append(polymer_result)
                    print(f"    {name}: {polymer_result['avg_latency_s']:.3f}s avg ({polymer_result['success_rate']:.0f}% success)")
                else:
                    print(f"    {name}: All runs failed")

        # Calculate overall statistics
        if results:
            all_latencies = [r["avg_latency_s"] for r in results]
            all_success_rates = [r["success_rate"] for r in results]

            summary = {
                "total_polymers_tested": len(results),
                "overall_avg_latency_s": statistics.mean(all_latencies),
                "overall_min_latency_s": min(all_latencies),
                "overall_max_latency_s": max(all_latencies),
                "overall_success_rate": statistics.mean(all_success_rates),
                "detailed_results": results
            }
        else:
            summary = {"error": "No successful predictions completed"}

        return summary

    def test_multi_property_performance(self) -> Dict:
        """Test performance when predicting multiple properties"""
        print("Testing Multi-Property Performance...")

        # Test with increasing number of properties
        property_tests = [
            (["Glass Transition Temperature (Tg)"], "single"),
            (["Glass Transition Temperature (Tg)", "Melting Temperature (Tm)"], "dual"),
            (self.properties, "all")
        ]

        results = []
        test_polymer = "CC"  # Simple polymer for consistency

        for properties, test_name in property_tests:
            print(f"  Testing {test_name} property prediction...")

            test_runs = []
            for i in range(3):
                result = self._make_gradio_request(test_polymer, properties)
                test_runs.append(result)

                if result["success"]:
                    print(f"    Run {i+1}: {result['latency_s']:.3f}s")
                else:
                    print(f"    Run {i+1}: Failed")

                time.sleep(1)

            successful_runs = [r for r in test_runs if r["success"]]

            if successful_runs:
                latencies = [r["latency_s"] for r in successful_runs]

                result_summary = {
                    "test_type": test_name,
                    "num_properties": len(properties),
                    "properties": properties,
                    "successful_runs": len(successful_runs),
                    "success_rate": len(successful_runs) / len(test_runs) * 100,
                    "avg_latency_s": statistics.mean(latencies),
                    "min_latency_s": min(latencies),
                    "max_latency_s": max(latencies)
                }

                results.append(result_summary)
                print(f"    {test_name}: {result_summary['avg_latency_s']:.3f}s avg")

        return {"multi_property_tests": results}

    def test_concurrent_performance(self, num_workers: int = 3) -> Dict:
        """Test concurrent prediction performance"""
        print(f"Testing Concurrent Performance ({num_workers} workers)...")

        def worker_function(worker_id: int, polymer_data: Tuple[str, str]) -> Dict:
            """Worker function for concurrent testing"""
            name, smiles = polymer_data

            result = self._make_gradio_request(smiles, ["Glass Transition Temperature (Tg)"])
            result["worker_id"] = worker_id
            result["polymer"] = name
            result["smiles"] = smiles

            return result

        # Prepare test data
        test_data = list(self.test_polymers["simple"].items())

        # Execute concurrent requests
        start_time = time.time()
        concurrent_results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_function, i, test_data[i % len(test_data)])
                for i in range(num_workers)
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    concurrent_results.append(result)
                except Exception as e:
                    concurrent_results.append({
                        "success": False,
                        "error": str(e),
                        "latency_s": 0
                    })

        total_time = time.time() - start_time

        # Analyze results
        successful_results = [r for r in concurrent_results if r.get("success", False)]

        if successful_results:
            latencies = [r["latency_s"] for r in successful_results]

            summary = {
                "num_workers": num_workers,
                "total_requests": len(concurrent_results),
                "successful_requests": len(successful_results),
                "success_rate": len(successful_results) / len(concurrent_results) * 100,
                "total_time_s": total_time,
                "throughput_per_sec": len(successful_results) / total_time,
                "avg_latency_s": statistics.mean(latencies),
                "min_latency_s": min(latencies),
                "max_latency_s": max(latencies),
                "detailed_results": concurrent_results
            }

            print(f"  Concurrent throughput: {summary['throughput_per_sec']:.2f} predictions/sec")
            print(f"  Success rate: {summary['success_rate']:.1f}%")
        else:
            summary = {
                "error": "No successful concurrent predictions",
                "total_time_s": total_time,
                "total_requests": len(concurrent_results)
            }

        return summary

    def test_stress_limits(self) -> Dict:
        """Test system limits and stress performance"""
        print("Testing Stress Limits...")

        stress_results = {}

        # Test 1: Large SMILES string
        print("  Testing large molecule handling...")
        large_smiles = "CC(C)(c1ccc(C(C)(C)c2ccc(C(C)(C)c3ccc(C(C)(C)c4ccc(C(C)(C)C)cc4)cc3)cc2)cc1)" * 2

        large_molecule_result = self._make_gradio_request(large_smiles, ["Glass Transition Temperature (Tg)"])
        stress_results["large_molecule"] = {
            "smiles_length": len(large_smiles),
            "success": large_molecule_result["success"],
            "latency_s": large_molecule_result["latency_s"],
            "error": large_molecule_result.get("error")
        }

        print(f"    Large molecule ({len(large_smiles)} chars): {'Success' if large_molecule_result['success'] else 'Failed'}")

        # Test 2: All properties at once
        print("  Testing all properties prediction...")
        all_props_result = self._make_gradio_request("CC", self.properties)
        stress_results["all_properties"] = {
            "num_properties": len(self.properties),
            "success": all_props_result["success"],
            "latency_s": all_props_result["latency_s"],
            "error": all_props_result.get("error")
        }

        print(f"    All properties: {'Success' if all_props_result['success'] else 'Failed'}")

        # Test 3: Rapid sequential requests
        print("  Testing rapid sequential requests...")
        rapid_results = []
        for i in range(5):
            result = self._make_gradio_request("CC(C)", ["Density"])
            rapid_results.append(result)
            time.sleep(0.1)  # Very short delay

        successful_rapid = [r for r in rapid_results if r["success"]]
        stress_results["rapid_sequential"] = {
            "total_requests": len(rapid_results),
            "successful_requests": len(successful_rapid),
            "success_rate": len(successful_rapid) / len(rapid_results) * 100,
            "avg_latency_s": statistics.mean([r["latency_s"] for r in successful_rapid]) if successful_rapid else 0
        }

        print(f"    Rapid requests: {len(successful_rapid)}/{len(rapid_results)} successful")

        return stress_results

    def run_comprehensive_analysis(self) -> Dict:
        """Run complete performance analysis"""
        print("="*70)
        print("COMPREHENSIVE POLYID HF SPACE PERFORMANCE ANALYSIS")
        print("="*70)
        print(f"Target: {self.space_url}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        results = {
            "timestamp": datetime.now().isoformat(),
            "space_url": self.space_url,
            "api_info": self.api_info
        }

        # Run all tests
        try:
            results["interface_response"] = self.test_interface_response_time()

            if results["interface_response"]["status"] == "success":
                results["prediction_latency"] = self.test_prediction_latency()
                results["multi_property_performance"] = self.test_multi_property_performance()
                results["concurrent_performance"] = self.test_concurrent_performance()
                results["stress_limits"] = self.test_stress_limits()
            else:
                print("Interface not accessible - skipping detailed tests")

        except Exception as e:
            results["error"] = f"Analysis failed: {str(e)}"
            print(f"Error during analysis: {e}")

        return results

    def generate_performance_report(self, results: Dict) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("="*80)
        report.append("POLYID HF SPACE PERFORMANCE ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Space URL: {results['space_url']}")
        report.append(f"API Function Index: {results['api_info']['fn_index']}")
        report.append("")

        # Interface Performance
        if "interface_response" in results:
            interface = results["interface_response"]
            report.append("INTERFACE PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Status: {'OK' if interface['status'] == 'success' else 'FAILED'}")
            report.append(f"Load Time: {interface['load_time_s']:.3f}s")
            report.append(f"Content Size: {interface.get('content_size_bytes', 0):,} bytes")

            if "components_detected" in interface:
                report.append("Components Detected:")
                for comp, detected in interface["components_detected"].items():
                    status = "YES" if detected else "NO"
                    report.append(f"  {comp.capitalize()}: {status}")
            report.append("")

        # Prediction Latency
        if "prediction_latency" in results:
            latency = results["prediction_latency"]
            if "overall_avg_latency_s" in latency:
                report.append("PREDICTION LATENCY PERFORMANCE")
                report.append("-" * 40)
                report.append(f"Average Latency: {latency['overall_avg_latency_s']:.3f}s")
                report.append(f"Range: {latency['overall_min_latency_s']:.3f}s - {latency['overall_max_latency_s']:.3f}s")
                report.append(f"Overall Success Rate: {latency['overall_success_rate']:.1f}%")
                report.append(f"Polymers Tested: {latency['total_polymers_tested']}")

                # Latency by complexity
                if "detailed_results" in latency:
                    report.append("\nLatency by Complexity:")
                    for result in latency["detailed_results"]:
                        report.append(f"  {result['complexity'].capitalize()} - {result['polymer']}: {result['avg_latency_s']:.3f}s")
                report.append("")

        # Multi-Property Performance
        if "multi_property_performance" in results:
            multi_prop = results["multi_property_performance"]
            if "multi_property_tests" in multi_prop:
                report.append("MULTI-PROPERTY PERFORMANCE")
                report.append("-" * 40)
                for test in multi_prop["multi_property_tests"]:
                    report.append(f"{test['test_type'].capitalize()} ({test['num_properties']} properties): {test['avg_latency_s']:.3f}s")
                report.append("")

        # Concurrent Performance
        if "concurrent_performance" in results:
            concurrent = results["concurrent_performance"]
            if "throughput_per_sec" in concurrent:
                report.append("CONCURRENT PERFORMANCE")
                report.append("-" * 40)
                report.append(f"Throughput: {concurrent['throughput_per_sec']:.2f} predictions/sec")
                report.append(f"Workers: {concurrent['num_workers']}")
                report.append(f"Success Rate: {concurrent['success_rate']:.1f}%")
                report.append(f"Average Latency: {concurrent['avg_latency_s']:.3f}s")
                report.append("")

        # Stress Test Results
        if "stress_limits" in results:
            stress = results["stress_limits"]
            report.append("STRESS TEST RESULTS")
            report.append("-" * 40)

            if "large_molecule" in stress:
                lm = stress["large_molecule"]
                report.append(f"Large Molecule ({lm['smiles_length']} chars): {'SUCCESS' if lm['success'] else 'FAILED'}")
                if lm['success']:
                    report.append(f"  Latency: {lm['latency_s']:.3f}s")

            if "all_properties" in stress:
                ap = stress["all_properties"]
                report.append(f"All Properties ({ap['num_properties']} props): {'SUCCESS' if ap['success'] else 'FAILED'}")
                if ap['success']:
                    report.append(f"  Latency: {ap['latency_s']:.3f}s")

            if "rapid_sequential" in stress:
                rs = stress["rapid_sequential"]
                report.append(f"Rapid Sequential: {rs['successful_requests']}/{rs['total_requests']} successful")
                report.append(f"  Success Rate: {rs['success_rate']:.1f}%")

            report.append("")

        # Performance Assessment & Recommendations
        report.append("PERFORMANCE ASSESSMENT & RECOMMENDATIONS")
        report.append("-" * 80)

        recommendations = []

        # Latency assessment
        if "prediction_latency" in results and "overall_avg_latency_s" in results["prediction_latency"]:
            avg_latency = results["prediction_latency"]["overall_avg_latency_s"]
            if avg_latency < 1.0:
                recommendations.append("✓ Excellent prediction latency (<1s average)")
            elif avg_latency < 2.0:
                recommendations.append("✓ Good prediction latency (<2s average)")
            elif avg_latency < 5.0:
                recommendations.append("⚠ Moderate prediction latency - consider optimization")
            else:
                recommendations.append("✗ High prediction latency - optimization needed")

        # Throughput assessment
        if "concurrent_performance" in results and "throughput_per_sec" in results["concurrent_performance"]:
            throughput = results["concurrent_performance"]["throughput_per_sec"]
            if throughput > 2.0:
                recommendations.append("✓ Good concurrent throughput")
            elif throughput > 1.0:
                recommendations.append("⚠ Moderate concurrent throughput")
            else:
                recommendations.append("✗ Low concurrent throughput - investigate bottlenecks")

        # Success rate assessment
        if "prediction_latency" in results and "overall_success_rate" in results["prediction_latency"]:
            success_rate = results["prediction_latency"]["overall_success_rate"]
            if success_rate > 95:
                recommendations.append("✓ Excellent reliability (>95% success rate)")
            elif success_rate > 90:
                recommendations.append("✓ Good reliability (>90% success rate)")
            elif success_rate > 80:
                recommendations.append("⚠ Moderate reliability - monitor error patterns")
            else:
                recommendations.append("✗ Poor reliability - investigate failures")

        # General recommendations
        recommendations.extend([
            "",
            "OPTIMIZATION RECOMMENDATIONS:",
            "• Implement request caching for repeated SMILES",
            "• Monitor memory usage during peak loads",
            "• Consider model optimization for faster inference",
            "• Implement request queuing for high concurrency",
            "• Add monitoring for error patterns and failures"
        ])

        for rec in recommendations:
            report.append(rec)

        report.append("")
        report.append("="*80)

        return "\n".join(report)

def main():
    """Main analysis workflow"""
    tester = AccuratePolyIDTester()

    # Run comprehensive analysis
    results = tester.run_comprehensive_analysis()

    # Generate report
    report = tester.generate_performance_report(results)
    print("\n" + report)

    # Save results
    timestamp = int(time.time())

    # Save JSON results
    json_file = f"accurate_performance_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save report
    report_file = f"accurate_performance_report_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\nResults saved:")
    print(f"  JSON: {json_file}")
    print(f"  Report: {report_file}")

if __name__ == "__main__":
    main()