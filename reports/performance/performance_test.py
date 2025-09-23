#!/usr/bin/env python3
"""
PolyID HF Space Performance Analysis
Comprehensive performance testing for https://jkbennitt-polyid-private.hf.space
"""

import requests
import time
import json
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import statistics

class PolyIDPerformanceTester:
    """Performance testing for PolyID HF Space"""

    def __init__(self, space_url: str = "https://jkbennitt-polyid-private.hf.space"):
        self.space_url = space_url.rstrip('/')
        self.api_url = f"{self.space_url}/api/predict"
        self.results = {}

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

    def test_space_availability(self) -> Dict:
        """Test basic space availability and response time"""
        print("Testing Space Availability...")

        start_time = time.time()
        try:
            response = requests.get(self.space_url, timeout=30)
            response_time = time.time() - start_time

            result = {
                "status": "success" if response.status_code == 200 else "error",
                "http_status": response.status_code,
                "response_time_s": response_time,
                "content_length": len(response.text) if response.status_code == 200 else 0
            }

            if response.status_code == 200:
                content = response.text.lower()
                result["components_found"] = {
                    "rdkit": "rdkit" in content,
                    "nfp": "nfp" in content,
                    "polyid": "polyid" in content,
                    "tensorflow": "tensorflow" in content,
                    "gradio": "gradio" in content
                }

            print(f"  Space accessible: {response.status_code == 200}")
            print(f"  Response time: {response_time:.3f}s")
            return result

        except Exception as e:
            print(f"  Error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response_time_s": time.time() - start_time
            }

    def test_single_prediction_latency(self) -> Dict:
        """Test latency for single predictions"""
        print("Testing Single Prediction Latency...")

        results = []

        for complexity, polymers in self.test_polymers.items():
            print(f"  Testing {complexity} polymers...")

            for name, smiles in polymers.items():
                latency_results = []

                # Test each polymer 3 times for consistency
                for i in range(3):
                    start_time = time.time()

                    try:
                        payload = {
                            "data": [smiles, ["Glass Transition Temperature (Tg)"]]
                        }

                        response = requests.post(
                            self.api_url,
                            json=payload,
                            timeout=60
                        )

                        latency = time.time() - start_time
                        latency_results.append(latency)

                        if response.status_code != 200:
                            print(f"    {name}: HTTP {response.status_code}")

                    except Exception as e:
                        print(f"    {name}: Error - {str(e)}")
                        latency_results.append(None)

                    time.sleep(1)  # Rate limiting

                # Calculate statistics for this polymer
                valid_latencies = [l for l in latency_results if l is not None]
                if valid_latencies:
                    result = {
                        "polymer": name,
                        "complexity": complexity,
                        "smiles": smiles,
                        "avg_latency_s": statistics.mean(valid_latencies),
                        "min_latency_s": min(valid_latencies),
                        "max_latency_s": max(valid_latencies),
                        "std_latency_s": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0,
                        "success_rate": len(valid_latencies) / len(latency_results) * 100
                    }
                    results.append(result)

                    print(f"    {name}: {result['avg_latency_s']:.3f}s avg")

        # Overall statistics
        all_latencies = [r['avg_latency_s'] for r in results]
        if all_latencies:
            summary = {
                "total_tests": len(results),
                "overall_avg_latency_s": statistics.mean(all_latencies),
                "overall_min_latency_s": min(all_latencies),
                "overall_max_latency_s": max(all_latencies),
                "overall_std_latency_s": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
                "detailed_results": results
            }
        else:
            summary = {"error": "No successful predictions"}

        return summary

    def test_concurrent_throughput(self, num_workers: int = 4) -> Dict:
        """Test concurrent prediction throughput"""
        print(f"Testing Concurrent Throughput ({num_workers} workers)...")

        def make_prediction(polymer_data):
            """Single prediction worker function"""
            name, smiles = polymer_data
            start_time = time.time()

            try:
                payload = {
                    "data": [smiles, ["Glass Transition Temperature (Tg)"]]
                }

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=60
                )

                latency = time.time() - start_time

                return {
                    "polymer": name,
                    "smiles": smiles,
                    "latency_s": latency,
                    "success": response.status_code == 200,
                    "http_status": response.status_code
                }

            except Exception as e:
                return {
                    "polymer": name,
                    "smiles": smiles,
                    "latency_s": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }

        # Prepare test data - use all polymers
        test_data = []
        for polymers in self.test_polymers.values():
            test_data.extend(polymers.items())

        # Execute concurrent requests
        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_prediction, polymer_data)
                      for polymer_data in test_data]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e)
                    })

        total_time = time.time() - start_time

        # Calculate metrics
        successful_results = [r for r in results if r.get("success", False)]

        if successful_results:
            latencies = [r["latency_s"] for r in successful_results]

            summary = {
                "total_requests": len(test_data),
                "successful_requests": len(successful_results),
                "success_rate": len(successful_results) / len(test_data) * 100,
                "total_time_s": total_time,
                "throughput_per_sec": len(successful_results) / total_time,
                "avg_latency_s": statistics.mean(latencies),
                "min_latency_s": min(latencies),
                "max_latency_s": max(latencies),
                "workers_used": num_workers,
                "detailed_results": results
            }
        else:
            summary = {
                "error": "No successful concurrent predictions",
                "total_requests": len(test_data),
                "total_time_s": total_time
            }

        print(f"  Throughput: {summary.get('throughput_per_sec', 0):.2f} predictions/sec")
        print(f"  Success rate: {summary.get('success_rate', 0):.1f}%")

        return summary

    def test_complex_molecule_handling(self) -> Dict:
        """Test performance with complex polymer structures"""
        print("Testing Complex Molecule Handling...")

        results = []

        # Test complex polymers with multiple properties
        complex_polymers = self.test_polymers["complex"]

        for name, smiles in complex_polymers.items():
            print(f"  Testing {name}...")

            start_time = time.time()

            try:
                payload = {
                    "data": [smiles, self.properties]  # All properties
                }

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=120  # Longer timeout for complex molecules
                )

                latency = time.time() - start_time

                result = {
                    "polymer": name,
                    "smiles": smiles,
                    "smiles_length": len(smiles),
                    "num_properties": len(self.properties),
                    "latency_s": latency,
                    "success": response.status_code == 200,
                    "http_status": response.status_code
                }

                if response.status_code == 200:
                    try:
                        data = response.json()
                        result["response_size"] = len(str(data))
                    except:
                        result["response_size"] = 0

                results.append(result)
                print(f"    {name}: {latency:.3f}s")

            except Exception as e:
                results.append({
                    "polymer": name,
                    "smiles": smiles,
                    "latency_s": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                })
                print(f"    {name}: Error - {str(e)}")

            time.sleep(2)  # Rate limiting

        # Calculate summary
        successful_results = [r for r in results if r.get("success", False)]

        if successful_results:
            latencies = [r["latency_s"] for r in successful_results]

            summary = {
                "total_complex_polymers": len(complex_polymers),
                "successful_predictions": len(successful_results),
                "success_rate": len(successful_results) / len(complex_polymers) * 100,
                "avg_latency_s": statistics.mean(latencies),
                "max_latency_s": max(latencies),
                "avg_smiles_length": statistics.mean([r["smiles_length"] for r in successful_results]),
                "detailed_results": results
            }
        else:
            summary = {
                "error": "No successful complex polymer predictions",
                "total_complex_polymers": len(complex_polymers)
            }

        return summary

    def test_startup_performance(self) -> Dict:
        """Test cold start and warm-up performance"""
        print("Testing Startup Performance...")

        # Test initial request (cold start)
        print("  Testing cold start...")
        cold_start_time = time.time()

        try:
            payload = {
                "data": ["CC", ["Glass Transition Temperature (Tg)"]]
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # Longer timeout for cold start
            )

            cold_start_latency = time.time() - cold_start_time
            cold_start_success = response.status_code == 200

        except Exception as e:
            cold_start_latency = time.time() - cold_start_time
            cold_start_success = False

        print(f"    Cold start: {cold_start_latency:.3f}s")

        # Test subsequent requests (warm state)
        print("  Testing warm state...")
        warm_latencies = []

        for i in range(3):
            start_time = time.time()

            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=60
                )

                latency = time.time() - start_time
                warm_latencies.append(latency)

            except Exception as e:
                warm_latencies.append(time.time() - start_time)

            time.sleep(1)

        avg_warm_latency = statistics.mean(warm_latencies) if warm_latencies else 0
        print(f"    Warm state avg: {avg_warm_latency:.3f}s")

        # Calculate speedup
        speedup = cold_start_latency / avg_warm_latency if avg_warm_latency > 0 else 0

        return {
            "cold_start_latency_s": cold_start_latency,
            "cold_start_success": cold_start_success,
            "warm_state_avg_latency_s": avg_warm_latency,
            "warm_state_latencies": warm_latencies,
            "speedup_factor": speedup,
            "optimization_effective": speedup > 1.5
        }

    def run_comprehensive_analysis(self) -> Dict:
        """Run all performance tests"""
        print("="*60)
        print("PolyID HF Space Performance Analysis")
        print("="*60)
        print(f"Target: {self.space_url}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run all tests
        test_results = {}

        test_results["availability"] = self.test_space_availability()

        if test_results["availability"]["status"] == "success":
            test_results["single_prediction_latency"] = self.test_single_prediction_latency()
            test_results["concurrent_throughput"] = self.test_concurrent_throughput()
            test_results["complex_molecule_handling"] = self.test_complex_molecule_handling()
            test_results["startup_performance"] = self.test_startup_performance()
        else:
            print("Space not available - skipping detailed tests")

        # Store complete results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "space_url": self.space_url,
            "test_results": test_results
        }

        return self.results

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.results:
            return "No test results available"

        report = []
        report.append("="*80)
        report.append("POLYID HF SPACE PERFORMANCE ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Space URL: {self.results['space_url']}")
        report.append("")

        test_results = self.results["test_results"]

        # Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-"*80)

        if "availability" in test_results:
            avail = test_results["availability"]
            report.append(f"Space Accessibility: {'OK' if avail['status'] == 'success' else 'FAILED'}")
            if avail["status"] == "success":
                report.append(f"  Response Time: {avail['response_time_s']:.3f}s")

        if "single_prediction_latency" in test_results:
            latency = test_results["single_prediction_latency"]
            if "overall_avg_latency_s" in latency:
                report.append(f"Average Prediction Latency: {latency['overall_avg_latency_s']:.3f}s")
                report.append(f"  Range: {latency['overall_min_latency_s']:.3f}s - {latency['overall_max_latency_s']:.3f}s")

        if "concurrent_throughput" in test_results:
            throughput = test_results["concurrent_throughput"]
            if "throughput_per_sec" in throughput:
                report.append(f"Concurrent Throughput: {throughput['throughput_per_sec']:.2f} predictions/sec")
                report.append(f"  Success Rate: {throughput['success_rate']:.1f}%")

        if "startup_performance" in test_results:
            startup = test_results["startup_performance"]
            report.append(f"Cold Start Performance: {startup['cold_start_latency_s']:.3f}s")
            report.append(f"Warm State Performance: {startup['warm_state_avg_latency_s']:.3f}s")
            if startup["speedup_factor"] > 0:
                report.append(f"  Speedup Factor: {startup['speedup_factor']:.2f}x")

        report.append("")

        # Detailed Results
        report.append("DETAILED TEST RESULTS")
        report.append("-"*80)

        # Single Prediction Analysis
        if "single_prediction_latency" in test_results:
            latency = test_results["single_prediction_latency"]
            report.append("")
            report.append("Single Prediction Latency by Complexity:")

            if "detailed_results" in latency:
                for result in latency["detailed_results"]:
                    report.append(f"  {result['complexity'].capitalize()} - {result['polymer']}: {result['avg_latency_s']:.3f}s")

        # Complex Molecule Handling
        if "complex_molecule_handling" in test_results:
            complex_test = test_results["complex_molecule_handling"]
            report.append("")
            report.append("Complex Molecule Handling:")
            report.append(f"  Success Rate: {complex_test.get('success_rate', 0):.1f}%")

            if "detailed_results" in complex_test:
                for result in complex_test["detailed_results"]:
                    if result.get("success", False):
                        report.append(f"  {result['polymer']}: {result['latency_s']:.3f}s (SMILES length: {result['smiles_length']})")

        report.append("")

        # Performance Recommendations
        report.append("PERFORMANCE RECOMMENDATIONS")
        report.append("-"*80)

        recommendations = []

        # Latency recommendations
        if "single_prediction_latency" in test_results:
            latency = test_results["single_prediction_latency"]
            avg_latency = latency.get("overall_avg_latency_s", 0)

            if avg_latency > 3.0:
                recommendations.append("• High prediction latency detected - consider model optimization")
            elif avg_latency > 1.5:
                recommendations.append("• Moderate prediction latency - monitor performance trends")
            else:
                recommendations.append("• Good prediction latency performance")

        # Throughput recommendations
        if "concurrent_throughput" in test_results:
            throughput = test_results["concurrent_throughput"]
            tps = throughput.get("throughput_per_sec", 0)

            if tps < 1.0:
                recommendations.append("• Low concurrent throughput - investigate bottlenecks")
            elif tps < 3.0:
                recommendations.append("• Moderate throughput - consider load balancing")
            else:
                recommendations.append("• Good concurrent performance")

        # Startup recommendations
        if "startup_performance" in test_results:
            startup = test_results["startup_performance"]
            cold_start = startup.get("cold_start_latency_s", 0)

            if cold_start > 10.0:
                recommendations.append("• Long cold start time - implement model caching")
            elif cold_start > 5.0:
                recommendations.append("• Moderate cold start time - consider warm-up optimization")

        # General recommendations
        recommendations.extend([
            "• Monitor memory usage during peak loads",
            "• Implement request queuing for high concurrency",
            "• Consider batch processing for multiple predictions",
            "• Optimize model loading for faster startup"
        ])

        for rec in recommendations:
            report.append(rec)

        report.append("")
        report.append("="*80)

        return "\n".join(report)

def main():
    """Main performance testing workflow"""
    tester = PolyIDPerformanceTester()

    # Run comprehensive analysis
    results = tester.run_comprehensive_analysis()

    # Generate and display report
    report = tester.generate_performance_report()
    print("\n" + report)

    # Save results
    timestamp = int(time.time())

    # Save JSON results
    with open(f"polyid_performance_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save text report
    with open(f"polyid_performance_report_{timestamp}.txt", "w") as f:
        f.write(report)

    print(f"\nResults saved:")
    print(f"  JSON: polyid_performance_results_{timestamp}.json")
    print(f"  Report: polyid_performance_report_{timestamp}.txt")

if __name__ == "__main__":
    main()