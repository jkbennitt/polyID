#!/usr/bin/env python3
"""
Chemistry Stack Performance Analysis for PolyID
Tests the actual app.py deployment to identify performance bottlenecks
"""

import os
import sys
import time
import psutil
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

@dataclass
class ChemistryTestResult:
    """Test result for chemistry stack performance"""
    test_name: str
    success: bool
    duration: float
    memory_mb: float
    cpu_percent: float
    throughput: Optional[float] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Dict = None

class ChemistryStackTester:
    """Performance tester focused on chemistry stack components"""

    def __init__(self):
        self.process = psutil.Process()
        self.results = []

    def log(self, message):
        """Simple logging"""
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_percent(self):
        """Get current CPU usage"""
        return self.process.cpu_percent()

    def test_app_imports(self) -> ChemistryTestResult:
        """Test app.py import performance and dependency loading"""
        self.log("Testing app.py imports and dependency loading...")

        start_time = time.time()
        start_memory = self.get_memory_usage()

        import_details = {}
        import_errors = []

        try:
            # Import app.py and capture startup output
            import app

            # Check what's available
            import_details['rdkit_available'] = app.rdkit is not None
            import_details['nfp_available'] = app.nfp is not None
            import_details['shortuuid_available'] = app.shortuuid is not None
            import_details['polyid_available'] = app.POLYID_AVAILABLE

            # Test basic function availability
            import_details['validate_smiles_available'] = hasattr(app, 'validate_smiles')
            import_details['calculate_molecular_properties_available'] = hasattr(app, 'calculate_molecular_properties')
            import_details['predict_polymer_properties_available'] = hasattr(app, 'predict_polymer_properties')

            success = True

        except Exception as e:
            import_errors.append(str(e))
            success = False

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        return ChemistryTestResult(
            test_name="App Imports",
            success=success,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            error="; ".join(import_errors) if import_errors else None,
            details=import_details
        )

    def test_smiles_validation_performance(self) -> ChemistryTestResult:
        """Test SMILES validation performance using app functions"""
        self.log("Testing SMILES validation performance...")

        try:
            from app import validate_smiles
        except ImportError as e:
            return ChemistryTestResult(
                test_name="SMILES Validation",
                success=False,
                duration=0,
                memory_mb=0,
                cpu_percent=0,
                error=f"Cannot import validate_smiles: {e}"
            )

        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Test polymer SMILES with varying complexity
        test_smiles = [
            # Simple polymers
            "CC",  # Polyethylene
            "CC(C)",  # Polypropylene
            "CC(Cl)",  # PVC

            # Medium complexity
            "CC(c1ccccc1)",  # Polystyrene
            "CC(C)(C(=O)OC)",  # PMMA
            "CC(O)C(=O)",  # PLA

            # Complex polymers
            "COC(=O)c1ccc(C(=O)O)cc1",  # PET monomer
            "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",  # PC monomer
            "c1ccc2c(c1)C(=O)N(c1ccc(C(=O)c3ccc(N4C(=O)c5ccccc5C4=O)cc3)cc1)C2=O",  # Complex polyimide

            # Edge cases
            "",  # Empty string
            "INVALID",  # Invalid SMILES
            "C" * 100,  # Very long chain
        ]

        validation_results = []
        validation_times = []
        successful_validations = 0

        for smiles in test_smiles:
            validation_start = time.time()

            try:
                is_valid, message = validate_smiles(smiles)
                validation_time = time.time() - validation_start

                validation_results.append({
                    'smiles': smiles[:50] + '...' if len(smiles) > 50 else smiles,
                    'valid': is_valid,
                    'message': message,
                    'time_ms': validation_time * 1000
                })

                validation_times.append(validation_time)

                if is_valid:
                    successful_validations += 1

            except Exception as e:
                validation_results.append({
                    'smiles': smiles[:50] + '...' if len(smiles) > 50 else smiles,
                    'error': str(e),
                    'time_ms': (time.time() - validation_start) * 1000
                })

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        # Calculate performance metrics
        avg_validation_time = sum(validation_times) / len(validation_times) if validation_times else 0
        throughput = len(test_smiles) / duration if duration > 0 else 0

        return ChemistryTestResult(
            test_name="SMILES Validation",
            success=successful_validations > 0,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            throughput=throughput,
            response_time_ms=avg_validation_time * 1000,
            details={
                'total_smiles': len(test_smiles),
                'successful_validations': successful_validations,
                'validation_rate': successful_validations / len(test_smiles) * 100,
                'avg_validation_time_ms': avg_validation_time * 1000,
                'min_validation_time_ms': min(validation_times) * 1000 if validation_times else 0,
                'max_validation_time_ms': max(validation_times) * 1000 if validation_times else 0,
                'validation_results': validation_results[:5]  # First 5 results as sample
            }
        )

    def test_molecular_properties_performance(self) -> ChemistryTestResult:
        """Test molecular property calculation performance"""
        self.log("Testing molecular property calculation performance...")

        try:
            from app import calculate_molecular_properties
        except ImportError as e:
            return ChemistryTestResult(
                test_name="Molecular Properties",
                success=False,
                duration=0,
                memory_mb=0,
                cpu_percent=0,
                error=f"Cannot import calculate_molecular_properties: {e}"
            )

        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Test various polymer types
        test_polymers = [
            "CC",  # PE
            "CC(C)",  # PP
            "CC(c1ccccc1)",  # PS
            "CC(C)(C(=O)OC)",  # PMMA
            "CC(O)C(=O)",  # PLA
        ]

        property_results = []
        calculation_times = []
        successful_calculations = 0

        for smiles in test_polymers:
            calc_start = time.time()

            try:
                mol_props = calculate_molecular_properties(smiles)
                calc_time = time.time() - calc_start

                if 'error' not in mol_props:
                    successful_calculations += 1
                    property_results.append({
                        'smiles': smiles,
                        'properties': mol_props,
                        'time_ms': calc_time * 1000
                    })
                else:
                    property_results.append({
                        'smiles': smiles,
                        'error': mol_props['error'],
                        'time_ms': calc_time * 1000
                    })

                calculation_times.append(calc_time)

            except Exception as e:
                property_results.append({
                    'smiles': smiles,
                    'exception': str(e),
                    'time_ms': (time.time() - calc_start) * 1000
                })

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        # Calculate performance metrics
        avg_calc_time = sum(calculation_times) / len(calculation_times) if calculation_times else 0
        throughput = len(test_polymers) / duration if duration > 0 else 0

        return ChemistryTestResult(
            test_name="Molecular Properties",
            success=successful_calculations > 0,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            throughput=throughput,
            response_time_ms=avg_calc_time * 1000,
            details={
                'total_polymers': len(test_polymers),
                'successful_calculations': successful_calculations,
                'success_rate': successful_calculations / len(test_polymers) * 100,
                'avg_calculation_time_ms': avg_calc_time * 1000,
                'property_results': property_results[:3]  # First 3 results as sample
            }
        )

    def test_prediction_pipeline_performance(self) -> ChemistryTestResult:
        """Test full prediction pipeline performance"""
        self.log("Testing prediction pipeline performance...")

        try:
            from app import predict_polymer_properties
        except ImportError as e:
            return ChemistryTestResult(
                test_name="Prediction Pipeline",
                success=False,
                duration=0,
                memory_mb=0,
                cpu_percent=0,
                error=f"Cannot import predict_polymer_properties: {e}"
            )

        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Test polymers and properties
        test_polymers = [
            "CC",  # PE
            "CC(C)",  # PP
            "CC(c1ccccc1)",  # PS
        ]

        test_properties = [
            ["Glass Transition Temperature (Tg)"],
            ["Melting Temperature (Tm)"],
            ["Density"],
            ["Glass Transition Temperature (Tg)", "Melting Temperature (Tm)"],
            ["Glass Transition Temperature (Tg)", "Melting Temperature (Tm)", "Density"],
        ]

        prediction_results = []
        prediction_times = []
        successful_predictions = 0

        for smiles in test_polymers:
            for properties in test_properties:
                pred_start = time.time()

                try:
                    predictions = predict_polymer_properties(smiles, properties)
                    pred_time = time.time() - pred_start

                    if 'error' not in predictions:
                        successful_predictions += 1
                        prediction_results.append({
                            'smiles': smiles,
                            'properties_requested': properties,
                            'predictions': {prop: result['value'] for prop, result in predictions.items()},
                            'time_ms': pred_time * 1000
                        })
                    else:
                        prediction_results.append({
                            'smiles': smiles,
                            'properties_requested': properties,
                            'error': predictions['error'],
                            'time_ms': pred_time * 1000
                        })

                    prediction_times.append(pred_time)

                except Exception as e:
                    prediction_results.append({
                        'smiles': smiles,
                        'properties_requested': properties,
                        'exception': str(e),
                        'time_ms': (time.time() - pred_start) * 1000
                    })

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        # Calculate performance metrics
        total_tests = len(test_polymers) * len(test_properties)
        avg_pred_time = sum(prediction_times) / len(prediction_times) if prediction_times else 0
        throughput = total_tests / duration if duration > 0 else 0

        return ChemistryTestResult(
            test_name="Prediction Pipeline",
            success=successful_predictions > 0,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            throughput=throughput,
            response_time_ms=avg_pred_time * 1000,
            details={
                'total_tests': total_tests,
                'successful_predictions': successful_predictions,
                'success_rate': successful_predictions / total_tests * 100,
                'avg_prediction_time_ms': avg_pred_time * 1000,
                'prediction_results': prediction_results[:3]  # First 3 results as sample
            }
        )

    def run_all_tests(self) -> List[ChemistryTestResult]:
        """Run all chemistry stack performance tests"""
        self.log("Starting chemistry stack performance analysis...")
        self.log(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")

        tests = [
            self.test_app_imports,
            self.test_smiles_validation_performance,
            self.test_molecular_properties_performance,
            self.test_prediction_pipeline_performance,
        ]

        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.results.append(result)

                status = "PASS" if result.success else "FAIL"
                self.log(f"{result.test_name}: {status} ({result.duration:.2f}s, {result.memory_mb:.1f}MB)")

                if result.error:
                    self.log(f"  Error: {result.error}")

                if result.throughput:
                    self.log(f"  Throughput: {result.throughput:.2f} items/sec")

                if result.response_time_ms:
                    self.log(f"  Avg Response Time: {result.response_time_ms:.1f}ms")

            except Exception as e:
                self.log(f"Test {test_func.__name__} failed: {e}")
                results.append(ChemistryTestResult(
                    test_name=test_func.__name__,
                    success=False,
                    duration=0,
                    memory_mb=0,
                    cpu_percent=0,
                    error=str(e)
                ))

        return results

    def generate_performance_report(self, results: List[ChemistryTestResult]) -> str:
        """Generate chemistry stack performance report"""
        report = []
        report.append("=" * 80)
        report.append("POLYID CHEMISTRY STACK PERFORMANCE ANALYSIS")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
        report.append("")

        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        total_duration = sum(r.duration for r in results)
        total_memory = sum(r.memory_mb for r in results)

        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Tests Run: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {total_tests - passed_tests}")
        report.append(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        report.append(f"Total Duration: {total_duration:.2f}s")
        report.append(f"Total Memory Impact: {total_memory:.1f}MB")
        report.append("")

        # Performance Summary
        response_times = [r.response_time_ms for r in results if r.response_time_ms]
        throughputs = [r.throughput for r in results if r.throughput]

        if response_times:
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            report.append(f"Average Response Time: {sum(response_times)/len(response_times):.1f}ms")
            report.append(f"Fastest Response: {min(response_times):.1f}ms")
            report.append(f"Slowest Response: {max(response_times):.1f}ms")

        if throughputs:
            report.append(f"Average Throughput: {sum(throughputs)/len(throughputs):.1f} items/sec")
            report.append(f"Peak Throughput: {max(throughputs):.1f} items/sec")
        report.append("")

        # Detailed Results
        report.append("DETAILED RESULTS")
        report.append("-" * 80)

        for result in results:
            status = "PASS" if result.success else "FAIL"
            report.append(f"\n{result.test_name}: {status}")
            report.append(f"  Duration: {result.duration:.3f}s")
            report.append(f"  Memory: {result.memory_mb:.1f}MB")
            report.append(f"  CPU: {result.cpu_percent:.1f}%")

            if result.throughput:
                report.append(f"  Throughput: {result.throughput:.2f} items/sec")

            if result.response_time_ms:
                report.append(f"  Response Time: {result.response_time_ms:.1f}ms")

            if result.error:
                report.append(f"  Error: {result.error}")

            if result.details:
                report.append("  Key Metrics:")
                for key, value in result.details.items():
                    if key.endswith('_rate') or key.endswith('_percent'):
                        report.append(f"    {key}: {value:.1f}%")
                    elif key.endswith('_time_ms') or key.endswith('_ms'):
                        report.append(f"    {key}: {value:.1f}ms")
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        if isinstance(value, float):
                            report.append(f"    {key}: {value:.3f}")
                        else:
                            report.append(f"    {key}: {value}")

        # Performance Assessment
        report.append("\n\nCHEMISTRY STACK ASSESSMENT")
        report.append("-" * 80)

        # Identify bottlenecks
        bottlenecks = []
        recommendations = []

        # Check import performance
        import_test = next((r for r in results if r.test_name == "App Imports"), None)
        if import_test and not import_test.success:
            bottlenecks.append("App import failures - dependency issues")
            recommendations.append("Verify all chemistry dependencies are properly installed")
        elif import_test and import_test.details:
            missing_deps = [k for k, v in import_test.details.items() if k.endswith('_available') and not v]
            if missing_deps:
                bottlenecks.append(f"Missing dependencies: {missing_deps}")
                recommendations.append("Install missing chemistry stack components")

        # Check response times
        slow_tests = [r for r in results if r.response_time_ms and r.response_time_ms > 500]
        if slow_tests:
            bottlenecks.append("Slow response times (>500ms)")
            recommendations.append("Optimize molecular processing pipeline")

        # Check throughput
        low_throughput_tests = [r for r in results if r.throughput and r.throughput < 5]
        if low_throughput_tests:
            bottlenecks.append("Low throughput (<5 items/sec)")
            recommendations.append("Implement batch processing and caching")

        # Check memory usage
        high_memory_tests = [r for r in results if r.memory_mb > 100]
        if high_memory_tests:
            bottlenecks.append("High memory usage (>100MB)")
            recommendations.append("Optimize memory allocation and cleanup")

        if bottlenecks:
            report.append("IDENTIFIED BOTTLENECKS:")
            for bottleneck in bottlenecks:
                report.append(f"  - {bottleneck}")
        else:
            report.append("No critical bottlenecks identified.")

        report.append("")

        if recommendations:
            report.append("OPTIMIZATION RECOMMENDATIONS:")
            for rec in recommendations:
                report.append(f"  - {rec}")
        else:
            report.append("Chemistry stack performance is optimal.")

        # Deployment recommendations
        report.append("")
        report.append("HUGGING FACE SPACES DEPLOYMENT RECOMMENDATIONS:")
        report.append("  - Use Standard GPU Spaces for full chemistry stack compatibility")
        report.append("  - Implement model and preprocessor caching")
        report.append("  - Add request batching for concurrent users")
        report.append("  - Monitor memory usage with alerts")
        report.append("  - Implement graceful error handling for invalid SMILES")
        report.append("  - Use lazy loading for large models")
        report.append("  - Set up health checks and monitoring")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

def main():
    """Main function"""
    try:
        tester = ChemistryStackTester()
        results = tester.run_all_tests()

        # Generate and display report
        report = tester.generate_performance_report(results)
        print("\n" + report)

        # Return success status
        passed_tests = sum(1 for r in results if r.success)
        return passed_tests > len(results) // 2  # Success if > 50% pass

    except Exception as e:
        print(f"Chemistry stack performance test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
