#!/usr/bin/env python3
"""
Comprehensive PolyID Performance Analysis
Focused testing without Unicode issues for Windows compatibility
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
class TestResult:
    """Simple test result container"""
    name: str
    success: bool
    duration: float
    memory_mb: float
    cpu_percent: float
    throughput: Optional[float] = None
    error: Optional[str] = None
    details: Dict = None

class PolyIDPerformanceTest:
    """Simplified performance testing focused on core metrics"""

    def __init__(self):
        self.results = []
        self.start_memory = 0
        self.process = psutil.Process()

    def log(self, message):
        """Simple logging without Unicode issues"""
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_percent(self):
        """Get current CPU usage"""
        return self.process.cpu_percent()

    def test_imports(self) -> TestResult:
        """Test dependency imports and availability"""
        self.log("Testing imports and dependencies...")

        start_time = time.time()
        start_memory = self.get_memory_usage()

        dependencies = {}
        errors = []

        # Test TensorFlow
        try:
            import tensorflow as tf
            dependencies['tensorflow'] = tf.__version__
            # Test GPU availability
            gpu_devices = tf.config.list_physical_devices('GPU')
            dependencies['gpu_devices'] = len(gpu_devices)
            dependencies['gpu_available'] = len(gpu_devices) > 0
        except ImportError as e:
            dependencies['tensorflow'] = False
            errors.append(f"TensorFlow: {e}")

        # Test RDKit
        try:
            import rdkit
            from rdkit import Chem
            dependencies['rdkit'] = True
            # Test basic functionality
            mol = Chem.MolFromSmiles("CC")
            dependencies['rdkit_functional'] = mol is not None
        except ImportError as e:
            dependencies['rdkit'] = False
            errors.append(f"RDKit: {e}")

        # Test NFP
        try:
            import nfp
            dependencies['nfp'] = True
        except ImportError as e:
            dependencies['nfp'] = False
            errors.append(f"NFP: {e}")

        # Test PolyID
        try:
            from polyid.polyid import SingleModel, MultiModel
            from polyid.parameters import Parameters
            dependencies['polyid'] = True
        except ImportError as e:
            dependencies['polyid'] = False
            errors.append(f"PolyID: {e}")

        # Test Gradio
        try:
            import gradio as gr
            dependencies['gradio'] = gr.__version__
        except ImportError as e:
            dependencies['gradio'] = False
            errors.append(f"Gradio: {e}")

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        success = dependencies.get('rdkit', False) and dependencies.get('tensorflow', False)

        return TestResult(
            name="Import Test",
            success=success,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            error="; ".join(errors) if errors else None,
            details={
                'dependencies': dependencies,
                'critical_missing': not success
            }
        )

    def test_rdkit_performance(self) -> TestResult:
        """Test RDKit molecular processing performance"""
        self.log("Testing RDKit performance...")

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
        except ImportError:
            return TestResult(
                name="RDKit Performance",
                success=False,
                duration=0,
                memory_mb=0,
                cpu_percent=0,
                error="RDKit not available"
            )

        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Test polymer SMILES
        test_smiles = [
            "CC",  # Polyethylene
            "CC(C)",  # Polypropylene
            "CC(c1ccccc1)",  # Polystyrene
            "CC(C)(C(=O)OC)",  # PMMA
            "COC(=O)c1ccc(C(=O)O)cc1",  # PET
        ]

        results = []
        processing_times = []

        for smiles in test_smiles:
            mol_start = time.time()
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Calculate descriptors
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    atoms = mol.GetNumAtoms()
                    bonds = mol.GetNumBonds()
                    rings = rdMolDescriptors.CalcNumRings(mol)

                    results.append({
                        'smiles': smiles,
                        'mw': mw,
                        'logp': logp,
                        'tpsa': tpsa,
                        'atoms': atoms,
                        'bonds': bonds,
                        'rings': rings
                    })

                processing_time = time.time() - mol_start
                processing_times.append(processing_time)

            except Exception as e:
                self.log(f"Error processing {smiles}: {e}")

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        # Calculate throughput
        throughput = len(results) / duration if duration > 0 else 0

        # Performance metrics
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        return TestResult(
            name="RDKit Performance",
            success=len(results) > 0,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            throughput=throughput,
            details={
                'molecules_processed': len(results),
                'avg_processing_time_ms': avg_processing_time * 1000,
                'throughput_per_sec': throughput,
                'results_sample': results[:2]  # First 2 results as sample
            }
        )

    def test_tensorflow_performance(self) -> TestResult:
        """Test TensorFlow performance and GPU utilization"""
        self.log("Testing TensorFlow performance...")

        try:
            import tensorflow as tf
        except ImportError:
            return TestResult(
                name="TensorFlow Performance",
                success=False,
                duration=0,
                memory_mb=0,
                cpu_percent=0,
                error="TensorFlow not available"
            )

        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Test basic TensorFlow operations
        test_results = {}

        try:
            # Test tensor operations
            a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
            b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

            # Matrix multiplication
            c = tf.matmul(a, b)
            test_results['basic_ops'] = True
            test_results['matmul_result'] = c.numpy().tolist()

            # Test GPU availability
            gpu_devices = tf.config.list_physical_devices('GPU')
            test_results['gpu_devices'] = len(gpu_devices)
            test_results['gpu_available'] = len(gpu_devices) > 0

            if gpu_devices:
                # Test GPU operations
                try:
                    with tf.device('/GPU:0'):
                        gpu_a = tf.random.normal([100, 100])
                        gpu_b = tf.random.normal([100, 100])
                        gpu_c = tf.matmul(gpu_a, gpu_b)
                    test_results['gpu_ops'] = True
                except Exception as e:
                    test_results['gpu_ops'] = False
                    test_results['gpu_error'] = str(e)

            # Test model creation
            try:
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                    tf.keras.layers.Dense(1)
                ])
                test_results['model_creation'] = True
                test_results['model_params'] = model.count_params()
            except Exception as e:
                test_results['model_creation'] = False
                test_results['model_error'] = str(e)

        except Exception as e:
            test_results['error'] = str(e)

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        success = test_results.get('basic_ops', False)

        return TestResult(
            name="TensorFlow Performance",
            success=success,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            details=test_results
        )

    def test_app_simulation(self) -> TestResult:
        """Test PolyID app simulation performance"""
        self.log("Testing app simulation...")

        start_time = time.time()
        start_memory = self.get_memory_usage()

        try:
            # Import app functions
            from app import validate_smiles, calculate_molecular_properties, predict_polymer_properties

            test_polymers = [
                "CC",  # PE
                "CC(C)",  # PP
                "CC(c1ccccc1)",  # PS
            ]

            results = []
            response_times = []

            for smiles in test_polymers:
                request_start = time.time()

                try:
                    # Simulate full app workflow
                    validation = validate_smiles(smiles)
                    if validation[0]:  # If valid
                        mol_props = calculate_molecular_properties(smiles)
                        predictions = predict_polymer_properties(
                            smiles,
                            ["Glass Transition Temperature (Tg)", "Melting Temperature (Tm)"]
                        )

                        results.append({
                            'smiles': smiles,
                            'valid': True,
                            'mol_props': mol_props if 'error' not in mol_props else None,
                            'predictions': predictions if 'error' not in predictions else None
                        })
                    else:
                        results.append({
                            'smiles': smiles,
                            'valid': False,
                            'error': validation[1]
                        })

                except Exception as e:
                    results.append({
                        'smiles': smiles,
                        'error': str(e)
                    })

                response_time = time.time() - request_start
                response_times.append(response_time)

        except ImportError as e:
            return TestResult(
                name="App Simulation",
                success=False,
                duration=0,
                memory_mb=0,
                cpu_percent=0,
                error=f"Cannot import app functions: {e}"
            )

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        # Calculate metrics
        successful_requests = sum(1 for r in results if r.get('valid', False))
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        throughput = len(results) / duration if duration > 0 else 0

        return TestResult(
            name="App Simulation",
            success=successful_requests > 0,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            throughput=throughput,
            details={
                'total_requests': len(results),
                'successful_requests': successful_requests,
                'success_rate': successful_requests / len(results) * 100 if results else 0,
                'avg_response_time_ms': avg_response_time * 1000,
                'throughput_per_sec': throughput,
                'results_sample': results[:2]
            }
        )

    def test_memory_stress(self) -> TestResult:
        """Test memory usage under stress conditions"""
        self.log("Testing memory stress...")

        start_time = time.time()
        start_memory = self.get_memory_usage()

        memory_snapshots = [start_memory]

        try:
            # Test 1: Large data structures
            large_lists = []
            for i in range(10):
                large_lists.append([j for j in range(100000)])
                memory_snapshots.append(self.get_memory_usage())

            # Test 2: Repeated operations
            if True:  # RDKit available check
                try:
                    from rdkit import Chem
                    from rdkit.Chem import Descriptors

                    for i in range(100):
                        mol = Chem.MolFromSmiles("CC(C)(C(=O)OC)")
                        if mol:
                            mw = Descriptors.MolWt(mol)
                        if i % 20 == 0:
                            memory_snapshots.append(self.get_memory_usage())
                except ImportError:
                    pass

            # Test 3: Memory cleanup
            del large_lists
            import gc
            gc.collect()
            memory_snapshots.append(self.get_memory_usage())

        except Exception as e:
            return TestResult(
                name="Memory Stress",
                success=False,
                duration=time.time() - start_time,
                memory_mb=self.get_memory_usage() - start_memory,
                cpu_percent=self.get_cpu_percent(),
                error=str(e)
            )

        duration = time.time() - start_time
        final_memory = self.get_memory_usage()
        memory_usage = final_memory - start_memory

        # Memory analysis
        peak_memory = max(memory_snapshots)
        memory_growth = peak_memory - start_memory
        memory_recovered = peak_memory - final_memory
        recovery_rate = (memory_recovered / memory_growth * 100) if memory_growth > 0 else 100

        return TestResult(
            name="Memory Stress",
            success=True,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            details={
                'peak_memory_mb': peak_memory,
                'memory_growth_mb': memory_growth,
                'memory_recovered_mb': memory_recovered,
                'recovery_rate_percent': recovery_rate,
                'memory_snapshots': memory_snapshots[:10]  # First 10 snapshots
            }
        )

    def test_concurrent_load(self) -> TestResult:
        """Test concurrent processing capabilities"""
        self.log("Testing concurrent load...")

        import threading
        import concurrent.futures

        start_time = time.time()
        start_memory = self.get_memory_usage()

        def worker_task(task_id):
            """Simulate a worker task"""
            task_start = time.time()
            results = []

            try:
                # Simulate processing work
                for i in range(10):
                    # Simple computation
                    result = sum(j ** 2 for j in range(100))
                    results.append(result)

                # If RDKit available, do molecular work
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles("CC")
                    if mol:
                        results.append(mol.GetNumAtoms())
                except ImportError:
                    pass

                task_duration = time.time() - task_start
                return {
                    'task_id': task_id,
                    'duration': task_duration,
                    'results_count': len(results),
                    'success': True
                }

            except Exception as e:
                return {
                    'task_id': task_id,
                    'duration': time.time() - task_start,
                    'success': False,
                    'error': str(e)
                }

        # Run concurrent tasks
        num_workers = 4
        task_results = []

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_task, i) for i in range(num_workers * 2)]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        task_results.append(result)
                    except Exception as e:
                        task_results.append({
                            'success': False,
                            'error': str(e)
                        })

        except Exception as e:
            return TestResult(
                name="Concurrent Load",
                success=False,
                duration=time.time() - start_time,
                memory_mb=self.get_memory_usage() - start_memory,
                cpu_percent=self.get_cpu_percent(),
                error=str(e)
            )

        duration = time.time() - start_time
        memory_usage = self.get_memory_usage() - start_memory

        # Analyze results
        successful_tasks = sum(1 for r in task_results if r.get('success', False))
        total_tasks = len(task_results)
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        throughput = successful_tasks / duration if duration > 0 else 0

        avg_task_duration = sum(r.get('duration', 0) for r in task_results if r.get('success', False)) / successful_tasks if successful_tasks > 0 else 0

        return TestResult(
            name="Concurrent Load",
            success=successful_tasks > 0,
            duration=duration,
            memory_mb=memory_usage,
            cpu_percent=self.get_cpu_percent(),
            throughput=throughput,
            details={
                'total_tasks': total_tasks,
                'successful_tasks': successful_tasks,
                'success_rate': success_rate,
                'avg_task_duration_ms': avg_task_duration * 1000,
                'throughput_per_sec': throughput,
                'workers': num_workers
            }
        )

    def run_all_tests(self) -> List[TestResult]:
        """Run all performance tests"""
        self.log("Starting comprehensive PolyID performance analysis...")
        self.log(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")

        tests = [
            self.test_imports,
            self.test_rdkit_performance,
            self.test_tensorflow_performance,
            self.test_app_simulation,
            self.test_memory_stress,
            self.test_concurrent_load
        ]

        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.results.append(result)

                status = "PASS" if result.success else "FAIL"
                self.log(f"{result.name}: {status} ({result.duration:.2f}s, {result.memory_mb:.1f}MB)")

                if result.error:
                    self.log(f"  Error: {result.error}")

                if result.throughput:
                    self.log(f"  Throughput: {result.throughput:.2f} items/sec")

            except Exception as e:
                self.log(f"Test {test_func.__name__} failed: {e}")
                results.append(TestResult(
                    name=test_func.__name__,
                    success=False,
                    duration=0,
                    memory_mb=0,
                    cpu_percent=0,
                    error=str(e)
                ))

        return results

    def generate_report(self, results: List[TestResult]) -> str:
        """Generate performance analysis report"""
        report = []
        report.append("=" * 80)
        report.append("POLYID PERFORMANCE ANALYSIS REPORT")
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

        # Detailed Results
        report.append("DETAILED RESULTS")
        report.append("-" * 80)

        for result in results:
            status = "PASS" if result.success else "FAIL"
            report.append(f"\n{result.name}: {status}")
            report.append(f"  Duration: {result.duration:.3f}s")
            report.append(f"  Memory: {result.memory_mb:.1f}MB")
            report.append(f"  CPU: {result.cpu_percent:.1f}%")

            if result.throughput:
                report.append(f"  Throughput: {result.throughput:.2f} items/sec")

            if result.error:
                report.append(f"  Error: {result.error}")

            if result.details:
                report.append("  Details:")
                for key, value in result.details.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            report.append(f"    {key}: {value:.3f}")
                        else:
                            report.append(f"    {key}: {value}")
                    elif isinstance(value, bool):
                        report.append(f"    {key}: {value}")
                    elif isinstance(value, str) and len(value) < 100:
                        report.append(f"    {key}: {value}")
                    elif isinstance(value, (list, dict)) and len(str(value)) < 200:
                        report.append(f"    {key}: {value}")

        # Performance Assessment
        report.append("\n\nPERFORMANCE ASSESSMENT")
        report.append("-" * 80)

        # Identify issues and recommendations
        issues = []
        recommendations = []

        # Check import status
        import_test = next((r for r in results if r.name == "Import Test"), None)
        if import_test and not import_test.success:
            issues.append("Critical dependencies missing")
            recommendations.append("Install missing dependencies (RDKit, TensorFlow, PolyID)")

        # Check memory usage
        memory_test = next((r for r in results if r.name == "Memory Stress"), None)
        if memory_test and memory_test.details:
            recovery_rate = memory_test.details.get('recovery_rate_percent', 0)
            if recovery_rate < 50:
                issues.append("Poor memory recovery")
                recommendations.append("Implement explicit garbage collection")

        # Check app performance
        app_test = next((r for r in results if r.name == "App Simulation"), None)
        if app_test and app_test.details:
            avg_response = app_test.details.get('avg_response_time_ms', 0)
            if avg_response > 1000:  # > 1 second
                issues.append("High response times")
                recommendations.append("Optimize prediction pipeline")

        # Check concurrent performance
        concurrent_test = next((r for r in results if r.name == "Concurrent Load"), None)
        if concurrent_test and concurrent_test.details:
            success_rate = concurrent_test.details.get('success_rate', 0)
            if success_rate < 90:
                issues.append("Concurrent processing issues")
                recommendations.append("Improve thread safety and resource management")

        if issues:
            report.append("IDENTIFIED ISSUES:")
            for issue in issues:
                report.append(f"  - {issue}")
        else:
            report.append("No critical performance issues identified.")

        report.append("")

        if recommendations:
            report.append("RECOMMENDATIONS:")
            for rec in recommendations:
                report.append(f"  - {rec}")
        else:
            report.append("Performance appears optimal for current configuration.")

        # Additional recommendations
        report.append("")
        report.append("DEPLOYMENT RECOMMENDATIONS:")
        report.append("  - Monitor memory usage in production")
        report.append("  - Implement caching for repeated operations")
        report.append("  - Use TensorFlow optimization for inference")
        report.append("  - Consider request batching for high load")
        report.append("  - Set up monitoring and alerting")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def save_results(self, results: List[TestResult], output_dir: str = "."):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = int(time.time())

        # Save JSON data
        json_data = {
            'timestamp': timestamp,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'python_version': sys.version
            },
            'results': [
                {
                    'name': r.name,
                    'success': r.success,
                    'duration': r.duration,
                    'memory_mb': r.memory_mb,
                    'cpu_percent': r.cpu_percent,
                    'throughput': r.throughput,
                    'error': r.error,
                    'details': r.details
                }
                for r in results
            ]
        }

        json_file = output_path / f"polyid_performance_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Save text report
        report = self.generate_report(results)
        report_file = output_path / f"polyid_performance_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        self.log(f"Results saved to {json_file}")
        self.log(f"Report saved to {report_file}")

        return str(report_file), str(json_file)

def main():
    """Main function"""
    try:
        tester = PolyIDPerformanceTest()
        results = tester.run_all_tests()

        # Generate and display report
        report = tester.generate_report(results)
        print("\n" + report)

        # Save results
        report_file, json_file = tester.save_results(results)

        # Return success status
        passed_tests = sum(1 for r in results if r.success)
        return passed_tests > len(results) // 2  # Success if > 50% pass

    except Exception as e:
        print(f"Performance test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)