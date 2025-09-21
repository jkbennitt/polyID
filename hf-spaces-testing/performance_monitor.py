#!/usr/bin/env python3
"""
PolyID Hugging Face Spaces Performance Monitor

Comprehensive performance monitoring and analysis framework for the PolyID deployment
on Hugging Face Spaces. This tool provides detailed insights into:

1. Memory Usage: Monitor memory consumption during model loading and predictions
2. GPU Utilization: Assess TensorFlow GPU usage effectiveness
3. Response Times: Measure prediction latency and interface responsiveness
4. Throughput: Test concurrent usage capabilities and batch processing
5. Resource Bottlenecks: Identify performance limitations and optimization opportunities
6. Caching Efficiency: Evaluate preprocessing and model caching effectiveness
7. Scalability: Assess performance under different load conditions
8. Cold Start Times: Monitor initial loading and warm-up performance

Usage:
    python performance_monitor.py --mode [full|quick|gpu|memory|throughput]

    Modes:
    - full: Complete performance analysis (default)
    - quick: Basic performance metrics
    - gpu: GPU-focused analysis
    - memory: Memory usage profiling
    - throughput: Concurrent load testing
"""

import os
import sys
import time
import psutil
import threading
import queue
import warnings
import traceback
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import dependencies with availability checking
try:
    import tensorflow as tf
    TF_AVAILABLE = True

    # Configure TensorFlow for monitoring
    tf.config.run_functions_eagerly(True)  # For detailed profiling

    # Set up GPU memory growth to prevent allocation issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"GPU memory growth setup failed: {e}")

except ImportError:
    TF_AVAILABLE = False
    tf = None

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    rdkit = None

try:
    import nfp
    NFP_AVAILABLE = True
except ImportError:
    NFP_AVAILABLE = False
    nfp = None

try:
    from polyid.polyid import SingleModel, MultiModel
    from polyid.parameters import Parameters
    from polyid.preprocessors.preprocessors import PolymerPreprocessor
    from polyid.preprocessors.features import atom_features_v1, bond_features_v1
    POLYID_AVAILABLE = True
except ImportError:
    POLYID_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics"""
    timestamp: float
    test_name: str
    duration: float
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    throughput: Optional[float] = None
    success_rate: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemInfo:
    """System information and capabilities"""
    python_version: str
    cpu_count: int
    memory_total_gb: float
    tensorflow_version: Optional[str]
    gpu_available: bool
    gpu_devices: List[str]
    dependencies: Dict[str, bool]

class ResourceMonitor:
    """Context manager for monitoring system resources during operations"""

    def __init__(self, test_name: str = "Unknown"):
        self.test_name = test_name
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.peak_memory = None
        self.measurements = []

    def __enter__(self):
        self.start_time = time.time()
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = process.cpu_percent()
        self.peak_memory = self.start_memory

        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._monitoring = False
        self._monitor_thread.join()

    def _monitor_resources(self):
        """Monitor resources in background thread"""
        while self._monitoring:
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024
                current_cpu = process.cpu_percent()

                self.peak_memory = max(self.peak_memory, current_memory)

                measurement = {
                    'timestamp': time.time() - self.start_time,
                    'memory_mb': current_memory,
                    'cpu_percent': current_cpu
                }

                # Add GPU metrics if available
                if TF_AVAILABLE and tf.config.list_physical_devices('GPU'):
                    try:
                        gpu_details = tf.config.experimental.get_device_details(
                            tf.config.list_physical_devices('GPU')[0]
                        )
                        # Note: Getting real-time GPU utilization requires nvidia-ml-py
                        # For now, we'll track TensorFlow GPU memory allocation
                        measurement['gpu_available'] = True
                    except Exception as e:
                        measurement['gpu_available'] = False

                self.measurements.append(measurement)
                time.sleep(0.1)  # Sample every 100ms

            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break

    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics from monitoring session"""
        end_time = time.time()
        duration = end_time - self.start_time

        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_delta = end_memory - self.start_memory

        # Calculate average CPU usage
        if self.measurements:
            avg_cpu = np.mean([m['cpu_percent'] for m in self.measurements])
        else:
            avg_cpu = process.cpu_percent()

        # GPU metrics
        gpu_utilization = None
        gpu_memory_used = None
        gpu_memory_total = None

        if TF_AVAILABLE and tf.config.list_physical_devices('GPU'):
            try:
                # Get TensorFlow GPU memory info
                gpu_details = tf.config.experimental.get_device_details(
                    tf.config.list_physical_devices('GPU')[0]
                )
                # Note: This is limited without nvidia-ml-py for real GPU utilization
                gpu_utilization = 0.0  # Placeholder

            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")

        return PerformanceMetrics(
            timestamp=end_time,
            test_name=self.test_name,
            duration=duration,
            memory_usage_mb=memory_delta,
            memory_peak_mb=self.peak_memory,
            cpu_percent=avg_cpu,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            details={'measurements': self.measurements}
        )

class PolyIDPerformanceAnalyzer:
    """Comprehensive performance analyzer for PolyID deployment"""

    def __init__(self):
        self.system_info = self._gather_system_info()
        self.metrics: List[PerformanceMetrics] = []
        self.test_polymers = self._prepare_test_data()

    def _gather_system_info(self) -> SystemInfo:
        """Gather system information and capabilities"""

        # GPU information
        gpu_available = False
        gpu_devices = []
        tensorflow_version = None

        if TF_AVAILABLE:
            tensorflow_version = tf.__version__
            gpu_devices = [device.name for device in tf.config.list_physical_devices('GPU')]
            gpu_available = len(gpu_devices) > 0

        return SystemInfo(
            python_version=sys.version,
            cpu_count=psutil.cpu_count(),
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            tensorflow_version=tensorflow_version,
            gpu_available=gpu_available,
            gpu_devices=gpu_devices,
            dependencies={
                'tensorflow': TF_AVAILABLE,
                'rdkit': RDKIT_AVAILABLE,
                'nfp': NFP_AVAILABLE,
                'polyid': POLYID_AVAILABLE,
                'gradio': GRADIO_AVAILABLE
            }
        )

    def _prepare_test_data(self) -> Dict[str, str]:
        """Prepare test polymer dataset for performance testing"""
        return {
            # Simple polymers for baseline testing
            "polyethylene": "CC",
            "polypropylene": "CC(C)",
            "polystyrene": "CC(c1ccccc1)",

            # Medium complexity polymers
            "pmma": "CC(C)(C(=O)OC)",
            "pla": "CC(O)C(=O)",
            "pvc": "CC(Cl)",

            # Complex polymers for stress testing
            "pet": "COC(=O)c1ccc(C(=O)O)cc1.OCCO",
            "polycarbonate": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1.O=C(Cl)Cl",
            "polyimide": "c1ccc2c(c1)C(=O)N(c1ccc(C(=O)c3ccc(N4C(=O)c5ccccc5C4=O)cc3)cc1)C2=O",

            # Stress test polymers
            "complex_branched": "CC(C)(CC(C)(C)c1ccccc1)c1ccc(C(C)(C)CC(C)(C)C)cc1",
            "fluorinated": "C(C(F)(F)F)(C(F)(F)F)C(F)(F)F",
        }

    def test_cold_start_performance(self) -> PerformanceMetrics:
        """Test initial loading and warm-up performance"""
        logger.info("Testing cold start performance...")

        with ResourceMonitor("Cold Start Performance") as monitor:
            try:
                # Simulate cold start by importing and initializing components
                if RDKIT_AVAILABLE:
                    # Test RDKit initialization
                    mol = Chem.MolFromSmiles("CC")
                    if mol:
                        mw = Descriptors.MolWt(mol)

                if TF_AVAILABLE:
                    # Test TensorFlow initialization
                    tf.constant([1, 2, 3, 4])

                    # Test GPU availability
                    if tf.config.list_physical_devices('GPU'):
                        with tf.device('/GPU:0'):
                            tf.constant([1.0, 2.0, 3.0])

                if POLYID_AVAILABLE:
                    # Test PolyID component initialization
                    params = Parameters()

                    # Test preprocessor initialization
                    preprocessor = PolymerPreprocessor(
                        atom_features=atom_features_v1,
                        bond_features=bond_features_v1,
                        explicit_hs=False
                    )

                success = True
                details = {
                    "components_loaded": {
                        "rdkit": RDKIT_AVAILABLE,
                        "tensorflow": TF_AVAILABLE,
                        "polyid": POLYID_AVAILABLE
                    },
                    "gpu_initialization": TF_AVAILABLE and bool(tf.config.list_physical_devices('GPU'))
                }

            except Exception as e:
                success = False
                details = {"error": str(e), "traceback": traceback.format_exc()}

        metrics = monitor.get_metrics()
        metrics.success_rate = 100.0 if success else 0.0
        metrics.details.update(details)

        return metrics

    def test_memory_usage_patterns(self) -> PerformanceMetrics:
        """Test memory usage patterns during different operations"""
        logger.info("Testing memory usage patterns...")

        with ResourceMonitor("Memory Usage Patterns") as monitor:
            try:
                memory_snapshots = []

                # Baseline memory
                process = psutil.Process()
                baseline_memory = process.memory_info().rss / 1024 / 1024
                memory_snapshots.append(("baseline", baseline_memory))

                # Test SMILES processing memory usage
                if RDKIT_AVAILABLE:
                    molecules = []
                    for name, smiles in self.test_polymers.items():
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            molecules.append(mol)

                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(("rdkit_processing", current_memory))

                # Test batch processing memory usage
                if RDKIT_AVAILABLE:
                    batch_results = []
                    for i in range(10):  # Process 10 batches
                        batch_mols = []
                        for name, smiles in self.test_polymers.items():
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                # Calculate descriptors
                                descriptors = {
                                    'mw': Descriptors.MolWt(mol),
                                    'logp': Descriptors.MolLogP(mol),
                                    'tpsa': Descriptors.TPSA(mol)
                                }
                                batch_mols.append(descriptors)
                        batch_results.append(batch_mols)

                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(("batch_processing", current_memory))

                # Test TensorFlow memory usage
                if TF_AVAILABLE:
                    # Create and manipulate tensors
                    tensors = []
                    for i in range(5):
                        tensor = tf.random.normal([1000, 1000])
                        result = tf.matmul(tensor, tensor)
                        tensors.append(result)

                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(("tensorflow_operations", current_memory))

                    # Clear tensors
                    del tensors
                    if hasattr(tf, 'keras'):
                        tf.keras.backend.clear_session()

                # Final memory check
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_snapshots.append(("final", final_memory))

                # Calculate memory efficiency metrics
                peak_memory = max(snapshot[1] for snapshot in memory_snapshots)
                memory_overhead = peak_memory - baseline_memory
                memory_recovery = peak_memory - final_memory

                details = {
                    "memory_snapshots": memory_snapshots,
                    "baseline_memory_mb": baseline_memory,
                    "peak_memory_mb": peak_memory,
                    "memory_overhead_mb": memory_overhead,
                    "memory_recovery_mb": memory_recovery,
                    "memory_efficiency": (memory_recovery / memory_overhead * 100) if memory_overhead > 0 else 100.0
                }

                success = True

            except Exception as e:
                success = False
                details = {"error": str(e), "traceback": traceback.format_exc()}

        metrics = monitor.get_metrics()
        metrics.success_rate = 100.0 if success else 0.0
        metrics.details.update(details)

        return metrics

    def test_gpu_utilization(self) -> PerformanceMetrics:
        """Test GPU utilization and TensorFlow performance"""
        logger.info("Testing GPU utilization...")

        with ResourceMonitor("GPU Utilization") as monitor:
            try:
                gpu_tests = {}

                if not TF_AVAILABLE:
                    success = False
                    details = {"error": "TensorFlow not available"}

                elif not tf.config.list_physical_devices('GPU'):
                    success = False
                    details = {"error": "No GPU devices available"}

                else:
                    # Test GPU device availability
                    gpu_devices = tf.config.list_physical_devices('GPU')
                    gpu_tests["devices_available"] = len(gpu_devices)

                    # Test GPU memory allocation
                    try:
                        with tf.device('/GPU:0'):
                            # Create tensors on GPU
                            a = tf.random.normal([1000, 1000])
                            b = tf.random.normal([1000, 1000])

                            # Perform matrix operations
                            start_time = time.time()
                            c = tf.matmul(a, b)
                            gpu_compute_time = time.time() - start_time

                            gpu_tests["gpu_compute_time"] = gpu_compute_time
                            gpu_tests["gpu_allocation_success"] = True

                    except Exception as e:
                        gpu_tests["gpu_allocation_success"] = False
                        gpu_tests["gpu_error"] = str(e)

                    # Compare CPU vs GPU performance
                    try:
                        # CPU computation
                        with tf.device('/CPU:0'):
                            a_cpu = tf.random.normal([500, 500])
                            b_cpu = tf.random.normal([500, 500])

                            start_time = time.time()
                            c_cpu = tf.matmul(a_cpu, b_cpu)
                            cpu_compute_time = time.time() - start_time

                        gpu_tests["cpu_compute_time"] = cpu_compute_time

                        if "gpu_compute_time" in gpu_tests:
                            speedup = cpu_compute_time / gpu_tests["gpu_compute_time"]
                            gpu_tests["gpu_speedup"] = speedup

                    except Exception as e:
                        gpu_tests["cpu_comparison_error"] = str(e)

                    # Test memory growth configuration
                    try:
                        for gpu in gpu_devices:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        gpu_tests["memory_growth_configured"] = True
                    except Exception as e:
                        gpu_tests["memory_growth_configured"] = False
                        gpu_tests["memory_growth_error"] = str(e)

                    success = gpu_tests.get("gpu_allocation_success", False)
                    details = {"gpu_tests": gpu_tests}

            except Exception as e:
                success = False
                details = {"error": str(e), "traceback": traceback.format_exc()}

        metrics = monitor.get_metrics()
        metrics.success_rate = 100.0 if success else 0.0
        metrics.details.update(details)

        return metrics

    def test_polymer_processing_performance(self) -> PerformanceMetrics:
        """Test polymer processing pipeline performance"""
        logger.info("Testing polymer processing performance...")

        with ResourceMonitor("Polymer Processing Performance") as monitor:
            try:
                processing_results = {}

                if not RDKIT_AVAILABLE:
                    success = False
                    details = {"error": "RDKit not available"}
                else:
                    # Test individual polymer processing
                    individual_times = []
                    for name, smiles in self.test_polymers.items():
                        start_time = time.time()

                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            # Calculate molecular descriptors
                            descriptors = {
                                'mw': Descriptors.MolWt(mol),
                                'logp': Descriptors.MolLogP(mol),
                                'tpsa': Descriptors.TPSA(mol),
                                'num_atoms': mol.GetNumAtoms(),
                                'num_bonds': mol.GetNumBonds(),
                                'num_rings': rdMolDescriptors.CalcNumRings(mol)
                            }

                        processing_time = time.time() - start_time
                        individual_times.append(processing_time)

                    processing_results["individual_processing"] = {
                        "avg_time_ms": np.mean(individual_times) * 1000,
                        "max_time_ms": np.max(individual_times) * 1000,
                        "min_time_ms": np.min(individual_times) * 1000,
                        "std_time_ms": np.std(individual_times) * 1000
                    }

                    # Test batch processing
                    start_time = time.time()
                    batch_results = []

                    for name, smiles in self.test_polymers.items():
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            descriptors = {
                                'name': name,
                                'mw': Descriptors.MolWt(mol),
                                'logp': Descriptors.MolLogP(mol),
                                'tpsa': Descriptors.TPSA(mol)
                            }
                            batch_results.append(descriptors)

                    batch_time = time.time() - start_time

                    processing_results["batch_processing"] = {
                        "total_time_s": batch_time,
                        "polymers_processed": len(batch_results),
                        "throughput_per_sec": len(batch_results) / batch_time if batch_time > 0 else 0
                    }

                    # Test PolyID specific operations if available
                    if POLYID_AVAILABLE:
                        try:
                            # Test preprocessor performance
                            start_time = time.time()

                            preprocessor = PolymerPreprocessor(
                                atom_features=atom_features_v1,
                                bond_features=bond_features_v1,
                                explicit_hs=False
                            )

                            # Create test dataframe
                            df_test = pd.DataFrame([
                                {"smiles_polymer": smiles, "polymer_name": name}
                                for name, smiles in list(self.test_polymers.items())[:5]
                            ])

                            # Test graph generation
                            graphs_created = 0
                            for _, row in df_test.iterrows():
                                try:
                                    graph = preprocessor.create_nx_graph(row)
                                    graphs_created += 1
                                except Exception as e:
                                    logger.warning(f"Graph creation failed for {row['polymer_name']}: {e}")

                            preprocessor_time = time.time() - start_time

                            processing_results["polyid_preprocessing"] = {
                                "preprocessor_creation_time_s": preprocessor_time,
                                "graphs_created": graphs_created,
                                "graph_creation_success_rate": graphs_created / len(df_test) * 100
                            }

                        except Exception as e:
                            processing_results["polyid_preprocessing_error"] = str(e)

                    success = len(batch_results) > 0
                    details = {"processing_results": processing_results}

            except Exception as e:
                success = False
                details = {"error": str(e), "traceback": traceback.format_exc()}

        metrics = monitor.get_metrics()
        metrics.success_rate = 100.0 if success else 0.0
        metrics.throughput = details.get("processing_results", {}).get("batch_processing", {}).get("throughput_per_sec", 0)
        metrics.details.update(details)

        return metrics

    def test_concurrent_performance(self, num_threads: int = 4) -> PerformanceMetrics:
        """Test concurrent processing performance"""
        logger.info(f"Testing concurrent performance with {num_threads} threads...")

        with ResourceMonitor(f"Concurrent Performance ({num_threads} threads)") as monitor:
            try:
                def process_polymer_batch(polymer_batch, thread_id):
                    """Process a batch of polymers in a thread"""
                    thread_results = []
                    thread_start_time = time.time()

                    if RDKIT_AVAILABLE:
                        for name, smiles in polymer_batch:
                            try:
                                mol = Chem.MolFromSmiles(smiles)
                                if mol:
                                    descriptors = {
                                        'name': name,
                                        'thread_id': thread_id,
                                        'mw': Descriptors.MolWt(mol),
                                        'logp': Descriptors.MolLogP(mol),
                                        'processing_time': time.time() - thread_start_time
                                    }
                                    thread_results.append(descriptors)
                            except Exception as e:
                                logger.warning(f"Processing failed for {name} in thread {thread_id}: {e}")

                    thread_duration = time.time() - thread_start_time
                    return thread_results, thread_duration

                # Distribute polymers across threads
                polymer_list = list(self.test_polymers.items())
                batch_size = len(polymer_list) // num_threads

                if batch_size == 0:
                    batch_size = 1
                    num_threads = len(polymer_list)

                batches = [
                    polymer_list[i:i + batch_size]
                    for i in range(0, len(polymer_list), batch_size)
                ]

                # Execute concurrent processing
                start_time = time.time()
                all_results = []
                thread_durations = []

                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    future_to_thread = {
                        executor.submit(process_polymer_batch, batch, i): i
                        for i, batch in enumerate(batches[:num_threads])
                    }

                    for future in as_completed(future_to_thread):
                        thread_id = future_to_thread[future]
                        try:
                            results, duration = future.result()
                            all_results.extend(results)
                            thread_durations.append(duration)
                        except Exception as e:
                            logger.error(f"Thread {thread_id} failed: {e}")

                total_duration = time.time() - start_time

                # Calculate concurrent performance metrics
                concurrent_results = {
                    "threads_used": len(batches[:num_threads]),
                    "total_polymers_processed": len(all_results),
                    "total_duration_s": total_duration,
                    "avg_thread_duration_s": np.mean(thread_durations) if thread_durations else 0,
                    "max_thread_duration_s": np.max(thread_durations) if thread_durations else 0,
                    "concurrent_throughput": len(all_results) / total_duration if total_duration > 0 else 0,
                    "thread_efficiency": np.mean(thread_durations) / total_duration if total_duration > 0 else 0
                }

                # Compare with sequential processing
                sequential_start = time.time()
                sequential_results = []

                if RDKIT_AVAILABLE:
                    for name, smiles in polymer_list:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            sequential_results.append({
                                'name': name,
                                'mw': Descriptors.MolWt(mol)
                            })

                sequential_duration = time.time() - sequential_start
                sequential_throughput = len(sequential_results) / sequential_duration if sequential_duration > 0 else 0

                speedup = sequential_throughput / concurrent_results["concurrent_throughput"] if concurrent_results["concurrent_throughput"] > 0 else 0

                concurrent_results.update({
                    "sequential_duration_s": sequential_duration,
                    "sequential_throughput": sequential_throughput,
                    "speedup_factor": speedup,
                    "efficiency_percent": speedup / num_threads * 100 if num_threads > 0 else 0
                })

                success = len(all_results) > 0
                details = {"concurrent_results": concurrent_results}

            except Exception as e:
                success = False
                details = {"error": str(e), "traceback": traceback.format_exc()}

        metrics = monitor.get_metrics()
        metrics.success_rate = 100.0 if success else 0.0
        metrics.throughput = details.get("concurrent_results", {}).get("concurrent_throughput", 0)
        metrics.details.update(details)

        return metrics

    def test_caching_efficiency(self) -> PerformanceMetrics:
        """Test caching effectiveness for preprocessors and models"""
        logger.info("Testing caching efficiency...")

        with ResourceMonitor("Caching Efficiency") as monitor:
            try:
                caching_results = {}

                if not (RDKIT_AVAILABLE and POLYID_AVAILABLE):
                    success = False
                    details = {"error": "Required dependencies not available"}
                else:
                    # Test preprocessor caching
                    preprocessor_times = []

                    # First creation (no cache)
                    start_time = time.time()
                    preprocessor1 = PolymerPreprocessor(
                        atom_features=atom_features_v1,
                        bond_features=bond_features_v1,
                        explicit_hs=False
                    )
                    first_creation_time = time.time() - start_time
                    preprocessor_times.append(first_creation_time)

                    # Subsequent creations (potentially cached)
                    for i in range(3):
                        start_time = time.time()
                        preprocessor = PolymerPreprocessor(
                            atom_features=atom_features_v1,
                            bond_features=bond_features_v1,
                            explicit_hs=False
                        )
                        creation_time = time.time() - start_time
                        preprocessor_times.append(creation_time)

                    caching_results["preprocessor_creation"] = {
                        "first_creation_time_ms": first_creation_time * 1000,
                        "avg_subsequent_time_ms": np.mean(preprocessor_times[1:]) * 1000,
                        "speedup_factor": first_creation_time / np.mean(preprocessor_times[1:]) if np.mean(preprocessor_times[1:]) > 0 else 1.0
                    }

                    # Test molecular processing caching
                    test_smiles = "CC"  # Simple polymer for repeated processing

                    # First processing
                    start_time = time.time()
                    mol1 = Chem.MolFromSmiles(test_smiles)
                    if mol1:
                        mw1 = Descriptors.MolWt(mol1)
                        logp1 = Descriptors.MolLogP(mol1)
                    first_processing_time = time.time() - start_time

                    # Repeated processing
                    repeated_times = []
                    for i in range(5):
                        start_time = time.time()
                        mol = Chem.MolFromSmiles(test_smiles)
                        if mol:
                            mw = Descriptors.MolWt(mol)
                            logp = Descriptors.MolLogP(mol)
                        processing_time = time.time() - start_time
                        repeated_times.append(processing_time)

                    caching_results["molecular_processing"] = {
                        "first_processing_time_ms": first_processing_time * 1000,
                        "avg_repeated_time_ms": np.mean(repeated_times) * 1000,
                        "consistency": np.std(repeated_times) < 0.001,  # Low variance indicates caching
                        "speedup_factor": first_processing_time / np.mean(repeated_times) if np.mean(repeated_times) > 0 else 1.0
                    }

                    # Test TensorFlow operation caching
                    if TF_AVAILABLE:
                        tf_caching_results = {}

                        # Create a simple operation
                        @tf.function
                        def simple_operation(x):
                            return tf.square(x) + tf.sin(x)

                        # First execution (compilation)
                        test_tensor = tf.constant([1.0, 2.0, 3.0])
                        start_time = time.time()
                        result1 = simple_operation(test_tensor)
                        first_tf_time = time.time() - start_time

                        # Subsequent executions (cached)
                        tf_repeated_times = []
                        for i in range(5):
                            start_time = time.time()
                            result = simple_operation(test_tensor)
                            tf_time = time.time() - start_time
                            tf_repeated_times.append(tf_time)

                        tf_caching_results = {
                            "first_execution_time_ms": first_tf_time * 1000,
                            "avg_cached_time_ms": np.mean(tf_repeated_times) * 1000,
                            "speedup_factor": first_tf_time / np.mean(tf_repeated_times) if np.mean(tf_repeated_times) > 0 else 1.0,
                            "caching_effective": first_tf_time > np.mean(tf_repeated_times) * 2  # At least 2x speedup
                        }

                        caching_results["tensorflow_operations"] = tf_caching_results

                    # Calculate overall caching efficiency
                    efficiency_scores = []

                    if "preprocessor_creation" in caching_results:
                        prep_speedup = caching_results["preprocessor_creation"]["speedup_factor"]
                        efficiency_scores.append(min(prep_speedup, 10) / 10 * 100)  # Cap at 10x speedup

                    if "molecular_processing" in caching_results:
                        mol_speedup = caching_results["molecular_processing"]["speedup_factor"]
                        efficiency_scores.append(min(mol_speedup, 5) / 5 * 100)  # Cap at 5x speedup

                    if "tensorflow_operations" in caching_results:
                        tf_speedup = caching_results["tensorflow_operations"]["speedup_factor"]
                        efficiency_scores.append(min(tf_speedup, 20) / 20 * 100)  # Cap at 20x speedup

                    overall_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0

                    caching_results["overall_efficiency"] = {
                        "efficiency_score": overall_efficiency,
                        "components_tested": len(efficiency_scores),
                        "recommendation": (
                            "Excellent caching" if overall_efficiency > 80 else
                            "Good caching" if overall_efficiency > 60 else
                            "Moderate caching" if overall_efficiency > 40 else
                            "Poor caching - optimization needed"
                        )
                    }

                    success = True
                    details = {"caching_results": caching_results}

            except Exception as e:
                success = False
                details = {"error": str(e), "traceback": traceback.format_exc()}

        metrics = monitor.get_metrics()
        metrics.success_rate = 100.0 if success else 0.0
        metrics.details.update(details)

        return metrics

    def run_comprehensive_analysis(self, mode: str = "full") -> Dict[str, PerformanceMetrics]:
        """Run comprehensive performance analysis"""
        logger.info(f"Starting comprehensive performance analysis (mode: {mode})")

        results = {}

        # System info
        logger.info("System Information:")
        logger.info(f"  Python: {self.system_info.python_version}")
        logger.info(f"  CPU Cores: {self.system_info.cpu_count}")
        logger.info(f"  Memory: {self.system_info.memory_total_gb:.1f} GB")
        logger.info(f"  TensorFlow: {self.system_info.tensorflow_version}")
        logger.info(f"  GPU Available: {self.system_info.gpu_available}")
        logger.info(f"  Dependencies: {self.system_info.dependencies}")

        # Run tests based on mode
        if mode in ["full", "quick"]:
            results["cold_start"] = self.test_cold_start_performance()
            results["polymer_processing"] = self.test_polymer_processing_performance()

        if mode in ["full", "memory"]:
            results["memory_usage"] = self.test_memory_usage_patterns()

        if mode in ["full", "gpu"] and self.system_info.gpu_available:
            results["gpu_utilization"] = self.test_gpu_utilization()

        if mode in ["full", "throughput"]:
            results["concurrent_performance"] = self.test_concurrent_performance()

        if mode == "full":
            results["caching_efficiency"] = self.test_caching_efficiency()

        # Store results
        self.metrics.extend(results.values())

        return results

    def generate_performance_report(self, results: Dict[str, PerformanceMetrics]) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("🚀 PolyID Hugging Face Spaces - Performance Analysis Report")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Mode: Comprehensive Performance Monitoring")
        report.append("")

        # System Overview
        report.append("🖥️ SYSTEM OVERVIEW")
        report.append("-" * 40)
        report.append(f"Python Version: {self.system_info.python_version.split()[0]}")
        report.append(f"CPU Cores: {self.system_info.cpu_count}")
        report.append(f"Total Memory: {self.system_info.memory_total_gb:.1f} GB")
        report.append(f"TensorFlow Version: {self.system_info.tensorflow_version or 'Not Available'}")
        report.append(f"GPU Available: {'✅ Yes' if self.system_info.gpu_available else '❌ No'}")
        if self.system_info.gpu_devices:
            for i, device in enumerate(self.system_info.gpu_devices):
                report.append(f"  GPU {i}: {device}")
        report.append("")

        # Dependencies Status
        report.append("📦 DEPENDENCIES STATUS")
        report.append("-" * 40)
        for dep, available in self.system_info.dependencies.items():
            status = "✅ Available" if available else "❌ Missing"
            report.append(f"{dep.capitalize()}: {status}")
        report.append("")

        # Performance Summary
        report.append("📊 PERFORMANCE SUMMARY")
        report.append("-" * 40)

        total_tests = len(results)
        successful_tests = sum(1 for metrics in results.values() if metrics.success_rate and metrics.success_rate > 0)

        report.append(f"Tests Completed: {successful_tests}/{total_tests}")
        report.append(f"Overall Success Rate: {successful_tests/total_tests*100:.1f}%")

        # Performance metrics
        total_duration = sum(metrics.duration for metrics in results.values())
        avg_memory_usage = np.mean([metrics.memory_usage_mb for metrics in results.values()])
        peak_memory = max(metrics.memory_peak_mb for metrics in results.values())

        report.append(f"Total Analysis Duration: {total_duration:.2f}s")
        report.append(f"Average Memory Usage: {avg_memory_usage:.1f} MB")
        report.append(f"Peak Memory Usage: {peak_memory:.1f} MB")
        report.append("")

        # Detailed Test Results
        report.append("🔍 DETAILED TEST RESULTS")
        report.append("-" * 80)

        for test_name, metrics in results.items():
            status = "✅ PASS" if metrics.success_rate and metrics.success_rate > 0 else "❌ FAIL"
            report.append(f"\n{test_name.upper().replace('_', ' ')}: {status}")
            report.append(f"  Duration: {metrics.duration:.3f}s")
            report.append(f"  Memory Usage: {metrics.memory_usage_mb:.1f} MB")
            report.append(f"  Peak Memory: {metrics.memory_peak_mb:.1f} MB")
            report.append(f"  CPU Usage: {metrics.cpu_percent:.1f}%")

            if metrics.throughput:
                report.append(f"  Throughput: {metrics.throughput:.2f} items/sec")

            if metrics.success_rate is not None:
                report.append(f"  Success Rate: {metrics.success_rate:.1f}%")

            # Add specific details based on test type
            if test_name == "cold_start" and metrics.details:
                components = metrics.details.get("components_loaded", {})
                report.append(f"  Components Loaded: {sum(components.values())}/{len(components)}")

            elif test_name == "memory_usage" and metrics.details:
                memory_efficiency = metrics.details.get("memory_efficiency", 0)
                report.append(f"  Memory Efficiency: {memory_efficiency:.1f}%")

            elif test_name == "gpu_utilization" and metrics.details:
                gpu_tests = metrics.details.get("gpu_tests", {})
                if "gpu_speedup" in gpu_tests:
                    report.append(f"  GPU Speedup: {gpu_tests['gpu_speedup']:.2f}x")

            elif test_name == "polymer_processing" and metrics.details:
                processing = metrics.details.get("processing_results", {}).get("batch_processing", {})
                if "throughput_per_sec" in processing:
                    report.append(f"  Processing Throughput: {processing['throughput_per_sec']:.2f} polymers/sec")

            elif test_name == "concurrent_performance" and metrics.details:
                concurrent = metrics.details.get("concurrent_results", {})
                if "efficiency_percent" in concurrent:
                    report.append(f"  Parallel Efficiency: {concurrent['efficiency_percent']:.1f}%")

            elif test_name == "caching_efficiency" and metrics.details:
                caching = metrics.details.get("caching_results", {}).get("overall_efficiency", {})
                if "efficiency_score" in caching:
                    report.append(f"  Caching Efficiency: {caching['efficiency_score']:.1f}%")
                    report.append(f"  Recommendation: {caching.get('recommendation', 'N/A')}")

        report.append("")

        # Performance Recommendations
        report.append("🎯 PERFORMANCE RECOMMENDATIONS")
        report.append("-" * 80)

        recommendations = []

        # Memory recommendations
        if peak_memory > 1000:  # > 1GB
            recommendations.append("• Consider implementing memory pooling for large dataset processing")
            recommendations.append("• Monitor memory usage during batch processing to prevent OOM errors")

        # GPU recommendations
        if "gpu_utilization" in results:
            gpu_metrics = results["gpu_utilization"]
            if gpu_metrics.success_rate and gpu_metrics.success_rate > 0:
                gpu_tests = gpu_metrics.details.get("gpu_tests", {})
                if gpu_tests.get("gpu_speedup", 0) < 2:
                    recommendations.append("• GPU speedup is low - consider optimizing tensor operations")
                    recommendations.append("• Ensure batch sizes are large enough to utilize GPU efficiently")
            else:
                recommendations.append("• GPU utilization test failed - check TensorFlow GPU configuration")

        # Throughput recommendations
        if "polymer_processing" in results:
            processing_metrics = results["polymer_processing"]
            throughput = processing_metrics.throughput or 0
            if throughput < 10:  # Less than 10 polymers/sec
                recommendations.append("• Polymer processing throughput is low - consider batch optimization")
                recommendations.append("• Implement parallel processing for molecular descriptor calculation")

        # Caching recommendations
        if "caching_efficiency" in results:
            caching_metrics = results["caching_efficiency"]
            if caching_metrics.details:
                efficiency = caching_metrics.details.get("caching_results", {}).get("overall_efficiency", {}).get("efficiency_score", 0)
                if efficiency < 60:
                    recommendations.append("• Implement model caching to reduce cold start times")
                    recommendations.append("• Consider preprocessor caching for repeated operations")

        # Deployment recommendations
        recommendations.extend([
            "• Monitor memory usage in production to prevent crashes",
            "• Implement gradual model loading to improve cold start performance",
            "• Use TensorFlow model optimization for inference acceleration",
            "• Consider implementing request batching for concurrent users"
        ])

        if not recommendations:
            recommendations.append("✅ Performance is good! No critical optimizations needed.")

        for rec in recommendations:
            report.append(rec)

        report.append("")

        # Deployment Optimization
        report.append("🚀 DEPLOYMENT OPTIMIZATION GUIDE")
        report.append("-" * 80)
        report.append("1. Memory Optimization:")
        report.append("   - Set TensorFlow memory growth: tf.config.experimental.set_memory_growth()")
        report.append("   - Use lazy loading for models and preprocessors")
        report.append("   - Implement garbage collection after batch processing")
        report.append("")
        report.append("2. GPU Optimization:")
        report.append("   - Configure mixed precision training: tf.keras.mixed_precision")
        report.append("   - Use tf.function for operation caching")
        report.append("   - Optimize batch sizes for GPU memory")
        report.append("")
        report.append("3. Caching Strategy:")
        report.append("   - Cache preprocessed molecular graphs")
        report.append("   - Implement model warm-up on startup")
        report.append("   - Use Redis or similar for cross-request caching")
        report.append("")
        report.append("4. Concurrency Optimization:")
        report.append("   - Use ThreadPoolExecutor for I/O operations")
        report.append("   - Implement request queuing for high load")
        report.append("   - Consider async processing for non-blocking operations")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def create_performance_visualization(self, results: Dict[str, PerformanceMetrics]) -> plt.Figure:
        """Create comprehensive performance visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Test Duration and Success Rate
        test_names = [name.replace('_', '\n') for name in results.keys()]
        durations = [metrics.duration for metrics in results.values()]
        success_rates = [metrics.success_rate or 0 for metrics in results.values()]
        colors = ['green' if rate > 0 else 'red' for rate in success_rates]

        bars1 = ax1.bar(test_names, durations, color=colors, alpha=0.7)
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_title('Test Duration by Category')
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Add success rate annotations
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.annotate(f'{rate:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # 2. Memory Usage Patterns
        memory_usage = [metrics.memory_usage_mb for metrics in results.values()]
        memory_peak = [metrics.memory_peak_mb for metrics in results.values()]

        x_pos = np.arange(len(test_names))
        ax2.bar(x_pos - 0.2, memory_usage, 0.4, label='Memory Delta', alpha=0.7, color='blue')
        ax2.bar(x_pos + 0.2, memory_peak, 0.4, label='Peak Memory', alpha=0.7, color='orange')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Memory Usage Patterns')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(test_names, rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. Throughput Analysis
        throughputs = []
        throughput_labels = []
        for name, metrics in results.items():
            if metrics.throughput and metrics.throughput > 0:
                throughputs.append(metrics.throughput)
                throughput_labels.append(name.replace('_', '\n'))

        if throughputs:
            ax3.barh(throughput_labels, throughputs, color='green', alpha=0.7)
            ax3.set_xlabel('Throughput (items/sec)')
            ax3.set_title('Processing Throughput')
            ax3.grid(axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No throughput data available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Processing Throughput - No Data')

        # 4. Performance Score Overview
        performance_scores = []
        score_labels = []

        for name, metrics in results.items():
            if metrics.success_rate is not None:
                # Calculate composite performance score
                score = metrics.success_rate

                # Adjust score based on efficiency metrics
                if metrics.throughput and metrics.throughput > 0:
                    score = min(100, score * (1 + metrics.throughput / 10))  # Bonus for high throughput

                if metrics.memory_usage_mb > 500:  # Penalty for high memory usage
                    score = score * 0.9

                performance_scores.append(score)
                score_labels.append(name.replace('_', '\n'))

        if performance_scores:
            colors_perf = ['green' if score > 80 else 'orange' if score > 60 else 'red'
                          for score in performance_scores]
            ax4.bar(score_labels, performance_scores, color=colors_perf, alpha=0.7)
            ax4.set_ylabel('Performance Score')
            ax4.set_title('Overall Performance Scores')
            ax4.set_ylim(0, 100)
            ax4.grid(axis='y', alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No performance scores available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance Scores - No Data')

        plt.tight_layout()
        return fig

def main():
    """Main function for performance monitoring"""
    parser = argparse.ArgumentParser(description='PolyID Performance Monitor')
    parser.add_argument('--mode', choices=['full', 'quick', 'gpu', 'memory', 'throughput'],
                       default='full', help='Analysis mode')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for reports and visualizations')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results as JSON file')

    args = parser.parse_args()

    try:
        # Create analyzer
        analyzer = PolyIDPerformanceAnalyzer()

        # Run analysis
        logger.info(f"Starting performance analysis in {args.mode} mode...")
        results = analyzer.run_comprehensive_analysis(mode=args.mode)

        # Generate report
        report = analyzer.generate_performance_report(results)
        print("\n" + "="*80)
        print(report)

        # Save report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        report_file = output_dir / f"polyid_performance_report_{args.mode}_{int(time.time())}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")

        # Create visualization
        try:
            fig = analyzer.create_performance_visualization(results)
            viz_file = output_dir / f"polyid_performance_viz_{args.mode}_{int(time.time())}.png"
            fig.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Visualization saved to: {viz_file}")
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")

        # Save JSON if requested
        if args.save_json:
            json_data = {
                'system_info': {
                    'python_version': analyzer.system_info.python_version,
                    'cpu_count': analyzer.system_info.cpu_count,
                    'memory_total_gb': analyzer.system_info.memory_total_gb,
                    'tensorflow_version': analyzer.system_info.tensorflow_version,
                    'gpu_available': analyzer.system_info.gpu_available,
                    'gpu_devices': analyzer.system_info.gpu_devices,
                    'dependencies': analyzer.system_info.dependencies
                },
                'results': {
                    name: {
                        'timestamp': metrics.timestamp,
                        'test_name': metrics.test_name,
                        'duration': metrics.duration,
                        'memory_usage_mb': metrics.memory_usage_mb,
                        'memory_peak_mb': metrics.memory_peak_mb,
                        'cpu_percent': metrics.cpu_percent,
                        'gpu_utilization': metrics.gpu_utilization,
                        'gpu_memory_used_mb': metrics.gpu_memory_used_mb,
                        'gpu_memory_total_mb': metrics.gpu_memory_total_mb,
                        'throughput': metrics.throughput,
                        'success_rate': metrics.success_rate,
                        'details': metrics.details
                    }
                    for name, metrics in results.items()
                }
            }

            json_file = output_dir / f"polyid_performance_data_{args.mode}_{int(time.time())}.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            logger.info(f"JSON data saved to: {json_file}")

        # Return success status
        successful_tests = sum(1 for metrics in results.values()
                             if metrics.success_rate and metrics.success_rate > 0)
        success_rate = successful_tests / len(results) if results else 0

        logger.info(f"Analysis completed with {success_rate*100:.1f}% success rate")
        return success_rate > 0.5  # Consider successful if >50% of tests passed

    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)