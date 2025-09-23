"""
PolyID - Comprehensive Optimization Testing Suite
Tests for Phase 1 and Phase 2 performance optimizations
"""

import pytest
import time
import sys
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our optimization components
from polyid.imports import TF_AVAILABLE, rdkit
from polyid.cache_manager import ModelCacheManager, get_cache_manager
from polyid.optimized_predictor import OptimizedPredictor
from polyid.molecular_viewer import MolecularViewer
from polyid.async_processor import AsyncPredictionProcessor, get_async_processor
from polyid.performance_monitor import PerformanceMonitor, get_performance_monitor

class TestPhase1Optimizations:
    """Test Phase 1 critical fixes and startup optimizations"""
    
    def test_tensorflow_configuration(self):
        """Test TensorFlow optimization settings"""
        assert TF_AVAILABLE, "TensorFlow should be available"
        
        # Check environment variables are set correctly
        assert os.environ.get('TF_CPP_MIN_LOG_LEVEL') == '2'
        assert os.environ.get('OMP_NUM_THREADS') == '4'
        assert os.environ.get('TF_ENABLE_ONEDNN_OPTS') == '0'
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    
    def test_rdkit_availability(self):
        """Test RDKit chemistry library availability"""
        assert rdkit is not None, "RDKit should be available"
        
        # Test basic functionality
        from rdkit import Chem
        mol = Chem.MolFromSmiles('CC')  # Ethane
        assert mol is not None, "RDKit should parse simple SMILES"
    
    def test_lazy_imports(self):
        """Test lazy import system performance"""
        from polyid.imports import LazyImporter
        
        # Create lazy importer
        lazy_tf = LazyImporter('tensorflow')
        
        # Should not import until accessed
        assert lazy_tf._module is None
        
        # Access should trigger import
        version = lazy_tf.__version__
        assert version is not None
        assert lazy_tf._module is not None
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization and basic functionality"""
        cache_manager = get_cache_manager()
        
        assert isinstance(cache_manager, ModelCacheManager)
        
        # Test cache operations
        test_key = "test_key"
        test_result = {"test": "data"}
        
        cache_manager.cache_prediction(test_key, test_result)
        cached_result = cache_manager.get_cached_prediction(test_key)
        
        assert cached_result == test_result
        
        # Test cache statistics
        stats = cache_manager.get_cache_stats()
        assert 'memory_cache_size' in stats
        assert stats['memory_cache_size'] >= 1

class TestPhase2PerformanceOptimizations:
    """Test Phase 2 performance pipeline optimizations"""
    
    def test_optimized_predictor_initialization(self):
        """Test optimized predictor lazy initialization"""
        predictor = OptimizedPredictor()
        
        # Should not be initialized yet
        assert not predictor._initialized
        
        # Initialization should work
        success = predictor.lazy_initialize()
        assert success, "Predictor initialization should succeed"
        assert predictor._initialized
    
    def test_optimized_predictor_performance(self):
        """Test prediction performance with timing"""
        predictor = OptimizedPredictor()
        
        test_smiles = "CC"
        test_properties = ["Glass Transition Temperature (Tg)"]
        
        # First prediction (may include initialization)
        start_time = time.time()
        result1 = predictor.predict_properties(test_smiles, test_properties)
        first_time = time.time() - start_time
        
        # Second prediction (should be faster due to caching and warm-up)
        start_time = time.time()
        result2 = predictor.predict_properties(test_smiles, test_properties)
        second_time = time.time() - start_time
        
        assert 'error' not in result1, f"First prediction failed: {result1}"
        assert 'error' not in result2, f"Second prediction failed: {result2}"
        
        # Second prediction should be faster (caching effect)
        assert second_time <= first_time * 1.2, "Caching should improve performance"
        
        # Both should be reasonably fast
        assert first_time < 10.0, "Initial prediction should complete in <10s"
        assert second_time < 2.0, "Cached prediction should complete in <2s"
    
    def test_molecular_viewer(self):
        """Test molecular visualization components"""
        viewer = MolecularViewer()
        
        # Test molecular image creation
        test_smiles = "CC"
        img_data = viewer.create_molecule_image(test_smiles)
        
        assert img_data.startswith("data:image/png;base64,"), "Should return base64 image data"
        
        # Test property chart creation
        test_properties = {
            "Glass Transition Temperature (Tg)": {"value": 300, "unit": "K", "confidence": "High"}
        }
        
        fig = viewer.create_property_radar_chart(test_properties)
        assert fig is not None, "Should create matplotlib figure"
        
        # Test cache functionality
        cache_stats = viewer.get_cache_stats()
        assert 'cache_size' in cache_stats
    
    def test_async_processor(self):
        """Test asynchronous processing capabilities"""
        processor = get_async_processor()
        
        assert isinstance(processor, AsyncPredictionProcessor)
        
        # Test single prediction processing
        test_data = [
            {"smiles": "CC", "properties": ["Glass Transition Temperature (Tg)"]},
            {"smiles": "CC(C)", "properties": ["Glass Transition Temperature (Tg)"]}
        ]
        
        start_time = time.time()
        results = processor.predict_batch_sync(test_data)
        processing_time = time.time() - start_time
        
        assert len(results) == 2, "Should process all items"
        assert processing_time < 30.0, "Batch processing should complete in reasonable time"
        
        # Check results format
        for result in results:
            assert isinstance(result, dict), "Results should be dictionaries"
    
    def test_performance_monitor(self):
        """Test performance monitoring system"""
        monitor = get_performance_monitor()
        
        assert isinstance(monitor, PerformanceMonitor)
        
        # Test metric recording
        monitor.record_metric("test_metric", 42.0, "units")
        
        # Test prediction performance recording
        monitor.record_prediction_performance(
            prediction_time=1.5,
            cache_hit=True,
            error_occurred=False,
            smiles="CC"
        )
        
        # Test metrics retrieval
        current_metrics = monitor.get_current_metrics()
        assert 'timestamp' in current_metrics
        assert 'system' in current_metrics
        assert 'application' in current_metrics
        
        # Test performance summary
        summary = monitor.get_performance_summary(time_window_minutes=1)
        assert 'total_predictions' in summary
        assert 'performance' in summary
        assert 'reliability' in summary
        
        # Test optimization recommendations
        recommendations = monitor.get_optimization_recommendations()
        assert isinstance(recommendations, list)

class TestIntegrationPerformance:
    """Integration tests for complete performance optimization"""
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline performance"""
        
        # Initialize all components
        predictor = OptimizedPredictor()
        cache_manager = get_cache_manager()
        performance_monitor = get_performance_monitor()
        
        test_molecules = [
            ("CC", "Polyethylene"),
            ("CC(C)", "Polypropylene"), 
            ("CC(c1ccccc1)", "Polystyrene")
        ]
        
        test_properties = ["Glass Transition Temperature (Tg)", "Melting Temperature (Tm)"]
        
        results = []
        total_start_time = time.time()
        
        for smiles, name in test_molecules:
            start_time = time.time()
            result = predictor.predict_properties(smiles, test_properties)
            prediction_time = time.time() - start_time
            
            results.append({
                'smiles': smiles,
                'name': name,
                'result': result,
                'time': prediction_time
            })
        
        total_time = time.time() - total_start_time
        
        # Validate results
        assert len(results) == 3, "Should process all test molecules"
        
        for result_data in results:
            assert 'error' not in result_data['result'], f"Prediction failed for {result_data['name']}"
            assert result_data['time'] < 5.0, f"Prediction too slow for {result_data['name']}"
        
        # Check overall performance
        assert total_time < 15.0, "Complete pipeline should finish in <15s"
        
        # Check cache effectiveness (second run should be faster)
        second_start_time = time.time()
        for smiles, _ in test_molecules:
            predictor.predict_properties(smiles, test_properties)
        second_total_time = time.time() - second_start_time
        
        assert second_total_time < total_time * 0.8, "Caching should improve repeat performance"
    
    def test_batch_processing_performance(self):
        """Test batch processing performance and scalability"""
        
        processor = get_async_processor()
        
        # Create test data
        test_data = []
        for i in range(20):  # Test with 20 molecules
            smiles = ["CC", "CC(C)", "CC(c1ccccc1)"][i % 3]
            test_data.append({
                "smiles": smiles,
                "properties": ["Glass Transition Temperature (Tg)"]
            })
        
        start_time = time.time()
        results = processor.predict_batch_sync(test_data)
        batch_time = time.time() - start_time
        
        # Validate batch results
        assert len(results) == 20, "Should process all batch items"
        
        successful_results = [r for r in results if 'error' not in r]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate >= 0.8, "Batch processing should have >80% success rate"
        assert batch_time < 60.0, "Batch processing should complete in reasonable time"
        
        # Check performance stats
        stats = processor.get_performance_stats()
        assert stats['processed_count'] >= 20
        assert stats['success_rate'] >= 0.8
    
    def test_csv_batch_processing(self):
        """Test CSV file batch processing"""
        
        processor = get_async_processor()
        
        # Create test CSV content
        test_csv_data = """smiles_polymer,polymer_name
CC,Polyethylene
CC(C),Polypropylene
CC(c1ccccc1),Polystyrene
CC(C)(C(=O)OC),PMMA
"""
        
        # Test CSV processing
        try:
            results_df = processor.process_csv_file(
                test_csv_data,
                smiles_column="smiles_polymer",
                properties=["Glass Transition Temperature (Tg)"]
            )
            
            assert isinstance(results_df, pd.DataFrame), "Should return DataFrame"
            assert len(results_df) == 4, "Should process all CSV rows"
            
            # Check for prediction columns
            expected_columns = ["Glass Transition Temperature (Tg)_pred", 
                              "Glass Transition Temperature (Tg)_confidence"]
            for col in expected_columns:
                assert col in results_df.columns, f"Should have column {col}"
            
            # Test export functionality
            csv_export = processor.export_results(results_df, 'csv')
            assert isinstance(csv_export, bytes), "Should export as bytes"
            
            json_export = processor.export_results(results_df, 'json')
            assert isinstance(json_export, bytes), "Should export JSON as bytes"
            
        except Exception as e:
            pytest.fail(f"CSV batch processing failed: {e}")

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_startup_time_benchmark(self):
        """Benchmark application startup time"""
        
        # Test component initialization times
        components = [
            ("Cache Manager", lambda: get_cache_manager()),
            ("Optimized Predictor", lambda: OptimizedPredictor()),
            ("Async Processor", lambda: get_async_processor()),
            ("Performance Monitor", lambda: get_performance_monitor()),
            ("Molecular Viewer", lambda: MolecularViewer())
        ]
        
        initialization_times = {}
        
        for name, init_func in components:
            start_time = time.time()
            component = init_func()
            init_time = time.time() - start_time
            initialization_times[name] = init_time
            
            # Each component should initialize quickly
            assert init_time < 2.0, f"{name} initialization too slow: {init_time:.2f}s"
        
        total_init_time = sum(initialization_times.values())
        assert total_init_time < 5.0, f"Total initialization too slow: {total_init_time:.2f}s"
        
        print(f"\n🚀 Startup Performance Benchmark:")
        for name, init_time in initialization_times.items():
            print(f"  {name}: {init_time:.3f}s")
        print(f"  Total: {total_init_time:.3f}s")
    
    def test_prediction_speed_benchmark(self):
        """Benchmark prediction speed across different molecule types"""
        
        predictor = OptimizedPredictor()
        
        # Test molecules of varying complexity
        test_molecules = [
            ("CC", "Simple alkane"),
            ("CC(C)", "Branched alkane"),
            ("CC(c1ccccc1)", "Aromatic"),
            ("CC(C)(C(=O)OC)", "Ester functional group"),
            ("NCCCCCC(=O)", "Amide functional group"),
            ("CC(C)(c1ccc(O)cc1)c1ccc(O)cc1.O=C(Cl)Cl", "Complex polycarbonate")
        ]
        
        benchmark_results = []
        
        for smiles, description in test_molecules:
            times = []
            
            # Run multiple predictions for statistical significance
            for _ in range(3):
                start_time = time.time()
                result = predictor.predict_properties(smiles, ["Glass Transition Temperature (Tg)"])
                prediction_time = time.time() - start_time
                times.append(prediction_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            benchmark_results.append({
                'smiles': smiles,
                'description': description,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min(times),
                'max_time': max(times)
            })
            
            # Each prediction should be reasonably fast
            assert avg_time < 3.0, f"Prediction too slow for {description}: {avg_time:.2f}s"
        
        print(f"\n⚡ Prediction Speed Benchmark:")
        for result in benchmark_results:
            print(f"  {result['description']}: {result['avg_time']:.3f}±{result['std_time']:.3f}s")
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage during processing"""
        
        try:
            import psutil
            process = psutil.Process()
            
            # Measure baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create components and measure memory growth
            predictor = OptimizedPredictor()
            after_predictor = process.memory_info().rss / 1024 / 1024
            
            viewer = MolecularViewer()
            after_viewer = process.memory_info().rss / 1024 / 1024
            
            processor = get_async_processor()
            after_processor = process.memory_info().rss / 1024 / 1024
            
            # Perform some predictions
            test_smiles = ["CC", "CC(C)", "CC(c1ccccc1)"] * 10
            for smiles in test_smiles:
                predictor.predict_properties(smiles, ["Glass Transition Temperature (Tg)"])
            
            final_memory = process.memory_info().rss / 1024 / 1024
            
            memory_growth = final_memory - baseline_memory
            
            print(f"\n💾 Memory Usage Benchmark:")
            print(f"  Baseline: {baseline_memory:.1f} MB")
            print(f"  After Predictor: {after_predictor:.1f} MB (+{after_predictor-baseline_memory:.1f})")
            print(f"  After Viewer: {after_viewer:.1f} MB (+{after_viewer-after_predictor:.1f})")
            print(f"  After Processor: {after_processor:.1f} MB (+{after_processor-after_viewer:.1f})")
            print(f"  After Predictions: {final_memory:.1f} MB")
            print(f"  Total Growth: {memory_growth:.1f} MB")
            
            # Memory growth should be reasonable
            assert memory_growth < 500, f"Memory growth too high: {memory_growth:.1f} MB"
            
        except ImportError:
            pytest.skip("psutil not available for memory benchmarking")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])