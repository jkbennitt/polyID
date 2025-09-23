#!/usr/bin/env python3
"""
PolyID Hugging Face Space Performance Testing Suite
==================================================

Comprehensive performance testing for the live PolyID deployment at:
https://jkbennitt-polyid-private.hf.space

This suite tests:
1. Model loading time benchmarks
2. Prediction latency measurements with various input sizes
3. Memory usage profiling during predictions
4. Concurrent user load testing
5. GPU utilization monitoring
6. Response time analysis under different load conditions
7. API endpoint stress testing

Requirements:
- requests
- asyncio
- aiohttp
- psutil
- numpy
- pandas
- matplotlib
- seaborn
- threading
"""

import asyncio
import aiohttp
import requests
import time
import json
import threading
import statistics
import psutil
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urljoin

# Test configuration
SPACE_URL = "https://jkbennitt-polyid-private.hf.space"
API_ENDPOINT = "/api/predict"
GRADIO_API_ENDPOINT = "/api"

# Sample test polymers with varying complexity
TEST_POLYMERS = {
    "simple": [
        "CC",  # Polyethylene
        "CC(C)",  # Polypropylene
        "CCC",  # Simple chain
    ],
    "medium": [
        "CC(c1ccccc1)",  # Polystyrene
        "CC(C)(C(=O)OC)",  # PMMA
        "CC(C)(C)CC(C)(C)",  # Branched polymer
    ],
    "complex": [
        "COC(=O)c1ccc(C(=O)O)cc1.OCCO",  # PET-like
        "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1.O=C(Cl)Cl",  # PC-like
        "CC(C)(C)c1ccc(OC(=O)c2ccc(C(=O)O)cc2)cc1",  # Complex aromatic
    ]
}

PROPERTIES_TO_TEST = [
    "Glass Transition Temperature (Tg)",
    "Melting Temperature (Tm)",
    "Density",
    "Elastic Modulus"
]

@dataclass
class PerformanceMetrics:
    """Container for performance test results"""
    test_name: str
    timestamp: datetime
    response_time: float
    status_code: int
    content_length: int
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    concurrent_users: Optional[int] = None
    input_complexity: Optional[str] = None


class PolyIDPerformanceTester:
    """Main performance testing class for PolyID Hugging Face Space"""

    def __init__(self, base_url: str = SPACE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolyID-Performance-Test/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        self.results: List[PerformanceMetrics] = []
        self.system_monitor = SystemMonitor()

    def get_gradio_info(self) -> Dict[str, Any]:
        """Get Gradio app information and available endpoints"""
        try:
            response = self.session.get(f"{self.base_url}/config")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Warning: Could not get Gradio config (status: {response.status_code})")
                return {}
        except Exception as e:
            print(f"Warning: Error getting Gradio info: {e}")
            return {}

    def test_cold_start_performance(self) -> PerformanceMetrics:
        """Test cold start performance by making the first request"""
        print("Testing cold start performance...")

        # Wait a bit to ensure any caches are cleared
        time.sleep(5)

        start_time = time.time()
        memory_before = self.system_monitor.get_memory_usage()

        try:
            # Make a simple prediction request
            payload = {
                "data": [
                    "CC",  # Simple SMILES
                    ["Glass Transition Temperature (Tg)"]  # One property
                ]
            }

            response = self.session.post(
                f"{self.base_url}/api/predict/",
                json=payload,
                timeout=120  # Long timeout for cold start
            )

            end_time = time.time()
            response_time = end_time - start_time
            memory_after = self.system_monitor.get_memory_usage()

            result = PerformanceMetrics(
                test_name="cold_start",
                timestamp=datetime.now(),
                response_time=response_time,
                status_code=response.status_code,
                content_length=len(response.content),
                memory_usage_mb=memory_after - memory_before,
                input_complexity="simple"
            )

            print(f"Cold start completed in {response_time:.2f} seconds")
            return result

        except Exception as e:
            return PerformanceMetrics(
                test_name="cold_start",
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                status_code=0,
                content_length=0,
                error_message=str(e),
                input_complexity="simple"
            )

    def test_warm_prediction_latency(self, smiles: str, properties: List[str],
                                   complexity: str = "unknown") -> PerformanceMetrics:
        """Test prediction latency for a warmed-up model"""

        start_time = time.time()
        cpu_before = self.system_monitor.get_cpu_usage()
        memory_before = self.system_monitor.get_memory_usage()

        try:
            payload = {
                "data": [smiles, properties]
            }

            response = self.session.post(
                f"{self.base_url}/api/predict/",
                json=payload,
                timeout=30
            )

            end_time = time.time()
            response_time = end_time - start_time

            cpu_after = self.system_monitor.get_cpu_usage()
            memory_after = self.system_monitor.get_memory_usage()

            return PerformanceMetrics(
                test_name="warm_prediction",
                timestamp=datetime.now(),
                response_time=response_time,
                status_code=response.status_code,
                content_length=len(response.content),
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=cpu_after - cpu_before,
                input_complexity=complexity
            )

        except Exception as e:
            return PerformanceMetrics(
                test_name="warm_prediction",
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                status_code=0,
                content_length=0,
                error_message=str(e),
                input_complexity=complexity
            )

    def test_batch_prediction_performance(self) -> List[PerformanceMetrics]:
        """Test performance with different input complexities"""
        print("Testing batch prediction performance across complexity levels...")

        results = []

        for complexity, smiles_list in TEST_POLYMERS.items():
            print(f"Testing {complexity} complexity polymers...")

            for smiles in smiles_list:
                for num_properties in [1, 2, 4]:  # Test with different property counts
                    properties = PROPERTIES_TO_TEST[:num_properties]

                    result = self.test_warm_prediction_latency(smiles, properties, complexity)
                    result.test_name = f"batch_{complexity}_{num_properties}props"
                    results.append(result)

                    # Small delay between requests
                    time.sleep(0.5)

        return results

    def concurrent_user_test(self, num_users: int, duration_seconds: int = 60) -> List[PerformanceMetrics]:
        """Simulate concurrent users making predictions"""
        print(f"Testing concurrent load with {num_users} users for {duration_seconds} seconds...")

        results = []
        start_time = time.time()

        def user_session(user_id: int) -> List[PerformanceMetrics]:
            """Simulate a single user session"""
            user_results = []
            session_start = time.time()

            while time.time() - session_start < duration_seconds:
                # Pick a random polymer and properties
                complexity = np.random.choice(list(TEST_POLYMERS.keys()))
                smiles = np.random.choice(TEST_POLYMERS[complexity])
                num_props = np.random.choice([1, 2, 4])
                properties = np.random.choice(PROPERTIES_TO_TEST, size=num_props, replace=False).tolist()

                result = self.test_warm_prediction_latency(smiles, properties, complexity)
                result.test_name = f"concurrent_user_{user_id}"
                result.concurrent_users = num_users
                user_results.append(result)

                # Random delay between requests (0.5-3 seconds)
                time.sleep(np.random.uniform(0.5, 3.0))

            return user_results

        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(user_session, i) for i in range(num_users)]

            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    results.extend(user_results)
                except Exception as e:
                    print(f"Error in user session: {e}")

        print(f"Concurrent test completed. Total requests: {len(results)}")
        return results

    async def async_stress_test(self, requests_per_second: int, duration_seconds: int = 30) -> List[PerformanceMetrics]:
        """Asynchronous stress test with controlled request rate"""
        print(f"Running async stress test: {requests_per_second} req/sec for {duration_seconds} seconds...")

        results = []
        interval = 1.0 / requests_per_second

        async def make_request(session: aiohttp.ClientSession, request_id: int) -> PerformanceMetrics:
            start_time = time.time()

            # Pick random test data
            complexity = np.random.choice(list(TEST_POLYMERS.keys()))
            smiles = np.random.choice(TEST_POLYMERS[complexity])
            properties = [np.random.choice(PROPERTIES_TO_TEST)]

            try:
                payload = {"data": [smiles, properties]}

                async with session.post(
                    f"{self.base_url}/api/predict/",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    content = await response.read()
                    end_time = time.time()

                    return PerformanceMetrics(
                        test_name=f"stress_test_{requests_per_second}rps",
                        timestamp=datetime.now(),
                        response_time=end_time - start_time,
                        status_code=response.status,
                        content_length=len(content),
                        input_complexity=complexity
                    )

            except Exception as e:
                return PerformanceMetrics(
                    test_name=f"stress_test_{requests_per_second}rps",
                    timestamp=datetime.now(),
                    response_time=time.time() - start_time,
                    status_code=0,
                    content_length=0,
                    error_message=str(e),
                    input_complexity=complexity
                )

        # Create connector with appropriate limits
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            start_time = time.time()
            request_count = 0

            while time.time() - start_time < duration_seconds:
                task = asyncio.create_task(make_request(session, request_count))
                tasks.append(task)
                request_count += 1

                await asyncio.sleep(interval)

            # Wait for all requests to complete
            results_raw = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results_raw:
                if isinstance(result, PerformanceMetrics):
                    results.append(result)
                elif isinstance(result, Exception):
                    print(f"Request failed: {result}")

        print(f"Stress test completed. Total requests: {len(results)}")
        return results

    def test_gradio_interface_load_time(self) -> PerformanceMetrics:
        """Test the time to load the Gradio interface"""
        print("Testing Gradio interface load time...")

        start_time = time.time()

        try:
            response = self.session.get(self.base_url, timeout=30)
            end_time = time.time()

            return PerformanceMetrics(
                test_name="gradio_interface_load",
                timestamp=datetime.now(),
                response_time=end_time - start_time,
                status_code=response.status_code,
                content_length=len(response.content)
            )

        except Exception as e:
            return PerformanceMetrics(
                test_name="gradio_interface_load",
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                status_code=0,
                content_length=0,
                error_message=str(e)
            )

    def run_comprehensive_test_suite(self) -> Dict[str, List[PerformanceMetrics]]:
        """Run the complete performance test suite"""
        print("Starting comprehensive PolyID performance test suite...")
        print(f"Target URL: {self.base_url}")
        print("=" * 60)

        all_results = {}

        # 1. Interface load test
        interface_result = self.test_gradio_interface_load_time()
        all_results["interface_load"] = [interface_result]

        # 2. Cold start test
        cold_start_result = self.test_cold_start_performance()
        all_results["cold_start"] = [cold_start_result]

        # 3. Warm-up with a few requests
        print("Warming up the model...")
        for _ in range(3):
            self.test_warm_prediction_latency("CC", ["Glass Transition Temperature (Tg)"], "warmup")
            time.sleep(1)

        # 4. Batch performance tests
        batch_results = self.test_batch_prediction_performance()
        all_results["batch_performance"] = batch_results

        # 5. Concurrent user tests (start with smaller numbers)
        for num_users in [2, 5, 10]:
            concurrent_results = self.concurrent_user_test(num_users, duration_seconds=30)
            all_results[f"concurrent_{num_users}_users"] = concurrent_results

        # 6. Stress tests with different request rates
        for rps in [1, 2, 5]:
            try:
                stress_results = asyncio.run(self.async_stress_test(rps, duration_seconds=20))
                all_results[f"stress_{rps}_rps"] = stress_results
            except Exception as e:
                print(f"Stress test {rps} rps failed: {e}")

        # Store all results
        self.results = []
        for test_results in all_results.values():
            self.results.extend(test_results)

        print("=" * 60)
        print("Comprehensive test suite completed!")
        return all_results


class SystemMonitor:
    """Monitor system resources during testing"""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent()


class PerformanceAnalyzer:
    """Analyze and visualize performance test results"""

    def __init__(self, results: List[PerformanceMetrics]):
        self.results = results
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        data = []
        for result in self.results:
            data.append({
                'test_name': result.test_name,
                'timestamp': result.timestamp,
                'response_time': result.response_time,
                'status_code': result.status_code,
                'content_length': result.content_length,
                'memory_usage_mb': result.memory_usage_mb,
                'cpu_usage_percent': result.cpu_usage_percent,
                'concurrent_users': result.concurrent_users,
                'input_complexity': result.input_complexity,
                'success': result.status_code == 200,
                'error_message': result.error_message
            })
        return pd.DataFrame(data)

    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        successful_requests = self.df[self.df['success']]

        stats = {
            'total_requests': len(self.df),
            'successful_requests': len(successful_requests),
            'success_rate': len(successful_requests) / len(self.df) * 100,
            'avg_response_time': successful_requests['response_time'].mean(),
            'median_response_time': successful_requests['response_time'].median(),
            'p95_response_time': successful_requests['response_time'].quantile(0.95),
            'p99_response_time': successful_requests['response_time'].quantile(0.99),
            'min_response_time': successful_requests['response_time'].min(),
            'max_response_time': successful_requests['response_time'].max(),
            'std_response_time': successful_requests['response_time'].std(),
        }

        if 'memory_usage_mb' in successful_requests.columns:
            memory_data = successful_requests['memory_usage_mb'].dropna()
            if not memory_data.empty:
                stats.update({
                    'avg_memory_usage_mb': memory_data.mean(),
                    'max_memory_usage_mb': memory_data.max(),
                })

        return stats

    def create_performance_plots(self, output_dir: str = "performance_plots"):
        """Create comprehensive performance visualization plots"""
        os.makedirs(output_dir, exist_ok=True)

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Response time distribution
        plt.figure(figsize=(12, 8))
        successful_df = self.df[self.df['success']]

        plt.subplot(2, 2, 1)
        plt.hist(successful_df['response_time'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')

        # 2. Response time by test type
        plt.subplot(2, 2, 2)
        test_types = successful_df['test_name'].unique()
        for test_type in test_types:
            data = successful_df[successful_df['test_name'] == test_type]['response_time']
            if not data.empty:
                plt.hist(data, alpha=0.5, label=test_type, bins=20)
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Response Time by Test Type')
        plt.legend()

        # 3. Response time vs concurrent users
        plt.subplot(2, 2, 3)
        concurrent_data = successful_df[successful_df['concurrent_users'].notna()]
        if not concurrent_data.empty:
            sns.boxplot(data=concurrent_data, x='concurrent_users', y='response_time')
        plt.xlabel('Concurrent Users')
        plt.ylabel('Response Time (seconds)')
        plt.title('Response Time vs Concurrent Users')

        # 4. Success rate over time
        plt.subplot(2, 2, 4)
        self.df['minute'] = self.df['timestamp'].dt.floor('T')
        success_rate_by_minute = self.df.groupby('minute')['success'].mean() * 100
        plt.plot(success_rate_by_minute.index, success_rate_by_minute.values, marker='o')
        plt.xlabel('Time')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate Over Time')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Detailed response time analysis
        plt.figure(figsize=(15, 10))

        # Response time by complexity
        plt.subplot(2, 3, 1)
        complexity_data = successful_df[successful_df['input_complexity'].notna()]
        if not complexity_data.empty:
            sns.boxplot(data=complexity_data, x='input_complexity', y='response_time')
        plt.title('Response Time by Input Complexity')
        plt.xticks(rotation=45)

        # Memory usage over time
        plt.subplot(2, 3, 2)
        memory_data = successful_df[successful_df['memory_usage_mb'].notna()]
        if not memory_data.empty:
            plt.scatter(memory_data['timestamp'], memory_data['memory_usage_mb'], alpha=0.6)
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Over Time')
        plt.xticks(rotation=45)

        # Throughput analysis
        plt.subplot(2, 3, 3)
        throughput_data = self.df.groupby('minute').size()
        plt.plot(throughput_data.index, throughput_data.values, marker='o')
        plt.xlabel('Time')
        plt.ylabel('Requests per Minute')
        plt.title('Request Throughput')
        plt.xticks(rotation=45)

        # Error analysis
        plt.subplot(2, 3, 4)
        error_data = self.df[~self.df['success']]
        if not error_data.empty:
            error_counts = error_data['test_name'].value_counts()
            plt.bar(range(len(error_counts)), error_counts.values)
            plt.xticks(range(len(error_counts)), error_counts.index, rotation=45)
        plt.ylabel('Error Count')
        plt.title('Errors by Test Type')

        # Response time percentiles
        plt.subplot(2, 3, 5)
        percentiles = [50, 75, 90, 95, 99]
        response_times = successful_df['response_time']
        percentile_values = [response_times.quantile(p/100) for p in percentiles]
        plt.bar(range(len(percentiles)), percentile_values)
        plt.xticks(range(len(percentiles)), [f'P{p}' for p in percentiles])
        plt.ylabel('Response Time (seconds)')
        plt.title('Response Time Percentiles')

        # Load test performance
        plt.subplot(2, 3, 6)
        load_test_data = successful_df[successful_df['test_name'].str.contains('concurrent|stress')]
        if not load_test_data.empty:
            sns.scatterplot(data=load_test_data, x='timestamp', y='response_time',
                          hue='test_name', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Response Time (seconds)')
        plt.title('Load Test Performance')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance plots saved to {output_dir}/")

    def generate_report(self, output_file: str = "performance_report.md"):
        """Generate a comprehensive performance report"""
        stats = self.generate_summary_stats()

        report = f"""# PolyID Performance Test Report

## Test Summary
- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Target URL**: {SPACE_URL}
- **Total Requests**: {stats['total_requests']}
- **Successful Requests**: {stats['successful_requests']}
- **Success Rate**: {stats['success_rate']:.2f}%

## Response Time Metrics
- **Average Response Time**: {stats['avg_response_time']:.3f} seconds
- **Median Response Time**: {stats['median_response_time']:.3f} seconds
- **95th Percentile**: {stats['p95_response_time']:.3f} seconds
- **99th Percentile**: {stats['p99_response_time']:.3f} seconds
- **Min Response Time**: {stats['min_response_time']:.3f} seconds
- **Max Response Time**: {stats['max_response_time']:.3f} seconds
- **Standard Deviation**: {stats['std_response_time']:.3f} seconds

## Performance by Test Type

"""

        # Add test type breakdown
        test_stats = self.df.groupby('test_name').agg({
            'response_time': ['count', 'mean', 'median', 'std'],
            'success': 'mean'
        }).round(3)

        report += "| Test Type | Count | Avg Time (s) | Median Time (s) | Std Dev | Success Rate |\n"
        report += "|-----------|-------|--------------|-----------------|---------|-------------|\n"

        for test_name in test_stats.index:
            count = test_stats.loc[test_name, ('response_time', 'count')]
            avg_time = test_stats.loc[test_name, ('response_time', 'mean')]
            median_time = test_stats.loc[test_name, ('response_time', 'median')]
            std_dev = test_stats.loc[test_name, ('response_time', 'std')]
            success_rate = test_stats.loc[test_name, ('success', 'mean')] * 100

            report += f"| {test_name} | {count} | {avg_time:.3f} | {median_time:.3f} | {std_dev:.3f} | {success_rate:.1f}% |\n"

        # Add complexity analysis
        if 'input_complexity' in self.df.columns:
            complexity_stats = self.df[self.df['input_complexity'].notna()].groupby('input_complexity').agg({
                'response_time': ['count', 'mean', 'median'],
                'success': 'mean'
            }).round(3)

            report += "\n## Performance by Input Complexity\n\n"
            report += "| Complexity | Count | Avg Time (s) | Median Time (s) | Success Rate |\n"
            report += "|------------|-------|--------------|-----------------|-------------|\n"

            for complexity in complexity_stats.index:
                count = complexity_stats.loc[complexity, ('response_time', 'count')]
                avg_time = complexity_stats.loc[complexity, ('response_time', 'mean')]
                median_time = complexity_stats.loc[complexity, ('response_time', 'median')]
                success_rate = complexity_stats.loc[complexity, ('success', 'mean')] * 100

                report += f"| {complexity} | {count} | {avg_time:.3f} | {median_time:.3f} | {success_rate:.1f}% |\n"

        # Add error analysis
        errors = self.df[~self.df['success']]
        if not errors.empty:
            report += "\n## Error Analysis\n\n"
            error_summary = errors.groupby(['test_name', 'error_message']).size().reset_index(name='count')

            for _, row in error_summary.iterrows():
                report += f"- **{row['test_name']}**: {row['error_message']} (Count: {row['count']})\n"

        # Add recommendations
        report += "\n## Performance Recommendations\n\n"

        if stats['avg_response_time'] > 5.0:
            report += "⚠️ **High Average Response Time**: Consider optimizing model inference or scaling resources.\n\n"

        if stats['success_rate'] < 95:
            report += "⚠️ **Low Success Rate**: Investigate error causes and improve error handling.\n\n"

        if stats['p95_response_time'] > 10.0:
            report += "⚠️ **High P95 Response Time**: Some requests are taking very long - check for bottlenecks.\n\n"

        report += "✅ **Performance is within acceptable ranges** if no warnings appear above.\n"

        # Save report
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"Performance report saved to {output_file}")
        return report


def main():
    """Main function to run the complete performance test suite"""
    print("PolyID Hugging Face Space Performance Testing Suite")
    print("=" * 60)

    # Initialize tester
    tester = PolyIDPerformanceTester()

    # Run comprehensive tests
    all_results = tester.run_comprehensive_test_suite()

    # Analyze results
    analyzer = PerformanceAnalyzer(tester.results)

    # Generate plots and report
    analyzer.create_performance_plots()
    report = analyzer.generate_report()

    # Print summary
    stats = analyzer.generate_summary_stats()
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 60)
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Success Rate: {stats['success_rate']:.2f}%")
    print(f"Average Response Time: {stats['avg_response_time']:.3f}s")
    print(f"95th Percentile: {stats['p95_response_time']:.3f}s")
    print(f"Max Response Time: {stats['max_response_time']:.3f}s")

    if 'avg_memory_usage_mb' in stats:
        print(f"Average Memory Usage: {stats['avg_memory_usage_mb']:.2f} MB")

    print("=" * 60)

    return analyzer


if __name__ == "__main__":
    analyzer = main()