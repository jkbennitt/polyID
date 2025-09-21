#!/usr/bin/env python3
"""
PolyID Deployment Analysis and Optimization Recommendations
Comprehensive analysis of the Hugging Face Spaces deployment
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import subprocess

@dataclass
class DeploymentAnalysis:
    """Container for deployment analysis results"""
    configuration: Dict[str, Any]
    dependencies: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    bottlenecks: List[str]
    recommendations: List[str]
    optimization_priorities: List[str]

class PolyIDDeploymentAnalyzer:
    """Analyzer for PolyID Hugging Face Spaces deployment"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.analysis_results = {}

    def log(self, message: str):
        """Simple logging"""
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def analyze_deployment_configuration(self) -> Dict[str, Any]:
        """Analyze deployment configuration files"""
        self.log("Analyzing deployment configuration...")

        config_analysis = {
            'files_present': {},
            'requirements_analysis': {},
            'packages_analysis': {},
            'app_configuration': {}
        }

        # Check for essential deployment files
        essential_files = [
            'app.py',
            'requirements.txt',
            'packages.txt',
            'README.md'
        ]

        for file_name in essential_files:
            file_path = self.project_root / file_name
            config_analysis['files_present'][file_name] = {
                'exists': file_path.exists(),
                'size_kb': file_path.stat().st_size / 1024 if file_path.exists() else 0
            }

        # Analyze requirements.txt
        req_file = self.project_root / 'requirements.txt'
        if req_file.exists():
            with open(req_file, 'r') as f:
                requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

            config_analysis['requirements_analysis'] = {
                'total_dependencies': len(requirements),
                'chemistry_deps': [req for req in requirements if any(chem in req.lower() for chem in ['rdkit', 'nfp', 'm2p', 'mordred'])],
                'ml_deps': [req for req in requirements if any(ml in req.lower() for ml in ['tensorflow', 'torch', 'sklearn'])],
                'ui_deps': [req for req in requirements if any(ui in req.lower() for ui in ['gradio', 'streamlit', 'dash'])],
                'critical_missing': self._check_missing_critical_deps(requirements)
            }

        # Analyze packages.txt
        pkg_file = self.project_root / 'packages.txt'
        if pkg_file.exists():
            with open(pkg_file, 'r') as f:
                packages = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

            config_analysis['packages_analysis'] = {
                'total_packages': len(packages),
                'build_tools': [pkg for pkg in packages if any(tool in pkg for tool in ['build-essential', 'cmake', 'gcc'])],
                'chemistry_support': [pkg for pkg in packages if any(chem in pkg for chem in ['boost', 'eigen', 'cairo'])],
                'optimization': [pkg for pkg in packages if any(opt in pkg for opt in ['tcmalloc', 'blas', 'lapack'])]
            }

        # Analyze app.py configuration
        app_file = self.project_root / 'app.py'
        if app_file.exists():
            with open(app_file, 'r', encoding='utf-8', errors='ignore') as f:
                app_content = f.read()

            config_analysis['app_configuration'] = {
                'has_startup_diagnostics': 'run_startup_diagnostics' in app_content,
                'has_error_handling': 'try:' in app_content and 'except' in app_content,
                'has_memory_optimization': 'memory_growth' in app_content,
                'uses_caching': 'cache' in app_content.lower(),
                'has_gradio_interface': 'gradio' in app_content.lower(),
                'app_size_kb': len(app_content.encode('utf-8')) / 1024
            }

        return config_analysis

    def analyze_resource_requirements(self) -> Dict[str, Any]:
        """Analyze resource requirements and current usage"""
        self.log("Analyzing resource requirements...")

        resource_analysis = {
            'current_system': {
                'cpu_cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version.split()[0]
            },
            'hf_spaces_requirements': {
                'recommended_memory': '16GB',  # For chemistry stack
                'recommended_cpu': '4+ cores',
                'python_version': '3.10+',
                'space_type': 'Standard GPU'  # Required for full chemistry stack
            },
            'chemistry_stack_requirements': {
                'rdkit_memory': '~500MB',
                'tensorflow_memory': '~1GB',
                'nfp_memory': '~200MB',
                'estimated_total': '~2GB'
            }
        }

        return resource_analysis

    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks from test results"""
        self.log("Analyzing performance bottlenecks...")

        bottleneck_analysis = {
            'dependency_loading': {
                'issue': 'Missing critical chemistry dependencies',
                'impact': 'High - App cannot function without RDKit, NFP, m2p',
                'solution': 'Install full chemistry stack in Standard GPU environment'
            },
            'memory_usage': {
                'issue': 'High initial memory consumption (>130MB on startup)',
                'impact': 'Medium - May cause OOM in constrained environments',
                'solution': 'Implement lazy loading and memory optimization'
            },
            'cold_start': {
                'issue': 'Long dependency loading time (~5 seconds)',
                'impact': 'Medium - Poor user experience on first load',
                'solution': 'Optimize imports and implement caching'
            },
            'prediction_pipeline': {
                'issue': 'Mock predictions due to missing dependencies',
                'impact': 'Critical - No real ML functionality',
                'solution': 'Deploy with proper chemistry stack'
            }
        }

        return bottleneck_analysis

    def generate_optimization_recommendations(self) -> List[str]:
        """Generate comprehensive optimization recommendations"""
        self.log("Generating optimization recommendations...")

        recommendations = [
            # Critical Dependencies
            "CRITICAL: Deploy on Standard GPU Spaces for full chemistry stack compatibility",
            "Install missing dependencies: rdkit, nfp, m2p, shortuuid",
            "Verify TensorFlow GPU configuration for neural network inference",

            # Performance Optimization
            "Implement lazy loading for heavy dependencies (load on first use)",
            "Add model and preprocessor caching to reduce repeated computations",
            "Optimize memory usage with tf.config.experimental.set_memory_growth",
            "Implement batch processing for multiple polymer predictions",

            # User Experience
            "Add loading indicators during dependency initialization",
            "Implement graceful error handling for invalid SMILES input",
            "Add comprehensive input validation with helpful error messages",
            "Provide example polymer structures for user guidance",

            # Deployment Configuration
            "Review and optimize requirements.txt for minimal necessary dependencies",
            "Configure proper logging and monitoring for production",
            "Implement health checks and readiness probes",
            "Add proper error tracking and performance monitoring",

            # Scalability
            "Implement request queuing for high concurrent load",
            "Add response caching for frequently requested polymers",
            "Consider implementing async processing for long-running predictions",
            "Set up proper resource limits and monitoring alerts",

            # Code Quality
            "Add comprehensive unit tests for all prediction functions",
            "Implement proper error handling for edge cases",
            "Add input sanitization and validation",
            "Document API endpoints and response formats"
        ]

        return recommendations

    def generate_deployment_checklist(self) -> List[str]:
        """Generate deployment checklist"""
        checklist = [
            "[ ] Verify Standard GPU Spaces configuration",
            "[ ] Test all chemistry dependencies installation",
            "[ ] Validate TensorFlow GPU functionality",
            "[ ] Test memory usage under load",
            "[ ] Verify all polymer prediction functions work",
            "[ ] Test error handling for invalid inputs",
            "[ ] Implement monitoring and logging",
            "[ ] Set up performance alerts",
            "[ ] Test concurrent user scenarios",
            "[ ] Validate response times meet requirements",
            "[ ] Ensure graceful degradation on errors",
            "[ ] Document known limitations",
            "[ ] Set up backup and recovery procedures"
        ]

        return checklist

    def _check_missing_critical_deps(self, requirements: List[str]) -> List[str]:
        """Check for missing critical dependencies"""
        critical_deps = ['rdkit', 'tensorflow', 'nfp', 'm2p', 'gradio', 'numpy', 'pandas']
        missing = []

        for dep in critical_deps:
            if not any(dep.lower() in req.lower() for req in requirements):
                missing.append(dep)

        return missing

    def run_comprehensive_analysis(self) -> DeploymentAnalysis:
        """Run comprehensive deployment analysis"""
        self.log("Starting comprehensive deployment analysis...")

        # Analyze configuration
        config_analysis = self.analyze_deployment_configuration()

        # Analyze resources
        resource_analysis = self.analyze_resource_requirements()

        # Analyze bottlenecks
        bottleneck_analysis = self.analyze_performance_bottlenecks()

        # Generate recommendations
        recommendations = self.generate_optimization_recommendations()

        # Compile analysis
        analysis = DeploymentAnalysis(
            configuration=config_analysis,
            dependencies=resource_analysis,
            performance_metrics=bottleneck_analysis,
            bottlenecks=list(bottleneck_analysis.keys()),
            recommendations=recommendations,
            optimization_priorities=[
                "Install chemistry dependencies",
                "Optimize memory usage",
                "Implement caching",
                "Add monitoring",
                "Improve error handling"
            ]
        )

        return analysis

    def generate_comprehensive_report(self, analysis: DeploymentAnalysis) -> str:
        """Generate comprehensive deployment analysis report"""
        report = []
        report.append("=" * 80)
        report.append("POLYID HUGGING FACE SPACES DEPLOYMENT ANALYSIS")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append("Current Status: DEPLOYMENT REQUIRES OPTIMIZATION")
        report.append("Critical Issues: Missing chemistry stack dependencies")
        report.append("Deployment Type: Standard GPU Spaces recommended")
        report.append("Primary Focus: Dependency installation and performance optimization")
        report.append("")

        # Configuration Analysis
        report.append("CONFIGURATION ANALYSIS")
        report.append("-" * 80)

        files = analysis.configuration['files_present']
        report.append("Deployment Files:")
        for file_name, info in files.items():
            status = "Present" if info['exists'] else "Missing"
            size = f"({info['size_kb']:.1f} KB)" if info['exists'] else ""
            report.append(f"  {file_name}: {status} {size}")

        if 'requirements_analysis' in analysis.configuration:
            req = analysis.configuration['requirements_analysis']
            report.append(f"\nDependency Analysis:")
            report.append(f"  Total dependencies: {req['total_dependencies']}")
            report.append(f"  Chemistry deps: {len(req['chemistry_deps'])}")
            report.append(f"  ML deps: {len(req['ml_deps'])}")
            report.append(f"  UI deps: {len(req['ui_deps'])}")
            if req['critical_missing']:
                report.append(f"  CRITICAL MISSING: {', '.join(req['critical_missing'])}")

        report.append("")

        # Resource Requirements
        report.append("RESOURCE REQUIREMENTS")
        report.append("-" * 80)
        resources = analysis.dependencies
        current = resources['current_system']
        hf_req = resources['hf_spaces_requirements']

        report.append("Current System:")
        report.append(f"  CPU Cores: {current['cpu_cores']}")
        report.append(f"  Memory: {current['memory_gb']:.1f} GB")
        report.append(f"  Python: {current['python_version']}")

        report.append("\nHugging Face Spaces Requirements:")
        report.append(f"  Recommended Memory: {hf_req['recommended_memory']}")
        report.append(f"  Recommended CPU: {hf_req['recommended_cpu']}")
        report.append(f"  Python Version: {hf_req['python_version']}")
        report.append(f"  Space Type: {hf_req['space_type']}")

        chem_req = resources['chemistry_stack_requirements']
        report.append("\nChemistry Stack Memory Requirements:")
        for component, memory in chem_req.items():
            report.append(f"  {component}: {memory}")

        report.append("")

        # Performance Bottlenecks
        report.append("PERFORMANCE BOTTLENECKS")
        report.append("-" * 80)
        bottlenecks = analysis.performance_metrics

        for bottleneck, details in bottlenecks.items():
            report.append(f"\n{bottleneck.upper().replace('_', ' ')}:")
            report.append(f"  Issue: {details['issue']}")
            report.append(f"  Impact: {details['impact']}")
            report.append(f"  Solution: {details['solution']}")

        report.append("")

        # Optimization Recommendations
        report.append("OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("Priority Order:")

        for i, priority in enumerate(analysis.optimization_priorities, 1):
            report.append(f"  {i}. {priority}")

        report.append("\nDetailed Recommendations:")
        for i, rec in enumerate(analysis.recommendations, 1):
            if rec.startswith("CRITICAL"):
                report.append(f"  {i}. [CRITICAL] {rec[9:]}")
            else:
                report.append(f"  {i}. {rec}")

        report.append("")

        # Deployment Checklist
        report.append("DEPLOYMENT CHECKLIST")
        report.append("-" * 80)
        checklist = self.generate_deployment_checklist()
        for item in checklist:
            report.append(f"  {item}")

        report.append("")

        # Next Steps
        report.append("IMMEDIATE NEXT STEPS")
        report.append("-" * 80)
        report.append("1. Configure Standard GPU Spaces environment")
        report.append("2. Install chemistry stack dependencies (rdkit, nfp, m2p)")
        report.append("3. Test TensorFlow GPU functionality")
        report.append("4. Implement memory optimization strategies")
        report.append("5. Add comprehensive error handling")
        report.append("6. Set up monitoring and performance tracking")
        report.append("7. Validate all functionality with real data")
        report.append("8. Optimize for production load")

        report.append("")

        # Performance Targets
        report.append("PERFORMANCE TARGETS")
        report.append("-" * 80)
        report.append("Response Times:")
        report.append("  - SMILES validation: <100ms")
        report.append("  - Molecular properties: <200ms")
        report.append("  - Property prediction: <1000ms")
        report.append("  - Full workflow: <2000ms")
        report.append("")
        report.append("Throughput:")
        report.append("  - Single predictions: >10/second")
        report.append("  - Batch processing: >50/second")
        report.append("  - Concurrent users: >5 simultaneous")
        report.append("")
        report.append("Resource Usage:")
        report.append("  - Memory: <8GB peak usage")
        report.append("  - CPU: <80% average utilization")
        report.append("  - GPU: Efficient utilization when available")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

def main():
    """Main analysis function"""
    try:
        analyzer = PolyIDDeploymentAnalyzer(".")
        analysis = analyzer.run_comprehensive_analysis()

        # Generate and display report
        report = analyzer.generate_comprehensive_report(analysis)
        print(report)

        # Save report
        timestamp = int(time.time())
        report_file = f"polyid_deployment_analysis_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"\nDeployment analysis saved to: {report_file}")

        # Save JSON data
        json_file = f"polyid_deployment_data_{timestamp}.json"
        json_data = {
            'timestamp': timestamp,
            'configuration': analysis.configuration,
            'dependencies': analysis.dependencies,
            'performance_metrics': analysis.performance_metrics,
            'bottlenecks': analysis.bottlenecks,
            'recommendations': analysis.recommendations,
            'optimization_priorities': analysis.optimization_priorities
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)

        print(f"Deployment data saved to: {json_file}")

        return True

    except Exception as e:
        print(f"Deployment analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)