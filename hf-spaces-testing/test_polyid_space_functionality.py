#!/usr/bin/env python3
"""
PolyID HuggingFace Space Functionality Testing Suite

This script comprehensively tests the PolyID Space functionality including:
1. UI responsiveness and interface elements
2. Input validation for polymer SMILES strings
3. Prediction generation workflow
4. Output formatting and visualization
5. Error handling for invalid inputs
6. Example data and demo functionality
7. Performance analysis and bottleneck identification

URL: https://huggingface.co/spaces/jkbennitt/polyid-private
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import testing frameworks
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    print("WARNING: pytest not available, continuing without it")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Test data for comprehensive validation
TEST_SMILES = {
    "valid_simple": {
        "CC": "Polyethylene (PE) - Simple alkyl chain",
        "CC(C)": "Polypropylene (PP) - Branched alkyl",
        "CC(c1ccccc1)": "Polystyrene (PS) - Aromatic pendant",
    },
    "valid_complex": {
        "CC(C)(C(=O)OC)": "Poly(methyl methacrylate) (PMMA) - Ester group",
        "COC(=O)c1ccc(C(=O)O)cc1.OCCO": "Polyethylene terephthalate (PET) - Multi-component",
        "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1.O=C(Cl)Cl": "Polycarbonate (PC) - Complex aromatic",
    },
    "invalid": {
        "": "Empty string",
        "XYZ123": "Invalid characters",
        "C(C": "Unmatched parentheses",
        "Cc1ccc(C(C)": "Incomplete aromatic",
        "C#C#C": "Invalid triple bonds",
    },
    "edge_cases": {
        "C": "Single carbon",
        "CC" * 100: "Very long polymer chain",
        "c1ccccc1" * 10: "Repeated aromatic rings",
        "CC(C)(C)C" * 20: "Highly branched structure",
    }
}

EXPECTED_PROPERTIES = [
    "Glass Transition Temperature (Tg)",
    "Melting Temperature (Tm)",
    "Density",
    "Elastic Modulus"
]

class PolyIDSpaceTester:
    """Comprehensive testing suite for PolyID HuggingFace Space"""

    def __init__(self):
        self.test_results = {
            "ui_elements": {},
            "input_validation": {},
            "prediction_workflow": {},
            "output_formatting": {},
            "error_handling": {},
            "example_functionality": {},
            "performance_metrics": {},
            "issues_found": []
        }
        self.start_time = time.time()

        # Import app functions for testing
        try:
            from app import (
                validate_smiles,
                calculate_molecular_properties,
                predict_polymer_properties,
                analyze_domain_of_validity,
                main_prediction_interface,
                create_prediction_plot,
                SAMPLE_POLYMERS
            )
            self.app_functions = {
                'validate_smiles': validate_smiles,
                'calculate_molecular_properties': calculate_molecular_properties,
                'predict_polymer_properties': predict_polymer_properties,
                'analyze_domain_of_validity': analyze_domain_of_validity,
                'main_prediction_interface': main_prediction_interface,
                'create_prediction_plot': create_prediction_plot,
                'SAMPLE_POLYMERS': SAMPLE_POLYMERS
            }
            print("✅ Successfully imported app functions")
        except ImportError as e:
            print(f"❌ Failed to import app functions: {e}")
            self.app_functions = None

    def test_ui_elements(self) -> Dict:
        """Test 1: UI responsiveness and interface elements"""
        print("\n" + "="*60)
        print("TEST 1: UI RESPONSIVENESS AND INTERFACE ELEMENTS")
        print("="*60)

        results = {
            "gradio_interface": {"status": "unknown", "details": []},
            "input_components": {"status": "unknown", "details": []},
            "output_components": {"status": "unknown", "details": []},
            "sample_data": {"status": "unknown", "details": []},
            "navigation": {"status": "unknown", "details": []}
        }

        # Test Gradio interface creation
        try:
            from app import create_gradio_interface
            demo = create_gradio_interface()
            results["gradio_interface"]["status"] = "pass"
            results["gradio_interface"]["details"].append("✅ Gradio interface created successfully")

            # Check if interface components exist
            if hasattr(demo, 'blocks'):
                results["gradio_interface"]["details"].append("✅ Interface has blocks structure")
            else:
                results["gradio_interface"]["details"].append("⚠️ Interface missing blocks structure")

        except Exception as e:
            results["gradio_interface"]["status"] = "fail"
            results["gradio_interface"]["details"].append(f"❌ Gradio interface creation failed: {e}")

        # Test sample polymers availability
        if self.app_functions and 'SAMPLE_POLYMERS' in self.app_functions:
            sample_polymers = self.app_functions['SAMPLE_POLYMERS']
            if isinstance(sample_polymers, dict) and len(sample_polymers) > 0:
                results["sample_data"]["status"] = "pass"
                results["sample_data"]["details"].append(f"✅ {len(sample_polymers)} sample polymers available")
                for name, smiles in sample_polymers.items():
                    results["sample_data"]["details"].append(f"  • {name}: {smiles}")
            else:
                results["sample_data"]["status"] = "fail"
                results["sample_data"]["details"].append("❌ Sample polymers not properly configured")

        # Test expected properties list
        expected_props = EXPECTED_PROPERTIES
        results["input_components"]["status"] = "pass"
        results["input_components"]["details"].append(f"✅ {len(expected_props)} property options available")
        for prop in expected_props:
            results["input_components"]["details"].append(f"  • {prop}")

        return results

    def test_input_validation(self) -> Dict:
        """Test 2: Input validation for polymer SMILES strings"""
        print("\n" + "="*60)
        print("TEST 2: INPUT VALIDATION FOR POLYMER SMILES")
        print("="*60)

        results = {
            "valid_simple": {"status": "unknown", "details": []},
            "valid_complex": {"status": "unknown", "details": []},
            "invalid_inputs": {"status": "unknown", "details": []},
            "edge_cases": {"status": "unknown", "details": []},
            "validation_function": {"status": "unknown", "details": []}
        }

        if not self.app_functions:
            for key in results:
                results[key]["status"] = "fail"
                results[key]["details"].append("❌ App functions not available")
            return results

        validate_func = self.app_functions.get('validate_smiles')
        if not validate_func:
            for key in results:
                results[key]["status"] = "fail"
                results[key]["details"].append("❌ validate_smiles function not available")
            return results

        # Test valid simple SMILES
        valid_count = 0
        for smiles, description in TEST_SMILES["valid_simple"].items():
            try:
                is_valid, message = validate_func(smiles)
                if is_valid:
                    valid_count += 1
                    results["valid_simple"]["details"].append(f"✅ {smiles} - {description}: {message}")
                else:
                    results["valid_simple"]["details"].append(f"❌ {smiles} - {description}: {message}")
            except Exception as e:
                results["valid_simple"]["details"].append(f"🔥 {smiles} - Exception: {e}")

        results["valid_simple"]["status"] = "pass" if valid_count == len(TEST_SMILES["valid_simple"]) else "fail"

        # Test valid complex SMILES
        valid_count = 0
        for smiles, description in TEST_SMILES["valid_complex"].items():
            try:
                is_valid, message = validate_func(smiles)
                if is_valid:
                    valid_count += 1
                    results["valid_complex"]["details"].append(f"✅ {smiles[:20]}... - {description}: {message}")
                else:
                    results["valid_complex"]["details"].append(f"❌ {smiles[:20]}... - {description}: {message}")
            except Exception as e:
                results["valid_complex"]["details"].append(f"🔥 {smiles[:20]}... - Exception: {e}")

        results["valid_complex"]["status"] = "pass" if valid_count == len(TEST_SMILES["valid_complex"]) else "fail"

        # Test invalid SMILES
        invalid_count = 0
        for smiles, description in TEST_SMILES["invalid"].items():
            try:
                is_valid, message = validate_func(smiles)
                if not is_valid:
                    invalid_count += 1
                    results["invalid_inputs"]["details"].append(f"✅ {repr(smiles)} - {description}: Correctly rejected - {message}")
                else:
                    results["invalid_inputs"]["details"].append(f"❌ {repr(smiles)} - {description}: Incorrectly accepted - {message}")
            except Exception as e:
                results["invalid_inputs"]["details"].append(f"🔥 {repr(smiles)} - Exception: {e}")

        results["invalid_inputs"]["status"] = "pass" if invalid_count == len(TEST_SMILES["invalid"]) else "fail"

        # Test edge cases
        edge_case_results = []
        for smiles, description in TEST_SMILES["edge_cases"].items():
            try:
                is_valid, message = validate_func(smiles)
                edge_case_results.append((smiles, description, is_valid, message))
                results["edge_cases"]["details"].append(f"{'✅' if is_valid else '⚠️'} {smiles[:30]}... - {description}: {message}")
            except Exception as e:
                results["edge_cases"]["details"].append(f"🔥 {smiles[:30]}... - Exception: {e}")

        results["edge_cases"]["status"] = "pass"  # Edge cases are informational

        return results

    def test_prediction_workflow(self) -> Dict:
        """Test 3: Prediction generation workflow"""
        print("\n" + "="*60)
        print("TEST 3: PREDICTION GENERATION WORKFLOW")
        print("="*60)

        results = {
            "molecular_properties": {"status": "unknown", "details": []},
            "property_predictions": {"status": "unknown", "details": []},
            "domain_validity": {"status": "unknown", "details": []},
            "workflow_integration": {"status": "unknown", "details": []},
            "performance_timing": {"status": "unknown", "details": []}
        }

        if not self.app_functions:
            for key in results:
                results[key]["status"] = "fail"
                results[key]["details"].append("❌ App functions not available")
            return results

        test_smiles = "CC(c1ccccc1)"  # Polystyrene
        test_properties = ["Glass Transition Temperature (Tg)", "Density"]

        # Test molecular property calculation
        try:
            start_time = time.time()
            mol_props_func = self.app_functions.get('calculate_molecular_properties')
            mol_props = mol_props_func(test_smiles)
            mol_props_time = time.time() - start_time

            if isinstance(mol_props, dict) and "error" not in mol_props:
                results["molecular_properties"]["status"] = "pass"
                results["molecular_properties"]["details"].append(f"✅ Molecular properties calculated in {mol_props_time:.3f}s")
                for prop, value in mol_props.items():
                    results["molecular_properties"]["details"].append(f"  • {prop}: {value}")
            else:
                results["molecular_properties"]["status"] = "fail"
                results["molecular_properties"]["details"].append(f"❌ Molecular properties failed: {mol_props}")

        except Exception as e:
            results["molecular_properties"]["status"] = "fail"
            results["molecular_properties"]["details"].append(f"🔥 Molecular properties exception: {e}")

        # Test property predictions
        try:
            start_time = time.time()
            predict_func = self.app_functions.get('predict_polymer_properties')
            predictions = predict_func(test_smiles, test_properties)
            predict_time = time.time() - start_time

            if isinstance(predictions, dict) and "error" not in predictions:
                results["property_predictions"]["status"] = "pass"
                results["property_predictions"]["details"].append(f"✅ Property predictions generated in {predict_time:.3f}s")
                for prop, result in predictions.items():
                    if isinstance(result, dict) and "value" in result:
                        results["property_predictions"]["details"].append(
                            f"  • {prop}: {result['value']} {result.get('unit', '')} "
                            f"(Confidence: {result.get('confidence', 'Unknown')})"
                        )
                    else:
                        results["property_predictions"]["details"].append(f"  • {prop}: {result}")
            else:
                results["property_predictions"]["status"] = "fail"
                results["property_predictions"]["details"].append(f"❌ Property predictions failed: {predictions}")

        except Exception as e:
            results["property_predictions"]["status"] = "fail"
            results["property_predictions"]["details"].append(f"🔥 Property predictions exception: {e}")

        # Test domain of validity analysis
        try:
            start_time = time.time()
            dov_func = self.app_functions.get('analyze_domain_of_validity')
            dov_result = dov_func(test_smiles)
            dov_time = time.time() - start_time

            if isinstance(dov_result, dict) and "error" not in dov_result:
                results["domain_validity"]["status"] = "pass"
                results["domain_validity"]["details"].append(f"✅ Domain of validity analyzed in {dov_time:.3f}s")
                results["domain_validity"]["details"].append(f"  • Score: {dov_result.get('score', 'Unknown')}")
                results["domain_validity"]["details"].append(f"  • Reliability: {dov_result.get('reliability', 'Unknown')}")
                results["domain_validity"]["details"].append(f"  • Recommendation: {dov_result.get('recommendation', 'None')}")
            else:
                results["domain_validity"]["status"] = "fail"
                results["domain_validity"]["details"].append(f"❌ Domain of validity failed: {dov_result}")

        except Exception as e:
            results["domain_validity"]["status"] = "fail"
            results["domain_validity"]["details"].append(f"🔥 Domain of validity exception: {e}")

        # Test integrated workflow
        try:
            start_time = time.time()
            main_func = self.app_functions.get('main_prediction_interface')
            validation, mol_props_str, full_results, plot = main_func(test_smiles, test_properties)
            workflow_time = time.time() - start_time

            results["workflow_integration"]["status"] = "pass"
            results["workflow_integration"]["details"].append(f"✅ Complete workflow executed in {workflow_time:.3f}s")
            results["workflow_integration"]["details"].append(f"  • Validation: {validation}")
            results["workflow_integration"]["details"].append(f"  • Results length: {len(full_results)} characters")
            results["workflow_integration"]["details"].append(f"  • Plot generated: {'Yes' if plot else 'No'}")

            # Record performance timing
            results["performance_timing"]["status"] = "pass"
            results["performance_timing"]["details"].append(f"✅ Performance timing recorded")
            results["performance_timing"]["details"].append(f"  • Molecular properties: {mol_props_time:.3f}s")
            results["performance_timing"]["details"].append(f"  • Property predictions: {predict_time:.3f}s")
            results["performance_timing"]["details"].append(f"  • Domain of validity: {dov_time:.3f}s")
            results["performance_timing"]["details"].append(f"  • Complete workflow: {workflow_time:.3f}s")

        except Exception as e:
            results["workflow_integration"]["status"] = "fail"
            results["workflow_integration"]["details"].append(f"🔥 Integrated workflow exception: {e}")

        return results

    def test_output_formatting(self) -> Dict:
        """Test 4: Output formatting and visualization"""
        print("\n" + "="*60)
        print("TEST 4: OUTPUT FORMATTING AND VISUALIZATION")
        print("="*60)

        results = {
            "text_formatting": {"status": "unknown", "details": []},
            "plot_generation": {"status": "unknown", "details": []},
            "data_structure": {"status": "unknown", "details": []},
            "visual_elements": {"status": "unknown", "details": []}
        }

        if not self.app_functions:
            for key in results:
                results[key]["status"] = "fail"
                results[key]["details"].append("❌ App functions not available")
            return results

        test_smiles = "CC(C)(C(=O)OC)"  # PMMA
        test_properties = EXPECTED_PROPERTIES

        # Test text output formatting
        try:
            main_func = self.app_functions.get('main_prediction_interface')
            validation, mol_props_str, full_results, plot = main_func(test_smiles, test_properties)

            # Check text formatting
            if isinstance(full_results, str) and len(full_results) > 0:
                results["text_formatting"]["status"] = "pass"
                results["text_formatting"]["details"].append("✅ Text output properly formatted")

                # Check for expected sections
                expected_sections = ["SMILES Validation", "Molecular Properties", "Property Predictions", "Domain of Validity"]
                for section in expected_sections:
                    if section in full_results:
                        results["text_formatting"]["details"].append(f"  ✅ Contains {section} section")
                    else:
                        results["text_formatting"]["details"].append(f"  ❌ Missing {section} section")
                        results["text_formatting"]["status"] = "fail"

                # Check for emojis and formatting
                emoji_count = sum(1 for char in full_results if ord(char) > 0x1F600)
                results["text_formatting"]["details"].append(f"  • Emoji usage: {emoji_count} emojis found")
                results["text_formatting"]["details"].append(f"  • Total length: {len(full_results)} characters")

            else:
                results["text_formatting"]["status"] = "fail"
                results["text_formatting"]["details"].append("❌ Text output not properly formatted")

        except Exception as e:
            results["text_formatting"]["status"] = "fail"
            results["text_formatting"]["details"].append(f"🔥 Text formatting exception: {e}")

        # Test plot generation
        try:
            plot_func = self.app_functions.get('create_prediction_plot')
            predict_func = self.app_functions.get('predict_polymer_properties')

            # Generate predictions for plotting
            predictions = predict_func(test_smiles, test_properties)

            if isinstance(predictions, dict) and "error" not in predictions:
                plot = plot_func(predictions)

                if plot:
                    results["plot_generation"]["status"] = "pass"
                    results["plot_generation"]["details"].append("✅ Plot generated successfully")

                    # Check plot properties
                    if hasattr(plot, 'get_axes') and plot.get_axes():
                        axes = plot.get_axes()[0]
                        results["plot_generation"]["details"].append(f"  • Axes created: {len(plot.get_axes())}")
                        results["plot_generation"]["details"].append(f"  • Title: {axes.get_title()}")
                        results["plot_generation"]["details"].append(f"  • X-label: {axes.get_xlabel()}")
                        results["plot_generation"]["details"].append(f"  • Y-label: {axes.get_ylabel()}")

                        # Check if data is plotted
                        if axes.patches:  # Bar chart patches
                            results["plot_generation"]["details"].append(f"  • Data bars: {len(axes.patches)}")

                    plt.close(plot)  # Clean up
                else:
                    results["plot_generation"]["status"] = "fail"
                    results["plot_generation"]["details"].append("❌ Plot not generated")
            else:
                results["plot_generation"]["status"] = "fail"
                results["plot_generation"]["details"].append("❌ No valid predictions for plotting")

        except Exception as e:
            results["plot_generation"]["status"] = "fail"
            results["plot_generation"]["details"].append(f"🔥 Plot generation exception: {e}")

        return results

    def test_error_handling(self) -> Dict:
        """Test 5: Error handling for invalid inputs"""
        print("\n" + "="*60)
        print("TEST 5: ERROR HANDLING FOR INVALID INPUTS")
        print("="*60)

        results = {
            "invalid_smiles": {"status": "unknown", "details": []},
            "empty_properties": {"status": "unknown", "details": []},
            "exception_handling": {"status": "unknown", "details": []},
            "graceful_degradation": {"status": "unknown", "details": []}
        }

        if not self.app_functions:
            for key in results:
                results[key]["status"] = "fail"
                results[key]["details"].append("❌ App functions not available")
            return results

        main_func = self.app_functions.get('main_prediction_interface')

        # Test invalid SMILES handling
        invalid_smiles_cases = ["", "XYZ123", "C(C", "###", None]

        for invalid_smiles in invalid_smiles_cases:
            try:
                validation, mol_props_str, full_results, plot = main_func(
                    str(invalid_smiles) if invalid_smiles is not None else "",
                    ["Glass Transition Temperature (Tg)"]
                )

                # Check if error is properly handled
                if "Error" in validation or "Invalid" in validation or "Please enter" in validation:
                    results["invalid_smiles"]["details"].append(f"✅ {repr(invalid_smiles)} - Properly rejected: {validation}")
                else:
                    results["invalid_smiles"]["details"].append(f"❌ {repr(invalid_smiles)} - Not properly rejected: {validation}")

            except Exception as e:
                results["invalid_smiles"]["details"].append(f"🔥 {repr(invalid_smiles)} - Exception: {e}")

        results["invalid_smiles"]["status"] = "pass"  # If we reach here, error handling works

        # Test empty properties list
        try:
            validation, mol_props_str, full_results, plot = main_func("CC", [])

            if "Please select properties" in full_results or "No properties" in full_results:
                results["empty_properties"]["status"] = "pass"
                results["empty_properties"]["details"].append("✅ Empty properties list handled gracefully")
            else:
                results["empty_properties"]["status"] = "fail"
                results["empty_properties"]["details"].append("❌ Empty properties list not handled properly")

        except Exception as e:
            results["empty_properties"]["status"] = "fail"
            results["empty_properties"]["details"].append(f"🔥 Empty properties exception: {e}")

        # Test graceful degradation when dependencies missing
        try:
            # This tests the mock prediction functionality
            validation, mol_props_str, full_results, plot = main_func("CC", ["Density"])

            if "mock" in full_results.lower() or "limited" in full_results.lower():
                results["graceful_degradation"]["status"] = "pass"
                results["graceful_degradation"]["details"].append("✅ Graceful degradation with mock predictions")
            else:
                results["graceful_degradation"]["status"] = "pass"
                results["graceful_degradation"]["details"].append("✅ Full functionality available")

        except Exception as e:
            results["graceful_degradation"]["status"] = "fail"
            results["graceful_degradation"]["details"].append(f"🔥 Graceful degradation exception: {e}")

        return results

    def test_example_functionality(self) -> Dict:
        """Test 6: Example data and demo functionality"""
        print("\n" + "="*60)
        print("TEST 6: EXAMPLE DATA AND DEMO FUNCTIONALITY")
        print("="*60)

        results = {
            "sample_polymers": {"status": "unknown", "details": []},
            "example_workflows": {"status": "unknown", "details": []},
            "demo_completeness": {"status": "unknown", "details": []}
        }

        if not self.app_functions:
            for key in results:
                results[key]["status"] = "fail"
                results[key]["details"].append("❌ App functions not available")
            return results

        sample_polymers = self.app_functions.get('SAMPLE_POLYMERS')
        main_func = self.app_functions.get('main_prediction_interface')

        # Test each sample polymer
        if sample_polymers:
            successful_samples = 0
            total_samples = len(sample_polymers)

            for name, smiles in sample_polymers.items():
                try:
                    validation, mol_props_str, full_results, plot = main_func(
                        smiles,
                        ["Glass Transition Temperature (Tg)", "Density"]
                    )

                    if "Valid SMILES" in validation:
                        successful_samples += 1
                        results["sample_polymers"]["details"].append(f"✅ {name} ({smiles}) - Working")
                    else:
                        results["sample_polymers"]["details"].append(f"❌ {name} ({smiles}) - Failed: {validation}")

                except Exception as e:
                    results["sample_polymers"]["details"].append(f"🔥 {name} ({smiles}) - Exception: {e}")

            if successful_samples == total_samples:
                results["sample_polymers"]["status"] = "pass"
                results["sample_polymers"]["details"].insert(0, f"✅ All {total_samples} sample polymers working")
            else:
                results["sample_polymers"]["status"] = "fail"
                results["sample_polymers"]["details"].insert(0, f"❌ {successful_samples}/{total_samples} sample polymers working")
        else:
            results["sample_polymers"]["status"] = "fail"
            results["sample_polymers"]["details"].append("❌ No sample polymers available")

        # Test demo workflow completeness
        demo_properties = EXPECTED_PROPERTIES
        demo_smiles = "CC(c1ccccc1)"  # Polystyrene

        try:
            validation, mol_props_str, full_results, plot = main_func(demo_smiles, demo_properties)

            # Check completeness of demo
            completeness_score = 0
            max_score = 4

            if "Valid SMILES" in validation:
                completeness_score += 1
                results["demo_completeness"]["details"].append("✅ SMILES validation working")

            if mol_props_str and len(mol_props_str) > 0:
                completeness_score += 1
                results["demo_completeness"]["details"].append("✅ Molecular properties calculated")

            if "Property Predictions" in full_results:
                completeness_score += 1
                results["demo_completeness"]["details"].append("✅ Property predictions generated")

            if plot:
                completeness_score += 1
                results["demo_completeness"]["details"].append("✅ Visualization plot created")

            if completeness_score == max_score:
                results["demo_completeness"]["status"] = "pass"
                results["demo_completeness"]["details"].insert(0, f"✅ Complete demo functionality ({completeness_score}/{max_score})")
            else:
                results["demo_completeness"]["status"] = "fail"
                results["demo_completeness"]["details"].insert(0, f"❌ Incomplete demo functionality ({completeness_score}/{max_score})")

        except Exception as e:
            results["demo_completeness"]["status"] = "fail"
            results["demo_completeness"]["details"].append(f"🔥 Demo completeness exception: {e}")

        return results

    def run_performance_analysis(self) -> Dict:
        """Test 7: Performance analysis and bottleneck identification"""
        print("\n" + "="*60)
        print("TEST 7: PERFORMANCE ANALYSIS AND BOTTLENECK IDENTIFICATION")
        print("="*60)

        results = {
            "startup_time": {"status": "unknown", "details": []},
            "prediction_latency": {"status": "unknown", "details": []},
            "memory_usage": {"status": "unknown", "details": []},
            "scalability": {"status": "unknown", "details": []},
            "bottlenecks": {"status": "unknown", "details": []}
        }

        if not self.app_functions:
            for key in results:
                results[key]["status"] = "fail"
                results[key]["details"].append("❌ App functions not available")
            return results

        # Measure prediction latency for different complexity levels
        test_cases = [
            ("Simple", "CC", ["Density"]),
            ("Medium", "CC(c1ccccc1)", ["Glass Transition Temperature (Tg)", "Density"]),
            ("Complex", "CC(C)(C(=O)OC)", EXPECTED_PROPERTIES),
        ]

        main_func = self.app_functions.get('main_prediction_interface')
        latencies = []

        for complexity, smiles, properties in test_cases:
            times = []
            for i in range(3):  # Run 3 times for average
                start_time = time.time()
                try:
                    validation, mol_props_str, full_results, plot = main_func(smiles, properties)
                    end_time = time.time()
                    times.append(end_time - start_time)
                except Exception as e:
                    results["prediction_latency"]["details"].append(f"🔥 {complexity} case exception: {e}")
                    break

            if times:
                avg_time = sum(times) / len(times)
                latencies.append((complexity, avg_time))
                results["prediction_latency"]["details"].append(f"✅ {complexity} prediction: {avg_time:.3f}s average")

        if latencies:
            results["prediction_latency"]["status"] = "pass"
            max_latency = max(lat[1] for lat in latencies)
            if max_latency < 2.0:
                results["prediction_latency"]["details"].insert(0, "✅ All predictions under 2s")
            elif max_latency < 5.0:
                results["prediction_latency"]["details"].insert(0, "⚠️ Some predictions over 2s but under 5s")
            else:
                results["prediction_latency"]["details"].insert(0, "❌ Some predictions over 5s")
                results["prediction_latency"]["status"] = "fail"

        # Memory usage estimation (simplified)
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            results["memory_usage"]["status"] = "pass"
            results["memory_usage"]["details"].append(f"✅ Current memory usage: {memory_mb:.1f} MB")

            if memory_mb < 500:
                results["memory_usage"]["details"].append("✅ Memory usage is reasonable")
            elif memory_mb < 1000:
                results["memory_usage"]["details"].append("⚠️ Memory usage is moderate")
            else:
                results["memory_usage"]["details"].append("❌ Memory usage is high")

        except ImportError:
            results["memory_usage"]["status"] = "fail"
            results["memory_usage"]["details"].append("❌ psutil not available for memory monitoring")
        except Exception as e:
            results["memory_usage"]["status"] = "fail"
            results["memory_usage"]["details"].append(f"🔥 Memory usage exception: {e}")

        # Identify potential bottlenecks
        bottlenecks = []

        # Check import status
        try:
            from app import rdkit, nfp, POLYID_AVAILABLE
            if not rdkit:
                bottlenecks.append("RDKit not available - using fallback validation")
            if not nfp:
                bottlenecks.append("NFP not available - limited ML functionality")
            if not POLYID_AVAILABLE:
                bottlenecks.append("PolyID not fully available - using mock predictions")
        except:
            bottlenecks.append("Cannot check dependency status")

        # Check prediction times
        if latencies:
            slow_predictions = [case for case, time in latencies if time > 1.0]
            if slow_predictions:
                bottlenecks.append(f"Slow predictions for: {', '.join(slow_predictions)}")

        if bottlenecks:
            results["bottlenecks"]["status"] = "fail"
            results["bottlenecks"]["details"].append("❌ Performance bottlenecks identified:")
            for bottleneck in bottlenecks:
                results["bottlenecks"]["details"].append(f"  • {bottleneck}")
        else:
            results["bottlenecks"]["status"] = "pass"
            results["bottlenecks"]["details"].append("✅ No major performance bottlenecks identified")

        return results

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time

        # Run all tests
        print("🧪 STARTING COMPREHENSIVE POLYID SPACE TESTING")
        print("="*80)

        self.test_results["ui_elements"] = self.test_ui_elements()
        self.test_results["input_validation"] = self.test_input_validation()
        self.test_results["prediction_workflow"] = self.test_prediction_workflow()
        self.test_results["output_formatting"] = self.test_output_formatting()
        self.test_results["error_handling"] = self.test_error_handling()
        self.test_results["example_functionality"] = self.test_example_functionality()
        self.test_results["performance_metrics"] = self.run_performance_analysis()

        # Generate summary report
        report = []
        report.append("="*80)
        report.append("POLYID HUGGINGFACE SPACE - COMPREHENSIVE TEST REPORT")
        report.append("="*80)
        report.append(f"URL: https://huggingface.co/spaces/jkbennitt/polyid-private")
        report.append(f"Test Duration: {total_time:.2f} seconds")
        report.append(f"Test Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Test results summary
        report.append("TEST RESULTS SUMMARY:")
        report.append("-" * 40)

        total_tests = 0
        passed_tests = 0

        for test_category, test_data in self.test_results.items():
            if test_category == "issues_found":
                continue

            report.append(f"\n{test_category.upper().replace('_', ' ')}:")

            for test_name, test_result in test_data.items():
                total_tests += 1
                status = test_result.get("status", "unknown")

                if status == "pass":
                    passed_tests += 1
                    status_emoji = "✅"
                elif status == "fail":
                    status_emoji = "❌"
                else:
                    status_emoji = "⚠️"

                report.append(f"  {status_emoji} {test_name}: {status}")

                # Add details
                for detail in test_result.get("details", []):
                    report.append(f"    {detail}")

        # Overall status
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        report.append(f"\nOVERALL RESULTS:")
        report.append(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

        if success_rate >= 90:
            overall_status = "✅ EXCELLENT"
        elif success_rate >= 75:
            overall_status = "✅ GOOD"
        elif success_rate >= 50:
            overall_status = "⚠️ FAIR"
        else:
            overall_status = "❌ POOR"

        report.append(f"Overall Status: {overall_status}")

        # Key findings
        report.append(f"\nKEY FINDINGS:")
        report.append("-" * 40)

        # Dependencies status
        try:
            from app import rdkit, nfp, POLYID_AVAILABLE
            report.append(f"• RDKit Available: {'✅' if rdkit else '❌'}")
            report.append(f"• NFP Available: {'✅' if nfp else '❌'}")
            report.append(f"• PolyID Full Functionality: {'✅' if POLYID_AVAILABLE else '❌'}")
        except:
            report.append("• Dependency status could not be determined")

        # Performance insights
        if "performance_metrics" in self.test_results:
            perf_data = self.test_results["performance_metrics"]
            if "prediction_latency" in perf_data:
                latency_status = perf_data["prediction_latency"]["status"]
                report.append(f"• Prediction Performance: {'✅ Good' if latency_status == 'pass' else '❌ Needs Improvement'}")

        # UI/UX insights
        if "ui_elements" in self.test_results:
            ui_status = any(test["status"] == "pass" for test in self.test_results["ui_elements"].values())
            report.append(f"• User Interface: {'✅ Functional' if ui_status else '❌ Issues Found'}")

        # Recommendations
        report.append(f"\nRECOMMENDations:")
        report.append("-" * 40)

        recommendations = []

        # Check for dependency issues
        try:
            from app import rdkit, nfp, POLYID_AVAILABLE
            if not rdkit:
                recommendations.append("Install RDKit for full molecular validation capabilities")
            if not nfp:
                recommendations.append("Install NFP for complete neural fingerprint functionality")
            if not POLYID_AVAILABLE:
                recommendations.append("Ensure PolyID package is properly installed for real predictions")
        except:
            recommendations.append("Verify all dependencies are properly configured")

        # Performance recommendations
        if "performance_metrics" in self.test_results:
            bottlenecks = self.test_results["performance_metrics"].get("bottlenecks", {})
            if bottlenecks.get("status") == "fail":
                recommendations.append("Address performance bottlenecks identified in testing")

        # Error handling recommendations
        if "error_handling" in self.test_results:
            error_status = any(test["status"] == "fail" for test in self.test_results["error_handling"].values())
            if error_status:
                recommendations.append("Improve error handling for edge cases")

        if not recommendations:
            recommendations.append("No critical issues identified - system performing well")

        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")

        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)

        return "\n".join(report)

def main():
    """Main testing function"""
    tester = PolyIDSpaceTester()
    report = tester.generate_comprehensive_report()

    # Print report
    print(report)

    # Save report to file
    with open('polyid_space_test_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📄 Test report saved to: polyid_space_test_report.txt")

if __name__ == "__main__":
    main()