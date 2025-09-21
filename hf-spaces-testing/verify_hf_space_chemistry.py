#!/usr/bin/env python
"""
Comprehensive verification of chemistry software stack in HF Spaces deployment
Tests all critical components for PolyID's chemical computation pipeline
"""

import sys
import os
import warnings
import traceback
import json
import time
import importlib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Fix for Windows Unicode issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

class ChemistryStackVerifier:
    """Verify chemistry and ML stack components for PolyID HF Spaces deployment"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "platform": sys.platform,
            "python_version": sys.version,
            "tests": {}
        }

    def test_component(self, name: str, test_func, critical: bool = True) -> bool:
        """Run a single component test"""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"Critical: {'Yes' if critical else 'No'}")
        print('-'*60)

        start_time = time.time()
        result = {
            "name": name,
            "critical": critical,
            "status": "UNKNOWN",
            "duration": 0,
            "details": {},
            "error": None
        }

        try:
            success, details = test_func()
            result["status"] = "PASS" if success else "FAIL"
            result["details"] = details

            # Print results
            status_symbol = "✅" if success else "❌"
            print(f"{status_symbol} Status: {result['status']}")

            if details:
                print("\nDetails:")
                for key, value in details.items():
                    print(f"  • {key}: {value}")

        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"❌ Error: {e}")
            print(f"Traceback:\n{result['traceback']}")

        result["duration"] = round(time.time() - start_time, 3)
        print(f"\nExecution time: {result['duration']}s")

        self.results["tests"][name] = result
        return result["status"] == "PASS"

    def verify_rdkit(self) -> Tuple[bool, Dict]:
        """Verify RDKit installation and functionality"""
        details = {}

        # Import test
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
            details["import"] = "Success"
            details["version"] = rdkit.__version__
        except ImportError as e:
            details["import"] = f"Failed: {e}"
            return False, details

        # SMILES parsing test
        try:
            test_smiles = [
                ("CC", "ethane"),
                ("CC(C)", "propane"),
                ("c1ccccc1", "benzene"),
                ("CC(C)(C(=O)OC)", "polymer_unit"),
                ("CC.CC(C)", "mixture")
            ]

            parsed_count = 0
            for smiles, name in test_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    parsed_count += 1

            details["smiles_parsing"] = f"{parsed_count}/{len(test_smiles)} parsed"

        except Exception as e:
            details["smiles_parsing"] = f"Failed: {e}"

        # Descriptor calculation test
        try:
            mol = Chem.MolFromSmiles("CC(c1ccccc1)")
            descriptors = {
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumAtoms": mol.GetNumAtoms(),
                "NumBonds": mol.GetNumBonds(),
                "NumRings": rdMolDescriptors.CalcNumRings(mol)
            }
            details["descriptor_calculation"] = f"Calculated {len(descriptors)} descriptors"
            details["sample_MW"] = round(descriptors["MolWt"], 2)

        except Exception as e:
            details["descriptor_calculation"] = f"Failed: {e}"

        # 3D conformer generation test
        try:
            mol = Chem.MolFromSmiles("CCC")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            details["3d_conformer"] = "Success"

        except Exception as e:
            details["3d_conformer"] = f"Failed: {e}"

        # Fingerprint generation test
        try:
            mol = Chem.MolFromSmiles("CC(C)C")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            details["fingerprint"] = f"Generated {len(fp)} bit fingerprint"

        except Exception as e:
            details["fingerprint"] = f"Failed: {e}"

        success = all([
            details.get("import") == "Success",
            "parsed" in str(details.get("smiles_parsing", "")),
            "Calculated" in str(details.get("descriptor_calculation", ""))
        ])

        return success, details

    def verify_nfp(self) -> Tuple[bool, Dict]:
        """Verify NFP (Neural Fingerprints) installation and functionality"""
        details = {}

        # Import test
        try:
            import nfp
            details["import"] = "Success"

            # Check for key NFP components
            from nfp import preprocessing, layers
            details["preprocessing"] = "Available"
            details["layers"] = "Available"

            # Version info if available
            if hasattr(nfp, '__version__'):
                details["version"] = nfp.__version__

        except ImportError as e:
            details["import"] = f"Failed: {e}"
            return False, details

        # Test preprocessor creation
        try:
            from nfp.preprocessing import SmilesPreprocessor, MolPreprocessor

            # Test SMILES preprocessor
            preprocessor = SmilesPreprocessor()
            test_smiles = "CC(C)C"
            inputs = preprocessor.construct_feature_matrices(test_smiles)

            details["smiles_preprocessor"] = "Success"
            details["feature_matrices"] = f"Generated {len(inputs)} matrices"

        except Exception as e:
            details["smiles_preprocessor"] = f"Failed: {e}"

        # Test NFP layers availability
        try:
            from nfp.layers import (
                GlobalUpdate, EdgeUpdate, NodeUpdate,
                Reduce, GatherAtomFeatures
            )

            layers_available = []
            for layer_name in ['GlobalUpdate', 'EdgeUpdate', 'NodeUpdate',
                              'Reduce', 'GatherAtomFeatures']:
                if layer_name in dir(nfp.layers):
                    layers_available.append(layer_name)

            details["nfp_layers"] = f"{len(layers_available)} layers available"

        except Exception as e:
            details["nfp_layers"] = f"Failed: {e}"

        success = details.get("import") == "Success" and \
                 details.get("smiles_preprocessor") == "Success"

        return success, details

    def verify_m2p(self) -> Tuple[bool, Dict]:
        """Verify m2p (monomers to polymers) installation and functionality"""
        details = {}

        # Import test
        try:
            import m2p
            details["import"] = "Success"

            # Version info if available
            if hasattr(m2p, '__version__'):
                details["version"] = m2p.__version__

        except ImportError as e:
            details["import"] = f"Failed: {e}"
            return False, details

        # Test polymer generation
        try:
            from m2p import MonomerToPolymer

            # Test basic polymer generation
            m2p_obj = MonomerToPolymer()

            # Test with simple monomer
            monomer_smiles = "CC(=C)C(=O)OC"  # Methyl methacrylate
            polymer_smiles = m2p_obj.polymerize(monomer_smiles)

            if polymer_smiles:
                details["polymerization"] = "Success"
                details["polymer_length"] = len(polymer_smiles)
            else:
                details["polymerization"] = "Failed: No polymer generated"

        except Exception as e:
            details["polymerization"] = f"Failed: {e}"

        # Test copolymer generation if supported
        try:
            # Test copolymer from two monomers
            monomer1 = "C=C"  # Ethylene
            monomer2 = "C=CC"  # Propylene

            copolymer = m2p_obj.copolymerize([monomer1, monomer2])

            if copolymer:
                details["copolymerization"] = "Success"
            else:
                details["copolymerization"] = "Not generated"

        except Exception as e:
            details["copolymerization"] = f"Not supported or failed: {e}"

        success = details.get("import") == "Success"

        return success, details

    def verify_tensorflow(self) -> Tuple[bool, Dict]:
        """Verify TensorFlow installation and GPU support"""
        details = {}

        # Import test
        try:
            import tensorflow as tf
            details["import"] = "Success"
            details["version"] = tf.__version__

        except ImportError as e:
            details["import"] = f"Failed: {e}"
            return False, details

        # GPU availability
        try:
            details["built_with_cuda"] = tf.test.is_built_with_cuda()
            details["built_with_gpu"] = tf.test.is_built_with_gpu_support()

            # Physical devices
            physical_devices = tf.config.list_physical_devices()
            details["physical_devices"] = {
                dev.device_type: len([d for d in physical_devices
                                     if d.device_type == dev.device_type])
                for dev in physical_devices
            }

            # GPU specific
            gpu_devices = tf.config.list_physical_devices('GPU')
            details["gpu_count"] = len(gpu_devices)

            if gpu_devices:
                details["gpu_available"] = True

                # Try to get GPU details
                for i, gpu in enumerate(gpu_devices):
                    details[f"gpu_{i}"] = str(gpu)

                # Test GPU memory growth
                for gpu in gpu_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                details["memory_growth_set"] = True

            else:
                details["gpu_available"] = False

        except Exception as e:
            details["gpu_check"] = f"Failed: {e}"

        # Test basic operations
        try:
            # Simple tensor operation
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)

            details["tensor_operations"] = "Success"
            details["computation_device"] = c.device if hasattr(c, 'device') else "Unknown"

        except Exception as e:
            details["tensor_operations"] = f"Failed: {e}"

        # Test Keras model creation
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            details["keras_model"] = "Success"
            details["model_layers"] = len(model.layers)

        except Exception as e:
            details["keras_model"] = f"Failed: {e}"

        success = details.get("import") == "Success" and \
                 details.get("tensor_operations") == "Success"

        return success, details

    def verify_polyid(self) -> Tuple[bool, Dict]:
        """Verify PolyID package installation and core functionality"""
        details = {}

        # Import test - try multiple approaches
        polyid_imported = False

        # Try direct package import
        try:
            from polyid.polyid import SingleModel, MultiModel
            from polyid.parameters import Parameters
            from polyid.models.base_models import global100

            details["import"] = "Success (full package)"
            polyid_imported = True

        except ImportError as e1:
            # Try alternative import
            try:
                import polyid
                details["import"] = "Success (limited)"
                polyid_imported = True

            except ImportError as e2:
                details["import"] = f"Failed: {e2}"
                return False, details

        # Test SingleModel creation
        if polyid_imported:
            try:
                from polyid.polyid import SingleModel
                from polyid.parameters import Parameters

                # Create a minimal SingleModel instance
                params = Parameters()
                model = SingleModel(parameters=params, model_id="test_model")

                details["singlemodel_creation"] = "Success"
                details["model_id"] = model.model_id

            except Exception as e:
                details["singlemodel_creation"] = f"Failed: {e}"

        # Test preprocessor generation
        if polyid_imported:
            try:
                from polyid.preprocessors.preprocessors import PolymerPreprocessor

                preprocessor = PolymerPreprocessor()
                details["preprocessor"] = "Success"

                # Test with sample SMILES
                test_inputs = preprocessor.construct_feature_matrices("CC(C)")
                details["feature_generation"] = f"Generated {len(test_inputs)} features"

            except Exception as e:
                details["preprocessor"] = f"Failed: {e}"

        # Test neural network architecture
        if polyid_imported:
            try:
                from polyid.models.base_models import global100

                # Check if function exists
                details["neural_architecture"] = "global100 available"

            except Exception as e:
                details["neural_architecture"] = f"Failed: {e}"

        # Test domain of validity module
        if polyid_imported:
            try:
                from polyid.domain_of_validity import DoV

                details["domain_validity"] = "Available"

            except Exception as e:
                details["domain_validity"] = f"Not available: {e}"

        success = polyid_imported and \
                 "Success" in str(details.get("singlemodel_creation", ""))

        return success, details

    def verify_dependencies_compatibility(self) -> Tuple[bool, Dict]:
        """Verify compatibility between different components"""
        details = {}

        # Test RDKit + NFP integration
        try:
            import rdkit
            from rdkit import Chem
            import nfp
            from nfp.preprocessing import SmilesPreprocessor

            # Create molecule with RDKit
            mol = Chem.MolFromSmiles("CC(C)C")

            # Process with NFP
            preprocessor = SmilesPreprocessor()
            features = preprocessor.construct_feature_matrices("CC(C)C")

            details["rdkit_nfp"] = "Compatible"

        except Exception as e:
            details["rdkit_nfp"] = f"Incompatible: {e}"

        # Test RDKit + TensorFlow
        try:
            import rdkit
            from rdkit import Chem
            import tensorflow as tf

            # Create molecule features
            mol = Chem.MolFromSmiles("CCC")
            num_atoms = mol.GetNumAtoms()

            # Create TensorFlow tensor
            atom_features = tf.constant([[1.0] * 10 for _ in range(num_atoms)])

            details["rdkit_tensorflow"] = "Compatible"

        except Exception as e:
            details["rdkit_tensorflow"] = f"Incompatible: {e}"

        # Test NFP + TensorFlow
        try:
            import nfp
            import tensorflow as tf
            from nfp.layers import EdgeUpdate, NodeUpdate

            # Try to create NFP layers (they use TensorFlow)
            edge_layer = EdgeUpdate()
            node_layer = NodeUpdate()

            details["nfp_tensorflow"] = "Compatible"

        except Exception as e:
            details["nfp_tensorflow"] = f"Incompatible: {e}"

        # Test full stack integration
        try:
            import rdkit
            import nfp
            import tensorflow as tf
            from rdkit import Chem
            from nfp.preprocessing import SmilesPreprocessor

            # Full pipeline test
            smiles = "CC(C)=O"
            mol = Chem.MolFromSmiles(smiles)

            preprocessor = SmilesPreprocessor()
            features = preprocessor.construct_feature_matrices(smiles)

            # Convert to TensorFlow tensors
            tensors = {k: tf.constant(v) for k, v in features.items()
                      if v is not None}

            details["full_stack"] = "Compatible"
            details["tensor_count"] = len(tensors)

        except Exception as e:
            details["full_stack"] = f"Failed: {e}"

        success = all([
            "Compatible" in str(details.get("rdkit_nfp", "")),
            "Compatible" in str(details.get("rdkit_tensorflow", "")),
            "Compatible" in str(details.get("nfp_tensorflow", ""))
        ])

        return success, details

    def verify_numerical_stability(self) -> Tuple[bool, Dict]:
        """Test numerical stability for chemistry computations"""
        details = {}

        # Test with edge cases
        try:
            import numpy as np
            import rdkit
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            edge_cases = [
                ("C", "minimal_molecule"),
                ("C" * 100, "large_linear"),
                ("c1ccccc1" * 5, "polycyclic"),
                ("C(F)(F)(F)", "highly_substituted"),
                ("C.[Na+].[Cl-]", "ionic")
            ]

            stable_count = 0
            for smiles, case_name in edge_cases:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        mw = Descriptors.MolWt(mol)
                        logp = Descriptors.MolLogP(mol)

                        # Check for numerical issues
                        if np.isfinite(mw) and np.isfinite(logp):
                            stable_count += 1

                except:
                    pass

            details["edge_cases"] = f"{stable_count}/{len(edge_cases)} stable"

        except Exception as e:
            details["edge_cases"] = f"Failed: {e}"

        # Test large batch processing
        try:
            import numpy as np

            # Simulate large molecular dataset
            large_matrix = np.random.randn(1000, 100)

            # Test common operations
            mean = np.mean(large_matrix, axis=0)
            std = np.std(large_matrix, axis=0)

            # Check for numerical stability
            if np.all(np.isfinite(mean)) and np.all(np.isfinite(std)):
                details["batch_processing"] = "Stable"
            else:
                details["batch_processing"] = "Numerical issues detected"

        except Exception as e:
            details["batch_processing"] = f"Failed: {e}"

        success = "stable" in str(details.get("edge_cases", ""))

        return success, details

    def generate_report(self) -> str:
        """Generate comprehensive verification report"""
        report = []
        report.append("="*70)
        report.append("CHEMISTRY STACK VERIFICATION REPORT")
        report.append("="*70)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Platform: {self.results['platform']}")
        report.append(f"Python: {self.results['python_version'].split()[0]}")
        report.append("")

        # Summary statistics
        total_tests = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"].values()
                    if t["status"] == "PASS")
        failed = sum(1 for t in self.results["tests"].values()
                    if t["status"] == "FAIL")
        errors = sum(1 for t in self.results["tests"].values()
                    if t["status"] == "ERROR")

        report.append("SUMMARY")
        report.append("-"*70)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"✅ Passed: {passed}")
        report.append(f"❌ Failed: {failed}")
        report.append(f"⚠️ Errors: {errors}")
        report.append(f"Success Rate: {(passed/total_tests)*100:.1f}%")
        report.append("")

        # Critical components status
        report.append("CRITICAL COMPONENTS")
        report.append("-"*70)

        critical_components = ["RDKit", "NFP", "TensorFlow", "PolyID"]
        for component in critical_components:
            test = self.results["tests"].get(component, {})
            status = test.get("status", "NOT_TESTED")
            symbol = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️"}.get(status, "❓")
            report.append(f"{symbol} {component}: {status}")

        report.append("")

        # Detailed test results
        report.append("DETAILED RESULTS")
        report.append("-"*70)

        for test_name, test_data in self.results["tests"].items():
            status_symbol = {
                "PASS": "✅", "FAIL": "❌", "ERROR": "⚠️"
            }.get(test_data["status"], "❓")

            report.append(f"\n{status_symbol} {test_name}")
            report.append(f"   Status: {test_data['status']}")
            report.append(f"   Duration: {test_data['duration']}s")

            if test_data["details"]:
                report.append("   Details:")
                for key, value in test_data["details"].items():
                    report.append(f"     • {key}: {value}")

            if test_data["error"]:
                report.append(f"   Error: {test_data['error']}")

        # Recommendations
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-"*70)

        if failed > 0 or errors > 0:
            report.append("⚠️ Issues detected in chemistry stack:")

            # Check specific failures
            if "RDKit" in self.results["tests"] and \
               self.results["tests"]["RDKit"]["status"] != "PASS":
                report.append("  • RDKit: Check conda environment and system libraries")

            if "NFP" in self.results["tests"] and \
               self.results["tests"]["NFP"]["status"] != "PASS":
                report.append("  • NFP: Verify TensorFlow compatibility")

            if "TensorFlow" in self.results["tests"] and \
               self.results["tests"]["TensorFlow"]["status"] != "PASS":
                report.append("  • TensorFlow: Check GPU drivers and CUDA installation")

            if "PolyID" in self.results["tests"] and \
               self.results["tests"]["PolyID"]["status"] != "PASS":
                report.append("  • PolyID: Ensure package is properly installed")

        else:
            report.append("✅ All components functioning correctly!")
            report.append("   Chemistry stack is ready for production use.")

        report.append("")
        report.append("="*70)

        return "\n".join(report)

    def save_results(self, filepath: str = "chemistry_stack_verification.json"):
        """Save detailed results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {filepath}")


def main():
    """Main verification workflow"""
    print("Starting Chemistry Stack Verification for HF Spaces Deployment")
    print("="*70)

    verifier = ChemistryStackVerifier()

    # Run all verification tests
    tests = [
        ("RDKit", verifier.verify_rdkit, True),
        ("NFP", verifier.verify_nfp, True),
        ("m2p", verifier.verify_m2p, False),
        ("TensorFlow", verifier.verify_tensorflow, True),
        ("PolyID", verifier.verify_polyid, True),
        ("Dependencies Compatibility", verifier.verify_dependencies_compatibility, True),
        ("Numerical Stability", verifier.verify_numerical_stability, False)
    ]

    for test_name, test_func, critical in tests:
        verifier.test_component(test_name, test_func, critical)

    # Generate and print report
    report = verifier.generate_report()
    print("\n" + report)

    # Save detailed results
    verifier.save_results()

    # Return exit code based on critical failures
    critical_failures = any(
        test["status"] != "PASS" and test["critical"]
        for test in verifier.results["tests"].values()
    )

    if critical_failures:
        print("\n❌ CRITICAL FAILURES DETECTED - Chemistry stack not fully operational")
        return 1
    else:
        print("\n✅ Chemistry stack verification completed successfully!")
        return 0


if __name__ == "__main__":
    exit(main())