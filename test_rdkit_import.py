#!/usr/bin/env python3
"""
Test script to verify RDKit import and basic functionality.
This helps diagnose RDKit installation issues in Hugging Face Spaces.
"""

import sys
import os

def test_rdkit_import():
    """Test RDKit import and basic functionality."""
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")

    try:
        print("\n=== Testing RDKit Import ===")
        import rdkit
        print(f"✅ RDKit imported successfully")
        print(f"RDKit version: {rdkit.__version__}")

        print("\n=== Testing RDKit.Chem ===")
        from rdkit import Chem
        print(f"✅ RDKit.Chem imported successfully")

        print("\n=== Testing RDKit.Chem.AllChem ===")
        from rdkit.Chem import AllChem
        print(f"✅ RDKit.Chem.AllChem imported successfully")

        print("\n=== Testing SMILES Parsing ===")
        # Test basic SMILES parsing
        test_smiles = "CCO"  # Ethanol
        mol = Chem.MolFromSmiles(test_smiles)
        if mol is not None:
            print(f"✅ Successfully parsed SMILES: {test_smiles}")
            print(f"Molecule formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
        else:
            print(f"❌ Failed to parse SMILES: {test_smiles}")

        return True

    except ImportError as e:
        print(f"❌ RDKit import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ RDKit test failed: {e}")
        return False

def test_polyid_imports():
    """Test polyID specific imports that depend on RDKit."""
    print("\n=== Testing PolyID RDKit Dependencies ===")

    # Add src to path like the main app does
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

    try:
        print("Testing domain_of_validity module...")
        from polyid.domain_of_validity import DoV
        print("✅ DoV imported successfully")

        print("Testing features module...")
        from polyid.preprocessors.features import atom_features_meso
        print("✅ Features imported successfully")

        return True

    except ImportError as e:
        print(f"❌ PolyID RDKit dependency import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PolyID test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 RDKit Import Test for PolyID")
    print("=" * 50)

    rdkit_ok = test_rdkit_import()
    polyid_ok = test_polyid_imports() if rdkit_ok else False

    print("\n" + "=" * 50)
    if rdkit_ok and polyid_ok:
        print("🎉 All tests passed! RDKit is working correctly.")
        sys.exit(0)
    elif rdkit_ok:
        print("⚠️  RDKit works but PolyID dependencies have issues.")
        sys.exit(1)
    else:
        print("💥 RDKit import failed. Check installation.")
        sys.exit(1)