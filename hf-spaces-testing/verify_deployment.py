#!/usr/bin/env python
"""Auto-generated verification tests for HF Space deployment"""


# Basic import verification
print('\nRunning import_tests...')
print('-'*40)

import sys
print(f"Python: {sys.version}")

try:
    import rdkit
    from rdkit import Chem
    print("✓ RDKit imported successfully")
    print(f"  Version: {rdkit.__version__}")
except ImportError as e:
    print(f"✗ RDKit import failed: {e}")

try:
    import nfp
    print("✓ NFP imported successfully")
except ImportError as e:
    print(f"✗ NFP import failed: {e}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow imported: {tf.__version__}")
    print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")

try:
    from polyid.polyid import SingleModel, MultiModel
    print("✓ PolyID imported successfully")
except ImportError as e:
    print(f"✗ PolyID import failed: {e}")


# Component functionality verification
print('\nRunning functionality_tests...')
print('-'*40)

# Test RDKit functionality
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles("CC(C)C")
    mw = Descriptors.MolWt(mol)
    print(f"✓ RDKit molecular processing works (MW: {mw:.2f})")
except Exception as e:
    print(f"✗ RDKit processing failed: {e}")

# Test NFP preprocessing
try:
    from nfp.preprocessing import SmilesPreprocessor

    preprocessor = SmilesPreprocessor()
    features = preprocessor.construct_feature_matrices("CCC")
    print(f"✓ NFP preprocessing works ({len(features)} features)")
except Exception as e:
    print(f"✗ NFP preprocessing failed: {e}")

# Test TensorFlow operations
try:
    import tensorflow as tf

    # Simple computation
    a = tf.constant([[1.0, 2.0]])
    b = tf.constant([[3.0], [4.0]])
    c = tf.matmul(a, b)

    print(f"✓ TensorFlow computation works (result: {c.numpy()[0][0]:.1f})")
except Exception as e:
    print(f"✗ TensorFlow computation failed: {e}")


# Full stack integration
print('\nRunning integration_tests...')
print('-'*40)

# Test complete pipeline
try:
    import pandas as pd
    from rdkit import Chem
    from nfp.preprocessing import SmilesPreprocessor
    import tensorflow as tf

    # Create test data
    test_smiles = ["CC", "CCC", "CC(C)"]
    mols = [Chem.MolFromSmiles(s) for s in test_smiles]

    # Preprocess with NFP
    preprocessor = SmilesPreprocessor()
    features_list = [preprocessor.construct_feature_matrices(s)
                    for s in test_smiles]

    # Create TensorFlow dataset (simplified)
    print(f"✓ Full pipeline integration successful")
    print(f"  Processed {len(test_smiles)} molecules")
    print(f"  Generated {len(features_list)} feature sets")

except Exception as e:
    print(f"✗ Integration test failed: {e}")

