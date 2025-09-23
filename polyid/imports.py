"""
PolyID - Unified Import Management System
Centralized import management for optimal performance and compatibility
"""

import os
import sys
import warnings
from typing import Any, Dict, Optional

# Performance optimizations BEFORE any TensorFlow imports
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '2',  # Suppress TensorFlow logging
    'OMP_NUM_THREADS': '4',       # Limit OpenMP threads
    'TF_ENABLE_ONEDNN_OPTS': '0', # Disable oneDNN optimizations for stability
    'CUDA_VISIBLE_DEVICES': '',   # Force CPU mode to avoid CUDA issues
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true'  # Prevent memory allocation issues
})

# Configure Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Lazy importer for heavy dependencies
class LazyImporter:
    """Lazy importer to defer expensive imports until needed"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module: Optional[Any] = None

    def __getattr__(self, item: str) -> Any:
        if self._module is None:
            try:
                self._module = __import__(self.module_name, fromlist=[item])
            except ImportError as e:
                raise ImportError(f"Failed to import {self.module_name}: {e}")
        return getattr(self._module, item)

# Core scientific computing - Always import these as they're lightweight
import numpy as np
import pandas as pd

# Lazy imports for heavy dependencies
tf = LazyImporter('tensorflow')
keras = LazyImporter('tensorflow.keras')
layers = LazyImporter('tensorflow.keras.layers')
models = LazyImporter('tensorflow.keras.models')
callbacks = LazyImporter('tensorflow.keras.callbacks')
optimizers = LazyImporter('tensorflow.keras.optimizers')

# Chemistry libraries - Lazy load for performance
rdkit = LazyImporter('rdkit')
Chem = LazyImporter('rdkit.Chem')
Descriptors = LazyImporter('rdkit.Chem.Descriptors')
rdMolDescriptors = LazyImporter('rdkit.Chem.rdMolDescriptors')
nfp = LazyImporter('nfp')

# PolyID core modules - Lazy load to avoid circular imports
def get_polyid_module(module_name: str) -> Any:
    """Get PolyID module with proper error handling"""
    try:
        module_path = f'polyid.{module_name}'
        return LazyImporter(module_path)
    except ImportError:
        # Fallback for different package structures
        return LazyImporter(module_name)

# Configure TensorFlow for optimal CPU performance
def configure_tensorflow():
    """Configure TensorFlow for optimal performance"""
    try:
        # Import TensorFlow
        import tensorflow as tf

        # Configure threading for CPU optimization
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)

        # Configure memory growth
        physical_devices = tf.config.list_physical_devices('CPU')
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except:
                pass  # Memory growth may not be applicable for CPU

        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')

        return True
    except ImportError:
        return False

# Initialize TensorFlow configuration
TF_AVAILABLE = configure_tensorflow()

# Import cache manager
from polyid.cache_manager import get_cache_manager

# Export commonly used functions and classes
__all__ = [
    'np', 'pd', 'tf', 'keras', 'layers', 'models', 'callbacks', 'optimizers',
    'rdkit', 'Chem', 'Descriptors', 'rdMolDescriptors', 'nfp', 'get_polyid_module',
    'TF_AVAILABLE', 'LazyImporter', 'get_cache_manager'
]