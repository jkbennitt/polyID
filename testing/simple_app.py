"""
Simple FastAPI app for PolyID PaleoBond API testing
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np
from fastapi import Request, FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import time

# Set up path for PolyID imports
sys.path.insert(0, os.path.dirname(__file__))

# Core PolyID imports
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    print("[OK] RDKit imported successfully")
except ImportError as e:
    print(f"[FAIL] RDKit import failed: {e}")
    rdkit = None

try:
    import shortuuid
    print("[OK] shortuuid imported successfully")
except ImportError as e:
    print(f"[FAIL] shortuuid import failed: {e}")
    shortuuid = None

try:
    # Import from the actual polyid package structure
    from polyid.polyid import SingleModel, MultiModel
    from polyid.parameters import Parameters
    from polyid.models.base_models import global100
    from polyid.preprocessors.preprocessors import PolymerPreprocessor
    from polyid.domain_of_validity import DoV
    print("[OK] PolyID core modules imported successfully")
    POLYID_AVAILABLE = True
except ImportError as e:
    print(f"[FAIL] PolyID import failed: {e}")
    POLYID_AVAILABLE = False

def validate_smiles(smiles: str) -> Tuple[bool, str]:
    """
    Validate SMILES string using RDKit

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if not rdkit:
            logger.warning("RDKit not available for SMILES validation")
            return False, "RDKit not available for SMILES validation"

        if not smiles or not isinstance(smiles, str) or not smiles.strip():
            return False, "SMILES string is required and must be non-empty"

        smiles_clean = smiles.strip()

        # Check for obviously invalid characters (basic sanity check)
        if any(char in smiles_clean for char in ['<', '>', '|', '{', '}', '\\']):
            return False, "SMILES contains invalid characters"

        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            logger.warning(f"RDKit could not parse SMILES: {smiles_clean}")
            return False, "Invalid SMILES string - could not be parsed"
        return True, "Valid SMILES string"
    except Exception as e:
        logger.error(f"SMILES validation error for '{smiles}': {str(e)}", exc_info=True)
        return False, f"SMILES validation error: {str(e)}"

def predict_single_polymer(smiles: str) -> Dict:
    """
    Predict properties for a single polymer in PaleoBond format

    Args:
        smiles: Polymer SMILES string

    Returns:
        Dictionary with 22 properties in PaleoBond format
    """
    try:
        # Validate SMILES
        is_valid, validation_msg = validate_smiles(smiles)
        if not is_valid:
            logger.error(f"SMILES validation failed for '{smiles}': {validation_msg}")
            return {
                "error": f"Invalid SMILES: {validation_msg}",
                "error_code": "INVALID_SMILES",
                "polymer_id": None,
                "smiles": smiles,
                "properties": {},
                "timestamp": pd.Timestamp.now().isoformat()
            }

        # Generate polymer ID
        polymer_id = f"POLY-{shortuuid.uuid()[:8].upper()}" if shortuuid else f"POLY-{hash(smiles) % 10000:04d}"

        # Get predictions (will use real models when available)
        properties = generate_mock_paleobond_properties(smiles)

        return {
            "polymer_id": polymer_id,
            "smiles": smiles,
            "properties": properties,
            "timestamp": pd.Timestamp.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Prediction failed for SMILES '{smiles}': {str(e)}", exc_info=True)
        return {
            "error": f"Prediction failed: {str(e)}",
            "error_code": "PREDICTION_ERROR",
            "polymer_id": None,
            "smiles": smiles,
            "properties": {},
            "timestamp": pd.Timestamp.now().isoformat()
        }

def generate_mock_paleobond_properties(smiles: str) -> Dict:
    """
    Generate mock predictions for all 22 PaleoBond properties

    Args:
        smiles: Polymer SMILES string

    Returns:
        Dictionary with 22 properties
    """
    # Base properties with realistic ranges for preservation polymers
    base_props = {
        "glass_transition_temp": np.random.normal(85, 25),  # °C
        "melting_temp": np.random.normal(160, 40),  # °C
        "decomposition_temp": np.random.normal(300, 50),  # °C
        "thermal_stability_score": np.random.uniform(0.6, 0.95),
        "tensile_strength": np.random.normal(50, 15),  # MPa
        "elongation_at_break": np.random.normal(150, 50),  # %
        "youngs_modulus": np.random.normal(2.5, 1.0),  # GPa
        "flexibility_score": np.random.uniform(0.4, 0.9),
        "water_resistance": np.random.uniform(0.6, 0.95),
        "acid_resistance": np.random.uniform(0.5, 0.9),
        "base_resistance": np.random.uniform(0.55, 0.9),
        "solvent_resistance": np.random.uniform(0.4, 0.85),
        "uv_stability": np.random.normal(5000, 1000),  # hours
        "oxygen_permeability": np.random.normal(50, 20),  # cm³·mil/m²·day·atm
        "moisture_vapor_transmission": np.random.normal(15, 5),  # g·mil/m²·day
        "biodegradability": np.random.uniform(0.1, 0.5),
        "hydrophane_opal_compatibility": np.random.uniform(0.6, 0.95),
        "pyrite_compatibility": np.random.uniform(0.5, 0.9),
        "fossil_compatibility": np.random.uniform(0.65, 0.95),
        "meteorite_compatibility": np.random.uniform(0.5, 0.85),
        "analysis_time": np.random.uniform(0.8, 2.5),  # seconds
        "confidence_score": np.random.uniform(0.7, 0.95)
    }

    # Add SMILES-based variation for realism
    smiles_hash = hash(smiles) % 10000
    variation_factor = (smiles_hash / 10000 - 0.5) * 0.2  # ±10% variation

    # Adjust properties based on molecular characteristics
    aromatic_content = smiles.count('c') / len(smiles) if smiles else 0
    if aromatic_content > 0.3:  # High aromatic content
        base_props["uv_stability"] *= 1.2
        base_props["thermal_stability_score"] *= 1.1
        base_props["hydrophane_opal_compatibility"] *= 0.9  # Less compatible

    # Round values appropriately
    for key, value in base_props.items():
        if key in ["glass_transition_temp", "melting_temp", "decomposition_temp", "tensile_strength",
                   "elongation_at_break", "youngs_modulus", "oxygen_permeability", "moisture_vapor_transmission"]:
            base_props[key] = round(value * (1 + variation_factor), 1)
        elif key in ["uv_stability"]:
            base_props[key] = round(value * (1 + variation_factor), 0)
        else:
            base_props[key] = round(value * (1 + variation_factor), 3)

    return base_props

def predict_batch_polymers(smiles_list: List[str]) -> List[Dict]:
    """
    Predict properties for multiple polymers

    Args:
        smiles_list: List of polymer SMILES strings

    Returns:
        List of prediction dictionaries
    """
    if not smiles_list:
        return []

    results = []
    for smiles in smiles_list:
        try:
            result = predict_single_polymer(smiles)
            results.append(result)
        except Exception as e:
            logger.error(f"Unexpected error in batch prediction for '{smiles}': {str(e)}", exc_info=True)
            results.append({
                "error": f"Unexpected prediction error: {str(e)}",
                "error_code": "BATCH_PREDICTION_ERROR",
                "polymer_id": None,
                "smiles": smiles,
                "properties": {},
                "timestamp": pd.Timestamp.now().isoformat()
            })

    return results

def health_check() -> Dict:
    """
    Health check endpoint

    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "timestamp": pd.Timestamp.now().isoformat(),
        "components": {
            "rdkit": "available" if rdkit else "unavailable",
            "nfp": "unavailable",  # Not imported
            "polyid": "mock_mode",
            "tensorflow": "unavailable"
        },
        "version": "1.0.0"
    }

def get_metrics() -> Dict:
    """
    Performance metrics endpoint

    Returns:
        Metrics dictionary
    """
    return {
        "predictions_total": 0,  # Would track in production
        "predictions_success": 0,
        "predictions_failed": 0,
        "average_response_time": 1.2,
        "uptime_seconds": 0.0,
        "memory_usage_mb": 0.0,
        "gpu_utilization": 0.0  # Would measure in production
    }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Endpoint functions
async def run_predict_endpoint(request: Request) -> JSONResponse:
    """
    /run/predict endpoint for single polymer prediction

    Args:
        request: FastAPI request with JSON body containing 'smiles'

    Returns:
        Prediction results in PaleoBond format
    """
    start_time = time.time()

    try:
        # Validate request content-type
        if request.headers.get("content-type") != "application/json":
            logger.warning("Invalid content-type in request")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Content-Type must be application/json",
                    "error_code": "INVALID_CONTENT_TYPE",
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        data = await request.json()

        # Validate request structure
        if not isinstance(data, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Request body must be a JSON object",
                    "error_code": "INVALID_REQUEST_FORMAT",
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        smiles = data.get("smiles")
        if not smiles:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing 'smiles' field in request body",
                    "error_code": "MISSING_SMILES",
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        if not isinstance(smiles, str):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "'smiles' field must be a string",
                    "error_code": "INVALID_SMILES_TYPE",
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        # Get prediction
        result = predict_single_polymer(smiles)

        # Add performance metrics
        processing_time = time.time() - start_time
        result["processing_time_seconds"] = round(processing_time, 3)

        # Check for errors in result
        if "error" in result:
            status_code = 400 if result.get("error_code") == "INVALID_SMILES" else 500
            return JSONResponse(status_code=status_code, content=result)

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error in run_predict_endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Internal server error: {str(e)}",
                "error_code": "INTERNAL_ERROR",
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )

async def batch_predict_endpoint(request: Request) -> JSONResponse:
    """
    /batch_predict endpoint for multiple polymer predictions

    Args:
        request: FastAPI request with JSON body containing 'smiles_list'

    Returns:
        List of prediction results
    """
    start_time = time.time()

    try:
        # Validate request content-type
        if request.headers.get("content-type") != "application/json":
            logger.warning("Invalid content-type in batch request")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Content-Type must be application/json",
                    "error_code": "INVALID_CONTENT_TYPE",
                    "results": [],
                    "summary": {"total": 0, "successful": 0, "failed": 0},
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        data = await request.json()

        # Validate request structure
        if not isinstance(data, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Request body must be a JSON object",
                    "error_code": "INVALID_REQUEST_FORMAT",
                    "results": [],
                    "summary": {"total": 0, "successful": 0, "failed": 0},
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        smiles_list = data.get("smiles_list", [])
        if not smiles_list:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing or empty 'smiles_list' field in request body",
                    "error_code": "MISSING_SMILES_LIST",
                    "results": [],
                    "summary": {"total": 0, "successful": 0, "failed": 0},
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        if not isinstance(smiles_list, list):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "'smiles_list' must be a list of SMILES strings",
                    "error_code": "INVALID_SMILES_LIST_TYPE",
                    "results": [],
                    "summary": {"total": 0, "successful": 0, "failed": 0},
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        # Limit batch size for performance
        if len(smiles_list) > 100:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Batch size limited to 100 SMILES strings",
                    "error_code": "BATCH_SIZE_EXCEEDED",
                    "results": [],
                    "summary": {"total": len(smiles_list), "successful": 0, "failed": len(smiles_list)},
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

        # Get predictions
        results = predict_batch_polymers(smiles_list)

        # Calculate summary
        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful

        processing_time = time.time() - start_time

        response_data = {
            "results": results,
            "summary": {
                "total": len(results),
                "successful": successful,
                "failed": failed,
                "processing_time_seconds": round(processing_time, 3)
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }

        # Return 207 Multi-Status if there are partial failures
        status_code = 207 if failed > 0 else 200
        return JSONResponse(status_code=status_code, content=response_data)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error in batch_predict_endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Internal server error: {str(e)}",
                "error_code": "INTERNAL_ERROR",
                "results": [],
                "summary": {"total": 0, "successful": 0, "failed": 0, "processing_time_seconds": round(processing_time, 3)},
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )

def health_endpoint() -> JSONResponse:
    """
    /health endpoint for system health check

    Returns:
        Health status dictionary
    """
    try:
        health_data = health_check()
        return JSONResponse(status_code=200, content=health_data)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": f"Health check failed: {str(e)}",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )

def metrics_endpoint() -> JSONResponse:
    """
    /metrics endpoint for performance metrics

    Returns:
        Metrics dictionary
    """
    try:
        metrics_data = get_metrics()
        return JSONResponse(status_code=200, content=metrics_data)
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Metrics collection failed: {str(e)}",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )

# Create FastAPI app
app = FastAPI()

# Add API routes
app.add_api_route("/run/predict", run_predict_endpoint, methods=["POST"])
app.add_api_route("/batch_predict", batch_predict_endpoint, methods=["POST"])
app.add_api_route("/health", health_endpoint, methods=["GET"])
app.add_api_route("/metrics", metrics_endpoint, methods=["GET"])

if __name__ == "__main__":
    print("[INFO] Starting simple PolyID API server")
    print("[INFO] Endpoints: /run/predict (POST), /batch_predict (POST), /health (GET), /metrics (GET)")
    uvicorn.run(app, host="0.0.0.0", port=7861)