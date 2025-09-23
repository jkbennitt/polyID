# PolyID Integration Context for AI Agent

## Project Overview
You are working on the **PolyID repository** at https://github.com/jkbennitt/polyID to support **PaleoBond**, a production molecular generation platform for preservation formulation development. PaleoBond combines REINVENT4 reinforcement learning, MoLFormer-XL property prediction, and Rust-accelerated scoring to generate preservation formulations for materials like hydrophane opals, pyrite disease prevention, and fossil consolidation.

## Current Integration Status: 95% Complete

### ✅ What's Working
- **PolyID Standard GPU Space**: Deployed and running at `jkbennitt/polyid-private`
- **PaleoBond Integration**: Complete via [`python-backend/app/polyid_client.py`](python-backend/app/polyid_client.py:1)
- **Web Interface**: Fully functional with SMILES input and JSON response
- **ZeroGPU Integration**: Ready with `@spaces.GPU` decorators
- **Mock Mode Fallback**: Works when trained models unavailable
- **Response Format**: Matches PaleoBond's 22-property requirement structure

### ❌ Known Issues to Fix
1. **REST API Endpoints**: Returning 404 errors despite web interface working
2. **Model Training**: Currently using mock predictions, need real trained models
3. **Performance Optimization**: Cold start handling needs improvement
4. **Batch Processing**: Missing batch endpoint for multiple SMILES analysis

## Critical Integration Requirements

### 1. Response Format (MANDATORY)
PaleoBond expects this exact JSON structure:

```json
{
  "polymer_id": "POLY-1234",
  "smiles": "input_smiles_string",
  "properties": {
    "glass_transition_temp": 85.0,
    "melting_temp": 160.0,
    "decomposition_temp": 300.0,
    "thermal_stability_score": 0.8,
    "tensile_strength": 50.0,
    "elongation_at_break": 150.0,
    "youngs_modulus": 2.5,
    "flexibility_score": 0.7,
    "water_resistance": 0.75,
    "acid_resistance": 0.65,
    "base_resistance": 0.70,
    "solvent_resistance": 0.60,
    "uv_stability": 5000.0,
    "oxygen_permeability": 50.0,
    "moisture_vapor_transmission": 15.0,
    "biodegradability": 0.3,
    "hydrophane_opal_compatibility": 0.8,
    "pyrite_compatibility": 0.7,
    "fossil_compatibility": 0.75,
    "meteorite_compatibility": 0.65,
    "analysis_time": 1.2,
    "confidence_score": 0.85
  }
}
```

### 2. PaleoBond Client Integration
PaleoBond connects via [`gradio_client`](python-backend/app/polyid_client.py:12) library:

```python
from gradio_client import Client

class PolyIDClient:
    def __init__(self, space_url="jkbennitt/polyid-private"):
        self.gradio_client = Client(space_url, hf_token=hf_token)
    
    async def analyze_polymer(self, smiles: str):
        result = self.gradio_client.predict(smiles, api_name="/predict")
        return self._parse_gradio_response(result)
```

### 3. Performance Expectations
- **Cold Start Handling**: Must handle HF Space restarts gracefully
- **Timeout**: 120s default, up to 300s for cold starts
- **Retry Logic**: 3 attempts with exponential backoff
- **Batch Processing**: Support for multiple SMILES in single request
- **Caching**: PaleoBond caches predictions client-side

### 4. Authentication
- **Private Space**: Uses HF token from `POLYID_API_KEY` or `HUGGINGFACE_TOKEN`
- **Access**: PaleoBond authenticates via environment variables

## Preservation-Specific Requirements

### Target Applications
- **Hydrophane Opal Stabilization**: Deep penetration (>10mm), refractive index matching
- **Pyrite Disease Prevention**: Oxygen barriers, iron chelation  
- **Fossil Consolidation**: Controlled penetration, strength enhancement
- **Meteorite Protection**: Oxidation prevention, chloride extraction

### Critical Properties for Preservation
- **Molecular weight**: 200-500 Da
- **Refractive index**: 1.37-1.47  
- **UV stability**: >5000 hours
- **Penetration depth**: >10mm
- **Compatibility scores**: Key metrics for preservation applications

## Current Performance Metrics (PaleoBond Side)
- **Cold Start Detection**: Tracks requests >5min apart
- **Average Response Time**: Currently variable due to HF Space restarts
- **Success Rate**: High for web interface, API endpoints need fixing
- **TTF Response**: Time-to-first-response after cold starts monitored

## Development Environment Context

### PolyID Space Structure
```
app.py                 # Main Gradio application with @spaces.GPU
requirements.txt       # Python 3.13 compatible packages  
polyid/               # Core PolyID library
├── models/           # Model files (parameters.pk missing = mock mode)
├── base_models/      # Base model components
└── callbacks/        # Training callbacks
README.md             # HF Spaces configuration
```

### Key Files You'll Work With
- **`app.py`**: Main Gradio interface with ZeroGPU integration
- **`polyid/models.py`**: MultiModel class for predictions
- **`requirements.txt`**: Dependencies (already Python 3.13 compatible)
- **`README.md`**: HF Spaces metadata and documentation

## API Endpoints Needed

### Current (Working)
- **Web Interface**: `https://jkbennitt-polyid-private.hf.space` ✅
- **Direct Prediction**: Via Gradio client ✅

### Missing (To Implement)
- **REST API**: `/run/predict` endpoint returns 404 ❌
- **Batch Processing**: `/batch_predict` for multiple SMILES ❌
- **Health Check**: `/health` endpoint ❌
- **Performance Metrics**: `/metrics` endpoint ❌

## Testing Requirements

### Essential Tests
1. **Single SMILES Analysis**: Must return 22-property JSON
2. **Batch Processing**: Handle multiple SMILES efficiently  
3. **Error Handling**: Graceful failures with proper error JSON
4. **Cold Start Recovery**: Handle HF Space restarts
5. **API Endpoint Functionality**: Fix 404 errors

### Test SMILES Examples
```python
test_molecules = [
    "CCO",                                    # Ethanol (simple)
    "CC(C)(C)OC(=O)C=C",                     # Poly(tert-butyl acrylate)
    "CC=C(C)C(=O)OC",                        # PMMA
    "C=CC6H5",                               # Polystyrene
    "CC(=O)OC1=CC=CC=C1C(=O)O"              # Aspirin (complex)
]
```

## Environment Variables You Should Know
```bash
# PaleoBond uses these for PolyID integration:
POLYID_SPACE_URL=jkbennitt/polyid-private
POLYID_TIMEOUT=120
POLYID_MAX_RETRIES=3  
POLYID_CACHE_ENABLED=true
HUGGINGFACE_TOKEN=hf_your_token_here
```

## Success Criteria
- ✅ **API Endpoints Work**: Fix 404 errors, enable programmatic access
- ✅ **Batch Processing**: Handle multiple SMILES efficiently
- ✅ **Real Model Training**: Replace mock predictions with trained models
- ✅ **Performance Optimization**: Improve cold start and response times
- ✅ **Preservation Properties**: Accurate compatibility scores for preservation use cases
- ✅ **Production Stability**: Robust error handling and monitoring

## Integration Architecture

```mermaid
graph TD
    A[PaleoBond Frontend] --> B[PaleoBond FastAPI Backend]
    B --> C[PolyIDClient]
    C --> D[gradio_client]
    D --> E[HF Space: jkbennitt/polyid-private]
    E --> F[@spaces.GPU: analyze_polymer_gpu]
    F --> G[MultiModel or Mock Predictions]
    G --> H[22-Property JSON Response]
    H --> C
    C --> I[Cache + Monitoring]
    I --> B
    B --> A
```

## Priority Order
1. **Fix API 404 errors** - Critical for programmatic access
2. **Implement batch processing** - Performance improvement
3. **Add monitoring endpoints** - Production readiness  
4. **Train real models** - Replace mock predictions
5. **Optimize performance** - Cold start and response time improvements
6. **Add preservation-specific features** - Enhanced compatibility scoring

This context should provide you with everything needed to enhance the PolyID repository for optimal PaleoBond integration.