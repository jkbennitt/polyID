# Model Persistence and Deployment Strategy for PolyID on Hugging Face Spaces

> **✅ RECOMMENDED APPROACH**: See [OPTION_C_CLOUD_TRAINING_PLAN.md](OPTION_C_CLOUD_TRAINING_PLAN.md) for the approved implementation using dedicated training Space with cloud GPU.

## Executive Summary

This document provides a comprehensive strategy for deploying trained PolyID models on Hugging Face Spaces, comparing multiple deployment options. **Option C (Training Space → Model Hub → Inference Space) is the recommended approach for cloud-based training.**

**Current State:**
- HF Space: `jkbennitt/polyid-private` (Standard GPU)
- App: `app.py` with model loading from `models/` directory
- Models needed: MultiModel with multiple `.h5` (TensorFlow) and `.pk` (preprocessor/scaler/metadata) files
- Current implementation: No trained models, using mock predictions

**Selected Solution:** Option C - Dedicated training Space using HF cloud GPU compute (see plan above)

---

## 1. HF Spaces Storage Options Analysis

### 1.1 Ephemeral Storage (Default - FREE)
**What it is:** Temporary storage that resets on every Space restart/rebuild

**Characteristics:**
- Free for all Spaces
- Reset on restart, rebuild, or code changes
- Suitable for runtime data only
- Not suitable for model persistence

**Use cases for PolyID:**
- ❌ Storing trained models (lost on restart)
- ✅ Temporary prediction caches
- ✅ Runtime logs and session data

### 1.2 Persistent Storage (Paid Upgrade)
**What it is:** Disk space that persists across Space restarts

**Pricing (as of 2025):**
- Small (20GB): $5/month
- Medium (150GB): $25/month
- Large (1TB+): Custom pricing

**Key Features:**
- Survives Space restarts and rebuilds
- Accessible at `/data/` directory
- Billed continuously once added (even when Space is stopped)
- Cannot be downgraded (must delete and recreate)

**Environment Variable Setup:**
```python
import os
os.environ['HF_HOME'] = '/data/.huggingface'  # Cache models in persistent storage
```

**Use cases for PolyID:**
- ✅ Storing trained models (if training in Space)
- ✅ Large model checkpoints
- ⚠️ Additional cost required
- ⚠️ Still requires models to be trained or uploaded first

### 1.3 Hugging Face Hub Storage (Model Repository)
**What it is:** Git-based repository for storing and versioning models

**Characteristics:**
- Free for public models
- Built-in versioning with Git
- Git LFS for large files (>.h5, .pk files)
- **Xet-enabled repositories (NEW 2025):** Better than Git LFS, chunk-level deduplication, faster downloads
- Accessible via `huggingface_hub` library

**Key Features:**
- Version control for models
- Easy sharing and collaboration
- Integrated with HF ecosystem
- Download on startup

**Use cases for PolyID:**
- ✅ Store pre-trained models
- ✅ Version control for model iterations
- ✅ Share models across Spaces
- ✅ Free for public models
- ✅ **RECOMMENDED APPROACH**

### 1.4 Git LFS in Space Repository
**What it is:** Store large model files directly in your Space's Git repository

**Characteristics:**
- Models live in Space repo
- Git LFS tracks files >10MB
- Deployed with Space code
- Version controlled with code

**Limitations:**
- Repository size limits apply
- Increases repo clone time
- Less flexible than Hub storage
- Harder to update models independently

**Use cases for PolyID:**
- ⚠️ Small models only (<100MB total)
- ⚠️ Models rarely change
- ❌ Not recommended for PolyID (large model ensembles)

---

## 2. Model Deployment Strategies - Detailed Comparison

### Option A: Pre-train Locally → Upload to Space Repo via Git LFS

**How it works:**
1. Train models locally using PolyID training scripts
2. Add model files to Space repo with Git LFS
3. Commit and push to Space repo
4. Space loads models from repo on startup

**Pros:**
- ✅ Simple deployment (models with code)
- ✅ Version controlled together
- ✅ No additional storage costs
- ✅ Works offline once cloned

**Cons:**
- ❌ Large repo size (slow clones)
- ❌ Hard to update models independently
- ❌ Git LFS storage limits
- ❌ Inefficient for large model ensembles

**Best for:**
- Small to medium models (<500MB)
- Infrequent model updates
- Single model deployments

**PolyID suitability:** ⚠️ **NOT RECOMMENDED**
- PolyID uses MultiModel with multiple .h5 and .pk files
- Model ensembles can be >1GB
- Frequent retraining needs independent model updates

---

### Option B: Pre-train Locally → Store in HF Hub → Download at Startup ⭐ RECOMMENDED

**How it works:**
1. Train models locally using PolyID training scripts
2. Upload models to HF Hub model repository
3. Space downloads models from Hub on startup
4. Cache in `/tmp/` (ephemeral) or `/data/` (persistent storage)

**Pros:**
- ✅ Clean separation of code and models
- ✅ Easy model updates (just upload new version)
- ✅ Free for public models
- ✅ Built-in versioning
- ✅ Share models across multiple Spaces
- ✅ Optimized for ML model storage (Xet in 2025)
- ✅ Smaller Space repo size

**Cons:**
- ❌ Requires download on first startup (~30-60s)
- ❌ Needs internet access to update models
- ⚠️ Public models are visible (use private repo for sensitive models)

**Best for:**
- Large model ensembles
- Frequent model updates
- Multiple Spaces using same models
- **PolyID deployment ⭐**

**PolyID suitability:** ✅ **RECOMMENDED**
- Perfect for MultiModel architecture
- Easy to retrain and update
- Version control for model iterations
- Free for public models

---

### Option C: Training Space → Model Hub → Inference Space

**How it works:**
1. **Training Space:** Dedicated Space for training models
   - Uses persistent storage for training data
   - GPU for training
   - Uploads trained models to Hub
2. **Model Hub:** Stores versioned models
3. **Inference Space:** Production Space for predictions
   - Downloads models from Hub
   - Optimized for inference
   - Can use cheaper GPU tier

**Pros:**
- ✅ Complete ML workflow in HF ecosystem
- ✅ Separate training and inference resources
- ✅ Training Space can be paused when not training
- ✅ Inference Space always available
- ✅ Professional ML ops setup

**Cons:**
- ❌ More complex setup (two Spaces)
- ❌ Additional cost (training Space GPU)
- ❌ Requires orchestration between Spaces
- ❌ Overkill for most use cases

**Best for:**
- Production ML systems
- Frequent retraining needs
- Resource optimization (pause training Space)
- Enterprise deployments

**PolyID suitability:** ⚠️ **ADVANCED OPTION**
- Useful if training new models frequently
- Can optimize costs (training GPU only when needed)
- Better suited for production deployment at scale
- Probably overkill for current use case

---

## 3. Recommended Strategy for PolyID: Option B Implementation

### 3.1 Why Option B is Best for PolyID

1. **Model Architecture Fit:**
   - PolyID uses MultiModel with ensemble of SingleModels
   - Each model has `.h5` (TensorFlow weights) + `.pk` (preprocessor/scaler)
   - Easy to version as a collection in Hub repository

2. **Workflow Optimization:**
   - Train locally with full control
   - Upload once to Hub
   - Multiple Spaces can use same models
   - Update models independently of code

3. **Cost Efficiency:**
   - No persistent storage fees
   - Free Hub storage for public models
   - Only pay for inference GPU time

4. **Developer Experience:**
   - Clean separation of concerns
   - Easy model versioning
   - Simple update process

### 3.2 Implementation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Train PolyID Models                            │    │
│  │     - MultiModel.train_models()                    │    │
│  │     - Save to local folder                         │    │
│  │     - Validate model performance                   │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  2. Upload to Hugging Face Hub                     │    │
│  │     - huggingface-cli login                        │    │
│  │     - huggingface_hub.upload_folder()              │    │
│  │     - Tag version (e.g., v1.0.0)                   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 HUGGING FACE HUB (Storage)                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Model Repository: jkbennitt/polyid-models         │    │
│  │                                                     │    │
│  │  📁 model_fold_0/                                  │    │
│  │     ├── model.h5              (TensorFlow)         │    │
│  │     ├── preprocessor.pk       (Pickle)             │    │
│  │     └── scaler.pk             (Pickle)             │    │
│  │  📁 model_fold_1/                                  │    │
│  │  📁 model_fold_2/                                  │    │
│  │  📁 ...                                             │    │
│  │  📄 parameters.pk              (MultiModel config) │    │
│  │  📄 README.md                  (Model card)        │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              HUGGING FACE SPACE (Inference)                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Space: jkbennitt/polyid-private                   │    │
│  │                                                     │    │
│  │  On Startup:                                       │    │
│  │  1. Download models from Hub                       │    │
│  │     - snapshot_download()                          │    │
│  │     - Cache in /tmp/polyid_models/                 │    │
│  │  2. Load MultiModel                                │    │
│  │     - MultiModel.load_models(model_path)           │    │
│  │  3. Ready for predictions                          │    │
│  │                                                     │    │
│  │  Runtime:                                          │    │
│  │  - Use loaded models for predictions               │    │
│  │  - Return real predictions (not mocks)             │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Step-by-Step Implementation Guide

### 4.1 Prerequisites

**Local Environment:**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Authenticate with Hugging Face
huggingface-cli login
# Enter your HF token when prompted
```

**Create Model Repository on HF Hub:**
1. Go to https://huggingface.co/new
2. Select "Model" repository
3. Name: `polyid-models` (or your preferred name)
4. Visibility: Public (free) or Private ($9/month)
5. Click "Create repository"

### 4.2 Step 1: Train Models Locally

Use PolyID's standard training workflow:

```python
# train_polyid_models.py
import os
from polyid import MultiModel
from polyid.models.base_models import global100
from polyid.parameters import Parameters
import pandas as pd

# Load training data
df = pd.read_csv('data/polymer_data.csv')

# Initialize MultiModel
mm = MultiModel()

# Load and prepare data
mm.load_dataset(df, smiles_column='smiles_polymer')
mm.split_data(valid_folds=5, n_folds=5)

# Generate preprocessors
mm.generate_preprocessors()

# Generate data scaler
mm.generate_data_scaler()

# Train models
mm.train_models(
    build_fn=global100,
    save_folder='trained_models',  # Local save directory
    verbose=1
)

print("✅ Models trained successfully!")
print(f"📁 Models saved to: {os.path.abspath('trained_models')}")
```

**Expected Output Structure:**
```
trained_models/
├── model_fold_0/
│   ├── model.h5
│   ├── preprocessor.pk
│   └── scaler.pk
├── model_fold_1/
│   ├── model.h5
│   ├── preprocessor.pk
│   └── scaler.pk
├── ... (more folds)
└── parameters.pk
```

### 4.3 Step 2: Upload Models to Hugging Face Hub

**Method A: Using Python API (Recommended)**

```python
# upload_models_to_hub.py
from huggingface_hub import HfApi, create_repo, upload_folder
import os

# Configuration
HF_USERNAME = "jkbennitt"  # Your HF username
MODEL_REPO_NAME = "polyid-models"
LOCAL_MODEL_PATH = "trained_models"
REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"

# Initialize HF API
api = HfApi()

# Create repository (if it doesn't exist)
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=False,  # Set to True for private repo
        exist_ok=True
    )
    print(f"✅ Repository created/verified: {REPO_ID}")
except Exception as e:
    print(f"⚠️ Repository creation note: {e}")

# Upload entire model folder
upload_folder(
    folder_path=LOCAL_MODEL_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload trained PolyID models",
)

print(f"✅ Models uploaded successfully to: https://huggingface.co/{REPO_ID}")
```

**Method B: Using CLI**

```bash
# Upload entire folder
huggingface-cli upload jkbennitt/polyid-models ./trained_models --repo-type model

# Or upload with specific commit message
huggingface-cli upload jkbennitt/polyid-models ./trained_models \
    --repo-type model \
    --commit-message "Add PolyID v1.0 trained models"
```

### 4.4 Step 3: Create Model Card (README.md)

Create a `README.md` in your model repository to document the models:

```markdown
---
library_name: polyid
tags:
- polymer-prediction
- graph-neural-networks
- chemistry
- materials-science
license: bsd-3-clause
---

# PolyID Trained Models

Pre-trained graph neural network models for polymer property prediction.

## Model Details

**Framework:** PolyID (Graph Neural Networks for Polymer Properties)
**Architecture:** global100 (Message-passing neural network)
**Training Dataset:** [Describe your training data]
**Properties Predicted:**
- Glass Transition Temperature (Tg)
- Melting Temperature (Tm)
- Young's Modulus
- Permeability (O2, CO2)

## Model Structure

This repository contains a MultiModel ensemble with 5 cross-validation folds:

```
model_fold_0/
├── model.h5              # TensorFlow/Keras model weights
├── preprocessor.pk       # PolymerPreprocessor for SMILES processing
└── scaler.pk            # StandardScaler for feature normalization

model_fold_1/ ... (similar structure)
parameters.pk            # MultiModel configuration
```

## Usage

```python
from huggingface_hub import snapshot_download
from polyid import MultiModel

# Download models from Hub
model_path = snapshot_download(
    repo_id="jkbennitt/polyid-models",
    repo_type="model"
)

# Load MultiModel
model = MultiModel.load_models(model_path)

# Make predictions
import pandas as pd
df = pd.DataFrame({'smiles_polymer': ['CC', 'CCC']})
predictions = model.make_aggregate_predictions(df, funcs=['mean'])
print(predictions)
```

## Performance

- **Cross-validation R²:** [Your metrics]
- **MAE:** [Your metrics]
- **Domain of Validity:** [Coverage stats]

## Citation

```bibtex
@article{wilson2023polyid,
  title={PolyID: Artificial Intelligence for Discovering Performance-Advantaged and Sustainable Polymers},
  author={Wilson, Anand N. and others},
  journal={Macromolecules},
  year={2023}
}
```

## License

BSD-3-Clause
```

### 4.5 Step 4: Update Space `app.py` to Load Models from Hub

**Modify your `app.py`:**

```python
# app.py - Updated for HF Hub model loading
import os
import sys
import logging
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PolyID imports
sys.path.insert(0, os.path.dirname(__file__))
from polyid import MultiModel
from polyid.preprocessors.preprocessors import PolymerPreprocessor

# Global model cache
_POLYID_MODEL = None
_MODEL_PATH = None

def download_models_from_hub():
    """
    Download trained models from Hugging Face Hub on startup

    Returns:
        Path to downloaded models
    """
    try:
        logger.info("⏳ Downloading PolyID models from Hugging Face Hub...")

        # Download models to /tmp/ (ephemeral storage - free)
        model_path = snapshot_download(
            repo_id="jkbennitt/polyid-models",
            repo_type="model",
            cache_dir="/tmp/polyid_models",  # Ephemeral cache
            local_dir="/tmp/polyid_models/current",  # Extraction location
            local_dir_use_symlinks=False  # Copy files instead of symlinks
        )

        logger.info(f"✅ Models downloaded successfully to: {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"❌ Failed to download models from Hub: {e}")
        raise

def load_polyid_model():
    """
    Load trained PolyID model from Hugging Face Hub

    Returns:
        Loaded MultiModel or None if not available
    """
    global _POLYID_MODEL, _MODEL_PATH

    if _POLYID_MODEL is not None:
        logger.info("Using cached model")
        return _POLYID_MODEL

    try:
        # Download models from Hub (only on first load)
        if _MODEL_PATH is None:
            _MODEL_PATH = download_models_from_hub()

        logger.info(f"⏳ Loading PolyID models from: {_MODEL_PATH}")

        # Load MultiModel
        _POLYID_MODEL = MultiModel.load_models(_MODEL_PATH)

        logger.info("✅ PolyID models loaded successfully")
        return _POLYID_MODEL

    except Exception as e:
        logger.error(f"❌ Failed to load PolyID models: {e}")
        return None

def predict_polymer_properties(smiles: str, properties: list):
    """
    Predict polymer properties using trained models from Hub

    Args:
        smiles: Polymer SMILES string
        properties: List of properties to predict

    Returns:
        Dictionary with predictions
    """
    try:
        # Load model (cached after first call)
        model = load_polyid_model()

        if model is None:
            return {
                "error": "Models not available - check startup logs",
                "note": "Using mock predictions as fallback"
            }

        # Make real predictions
        df_input = pd.DataFrame({'smiles_polymer': [smiles]})
        predictions = model.make_aggregate_predictions(df_input, funcs=['mean'])

        if predictions.empty:
            raise ValueError("Empty prediction result")

        # Format results
        pred = predictions.iloc[0]
        results = {}

        # Map properties to prediction columns
        property_map = {
            'Tg': 'Glass_Transition_pred_mean',
            'Tm': 'Melt_Temp_pred_mean',
            'YoungMod': 'YoungMod_pred_mean',
            'O2_Permeability': 'log10_Permeability_O2_pred_mean',
            'CO2_Permeability': 'log10_Permeability_CO2_pred_mean'
        }

        for prop in properties:
            if prop in property_map:
                col_name = property_map[prop]
                if col_name in pred:
                    results[prop] = {
                        'value': float(pred[col_name]),
                        'unit': get_unit(prop),
                        'source': 'PolyID MultiModel (HF Hub)'
                    }

        return results

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {"error": f"Prediction failed: {str(e)}"}

def get_unit(property_name):
    """Get unit for property"""
    units = {
        'Tg': '°C',
        'Tm': '°C',
        'YoungMod': 'MPa',
        'O2_Permeability': 'log10(Barrer)',
        'CO2_Permeability': 'log10(Barrer)'
    }
    return units.get(property_name, '')

# Pre-load models on startup
logger.info("=" * 50)
logger.info("PolyID Startup - Loading models from HF Hub")
logger.info("=" * 50)

try:
    load_polyid_model()
    logger.info("✅ Startup model loading complete")
except Exception as e:
    logger.error(f"⚠️ Startup model loading failed: {e}")
    logger.info("   Space will use mock predictions")

# Rest of your Gradio interface code...
```

### 4.6 Step 5: Update Space Requirements

**Ensure `requirements.txt` includes:**

```txt
# HF Hub integration
huggingface_hub>=0.26.0

# PolyID dependencies
gradio>=5.46.0
tensorflow>=2.16.0
rdkit>=2023.9.1
nfp>=0.4.0
shortuuid>=1.0.11
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

### 4.7 Step 6: Add Environment Variables (Optional)

For optimized caching with persistent storage (if you add it later):

**In Space Settings → Variables:**
```bash
HF_HOME=/data/.huggingface  # Cache HF downloads in persistent storage
TRANSFORMERS_CACHE=/data/.huggingface/transformers
HF_DATASETS_CACHE=/data/.huggingface/datasets
```

**Without persistent storage (using ephemeral /tmp):**
```python
# In app.py - already configured above
cache_dir="/tmp/polyid_models"
```

---

## 5. Alternative Approaches

### 5.1 Using Space Secrets for Model Paths

Store model configuration in Space secrets:

```python
# In Space Settings → Variables
MODEL_REPO_ID="jkbennitt/polyid-models"
MODEL_VERSION="v1.0.0"  # Git tag or branch

# In app.py
import os
from huggingface_hub import snapshot_download

model_repo = os.getenv("MODEL_REPO_ID", "jkbennitt/polyid-models")
model_version = os.getenv("MODEL_VERSION", "main")

model_path = snapshot_download(
    repo_id=model_repo,
    revision=model_version,  # Specific version
    cache_dir="/tmp/models"
)
```

### 5.2 Model Versioning Strategy

Use Git tags for model versions:

```bash
# Tag model version
git tag v1.0.0
git push origin v1.0.0

# In app.py - download specific version
snapshot_download(
    repo_id="jkbennitt/polyid-models",
    revision="v1.0.0"  # Specific tag
)
```

### 5.3 Lazy Loading (Load on First Prediction)

Only load models when first prediction is made:

```python
def predict_polymer_properties(smiles: str):
    global _POLYID_MODEL

    # Lazy load on first use
    if _POLYID_MODEL is None:
        _POLYID_MODEL = load_polyid_model()

    # Make prediction
    # ...
```

---

## 6. Deployment Workflow Summary

### Complete Deployment Pipeline

```bash
# 1. Local: Train models
python train_polyid_models.py

# 2. Local: Upload to Hub
python upload_models_to_hub.py

# 3. Local: Update app.py (use code above)
# 4. Local: Commit and push to Space repo
git add app.py requirements.txt
git commit -m "Add HF Hub model loading"
git push

# 5. HF Space: Automatically rebuilds and downloads models
# 6. Test: Make predictions in Space UI
```

### Update Workflow (When Retraining Models)

```bash
# 1. Train new models locally
python train_polyid_models.py

# 2. Upload to Hub (overwrites or creates new version)
python upload_models_to_hub.py

# 3. Space automatically uses new models on next restart
# OR manually restart Space to reload immediately
```

---

## 7. Cost Analysis

### Option B Cost Breakdown (Recommended)

**Storage Costs:**
- HF Hub model storage: **FREE** (public repo)
- Space ephemeral storage: **FREE**
- Total storage: **$0/month**

**Compute Costs:**
- Standard GPU Space: **$0.60/hour** (Nvidia T4 small)
- Estimated monthly (24/7): **~$432/month** (with Pro subscription discount)
- OR **FREE** during Community GPU queue time

**Total Monthly Cost:** **$0** (using Community GPU) or **~$60-400** (dedicated GPU)

### Option A Cost Breakdown (Git LFS in Space Repo)

**Storage Costs:**
- Git LFS storage (1GB models): **FREE** (within limits)
- Total storage: **$0/month**

**Compute Costs:**
- Same as Option B

**Downsides:**
- Large repo size
- Slow deployments
- Hard to update models

### Option C Cost Breakdown (Training + Inference Spaces)

**Storage Costs:**
- HF Hub model storage: **FREE**
- Training Space persistent storage (150GB): **$25/month**

**Compute Costs:**
- Training Space (A10G GPU): **$1.00/hour** (only when training)
- Inference Space (T4 GPU): **$0.60/hour** (24/7)

**Total Monthly Cost:** **$25 + training hours + inference hours**

---

## 8. Troubleshooting

### Issue: Models fail to download

**Symptoms:**
```
HTTPError: 401 Client Error: Unauthorized
```

**Solutions:**
1. Check HF token is valid:
```bash
huggingface-cli whoami
```

2. For private repos, add token to Space secrets:
```python
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="jkbennitt/polyid-models",
    token=os.getenv("HF_TOKEN")  # Add as Space secret
)
```

### Issue: Models load slowly

**Symptoms:**
- Space startup takes >2 minutes
- First prediction is very slow

**Solutions:**
1. Use persistent storage to cache downloads:
```python
snapshot_download(
    repo_id="jkbennitt/polyid-models",
    cache_dir="/data/.huggingface"  # Requires persistent storage upgrade
)
```

2. Pre-warm cache by downloading in Dockerfile (Space builder):
```dockerfile
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='jkbennitt/polyid-models')"
```

### Issue: Out of memory loading models

**Symptoms:**
```
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM
```

**Solutions:**
1. Enable TensorFlow memory growth:
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

2. Upgrade to larger GPU:
- Settings → Hardware → A10G or A100

---

## 9. Security Considerations

### Public vs Private Model Repositories

**Public Models (Free):**
- ✅ Free storage
- ✅ Easy sharing
- ❌ Models visible to everyone
- ❌ Training data may be inferred

**Private Models ($9/month):**
- ✅ Protected models
- ✅ Access control
- ✅ Suitable for proprietary research
- ❌ Requires subscription

### Authentication Best Practices

**Never hardcode tokens:**
```python
# ❌ Bad - token in code
snapshot_download(repo_id="...", token="hf_...")

# ✅ Good - token from environment
snapshot_download(repo_id="...", token=os.getenv("HF_TOKEN"))
```

**Use Space secrets:**
1. Go to Space Settings → Variables
2. Add `HF_TOKEN` with your token
3. Set as "Secret" (encrypted)

---

## 10. Recommended Implementation for PolyID

### Final Recommendation: **Option B** with the following setup:

1. **Model Repository:** `jkbennitt/polyid-models` (HF Hub)
   - Public or private based on needs
   - Stores all trained MultiModel files
   - Version controlled with Git tags

2. **Space Configuration:**
   - Download models on startup using `snapshot_download()`
   - Cache in `/tmp/` (ephemeral - free)
   - Load into global variable for predictions

3. **Deployment Process:**
   - Train locally
   - Upload to Hub via Python API or CLI
   - Space auto-downloads on restart
   - Zero storage costs (using public repo + ephemeral cache)

4. **Update Process:**
   - Retrain models locally
   - Upload new version to Hub
   - Restart Space (or wait for next auto-restart)
   - Models automatically updated

### Expected Timeline

**Initial Setup:** 1-2 hours
- Create Hub repo: 5 min
- Train models: 30-60 min (depends on data)
- Upload to Hub: 5-10 min
- Update app.py: 30 min
- Deploy and test: 15 min

**Model Updates:** 15-30 minutes
- Retrain: (varies)
- Upload: 5 min
- Restart Space: 2 min
- Validation: 5-10 min

---

## 11. Next Steps

### Immediate Actions (To Get Real Predictions Working)

1. **Create model repository on HF Hub** (5 min)
   ```bash
   # Via web UI or CLI
   huggingface-cli repo create polyid-models --type model
   ```

2. **Train PolyID models locally** (30-60 min)
   ```bash
   python train_polyid_models.py
   ```

3. **Upload models to Hub** (5 min)
   ```bash
   python upload_models_to_hub.py
   ```

4. **Update app.py** (30 min)
   - Add `snapshot_download()` for model loading
   - Replace mock predictions with real model calls
   - Test locally first

5. **Deploy and validate** (15 min)
   - Push to Space repo
   - Monitor logs for successful model download
   - Test predictions in UI

### Future Enhancements

1. **Add model versioning**
   - Git tags for model versions
   - Specify version in Space config

2. **Implement caching optimization**
   - Add persistent storage for faster restarts
   - Cache preprocessed predictions

3. **Add model monitoring**
   - Track prediction performance
   - Log model usage metrics

4. **Consider Option C** (if frequent retraining needed)
   - Dedicated training Space
   - Automated model upload pipeline
   - Separate inference Space

---

## Appendix: Code Templates

### A. Complete Upload Script

```python
# upload_models_to_hub.py
from huggingface_hub import HfApi, create_repo, upload_folder
import os

HF_USERNAME = "jkbennitt"
MODEL_REPO_NAME = "polyid-models"
LOCAL_MODEL_PATH = "trained_models"
REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"

api = HfApi()

# Create repo
create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)

# Upload models
upload_folder(
    folder_path=LOCAL_MODEL_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload trained PolyID models"
)

print(f"✅ Models uploaded to: https://huggingface.co/{REPO_ID}")
```

### B. Complete Download and Load Script

```python
# In app.py
from huggingface_hub import snapshot_download
from polyid import MultiModel
import logging

logger = logging.getLogger(__name__)

def load_models_from_hub(repo_id="jkbennitt/polyid-models"):
    """Download and load PolyID models from HF Hub"""

    # Download models
    model_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir="/tmp/polyid_models",
        local_dir="/tmp/polyid_models/current",
        local_dir_use_symlinks=False
    )

    logger.info(f"Models downloaded to: {model_path}")

    # Load MultiModel
    model = MultiModel.load_models(model_path)
    logger.info("Models loaded successfully")

    return model

# Global model
_MODEL = load_models_from_hub()
```

### C. Prediction Function with Real Models

```python
def predict_properties(smiles: str, properties: list):
    """Make real predictions using loaded models"""

    import pandas as pd

    # Prepare input
    df = pd.DataFrame({'smiles_polymer': [smiles]})

    # Make predictions
    predictions = _MODEL.make_aggregate_predictions(df, funcs=['mean'])

    # Format results
    results = {}
    pred = predictions.iloc[0]

    for prop in properties:
        col_name = f"{prop}_pred_mean"
        if col_name in pred:
            results[prop] = {
                'value': float(pred[col_name]),
                'unit': get_unit(prop),
                'source': 'PolyID MultiModel'
            }

    return results
```

---

## Summary

This strategy provides a comprehensive approach to deploying trained PolyID models on Hugging Face Spaces:

- **Recommended:** Option B (HF Hub storage with startup download)
- **Zero additional storage costs** using public repos and ephemeral cache
- **Easy model updates** via Hub upload
- **Clean separation** of models and code
- **Version control** built-in with Git
- **Production ready** with proper error handling and logging

The implementation requires minimal code changes to `app.py` and provides a robust, scalable solution for PolyID deployment on HF Spaces.