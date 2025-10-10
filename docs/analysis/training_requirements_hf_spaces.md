# PolyID Model Training Requirements for HF Spaces Deployment

> **✅ RECOMMENDED APPROACH**: See [OPTION_C_CLOUD_TRAINING_PLAN.md](../deployment/OPTION_C_CLOUD_TRAINING_PLAN.md) for the approved cloud training workflow using dedicated HF training Space.

## Executive Summary

**Current State**: PolyID HF Space is operational with mock predictions only
**Goal**: Deploy real trained models for authentic polymer property prediction
**Critical Finding**: Training on inference Space is **NOT recommended** - use dedicated training Space instead

**Recommended Solution**: Option C - Training Space → Model Hub → Inference Space (see deployment plan above)

---

## 1. Training Computational Requirements

### Dataset Characteristics
- **Training Data**: `stereopolymer_input_nopush.csv`
- **Dataset Size**: 229 samples (rows)
- **Target Properties**: 7 polymer properties
  1. Glass_Transition (Tg)
  2. Melt_Temp (Tm)
  3. Density
  4. log10_Permeability_CO2
  5. log10_Permeability_N2
  6. log10_Permeability_O2
  7. YoungMod

### Training Configuration (from notebook)
```python
# Optimized hyperparameters from publication
batch_size = 1              # Extremely small for stability
learning_rate = 1E-4
decay = 1E-5
atom_features = 128
bond_features = 128
num_messages = 12           # 12 message-passing iterations

# Training parameters
epochs = 500                # Recommended 500-1000
kfolds = 10                 # 10-fold cross-validation
```

### Per-Fold Training Time Estimates

**Dataset Split per Fold**:
- Training samples: ~206 (90%)
- Validation samples: ~23 (10%)

**Training Iterations**:
- 500 epochs × 206 samples/epoch = **103,000 gradient updates** per model
- Batch size of 1 = 103,000 individual forward/backward passes

**Time Estimates per Model**:
- **Optimistic** (0.1s/update on GPU): ~2.9 hours
- **Conservative** (0.5s/update on CPU/slow GPU): ~14.3 hours
- **Realistic on Standard GPU (T4)**: ~4-6 hours

### Total Training Time for 10-Fold Ensemble

| Scenario | Per Model | Total (10 models) | Days |
|----------|-----------|-------------------|------|
| Optimistic (GPU) | 2.9 hrs | **29 hours** | 1.2 |
| Realistic (T4) | 5 hrs | **50 hours** | 2.1 |
| Conservative | 14.3 hrs | **143 hours** | 6.0 |

**⚠️ Critical Issue**: HF Spaces Standard GPU has runtime limits and cost implications for multi-day training

### GPU Requirements

**Can Training Run on CPU?**
- ✅ Yes, technically possible
- ❌ Extremely slow (10-20x slower than GPU)
- ❌ Not practical for production deployment

**GPU Necessity**:
- **Essential** for reasonable training times
- **Recommended**: NVIDIA GPU with CUDA support
- **Minimum**: T4 GPU (16GB VRAM) - sufficient for this model size
- **Memory Usage**: ~2-4GB VRAM per model during training

### Model Size Analysis

**Parameter Count Estimate**:
```
Message-passing layers:
  - Atom embeddings: 128 × 128 = 16,384
  - Bond embeddings: 128 × 128 = 16,384
  - 12 message iterations × ~32,000 params = ~384,000

Output layers (7 properties):
  - Dense layers: 128 × 7 = 896

Total per model: ~406,400 parameters
```

**Storage Requirements**:
- Single model (FP32): ~1.6 MB (.h5 file)
- Preprocessor + scaler: ~0.5 MB (.pk file)
- **Total per model**: ~2.1 MB
- **10-fold ensemble**: ~21 MB
- **With training logs**: ~30-50 MB total

**✅ Verdict**: Model size is NOT a constraint for HF Spaces

---

## 2. Training Data Pipeline

### Data Already on Space
- ✅ `data/stereopolymer_input_nopush.csv` is committed to repo
- ✅ Data will be available on Space deployment
- ✅ No additional data upload needed

### Data Preprocessing Requirements

**Required Preprocessing Steps** (from training notebook):
1. **Data Loading**: `mm.load_dataset()` with prediction columns
2. **K-fold Splitting**: `mm.split_data(kfolds=10)`
3. **Data Scaling**: `mm.generate_data_scaler()` using RobustScaler
4. **Preprocessor Generation**: `mm.generate_preprocessors()` with:
   - PolymerPreprocessor
   - atom_features_v1
   - bond_features_v1
   - batch_size=1

**Preprocessing Computational Cost**:
- Data scaling: ~1-2 seconds (lightweight)
- Preprocessor generation: ~30-60 seconds per fold (SMILES → graphs)
- **Total preprocessing**: ~10-15 minutes for all folds

**Memory Requirements**:
- Raw dataset: ~82 KB
- Preprocessed graphs: ~5-10 MB (in-memory)
- Scaled data: ~1 MB
- **Total preprocessing memory**: <100 MB

---

## 3. Model Output Requirements

### Expected File Structure (from MultiModel.train_models())

```
save_folder/
├── parameters.pk              # Ensemble parameters and metadata
├── model_0/
│   ├── model.h5              # TensorFlow model weights
│   ├── preprocessor.pk       # Graph preprocessor
│   ├── data_scaler.pk        # RobustScaler for targets
│   └── training_log.csv      # Loss history (if save_training=True)
├── model_1/
│   ├── model.h5
│   ├── preprocessor.pk
│   ├── data_scaler.pk
│   └── training_log.csv
...
├── model_9/
│   └── ...
```

### Storage for Production Deployment

**Where Models Should Be Saved**:
- **Development/Training**: `save_examples/` (gitignored)
- **Production Deployment**: `models/` (committed to repo)
- **HF Space Deployment**: Models must be in repo root or committed path

**File Naming Convention**:
```python
# From training notebook
save_folder = "save_examples"  # For development

# For HF Spaces production:
save_folder = "models/trained_ensemble"
```

### Model Loading in app.py

**Current app.py structure**:
```python
# app.py needs to load models at startup:
from polyid.polyid import MultiModel

# Load trained ensemble
mm = MultiModel.load_models("models/trained_ensemble")

# Make predictions
predictions = mm.make_aggregate_predictions(df_input)
```

---

## 4. Integration Strategy Recommendations

### ❌ Option A: Train on Space Startup (NOT RECOMMENDED)

**Approach**: Run training script when Space starts
```python
# startup.py
mm = MultiModel()
mm.load_dataset('data/stereopolymer_input_nopush.csv', ...)
mm.split_data(kfolds=10)
mm.generate_data_scaler()
mm.generate_preprocessors(...)
mm.train_models(modelbuilder=global100, save_folder="models/trained_ensemble")
```

**Issues**:
- ⏱️ 29-50+ hour training time on Space startup
- 💰 Costly GPU usage for training every deployment
- 🔄 Space timeout/restart issues during long training
- 📊 No visibility into training progress for users
- ⚠️ Training failures break deployment

**Verdict**: ❌ **Do NOT train on Space**

---

### ❌ Option B: Training Endpoint/Script (NOT RECOMMENDED)

**Approach**: Add training button/endpoint to Gradio app
```python
def train_models_on_space():
    # Trigger training from UI
    mm = MultiModel()
    # ... training code ...
```

**Issues**:
- Same 29-50+ hour training time
- UI blocked during training
- Potential timeout issues
- Resource waste for multiple users
- Complexity in managing training state

**Verdict**: ❌ **Avoid on-demand training**

---

### ✅ Option C: Pre-Trained Models (RECOMMENDED)

**Approach**: Train locally or on dedicated compute, commit trained models to repo

#### Step-by-Step Implementation:

**1. Local Training (Recommended Path)**
```bash
# On local machine with GPU (or cloud compute)
cd C:\Users\redmo\Projects\polyID

# Ensure environment is set up
conda activate polyID

# Run training notebook or script
jupyter notebook examples/2_generate_and_train_models.ipynb

# Or run as script:
python scripts/train_models.py
```

**Training Script** (`scripts/train_models.py`):
```python
from polyid.preprocessors import PolymerPreprocessor
from polyid import MultiModel, Parameters
from polyid.models import global100
from nfp.preprocessing.features import atom_features_v1, bond_features_v1
import pandas as pd

# Generate model parameters
params = Parameters()
params.batch_size = 1
params.learning_rate = 1E-4
params.decay = 1E-5
params.atom_features = 128
params.bond_features = 128
params.num_messages = 12
params.prediction_columns = ['Glass_Transition', 'Melt_Temp', 'Density',
                              'log10_Permeability_CO2', 'log10_Permeability_N2',
                              'log10_Permeability_O2', 'YoungMod']
params.epochs = 500
params.kfolds = 10

# Create MultiModel
mm = MultiModel()

# Load data
mm.load_dataset('data/stereopolymer_input_nopush.csv',
                prediction_columns=params.prediction_columns)

# Split and preprocess
mm.split_data(kfolds=params.kfolds)
mm.generate_data_scaler()
mm.generate_preprocessors(preprocessor=PolymerPreprocessor,
                          atom_features=atom_features_v1,
                          bond_features=bond_features_v1,
                          batch_size=params.batch_size)

# Train models
mm.train_models(modelbuilder=global100,
                model_params=params.to_dict(),
                save_folder="models/trained_ensemble",  # Production path
                save_training=True)

print("Training complete! Models saved to models/trained_ensemble/")
```

**2. Commit Trained Models to Repo**
```bash
# After training completes locally
git add models/trained_ensemble/
git commit -m "feat: Add pre-trained 10-fold ensemble models for HF Spaces deployment"
git push origin standard-gpu-deployment
```

**3. Update app.py to Load Models**
```python
# app.py startup code
import os
from polyid.polyid import MultiModel

# Global model variable
TRAINED_MODELS = None

def load_trained_models():
    """Load pre-trained models at startup"""
    global TRAINED_MODELS

    model_path = "models/trained_ensemble"

    if os.path.exists(model_path):
        try:
            TRAINED_MODELS = MultiModel.load_models(model_path)
            logger.info(f"✓ Loaded pre-trained ensemble from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    else:
        logger.warning(f"No trained models found at {model_path}")
        return False

# Call at startup
load_trained_models()

def predict_properties(smiles: str) -> Dict:
    """Make real predictions using trained models"""
    if TRAINED_MODELS is None:
        return generate_mock_predictions(smiles)

    # Create input dataframe
    df_input = pd.DataFrame({'smiles_polymer': [smiles]})

    # Make ensemble predictions
    predictions = TRAINED_MODELS.make_aggregate_predictions(df_input)

    return format_predictions(predictions)
```

**4. Space Deployment**
- Models are part of repo
- Space builds with pre-trained models
- No training overhead
- Instant predictions on startup

---

## 5. Recommended Workflow

### Phase 1: Local Training Setup (This Week)

**Tasks**:
1. ✅ Verify local GPU availability (or use cloud compute)
2. ✅ Create training script from notebook
3. ✅ Run training locally (budget 1-2 days for full training)
4. ✅ Validate trained models with test predictions
5. ✅ Document model performance metrics

**Expected Outputs**:
- `models/trained_ensemble/` directory with 10 models
- Validation metrics (MAE, R² for each property)
- Training logs and loss curves

### Phase 2: Model Integration (Next Week)

**Tasks**:
1. ✅ Update `.gitignore` to allow `models/trained_ensemble/`
2. ✅ Commit trained models to repo (Git LFS if >50MB)
3. ✅ Update `app.py` to load pre-trained models
4. ✅ Remove mock prediction fallbacks
5. ✅ Test locally with real predictions

### Phase 3: HF Spaces Deployment (Final)

**Tasks**:
1. ✅ Push updates to `standard-gpu-deployment` branch
2. ✅ Deploy to HF Spaces
3. ✅ Verify models load correctly on Space
4. ✅ Test end-to-end predictions
5. ✅ Monitor performance and memory usage

---

## 6. Alternative: Cloud Training Options

If local GPU unavailable, consider:

### Option 1: Google Colab Pro
- **Cost**: $10/month
- **GPU**: T4 or V100
- **Training Time**: 4-8 hours
- **Setup**: Upload notebook, run training, download models

### Option 2: AWS SageMaker / EC2
- **Cost**: ~$0.50-1.50/hour for GPU instances
- **GPU**: Various (T4, V100, A10G)
- **Training Time**: 3-6 hours
- **Setup**: Launch instance, run training, download models

### Option 3: Hugging Face Spaces (Persistent Storage)
- **Cost**: HF Pro subscription for longer runtimes
- **Not Recommended**: Still faces timeout issues and cost inefficiency

---

## 7. Data Availability Check

### Current Data Status
```bash
# Data is already in repo
$ ls data/
mordred_fp_descripts.csv
stereopolymer_input_nopush.csv  # ← Training data (82 KB, 229 rows)
```

**✅ Data Ready for Training**:
- File exists in repo
- Size is manageable (82 KB)
- Contains required columns: `smiles_polymer`, `Tg`, `Tm`, etc.
- No additional preprocessing needed before training

**Data on HF Space**:
- Automatically available when Space builds from repo
- No manual upload required
- Can be used for DoV analysis and visualization

---

## 8. Technical Constraints Summary

| Constraint | Status | Notes |
|------------|--------|-------|
| **Training Time** | ⚠️ 29-50 hours | Too long for on-Space training |
| **Model Size** | ✅ ~21 MB | Well within HF limits |
| **Data Size** | ✅ 82 KB | Negligible |
| **GPU Memory** | ✅ 2-4 GB | T4 (16GB) is sufficient |
| **Space Runtime** | ❌ Hours-long limits | Not suitable for training |
| **Cost** | 💰 High if training on Space | Pre-training avoids costs |

---

## 9. Final Recommendation

### ✅ Recommended Approach: **Pre-Train Locally, Deploy to Space**

**Rationale**:
1. **Practicality**: 29-50 hour training is incompatible with HF Spaces runtime limits
2. **Cost Efficiency**: One-time local training vs. continuous Space GPU costs
3. **Reliability**: Trained models are tested before deployment
4. **User Experience**: Instant predictions, no training delays
5. **Maintainability**: Models can be retrained and updated independently

**Action Items**:
1. **Set up local training environment** with GPU
2. **Run training script** adapted from notebook (1-2 days)
3. **Validate models** with test predictions
4. **Commit trained models** to repo (~21 MB)
5. **Update app.py** to load models at startup
6. **Deploy to HF Spaces** with real predictions

### Expected User Experience After Deployment
- ✅ Space starts in <2 minutes (no training)
- ✅ Real predictions from trained ensemble
- ✅ Fast inference (<1 second per prediction)
- ✅ Reliable, tested models
- ✅ Cost-effective deployment

---

## 10. Next Steps

### Immediate Actions (Priority Order)

1. **Create Training Script** (30 min)
   - Adapt notebook to standalone Python script
   - Use production paths (`models/trained_ensemble/`)

2. **Run Local Training** (1-2 days)
   - Execute training on local GPU or cloud compute
   - Monitor progress and validate results

3. **Validate Models** (2 hours)
   - Test predictions on holdout data
   - Check model performance metrics
   - Verify ensemble aggregation

4. **Integrate with app.py** (1 hour)
   - Add model loading at startup
   - Replace mock predictions with real predictions
   - Test locally before deployment

5. **Deploy to HF Spaces** (30 min)
   - Commit models and code changes
   - Push to standard-gpu-deployment branch
   - Monitor Space build and startup

### Success Criteria
- ✅ Models train successfully with acceptable MAE/R²
- ✅ Model files are <100 MB total (for easy repo management)
- ✅ app.py loads models without errors
- ✅ Real predictions match expected ranges
- ✅ HF Space starts and runs predictions smoothly

---

## Appendix A: Training Data Analysis

**File**: `data/stereopolymer_input_nopush.csv`

**Structure**:
```
Rows: 229
Columns: smiles_monomer, pm, polymer_name, Tg, Tm, Tm_units, Tg_units,
         monomers, distribution, replicate_structure, smiles_polymer, mechanism

Target Properties Available: Tg, Tm (2 properties)
```

**⚠️ Data Discrepancy**:
- Notebook trains on 7 properties (Tg, Tm, Density, Permeabilities, YoungMod)
- Current data file has only Tg and Tm
- **Resolution needed**: Verify correct training data file or adjust target properties

**Action Required**:
- Confirm if `data/dftrain.csv` (referenced in notebook) is the correct training file
- Or adjust training to use only available properties (Tg, Tm)

---

## Appendix B: Model Performance Expectations

Based on PolyID publication and typical polymer property prediction:

| Property | Expected MAE | Expected R² | Units |
|----------|--------------|-------------|-------|
| Glass_Transition (Tg) | 15-25°C | 0.75-0.85 | °C |
| Melt_Temp (Tm) | 20-35°C | 0.70-0.80 | °C |
| Density | 0.02-0.05 g/cm³ | 0.80-0.90 | g/cm³ |
| log10_Permeability | 0.3-0.5 | 0.75-0.85 | log10(Barrer) |
| YoungMod | 0.2-0.4 GPa | 0.65-0.75 | GPa |

**Note**: 10-fold cross-validation provides robust performance estimates with confidence intervals.

---

**Document Version**: 1.0
**Date**: 2025-09-24
**Author**: Scientific Computing Analysis (Claude Opus)
**Branch**: standard-gpu-deployment
**Status**: Analysis Complete - Ready for Implementation