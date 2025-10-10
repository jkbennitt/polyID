# Option C: Cloud Training on Hugging Face - Implementation Plan

**Version**: 1.0
**Date**: September 24, 2025
**Status**: Ready for Approval
**Deployment Strategy**: Training Space → Model Hub → Inference Space

---

## Executive Summary

Use **Hugging Face cloud compute** to train PolyID models in a dedicated training Space, store trained models in HF Hub, and load them in the production inference Space (polyid-private). This is the professional ML ops workflow for production deployments.

**Key Benefits**:
- ✅ **Cloud GPU Training**: Use HF Pro GPU compute (no local hardware needed)
- ✅ **Clean Separation**: Training and inference are independent
- ✅ **Cost Efficient**: Only pay for GPU when training (pause training Space when done)
- ✅ **Professional**: Production-ready ML workflow
- ✅ **Version Control**: Model versioning via HF Hub

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING SPACE (polyid-trainer)                            │
│  - Hardware: A10G GPU ($1.00/hour)                          │
│  - Purpose: Train models on-demand                          │
│  - UI: Gradio button to start training                      │
│  - Auto-upload to Hub when complete                         │
│  - Can be PAUSED when not training                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (Upload trained models)
┌─────────────────────────────────────────────────────────────┐
│  MODEL HUB (jkbennitt/polyid-models)                        │
│  - Storage: FREE (public repo)                              │
│  - Versioning: Git tags (v1.0, v2.0, etc.)                  │
│  - Content: .h5 and .pk model files                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (Download on startup)
┌─────────────────────────────────────────────────────────────┐
│  INFERENCE SPACE (polyid-private)                           │
│  - Hardware: Standard GPU (existing)                        │
│  - Purpose: 24/7 predictions                                │
│  - Startup: Download models from Hub                        │
│  - Runtime: Real predictions (no mocks)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Space Components

### Space Configuration
- **Name**: `jkbennitt/polyid-trainer` (or your choice)
- **SDK**: Gradio
- **Hardware**: A10G GPU (recommended for training)
- **Visibility**: Private
- **Persistent Storage**: Optional (20GB = $5/month for model checkpoints)

### App Features (app.py)
```
┌─────────────────────────────────────┐
│   PolyID Model Training UI          │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Training Configuration       │   │
│  │  - Epochs: [500]             │   │
│  │  - K-folds: [5]              │   │
│  │  - Properties: [✓ All]       │   │
│  └─────────────────────────────┘   │
│                                     │
│  [Start Training] (Button)          │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Training Progress            │   │
│  │  Fold 1/5: Epoch 245/500     │   │
│  │  Loss: 0.234                 │   │
│  │  ETA: 2.5 hours              │   │
│  └─────────────────────────────┘   │
│                                     │
│  ✅ Training complete!              │
│  Models uploaded to Hub              │
│  jkbennitt/polyid-models             │
└─────────────────────────────────────┘
```

### Training Workflow
1. User clicks "Start Training" button
2. Space runs training script (30-50 hours)
3. Progress displayed in UI (live updates)
4. On completion: auto-upload to jkbennitt/polyid-models
5. User pauses Space to stop GPU costs
6. User restarts inference Space to load new models

---

## Model Hub Repository

### Repository Structure
```
jkbennitt/polyid-models/
├── model_fold_0/
│   ├── model.h5              # TensorFlow weights
│   ├── preprocessor.pk       # SMILES preprocessor
│   └── scaler.pk            # Feature scaler
├── model_fold_1/
│   └── ... (same structure)
├── model_fold_2/
├── model_fold_3/
├── model_fold_4/
├── parameters.pk             # MultiModel config
└── README.md                # Model card with performance metrics
```

### Versioning Strategy
- **Main branch**: Latest stable models
- **Git tags**: Version releases (v1.0.0, v1.1.0, etc.)
- **Branches**: Experimental models (experimental, dev)

### Model Card (README.md)
```markdown
# PolyID Trained Models

**Version**: 1.0.0
**Training Date**: 2025-09-24
**Hardware**: A10G GPU (HF Spaces)
**Training Time**: 42 hours

## Performance Metrics
- Glass Transition (Tg): MAE = 18.5°C, R² = 0.82
- Melting Temp (Tm): MAE = 24.3°C, R² = 0.78
- Young's Modulus: MAE = 0.3 GPa, R² = 0.71

## Usage
```python
from huggingface_hub import snapshot_download
from polyid import MultiModel

model_path = snapshot_download(repo_id="jkbennitt/polyid-models")
model = MultiModel.load_models(model_path)
```
```

---

## Inference Space Updates

### Updated app.py (Model Loading)
```python
from huggingface_hub import snapshot_download
from polyid import MultiModel

# Global model cache
_POLYID_MODEL = None

def load_polyid_model():
    """Load trained models from HF Hub"""
    global _POLYID_MODEL

    if _POLYID_MODEL is not None:
        return _POLYID_MODEL

    try:
        logger.info("Downloading models from HF Hub...")

        # Download from Hub to ephemeral cache
        model_path = snapshot_download(
            repo_id="jkbennitt/polyid-models",
            repo_type="model",
            cache_dir="/tmp/polyid_models",
            local_dir="/tmp/polyid_models/current",
            local_dir_use_symlinks=False
        )

        logger.info(f"Models downloaded to {model_path}")

        # Load MultiModel
        _POLYID_MODEL = MultiModel.load_models(model_path)
        logger.info("✅ Models loaded successfully from Hub")

        return _POLYID_MODEL

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return None

# Load on startup
MODEL = load_polyid_model()
```

### Startup Behavior
```
[INFO] PolyID Inference Space Starting...
[INFO] Downloading models from HF Hub...
[INFO] Downloading: jkbennitt/polyid-models
[INFO] ⏳ Download progress: 15/21 MB (71%)
[INFO] ✅ Models downloaded to /tmp/polyid_models/current
[INFO] Loading MultiModel from downloaded files...
[INFO] ✅ Models loaded successfully from Hub
[INFO] Space ready for predictions!
```

**Startup Time**: 30-60 seconds (one-time download per restart)

---

## Cost Analysis

### One-Time Training Cost
- **Hardware**: A10G GPU @ $1.00/hour
- **Training Time**: 30-50 hours (estimated)
- **Total Training Cost**: **$30-50** (one-time)

### Ongoing Monthly Costs

**Training Space** (paused when not training):
- Active (training): $1.00/hour × hours training
- Paused: **~$0/month** (minimal storage costs)

**Model Hub**:
- Public repo: **$0/month**
- Private repo: **$9/month** (optional)

**Inference Space** (24/7):
- Your current Standard GPU setup
- Costs unchanged from current deployment

### Example Monthly Cost Scenario

**First Month** (initial training):
- Training: $40 (one-time, 40 hours)
- Inference: Your current GPU costs
- Hub: $0 (public repo)
- **Total extra**: ~$40

**Subsequent Months** (no retraining):
- Training Space: $0 (paused)
- Inference: Your current GPU costs
- Hub: $0
- **Total extra**: $0

**Retraining Month** (quarterly):
- Training: $40 (40 hours)
- Inference: Your current GPU costs
- Hub: $0
- **Total extra**: ~$40 every 3-4 months

---

## Implementation Steps

### Phase 1: Create Training Space (1 hour)

**1.1 Create New Space**
- Go to: https://huggingface.co/new-space
- Name: `polyid-trainer`
- SDK: Gradio
- Hardware: A10G GPU
- Visibility: Private

**1.2 Upload Training Files**
- `app.py` - Gradio UI for training
- `train_models.py` - Training logic
- `requirements.txt` - Dependencies
- `data/stereopolymer_input_nopush.csv` - Training data

**1.3 Configure Space**
- Add HF token as secret (for Hub upload)
- Enable persistent storage (optional, for checkpoints)

### Phase 2: Create Model Hub Repo (5 minutes)

**2.1 Create Repository**
```bash
huggingface-cli login
huggingface-cli repo create polyid-models --type model
```

**2.2 Set Visibility**
- Public (free) or Private ($9/month)

### Phase 3: Run Initial Training (30-50 hours)

**3.1 Start Training**
- Open training Space UI
- Click "Start Training"
- Monitor progress

**3.2 Wait for Completion**
- Training runs: 30-50 hours
- Auto-uploads to Hub when done
- Space shows completion message

**3.3 Pause Training Space**
- Settings → Pause Space
- Stops GPU billing

### Phase 4: Update Inference Space (30 minutes)

**4.1 Update app.py**
- Add `huggingface_hub` import
- Replace `load_polyid_model()` with Hub download version
- Remove mock prediction fallbacks

**4.2 Update requirements.txt**
```txt
huggingface_hub>=0.26.0
```

**4.3 Deploy**
```bash
git add app.py requirements.txt
git commit -m "feat: Load models from HF Hub (cloud-trained)"
git push space standard-gpu-deployment:main
```

**4.4 Verify**
- Check Space logs for successful model download
- Test predictions in UI
- Confirm real predictions (not mocks)

---

## Training Space Code

### app.py (Training UI)
```python
import gradio as gr
import logging
from pathlib import Path
import sys
import os

# Add polyid to path
sys.path.insert(0, str(Path(__file__).parent))

from train_models import train_polyid_models
from huggingface_hub import HfApi, upload_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global training state
training_status = {"running": False, "progress": ""}

def start_training(epochs, kfolds):
    """Start model training and upload to Hub"""
    global training_status

    if training_status["running"]:
        return "❌ Training already in progress!"

    try:
        training_status["running"] = True
        training_status["progress"] = "Starting training..."

        # Train models
        logger.info(f"Starting training: {epochs} epochs, {kfolds} folds")
        success = train_polyid_models(
            data_path='data/stereopolymer_input_nopush.csv',
            save_path='trained_models',
            epochs=int(epochs),
            kfolds=int(kfolds)
        )

        if not success:
            training_status["running"] = False
            return "❌ Training failed! Check logs for details."

        training_status["progress"] = "Training complete! Uploading to Hub..."

        # Upload to HF Hub
        logger.info("Uploading models to Hugging Face Hub...")
        api = HfApi()
        upload_folder(
            folder_path="trained_models",
            repo_id="jkbennitt/polyid-models",
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
            commit_message=f"Trained models ({epochs} epochs, {kfolds} folds)"
        )

        training_status["running"] = False
        training_status["progress"] = "Complete!"

        return f"✅ Training complete! Models uploaded to jkbennitt/polyid-models"

    except Exception as e:
        training_status["running"] = False
        logger.error(f"Training error: {e}", exc_info=True)
        return f"❌ Training failed: {str(e)}"

def get_training_status():
    """Get current training status"""
    if training_status["running"]:
        return f"🔄 Training in progress...\n{training_status['progress']}"
    return "✅ Ready to train"

# Gradio UI
with gr.Blocks(title="PolyID Model Training") as demo:
    gr.Markdown("# PolyID Model Training on HF Cloud GPU")
    gr.Markdown("Train PolyID models using Hugging Face cloud compute")

    with gr.Row():
        with gr.Column():
            epochs_input = gr.Number(label="Epochs", value=500, precision=0)
            kfolds_input = gr.Number(label="K-Folds", value=5, precision=0)
            train_button = gr.Button("🚀 Start Training", variant="primary")

        with gr.Column():
            status_display = gr.Textbox(label="Training Status", value="Ready", lines=3)
            result_display = gr.Textbox(label="Result", lines=5)

    gr.Markdown("""
    ### Training Configuration
    - **Epochs**: Number of training epochs (recommended: 500-1000)
    - **K-Folds**: Cross-validation folds (recommended: 5-10)
    - **Expected Time**: 30-50 hours on A10G GPU
    - **Auto-Upload**: Models automatically uploaded to jkbennitt/polyid-models

    ### After Training
    1. Models will be uploaded to Hugging Face Hub
    2. Pause this Space to stop GPU billing
    3. Restart your inference Space to load new models
    """)

    # Event handlers
    train_button.click(
        fn=start_training,
        inputs=[epochs_input, kfolds_input],
        outputs=result_display
    )

    # Auto-refresh status every 10 seconds
    demo.load(fn=get_training_status, outputs=status_display, every=10)

if __name__ == "__main__":
    demo.launch()
```

### train_models.py (Training Logic)
```python
"""
Model training script for PolyID on Hugging Face Spaces
Adapted from examples/2_generate_and_train_models.ipynb
"""

import logging
from pathlib import Path

from polyid.preprocessors import PolymerPreprocessor
from polyid import MultiModel, Parameters
from polyid.models import global100
from nfp.preprocessing.features import atom_features_v1, bond_features_v1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_polyid_models(
    data_path='data/stereopolymer_input_nopush.csv',
    save_path='trained_models',
    epochs=500,
    kfolds=5
):
    """
    Train PolyID models with k-fold cross-validation

    Args:
        data_path: Path to training CSV
        save_path: Where to save trained models
        epochs: Number of training epochs
        kfolds: Number of cross-validation folds

    Returns:
        bool: True if training successful
    """

    try:
        logger.info(f"Starting PolyID training: {epochs} epochs, {kfolds} folds")

        # Model parameters (from publication)
        params = Parameters()
        params.batch_size = 1
        params.learning_rate = 1E-4
        params.decay = 1E-5
        params.atom_features = 128
        params.bond_features = 128
        params.num_messages = 12
        params.prediction_columns = [
            'Glass_Transition', 'Melt_Temp', 'Density',
            'log10_Permeability_CO2', 'log10_Permeability_N2',
            'log10_Permeability_O2', 'YoungMod'
        ]
        params.epochs = epochs
        params.kfolds = kfolds

        # Create MultiModel
        mm = MultiModel()

        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        mm.load_dataset(data_path, prediction_columns=params.prediction_columns)

        # Split data
        logger.info(f"Splitting data into {kfolds} folds")
        mm.split_data(kfolds=kfolds)

        # Generate preprocessors
        logger.info("Generating preprocessors...")
        mm.generate_preprocessors(
            preprocessor=PolymerPreprocessor,
            atom_features=atom_features_v1,
            bond_features=bond_features_v1,
            batch_size=params.batch_size
        )

        # Generate data scaler
        logger.info("Generating data scaler...")
        mm.generate_data_scaler()

        # Train models
        logger.info(f"Training {kfolds} models...")
        logger.info("⏳ This will take 30-50 hours on A10G GPU")

        mm.train_models(
            modelbuilder=global100,
            model_params=params.to_dict(),
            save_folder=save_path,
            save_training=True
        )

        logger.info(f"✅ Training complete! Models saved to {save_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        return False
```

---

## User Workflow

### Initial Training
1. **Create** training Space with code above
2. **Start** training via UI button
3. **Wait** 30-50 hours (monitor progress)
4. **Pause** training Space when complete
5. **Restart** inference Space to load new models

### Retraining (Future)
1. **Resume** training Space
2. **Adjust** hyperparameters in UI
3. **Start** new training run
4. **Wait** for completion
5. **Pause** Space again
6. **Restart** inference Space

### Daily Operations
- Training Space: **Paused** (costs ~$0)
- Inference Space: **Running 24/7** (existing costs)
- Models: **Automatically loaded** from Hub on inference Space restart

---

## Next Steps

### Awaiting Approval
Please review this plan and confirm:
- ✅ Use dedicated training Space (polyid-trainer)
- ✅ A10G GPU hardware ($1/hour, ~$40 one-time training)
- ✅ Store models in HF Hub (jkbennitt/polyid-models)
- ✅ Inference Space downloads from Hub
- ✅ Pause training Space when not training

### After Approval
I will:
1. Create training Space files (app.py, train_models.py)
2. Create model Hub repository
3. Update inference Space to load from Hub
4. Provide deployment commands
5. Guide you through first training run

**Ready to proceed when you approve this plan!** 🚀
