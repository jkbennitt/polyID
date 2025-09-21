# Option 1: Standard GPU Spaces Implementation

## Implementation Overview

**Goal**: Implement PolyID for Standard GPU Spaces from clean fork state, incorporating insights gained from ZeroGPU compatibility investigation to ensure robust chemistry stack deployment.

**Approach**: Start fresh from clean fork and build with proven configurations and dependency management strategies learned during our investigation.

**Key Benefits**:
- Full chemistry package compatibility without restrictions
- Proven dependency versions and configurations
- Flexible Python version management
- Comprehensive system dependency support

---

## Claude Code Prompt

Copy and paste this entire prompt to Claude Code when working on the `standard-gpu-deployment` branch:

```
I need you to implement Option 1: Standard GPU Spaces deployment for the PolyID project. This is a FRESH implementation starting from the clean fork state, incorporating valuable insights gained from our ZeroGPU compatibility investigation.

## Context

PolyID is a polymer property prediction application using graph neural networks. During investigation, we discovered ZeroGPU Spaces have fundamental compatibility limitations with chemistry packages (RDKit, NFP, m2p). The complete analysis is in `docs/ZeroGPU_Compatibility_Analysis.md`.

**Important**: You are starting from a CLEAN FORK STATE - not modifying existing ZeroGPU configurations. The branch has been reset to the original fork point before any ZeroGPU work.

## Insights Gained from Investigation

### Optimal Dependency Versions (Latest Stable):
- `rdkit>=2024.3.1` - Latest RDKit with Python 3.11+ support
- `nfp>=0.4.0` - Latest neural fingerprint with modern TensorFlow
- `shortuuid>=1.0.11` - Latest UUID generation
- `gradio>=5.48.0` - Latest Gradio with all features
- `tensorflow>=2.16.0` - Latest stable TensorFlow

### System Dependencies (chemistry stack):
```
libboost-dev, libcairo2-dev, libeigen3-dev, libgomp1, python3-dev, build-essential
cmake, pkg-config, libboost-python-dev, libboost-serialization-dev,
libboost-system-dev, libboost-thread-dev
libxrender1, libfontconfig1, libice6, libsm6, libxext6, libxrandr2, libxss1
```

### Optimal Python Version:
- Use `python_version: "3.11"` for best balance of stability and modern features
- Much better chemistry package compatibility than 3.10
- No ZeroGPU constraints - can use latest versions

## Current State
- Clean fork state with original PolyID code
- No ZeroGPU configurations present
- Original app.py without GPU decorators
- Need to set up for Standard GPU Spaces from scratch

## Implementation Requirements

### 1. Create README.md for Standard GPU Deployment

Create a new README.md with proper Hugging Face Spaces configuration:

```yaml
---
title: PolyID
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.48.0"
app_file: app.py
python_version: "3.11"
pinned: false
license: bsd-3-clause
short_description: PolyID polymer property prediction using graph neural networks
---
```

Include content explaining this is a Standard GPU Spaces deployment with full chemistry package support and modern Python 3.11 for optimal performance.

### 2. Create app.py for Standard GPU

Create the main Gradio application file. You'll need to:

**Import the core PolyID functionality:**
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polyid import MultiModel, SingleModel
import gradio as gr
```

**Set up standard GPU allocation** (no special decorators needed - HF Spaces automatically provides GPU)

**Implement the prediction interface** following the original PolyID pattern but optimized for Gradio.

### 3. Create requirements.txt with latest stable versions

Create requirements.txt using optimal modern versions for Python 3.11:

```txt
# PolyID Standard GPU Dependencies - Latest Stable for Python 3.11
gradio>=5.48.0
torch>=2.4.0
transformers>=4.45.0
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.4.0
scipy>=1.11.0
tensorflow>=2.16.0
networkx>=3.2.0
tqdm>=4.66.0
shortuuid>=1.0.11

# Chemistry and molecular packages (latest versions)
rdkit>=2024.3.1
nfp>=0.4.0
m2p>=0.2.0
```

### 4. Create packages.txt with comprehensive system dependencies

Create packages.txt incorporating all system dependencies we discovered are needed:

```txt
# Standard system dependencies
libboost-dev
libcairo2-dev
libeigen3-dev
libgomp1
python3-dev
build-essential

# Chemistry package dependencies (learned from investigation)
cmake
pkg-config
libboost-python-dev
libboost-serialization-dev
libboost-system-dev
libboost-thread-dev

# Graphics and display libraries
libxrender1
libfontconfig1
libice6
libsm6
libxext6
libxrandr2
libxss1
```

### 5. Ensure Direct Package Imports (No Defensive Imports Needed)

Since Standard GPU Spaces should support all packages, use direct imports in your Python files:

**In `src/polyid/polyid.py`:**
```python
import nfp
from nfp import EdgeUpdate, GlobalUpdate, NodeUpdate, masked_mean_absolute_error
import shortuuid
```

**Optional Enhancement**: You could add defensive imports as a best practice for robustness, but they're not required like they were for ZeroGPU.

### 6. Select Appropriate GPU Hardware

For PolyID deployment, choose GPU hardware based on your needs:

**Recommended Options:**
- **Nvidia T4 - small** ($0.40/hour): Good for development and light usage
- **Nvidia A10G - small** ($1.00/hour): Better for production with more models
- **Nvidia A100 - large** ($4.00/hour): Best performance for heavy workloads

**Hardware Selection Process:**
1. Go to Space Settings
2. Select "Hardware" tab
3. Choose appropriate GPU tier
4. Note: Standard GPU provides better chemistry package compatibility than ZeroGPU

### 7. Update docs/CLAUDE.md (Optional)

If you want to update the project documentation:
- Change Python version reference to "3.11"
- Update development commands to reflect standard GPU deployment
- Note modern package versions and GPU hardware selection

### 8. Testing Strategy

Create comprehensive tests to validate:
- All chemistry packages import successfully
- RDKit molecular processing works
- NFP neural fingerprint functionality works
- Full polymer property prediction pipeline
- API endpoint functionality

## Success Criteria

1. **All dependencies install successfully** in standard GPU environment
2. **Chemistry packages function correctly** (RDKit, NFP, m2p)
3. **Real polymer predictions** instead of mock mode
4. **Clean deployment** without dependency errors
5. **Full functionality restored** for polymer property prediction

## Implementation Steps

1. Start by examining the current clean fork state
2. Create README.md with Standard GPU configuration
3. Create app.py for Gradio interface with standard GPU
4. Create requirements.txt with proven dependency versions
5. Create packages.txt with comprehensive system dependencies
6. Ensure all PolyID Python files use direct imports
7. Test the implementation locally if possible
8. Deploy to Standard GPU Spaces and validate
9. Commit the working implementation

## Notes

- This approach prioritizes **functionality over cost optimization**
- Standard GPU Spaces provide full compatibility without restrictions
- Incorporates all lessons learned from ZeroGPU investigation
- Uses proven dependency versions and configurations
- No defensive imports needed (but can be added for robustness)

Please implement this step by step, testing each component before moving to the next step.
```

---

## Implementation Checklist

- [ ] Examine clean fork state and current structure
- [ ] Create README.md with Standard GPU configuration
- [ ] Create app.py for Gradio interface (no ZeroGPU decorators)
- [ ] Create requirements.txt with proven dependency versions
- [ ] Create packages.txt with comprehensive system dependencies
- [ ] Verify all PolyID Python files use direct imports
- [ ] Test locally if possible
- [ ] Deploy to Standard GPU Spaces
- [ ] Validate full chemistry stack functionality
- [ ] Test polymer property prediction pipeline

## Expected Results

- **Full chemistry package support**: RDKit, NFP, m2p working correctly
- **Real polymer predictions**: Complete PolyID functionality
- **Reliable deployment**: No dependency installation failures
- **Clean implementation**: Built from scratch with proven patterns

## Troubleshooting

If issues arise:
1. Check Hugging Face Spaces logs for specific errors
2. Verify standard GPU hardware is selected in Spaces settings
3. Refer to proven dependency versions from our investigation
4. Review packages.txt against comprehensive system dependencies

---

**Branch**: `standard-gpu-deployment`
**Approach**: Fresh implementation for Standard GPU Spaces
**Priority**: Full functionality using proven configurations