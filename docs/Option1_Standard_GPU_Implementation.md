# Option 1: Standard GPU Spaces Implementation

## Implementation Overview

**Goal**: Migrate PolyID from ZeroGPU to Standard GPU Spaces to resolve dependency compatibility issues and restore full chemistry stack functionality.

**Key Changes**:
- Remove ZeroGPU decorators and constraints
- Update to standard GPU configuration
- Clean dependency management without ZeroGPU limitations
- Enable full RDKit, NFP, and chemistry package support

---

## Claude Code Prompt

Copy and paste this entire prompt to Claude Code when working on the `standard-gpu-deployment` branch:

```
I need you to implement Option 1: Standard GPU Spaces deployment for the PolyID project. This involves migrating from ZeroGPU to standard GPU Spaces to resolve dependency compatibility issues.

## Context

PolyID is a polymer property prediction application using graph neural networks. We've discovered that ZeroGPU Spaces have fundamental compatibility limitations with chemistry packages (RDKit, NFP, m2p) causing recurring dependency failures. The analysis is documented in `docs/ZeroGPU_Compatibility_Analysis.md`.

## Current State

- The project is currently configured for ZeroGPU with Python 3.10.13 exactly
- Dependencies fail to install due to ZeroGPU environment restrictions
- App falls back to mock mode instead of real functionality
- All chemistry packages (RDKit, NFP, m2p) fail to import

## Implementation Requirements

### 1. Update README.md Configuration

Change the frontmatter from:
```yaml
title: PolyID ZeroGPU
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.46.0"
app_file: app.py
python_version: "3.10.13"
pinned: false
license: bsd-3-clause
short_description: PolyID polymer property prediction with ZeroGPU acceleration
```

To:
```yaml
title: PolyID
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.46.0"
app_file: app.py
python_version: "3.10"
pinned: false
license: bsd-3-clause
short_description: PolyID polymer property prediction using graph neural networks
```

Update the description to remove ZeroGPU references and update Python version flexibility.

### 2. Modify app.py

Remove ZeroGPU decorators and imports:

**Remove these lines:**
```python
import spaces
@spaces.GPU
```

**Keep the core functionality** but remove GPU decorator from the prediction function. The function should work with standard GPU allocation.

**Update any ZeroGPU-specific comments** to reflect standard GPU usage.

### 3. Clean requirements.txt

Create a clean requirements.txt focused on compatibility without ZeroGPU restrictions:

```txt
# PolyID Standard GPU Dependencies
gradio>=5.46.0
torch>=2.1.0
transformers>=4.20.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
tensorflow>=2.12.0
networkx>=2.6.0
tqdm>=4.60.0
shortuuid>=1.0.0

# Chemistry and molecular packages
rdkit>=2023.9.1
nfp>=0.3.0
m2p>=0.1.0
```

### 4. Update packages.txt

Ensure comprehensive system dependencies for chemistry packages:

```txt
# Standard system dependencies
libboost-dev
libcairo2-dev
libeigen3-dev
libgomp1
python3-dev
build-essential

# Chemistry package dependencies
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

### 5. Remove Defensive Imports

Since we're moving to standard GPU Spaces with full package support, remove the defensive try/catch imports and mock fallbacks from:
- `src/polyid/polyid.py`
- `src/polyid/models/base_models.py`
- `src/polyid/models/tacticity_models.py`
- `src/polyid/preprocessors/preprocessors.py`

Restore direct imports for RDKit, NFP, and other packages.

### 6. Update Documentation

Update `docs/CLAUDE.md`:
- Remove ZeroGPU references
- Update Python version to be flexible (3.10+)
- Update development commands to reflect standard GPU deployment
- Remove mock mode as primary option (it becomes backup only)

### 7. Testing Strategy

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

1. Start by reading the current configuration files to understand the changes needed
2. Update README.md frontmatter and content
3. Modify app.py to remove ZeroGPU decorators
4. Clean up requirements.txt and packages.txt
5. Remove defensive imports and restore direct package imports
6. Update documentation to reflect standard GPU deployment
7. Test the changes to ensure full functionality
8. Commit the implementation with clear description

## Notes

- This approach prioritizes **functionality over cost optimization**
- Standard GPU Spaces may have resource costs but provide full compatibility
- The goal is to restore PolyID to its full intended functionality
- All chemistry packages should work without restrictions

Please implement this step by step, ensuring each change is properly tested before moving to the next step.
```

---

## Implementation Checklist

- [ ] Update README.md frontmatter and content
- [ ] Remove ZeroGPU decorators from app.py
- [ ] Clean requirements.txt without ZeroGPU limitations
- [ ] Update packages.txt with comprehensive dependencies
- [ ] Remove defensive imports from all Python files
- [ ] Update docs/CLAUDE.md
- [ ] Test full chemistry stack functionality
- [ ] Verify API endpoint functionality
- [ ] Deploy and validate on standard GPU Spaces
- [ ] Update project documentation

## Expected Results

- **Full chemistry package support**: RDKit, NFP, m2p working correctly
- **Real polymer predictions**: No more mock mode fallbacks
- **Reliable deployment**: No recurring dependency failures
- **Complete functionality**: All PolyID features working as intended

## Troubleshooting

If issues arise:
1. Check Hugging Face Spaces logs for specific errors
2. Verify standard GPU hardware is selected in Spaces settings
3. Ensure all packages in requirements.txt are compatible with Python 3.10
4. Review packages.txt for missing system dependencies

---

**Branch**: `standard-gpu-deployment`
**Approach**: Clean migration from ZeroGPU to Standard GPU Spaces
**Priority**: Full functionality over cost optimization