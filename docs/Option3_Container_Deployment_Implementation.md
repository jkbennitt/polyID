# Option 3: Custom Container Deployment Implementation

## Implementation Overview

**Goal**: Deploy PolyID using a custom Docker container with pre-installed chemistry packages to bypass Hugging Face Spaces dependency installation limitations entirely.

**Key Benefits**:
- Reproducible environment with all dependencies pre-installed
- Bypasses any Spaces installation restrictions
- Most reliable approach for complex chemistry stack
- Complete control over the runtime environment

---

## Claude Code Prompt

Copy and paste this entire prompt to Claude Code when working on the `container-deployment` branch:

```
I need you to implement Option 3: Custom Container Deployment for the PolyID project. This involves creating a custom Docker container with all chemistry dependencies pre-installed to bypass Hugging Face Spaces installation limitations.

## Context

PolyID is a polymer property prediction application requiring complex chemistry packages (RDKit, NFP, m2p) that have compatibility issues with Hugging Face Spaces environments. The analysis is documented in `docs/ZeroGPU_Compatibility_Analysis.md`.

The container approach pre-builds all dependencies, ensuring a reliable, reproducible environment that works regardless of Spaces limitations.

## Current State

- Dependencies fail in both ZeroGPU and potentially standard Spaces
- Chemistry packages require complex system-level dependencies
- Need bulletproof solution that guarantees package availability

## Implementation Requirements

### 1. Create Dockerfile

Create a multi-stage Dockerfile that pre-installs all chemistry packages:

```dockerfile
# PolyID Custom Container
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libboost-dev \
    libboost-python-dev \
    libboost-serialization-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libcairo2-dev \
    libeigen3-dev \
    libgomp1 \
    python3-dev \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libsm6 \
    libxext6 \
    libxrandr2 \
    libxss1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create user for security
RUN useradd -m -u 1000 user
USER user

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
```

### 2. Update README.md for Container Deployment

Change the frontmatter to:
```yaml
title: PolyID
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
```

Add container-specific documentation explaining the Docker approach.

### 3. Create Container-Optimized requirements.txt

```txt
# PolyID Container Dependencies
# Pre-installed in Docker environment for reliability

# Web interface and deployment
gradio>=5.46.0

# Machine learning frameworks
torch>=2.1.0
tensorflow>=2.12.0
transformers>=4.20.0

# Scientific computing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
networkx>=2.6.0

# Utilities
tqdm>=4.60.0
shortuuid>=1.0.0

# Chemistry and molecular packages (pre-built in container)
rdkit>=2023.9.1
nfp>=0.3.0
m2p>=0.1.0

# Optional monitoring and debugging
psutil
```

### 4. Update app.py for Container Environment

Modify app.py to work optimally in container:

**Add container-specific configurations:**
```python
import os
import sys

# Container environment detection
def is_container():
    return os.path.exists('/.dockerenv') or os.environ.get('CONTAINER') == 'true'

# Configure for container deployment
if is_container():
    print("Running in container environment")
    # Container-specific optimizations
```

**Update Gradio interface:**
```python
# Launch configuration for container
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    interface = create_interface()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=False
    )
```

### 5. Create .dockerignore

Create `.dockerignore` to optimize build:
```
.git
.github
*.md
docs/
examples/
tests/
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.coverage
.env
```

### 6. Remove Defensive Imports

Since the container pre-installs all packages, remove defensive imports and mocks:
- Direct imports for all chemistry packages
- Remove try/catch blocks
- Remove mock fallback classes
- Restore full functionality without safety nets

### 7. Create Container Testing Script

Create `scripts/container/test_container.py`:
```python
#!/usr/bin/env python3
"""Test script for container deployment validation."""

def test_imports():
    """Test all critical package imports."""
    try:
        import rdkit
        print("✓ RDKit imported successfully")

        import nfp
        print("✓ NFP imported successfully")

        import m2p
        print("✓ m2p imported successfully")

        from polyid import MultiModel
        print("✓ PolyID package imported successfully")

        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_functionality():
    """Test core PolyID functionality."""
    # Add basic functionality tests
    pass

if __name__ == "__main__":
    print("Container Deployment Tests")
    print("=" * 30)

    if test_imports():
        print("\n✓ All imports successful")
        test_functionality()
    else:
        print("\n✗ Import tests failed")
        exit(1)
```

### 8. Update Documentation

Update `docs/CLAUDE.md` with container-specific guidance:
- Docker build and run commands
- Container development workflow
- Debugging container issues
- Container-specific testing

## Container Development Workflow

### Building the Container
```bash
# Build the container
docker build -t polyid:latest .

# Test locally
docker run -p 7860:7860 polyid:latest

# Test with mock mode
docker run -p 7860:7860 polyid:latest python app.py --mock
```

### Deployment to Hugging Face Spaces

1. Configure Space for Docker SDK
2. Ensure Dockerfile is in root directory
3. Push to repository - HF will build container automatically
4. Monitor build logs for any issues

## Success Criteria

1. **Container builds successfully** with all dependencies
2. **All chemistry packages work** in container environment
3. **Reproducible deployments** across any compatible environment
4. **No dependency installation failures** during runtime
5. **Full PolyID functionality** available in container

## Implementation Steps

1. Create Dockerfile with multi-stage build
2. Update README.md for Docker SDK
3. Optimize requirements.txt for container
4. Modify app.py for container deployment
5. Create .dockerignore for efficient builds
6. Remove defensive imports - restore direct package usage
7. Create container testing scripts
8. Update documentation with container workflow
9. Test locally with Docker
10. Deploy to Hugging Face Spaces with Docker SDK

## Notes

- **Most reliable approach** for complex dependency stacks
- **Reproducible environment** across development and production
- **Bypasses all Spaces installation limitations**
- **May have longer cold start times** due to container initialization
- **Requires Docker expertise** for troubleshooting

## Troubleshooting

If container issues arise:
1. Check Docker build logs for dependency installation errors
2. Test locally with `docker run` before deploying
3. Verify all system dependencies are in Dockerfile
4. Check container resource requirements vs Spaces limits
5. Monitor Hugging Face Spaces container logs

Please implement this step by step, testing the container locally before deploying to Hugging Face Spaces.
```

---

## Implementation Checklist

- [ ] Create multi-stage Dockerfile with chemistry packages
- [ ] Update README.md for Docker SDK configuration
- [ ] Create container-optimized requirements.txt
- [ ] Modify app.py for container environment
- [ ] Create .dockerignore for build optimization
- [ ] Remove defensive imports and restore direct package usage
- [ ] Create container testing scripts
- [ ] Update docs/CLAUDE.md with container workflow
- [ ] Test container build locally
- [ ] Test container functionality locally
- [ ] Deploy to HF Spaces with Docker SDK
- [ ] Validate deployed container functionality

## Expected Results

- **Bulletproof deployment**: All dependencies guaranteed to work
- **Reproducible environment**: Same behavior across all deployments
- **No installation failures**: Dependencies pre-built in container
- **Full chemistry stack**: RDKit, NFP, m2p working perfectly
- **Container portability**: Can run anywhere Docker is supported

## Advantages of Container Approach

1. **Reliability**: Eliminates dependency installation uncertainty
2. **Reproducibility**: Exact same environment every time
3. **Portability**: Works on any Docker-compatible platform
4. **Control**: Complete control over runtime environment
5. **Debugging**: Can test exact deployment environment locally

## Considerations

- **Build time**: Initial container build may take longer
- **Image size**: Container may be larger than standard deployment
- **Cold starts**: Container initialization may add startup time
- **Complexity**: Requires Docker knowledge for maintenance

---

**Branch**: `container-deployment`
**Approach**: Pre-built Docker container with all dependencies
**Priority**: Maximum reliability and reproducibility