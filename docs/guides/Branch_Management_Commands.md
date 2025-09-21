# Branch Management Commands

## Overview

This document provides the exact commands to preserve your ZeroGPU troubleshooting work and set up clean branches for the two implementation approaches.

## Current Situation

- **Current main branch**: Contains 9 commits of ZeroGPU dependency troubleshooting
- **Target clean state**: Commit `55577cf` - "Merge branch 'tf2.14' into master" (good fork state)
- **Goal**: Preserve all work while creating clean starting points for new approaches

## Step-by-Step Commands

### 1. Preserve Current ZeroGPU Work

```bash
# Create a branch to preserve all ZeroGPU troubleshooting work
git checkout -b stale-zerogpu-attempts main

# Verify this branch contains all your recent work
git log --oneline -10
# Should show commits: 9515bc0, 0572dc4, 0d97cab, 76b7d17, etc.
```

### 2. Reset Main to Clean Fork State

```bash
# Switch back to main
git checkout main

# Reset to the clean fork state (before ZeroGPU troubleshooting)
git reset --hard 55577cf

# Verify clean state
git log --oneline -5
# Should show: 55577cf Merge branch 'tf2.14' into master
```

### 3. Create Implementation Branches

```bash
# Create Option 1: Standard GPU deployment branch
git checkout -b standard-gpu-deployment main

# Create Option 3: Container deployment branch
git checkout -b container-deployment main

# Verify all branches exist
git branch -a
# Should show:
# * container-deployment
#   main
#   standard-gpu-deployment
#   stale-zerogpu-attempts
```

### 4. Push All Branches to Repository

```bash
# Push the stale branch (preserve work)
git push origin stale-zerogpu-attempts

# Push the implementation branches
git push origin standard-gpu-deployment
git push origin container-deployment

# Force push the reset main branch (CAREFUL!)
git push --force-with-lease origin main
```

## Branch Purposes

### `stale-zerogpu-attempts`
- **Purpose**: Preserve all ZeroGPU troubleshooting work
- **Content**: Contains all dependency fixes, defensive imports, cache invalidation attempts
- **Use**: Reference for learning, documentation, potential future use

### `standard-gpu-deployment`
- **Purpose**: Implement Option 1 - Standard GPU Spaces
- **Starting point**: Clean fork state
- **Goal**: Remove ZeroGPU, enable full chemistry stack
- **Implementation guide**: `docs/Option1_Standard_GPU_Implementation.md`

### `container-deployment`
- **Purpose**: Implement Option 3 - Custom Docker Container
- **Starting point**: Clean fork state
- **Goal**: Pre-built container with all dependencies
- **Implementation guide**: `docs/Option3_Container_Deployment_Implementation.md`

### `main`
- **Purpose**: Clean state for future merging of successful approach
- **State**: Reset to clean fork point
- **Future**: Will receive merge from successful implementation branch

## Working with Branches

### To Work on Option 1 (Standard GPU):
```bash
git checkout standard-gpu-deployment
# Follow docs/Option1_Standard_GPU_Implementation.md
# Copy the Claude Code prompt from that document
```

### To Work on Option 3 (Container):
```bash
git checkout container-deployment
# Follow docs/Option3_Container_Deployment_Implementation.md
# Copy the Claude Code prompt from that document
```

### To Reference Previous Work:
```bash
git checkout stale-zerogpu-attempts
# Review what was tried for ZeroGPU
# Check specific commits for learning
git show 9515bc0  # Last comprehensive fix attempt
```

## Hugging Face Spaces Deployment

### For Standard GPU Branch:
```bash
git checkout standard-gpu-deployment
# After implementation...
git push space standard-gpu-deployment:main
```

### For Container Branch:
```bash
git checkout container-deployment
# After implementation...
git push space container-deployment:main
```

## Verification Commands

### Check Branch Status:
```bash
git branch -vv
# Shows all branches with last commits
```

### Compare Branches:
```bash
# See what's different between branches
git log stale-zerogpu-attempts..main --oneline
git log main..stale-zerogpu-attempts --oneline
```

### Check Remote Synchronization:
```bash
git remote -v
# Should show both GitHub and HF Spaces remotes
```

## Recovery Commands (If Needed)

### If Something Goes Wrong:
```bash
# You can always recover from the stale branch
git checkout stale-zerogpu-attempts
git checkout -b recovery-branch

# Or reset main back to previous state
git checkout main
git reset --hard stale-zerogpu-attempts
```

### Find Any Commit:
```bash
git reflog
# Shows all recent commits, even if "lost"
```

## Summary

After running these commands, you'll have:
- ✅ **Preserved work**: All ZeroGPU troubleshooting in `stale-zerogpu-attempts`
- ✅ **Clean starting points**: Two branches ready for implementation
- ✅ **Clear guidance**: Detailed implementation documents for each approach
- ✅ **Safe recovery**: Can always get back to any previous state

Choose your preferred approach and follow the corresponding implementation document!