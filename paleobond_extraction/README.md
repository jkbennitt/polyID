# PaleoBond Chemistry Extraction

Minimal extraction from PolyID for use with Claude Agent SDK.

## Files

| File | Description |
|------|-------------|
| `chemistry_utils.py` | Core chemistry utilities (validation, descriptors, DoV) |
| `agent_sdk_tools.py` | Claude Agent SDK tool definitions |
| `agent_sdk_hooks.py` | Pre/Post tool hooks for quality control |
| `complete_example.py` | Full working example |

## Quick Start

```bash
# 1. Install dependencies
pip install claude-agent-sdk rdkit pandas numpy

# 2. Copy files to your PaleoBond project
cp chemistry_utils.py agent_sdk_tools.py agent_sdk_hooks.py /path/to/paleobond/

# 3. Run the example
python complete_example.py
```

## Dependencies

```txt
claude-agent-sdk
rdkit>=2023.9.1
pandas>=2.1.0
numpy>=1.26.0
```

## Tool Reference

| Tool | Description |
|------|-------------|
| `validate_smiles` | Validate SMILES and get canonical form |
| `get_molecular_descriptors` | Calculate MW, LogP, TPSA, etc. |
| `analyze_polymer` | Comprehensive polymer analysis |
| `check_domain_of_validity` | Check prediction reliability |
| `calculate_similarity` | Tanimoto similarity between molecules |
| `find_similar_polymers` | Find similar molecules in reference set |
| `estimate_polymer_properties` | Heuristic property estimates |
| `batch_analyze_polymers` | Batch analysis (max 50) |

## Hook Reference

| Hook | Type | Purpose |
|------|------|---------|
| `validate_smiles_hook` | PreToolUse | Block invalid SMILES |
| `batch_size_limit_hook` | PreToolUse | Limit batch sizes |
| `dangerous_smiles_hook` | PreToolUse | Block problematic inputs |
| `domain_of_validity_hook` | PostToolUse | Warn about unreliable predictions |
| `log_predictions_hook` | PostToolUse | Log for analytics |
| `add_uncertainty_disclaimer_hook` | PostToolUse | Add caveats to estimates |
