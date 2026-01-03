#!/usr/bin/env python3
"""
PaleoBond + Claude Agent SDK Complete Example
==============================================

This is a complete, runnable example showing how to integrate
chemistry tools with the Claude Agent SDK for PaleoBond.

Copy this entire file to get started quickly.

Requirements:
    pip install claude-agent-sdk rdkit pandas numpy

Usage:
    python complete_example.py
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# =============================================================================
# CHEMISTRY UTILITIES (from chemistry_utils.py)
# =============================================================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


def validate_smiles(smiles: str):
    """Validate SMILES and return (is_valid, message, mol)."""
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES must be a non-empty string", None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return False, "RDKit could not parse SMILES", None
        return True, f"Valid (canonical: {Chem.MolToSmiles(mol)})", mol
    except Exception as e:
        return False, f"Parse error: {e}", None


def get_molecular_descriptors(smiles: str) -> Optional[Dict]:
    """Calculate molecular descriptors."""
    is_valid, _, mol = validate_smiles(smiles)
    if not is_valid:
        return None
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logP": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "fraction_sp3": round(Descriptors.FractionCSP3(mol), 3),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "canonical_smiles": Chem.MolToSmiles(mol)
    }


class DomainOfValidity:
    """Check if molecules are within prediction domain."""

    def __init__(self, radius: int = 2):
        self.radius = radius
        self.training_bits = set()
        self.is_fitted = False

    def fit(self, smiles_list: List[str]):
        for smiles in smiles_list:
            is_valid, _, mol = validate_smiles(smiles)
            if is_valid:
                fp = AllChem.GetMorganFingerprint(mol, self.radius)
                self.training_bits.update(fp.GetNonzeroElements().keys())
        self.is_fitted = True
        return self

    def score(self, smiles: str) -> Dict:
        is_valid, _, mol = validate_smiles(smiles)
        if not is_valid:
            return {"dov_score": 0.0, "in_domain": False, "confidence": "invalid"}

        fp = AllChem.GetMorganFingerprint(mol, self.radius)
        query_bits = set(fp.GetNonzeroElements().keys())

        if not query_bits:
            return {"dov_score": 0.0, "in_domain": False, "confidence": "empty"}

        coverage = len(query_bits & self.training_bits) / len(query_bits)
        novel = len(query_bits - self.training_bits)

        return {
            "dov_score": round(coverage, 3),
            "in_domain": coverage >= 0.7 and novel <= 3,
            "novel_bits": novel,
            "confidence": "high" if coverage >= 0.9 else "medium" if coverage >= 0.7 else "low"
        }


# Global DoV model
_DOV = DomainOfValidity().fit([
    "C=CC(=O)OC", "C=CC(=O)OCC", "CC(=C)C(=O)OC",
    "C=Cc1ccccc1", "C=C", "CC=C", "C=CCl"
])


# =============================================================================
# AGENT SDK TOOLS
# =============================================================================

try:
    from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeSDKClient, ClaudeAgentOptions, HookMatcher
    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False
    print("⚠️  claude-agent-sdk not installed. Running in mock mode.")
    print("   Install with: pip install claude-agent-sdk")

    def tool(name, desc, params):
        def decorator(func):
            return func
        return decorator

    def create_sdk_mcp_server(**kwargs):
        return None


@tool("validate_smiles", "Validate a SMILES string", {"smiles": str})
async def validate_smiles_tool(args):
    smiles = args.get("smiles", "")
    is_valid, msg, mol = validate_smiles(smiles)
    result = {"smiles": smiles, "is_valid": is_valid, "message": msg}
    if is_valid:
        result["canonical"] = Chem.MolToSmiles(mol)
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


@tool("get_molecular_descriptors", "Calculate molecular descriptors", {"smiles": str})
async def get_descriptors_tool(args):
    desc = get_molecular_descriptors(args.get("smiles", ""))
    if desc is None:
        return {"content": [{"type": "text", "text": '{"error": "Invalid SMILES"}'}]}
    return {"content": [{"type": "text", "text": json.dumps(desc, indent=2)}]}


@tool("check_domain_of_validity", "Check prediction reliability", {"smiles": str})
async def check_dov_tool(args):
    result = _DOV.score(args.get("smiles", ""))
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


@tool("analyze_polymer", "Comprehensive polymer analysis", {"smiles": str})
async def analyze_polymer_tool(args):
    smiles = args.get("smiles", "")
    is_valid, msg, mol = validate_smiles(smiles)

    result = {"smiles": smiles, "is_valid": is_valid}
    if is_valid:
        result["descriptors"] = get_molecular_descriptors(smiles)
        result["domain_of_validity"] = _DOV.score(smiles)

    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


# Create MCP server
TOOL_LIST = [validate_smiles_tool, get_descriptors_tool, check_dov_tool, analyze_polymer_tool]

if AGENT_SDK_AVAILABLE:
    chemistry_server = create_sdk_mcp_server(
        name="chemistry",
        version="1.0.0",
        tools=TOOL_LIST
    )
else:
    chemistry_server = None


# =============================================================================
# HOOKS
# =============================================================================

async def validate_smiles_hook(input_data, tool_use_id, context):
    """Block invalid SMILES before tool execution."""
    smiles = input_data.get("tool_input", {}).get("smiles", "")
    if smiles:
        is_valid, msg, _ = validate_smiles(smiles)
        if not is_valid:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Invalid SMILES: {msg}"
                }
            }
    return {}


async def dov_warning_hook(input_data, tool_use_id, context):
    """Warn about low DoV scores after predictions."""
    smiles = input_data.get("tool_input", {}).get("smiles", "")
    if smiles:
        dov = _DOV.score(smiles)
        if not dov.get("in_domain"):
            return {
                "hookSpecificOutput": {
                    "feedback": f"⚠️ Low confidence (DoV: {dov['dov_score']:.2f})"
                }
            }
    return {}


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

async def run_agent_example():
    """Run the Agent SDK example."""
    if not AGENT_SDK_AVAILABLE:
        print("\n❌ Cannot run agent example - SDK not installed")
        print("   Running tool tests instead...\n")

        # Test tools directly
        result = await validate_smiles_tool({"smiles": "CC(C)(C)OC(=O)C=C"})
        print("validate_smiles result:")
        print(result["content"][0]["text"])

        result = await get_descriptors_tool({"smiles": "CC(C)(C)OC(=O)C=C"})
        print("\nget_molecular_descriptors result:")
        print(result["content"][0]["text"])

        result = await check_dov_tool({"smiles": "CC(C)(C)OC(=O)C=C"})
        print("\ncheck_domain_of_validity result:")
        print(result["content"][0]["text"])

        result = await analyze_polymer_tool({"smiles": "CC(C)(C)OC(=O)C=C"})
        print("\nanalyze_polymer result:")
        print(result["content"][0]["text"])
        return

    # Full Agent SDK example
    print("\n🚀 Running Agent SDK example...")

    options = ClaudeAgentOptions(
        system_prompt="""You are a polymer chemistry expert for PaleoBond.

When analyzing polymers:
1. First validate the SMILES
2. Get molecular descriptors
3. Check domain of validity
4. Provide preservation suitability assessment

Focus on properties relevant to fossil/opal/meteorite preservation.""",

        mcp_servers={"chemistry": chemistry_server},

        allowed_tools=[
            "mcp__chemistry__validate_smiles",
            "mcp__chemistry__get_molecular_descriptors",
            "mcp__chemistry__check_domain_of_validity",
            "mcp__chemistry__analyze_polymer"
        ],

        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher="mcp__chemistry__*",
                    hooks=[validate_smiles_hook],
                    timeout=30
                )
            ],
            "PostToolUse": [
                HookMatcher(
                    matcher="mcp__chemistry__analyze_polymer",
                    hooks=[dov_warning_hook],
                    timeout=30
                )
            ]
        }
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.send(
            "Analyze this polymer for fossil preservation applications: CC(C)(C)OC(=O)C=C"
        )

        async for msg in client.receive_response():
            print(msg)


# =============================================================================
# COPY-PASTE CONFIGURATIONS
# =============================================================================

COPY_PASTE_CONFIG = '''
# =============================================================================
# COPY-PASTE: requirements.txt
# =============================================================================

claude-agent-sdk
rdkit>=2023.9.1
pandas>=2.1.0
numpy>=1.26.0

# =============================================================================
# COPY-PASTE: Allowed Tools List
# =============================================================================

ALLOWED_TOOLS = [
    "mcp__chemistry__validate_smiles",
    "mcp__chemistry__get_molecular_descriptors",
    "mcp__chemistry__check_domain_of_validity",
    "mcp__chemistry__analyze_polymer",
]

# =============================================================================
# COPY-PASTE: System Prompt for Polymer Analysis
# =============================================================================

SYSTEM_PROMPT = """You are a polymer chemistry expert assistant for PaleoBond.

When analyzing polymers for preservation applications:

1. **Validation First**: Always validate SMILES before analysis
2. **Descriptors**: Get molecular weight, LogP, TPSA for property assessment
3. **Domain Check**: Verify predictions are reliable using DoV
4. **Application Fit**: Evaluate suitability for specific preservation use cases:
   - Fossil consolidation: Need penetration + strength
   - Opal stabilization: Need UV stability + water resistance
   - Pyrite prevention: Need oxygen barrier + acid resistance
   - Meteorite protection: Need oxidation prevention

Always communicate uncertainty when DoV confidence is low.
Recommend experimental validation for critical applications."""

# =============================================================================
# COPY-PASTE: Minimal Agent Setup
# =============================================================================

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def analyze_polymer(smiles: str):
    """Minimal function to analyze a polymer."""
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"chemistry": chemistry_server},
        allowed_tools=ALLOWED_TOOLS
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.send(f"Analyze this polymer: {smiles}")
        async for msg in client.receive_response():
            yield msg
'''


if __name__ == "__main__":
    print("=" * 70)
    print("PaleoBond + Claude Agent SDK - Complete Example")
    print("=" * 70)

    # Run the example
    asyncio.run(run_agent_example())

    print("\n" + "=" * 70)
    print("COPY-PASTE CONFIGURATIONS")
    print("=" * 70)
    print(COPY_PASTE_CONFIG)
