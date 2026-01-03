"""
PaleoBond Chemistry Tools for Claude Agent SDK
Copy these tool definitions to your PaleoBond Agent SDK implementation.

Dependencies:
    pip install claude-agent-sdk rdkit pandas numpy

Usage:
    from agent_sdk_tools import chemistry_server, CHEMISTRY_TOOLS

    # In your Agent SDK setup:
    options = ClaudeAgentOptions(
        mcp_servers={"chemistry": chemistry_server},
        allowed_tools=[f"mcp__chemistry__{t}" for t in CHEMISTRY_TOOLS]
    )
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Import the chemistry utilities
from chemistry_utils import (
    validate_smiles,
    get_molecular_descriptors,
    get_extended_descriptors,
    get_fingerprint_bits,
    DomainOfValidity,
    tanimoto_similarity,
    find_similar_molecules,
    analyze_molecule,
    generate_polymer_id,
    estimate_polymer_properties
)

# Try to import Agent SDK - if not available, provide mock for testing
try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False
    # Mock decorators for testing without SDK
    def tool(name, description, params):
        def decorator(func):
            func._tool_name = name
            func._tool_description = description
            func._tool_params = params
            return func
        return decorator

    def create_sdk_mcp_server(**kwargs):
        return {"mock": True, **kwargs}


# =============================================================================
# GLOBAL STATE - Domain of Validity Model
# =============================================================================

# Load or initialize DoV model
_DOV_MODEL: Optional[DomainOfValidity] = None


def get_dov_model() -> DomainOfValidity:
    """Get or create the DoV model."""
    global _DOV_MODEL
    if _DOV_MODEL is None:
        _DOV_MODEL = DomainOfValidity()
        # Pre-fit with some common polymer monomers
        # Replace with your actual training data
        default_training = [
            "C=CC(=O)OC",  # Methyl acrylate
            "C=CC(=O)OCC",  # Ethyl acrylate
            "CC(=C)C(=O)OC",  # Methyl methacrylate
            "C=Cc1ccccc1",  # Styrene
            "C=C",  # Ethylene
            "CC=C",  # Propylene
            "C=CCl",  # Vinyl chloride
            "C=CC#N",  # Acrylonitrile
            "CC(C)(C)OC(=O)C=C",  # tert-Butyl acrylate
        ]
        _DOV_MODEL.fit(default_training)
    return _DOV_MODEL


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@tool(
    "validate_smiles",
    "Validate a SMILES string and check if it represents a valid molecular structure. Returns validation status and canonical SMILES.",
    {"smiles": str}
)
async def validate_smiles_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate SMILES string and return canonical form."""
    smiles = args.get("smiles", "")

    is_valid, message, mol = validate_smiles(smiles)

    result = {
        "input_smiles": smiles,
        "is_valid": is_valid,
        "message": message
    }

    if is_valid and mol:
        from rdkit import Chem
        result["canonical_smiles"] = Chem.MolToSmiles(mol)
        result["polymer_id"] = generate_polymer_id(smiles)

    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


@tool(
    "get_molecular_descriptors",
    "Calculate molecular descriptors (MW, LogP, TPSA, H-bond donors/acceptors, etc.) for a molecule from its SMILES string.",
    {"smiles": str}
)
async def get_descriptors_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate molecular descriptors."""
    smiles = args.get("smiles", "")

    descriptors = get_extended_descriptors(smiles)

    if descriptors is None:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": f"Invalid SMILES: {smiles}"})
            }]
        }

    return {"content": [{"type": "text", "text": json.dumps(descriptors, indent=2)}]}


@tool(
    "analyze_polymer",
    "Comprehensive analysis of a polymer molecule including validation, descriptors, fingerprints, and property estimates.",
    {"smiles": str}
)
async def analyze_polymer_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive polymer analysis."""
    smiles = args.get("smiles", "")

    analysis = analyze_molecule(smiles)

    return {"content": [{"type": "text", "text": json.dumps(analysis, indent=2)}]}


@tool(
    "check_domain_of_validity",
    "Check if a molecule is within the domain of validity for reliable predictions. Returns confidence score and coverage metrics.",
    {"smiles": str}
)
async def check_dov_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Check domain of validity for a molecule."""
    smiles = args.get("smiles", "")

    dov = get_dov_model()
    score = dov.score(smiles)

    result = {
        "smiles": smiles,
        **score,
        "recommendation": (
            "Predictions should be reliable" if score.get("in_domain")
            else "Predictions may be unreliable - molecule is outside training domain"
        )
    }

    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


@tool(
    "calculate_similarity",
    "Calculate Tanimoto similarity between two molecules using Morgan fingerprints.",
    {"smiles1": str, "smiles2": str}
)
async def calculate_similarity_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate molecular similarity."""
    smiles1 = args.get("smiles1", "")
    smiles2 = args.get("smiles2", "")

    similarity = tanimoto_similarity(smiles1, smiles2)

    if similarity is None:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": "One or both SMILES are invalid"})
            }]
        }

    result = {
        "smiles1": smiles1,
        "smiles2": smiles2,
        "tanimoto_similarity": round(similarity, 4),
        "interpretation": (
            "Very similar" if similarity > 0.85 else
            "Similar" if similarity > 0.7 else
            "Moderately similar" if similarity > 0.5 else
            "Different"
        )
    }

    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


@tool(
    "find_similar_polymers",
    "Find the most similar polymers from a reference set to a query molecule.",
    {"query_smiles": str, "reference_smiles": list, "top_k": int}
)
async def find_similar_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Find similar molecules from reference set."""
    query = args.get("query_smiles", "")
    references = args.get("reference_smiles", [])
    top_k = args.get("top_k", 5)

    if not references:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": "reference_smiles list is required"})
            }]
        }

    similar = find_similar_molecules(query, references, top_k=top_k)

    result = {
        "query_smiles": query,
        "top_matches": [
            {"smiles": s, "similarity": round(sim, 4)}
            for s, sim in similar
        ]
    }

    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


@tool(
    "estimate_polymer_properties",
    "Estimate polymer properties (Tg, flexibility, hydrophilicity) from molecular descriptors. NOTE: These are heuristic estimates - use for screening only.",
    {"smiles": str}
)
async def estimate_properties_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate polymer properties from descriptors."""
    smiles = args.get("smiles", "")

    descriptors = get_extended_descriptors(smiles)
    if descriptors is None:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": f"Invalid SMILES: {smiles}"})
            }]
        }

    estimates = estimate_polymer_properties(descriptors)

    result = {
        "smiles": smiles,
        "polymer_id": generate_polymer_id(smiles),
        "estimates": estimates,
        "descriptors_used": {
            "molecular_weight": descriptors["molecular_weight"],
            "logP": descriptors["logP"],
            "aromatic_rings": descriptors["aromatic_rings"],
            "rotatable_bonds": descriptors["rotatable_bonds"]
        }
    }

    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


@tool(
    "batch_analyze_polymers",
    "Analyze multiple polymers in batch. Returns analysis for each SMILES.",
    {"smiles_list": list}
)
async def batch_analyze_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Batch analyze multiple polymers."""
    smiles_list = args.get("smiles_list", [])

    if not smiles_list:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": "smiles_list is required"})
            }]
        }

    results = []
    for smiles in smiles_list[:50]:  # Limit to 50 for performance
        analysis = analyze_molecule(smiles)
        results.append(analysis)

    summary = {
        "total": len(results),
        "valid": sum(1 for r in results if r.get("is_valid")),
        "invalid": sum(1 for r in results if not r.get("is_valid"))
    }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "summary": summary,
                "results": results
            }, indent=2)
        }]
    }


# =============================================================================
# MCP SERVER CREATION
# =============================================================================

# List of all tool names for allowed_tools configuration
CHEMISTRY_TOOLS = [
    "validate_smiles",
    "get_molecular_descriptors",
    "analyze_polymer",
    "check_domain_of_validity",
    "calculate_similarity",
    "find_similar_polymers",
    "estimate_polymer_properties",
    "batch_analyze_polymers"
]

# Create the MCP server with all tools
if AGENT_SDK_AVAILABLE:
    chemistry_server = create_sdk_mcp_server(
        name="chemistry",
        version="1.0.0",
        tools=[
            validate_smiles_tool,
            get_descriptors_tool,
            analyze_polymer_tool,
            check_dov_tool,
            calculate_similarity_tool,
            find_similar_tool,
            estimate_properties_tool,
            batch_analyze_tool
        ]
    )
else:
    chemistry_server = None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

EXAMPLE_USAGE = '''
# =============================================================================
# HOW TO USE THESE TOOLS IN PALEOBOND
# =============================================================================

# 1. Copy chemistry_utils.py and agent_sdk_tools.py to your PaleoBond project

# 2. Install dependencies:
#    pip install claude-agent-sdk rdkit pandas numpy

# 3. In your PaleoBond Agent SDK setup:

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from agent_sdk_tools import chemistry_server, CHEMISTRY_TOOLS

async def main():
    options = ClaudeAgentOptions(
        system_prompt="""You are a polymer chemistry expert assistant for PaleoBond.

        When analyzing polymers:
        1. First validate the SMILES using validate_smiles
        2. Get molecular descriptors using get_molecular_descriptors
        3. Check domain of validity using check_domain_of_validity
        4. Estimate properties using estimate_polymer_properties

        For preservation applications, evaluate:
        - UV stability (higher aromatic content = better)
        - Flexibility (more rotatable bonds = more flexible)
        - Hydrophilicity (lower logP = more hydrophilic)

        Always provide confidence assessments based on DoV scores.""",

        mcp_servers={"chemistry": chemistry_server},

        allowed_tools=[
            f"mcp__chemistry__{tool}" for tool in CHEMISTRY_TOOLS
        ]
    )

    async with ClaudeSDKClient(options=options) as client:
        # Example: Analyze a polymer
        await client.send("Analyze this polymer for fossil preservation: CC(C)(C)OC(=O)C=C")

        async for msg in client.receive_response():
            print(msg)


# 4. Run your agent:
import anyio
anyio.run(main)

# =============================================================================
# ALLOWED TOOLS LIST (copy this to your config)
# =============================================================================

ALLOWED_CHEMISTRY_TOOLS = [
    "mcp__chemistry__validate_smiles",
    "mcp__chemistry__get_molecular_descriptors",
    "mcp__chemistry__analyze_polymer",
    "mcp__chemistry__check_domain_of_validity",
    "mcp__chemistry__calculate_similarity",
    "mcp__chemistry__find_similar_polymers",
    "mcp__chemistry__estimate_polymer_properties",
    "mcp__chemistry__batch_analyze_polymers"
]
'''


if __name__ == "__main__":
    print(EXAMPLE_USAGE)

    # Test the tools without Agent SDK
    import asyncio

    async def test_tools():
        print("\n" + "=" * 60)
        print("Testing Tools")
        print("=" * 60)

        # Test validate_smiles
        result = await validate_smiles_tool({"smiles": "CC(C)(C)OC(=O)C=C"})
        print("\n1. validate_smiles:")
        print(result["content"][0]["text"])

        # Test get_molecular_descriptors
        result = await get_descriptors_tool({"smiles": "CC(C)(C)OC(=O)C=C"})
        print("\n2. get_molecular_descriptors:")
        print(result["content"][0]["text"])

        # Test check_domain_of_validity
        result = await check_dov_tool({"smiles": "CC(C)(C)OC(=O)C=C"})
        print("\n3. check_domain_of_validity:")
        print(result["content"][0]["text"])

        # Test estimate_polymer_properties
        result = await estimate_properties_tool({"smiles": "CC(C)(C)OC(=O)C=C"})
        print("\n4. estimate_polymer_properties:")
        print(result["content"][0]["text"])

    asyncio.run(test_tools())
