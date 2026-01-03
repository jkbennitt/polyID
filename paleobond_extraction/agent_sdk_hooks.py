"""
PaleoBond Agent SDK Hooks for Chemistry Workflows
Copy these hooks to your PaleoBond Agent SDK implementation.

Hooks provide deterministic control over agent behavior:
- PreToolUse: Validate inputs before tool execution
- PostToolUse: Process outputs and provide feedback

Usage:
    from agent_sdk_hooks import (
        validate_smiles_hook,
        domain_of_validity_hook,
        log_predictions_hook,
        HOOK_CONFIG
    )
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import chemistry utilities
from chemistry_utils import validate_smiles, DomainOfValidity, get_dov_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paleobond.hooks")


# =============================================================================
# PRE-TOOL-USE HOOKS
# =============================================================================

async def validate_smiles_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Pre-tool hook to validate SMILES before any chemistry tool runs.

    This hook intercepts tool calls that include a 'smiles' parameter
    and validates the SMILES string before allowing execution.

    Args:
        input_data: Tool invocation data including tool_name and tool_input
        tool_use_id: Unique ID for this tool use
        context: Additional context from the agent

    Returns:
        Hook response - empty dict to allow, or deny decision
    """
    tool_input = input_data.get("tool_input", {})

    # Check for smiles parameter
    smiles = tool_input.get("smiles") or tool_input.get("query_smiles")

    if not smiles:
        # No SMILES to validate, allow the call
        return {}

    # Validate the SMILES
    is_valid, message, _ = validate_smiles(smiles)

    if not is_valid:
        logger.warning(f"Hook blocked invalid SMILES: {smiles} - {message}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Invalid SMILES: {message}"
            }
        }

    logger.debug(f"Hook validated SMILES: {smiles}")
    return {}


async def batch_size_limit_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Pre-tool hook to limit batch sizes for performance.

    Prevents excessively large batch requests that could timeout.
    """
    tool_input = input_data.get("tool_input", {})
    smiles_list = tool_input.get("smiles_list", [])

    MAX_BATCH_SIZE = 50

    if len(smiles_list) > MAX_BATCH_SIZE:
        logger.warning(f"Hook blocked batch of {len(smiles_list)} (max: {MAX_BATCH_SIZE})")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Batch size {len(smiles_list)} exceeds maximum of {MAX_BATCH_SIZE}"
            }
        }

    return {}


async def dangerous_smiles_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Pre-tool hook to block potentially problematic molecular structures.

    This is a safety check for known problematic patterns.
    """
    tool_input = input_data.get("tool_input", {})
    smiles = tool_input.get("smiles", "")

    # Block very long SMILES (likely malformed or attacks)
    if len(smiles) > 1000:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "SMILES string too long (max 1000 characters)"
            }
        }

    return {}


# =============================================================================
# POST-TOOL-USE HOOKS
# =============================================================================

async def domain_of_validity_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Post-tool hook to check domain of validity after predictions.

    Provides feedback to Claude about prediction reliability.
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    tool_output = input_data.get("tool_output", "")

    # Only check for prediction tools
    prediction_tools = ["estimate_polymer_properties", "analyze_polymer"]
    if not any(t in tool_name for t in prediction_tools):
        return {}

    smiles = tool_input.get("smiles", "")
    if not smiles:
        return {}

    # Check DoV
    try:
        dov = get_dov_model()
        dov_result = dov.score(smiles)

        if not dov_result.get("in_domain", True):
            feedback = (
                f"⚠️ DOMAIN OF VALIDITY WARNING: The molecule '{smiles}' "
                f"has a DoV score of {dov_result.get('dov_score', 0):.2f} "
                f"(confidence: {dov_result.get('confidence', 'unknown')}). "
                f"Predictions may be unreliable. Consider flagging this to the user."
            )
            logger.info(f"DoV warning for {smiles}: score={dov_result.get('dov_score')}")

            return {
                "hookSpecificOutput": {
                    "feedback": feedback
                }
            }
    except Exception as e:
        logger.error(f"DoV check failed: {e}")

    return {}


async def log_predictions_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Post-tool hook to log all predictions for monitoring/analytics.

    Logs prediction details to a file or database for later analysis.
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Only log prediction-related tools
    if "chemistry" not in tool_name:
        return {}

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "input_smiles": tool_input.get("smiles", tool_input.get("query_smiles", "")),
        "batch_size": len(tool_input.get("smiles_list", [])) if "smiles_list" in tool_input else 1
    }

    # Log to file (replace with database in production)
    logger.info(f"Prediction logged: {json.dumps(log_entry)}")

    return {}


async def add_uncertainty_disclaimer_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Post-tool hook to add uncertainty disclaimers to property estimates.

    Ensures Claude always caveats heuristic predictions appropriately.
    """
    tool_name = input_data.get("tool_name", "")

    if "estimate_polymer_properties" in tool_name:
        return {
            "hookSpecificOutput": {
                "feedback": (
                    "Remember: These property estimates are based on heuristics, "
                    "not trained ML models. Always communicate uncertainty to users "
                    "and recommend experimental validation for critical applications."
                )
            }
        }

    return {}


# =============================================================================
# HOOK CONFIGURATION
# =============================================================================

# Copy this configuration to your ClaudeAgentOptions

HOOK_CONFIG = {
    "PreToolUse": [
        {
            "matcher": "mcp__chemistry__*",
            "hooks": [
                validate_smiles_hook,
                dangerous_smiles_hook
            ],
            "timeout": 30
        },
        {
            "matcher": "mcp__chemistry__batch_analyze_polymers",
            "hooks": [batch_size_limit_hook],
            "timeout": 10
        }
    ],
    "PostToolUse": [
        {
            "matcher": "mcp__chemistry__estimate_polymer_properties",
            "hooks": [
                domain_of_validity_hook,
                add_uncertainty_disclaimer_hook,
                log_predictions_hook
            ],
            "timeout": 30
        },
        {
            "matcher": "mcp__chemistry__analyze_polymer",
            "hooks": [
                domain_of_validity_hook,
                log_predictions_hook
            ],
            "timeout": 30
        }
    ]
}


# =============================================================================
# EXAMPLE INTEGRATION
# =============================================================================

EXAMPLE_INTEGRATION = '''
# =============================================================================
# HOW TO USE HOOKS IN PALEOBOND
# =============================================================================

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher
from agent_sdk_hooks import (
    validate_smiles_hook,
    batch_size_limit_hook,
    dangerous_smiles_hook,
    domain_of_validity_hook,
    log_predictions_hook,
    add_uncertainty_disclaimer_hook
)
from agent_sdk_tools import chemistry_server, CHEMISTRY_TOOLS

async def main():
    options = ClaudeAgentOptions(
        system_prompt="You are a polymer chemistry expert for PaleoBond...",

        mcp_servers={"chemistry": chemistry_server},

        allowed_tools=[
            f"mcp__chemistry__{tool}" for tool in CHEMISTRY_TOOLS
        ],

        hooks={
            "PreToolUse": [
                # Validate all chemistry tools
                HookMatcher(
                    matcher="mcp__chemistry__*",
                    hooks=[validate_smiles_hook, dangerous_smiles_hook],
                    timeout=30
                ),
                # Extra check for batch operations
                HookMatcher(
                    matcher="mcp__chemistry__batch_analyze_polymers",
                    hooks=[batch_size_limit_hook],
                    timeout=10
                )
            ],
            "PostToolUse": [
                # Add DoV warnings and disclaimers for predictions
                HookMatcher(
                    matcher="mcp__chemistry__estimate_polymer_properties",
                    hooks=[
                        domain_of_validity_hook,
                        add_uncertainty_disclaimer_hook,
                        log_predictions_hook
                    ],
                    timeout=30
                ),
                HookMatcher(
                    matcher="mcp__chemistry__analyze_polymer",
                    hooks=[domain_of_validity_hook, log_predictions_hook],
                    timeout=30
                )
            ]
        }
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.send("Analyze CC(C)(C)OC(=O)C=C for fossil preservation")
        async for msg in client.receive_response():
            print(msg)

import anyio
anyio.run(main)
'''


if __name__ == "__main__":
    print(EXAMPLE_INTEGRATION)
