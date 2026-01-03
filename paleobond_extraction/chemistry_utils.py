"""
PaleoBond Chemistry Utilities
Minimal extraction from PolyID for use with Claude Agent SDK

Dependencies:
    pip install rdkit pandas numpy

Usage:
    from chemistry_utils import (
        validate_smiles,
        get_molecular_descriptors,
        DomainOfValidity,
        get_morgan_fingerprint
    )
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib
import json

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


# =============================================================================
# SMILES Validation
# =============================================================================

def validate_smiles(smiles: str) -> Tuple[bool, str, Optional[Chem.Mol]]:
    """
    Validate a SMILES string and return the RDKit molecule object.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, message, mol_object)
    """
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES must be a non-empty string", None

    smiles = smiles.strip()

    # Check for obviously invalid characters
    invalid_chars = ['<', '>', '|', '{', '}', '\\\\']
    for char in invalid_chars:
        if char in smiles:
            return False, f"SMILES contains invalid character: {char}", None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "RDKit could not parse SMILES", None

        # Get canonical SMILES
        canonical = Chem.MolToSmiles(mol)
        return True, f"Valid SMILES (canonical: {canonical})", mol

    except Exception as e:
        return False, f"SMILES parsing error: {str(e)}", None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form."""
    is_valid, _, mol = validate_smiles(smiles)
    if is_valid and mol:
        return Chem.MolToSmiles(mol)
    return None


# =============================================================================
# Molecular Descriptors
# =============================================================================

@dataclass
class MolecularDescriptors:
    """Container for computed molecular descriptors."""
    molecular_weight: float
    logP: float
    tpsa: float  # Topological Polar Surface Area
    rotatable_bonds: int
    hbd: int  # Hydrogen Bond Donors
    hba: int  # Hydrogen Bond Acceptors
    aromatic_rings: int
    heavy_atoms: int
    fraction_sp3: float
    num_rings: int
    molecular_formula: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecular_weight": round(self.molecular_weight, 2),
            "logP": round(self.logP, 2),
            "tpsa": round(self.tpsa, 2),
            "rotatable_bonds": self.rotatable_bonds,
            "hydrogen_bond_donors": self.hbd,
            "hydrogen_bond_acceptors": self.hba,
            "aromatic_rings": self.aromatic_rings,
            "heavy_atoms": self.heavy_atoms,
            "fraction_sp3": round(self.fraction_sp3, 3),
            "num_rings": self.num_rings,
            "molecular_formula": self.molecular_formula
        }


def get_molecular_descriptors(smiles: str) -> Optional[MolecularDescriptors]:
    """
    Calculate molecular descriptors for a SMILES string.

    Args:
        smiles: Valid SMILES string

    Returns:
        MolecularDescriptors object or None if invalid
    """
    is_valid, _, mol = validate_smiles(smiles)
    if not is_valid or mol is None:
        return None

    return MolecularDescriptors(
        molecular_weight=Descriptors.MolWt(mol),
        logP=Descriptors.MolLogP(mol),
        tpsa=Descriptors.TPSA(mol),
        rotatable_bonds=Descriptors.NumRotatableBonds(mol),
        hbd=Descriptors.NumHDonors(mol),
        hba=Descriptors.NumHAcceptors(mol),
        aromatic_rings=Descriptors.NumAromaticRings(mol),
        heavy_atoms=Descriptors.HeavyAtomCount(mol),
        fraction_sp3=Descriptors.FractionCSP3(mol),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol)
    )


def get_extended_descriptors(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Calculate extended set of molecular descriptors for property prediction.

    Args:
        smiles: Valid SMILES string

    Returns:
        Dictionary of descriptors or None if invalid
    """
    is_valid, _, mol = validate_smiles(smiles)
    if not is_valid or mol is None:
        return None

    return {
        # Basic properties
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
        "logP": round(Descriptors.MolLogP(mol), 2),
        "mr": round(Descriptors.MolMR(mol), 2),  # Molar Refractivity

        # Topological
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "labute_asa": round(Descriptors.LabuteASA(mol), 2),

        # Counts
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "saturated_rings": Descriptors.NumSaturatedRings(mol),
        "heteroatoms": Descriptors.NumHeteroatoms(mol),

        # Complexity
        "fraction_sp3": round(Descriptors.FractionCSP3(mol), 3),
        "num_stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),

        # Electronic
        "formal_charge": Chem.GetFormalCharge(mol),
        "num_radical_electrons": Descriptors.NumRadicalElectrons(mol),

        # Formula
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "canonical_smiles": Chem.MolToSmiles(mol)
    }


# =============================================================================
# Morgan Fingerprints
# =============================================================================

def get_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    as_bitvector: bool = True
) -> Optional[Any]:
    """
    Generate Morgan fingerprint for a molecule.

    Args:
        smiles: Valid SMILES string
        radius: Fingerprint radius (default 2 = ECFP4 equivalent)
        n_bits: Number of bits for folded fingerprint
        as_bitvector: If True, return bit vector; if False, return count vector

    Returns:
        Fingerprint object or None if invalid
    """
    is_valid, _, mol = validate_smiles(smiles)
    if not is_valid or mol is None:
        return None

    if as_bitvector:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        return AllChem.GetMorganFingerprint(mol, radius, useFeatures=False)


def get_fingerprint_bits(smiles: str, radius: int = 2) -> Optional[Dict[int, int]]:
    """
    Get Morgan fingerprint as dictionary of bit positions to counts.
    Useful for domain of validity analysis.

    Args:
        smiles: Valid SMILES string
        radius: Fingerprint radius

    Returns:
        Dictionary mapping bit positions to counts, or None if invalid
    """
    is_valid, _, mol = validate_smiles(smiles)
    if not is_valid or mol is None:
        return None

    fp = AllChem.GetMorganFingerprint(mol, radius, useFeatures=False)
    return dict(fp.GetNonzeroElements())


# =============================================================================
# Domain of Validity (DoV)
# =============================================================================

class DomainOfValidity:
    """
    Assess whether a molecule is within the domain of validity
    based on fingerprint similarity to training data.

    Usage:
        dov = DomainOfValidity()
        dov.fit(training_smiles_list)
        score = dov.score("CCO")
    """

    def __init__(self, radius: int = 2):
        """
        Initialize DoV checker.

        Args:
            radius: Morgan fingerprint radius
        """
        self.radius = radius
        self.training_bits: set = set()
        self.training_bit_counts: Dict[int, int] = {}
        self.is_fitted = False

    def fit(self, smiles_list: List[str]) -> "DomainOfValidity":
        """
        Fit the DoV model on training SMILES.

        Args:
            smiles_list: List of SMILES strings from training data

        Returns:
            self
        """
        all_bits: Dict[int, int] = {}

        for smiles in smiles_list:
            bits = get_fingerprint_bits(smiles, self.radius)
            if bits:
                for bit, count in bits.items():
                    all_bits[bit] = all_bits.get(bit, 0) + count

        self.training_bits = set(all_bits.keys())
        self.training_bit_counts = all_bits
        self.is_fitted = True

        return self

    def score(self, smiles: str) -> Dict[str, Any]:
        """
        Calculate domain of validity score for a molecule.

        Args:
            smiles: SMILES string to evaluate

        Returns:
            Dictionary with DoV metrics
        """
        if not self.is_fitted:
            raise ValueError("DoV model not fitted. Call fit() first.")

        query_bits = get_fingerprint_bits(smiles, self.radius)

        if query_bits is None:
            return {
                "is_valid_smiles": False,
                "dov_score": 0.0,
                "in_domain": False,
                "novel_bits": 0,
                "total_bits": 0,
                "coverage": 0.0
            }

        query_bit_set = set(query_bits.keys())

        # Count bits not in training data
        novel_bits = query_bit_set - self.training_bits
        covered_bits = query_bit_set & self.training_bits

        total_bits = len(query_bit_set)
        novel_count = len(novel_bits)
        coverage = len(covered_bits) / total_bits if total_bits > 0 else 0.0

        # DoV score: 1.0 = fully in domain, 0.0 = fully out of domain
        dov_score = coverage

        # Threshold for "in domain" - can be adjusted
        in_domain = dov_score >= 0.7 and novel_count <= 3

        return {
            "is_valid_smiles": True,
            "dov_score": round(dov_score, 3),
            "in_domain": in_domain,
            "novel_bits": novel_count,
            "total_bits": total_bits,
            "coverage": round(coverage, 3),
            "confidence": "high" if dov_score >= 0.9 else "medium" if dov_score >= 0.7 else "low"
        }

    def save(self, filepath: str) -> None:
        """Save DoV model to file."""
        data = {
            "radius": self.radius,
            "training_bits": list(self.training_bits),
            "training_bit_counts": self.training_bit_counts
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "DomainOfValidity":
        """Load DoV model from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        dov = cls(radius=data["radius"])
        dov.training_bits = set(data["training_bits"])
        dov.training_bit_counts = data["training_bit_counts"]
        dov.is_fitted = True

        return dov


# =============================================================================
# Tanimoto Similarity
# =============================================================================

def tanimoto_similarity(smiles1: str, smiles2: str, radius: int = 2) -> Optional[float]:
    """
    Calculate Tanimoto similarity between two molecules.

    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        radius: Morgan fingerprint radius

    Returns:
        Similarity score (0-1) or None if either SMILES is invalid
    """
    from rdkit import DataStructs

    fp1 = get_morgan_fingerprint(smiles1, radius)
    fp2 = get_morgan_fingerprint(smiles2, radius)

    if fp1 is None or fp2 is None:
        return None

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def find_similar_molecules(
    query_smiles: str,
    reference_smiles: List[str],
    top_k: int = 5,
    radius: int = 2
) -> List[Tuple[str, float]]:
    """
    Find most similar molecules from a reference set.

    Args:
        query_smiles: Query SMILES string
        reference_smiles: List of reference SMILES strings
        top_k: Number of top matches to return
        radius: Morgan fingerprint radius

    Returns:
        List of (smiles, similarity) tuples sorted by similarity
    """
    similarities = []

    for ref_smiles in reference_smiles:
        sim = tanimoto_similarity(query_smiles, ref_smiles, radius)
        if sim is not None:
            similarities.append((ref_smiles, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# =============================================================================
# Polymer-Specific Utilities
# =============================================================================

def generate_polymer_id(smiles: str) -> str:
    """Generate a unique ID for a polymer based on its SMILES."""
    canonical = canonicalize_smiles(smiles) or smiles
    hash_bytes = hashlib.md5(canonical.encode()).hexdigest()[:8].upper()
    return f"POLY-{hash_bytes}"


def estimate_polymer_properties(descriptors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate polymer properties from molecular descriptors.
    These are rough heuristics - use ML models for accurate predictions.

    Args:
        descriptors: Dictionary from get_extended_descriptors()

    Returns:
        Dictionary of estimated polymer properties
    """
    mw = descriptors.get("molecular_weight", 100)
    logP = descriptors.get("logP", 0)
    tpsa = descriptors.get("tpsa", 50)
    aromatic = descriptors.get("aromatic_rings", 0)
    rotatable = descriptors.get("rotatable_bonds", 0)
    hbd = descriptors.get("hbd", 0)
    hba = descriptors.get("hba", 0)

    # Rough heuristics based on polymer science principles
    # These are NOT accurate predictions - use trained models!

    # Glass transition temperature (very rough estimate)
    tg_base = 50 + (mw / 10) + (aromatic * 30) - (rotatable * 5)
    tg_estimate = max(-100, min(300, tg_base))

    # Flexibility score (0-1)
    flexibility = min(1.0, max(0.0, 0.5 + (rotatable * 0.05) - (aromatic * 0.1)))

    # Hydrophilicity (based on logP and TPSA)
    hydrophilicity = min(1.0, max(0.0, 0.5 - (logP * 0.1) + (tpsa / 200)))

    return {
        "estimated_tg_celsius": round(tg_estimate, 1),
        "flexibility_score": round(flexibility, 2),
        "hydrophilicity_score": round(hydrophilicity, 2),
        "note": "These are heuristic estimates only. Use trained ML models for accurate predictions."
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_molecule(smiles: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of a molecule.

    Args:
        smiles: SMILES string to analyze

    Returns:
        Complete analysis dictionary
    """
    is_valid, message, mol = validate_smiles(smiles)

    result = {
        "input_smiles": smiles,
        "is_valid": is_valid,
        "validation_message": message
    }

    if not is_valid:
        return result

    result["canonical_smiles"] = Chem.MolToSmiles(mol)
    result["polymer_id"] = generate_polymer_id(smiles)
    result["descriptors"] = get_extended_descriptors(smiles)
    result["fingerprint_bits"] = len(get_fingerprint_bits(smiles) or {})

    if result["descriptors"]:
        result["property_estimates"] = estimate_polymer_properties(result["descriptors"])

    return result


if __name__ == "__main__":
    # Example usage
    test_smiles = "CC(C)(C)OC(=O)C=C"  # tert-butyl acrylate

    print("=" * 60)
    print("Chemistry Utils - Example Usage")
    print("=" * 60)

    # Validate SMILES
    is_valid, msg, mol = validate_smiles(test_smiles)
    print(f"\nValidation: {msg}")

    # Get descriptors
    desc = get_molecular_descriptors(test_smiles)
    if desc:
        print(f"\nMolecular Weight: {desc.molecular_weight:.2f}")
        print(f"LogP: {desc.logP:.2f}")
        print(f"Formula: {desc.molecular_formula}")

    # Full analysis
    print("\nFull Analysis:")
    analysis = analyze_molecule(test_smiles)
    print(json.dumps(analysis, indent=2))

    # DoV example
    print("\nDomain of Validity Example:")
    training_smiles = ["CCO", "CCCO", "CCCCO", "CC(C)O", "c1ccccc1"]
    dov = DomainOfValidity()
    dov.fit(training_smiles)

    score = dov.score(test_smiles)
    print(f"DoV Score: {score['dov_score']}")
    print(f"In Domain: {score['in_domain']}")
    print(f"Confidence: {score['confidence']}")
