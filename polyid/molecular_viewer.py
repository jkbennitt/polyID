"""
PolyID - Interactive Molecular Structure Visualization
Enhanced molecular viewing capabilities using RDKit with performance optimizations
"""

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import threading
import time

# Import using our unified system
from polyid.imports import rdkit, Chem, Descriptors

class MolecularViewer:
    """High-performance interactive molecular structure visualization"""

    def __init__(self, image_size: Tuple[int, int] = (400, 400)):
        self.image_size = image_size
        self.cache = {}
        self._cache_lock = threading.Lock()
        
    def create_molecule_image(self, smiles: str, 
                            highlight_substructure: Optional[str] = None,
                            use_cache: bool = True) -> str:
        """Create optimized molecular structure image"""
        
        # Check cache first
        cache_key = f"{smiles}_{highlight_substructure}_{self.image_size}"
        if use_cache:
            with self._cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
        
        try:
            if not rdkit:
                return self._create_text_fallback("RDKit not available")
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._create_text_fallback("Invalid SMILES structure")

            # Generate 2D coordinates
            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(mol)

            # Handle substructure highlighting
            highlight_atoms = []
            highlight_bonds = []
            if highlight_substructure:
                pattern = Chem.MolFromSmarts(highlight_substructure)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    if matches:
                        highlight_atoms = list(matches[0])
                        highlight_bonds = self._get_highlight_bonds(mol, highlight_atoms)

            # Create high-quality image
            from rdkit.Chem import Draw
            img = Draw.MolToImage(
                mol,
                size=self.image_size,
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds,
                highlightColor=(0.8, 0.8, 1.0)  # Light blue highlighting
            )

            # Convert to base64 for web display
            img_str = self._image_to_base64(img)
            
            # Cache successful results
            if use_cache:
                with self._cache_lock:
                    self.cache[cache_key] = img_str
                    
            return img_str

        except Exception as e:
            return self._create_text_fallback(f"Visualization error: {str(e)}")

    def create_property_radar_chart(self, properties: Dict[str, Any]) -> plt.Figure:
        """Create enhanced radar chart showing predicted properties"""
        
        if not properties or all('error' in str(v) for v in properties.values()):
            return self._create_empty_plot("No valid properties to display")

        # Filter and normalize properties
        valid_props = {}
        for prop_name, prop_data in properties.items():
            if isinstance(prop_data, dict) and 'value' in prop_data:
                valid_props[prop_name] = prop_data

        if not valid_props:
            return self._create_empty_plot("No valid numerical properties")

        # Extract data for plotting
        prop_names = [self._shorten_property_name(name) for name in valid_props.keys()]
        values = []
        confidences = []
        
        for prop_data in valid_props.values():
            normalized_value = self._normalize_property_value(prop_data)
            values.append(normalized_value)
            confidences.append(prop_data.get('confidence', 'Medium'))

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(prop_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        values = values + [values[0]]  # Complete the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot the data
        line_color = self._get_confidence_color(confidences[0])  # Use first confidence as base
        ax.plot(angles, values, 'o-', linewidth=2.5, color=line_color, markersize=8)
        ax.fill(angles, values, alpha=0.25, color=line_color)

        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(prop_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Polymer Property Profile', size=16, weight='bold', pad=30)
        ax.grid(True, alpha=0.3)
        
        # Add confidence indicators
        confidence_text = ", ".join(set(confidences))
        ax.text(0.5, -0.1, f"Confidence: {confidence_text}", 
               transform=ax.transAxes, ha='center', fontsize=10)

        plt.tight_layout()
        return fig

    def create_property_comparison_chart(self, properties: Dict[str, Any]) -> plt.Figure:
        """Create horizontal bar chart comparing property values"""
        
        if not properties:
            return self._create_empty_plot("No properties to compare")

        # Extract valid properties
        valid_props = {}
        for prop_name, prop_data in properties.items():
            if isinstance(prop_data, dict) and 'value' in prop_data:
                valid_props[prop_name] = prop_data

        if not valid_props:
            return self._create_empty_plot("No valid numerical properties")

        # Prepare data
        prop_names = [self._shorten_property_name(name) for name in valid_props.keys()]
        values = [prop_data['value'] for prop_data in valid_props.values()]
        units = [prop_data.get('unit', '') for prop_data in valid_props.values()]
        confidences = [prop_data.get('confidence', 'Medium') for prop_data in valid_props.values()]

        # Create color mapping for confidence levels
        colors = [self._get_confidence_color(conf) for conf in confidences]

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(prop_names) * 0.8)))
        bars = ax.barh(prop_names, values, color=colors, alpha=0.8, height=0.6)

        # Add value labels on bars
        for i, (bar, value, unit, conf) in enumerate(zip(bars, values, units, confidences)):
            width = bar.get_width()
            label_x = width + abs(width) * 0.02
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f} {unit}', ha='left', va='center', 
                   fontweight='bold', fontsize=10)

        # Customize chart
        ax.set_xlabel('Property Values', fontsize=12, fontweight='bold')
        ax.set_title('Polymer Property Predictions', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add confidence legend
        from matplotlib.patches import Patch
        unique_confidences = list(set(confidences))
        legend_elements = [
            Patch(facecolor=self._get_confidence_color(conf), label=f'{conf} Confidence')
            for conf in unique_confidences
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

        plt.tight_layout()
        return fig

    def create_molecular_descriptors_chart(self, smiles: str) -> plt.Figure:
        """Create chart showing molecular descriptors"""
        
        try:
            if not rdkit:
                return self._create_empty_plot("RDKit not available for descriptors")
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._create_empty_plot("Invalid SMILES for descriptors")

            # Calculate molecular descriptors
            descriptors = {
                'Molecular Weight': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                'H-Bond Donors': Descriptors.NumHDonors(mol),
                'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
                'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            }

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            desc_names = list(descriptors.keys())
            desc_values = list(descriptors.values())
            
            # Normalize values for better visualization
            normalized_values = []
            for i, value in enumerate(desc_values):
                if desc_names[i] == 'Molecular Weight':
                    normalized_values.append(value / 100)  # Scale down MW
                elif desc_names[i] == 'LogP':
                    normalized_values.append((value + 5) * 2)  # Shift and scale LogP
                else:
                    normalized_values.append(value)

            bars = ax.bar(desc_names, normalized_values, 
                         color=plt.cm.viridis(np.linspace(0, 1, len(desc_names))),
                         alpha=0.8)

            # Add value labels
            for bar, original_value in zip(bars, desc_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{original_value:.1f}', ha='center', va='bottom', fontweight='bold')

            ax.set_ylabel('Descriptor Values (scaled)', fontweight='bold')
            ax.set_title(f'Molecular Descriptors for {smiles}', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            return fig

        except Exception as e:
            return self._create_empty_plot(f"Error calculating descriptors: {str(e)}")

    def _get_highlight_bonds(self, mol, highlight_atoms: List[int]) -> List[int]:
        """Get bonds to highlight based on highlighted atoms"""
        highlight_bonds = []
        atom_set = set(highlight_atoms)
        
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
                highlight_bonds.append(bond.GetIdx())
                
        return highlight_bonds

    def _image_to_base64(self, img) -> str:
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def _create_text_fallback(self, message: str) -> str:
        """Create a simple text-based fallback image"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               transform=ax.transAxes, fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def _create_empty_plot(self, message: str) -> plt.Figure:
        """Create an empty plot with a message"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    def _shorten_property_name(self, name: str) -> str:
        """Shorten property names for better display"""
        name_map = {
            "Glass Transition Temperature (Tg)": "Tg",
            "Melting Temperature (Tm)": "Tm", 
            "Density": "Density",
            "Elastic Modulus": "Modulus"
        }
        return name_map.get(name, name)

    def _normalize_property_value(self, prop_data: Dict[str, Any]) -> float:
        """Normalize property values to 0-1 scale for radar charts"""
        value = prop_data['value']
        unit = prop_data.get('unit', '')
        
        # Property-specific normalization ranges
        if 'Tg' in str(prop_data) or 'Glass Transition' in str(prop_data):
            return min(max((value - 200) / 300, 0), 1)  # Tg range ~200-500K
        elif 'Tm' in str(prop_data) or 'Melting' in str(prop_data):
            return min(max((value - 300) / 400, 0), 1)  # Tm range ~300-700K
        elif unit == 'g/cm³':
            return min(max((value - 0.5) / 2.0, 0), 1)  # Density range ~0.5-2.5
        elif unit == 'MPa':
            return min(max(value / 5000, 0), 1)  # Modulus range ~0-5000 MPa
        else:
            # Generic normalization
            return min(max(value / 1000, 0), 1)

    def _get_confidence_color(self, confidence: str) -> str:
        """Get color based on confidence level"""
        color_map = {
            'High': '#2E8B57',    # Sea Green
            'Medium': '#FF8C00',   # Dark Orange  
            'Low': '#DC143C'       # Crimson
        }
        return color_map.get(confidence, '#4682B4')  # Steel Blue default
        
    def clear_cache(self):
        """Clear the image cache"""
        with self._cache_lock:
            self.cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                'cache_size': len(self.cache),
                'cache_memory_est_mb': len(str(self.cache)) / 1024 / 1024
            }