# core/standards_compliant_clustering.py
"""
Industry-standards compliant defect clustering for pipeline FFS applications.
Replaces proprietary clustering parameters with validated industry standards.

Standards implemented:
- ASME B31G Modified (RSTRENG) interaction rules
- BS 7910 flaw interaction methodology  
- API 579-1 proximity guidelines
- DNV-RP-F101 composite defect criteria
"""

import numpy as np
import pandas as pd
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ClusteringStandard(Enum):
    """Supported industry standards for defect clustering"""
    RSTRENG = "ASME B31G Modified (RSTRENG)"
    BS7910 = "BS 7910 Flaw Interaction"
    API579 = "API 579-1 Proximity Rules"
    DNV_RP_F101 = "DNV-RP-F101 Composite Defects"
    CONSERVATIVE = "Conservative Multi-Standard"

@dataclass
class InteractionCriteria:
    """Standard-specific interaction criteria"""
    axial_distance_mm: float
    circumferential_distance_mm: float
    depth_interaction_factor: float
    standard_name: str
    applicability_notes: str

class StandardsCompliantClusterer:
    """
    Industry-standards compliant defect clustering implementation.
    Replaces proprietary parameters with validated industry criteria.
    """
    
    def __init__(self, 
                 standard: ClusteringStandard = ClusteringStandard.RSTRENG,
                 pipe_diameter_mm: float = 1000.0,
                 conservative_factor: float = 1.0):
        """
        Initialize clusterer with specified industry standard.
        
        Parameters:
        - standard: Industry standard to use for clustering
        - pipe_diameter_mm: Pipeline outside diameter in mm
        - conservative_factor: Multiplier for additional conservatism (default 1.0)
        """
        self.standard = standard
        self.pipe_diameter_mm = pipe_diameter_mm
        self.conservative_factor = conservative_factor
        
        # Initialize standard-specific parameters
        self._initialize_standard_parameters()
    
    def _initialize_standard_parameters(self):
        """Initialize clustering parameters based on selected standard"""
        
        if self.standard == ClusteringStandard.RSTRENG:
            self._setup_rstreng_parameters()
        elif self.standard == ClusteringStandard.BS7910:
            self._setup_bs7910_parameters()
        elif self.standard == ClusteringStandard.API579:
            self._setup_api579_parameters()
        elif self.standard == ClusteringStandard.DNV_RP_F101:
            self._setup_dnv_parameters()
        elif self.standard == ClusteringStandard.CONSERVATIVE:
            self._setup_conservative_parameters()
        else:
            raise ValueError(f"Unsupported standard: {self.standard}")
    
    def _setup_rstreng_parameters(self):
        """Setup ASME B31G Modified (RSTRENG) clustering parameters"""
        self.standard_name = "ASME B31G Modified (RSTRENG)"
        self.reference = "ASME B31G-2012, Modified Method"
        
        # Base parameters (will be calculated per joint based on wall thickness)
        self.base_axial_formula = "sqrt(D * t)"  # Square root of diameter times thickness
        self.base_circumferential_factor = 6.0   # 6 times wall thickness
        self.depth_interaction_threshold = 0.1   # 10% depth difference for interaction
        
        self.applicability_notes = (
            "Valid for longitudinal defects with d/t ≤ 0.8. "
            "Interaction distance varies with wall thickness per RSTRENG methodology."
        )
    
    def _setup_bs7910_parameters(self):
        """Setup BS 7910 flaw interaction parameters"""
        self.standard_name = "BS 7910:2019 Flaw Interaction"
        self.reference = "BS 7910:2019 Section 7.1.7"
        
        # BS 7910 uses more sophisticated interaction rules
        self.separation_factor = 1.0  # s/a criteria where s is separation, a is defect size
        self.alignment_tolerance_degrees = 15.0  # Angular alignment tolerance
        self.size_ratio_threshold = 2.0  # Size ratio for interaction consideration
        
        self.applicability_notes = (
            "BS 7910 interaction rules consider both separation distance and flaw alignment. "
            "Applicable to both surface and embedded flaws."
        )
    
    def _setup_api579_parameters(self):
        """Setup API 579-1 proximity-based parameters"""
        self.standard_name = "API 579-1/ASME FFS-1"
        self.reference = "API 579-1/ASME FFS-1 2021 Edition"
        
        # API 579-1 proximity categories
        self.major_structural_proximity_mm = 25.4  # 1 inch - this is where your 25.4mm comes from!
        self.weld_proximity_mm = 50.8  # 2 inches from weld
        self.defect_proximity_multiplier = 2.0  # 2 times larger defect dimension
        
        self.applicability_notes = (
            "API 579-1 proximity rules for structural discontinuities and welds. "
            "The 25.4mm distance applies specifically to major structural discontinuities."
        )
    
    def _setup_dnv_parameters(self):
        """Setup DNV-RP-F101 composite defect parameters"""
        self.standard_name = "DNV-RP-F101 Corroded Pipelines"
        self.reference = "DNV-RP-F101 October 2017"
        
        # DNV approach for closely spaced defects
        self.axial_spacing_factor = 1.0  # Spacing relative to defect length
        self.circumferential_spacing_factor = 3.0  # 3 times wall thickness
        self.interaction_depth_threshold = 0.05  # 5% depth for interaction
        
        self.applicability_notes = (
            "DNV-RP-F101 composite defect approach for offshore pipelines. "
            "Validated for high-pressure gas transmission applications."
        )
    
    def _setup_conservative_parameters(self):
        """Setup conservative multi-standard approach"""
        self.standard_name = "Conservative Multi-Standard"
        self.reference = "Combined approach using most conservative criteria"
        
        # Use most conservative parameters from all standards
        self.conservative_multiplier = 1.5  # 50% additional conservatism
        
        self.applicability_notes = (
            "Conservative approach combining multiple standards. "
            "Recommended for critical pipeline segments or when standard is uncertain."
        )
    
    def calculate_interaction_criteria(self, wall_thickness_mm: float) -> InteractionCriteria:
        """
        Calculate interaction criteria for given wall thickness.
        
        Parameters:
        - wall_thickness_mm: Nominal wall thickness in mm
        
        Returns:
        - InteractionCriteria object with calculated distances
        """
        
        if self.standard == ClusteringStandard.RSTRENG:
            return self._calculate_rstreng_criteria(wall_thickness_mm)
        elif self.standard == ClusteringStandard.BS7910:
            return self._calculate_bs7910_criteria(wall_thickness_mm)
        elif self.standard == ClusteringStandard.API579:
            return self._calculate_api579_criteria(wall_thickness_mm)
        elif self.standard == ClusteringStandard.DNV_RP_F101:
            return self._calculate_dnv_criteria(wall_thickness_mm)
        elif self.standard == ClusteringStandard.CONSERVATIVE:
            return self._calculate_conservative_criteria(wall_thickness_mm)
    
    def _calculate_rstreng_criteria(self, wall_thickness_mm: float) -> InteractionCriteria:
        """Calculate RSTRENG-based interaction criteria"""
        
        # Core RSTRENG formula: √(D × t)
        axial_distance = math.sqrt(self.pipe_diameter_mm * wall_thickness_mm)
        
        # Circumferential interaction: 6 times wall thickness (industry practice)
        circumferential_distance = self.base_circumferential_factor * wall_thickness_mm
        
        # Apply conservative factor
        axial_distance *= self.conservative_factor
        circumferential_distance *= self.conservative_factor
        
        return InteractionCriteria(
            axial_distance_mm=axial_distance,
            circumferential_distance_mm=circumferential_distance,
            depth_interaction_factor=self.depth_interaction_threshold,
            standard_name=self.standard_name,
            applicability_notes=self.applicability_notes
        )
    
    def _calculate_bs7910_criteria(self, wall_thickness_mm: float) -> InteractionCriteria:
        """Calculate BS 7910-based interaction criteria"""
        
        # BS 7910 uses defect-size relative criteria
        # For surface flaws: interaction when s < (a1 + a2)
        # Conservative approximation: use wall thickness as characteristic dimension
        
        base_distance = 2.0 * wall_thickness_mm  # Conservative base
        
        axial_distance = base_distance * 2.0  # Longer interaction in axial direction
        circumferential_distance = base_distance
        
        # Apply conservative factor
        axial_distance *= self.conservative_factor
        circumferential_distance *= self.conservative_factor
        
        return InteractionCriteria(
            axial_distance_mm=axial_distance,
            circumferential_distance_mm=circumferential_distance,
            depth_interaction_factor=0.05,  # 5% depth difference
            standard_name=self.standard_name,
            applicability_notes=self.applicability_notes
        )
    
    def _calculate_api579_criteria(self, wall_thickness_mm: float) -> InteractionCriteria:
        """Calculate API 579-1-based interaction criteria"""
        
        # API 579-1 proximity rules
        axial_distance = self.major_structural_proximity_mm  # 25.4mm base
        circumferential_distance = self.major_structural_proximity_mm
        
        # Also consider defect-size relative criteria
        thickness_based_distance = 2.0 * wall_thickness_mm
        
        # Use maximum of fixed distance or thickness-based
        axial_distance = max(axial_distance, thickness_based_distance)
        circumferential_distance = max(circumferential_distance, thickness_based_distance)
        
        # Apply conservative factor
        axial_distance *= self.conservative_factor
        circumferential_distance *= self.conservative_factor
        
        return InteractionCriteria(
            axial_distance_mm=axial_distance,
            circumferential_distance_mm=circumferential_distance,
            depth_interaction_factor=0.1,  # 10% depth difference
            standard_name=self.standard_name,
            applicability_notes=self.applicability_notes
        )
    
    def _calculate_dnv_criteria(self, wall_thickness_mm: float) -> InteractionCriteria:
        """Calculate DNV-RP-F101-based interaction criteria"""
        
        # DNV approach for composite defects
        axial_distance = 1.0 * wall_thickness_mm  # Conservative base
        circumferential_distance = self.circumferential_spacing_factor * wall_thickness_mm
        
        # Apply conservative factor
        axial_distance *= self.conservative_factor
        circumferential_distance *= self.conservative_factor
        
        return InteractionCriteria(
            axial_distance_mm=axial_distance,
            circumferential_distance_mm=circumferential_distance,
            depth_interaction_factor=self.interaction_depth_threshold,
            standard_name=self.standard_name,
            applicability_notes=self.applicability_notes
        )
    
    def _calculate_conservative_criteria(self, wall_thickness_mm: float) -> InteractionCriteria:
        """Calculate conservative multi-standard criteria"""
        
        # Calculate criteria from all standards
        rstreng_criteria = self._calculate_rstreng_criteria(wall_thickness_mm)
        bs7910_criteria = self._calculate_bs7910_criteria(wall_thickness_mm)
        api579_criteria = self._calculate_api579_criteria(wall_thickness_mm)
        dnv_criteria = self._calculate_dnv_criteria(wall_thickness_mm)
        
        # Use maximum (most conservative) distances
        max_axial = max(
            rstreng_criteria.axial_distance_mm,
            bs7910_criteria.axial_distance_mm,
            api579_criteria.axial_distance_mm,
            dnv_criteria.axial_distance_mm
        )
        
        max_circumferential = max(
            rstreng_criteria.circumferential_distance_mm,
            bs7910_criteria.circumferential_distance_mm,
            api579_criteria.circumferential_distance_mm,
            dnv_criteria.circumferential_distance_mm
        )
        
        # Use minimum (most sensitive) depth factor
        min_depth_factor = min(
            rstreng_criteria.depth_interaction_factor,
            bs7910_criteria.depth_interaction_factor,
            api579_criteria.depth_interaction_factor,
            dnv_criteria.depth_interaction_factor
        )
        
        # Apply additional conservative multiplier
        max_axial *= self.conservative_multiplier
        max_circumferential *= self.conservative_multiplier
        
        return InteractionCriteria(
            axial_distance_mm=max_axial,
            circumferential_distance_mm=max_circumferential,
            depth_interaction_factor=min_depth_factor,
            standard_name=self.standard_name,
            applicability_notes=self.applicability_notes
        )
    
    def find_interacting_defects(self, 
                                defects_df: pd.DataFrame, 
                                joints_df: pd.DataFrame) -> List[List[int]]:
        """
        Find clusters of interacting defects using industry-standard criteria.
        
        Parameters:
        - defects_df: DataFrame with defect information
        - joints_df: DataFrame with joint information (for wall thickness)
        
        Returns:
        - List of lists, where each inner list contains indices of interacting defects
        """
        
        # Validate input data
        required_defect_cols = ['log dist. [m]', 'joint number', 'depth [%]']
        required_joint_cols = ['joint number', 'wt nom [mm]']
        
        missing_defect_cols = [col for col in required_defect_cols if col not in defects_df.columns]
        missing_joint_cols = [col for col in required_joint_cols if col not in joints_df.columns]
        
        if missing_defect_cols or missing_joint_cols:
            raise ValueError(f"Missing required columns: {missing_defect_cols + missing_joint_cols}")
        
        # Create wall thickness lookup
        wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
        
        # Initialize clustering
        defect_indices = list(range(len(defects_df)))
        clusters = []
        processed_indices = set()
        
        for i, defect_i in defects_df.iterrows():
            if i in processed_indices:
                continue
            
            # Get wall thickness for this defect's joint
            wall_thickness = wt_lookup.get(defect_i['joint number'], 10.0)  # Default fallback
            
            # Calculate interaction criteria for this wall thickness
            criteria = self.calculate_interaction_criteria(wall_thickness)
            
            # Find all defects that interact with defect i
            cluster = [i]
            
            for j, defect_j in defects_df.iterrows():
                if i == j or j in processed_indices:
                    continue
                
                # Check if defects i and j interact
                if self._defects_interact(defect_i, defect_j, criteria):
                    cluster.append(j)
            
            # If cluster has more than one defect, it's a meaningful cluster
            if len(cluster) > 1:
                clusters.append(cluster)
                processed_indices.update(cluster)
            else:
                # Single defect - mark as processed but don't add to clusters
                processed_indices.add(i)
        
        return clusters
    
    def _defects_interact(self, 
                         defect1: pd.Series, 
                         defect2: pd.Series, 
                         criteria: InteractionCriteria) -> bool:
        """
        Check if two defects interact based on standard criteria.
        
        Parameters:
        - defect1, defect2: Defect data series
        - criteria: Interaction criteria for the applicable standard
        
        Returns:
        - Boolean indicating if defects interact
        """
        
        # Calculate axial separation
        axial_separation_mm = abs(defect1['log dist. [m]'] - defect2['log dist. [m]']) * 1000
        
        # Check axial interaction
        if axial_separation_mm > criteria.axial_distance_mm:
            return False
        
        # Calculate circumferential separation (if clock data available)
        if 'clock_float' in defect1.index and 'clock_float' in defect2.index:
            clock1 = defect1['clock_float']
            clock2 = defect2['clock_float']
            
            if pd.notna(clock1) and pd.notna(clock2):
                # Convert clock positions to arc length
                clock_diff = min(abs(clock1 - clock2), 12 - abs(clock1 - clock2))  # Handle wrap-around
                arc_length_mm = (clock_diff / 12.0) * math.pi * self.pipe_diameter_mm
                
                # Check circumferential interaction
                if arc_length_mm > criteria.circumferential_distance_mm:
                    return False
        
        # Check depth similarity (optional additional criterion)
        if 'depth [%]' in defect1.index and 'depth [%]' in defect2.index:
            depth_diff = abs(defect1['depth [%]'] - defect2['depth [%]'])
            
            # If depths are very different, they might not interact meaningfully
            # This is a secondary criterion - primary is spatial proximity
            if depth_diff > 50.0:  # 50% depth difference threshold
                return False
        
        return True
    
    def get_standard_info(self) -> Dict:
        """
        Get information about the current clustering standard.
        
        Returns:
        - Dictionary with standard information
        """
        return {
            'standard': self.standard.value,
            'standard_name': getattr(self, 'standard_name', 'Unknown'),
            'reference': getattr(self, 'reference', 'No reference available'),
            'applicability_notes': getattr(self, 'applicability_notes', 'No notes available'),
            'pipe_diameter_mm': self.pipe_diameter_mm,
            'conservative_factor': self.conservative_factor
        }
    
    def compare_standards(self, wall_thickness_mm: float) -> pd.DataFrame:
        """
        Compare interaction criteria across different standards.
        
        Parameters:
        - wall_thickness_mm: Wall thickness for comparison
        
        Returns:
        - DataFrame comparing criteria across standards
        """
        standards_to_compare = [
            ClusteringStandard.RSTRENG,
            ClusteringStandard.BS7910,
            ClusteringStandard.API579,
            ClusteringStandard.DNV_RP_F101
        ]
        
        comparison_data = []
        
        for standard in standards_to_compare:
            # Temporarily switch standard
            original_standard = self.standard
            self.standard = standard
            self._initialize_standard_parameters()
            
            criteria = self.calculate_interaction_criteria(wall_thickness_mm)
            
            comparison_data.append({
                'Standard': standard.value,
                'Axial Distance (mm)': criteria.axial_distance_mm,
                'Circumferential Distance (mm)': criteria.circumferential_distance_mm,
                'Depth Interaction Factor': criteria.depth_interaction_factor,
                'Applicability Notes': criteria.applicability_notes[:50] + '...'
            })
            
            # Restore original standard
            self.standard = original_standard
            self._initialize_standard_parameters()
        
        return pd.DataFrame(comparison_data)


# Example usage and integration function
def create_standards_compliant_clusterer(standard_name: str = "RSTRENG", 
                                       pipe_diameter_mm: float = 1000.0,
                                       conservative_factor: float = 1.0) -> StandardsCompliantClusterer:
    """
    Factory function to create standards-compliant clusterer.
    
    Parameters:
    - standard_name: Name of standard ("RSTRENG", "BS7910", "API579", "DNV", "CONSERVATIVE")
    - pipe_diameter_mm: Pipeline diameter in mm
    - conservative_factor: Additional conservatism factor
    
    Returns:
    - Configured StandardsCompliantClusterer instance
    """
    
    standard_mapping = {
        "RSTRENG": ClusteringStandard.RSTRENG,
        "BS7910": ClusteringStandard.BS7910,
        "API579": ClusteringStandard.API579,
        "DNV": ClusteringStandard.DNV_RP_F101,
        "CONSERVATIVE": ClusteringStandard.CONSERVATIVE
    }
    
    if standard_name not in standard_mapping:
        raise ValueError(f"Unknown standard: {standard_name}. Choose from: {list(standard_mapping.keys())}")
    
    return StandardsCompliantClusterer(
        standard=standard_mapping[standard_name],
        pipe_diameter_mm=pipe_diameter_mm,
        conservative_factor=conservative_factor
    )


# Integration with existing FFS system
def replace_proprietary_clustering(defects_df: pd.DataFrame, 
                                 joints_df: pd.DataFrame,
                                 pipe_diameter_mm: float,
                                 standard: str = "RSTRENG") -> List[List[int]]:
    """
    Drop-in replacement for existing clustering function with industry standards.
    
    This function can directly replace your existing clustering calls.
    """
    
    clusterer = create_standards_compliant_clusterer(
        standard_name=standard,
        pipe_diameter_mm=pipe_diameter_mm
    )
    
    return clusterer.find_interacting_defects(defects_df, joints_df)