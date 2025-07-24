# core/enhanced_ffs_clustering.py
"""
Enhanced FFS clustering with stress concentration factors and standards compliance.
Integrates with existing corrosion assessment and failure prediction systems.

This module replaces the existing proprietary clustering and adds:
1. Industry-standard interaction rules
2. Stress concentration factors for interacting defects
3. Integration with existing ERF calculations
"""

import pandas as pd
from typing import List
from dataclasses import dataclass
from .standards_compliant_clustering import create_standards_compliant_clusterer

@dataclass
class ClusterProperties:
    """Properties of a clustered defect for stress analysis"""
    defect_indices: List[int]
    combined_length_mm: float
    combined_width_mm: float
    max_depth_pct: float
    center_location_m: float
    stress_concentration_factor: float
    interaction_type: str
    standard_used: str

class EnhancedFFSClusterer:
    """
    Enhanced FFS clusterer with stress concentration factors.
    Integrates standards-compliant clustering with failure prediction.
    """
    
    def __init__(self, 
                 standard: str = "BS7910",
                 pipe_diameter_mm: float = 1000.0,
                 conservative_factor: float = 1.0,
                 include_stress_concentration: bool = True):
        """
        Initialize enhanced FFS clusterer.
        
        Parameters:
        - standard: Industry standard for clustering ("BS7910", "API579", "DNV")
        - pipe_diameter_mm: Pipeline outside diameter in mm
        - conservative_factor: Additional conservatism factor
        - include_stress_concentration: Whether to apply stress concentration factors
        """
        
        self.clusterer = create_standards_compliant_clusterer(
            standard_name=standard,
            pipe_diameter_mm=pipe_diameter_mm,
            conservative_factor=conservative_factor
        )
        
        self.pipe_diameter_mm = pipe_diameter_mm
        self.include_stress_concentration = include_stress_concentration
        self.standard = standard
    
    def create_combined_defects_dataframe(self, defects_df: pd.DataFrame, clusters: List[ClusterProperties]) -> pd.DataFrame:
        """
        Create a new DataFrame with combined defects for assessment.
        
        Parameters:
        - defects_df: Original defects DataFrame
        - clusters: List of cluster properties
        
        Returns:
        - DataFrame with combined defects replacing individual clustered defects
        """


        # If no actual clusters (all single defects), return original DataFrame
        real_clusters = [c for c in clusters if len(c.defect_indices) > 1]
        
        if not real_clusters:
            # No clustering needed - return original with minimal modifications
            result_df = defects_df.copy()
            result_df['is_combined'] = False
            result_df['stress_concentration_factor'] = 1.0
            return result_df
        
        combined_defects = []
        processed_indices = set()
        
        # Add combined defects for each cluster
        for i, cluster in enumerate(clusters):
            combined_defect = {
                'log dist. [m]': cluster.center_location_m,
                'length [mm]': cluster.combined_length_mm,
                'width [mm]': cluster.combined_width_mm,
                'depth [%]': cluster.max_depth_pct,
                'cluster_id': f'cluster_{i}',
                'is_combined': True,
                'stress_concentration_factor': cluster.stress_concentration_factor,
                'interaction_type': cluster.interaction_type,
                'original_defect_count': len(cluster.defect_indices),
                'original_defect_indices': cluster.defect_indices
            }
            
            # Copy other relevant columns from the first defect in cluster
            first_defect = defects_df.iloc[cluster.defect_indices[0]]
            for col in ['joint number', 'surface location', 'component / anomaly identification']:
                if col in defects_df.columns:
                    combined_defect[col] = first_defect[col]
            
            combined_defects.append(combined_defect)
            processed_indices.update(cluster.defect_indices)
        
        # Add remaining individual defects
        for idx, defect in defects_df.iterrows():
            if idx not in processed_indices:
                defect_dict = defect.to_dict()
                defect_dict.update({
                    'cluster_id': f'individual_{idx}',
                    'is_combined': False,
                    'stress_concentration_factor': 1.0,
                    'interaction_type': 'isolated',
                    'original_defect_count': 1,
                    'original_defect_indices': [idx]
                })
                combined_defects.append(defect_dict)
        
        return pd.DataFrame(combined_defects)