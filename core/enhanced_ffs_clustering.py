# core/enhanced_ffs_clustering.py
"""
Enhanced FFS clustering with stress concentration factors and standards compliance.
Integrates with existing corrosion assessment and failure prediction systems.

This module replaces the existing proprietary clustering and adds:
1. Industry-standard interaction rules
2. Stress concentration factors for interacting defects
3. Integration with existing ERF calculations
"""

import numpy as np
import pandas as pd
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import the standards-compliant clusterer
from .standards_compliant_clustering import (
    StandardsCompliantClusterer, 
    create_standards_compliant_clusterer,
    ClusteringStandard
)

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
                 standard: str = "RSTRENG",
                 pipe_diameter_mm: float = 1000.0,
                 conservative_factor: float = 1.0,
                 include_stress_concentration: bool = True):
        """
        Initialize enhanced FFS clusterer.
        
        Parameters:
        - standard: Industry standard for clustering ("RSTRENG", "BS7910", "API579", "DNV", "CONSERVATIVE")
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


    def find_enhanced_clusters(self, 
                            defects_df: pd.DataFrame, 
                            joints_df: pd.DataFrame,
                            show_progress: bool = True) -> List[ClusterProperties]:
        """
        OPTIMIZED: Enhanced clustering with progress tracking and caching
        """
        import streamlit as st
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🔍 Starting enhanced clustering analysis...")
        
        # Step 1: Get basic clusters (30% of work)
        if show_progress:
            status_text.text("📊 Finding basic interaction clusters...")
        
        cluster_indices = self.clusterer.find_interacting_defects(
            defects_df, joints_df, show_progress=False  # Don't double-show progress
        )
        
        if show_progress:
            progress_bar.progress(0.3)
            status_text.text(f"✅ Found {len(cluster_indices)} basic clusters")
        
        # Step 2: Enhanced analysis (70% of work)
        enhanced_clusters = []
        total_clusters = len(cluster_indices)
        
        if total_clusters == 0:
            if show_progress:
                progress_bar.progress(1.0)
                status_text.text("ℹ️ No clusters found for enhancement")
            return enhanced_clusters
        
        # OPTIMIZATION: Pre-calculate common values
        defects_array = defects_df.values  # Faster access than pandas indexing
        
        for i, cluster_defect_indices in enumerate(cluster_indices):
            # Update progress
            if show_progress:
                cluster_progress = 0.3 + (0.7 * (i + 1) / total_clusters)
                progress_bar.progress(cluster_progress)
                status_text.text(f"🔧 Enhancing cluster {i+1}/{total_clusters}...")
            
            # Get defects in this cluster (use .iloc for speed)
            cluster_defects = defects_df.iloc[cluster_defect_indices]
            
            # Calculate enhanced properties
            cluster_props = self._calculate_cluster_properties(cluster_defects, cluster_defect_indices)
            enhanced_clusters.append(cluster_props)
        
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text(f"✅ Enhanced clustering complete: {len(enhanced_clusters)} clusters analyzed")
        
        return enhanced_clusters


    def _calculate_cluster_properties(self, cluster_defects: pd.DataFrame, defect_indices: List[int]) -> ClusterProperties:
        """Calculate enhanced properties for a cluster including stress concentration."""
        
        # Basic geometric properties
        combined_length = self._calculate_combined_length(cluster_defects)
        combined_width = self._calculate_combined_width(cluster_defects)
        max_depth = cluster_defects['depth [%]'].max()
        center_location = cluster_defects['log dist. [m]'].mean()
        
        # Calculate stress concentration factor
        if self.include_stress_concentration:
            stress_factor = self._calculate_stress_concentration_factor(cluster_defects)
        else:
            stress_factor = 1.0
        
        # Determine interaction type
        interaction_type = self._classify_interaction_type(cluster_defects)
        
        return ClusterProperties(
            defect_indices=defect_indices,
            combined_length_mm=combined_length,
            combined_width_mm=combined_width,
            max_depth_pct=max_depth,
            center_location_m=center_location,
            stress_concentration_factor=stress_factor,
            interaction_type=interaction_type,
            standard_used=self.standard
        )

    def _calculate_combined_length(self, cluster_defects: pd.DataFrame) -> float:
        """Calculate combined axial length using envelope method."""
        if len(cluster_defects) == 1:
            return cluster_defects['length [mm]'].iloc[0]
        
        start_positions = (cluster_defects['log dist. [m]'] * 1000 - cluster_defects['length [mm]'] / 2)
        end_positions = (cluster_defects['log dist. [m]'] * 1000 + cluster_defects['length [mm]'] / 2)
        return end_positions.max() - start_positions.min()


    def _calculate_combined_width(self, cluster_defects: pd.DataFrame) -> float:
        """Calculate combined circumferential width."""
        if len(cluster_defects) == 1:
            return cluster_defects['width [mm]'].iloc[0]
        return cluster_defects['width [mm]'].sum()


    def _calculate_stress_concentration_factor(self, cluster_defects: pd.DataFrame) -> float:
        """Calculate stress concentration factor for interacting defects."""
        num_defects = len(cluster_defects)
        if num_defects == 1:
            return 1.0
        
        import math
        kt_base = 1.0 + 0.15 * math.log(num_defects)
        spacing_factor = self._calculate_spacing_factor(cluster_defects)
        kt_spacing = 1.0 + 0.1 / spacing_factor
        
        max_depth = cluster_defects['depth [%]'].max()
        depth_factor = max_depth / 100.0
        kt_depth = 1.0 + 0.2 * depth_factor
        
        total_area = (cluster_defects['length [mm]'] * cluster_defects['width [mm]']).sum()
        area_factor = math.log10(max(total_area, 100)) / 3.0
        kt_size = 1.0 + 0.1 * area_factor
        
        kt_combined = math.pow(kt_base * kt_spacing * kt_depth * kt_size, 0.5)
        return min(max(kt_combined, 1.0), 2.5)
    

    def _calculate_spacing_factor(self, cluster_defects: pd.DataFrame) -> float:
        """Calculate average spacing factor for defects in cluster."""
        if len(cluster_defects) <= 1:
            return 1.0
        
        import numpy as np
        spacings = []
        locations = cluster_defects['log dist. [m]'].values * 1000
        lengths = cluster_defects['length [mm]'].values
        
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                center_distance = abs(locations[i] - locations[j])
                char_dimension = (lengths[i] + lengths[j]) / 2
                normalized_spacing = center_distance / max(char_dimension, 1.0)
                spacings.append(normalized_spacing)
        
        return np.mean(spacings) if spacings else 1.0


    def _classify_interaction_type(self, cluster_defects: pd.DataFrame) -> str:
        """Classify the type of defect interaction."""
        num_defects = len(cluster_defects)
        if num_defects == 1:
            return "isolated"
        elif num_defects == 2:
            return "pair_interaction"
        elif num_defects <= 5:
            return "small_cluster"
        else:
            return "large_cluster"
    
    
    def apply_stress_concentration_to_assessment(self, 
                                               cluster: ClusterProperties,
                                               base_erf: float) -> float:
        """
        Apply stress concentration factor to ERF calculation.
        
        Parameters:
        - cluster: ClusterProperties with stress factor
        - base_erf: Base ERF calculated without stress concentration
        
        Returns:
        - Modified ERF accounting for stress concentration
        """
        
        if not self.include_stress_concentration:
            return base_erf
        
        # Apply stress concentration to ERF
        # ERF increases with stress concentration (worse condition)
        modified_erf = base_erf * cluster.stress_concentration_factor
        
        return modified_erf
    
    def create_combined_defects_dataframe(self, 
                                        defects_df: pd.DataFrame,
                                        clusters: List[ClusterProperties]) -> pd.DataFrame:
        """
        Create a new DataFrame with combined defects for assessment.
        
        Parameters:
        - defects_df: Original defects DataFrame
        - clusters: List of cluster properties
        
        Returns:
        - DataFrame with combined defects replacing individual clustered defects
        """
        
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


# Integration functions for existing system
def enhance_existing_assessment(defects_df: pd.DataFrame,
                              joints_df: pd.DataFrame,
                              pipe_diameter_mm: float,
                              standard: str = "RSTRENG") -> Tuple[pd.DataFrame, List[ClusterProperties]]:
    """
    Enhanced assessment that can be integrated into existing corrosion assessment.
    
    Parameters:
    - defects_df: Original defects DataFrame
    - joints_df: Joints DataFrame with wall thickness
    - pipe_diameter_mm: Pipeline diameter
    - standard: Clustering standard to use
    
    Returns:
    - Tuple of (combined_defects_df, cluster_info)
    """
    
    # Create enhanced clusterer
    clusterer = EnhancedFFSClusterer(
        standard=standard,
        pipe_diameter_mm=pipe_diameter_mm,
        include_stress_concentration=True
    )
    
    # Find enhanced clusters
    clusters = clusterer.find_enhanced_clusters(defects_df, joints_df)
    
    # Create combined defects DataFrame
    combined_df = clusterer.create_combined_defects_dataframe(defects_df, clusters)
    
    return combined_df, clusters


def integrate_with_corrosion_assessment(enhanced_df: pd.DataFrame,
                                      clusters: List[ClusterProperties],
                                      method: str = "modified_b31g") -> pd.DataFrame:
    """
    Integrate stress concentration factors with existing corrosion assessment.
    
    Parameters:
    - enhanced_df: DataFrame from compute_enhanced_corrosion_metrics
    - clusters: List of cluster properties with stress factors
    - method: Assessment method ("b31g", "modified_b31g", "simplified_eff_area")
    
    Returns:
    - DataFrame with stress-corrected ERF values
    """
    
    # Create a copy to avoid modifying original
    corrected_df = enhanced_df.copy()
    
    # Add stress concentration columns
    corrected_df[f'{method}_stress_factor'] = 1.0
    corrected_df[f'{method}_erf_stress_corrected'] = corrected_df[f'{method}_erf']
    
    # Apply stress concentration to clustered defects
    for idx, row in corrected_df.iterrows():
        if row.get('is_combined', False):
            # Find corresponding cluster
            cluster_id = row.get('cluster_id', '')
            
            # Extract cluster index from cluster_id
            if cluster_id.startswith('cluster_'):
                try:
                    cluster_idx = int(cluster_id.split('_')[1])
                    if cluster_idx < len(clusters):
                        cluster = clusters[cluster_idx]
                        
                        # Apply stress concentration
                        stress_factor = cluster.stress_concentration_factor
                        original_erf = row[f'{method}_erf']
                        corrected_erf = original_erf * stress_factor
                        
                        corrected_df.loc[idx, f'{method}_stress_factor'] = stress_factor
                        corrected_df.loc[idx, f'{method}_erf_stress_corrected'] = corrected_erf
                        
                except (ValueError, IndexError):
                    continue
    
    return corrected_df


def create_clustering_comparison_report(defects_df: pd.DataFrame,
                                      joints_df: pd.DataFrame,
                                      pipe_diameter_mm: float,
                                      wall_thickness_mm: float = 10.0) -> pd.DataFrame:
    """
    Create a comparison report of clustering results across different standards.
    
    Parameters:
    - defects_df: DataFrame with defect information
    - joints_df: DataFrame with joint information
    - pipe_diameter_mm: Pipeline diameter
    - wall_thickness_mm: Representative wall thickness for comparison
    
    Returns:
    - DataFrame comparing clustering results across standards
    """
    
    standards = ["RSTRENG", "BS7910", "API579", "DNV", "CONSERVATIVE"]
    comparison_results = []
    
    for standard in standards:
        try:
            clusterer = EnhancedFFSClusterer(
                standard=standard,
                pipe_diameter_mm=pipe_diameter_mm
            )
            
            clusters = clusterer.find_enhanced_clusters(defects_df, joints_df)
            
            # Calculate summary statistics
            total_clusters = len(clusters)
            total_defects_clustered = sum(len(c.defect_indices) for c in clusters)
            avg_cluster_size = (total_defects_clustered / total_clusters) if total_clusters > 0 else 0
            max_stress_factor = max((c.stress_concentration_factor for c in clusters), default=1.0)
            
            # Get interaction criteria for comparison
            criteria = clusterer.clusterer.calculate_interaction_criteria(wall_thickness_mm)
            
            comparison_results.append({
                'Standard': standard,
                'Total Clusters': total_clusters,
                'Defects Clustered': total_defects_clustered,
                'Avg Cluster Size': avg_cluster_size,
                'Max Stress Factor': max_stress_factor,
                'Axial Distance (mm)': criteria.axial_distance_mm,
                'Circumferential Distance (mm)': criteria.circumferential_distance_mm
            })
            
        except Exception as e:
            comparison_results.append({
                'Standard': standard,
                'Error': str(e)
            })
    
    return pd.DataFrame(comparison_results)


# Example integration with session state for Streamlit
def update_session_state_with_clustering(defects_df: pd.DataFrame,
                                        joints_df: pd.DataFrame,
                                        pipe_diameter_mm: float,
                                        standard: str = "RSTRENG"):
    """
    Update Streamlit session state with enhanced clustering results.
    
    This function can be called from your corrosion assessment view.
    """
    
    import streamlit as st
    
    # Perform enhanced clustering
    combined_df, clusters = enhance_existing_assessment(
        defects_df, joints_df, pipe_diameter_mm, standard
    )
    
    # Store in session state
    st.session_state.ffs_combined_defects = combined_df
    st.session_state.ffs_clusters = clusters
    st.session_state.use_ffs_combined = True
    st.session_state.clustering_standard = standard
    
    return combined_df, clusters