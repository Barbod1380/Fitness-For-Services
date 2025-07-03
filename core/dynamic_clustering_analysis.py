"""
Dynamic clustering simulation for pipeline defects.
Simulates defect growth over time and detects when clustering will occur,
then calculates actual remaining life considering future clustering events.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import streamlit as st
from dataclasses import dataclass
import warnings
from .ffs_defect_interaction import FFSDefectInteraction

@dataclass
class ClusteringEvent:
    """Represents a clustering event that occurs at a specific time."""
    year: float
    defect_indices: List[int]
    cluster_type: str  # 'new_cluster', 'cluster_growth', 'merge_clusters'
    combined_defect_props: Dict
    failure_time: float
    original_failure_times: List[float]

@dataclass
class FailureEvent:
    """Represents a failure event (individual or clustering-induced)."""
    year: float
    failure_type: str  # 'individual', 'clustering'
    defect_indices: List[int]
    failure_mode: str  # 'depth_80pct', 'pressure_failure'
    severity: str  # 'CRITICAL', 'HIGH_RISK', etc.
    details: Dict

class DynamicClusteringAnalyzer:
    """
    Simulates defect growth over time and predicts clustering-induced failures.
    """
    
    def __init__(self, 
                 ffs_rules: FFSDefectInteraction,
                 max_simulation_years: int = 50,
                 time_step_years: float = 0.5,
                 depth_failure_threshold: float = 80.0):
        """
        Initialize the dynamic clustering analyzer.
        
        Parameters:
        - ffs_rules: FFS interaction rules for clustering detection
        - max_simulation_years: Maximum years to simulate forward
        - time_step_years: Time resolution for simulation (0.5 = 6 months)
        - depth_failure_threshold: Depth percentage for failure (default 80%)
        """
        self.ffs_rules = ffs_rules
        self.max_years = max_simulation_years
        self.time_step = time_step_years
        self.depth_threshold = depth_failure_threshold
        
    def simulate_dynamic_clustering_failure(self,
                                          defects_df: pd.DataFrame,
                                          joints_df: pd.DataFrame,
                                          growth_rates_dict: Dict,
                                          pipe_diameter_mm: float) -> Dict:
        """
        Main simulation function that predicts clustering-induced failures.
        
        Parameters:
        - defects_df: Current defects with positions and sizes
        - joints_df: Joint information for wall thickness
        - growth_rates_dict: Growth rates for each defect (depth, length, width per year)
        - pipe_diameter_mm: Pipe diameter for clustering calculations
        
        Returns:
        - Dictionary with simulation results including earliest failures
        """
        
        # Step 1: Initialize simulation
        simulation_results = {
            'earliest_failure_time': float('inf'),
            'earliest_failure_mode': 'none',
            'clustering_events': [],
            'failure_events': [],
            'individual_failure_times': {},
            'simulation_timeline': []
        }
        
        # Step 2: Calculate individual failure times (baseline)
        individual_failures = self._calculate_individual_failure_times(
            defects_df, joints_df, growth_rates_dict
        )
        simulation_results['individual_failure_times'] = individual_failures
        
        # Step 3: Time-forward simulation
        current_defects = defects_df.copy()
        current_defects['defect_id'] = range(len(current_defects))
        active_clusters = {}  # Track existing clusters
        
        # Simulation loop
        time_points = np.arange(self.time_step, self.max_years + self.time_step, self.time_step)
        
        for current_time in time_points:
            # Project defects to current time
            projected_defects = self._project_defects_to_time(
                defects_df, growth_rates_dict, current_time
            )
            
            # Check for new clustering events
            clustering_events = self._detect_clustering_events(
                projected_defects, joints_df, pipe_diameter_mm, 
                current_time, active_clusters, growth_rates_dict
            )
            
            # Process clustering events
            for event in clustering_events:
                simulation_results['clustering_events'].append(event)
                
                # Calculate failure time for clustered defect
                cluster_failure_time = current_time + event.failure_time
                
                # Check if this creates an earlier failure
                if cluster_failure_time < simulation_results['earliest_failure_time']:
                    simulation_results['earliest_failure_time'] = cluster_failure_time
                    simulation_results['earliest_failure_mode'] = 'clustering'
                    simulation_results['earliest_failure_details'] = {
                        'clustering_time': current_time,
                        'defects_involved': event.defect_indices,
                        'cluster_type': event.cluster_type,
                        'combined_properties': event.combined_defect_props
                    }
                
                # Update active clusters
                cluster_id = f"cluster_{current_time}_{len(active_clusters)}"
                active_clusters[cluster_id] = {
                    'defect_indices': event.defect_indices,
                    'formation_time': current_time,
                    'properties': event.combined_defect_props
                }
            
            # Record timeline snapshot
            simulation_results['simulation_timeline'].append({
                'time': current_time,
                'active_clusters': len(active_clusters),
                'new_clusters': len(clustering_events),
                'total_defects': len(projected_defects)
            })
            
            # Early termination if we found a very early failure

        # Step 4: Compare with individual failures
        earliest_individual = min(individual_failures.values()) if individual_failures else float('inf')
        
        if earliest_individual < simulation_results['earliest_failure_time']:
            simulation_results['earliest_failure_time'] = earliest_individual
            simulation_results['earliest_failure_mode'] = 'individual'
            # Find which defect fails first
            for defect_id, failure_time in individual_failures.items():
                if failure_time == earliest_individual:
                    simulation_results['earliest_failure_details'] = {
                        'defect_id': defect_id,
                        'failure_type': 'individual_depth_threshold'
                    }
                    break
        
        # Step 5: Generate summary
        simulation_results['analysis_summary'] = self._generate_analysis_summary(simulation_results)
        
        return simulation_results
    
    def _calculate_individual_failure_times(self,
                                           defects_df: pd.DataFrame,
                                           joints_df: pd.DataFrame,
                                           growth_rates_dict: Dict) -> Dict:
        """Calculate when each defect would fail individually (baseline)."""
        
        individual_failures = {}
        wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
        
        for idx, defect in defects_df.iterrows():
            defect_id = defect.get('defect_id', idx)
            
            # Get current depth and growth rate
            current_depth_pct = defect['depth [%]']
            
            # Get growth rate for this defect
            if defect_id in growth_rates_dict:
                depth_growth_rate = growth_rates_dict[defect_id].get('depth_growth_pct_per_year', 0)
            else:
                # Use average growth rate as fallback
                depth_growth_rate = 2.0  # Conservative default
            
            # Calculate failure time
            if depth_growth_rate > 0:
                remaining_depth = self.depth_threshold - current_depth_pct
                failure_time = remaining_depth / depth_growth_rate
                individual_failures[defect_id] = max(0, failure_time)
            else:
                individual_failures[defect_id] = float('inf')  # No growth = no failure
        
        return individual_failures
    
    def _project_defects_to_time(self, original_defects: pd.DataFrame, growth_rates_dict: Dict, target_time: float) -> pd.DataFrame:
        """Project all defects to a future time based on their growth rates."""
        
        projected = original_defects.copy()
        
        for idx, defect in projected.iterrows():
            defect_id = defect.get('defect_id', idx)
            
            if defect_id in growth_rates_dict:
                growth_rates = growth_rates_dict[defect_id]
                
                # Project depth
                if 'depth_growth_pct_per_year' in growth_rates:
                    new_depth = defect['depth [%]'] + (growth_rates['depth_growth_pct_per_year'] * target_time)
                    projected.loc[idx, 'depth [%]'] = min(100, max(0, new_depth)) # type: ignore
                
                # Project length
                if 'length_growth_mm_per_year' in growth_rates:
                    new_length = defect['length [mm]'] + (growth_rates['length_growth_mm_per_year'] * target_time)
                    projected.loc[idx, 'length [mm]'] = max(defect['length [mm]'], new_length) # type: ignore
                
                # Project width
                if 'width_growth_mm_per_year' in growth_rates:
                    new_width = defect['width [mm]'] + (growth_rates['width_growth_mm_per_year'] * target_time)
                    projected.loc[idx, 'width [mm]'] = max(defect['width [mm]'], new_width) # type: ignore
        
        return projected
    
    def _detect_clustering_events(self,
                                projected_defects: pd.DataFrame,
                                joints_df: pd.DataFrame,
                                pipe_diameter_mm: float,
                                current_time: float,
                                existing_clusters: Dict, 
                                growth_rates_dict: Dict) -> List[ClusteringEvent]:
        """Detect new clustering events at the current time."""
        
        clustering_events = []
        
        try:
            # Find current clusters using FFS rules
            current_clusters = self.ffs_rules.find_interacting_defects(
                projected_defects, joints_df, pipe_diameter_mm
            )

            existing_cluster_sets = {
                cluster_id: set(cluster_info['defect_indices']) 
                for cluster_id, cluster_info in existing_clusters.items()
            }
            
            # Check for new clusters (groups with >1 defect that weren't clustered before)
            for cluster in current_clusters:
                if len(cluster) > 1:  # Only interested in actual clusters
                    cluster_set = set(cluster)
                    
                    # Check if this exact cluster already exists
                    is_existing = any(
                        cluster_set == existing_set 
                        for existing_set in existing_cluster_sets.values()
                    )
                    
                    # Also check if it's a subset or superset of existing clusters
                    is_evolution = False
                    for existing_set in existing_cluster_sets.values():
                        if cluster_set.issuperset(existing_set) or existing_set.issuperset(cluster_set):
                            is_evolution = True
                            break
                    
                    if not is_existing and (not is_evolution or current_time > 0):
                        # This is a genuinely new cluster
                        # Calculate combined defect properties
                        combined_props = self._calculate_combined_defect_properties(
                            projected_defects.iloc[cluster], joints_df
                        )
                        
                        # Calculate failure time for combined defect
                        failure_time = self._calculate_cluster_failure_time(combined_props, growth_rates_dict)
                        
                        # Create clustering event
                        event = ClusteringEvent(
                            year=current_time,
                            defect_indices=cluster,
                            cluster_type='new_cluster' if not is_evolution else 'cluster_evolution',
                            combined_defect_props=combined_props,
                            failure_time=failure_time,
                            original_failure_times=[]  # We'll calculate this properly
                        )
                        
                        clustering_events.append(event)
        
        except Exception as e:
            warnings.warn(f"Error detecting clustering at time {current_time}: {str(e)}")
        
        return clustering_events
    
    def _calculate_combined_defect_properties(self,
                                            cluster_defects: pd.DataFrame,
                                            joints_df: pd.DataFrame) -> Dict:
        """Calculate properties of a combined clustered defect."""
        
        # Use maximum depth (most conservative)
        max_depth = cluster_defects['depth [%]'].max()
        
        # Calculate total axial extent
        min_start = (cluster_defects['log dist. [m]'] * 1000 - cluster_defects['length [mm]'] / 2).min()
        max_end = (cluster_defects['log dist. [m]'] * 1000 + cluster_defects['length [mm]'] / 2).max()
        total_length = max_end - min_start
        
        # Calculate total circumferential extent (simplified)
        total_width = cluster_defects['width [mm]'].sum()  # Simplified - could be more sophisticated
        
        # Calculate center position
        center_location = cluster_defects['log dist. [m]'].mean()
        
        return {
            'combined_depth_pct': max_depth,
            'combined_length_mm': total_length,
            'combined_width_mm': total_width,
            'center_location_m': center_location,
            'num_original_defects': len(cluster_defects),
            'original_defect_ids': cluster_defects.get('defect_id', cluster_defects.index).tolist()
        }
    
    def _calculate_cluster_failure_time(self, combined_props: Dict, growth_rates_dict: Dict) -> float:
        """Calculate how long until the combined defect fails."""
        
        current_depth = combined_props['combined_depth_pct']
        
        if current_depth >= self.depth_threshold:
            return 0.0  # Already failed
        
        # Estimate growth rate for combined defect (conservative approach)
        # This could be more sophisticated based on the original defects' growth rates
        constituent_growth_rates = [growth_rates_dict.get(idx, {}).get('depth_growth_pct_per_year', 2.0)  for idx in combined_props['original_defect_ids']]
        estimated_growth_rate = np.mean(constituent_growth_rates) * 1.2  # 20% acceleration factor
        
        remaining_depth = self.depth_threshold - current_depth
        failure_time = remaining_depth / estimated_growth_rate + 0.00001
        
        return max(0, failure_time)
    
    def _generate_analysis_summary(self, simulation_results: Dict) -> Dict:
        """Generate a summary of the simulation results."""
        
        summary = {
            'total_simulation_years': self.max_years,
            'earliest_failure_time': simulation_results['earliest_failure_time'],
            'failure_mode': simulation_results['earliest_failure_mode'],
            'clustering_events_count': len(simulation_results['clustering_events']),
            'individual_vs_clustering_benefit': 0.0
        }
        
        # Calculate benefit of clustering analysis
        if simulation_results['individual_failure_times']:
            earliest_individual = min(simulation_results['individual_failure_times'].values())
            clustering_failure_time = simulation_results['earliest_failure_time']
            
            if simulation_results['earliest_failure_mode'] == 'clustering':
                # Clustering causes earlier failure
                summary['individual_vs_clustering_benefit'] = earliest_individual - clustering_failure_time
                summary['risk_insight'] = f"Clustering causes failure {summary['individual_vs_clustering_benefit']:.1f} years earlier than individual analysis predicted"
            else:
                summary['risk_insight'] = "Individual defect analysis is sufficient for this case"
        
        return summary


# Integration function to add to comparison.py
def perform_dynamic_clustering_analysis(comparison_results, earlier_year, later_year, 
                                      pipe_diameter_mm, joints_df):
    """
    Perform dynamic clustering analysis and integrate with existing comparison view.
    """
    
    # Extract growth rates from comparison results
    growth_rates_dict = {}
    if 'matches_df' in comparison_results and not comparison_results['matches_df'].empty:
        matches_df = comparison_results['matches_df']
        
        for idx, match in matches_df.iterrows():
            defect_id = match.get('new_defect_id', idx)
            growth_rates_dict[defect_id] = {
                'depth_growth_pct_per_year': match.get('growth_rate_pct_per_year', 2.0),
                'length_growth_mm_per_year': match.get('length_growth_rate_mm_per_year', 0.0),
                'width_growth_mm_per_year': match.get('width_growth_rate_mm_per_year', 0.0)
            }
    
    # Get current defects (later year)
    current_defects = st.session_state.datasets[later_year]['defects_df'].copy()
    current_defects['defect_id'] = range(len(current_defects))
    
    # Initialize dynamic analyzer
    ffs_rules = FFSDefectInteraction(
        axial_interaction_distance_mm=25.4,
        circumferential_interaction_method='sqrt_dt'
    )
    
    analyzer = DynamicClusteringAnalyzer(
        ffs_rules=ffs_rules,
        max_simulation_years=30,
        time_step_years=0.5
    )
    
    # Run simulation
    simulation_results = analyzer.simulate_dynamic_clustering_failure(
        current_defects, joints_df, growth_rates_dict, pipe_diameter_mm
    )
    
    return simulation_results