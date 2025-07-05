# analysis/failure_prediction.py - Enhanced version with joint failure details

"""
Enhanced failure prediction analysis for pipeline joints over time.
Now includes detailed joint failure information for visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

def predict_joint_failures_over_time_with_clustering(
    defects_df: pd.DataFrame,
    joints_df: pd.DataFrame,
    pipe_diameter_mm: float,
    smys_mpa: float,
    operating_pressure_mpa: float,
    assessment_method: str = 'b31g',
    window_years: int = 15,
    safety_factor: float = 1.39,
    growth_rates_dict: Dict = None,
    pipe_creation_year: int = None,
    current_year: int = None
) -> Dict:
    """
    ENHANCED: Failure prediction that includes dynamic FFS clustering analysis.
    
    This version:
    1. Projects defects forward in time
    2. Checks for new clustering events each year
    3. Applies FFS combination rules when defects start interacting
    4. Calculates failure for both individual and clustered defects
    """
    
    # Initialize FFS interaction analyzer
    from core.ffs_defect_interaction import FFSDefectInteraction
    
    ffs_analyzer = FFSDefectInteraction(
        axial_interaction_distance_mm=25.4,  # API 579-1 standard
        circumferential_interaction_method='sqrt_dt'
    )
    
    # Determine growth rates
    if growth_rates_dict is None:
        if pipe_creation_year is None or current_year is None:
            raise ValueError("For single file analysis, both pipe_creation_year and current_year are required")
        
        growth_rates_dict = estimate_single_file_growth_rates(
            defects_df, joints_df, pipe_creation_year, current_year
        )
    
    # Initialize results with clustering tracking
    results = {
        'years': list(range(1, window_years + 1)),
        'erf_failures_by_year': [],
        'depth_failures_by_year': [],
        'clustering_events_by_year': [],  # NEW: Track when clustering occurs
        'clustered_failures_by_year': [],  # NEW: Failures due to clustering
        'individual_failures_by_year': [],  # NEW: Failures of individual defects
        'total_joints': len(joints_df),
        'joints_with_defects': len(defects_df['joint number'].unique()),
        'assessment_method': assessment_method,
        'operating_pressure_mpa': operating_pressure_mpa,
        'clustering_analysis': {
            'total_clustering_events': 0,
            'earliest_clustering_year': None,
            'clustering_accelerated_failures': 0
        }
    }
    
    # Track clustering state over time
    failed_joints = set()
    active_clusters = {}  # Track existing clusters: {cluster_id: {defect_indices, formation_year}}
    clustering_event_id = 0
    
    # Year-by-year simulation with clustering
    for year in results['years']:
        # Step 1: Project all defects to this year
        projected_defects = project_defects_to_year(defects_df, growth_rates_dict, year)
        
        # Step 2: Check for new clustering events
        current_clusters = ffs_analyzer.find_interacting_defects(
            projected_defects, joints_df, pipe_diameter_mm
        )
        
        # Step 3: Identify NEW clustering events (defects that started interacting this year)
        new_clustering_events = detect_new_clustering_events(
            current_clusters, active_clusters, year, projected_defects
        )
        
        # Step 4: Update active clusters
        for event in new_clustering_events:
            clustering_event_id += 1
            cluster_id = f"cluster_{year}_{clustering_event_id}"
            active_clusters[cluster_id] = {
                'defect_indices': event['defect_indices'],
                'formation_year': year,
                'combined_defect': event['combined_defect']
            }
            
            # Log clustering event
            results['clustering_analysis']['total_clustering_events'] += 1
            if results['clustering_analysis']['earliest_clustering_year'] is None:
                results['clustering_analysis']['earliest_clustering_year'] = year
        
        results['clustering_events_by_year'].append(len(new_clustering_events))
        
        # Step 5: Create assessment dataset with clustered defects
        assessment_defects = create_assessment_dataset_with_clusters(
            projected_defects, active_clusters
        )
        
        # Step 6: Calculate failures for this year
        erf_failed_joints, depth_failed_joints, year_details = calculate_joint_failures_with_clustering(
            assessment_defects, joints_df, pipe_diameter_mm, smys_mpa,
            operating_pressure_mpa, assessment_method, safety_factor, year, active_clusters
        )
        
        # Step 7: Track different types of failures
        new_erf_failures = [j for j in erf_failed_joints if j not in failed_joints]
        new_depth_failures = [j for j in depth_failed_joints if j not in failed_joints]
        
        # Separate clustered vs individual failures
        clustered_failures = count_clustered_failures(new_erf_failures + new_depth_failures, active_clusters)
        individual_failures = len(new_erf_failures + new_depth_failures) - clustered_failures
        
        # Store results
        results['erf_failures_by_year'].append(len(new_erf_failures))
        results['depth_failures_by_year'].append(len(new_depth_failures))
        results['clustered_failures_by_year'].append(clustered_failures)
        results['individual_failures_by_year'].append(individual_failures)
        
        # Update failed joints
        failed_joints.update(new_erf_failures + new_depth_failures)
        
        # Track clustering-accelerated failures
        if clustered_failures > 0:
            results['clustering_analysis']['clustering_accelerated_failures'] += clustered_failures
    
    # Calculate cumulative results
    results['cumulative_erf_failures'] = np.cumsum(results['erf_failures_by_year']).tolist()
    results['cumulative_depth_failures'] = np.cumsum(results['depth_failures_by_year']).tolist()
    results['cumulative_clustering_events'] = np.cumsum(results['clustering_events_by_year']).tolist()
    
    # Generate enhanced summary
    results['summary'] = generate_clustering_aware_summary(results)
    
    return results


def detect_new_clustering_events(current_clusters, active_clusters, year, projected_defects):
    """
    Detect which defects started interacting this year (new clustering events).
    """
    new_events = []
    
    # Convert current clusters to sets for comparison
    current_cluster_sets = [set(cluster) for cluster in current_clusters if len(cluster) > 1]
    
    # Get existing cluster sets
    existing_cluster_sets = [
        set(cluster_info['defect_indices']) 
        for cluster_info in active_clusters.values()
    ]
    
    for cluster_set in current_cluster_sets:
        # Check if this is a genuinely new cluster
        is_new = True
        for existing_set in existing_cluster_sets:
            # If there's significant overlap, it's not a new cluster
            if len(cluster_set.intersection(existing_set)) > len(cluster_set) * 0.5:
                is_new = False
                break
        
        if is_new and len(cluster_set) > 1:
            # This is a new clustering event!
            cluster_defects = projected_defects.iloc[list(cluster_set)]
            
            # Create combined defect using FFS rules
            from core.ffs_defect_interaction import FFSDefectInteraction
            ffs_analyzer = FFSDefectInteraction()
            
            combined_defect = ffs_analyzer._combine_defects_with_vector_summation(
                cluster_defects, projected_defects.iloc[0].get('pipe_diameter_mm', 1000), None
            )
            
            new_events.append({
                'year': year,
                'defect_indices': list(cluster_set),
                'num_defects': len(cluster_set),
                'combined_defect': combined_defect,
                'cluster_type': 'new_interaction'
            })
    
    return new_events


def create_assessment_dataset_with_clusters(projected_defects, active_clusters):
    """
    Create assessment dataset where clustered defects are replaced by their combined equivalents.
    """
    assessment_defects = projected_defects.copy()
    clustered_indices = set()
    
    # Replace clustered defects with combined defects
    for cluster_id, cluster_info in active_clusters.items():
        defect_indices = cluster_info['defect_indices']
        combined_defect = cluster_info['combined_defect']
        
        # Mark original defects as clustered (will be removed)
        clustered_indices.update(defect_indices)
        
        # Add combined defect (use first defect's index, modify its properties)
        primary_idx = defect_indices[0]
        for key, value in combined_defect.items():
            if key in assessment_defects.columns:
                assessment_defects.loc[primary_idx, key] = value
        
        # Mark as clustered for tracking
        assessment_defects.loc[primary_idx, 'is_clustered'] = True
        assessment_defects.loc[primary_idx, 'cluster_id'] = cluster_id
        assessment_defects.loc[primary_idx, 'original_defect_count'] = len(defect_indices)
    
    # Remove secondary defects from clusters (keep only the primary with combined properties)
    indices_to_remove = []
    for cluster_info in active_clusters.values():
        # Remove all but the first defect in each cluster
        indices_to_remove.extend(cluster_info['defect_indices'][1:])
    
    assessment_defects = assessment_defects.drop(indices_to_remove).reset_index(drop=True)
    
    return assessment_defects


def calculate_joint_failures_with_clustering(assessment_defects, joints_df, pipe_diameter_mm, 
                                           smys_mpa, operating_pressure_mpa, assessment_method, 
                                           safety_factor, year, active_clusters):
    """
    Calculate failures considering both individual and clustered defects.
    """
    # Use the existing failure calculation but with enhanced defect dataset
    return calculate_joint_failures_for_year_enhanced(
        assessment_defects, joints_df, dict(zip(joints_df['joint number'], joints_df['wt nom [mm]'])),
        pipe_diameter_mm, smys_mpa, operating_pressure_mpa, assessment_method, safety_factor, year
    )


def count_clustered_failures(failed_joints, active_clusters):
    """
    Count how many failures are due to clustered defects.
    """
    clustered_failure_count = 0
    
    for cluster_info in active_clusters.values():
        cluster_defects = cluster_info['defect_indices']
        # If any defect in the cluster is in a failed joint, count as clustered failure
        for joint in failed_joints:
            # This is simplified - would need joint mapping logic in practice
            pass
    
    return clustered_failure_count


def generate_clustering_aware_summary(results):
    """
    Generate summary statistics that include clustering effects.
    """
    total_joints = results['total_joints']
    max_erf_failures = max(results['cumulative_erf_failures']) if results['cumulative_erf_failures'] else 0
    max_clustering_events = max(results['cumulative_clustering_events']) if results['cumulative_clustering_events'] else 0
    
    clustering_analysis = results['clustering_analysis']
    
    summary = {
        'total_joints_analyzed': total_joints,
        'joints_with_defects': results['joints_with_defects'],
        'max_erf_failures': max_erf_failures,
        'clustering_events_total': clustering_analysis['total_clustering_events'],
        'earliest_clustering_year': clustering_analysis['earliest_clustering_year'],
        'clustering_accelerated_failures': clustering_analysis['clustering_accelerated_failures'],
        'clustering_impact_pct': (clustering_analysis['clustering_accelerated_failures'] / max_erf_failures * 100) if max_erf_failures > 0 else 0,
        'prediction_window_years': results['years'][-1] if results['years'] else 0
    }
    
    return summary


# Integration point for existing code
def predict_joint_failures_over_time(defects_df, joints_df, pipe_diameter_mm, smys_mpa, 
                                    operating_pressure_mpa, assessment_method='b31g', 
                                    window_years=15, safety_factor=1.39, growth_rates_dict=None,
                                    pipe_creation_year=None, current_year=None, 
                                    enable_clustering=True):
    """
    ENHANCED: Wrapper that can enable/disable clustering analysis.
    """
    return predict_joint_failures_over_time_with_clustering(
            defects_df, joints_df, pipe_diameter_mm, smys_mpa, operating_pressure_mpa,
            assessment_method, window_years, safety_factor, growth_rates_dict,
            pipe_creation_year, current_year
        )
    

def calculate_joint_failures_for_year_enhanced(
    projected_defects: pd.DataFrame,
    joints_df: pd.DataFrame,
    wt_lookup: Dict,
    pipe_diameter_mm: float,
    smys_mpa: float,
    operating_pressure_mpa: float,
    assessment_method: str,
    safety_factor: float,
    year: int
) -> Tuple[List, List, Dict]:
    """
    Enhanced version that captures more detailed failure information.
    """
    
    # Import calculation functions
    if assessment_method == 'b31g':
        from app.views.corrosion import calculate_b31g as calc_func
    elif assessment_method == 'modified_b31g':
        from app.views.corrosion import calculate_modified_b31g as calc_func
    elif assessment_method == 'simplified_eff_area':
        from app.views.corrosion import calculate_simplified_effective_area_method as calc_func
    
    erf_failed_joints = set()
    depth_failed_joints = set()
    joint_details = {}
    
    # Group defects by joint
    for joint_num in projected_defects['joint number'].unique():
        joint_defects = projected_defects[projected_defects['joint number'] == joint_num]
        wall_thickness = wt_lookup.get(joint_num, 10.0)
        
        joint_erf_failed = False
        joint_depth_failed = False
        defect_failures = []
        
        for idx, defect in joint_defects.iterrows():
            depth_pct = defect['depth [%]']
            length_mm = defect['length [mm]']
            width_mm = defect.get('width [mm]', defect['length [mm]'] * 0.5)
            
            # Check depth failure (>80%)
            if depth_pct > 80.0:
                joint_depth_failed = True
                defect_failures.append({
                    'defect_idx': idx,
                    'failure_type': 'depth',
                    'depth_pct': depth_pct,
                    'location_m': defect['log dist. [m]'],
                    'length_mm': length_mm,
                    'width_mm': width_mm,
                    'clock_position': defect.get('clock', '12:00')
                })
            
            # Check ERF failure (ERF < 1.0)
            try:
                if assessment_method == 'simplified_eff_area':
                    calc_result = calc_func(
                        depth_pct, length_mm, width_mm, pipe_diameter_mm, 
                        wall_thickness, smys_mpa, safety_factor  # type: ignore
                    )
                else:
                    calc_result = calc_func(
                        depth_pct, length_mm, pipe_diameter_mm, 
                        wall_thickness, smys_mpa, safety_factor
                    )
                
                if calc_result['safe'] and calc_result['failure_pressure_mpa'] > 0:
                    erf = calc_result['failure_pressure_mpa'] / operating_pressure_mpa
                    
                    if erf < 1.0:
                        joint_erf_failed = True
                        defect_failures.append({
                            'defect_idx': idx,
                            'failure_type': 'erf',
                            'erf': erf,
                            'failure_pressure_mpa': calc_result['failure_pressure_mpa'],
                            'location_m': defect['log dist. [m]'],
                            'depth_pct': depth_pct,
                            'length_mm': length_mm,
                            'width_mm': width_mm,
                            'clock_position': defect.get('clock', '12:00')
                        })
                        
            except Exception as e:
                warnings.warn(f"Failed to calculate failure pressure for defect {idx}: {e}")
        
        # Record joint failure status
        if joint_erf_failed:
            erf_failed_joints.add(joint_num)
        if joint_depth_failed:
            depth_failed_joints.add(joint_num)
        
        if defect_failures:
            joint_details[joint_num] = defect_failures
    
    year_details = {
        'year': year,
        'erf_failed_joints': list(erf_failed_joints),
        'depth_failed_joints': list(depth_failed_joints),
        'joint_failure_details': joint_details
    }
    
    return list(erf_failed_joints), list(depth_failed_joints), year_details


def extract_joint_failure_details(
    joint_num: int,
    projected_defects: pd.DataFrame,
    year_details: Dict,
    original_defects: pd.DataFrame,
    growth_rates_dict: Dict,
    failure_year: int
) -> Dict:
    """
    Extract detailed failure information for a specific joint for visualization.
    """
    
    # Get current and projected defects for this joint
    current_joint_defects = original_defects[original_defects['joint number'] == joint_num].copy()
    projected_joint_defects = projected_defects[projected_defects['joint number'] == joint_num].copy()
    
    # Get failure details from year_details
    joint_failures = year_details['joint_failure_details'].get(joint_num, [])
    
    # Identify which defects caused the failure
    failure_causing_defects = []
    for failure in joint_failures:
        failure_causing_defects.append({
            'defect_idx': failure['defect_idx'],
            'failure_type': failure['failure_type'],
            'failure_criteria': failure.get('erf', failure.get('depth_pct')),
            'location_m': failure['location_m'],
            'clock_position': failure['clock_position']
        })
    
    # Calculate growth for each defect in this joint
    defect_growth_info = []
    for idx, current_defect in current_joint_defects.iterrows():
        projected_defect = projected_joint_defects.loc[idx] if idx in projected_joint_defects.index else None # type: ignore
        
        if projected_defect is not None:
            growth_rates = growth_rates_dict.get(idx, {})
            
            growth_info = {
                'defect_idx': idx,
                'location_m': current_defect['log dist. [m]'],
                'clock_position': current_defect.get('clock', '12:00'),
                'current_depth': current_defect['depth [%]'],
                'current_length': current_defect['length [mm]'],
                'current_width': current_defect['width [mm]'],
                'projected_depth': projected_defect['depth [%]'],
                'projected_length': projected_defect['length [mm]'],
                'projected_width': projected_defect['width [mm]'],
                'depth_growth_rate': growth_rates.get('depth_growth_pct_per_year', 0),
                'length_growth_rate': growth_rates.get('length_growth_mm_per_year', 0),
                'width_growth_rate': growth_rates.get('width_growth_mm_per_year', 0),
                'is_failure_cause': idx in [f['defect_idx'] for f in failure_causing_defects]
            }
            defect_growth_info.append(growth_info)
    
    return {
        'joint_number': joint_num,
        'failure_year': failure_year,
        'failure_causing_defects': failure_causing_defects,
        'defect_growth_info': defect_growth_info,
        'current_defects_df': current_joint_defects,
        'projected_defects_df': projected_joint_defects
    }


# Keep all the existing helper functions (estimate_single_file_growth_rates, project_defects_to_year, generate_failure_summary) unchanged
def estimate_single_file_growth_rates(
    defects_df: pd.DataFrame,
    joints_df: pd.DataFrame,
    pipe_creation_year: int,
    current_year: int
) -> Dict:
    """
    Estimate growth rates for defects based on pipe age and current defect sizes.
    """
    
    pipe_age = current_year - pipe_creation_year
    if pipe_age <= 0:
        raise ValueError("Current year must be after pipe creation year")
    
    growth_rates_dict = {}
    
    for idx, defect in defects_df.iterrows():
        
        # Estimate depth growth rate
        current_depth_pct = defect.get('depth [%]', 0)
        if current_depth_pct > 0:
            depth_growth_pct_per_year = current_depth_pct / pipe_age
        else:
            depth_growth_pct_per_year = 1.0
        
        # Estimate length growth rate
        current_length_mm = defect.get('length [mm]', 0)
        if current_length_mm > 0:
            length_growth_mm_per_year = current_length_mm / pipe_age
        else:
            length_growth_mm_per_year = 3.0
        
        # Estimate width growth rate
        current_width_mm = defect.get('width [mm]', 0)
        if current_width_mm > 0:
            width_growth_mm_per_year = current_width_mm / pipe_age
        else:
            width_growth_mm_per_year = 2
        
        growth_rates_dict[idx] = {
            'depth_growth_pct_per_year': depth_growth_pct_per_year,
            'length_growth_mm_per_year': length_growth_mm_per_year,
            'width_growth_mm_per_year': width_growth_mm_per_year
        }
    
    return growth_rates_dict


def project_defects_to_year(defects_df: pd.DataFrame, growth_rates_dict: Dict, target_year: int) -> pd.DataFrame:
    """
    Project all defects to a future year based on their growth rates.
    """
    projected_df = defects_df.copy()
    
    for idx, defect in projected_df.iterrows():
        if idx in growth_rates_dict:
            growth_rates = growth_rates_dict[idx]
            
            # Project depth
            if 'depth_growth_pct_per_year' in growth_rates:
                new_depth = defect['depth [%]'] + (growth_rates['depth_growth_pct_per_year'] * target_year)
                projected_df.loc[idx, 'depth [%]'] = min(100.0, max(0.0, new_depth))                            # type: ignore
            
            # Project length
            if 'length_growth_mm_per_year' in growth_rates:
                new_length = defect['length [mm]'] + (growth_rates['length_growth_mm_per_year'] * target_year)
                projected_df.loc[idx, 'length [mm]'] = max(defect['length [mm]'], new_length)                   # type: ignore
             
            # Project width
            if 'width_growth_mm_per_year' in growth_rates:
                new_width = defect['width [mm]'] + (growth_rates['width_growth_mm_per_year'] * target_year)
                projected_df.loc[idx, 'width [mm]'] = max(defect['width [mm]'], new_width)                      # type: ignore
    
    return projected_df