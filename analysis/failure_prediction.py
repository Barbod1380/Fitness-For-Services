# analysis/failure_prediction.py - Enhanced version with joint failure details

"""
Enhanced failure prediction analysis for pipeline joints over time.
Now includes detailed joint failure information for visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

def predict_joint_failures_over_time(
    defects_df: pd.DataFrame,
    joints_df: pd.DataFrame,
    pipe_diameter_mm: float,
    smys_mpa: float,
    operating_pressure_mpa: float,
    assessment_method: str = 'b31g',
    window_years: int = 15,
    safety_factor: float = 1.39,
    growth_rates_dict: Dict = None,  # type: ignore
    pipe_creation_year: int = None,  # type: ignore
    current_year: int = None  # type: ignore
) -> Dict:
    """
    Enhanced failure prediction with detailed joint failure information for visualization.
    """
    
    # Validate inputs
    if operating_pressure_mpa <= 0:
        raise ValueError("Operating pressure must be positive")
    
    if assessment_method not in ['b31g', 'modified_b31g', 'simplified_eff_area']:
        raise ValueError(f"Unknown assessment method: {assessment_method}")
    
    # Determine if we're in single-file or multi-year mode
    if growth_rates_dict is None:
        if pipe_creation_year is None or current_year is None:
            raise ValueError("For single file analysis, both pipe_creation_year and current_year are required")
        
        growth_rates_dict = estimate_single_file_growth_rates(
            defects_df, joints_df, pipe_creation_year, current_year
        )
    
    # Create wall thickness lookup
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    # Initialize results structure with enhanced failure details
    results = {
        'years': list(range(1, window_years + 1)),
        'erf_failures_by_year': [],
        'depth_failures_by_year': [],
        'total_joints': len(joints_df),
        'joints_with_defects': len(defects_df['joint number'].unique()),
        'assessment_method': assessment_method,
        'operating_pressure_mpa': operating_pressure_mpa,
        'failure_details': [],
        # NEW: Enhanced failure information for visualization
        'joint_failure_timeline': {},  # {joint_num: {year: failure_info}}
        'failing_joints_summary': [],  # List of joints that fail with summary info
        'current_defects_df': defects_df.copy(),  # Store current state for visualization
        'joints_df': joints_df.copy(),  # Store joint info
        'growth_rates_dict': growth_rates_dict,  # Store growth rates
        'pipe_diameter_mm': pipe_diameter_mm
    }
    
    # Track joints that have already failed
    failed_joints = set()
    
    # For each year in the prediction window
    for year in results['years']:
        # Project defects to this future year
        projected_defects = project_defects_to_year(defects_df, growth_rates_dict, year)
        
        # Calculate failures for this year
        erf_failed_joints, depth_failed_joints, year_details = calculate_joint_failures_for_year_enhanced(
            projected_defects, joints_df, wt_lookup, pipe_diameter_mm, smys_mpa,
            operating_pressure_mpa, assessment_method, safety_factor, year
        )
        
        # Only count new failures (joints that haven't failed before)
        new_erf_failures = [j for j in erf_failed_joints if j not in failed_joints]
        new_depth_failures = [j for j in depth_failed_joints if j not in failed_joints]
        
        results['erf_failures_by_year'].append(len(new_erf_failures))
        results['depth_failures_by_year'].append(len(new_depth_failures))
        results['failure_details'].append(year_details)
        
        # Process new failures for joint timeline
        all_new_failures = set(new_erf_failures + new_depth_failures)
        for joint_num in all_new_failures:
            if joint_num not in results['joint_failure_timeline']:
                results['joint_failure_timeline'][joint_num] = {}
            
            # Store detailed failure information for this joint
            joint_failure_info = extract_joint_failure_details(
                joint_num, projected_defects, year_details, 
                defects_df, growth_rates_dict, year
            )
            
            results['joint_failure_timeline'][joint_num][year] = joint_failure_info
            
            # Add to failing joints summary (first time this joint fails)
            if joint_num not in failed_joints:
                failure_mode = 'ERF' if joint_num in new_erf_failures else 'Depth'
                if joint_num in new_erf_failures and joint_num in new_depth_failures:
                    failure_mode = 'Both'
                
                joint_info = joints_df[joints_df['joint number'] == joint_num].iloc[0]
                results['failing_joints_summary'].append({
                    'joint_number': joint_num,
                    'failure_year': year,
                    'failure_mode': failure_mode,
                    'location_m': joint_info['log dist. [m]'],
                    'joint_length_m': joint_info['joint length [m]'],
                    'defect_count': len(projected_defects[projected_defects['joint number'] == joint_num])
                })
        
        # Update failed joints set
        failed_joints.update(all_new_failures)
    
    # Sort failing joints by failure year
    results['failing_joints_summary'].sort(key=lambda x: x['failure_year'])
    
    # Calculate cumulative failures
    results['cumulative_erf_failures'] = np.cumsum(results['erf_failures_by_year']).tolist()
    results['cumulative_depth_failures'] = np.cumsum(results['depth_failures_by_year']).tolist()
    
    # Add summary statistics
    results['summary'] = generate_failure_summary(results)
    
    return results


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
        # Get wall thickness for depth calculation
        joint_num = defect['joint number']
        
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


def project_defects_to_year(
    defects_df: pd.DataFrame,
    growth_rates_dict: Dict,
    target_year: int
) -> pd.DataFrame:
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


def generate_failure_summary(results: Dict) -> Dict:
    """Generate summary statistics for the failure prediction."""
    
    total_joints = results['total_joints']
    max_erf_failures = max(results['cumulative_erf_failures']) if results['cumulative_erf_failures'] else 0
    max_depth_failures = max(results['cumulative_depth_failures']) if results['cumulative_depth_failures'] else 0
    
    # Find first year with failures
    first_erf_failure_year = None
    first_depth_failure_year = None
    
    for i, (erf_count, depth_count) in enumerate(zip(results['erf_failures_by_year'], results['depth_failures_by_year'])):
        if erf_count > 0 and first_erf_failure_year is None:
            first_erf_failure_year = results['years'][i]
        if depth_count > 0 and first_depth_failure_year is None:
            first_depth_failure_year = results['years'][i]
    
    summary = {
        'total_joints_analyzed': total_joints,
        'joints_with_defects': results['joints_with_defects'],
        'max_erf_failures': max_erf_failures,
        'max_depth_failures': max_depth_failures,
        'pct_erf_failures': (max_erf_failures / total_joints * 100) if total_joints > 0 else 0,
        'pct_depth_failures': (max_depth_failures / total_joints * 100) if total_joints > 0 else 0,
        'first_erf_failure_year': first_erf_failure_year,
        'first_depth_failure_year': first_depth_failure_year,
        'prediction_window_years': results['years'][-1] if results['years'] else 0,
        'total_failing_joints': len(results['failing_joints_summary'])
    }
    
    return summary