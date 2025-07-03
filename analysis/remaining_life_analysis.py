import pandas as pd
import numpy as np
from typing import Dict
import streamlit as st
from utils import get_wall_thickness_for_defect, create_wall_thickness_lookup
from app.views.corrosion import calculate_b31g, calculate_modified_b31g, calculate_simplified_effective_area_method
    

def find_similar_defects(target_defect: pd.Series, historical_matches_df: pd.DataFrame, joint_tolerance = 5) -> pd.DataFrame:
    """
    Find defects similar to the target defect based on type, depth, and location.
    
    Parameters:
        - target_defect: Series containing the new defect information
        - historical_matches_df: DataFrame with defects that have growth history
        - joints_df: DataFrame with joint information for wall thickness lookup
    
    Returns:
        - DataFrame with similar defects that have growth history
    """
    
    if historical_matches_df.empty:
        return pd.DataFrame()
    
    similar_defects = historical_matches_df.copy()
    
    # Criteria 1: Defect type (High Priority)
    if 'defect_type' in target_defect and pd.notna(target_defect['defect_type']):
        similar_defects = similar_defects[similar_defects['defect_type'] == target_defect['defect_type']]

    # Criteria 2: Current depth range ±10% (High Priority)
    if 'new_depth_pct' in target_defect and pd.notna(target_defect['new_depth_pct']):
        target_depth = float(target_defect['new_depth_pct'])
        depth_tolerance = target_depth * 0.1  
        
        similar_defects = similar_defects[
            (similar_defects['new_depth_pct'] >= target_depth - depth_tolerance) &
            (similar_defects['new_depth_pct'] <= target_depth + depth_tolerance)
        ]
    
    # Criteria 3: Joint location proximity ±5 joints (Medium Priority)
    if 'joint number' in target_defect and pd.notna(target_defect['joint number']):
        target_joint = int(target_defect['joint number'])
        
        similar_defects = similar_defects[
            (similar_defects['joint number'] >= target_joint - joint_tolerance) &
            (similar_defects['joint number'] <= target_joint + joint_tolerance)
        ]
    
    return similar_defects


def estimate_growth_rate_for_new_defect(new_defect: pd.Series, historical_matches_df: pd.DataFrame, joints_df: pd.DataFrame, min_similar_defects: int = 3) -> Dict:
    """
    Estimate growth rate for a new defect based on similar historical defects.
    
    Parameters:
        - new_defect: Series containing the new defect information
        - historical_matches_df: DataFrame with defects that have growth history
        - joints_df: DataFrame with joint information
        - min_similar_defects: Minimum number of similar defects required for estimation
    
    Returns:
        - Dictionary with estimated growth rates and confidence information
    """

    # Find similar defects
    joint_tolerance = 5
    similar_defects = find_similar_defects(new_defect, historical_matches_df, joint_tolerance)

    while len(similar_defects) < min_similar_defects and joint_tolerance <= 20:
        joint_tolerance += 1
        similar_defects = find_similar_defects(new_defect, historical_matches_df, joint_tolerance)
    
    
    # Calculate statistical measures from similar defects
    length_growth_rates = similar_defects.get('length_growth_rate_mm_per_year', pd.Series()).dropna()
    width_growth_rates = similar_defects.get('width_growth_rate_mm_per_year', pd.Series()).dropna()
    depth_growth_rates = similar_defects['growth_rate_pct_per_year'].dropna()
    
    # Determine confidence level based on sample size
    if len(similar_defects) >= 10:
        confidence = 'HIGH'
    elif len(similar_defects) >= 5:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    result = {
        'estimated_depth_growth_rate_pct_per_year': depth_growth_rates.median() if not depth_growth_rates.empty else 1.0,
        'estimated_length_growth_rate_mm_per_year': length_growth_rates.median() if not length_growth_rates.empty else 3.0,
        'estimated_width_growth_rate_mm_per_year': width_growth_rates.median() if not width_growth_rates.empty else 2.0,
        'confidence_level': confidence,
        'similar_defects_count': len(similar_defects),
        'estimation_method': 'STATISTICAL_INFERENCE',
        'note': f'Based on {len(similar_defects)} similar defects with {confidence.lower()} confidence'
    }    

    return result


def calculate_remaining_life_single_defect(defect: pd.Series, growth_rate_pct_per_year: float) -> Dict:
    """
    Calculate remaining life for a single defect until it reaches 80% wall thickness.
    
    Parameters:
        - defect: Series containing defect information
        - wall_thickness_mm: Wall thickness in mm for the defect's joint
        - growth_rate_pct_per_year: Depth growth rate in % points per year
    
    Returns:
        - Dictionary with remaining life calculation results
    """
    
    try:
        current_depth_pct = float(defect.get('new_depth_pct', defect.get('depth [%]', 0)))
        critical_threshold_pct = 80.0  # B31G limit
        
        # Check if already at or above critical threshold
        if current_depth_pct >= critical_threshold_pct:
            return {
                'remaining_life_years': 0,
                'current_depth_pct': current_depth_pct,
                'critical_threshold_pct': critical_threshold_pct,
                'growth_rate_pct_per_year': growth_rate_pct_per_year,
                'status': 'CRITICAL',
                'note': 'Already at or above critical threshold'
            }
        
        # Check for zero or negative growth rate
        if growth_rate_pct_per_year <= 0:
            return {
                'remaining_life_years': float(100),
                'current_depth_pct': current_depth_pct,
                'critical_threshold_pct': critical_threshold_pct,
                'growth_rate_pct_per_year': growth_rate_pct_per_year,
                'status': 'STABLE',
                'note': 'Zero or negative growth - defect considered stable'
            }
        
        # Calculate time to reach critical threshold
        depth_difference = critical_threshold_pct - current_depth_pct
        remaining_life_years = depth_difference / growth_rate_pct_per_year

        # Determine status based on remaining life
        if remaining_life_years <= 2:
            status = 'HIGH_RISK'
        elif remaining_life_years <= 10:
            status = 'MEDIUM_RISK'
        else:
            status = 'LOW_RISK'
        
        return {
            'remaining_life_years': remaining_life_years,
            'current_depth_pct': current_depth_pct,
            'critical_threshold_pct': critical_threshold_pct,
            'growth_rate_pct_per_year': growth_rate_pct_per_year,
            'status': status,
            'note': f'Estimated to reach {critical_threshold_pct}% depth in {remaining_life_years:.1f} years'
        }
        
    except Exception as e:
        return {
            'remaining_life_years': float('nan'),
            'current_depth_pct': float('nan'),
            'critical_threshold_pct': 80.0,
            'growth_rate_pct_per_year': float('nan'),
            'status': 'ERROR',
            'note': f'Calculation error: {str(e)}'
        }

def calculate_average_growth_rates_for_similar_defects(defect: pd.Series, historical_matches_df: pd.DataFrame) -> Dict:
    """
    Calculate average growth rates for similar defects, replacing negative growth with positive averages.
    
    Parameters:
        - defect: Series containing the defect information
        - historical_matches_df: DataFrame with defects that have growth history
    
    Returns:
        - Dictionary with average growth rates for each dimension
    """
    
    # Find similar defects (reuse existing logic)
    similar_defects = find_similar_defects(defect, historical_matches_df)
    
    result = {
        'depth_growth_rate_pct_per_year': 2.0,  # Default conservative value
        'length_growth_rate_mm_per_year': 5.0,
        'width_growth_rate_mm_per_year': 3.0,
        'similar_defects_count': len(similar_defects),
        'confidence': 'LOW'
    }
    
    if len(similar_defects) < 3:
        result['note'] = f'Using defaults - only {len(similar_defects)} similar defects found'
        return result
    
    # Calculate averages for positive growth rates only
    dimensions = {
        'depth_growth_rate_pct_per_year': 'growth_rate_pct_per_year',
        'length_growth_rate_mm_per_year': 'length_growth_rate_mm_per_year', 
        'width_growth_rate_mm_per_year': 'width_growth_rate_mm_per_year'
    }
    
    for result_key, column_key in dimensions.items():
        if column_key in similar_defects.columns:
            # Filter positive growth rates
            positive_rates = similar_defects[similar_defects[column_key] > 0][column_key]
            if not positive_rates.empty:
                result[result_key] = positive_rates.mean()
        
    # Set confidence based on sample size
    if len(similar_defects) >= 10:
        result['confidence'] = 'HIGH'
    elif len(similar_defects) >= 5:
        result['confidence'] = 'MEDIUM'
    
    result['note'] = f'Based on {len(similar_defects)} similar defects'
    return result


def calculate_iterative_failure_pressure(initial_depth_pct: float, initial_length_mm: float, 
                                        initial_width_mm: float, depth_growth_rate: float,
                                        length_growth_rate: float, width_growth_rate: float,
                                        pipe_diameter_mm: float, wall_thickness_mm: float, 
                                        smys_mpa: float, safety_factor: float, 
                                        max_years: int = 100) -> Dict:
    """
    Calculate failure pressure iteratively as defect grows over time for all three methods.
    
    Parameters:
    - initial_depth_pct, initial_length_mm, initial_width_mm: Initial defect dimensions
    - depth_growth_rate, length_growth_rate, width_growth_rate: Growth rates per year
    - pipe_diameter_mm, wall_thickness_mm, smys_mpa, safety_factor: Pipe parameters
    - max_years: Maximum years to simulate
    
    Returns:
    - Dictionary with failure pressures over time for each method
    """

    years = list(range(0, max_years + 1))
    results = {
        'years': years,
        'b31g_failure_pressure': [],
        'modified_b31g_failure_pressure': [],
        'rstreng_failure_pressure': [],
        'depth_pct': [],
        'length_mm': [],
        'width_mm': []
    }
    
    for year in years:
        # Calculate current dimensions
        current_depth = initial_depth_pct + (depth_growth_rate * year)
        current_length = initial_length_mm + (length_growth_rate * year)
        current_width = initial_width_mm + (width_growth_rate * year)
        
        # Ensure dimensions don't go negative or exceed limits
        current_depth = max(0, min(current_depth, 100))  # Cap at 100%
        current_length = max(initial_length_mm * 0.1, current_length)  # Min 10% of original
        current_width = max(initial_width_mm * 0.1, current_width)
        
        results['depth_pct'].append(current_depth)
        results['length_mm'].append(current_length)
        results['width_mm'].append(current_width)
        
        # Calculate failure pressure for each method
        try:
            b31g_result = calculate_b31g(current_depth, current_length, pipe_diameter_mm, 
                                       wall_thickness_mm, smys_mpa, safety_factor)
            results['b31g_failure_pressure'].append(
                b31g_result['failure_pressure_mpa'] if b31g_result['safe'] else 0
            )
        except:
            results['b31g_failure_pressure'].append(0)
            
        try:
            mod_b31g_result = calculate_modified_b31g(current_depth, current_length, pipe_diameter_mm,
                                                    wall_thickness_mm, smys_mpa, safety_factor)
            results['modified_b31g_failure_pressure'].append(
                mod_b31g_result['failure_pressure_mpa'] if mod_b31g_result['safe'] else 0
            )
        except:
            results['modified_b31g_failure_pressure'].append(0)
            
        try:
            rstreng_result = calculate_simplified_effective_area_method(current_depth, current_length, current_width,
                                                           pipe_diameter_mm, wall_thickness_mm, smys_mpa, safety_factor)
            results['rstreng_failure_pressure'].append(
                rstreng_result['failure_pressure_mpa'] if rstreng_result['safe'] else 0
            )
        except:
            results['rstreng_failure_pressure'].append(0)
    
    return results
def calculate_pressure_based_remaining_life(defect: pd.Series, growth_rates: Dict, 
                                          pipe_diameter_mm: float, wall_thickness_mm: float,
                                          smys_mpa: float, safety_factor: float,
                                          operating_pressure_mpa: float) -> Dict:
    """
    Calculate remaining life until operating pressure exceeds failure pressure for each method.
    
    Parameters:
    - defect: Series containing defect information
    - growth_rates: Dictionary with growth rates for each dimension
    - pipe_diameter_mm, wall_thickness_mm, smys_mpa, safety_factor: Pipe parameters
    - operating_pressure_mpa: Operating pressure from user input
    
    Returns:
    - Dictionary with remaining life for each assessment method
    """
    try:
        # Get initial dimensions
        initial_depth = float(defect.get('new_depth_pct', defect.get('depth [%]', 0)))
        initial_length = float(defect.get('length [mm]', 0))
        initial_width = float(defect.get('width [mm]', 0))
        
        # Validate initial dimensions
        if initial_depth <= 0 or initial_length <= 0 or initial_width <= 0:
            return {
                'b31g_pressure_remaining_life': float('nan'),
                'modified_b31g_pressure_remaining_life': float('nan'), 
                'rstreng_pressure_remaining_life': float('nan'),
                'b31g_pressure_status': 'ERROR',
                'modified_b31g_pressure_status': 'ERROR',
                'rstreng_pressure_status': 'ERROR',
                'calculation_successful': False,
                'note': f'Invalid initial dimensions: depth={initial_depth}%, length={initial_length}mm, width={initial_width}mm'
            }
        
        # Get growth rates - ensure they're positive (replace negative with average)
        depth_rate = max(0, growth_rates['depth_growth_rate_pct_per_year'])
        length_rate = max(0, growth_rates['length_growth_rate_mm_per_year'])
        width_rate = max(0, growth_rates['width_growth_rate_mm_per_year'])
        
        # Calculate iterative failure pressures
        pressure_data = calculate_iterative_failure_pressure(
            initial_depth, initial_length, initial_width,
            depth_rate, length_rate, width_rate,
            pipe_diameter_mm, wall_thickness_mm, smys_mpa, safety_factor
        )
        
        results = {}
        methods = ['b31g', 'modified_b31g', 'rstreng']
        
        # Debug info for first defect
        debug_info = {
            'initial_depth_pct': initial_depth,
            'initial_length_mm': initial_length,
            'initial_width_mm': initial_width,
            'depth_rate': depth_rate,
            'length_rate': length_rate,
            'width_rate': width_rate,
            'operating_pressure': operating_pressure_mpa,
            'wall_thickness': wall_thickness_mm,
            'pipe_diameter': pipe_diameter_mm,
            'smys': smys_mpa
        }
        
        for method in methods:
            pressure_key = f'{method}_failure_pressure'
            failure_pressures = pressure_data[pressure_key]
            
            # Find first year where failure pressure drops below operating pressure
            remaining_life = float('inf')  # Default to infinite if never fails
            initial_failure_pressure = failure_pressures[0] if failure_pressures else 0
            
            # Debug: Store initial failure pressure
            debug_info[f'{method}_initial_failure_pressure'] = initial_failure_pressure
            
            for year, failure_pressure in enumerate(failure_pressures):
                # Failure occurs when failure pressure drops to or below operating pressure
                if failure_pressure > 0 and failure_pressure <= operating_pressure_mpa:
                    remaining_life = year
                    debug_info[f'{method}_failure_year'] = year
                    debug_info[f'{method}_failure_pressure_at_failure'] = failure_pressure
                    break
                elif failure_pressure <= 0:
                    # Method not applicable (defect too severe)
                    remaining_life = year
                    debug_info[f'{method}_failure_year'] = year
                    debug_info[f'{method}_failure_pressure_at_failure'] = 0
                    debug_info[f'{method}_failure_reason'] = 'Method limit exceeded'
                    break
            
            # Determine status
            if remaining_life == float('inf'):
                status = 'SAFE'
            elif remaining_life == 0:
                status = 'CRITICAL'  # Already failed or at limit
            elif remaining_life <= 2:
                status = 'CRITICAL'
            elif remaining_life <= 10:
                status = 'HIGH_RISK'
            else:
                status = 'LOW_RISK'
            
            results[f'{method}_pressure_remaining_life'] = remaining_life
            results[f'{method}_pressure_status'] = status
        
        results['calculation_successful'] = True
        results['debug_info'] = debug_info
        results['note'] = 'Pressure-based analysis completed'
        
        return results
        
    except Exception as e:
        return {
            'b31g_pressure_remaining_life': float('nan'),
            'modified_b31g_pressure_remaining_life': float('nan'), 
            'rstreng_pressure_remaining_life': float('nan'),
            'b31g_pressure_status': 'ERROR',
            'modified_b31g_pressure_status': 'ERROR',
            'rstreng_pressure_status': 'ERROR',
            'calculation_successful': False,
            'note': f'Calculation error: {str(e)}'
        }


def enhanced_calculate_remaining_life_analysis(comparison_results: Dict, joints_df: pd.DataFrame, 
                                             operating_pressure_mpa: float, pipe_diameter_mm: float,
                                             smys_mpa: float, safety_factor: float = 1.39) -> Dict:
    """
    Enhanced remaining life analysis including both depth-based and pressure-based failure criteria.
    
    Parameters:
    - comparison_results: Results dictionary from compare_defects function
    - joints_df: DataFrame with joint information including wall thickness
    - operating_pressure_mpa: Operating pressure from user input
    - pipe_diameter_mm: Pipe diameter in mm
    - smys_mpa: SMYS value in MPa
    - safety_factor: Safety factor for calculations
    
    Returns:
    - Dictionary with enhanced remaining life analysis results
    """

    # Create wall thickness lookup with validation
    try:
        wt_lookup = create_wall_thickness_lookup(joints_df, validate=True)
    except ValueError as e:
        return {
            'error': str(e),
            'analysis_possible': False
        }
    
    # Get matched and new defects
    matched_defects = comparison_results['matches_df'].copy()
    new_defects = comparison_results['new_defects'].copy()
    
    results = {
        'matched_defects_analysis': [],
        'new_defects_analysis': [],
        'summary_statistics': {},
        'analysis_possible': True,
        'operating_pressure_mpa': operating_pressure_mpa
    }
    
    # Analyze matched defects (have measured growth rates)
    for idx, defect in matched_defects.iterrows():
        try: 
            wall_thickness = get_wall_thickness_for_defect(defect, wt_lookup)
            
            # Original depth-based analysis
            depth_growth_rate = defect.get('growth_rate_pct_per_year', 0)
            depth_remaining_life = calculate_remaining_life_single_defect(defect, depth_growth_rate)
            
            # Enhanced pressure-based analysis
            measured_growth_rates = {
                'depth_growth_rate_pct_per_year': depth_growth_rate,
                'length_growth_rate_mm_per_year': defect.get('length_growth_rate_mm_per_year', 0),
                'width_growth_rate_mm_per_year': defect.get('width_growth_rate_mm_per_year', 0)
            }
            
            pressure_analysis = calculate_pressure_based_remaining_life(
                defect, measured_growth_rates, pipe_diameter_mm, wall_thickness,
                smys_mpa, safety_factor, operating_pressure_mpa
            )
            
            # Combine results
            combined_result = {
                'defect_id': defect.get('new_defect_id', idx),
                'log_dist': defect.get('log_dist', 0),
                'defect_type': defect.get('defect_type', 'Unknown'),
                'joint_number': defect.get('joint number', 0),
                'wall_thickness_mm': wall_thickness,
                'growth_rate_source': 'MEASURED',
                
                # Depth-based results
                'depth_based_remaining_life': depth_remaining_life['remaining_life_years'],
                'depth_based_status': depth_remaining_life['status'],
                
                # Pressure-based results for each method
                **pressure_analysis
            }
            results['matched_defects_analysis'].append(combined_result)
        
        except ValueError as e:
            # Log the error but continue processing other defects
            st.warning(f"Skipping defect at {defect.get('log_dist', 'unknown')}m: {e}")
            continue
    

    # Analyze new defects (estimate growth rates)
    for idx, defect in new_defects.iterrows():
        try:
            wall_thickness = get_wall_thickness_for_defect(defect, wt_lookup)
        
            # Estimate growth rates based on similar defects
            estimated_growth_rates = calculate_average_growth_rates_for_similar_defects(defect, matched_defects)
            
            # Original depth-based analysis
            estimated_depth_rate = estimated_growth_rates['depth_growth_rate_pct_per_year']
            depth_remaining_life = calculate_remaining_life_single_defect(defect, estimated_depth_rate)
            
            # Enhanced pressure-based analysis
            pressure_analysis = calculate_pressure_based_remaining_life(
                defect, estimated_growth_rates, pipe_diameter_mm, wall_thickness,
                smys_mpa, safety_factor, operating_pressure_mpa
            )
            
            # Combine results
            combined_result = {
                'defect_id': defect.get('defect_id', idx),
                'log_dist': defect.get('log dist. [m]', 0),
                'defect_type': defect.get('component / anomaly identification', 'Unknown'),
                'joint_number': defect.get('joint number', 0),
                'wall_thickness_mm': wall_thickness,
                'growth_rate_source': 'ESTIMATED',
                'estimation_confidence': estimated_growth_rates['confidence'],
                'similar_defects_count': estimated_growth_rates['similar_defects_count'],
                
                # Depth-based results
                'depth_based_remaining_life': depth_remaining_life['remaining_life_years'],
                'depth_based_status': depth_remaining_life['status'],
                
                # Pressure-based results for each method
                **pressure_analysis
            }
            
            results['new_defects_analysis'].append(combined_result)

        except ValueError as e:
            # Log the error but continue processing other defects
            st.warning(f"Skipping defect at {defect.get('log_dist', 'unknown')}m: {e}")
            continue

    
    # Calculate summary statistics
    all_analyses = results['matched_defects_analysis'] + results['new_defects_analysis']
    
    if all_analyses:
        summary_stats = {
            'total_defects_analyzed': len(all_analyses),
            'defects_with_measured_growth': len(results['matched_defects_analysis']),
            'defects_with_estimated_growth': len(results['new_defects_analysis']),
        }
        
        # Calculate statistics for each failure criterion
        methods = ['depth_based', 'b31g_pressure', 'modified_b31g_pressure', 'rstreng_pressure']
        
        for method in methods:
            if method == 'depth_based':
                lives = [a['depth_based_remaining_life'] for a in all_analyses 
                        if np.isfinite(a['depth_based_remaining_life'])]
                statuses = [a['depth_based_status'] for a in all_analyses]
            else:
                life_key = f'{method}_remaining_life'
                status_key = f'{method}_status'
                lives = [a[life_key] for a in all_analyses 
                        if np.isfinite(a[life_key])]
                statuses = [a[status_key] for a in all_analyses]
            
            if lives:
                summary_stats[f'{method}_avg_remaining_life'] = np.mean(lives) # type: ignore
                summary_stats[f'{method}_min_remaining_life'] = np.min(lives)
            
            # Count status distribution
            status_counts = {}
            for status in statuses:
                status_counts[status] = status_counts.get(status, 0) + 1
            summary_stats[f'{method}_status_distribution'] = status_counts # type: ignore
        
        results['summary_statistics'] = summary_stats
    
    return results