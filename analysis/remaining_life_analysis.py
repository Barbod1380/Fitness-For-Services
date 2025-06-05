"""
Functions for calculating remaining life of pipeline defects based on growth rates.
"""

import pandas as pd
import numpy as np
from typing import Dict
import streamlit as st

def find_similar_defects(target_defect: pd.Series, historical_matches_df: pd.DataFrame) -> pd.DataFrame:
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
        similar_defects = similar_defects[
            similar_defects['defect_type'] == target_defect['defect_type']
        ]
    
    # Criteria 2: Current depth range ±10% (High Priority)
    if 'new_depth_pct' in target_defect and pd.notna(target_defect['new_depth_pct']):
        target_depth = float(target_defect['new_depth_pct'])
        depth_tolerance = target_depth * 0.1  # ±10%
        
        similar_defects = similar_defects[
            (similar_defects['new_depth_pct'] >= target_depth - depth_tolerance) &
            (similar_defects['new_depth_pct'] <= target_depth + depth_tolerance)
        ]
    
    # Criteria 3: Joint location proximity ±5 joints (Medium Priority)
    if 'joint number' in target_defect and pd.notna(target_defect['joint number']):
        target_joint = int(target_defect['joint number'])
        joint_tolerance = 5
        
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
    similar_defects = find_similar_defects(new_defect, historical_matches_df)
    
    if len(similar_defects) < min_similar_defects:
        # Not enough similar defects - use conservative industry defaults
        return {
            'estimated_depth_growth_rate_pct_per_year': 2.0,
            'estimated_length_growth_rate_mm_per_year': 5.0,
            'estimated_width_growth_rate_mm_per_year': 3.0,
            'confidence_level': 'LOW',
            'similar_defects_count': len(similar_defects),
            'estimation_method': 'INDUSTRY_DEFAULT',
            'note': f'Used conservative defaults - only {len(similar_defects)} similar defects found'
        }
    
    # Calculate statistical measures from similar defects
    depth_growth_rates = similar_defects['growth_rate_pct_per_year'].dropna()
    length_growth_rates = similar_defects.get('length_growth_rate_mm_per_year', pd.Series()).dropna()
    width_growth_rates = similar_defects.get('width_growth_rate_mm_per_year', pd.Series()).dropna()
    
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


def calculate_remaining_life_analysis(comparison_results: Dict, joints_df: pd.DataFrame) -> Dict:
    """
    Perform remaining life analysis for all defects in the comparison results.
    
    Parameters:
    - comparison_results: Results dictionary from compare_defects function
    - joints_df: DataFrame with joint information including wall thickness
    
    Returns:
    - Dictionary with remaining life analysis results
    """
    if not comparison_results.get('has_depth_data', False):
        return {
            'error': 'No depth data available for remaining life analysis',
            'analysis_possible': False
        }
    
    # Get matched defects (have growth history)
    matched_defects = comparison_results['matches_df'].copy()
    
    # Get new defects (no growth history)
    new_defects = comparison_results['new_defects'].copy()
    
    # Create wall thickness lookup
    wt_lookup = {}
    if 'wt nom [mm]' in joints_df.columns:
        wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    results = {
        'matched_defects_analysis': [],
        'new_defects_analysis': [],
        'summary_statistics': {},
        'analysis_possible': True
    }
    
    # Analyze matched defects (have measured growth rates)
    for idx, defect in matched_defects.iterrows():
        wall_thickness = wt_lookup.get(defect.get('joint number')) 
        if wall_thickness is None:
            wall_thickness = 10.0  # Default value
            st.warning(f"Wall thickness for joint {defect.get('joint number')} not found. Using default value of {wall_thickness} mm.", width = "stretch")
        
        # Use measured growth rate
        growth_rate = defect.get('growth_rate_pct_per_year', 0)
        
        remaining_life = calculate_remaining_life_single_defect(defect, growth_rate)
        remaining_life.update({
            'defect_id': defect.get('new_defect_id', idx),
            'log_dist': defect.get('log_dist', 0),
            'defect_type': defect.get('defect_type', 'Unknown'),
            'joint_number': defect.get('joint number', 0),
            'growth_rate_source': 'MEASURED',
            'wall_thickness_mm': wall_thickness
        })
        
        results['matched_defects_analysis'].append(remaining_life)
    
    # Analyze new defects (estimate growth rates)
    for idx, defect in new_defects.iterrows():
        wall_thickness = wt_lookup.get(defect.get('joint number'), 10.0)
        
        # Estimate growth rate based on similar defects
        growth_estimation = estimate_growth_rate_for_new_defect(defect, matched_defects, joints_df)
        estimated_growth_rate = growth_estimation['estimated_depth_growth_rate_pct_per_year']
        
        remaining_life = calculate_remaining_life_single_defect(defect, estimated_growth_rate)
        remaining_life.update({
            'defect_id': defect.get('defect_id', idx),
            'log_dist': defect.get('log dist. [m]', 0),
            'defect_type': defect.get('component / anomaly identification', 'Unknown'),
            'joint_number': defect.get('joint number', 0),
            'growth_rate_source': 'ESTIMATED',
            'wall_thickness_mm': wall_thickness,
            'estimation_confidence': growth_estimation['confidence_level'],
            'similar_defects_count': growth_estimation['similar_defects_count'],
            'estimation_method': growth_estimation['estimation_method']
        })
        
        results['new_defects_analysis'].append(remaining_life)
    
    # Calculate summary statistics
    all_analyses = results['matched_defects_analysis'] + results['new_defects_analysis']
    
    if all_analyses:
        # Filter out infinite and NaN values for statistics
        finite_lives = [a['remaining_life_years'] for a in all_analyses 
                       if np.isfinite(a['remaining_life_years'])]
        
        # Count by risk status
        status_counts = {}
        for analysis in all_analyses:
            status = analysis['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        results['summary_statistics'] = {
            'total_defects_analyzed': len(all_analyses),
            'defects_with_measured_growth': len(results['matched_defects_analysis']),
            'defects_with_estimated_growth': len(results['new_defects_analysis']),
            'average_remaining_life_years': np.mean(finite_lives) if finite_lives else float('nan'),
            'median_remaining_life_years': np.median(finite_lives) if finite_lives else float('nan'),
            'min_remaining_life_years': np.min(finite_lives) if finite_lives else float('nan'),
            'status_distribution': status_counts
        }
    
    return results