"""
Corrosion assessment view for the Pipeline Analysis application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import base64
from app.ui_components import create_metrics_row
from app.services.state_manager import get_state
from core.ffs_defect_interaction import *


def compute_enhanced_corrosion_metrics(defects_df, joints_df, pipe_diameter_mm, smys_mpa, 
                                     safety_factor, analysis_pressure_mpa, min_allowable_pressure_mpa):
    """
    Enhanced version of compute_corrosion_metrics_for_dataframe with pressure-based assessment.
    """
    
    # First run the standard corrosion assessment
    enhanced_df = compute_corrosion_metrics_for_dataframe(
        defects_df, joints_df, pipe_diameter_mm, smys_mpa, safety_factor
    )
    
    # Add pressure analysis parameters
    enhanced_df['analysis_pressure_mpa'] = analysis_pressure_mpa
    enhanced_df['min_allowable_pressure_mpa'] = min_allowable_pressure_mpa
    
    # Initialize pressure-based assessment columns
    methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
    
    for method in methods:
        enhanced_df[f'{method}_pressure_status'] = 'UNKNOWN'
        enhanced_df[f'{method}_operational_status'] = 'UNKNOWN'
        enhanced_df[f'{method}_can_operate_analysis'] = False
        enhanced_df[f'{method}_can_operate_minimum'] = False
        enhanced_df[f'{method}_recommended_action'] = ''
        enhanced_df[f'{method}_max_safe_pressure_mpa'] = 0.0
        enhanced_df[f'{method}_pressure_margin_pct'] = 0.0
    
    # Perform pressure-based assessment for each defect and method
    for idx, row in enhanced_df.iterrows():
        for method in methods:
            failure_pressure = row[f'{method}_failure_pressure_mpa']
            safe_pressure = row[f'{method}_safe_pressure_mpa']
            is_safe = row[f'{method}_safe']
            
            if is_safe and failure_pressure > 0 and safe_pressure > 0:
                pressure_assessment = classify_defect_pressure_status(
                    failure_pressure, safe_pressure, 
                    analysis_pressure_mpa, min_allowable_pressure_mpa
                )
                
                # Update dataframe with assessment results
                for key, value in pressure_assessment.items():
                    column_name = f'{method}_{key}'
                    if column_name in enhanced_df.columns:
                        enhanced_df.loc[idx, column_name] = value
            else:
                # Handle failed calculations
                enhanced_df.loc[idx, f'{method}_pressure_status'] = 'CALCULATION_ERROR'
                enhanced_df.loc[idx, f'{method}_operational_status'] = 'UNKNOWN'
                enhanced_df.loc[idx, f'{method}_recommended_action'] = 'Review defect parameters'
    
    return enhanced_df


def classify_defect_pressure_status(failure_pressure_mpa, safe_pressure_mpa, 
                                  analysis_pressure_mpa, min_allowable_pressure_mpa):
    """
    Classify defect based on pressure-based criteria.
    
    Parameters:
    - failure_pressure_mpa: Calculated failure pressure
    - safe_pressure_mpa: Safe operating pressure (with safety factor)
    - analysis_pressure_mpa: Current/proposed operating pressure
    - min_allowable_pressure_mpa: Minimum pressure needed for operations
    
    Returns:
    - Dict with classification results
    """
    
    # Handle invalid calculations
    if failure_pressure_mpa <= 0 or safe_pressure_mpa <= 0:
        return {
            'pressure_status': 'CALCULATION_ERROR',
            'operational_status': 'UNKNOWN',
            'can_operate_at_analysis_pressure': False,
            'can_operate_at_min_pressure': False,
            'recommended_action': 'Review calculation inputs'
        }
    
    # Determine if defect is safe at analysis pressure
    safe_at_analysis = safe_pressure_mpa >= analysis_pressure_mpa
    safe_at_minimum = safe_pressure_mpa >= min_allowable_pressure_mpa
    
    # Classify pressure status
    if safe_at_analysis:
        if safe_pressure_mpa >= analysis_pressure_mpa * 1.2:  # 20% margin
            pressure_status = 'SAFE_WITH_MARGIN'
        else:
            pressure_status = 'SAFE_MINIMAL_MARGIN'
    else:
        pressure_status = 'UNSAFE'
    
    # Determine operational status
    if safe_at_analysis:
        operational_status = 'ACCEPTABLE'
        recommended_action = 'Normal operations'
    elif safe_at_minimum:
        operational_status = 'PRESSURE_DERATION_REQUIRED'
        max_safe_pressure = safe_pressure_mpa * 0.95  # 5% additional margin
        recommended_action = f'Reduce pressure to ≤{max_safe_pressure:.1f} MPa'
    else:
        operational_status = 'IMMEDIATE_REPAIR_REQUIRED'
        recommended_action = 'Repair defect before operations'
    
    # Calculate maximum safe operating pressure
    max_safe_operating_pressure = safe_pressure_mpa * 0.95  # 5% margin
    
    return {
        'pressure_status': pressure_status,
        'operational_status': operational_status,
        'can_operate_at_analysis_pressure': safe_at_analysis,
        'can_operate_at_min_pressure': safe_at_minimum,
        'recommended_action': recommended_action,
        'max_safe_operating_pressure_mpa': max_safe_operating_pressure,
        'pressure_margin_pct': ((safe_pressure_mpa - analysis_pressure_mpa) / analysis_pressure_mpa * 100) if analysis_pressure_mpa > 0 else 0
    }

def create_pressure_based_summary_stats(enhanced_df, analysis_pressure_mpa, min_allowable_pressure_mpa):
    """
    Create summary statistics for pressure-based assessment.
    """
    methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
    summary = {
        'analysis_pressure_mpa': analysis_pressure_mpa,
        'min_allowable_pressure_mpa': min_allowable_pressure_mpa,
        'total_defects': len(enhanced_df)
    }
    
    for method in methods:
        # Count defects by operational status
        operational_counts = enhanced_df[f'{method}_operational_status'].value_counts().to_dict()
        pressure_status_counts = enhanced_df[f'{method}_pressure_status'].value_counts().to_dict()
        
        # Calculate key metrics
        can_operate_analysis = enhanced_df[f'{method}_can_operate_analysis'].sum()
        can_operate_minimum = enhanced_df[f'{method}_can_operate_minimum'].sum()
        
        # Calculate percentages
        total_valid = enhanced_df[enhanced_df[f'{method}_safe'] == True].shape[0]
        pct_safe_analysis = (can_operate_analysis / total_valid * 100) if total_valid > 0 else 0
        pct_safe_minimum = (can_operate_minimum / total_valid * 100) if total_valid > 0 else 0
        
        # Find maximum safe operating pressure for the pipeline
        valid_pressures = enhanced_df[enhanced_df[f'{method}_safe'] == True][f'{method}_max_safe_pressure_mpa']
        max_pipeline_pressure = valid_pressures.min() if not valid_pressures.empty else 0
        
        summary[method] = {
            'operational_status_counts': operational_counts,
            'pressure_status_counts': pressure_status_counts,
            'defects_safe_at_analysis_pressure': can_operate_analysis,
            'defects_safe_at_minimum_pressure': can_operate_minimum,
            'pct_safe_at_analysis_pressure': pct_safe_analysis,
            'pct_safe_at_minimum_pressure': pct_safe_minimum,
            'max_pipeline_operating_pressure_mpa': max_pipeline_pressure,
            'total_valid_defects': total_valid
        }
    
    return summary


def create_pressure_assessment_visualization(enhanced_df, method='b31g'):
    """
    Create visualization showing pressure-based assessment results.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Filter valid data
    valid_data = enhanced_df[enhanced_df[f'{method}_safe'] == True].copy()
    
    if valid_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data for pressure assessment visualization",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Color mapping for operational status
    color_map = {
        'ACCEPTABLE': 'green',
        'PRESSURE_DERATION_REQUIRED': 'orange', 
        'IMMEDIATE_REPAIR_REQUIRED': 'red',
        'UNKNOWN': 'gray'
    }
    
    # Create scatter plot
    fig = go.Figure()
    
    for status, color in color_map.items():
        status_data = valid_data[valid_data[f'{method}_operational_status'] == status]
        if status_data.empty:
            continue
            
        hover_text = []
        for _, row in status_data.iterrows():
            hover_text.append(
                f"<b>Location:</b> {row['log dist. [m]']:.2f}m<br>"
                f"<b>Safe Pressure:</b> {row[f'{method}_safe_pressure_mpa']:.1f} MPa<br>"
                f"<b>Max Safe Operating:</b> {row[f'{method}_max_safe_pressure_mpa']:.1f} MPa<br>"
                f"<b>Pressure Margin:</b> {row[f'{method}_pressure_margin_pct']:.1f}%<br>"
                f"<b>Status:</b> {status.replace('_', ' ').title()}<br>"
                f"<b>Action:</b> {row[f'{method}_recommended_action']}"
            )
        
        fig.add_trace(go.Scatter(
            x=status_data['log dist. [m]'],
            y=status_data[f'{method}_safe_pressure_mpa'],
            mode='markers',
            marker=dict(size=10, color=color, opacity=0.7),
            name=f"{status.replace('_', ' ').title()} ({len(status_data)})",
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add horizontal lines for pressure references
    analysis_pressure = enhanced_df['analysis_pressure_mpa'].iloc[0]
    min_allowable = enhanced_df['min_allowable_pressure_mpa'].iloc[0]
    
    fig.add_hline(
        y=analysis_pressure, 
        line_dash="dash", 
        line_color="blue",
        annotation_text=f"Analysis Pressure: {analysis_pressure:.1f} MPa"
    )
    
    fig.add_hline(
        y=min_allowable,
        line_dash="dot", 
        line_color="purple",
        annotation_text=f"Min Allowable: {min_allowable:.1f} MPa"
    )
    
    fig.update_layout(
        title=f"Pressure-Based Assessment Results - {method.replace('_', ' ').title()}",
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title="Safe Operating Pressure (MPa)",
        height=500,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig



def calculate_b31g(defect_depth_pct, defect_length_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa, safety_factor=1.39):
    """
    Calculate remaining strength and failure pressure using the
    Original ASME B31G Level-1 method (2012 edition), with defect depth given as a percentage.
    This version distinguishes between short and long flaws.

    Parameters:
    - defect_depth_pct: Corrosion depth, as a percentage of wall thickness (e.g., 40 for 40 %).
    - defect_length_mm: Longitudinal extent of corrosion (L) in millimeters.
    - pipe_diameter_mm: Outside diameter of pipe (D) in millimeters.
    - wall_thickness_mm: Nominal wall thickness (t) in millimeters.
    - smys_mpa: Specified Minimum Yield Strength (SMYS) in MPa.
    - safety_factor: Safety factor to apply (default: 1.39 for gas pipelines)

    Returns:
    - dict containing:
        • method: "B31G Original Level-1"
        • safe (bool): False if invalid, True otherwise
        • failure_pressure_mpa: Predicted burst pressure in MPa (0 if invalid)
        • safe_pressure_mpa: failure_pressure_mpa / safety_factor (0 if invalid)
        • remaining_strength_pct: (1 - d/t)×100 (0 if invalid)
        • folias_factor_M: Folias factor M (None if not applicable or invalid)
        • flaw_type_applied: "Short Flaw" or "Long Flaw" or "N/A"
        • safety_factor_used: The safety factor that was applied
        • note: Explanation if invalid or which formulation was used
    """

    # 1. Convert defect depth percentage → absolute depth (mm)
    defect_depth_mm = (defect_depth_pct / 100.0) * wall_thickness_mm

    # 2. Compute depth-to-thickness ratio d/t
    d_t = defect_depth_mm / wall_thickness_mm

    # 3. If d/t > 0.8, Original B31G Level-1 is not valid
    if d_t > 0.8:
        return {
            "method": "B31G Original Level-1",
            "safe": False,
            "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0,
            "remaining_strength_pct": 0.0,
            "folias_factor_M": None,
            "flaw_type_applied": "N/A",
            "note": "Defect depth exceeds 80% of wall thickness (d/t > 0.8): Level-1 B31G not applicable."
        }

    # 4. Compute the dimensionless length parameter z = L² / (D × t) Ensure pipe_diameter_mm is not zero
    z = (defect_length_mm ** 2) / (pipe_diameter_mm * wall_thickness_mm)

    # 5. Overall applicability limit for z in Level 1 (Original B31G tables often go up to L/sqrt(Dt) ~ 7.07, so z ~ 50)
    if z > 50:
        return {
            "method": "B31G Original Level-1",
            "safe": False,
            "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0,
            "remaining_strength_pct": 0.0,
            "folias_factor_M": None,
            "flaw_type_applied": "N/A",
            "note": f"Defect parameter z = L²/(D·t) = {z:.2f} exceeds 50: Level-1 B31G not applicable for this simplified model."
        }

    # 7. Compute flow stress S_flow = 1.1 × SMYS (per B31G Level-1)
    S_flow_mpa = 1.1 * smys_mpa

    # 8. Compute the "intact pipe" hoop‐burst term (Barlow's formula at flow stress): P_o = 2·S_flow·t / D
    P_o_mpa = (2.0 * S_flow_mpa * wall_thickness_mm) / pipe_diameter_mm

    failure_pressure_mpa = 0.0
    M_calculated = None # Folias factor
    flaw_type_note = ""

    # Distinguish between Short Flaws (z <= 20) and Long Flaws (20 < z <= 50)
    if z <= 20:  # Short Flaw
        flaw_type_note = "Short Flaw (z <= 20)"
        # 6a. For short flaws, A/A_0 is approximated as (2/3)*(d/t) for parabolic flaw shape
        A_A0_ratio = (2.0 / 3.0) * d_t

        # 6b. Compute Folias factor M_p for short flaws
        # M_p = sqrt(1 + 0.8 * z)
        M_calculated = math.sqrt(1.0 + 0.8 * z)

        # 10a. Failure pressure for short flaws
        # Pf = P_o * [ (1 - A/A_0) / (1 - (A/A_0) / M_p) ]
        # Ensure denominator is not zero or negative (can happen if M_p is too small relative to A/A_0)
        denominator_short_flaw = 1.0 - (A_A0_ratio / M_calculated if M_calculated != 0 else float('inf'))
        if denominator_short_flaw <= 0:
             return {
                "method": "B31G Original Level-1",
                "safe": False,
                "failure_pressure_mpa": 0.0,
                "safe_pressure_mpa": 0.0,
                "remaining_strength_pct": (1.0 - d_t) * 100.0,
                "folias_factor_M": M_calculated,
                "flaw_type_applied": flaw_type_note,
                "note": f"Calculation error for short flaw: denominator (1 - (A/A0)/M) is not positive. A/A0={A_A0_ratio:.3f}, M={M_calculated:.3f}"
            }
        failure_pressure_mpa = P_o_mpa * ( (1.0 - A_A0_ratio) / denominator_short_flaw )

    else:  # Long Flaw (20 < z <= 50)
        flaw_type_note = "Long Flaw (20 < z <= 50)"
        # 10b. Failure pressure for long flaws
        # Pf = P_o * (1 - d/t)
        # Folias factor M is not directly used in this specific Pf equation for long flaws in original B31G.
        # However, we can still report an M value if needed for context, using the simpler form.
        M_calculated = math.sqrt(1.0 + 0.8 * z) # Calculated for reporting, not used in this Pf formula.
        failure_pressure_mpa = P_o_mpa * (1.0 - d_t)

    # 11. Apply the user-specified or default safety factor
    safe_pressure_mpa = failure_pressure_mpa / safety_factor

    # 12. Compute remaining strength as a percentage of original (based on remaining ligament)
    remaining_strength_pct = (1.0 - d_t) * 100.0

    # 13. Return the results with safety factor included
    return {
        "method": "B31G Original Level-1",
        "safe": True,
        "failure_pressure_mpa": failure_pressure_mpa,
        "safe_pressure_mpa": safe_pressure_mpa,
        "remaining_strength_pct": remaining_strength_pct,
        "folias_factor_M": M_calculated,
        "flaw_type_applied": flaw_type_note,
        "safety_factor_used": safety_factor,  # Add this
        "note": f"Calculation based on {flaw_type_note} criteria. z={z:.2f}, d/t={d_t:.3f}. Safety Factor: {safety_factor}"
    }


def calculate_modified_b31g(defect_depth_pct, defect_length_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa, safety_factor=1.39):
    """
    Calculate remaining strength and failure pressure using the Modified B31G
    Level-1 method (often referred to as the 0.85dL method).

    Parameters:
    - defect_depth_pct: Defect depth (%). Must be an absolute value.
    - defect_length_mm: Longitudinal extent of corrosion (L) in millimeters.
    - pipe_diameter_mm: Outside diameter of pipe (D) in millimeters.
    - wall_thickness_mm: Nominal wall thickness (t) in millimeters.
    - smys_mpa: Specified Minimum Yield Strength (SMYS) in MPa.
    - safety_factor: Safety factor to apply (default: 1.39 for gas pipelines)

    Returns:
    - dict containing:
        • method (str): "Modified B31G (0.85dL)"
        • safe (bool): False if invalid input or outside applicability, True otherwise.
        • failure_pressure_mpa (float): Predicted burst pressure in MPa (0 if invalid).
        • safe_pressure_mpa (float): failure_pressure_mpa / safety_factor (0 if invalid).
        • remaining_strength_factor_pct (float): RSF = (Sf / S_flow) * 100 (0 if invalid).
        • folias_factor_M (float): Folias factor M (None if invalid).
        • z_parameter (float): Dimensionless length parameter z (None if invalid).
        • d_over_t_ratio (float): d/t ratio (None if invalid).
        • note (str): Explanation if invalid or other relevant info.
    """

    # Depth to thickness ratio (d/t)
    defect_depth_mm = (defect_depth_pct / 100.0) * wall_thickness_mm
    d_t = defect_depth_mm / wall_thickness_mm

    # Applicability Limit: If depth is greater than 80% of wall thickness
    if d_t > 0.8:
        return {
            "method": "Modified B31G (0.85dL)", "safe": False, "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0, "remaining_strength_factor_pct": 0.0,
            "folias_factor_M": None, "z_parameter": None, "d_over_t_ratio": d_t,
            "note": f"Defect depth exceeds 80% of wall thickness (d/t = {d_t:.3f} > 0.8). Not applicable."
        }
    
    if d_t == 0: # No defect depth
        return {
            "method": "Modified B31G (0.85dL)", "safe": True, "failure_pressure_mpa": float('inf'), # Or MAOP based on intact pipe
            "safe_pressure_mpa": float('inf'), "remaining_strength_factor_pct": 100.0,
            "folias_factor_M": None, "z_parameter": None, "d_over_t_ratio": d_t,
            "note": "Zero defect depth; pipe considered at full strength for this assessment."
        }

    # --- Core Modified B31G Calculations ---

    # Dimensionless length parameter z = L² / (D * t)
    # All units should be consistent (e.g., mm). z is dimensionless.
    z = (defect_length_mm ** 2) / (pipe_diameter_mm * wall_thickness_mm)

    # Applicability Limit: for z (Level 1 methods often limited, e.g. z <= 50)
    if z > 50: # This limit is common for Level 1 type assessments
        return {
            "method": "Modified B31G (0.85dL)", "safe": False, "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0, "remaining_strength_factor_pct": 0.0,
            "folias_factor_M": None, "z_parameter": z, "d_over_t_ratio": d_t,
            "note": f"Dimensionless defect parameter z ({z:.2f}) exceeds 50. Level-1 Modified B31G may not be applicable."
        }

    # Folias factor M (polynomial form common for Modified B31G / RSTRENG Level 1)
    # M = sqrt(1 + 0.6275 * z - 0.003375 * z^2)
    # This form is generally valid for z <= 50.
    # Ensure the term inside sqrt is non-negative, though with z<=50 and typical coefficients, it should be.
    m_term = 1 + 0.6275 * z - 0.003375 * (z**2)
    if m_term < 0:
        return {
            "method": "Modified B31G (0.85dL)", "safe": False, "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0, "remaining_strength_factor_pct": 0.0,
            "folias_factor_M": None, "z_parameter": z, "d_over_t_ratio": d_t,
            "note": f"Invalid Folias factor calculation: term inside sqrt is negative ({m_term:.3f}). Check z value or Folias coefficients."
        }
    M = math.sqrt(m_term)
    if M == 0: # Avoid division by zero later
         return {
            "method": "Modified B31G (0.85dL)", "safe": False, "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0, "remaining_strength_factor_pct": 0.0,
            "folias_factor_M": M, "z_parameter": z, "d_over_t_ratio": d_t,
            "note": "Folias factor M calculated as zero, leading to division by zero."
        }

    # Flow Stress (S_flow) for Modified B31G
    # S_flow = SMYS + 69 MPa (or SMYS + 10 ksi)
    S_flow_mpa = smys_mpa + 69.0  # 69 MPa is approx 10 ksi

    # Area ratio A/A_0 for Modified B31G (0.85dL method)
    # A/A_0 = (0.85 * d * L) / (t * L) = 0.85 * (d/t)
    A_A0_ratio = 0.85 * d_t

    # Failure Stress (Sf) or Hoop Stress at failure
    # Sf = S_flow * [ (1 - A/A_0) / (1 - (A/A_0) / M) ]
    denominator_sf = (1.0 - (A_A0_ratio / M))
    if denominator_sf <= 0 : # Avoid division by zero or non-physical result
        return {
            "method": "Modified B31G (0.85dL)", "safe": False, "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0, "remaining_strength_factor_pct": 0.0,
            "folias_factor_M": M, "z_parameter": z, "d_over_t_ratio": d_t,
            "note": f"Failure stress calculation error: Denominator (1 - (A/A0)/M) is not positive ({denominator_sf:.3f}). Check inputs or M value."
        }
    
    failure_stress_mpa = S_flow_mpa * ( (1.0 - A_A0_ratio) / denominator_sf )

    # Failure Pressure (Pf) using Barlow's formula with failure stress
    # Pf = (2 * Sf * t) / D
    failure_pressure_mpa = (2.0 * failure_stress_mpa * wall_thickness_mm) / pipe_diameter_mm
    
    # Safe Operating Pressure (P_safe) with user-specified safety factor
    safe_pressure_mpa = failure_pressure_mpa / safety_factor

    # Remaining Strength Factor (RSF) as a percentage
    # RSF = Sf / S_flow
    remaining_strength_factor_pct = (failure_stress_mpa / S_flow_mpa) * 100.0 if S_flow_mpa != 0 else 0.0
    
    return {
        "method": "Modified B31G (0.85dL)",
        "safe": True,
        "failure_pressure_mpa": failure_pressure_mpa,
        "safe_pressure_mpa": safe_pressure_mpa,
        "remaining_strength_pct": remaining_strength_factor_pct,
        "folias_factor_M": M,
        "z_parameter": z,
        "d_over_t_ratio": d_t,
        "note": f"Calculation successful. z={z:.2f}, d/t={d_t:.3f}, A/A0_ratio={A_A0_ratio:.3f}"
    }



def calculate_simplified_effective_area_method(defect_depth_pct, defect_length_mm, defect_width_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa, safety_factor=1.5):
    """
    Calculate remaining strength using a SIMPLIFIED effective area approximation.
    
    ⚠️ CRITICAL WARNING: This is NOT true RSTRENG (B31G Level 2) analysis!
    
    True RSTRENG requires:
    - Detailed river-bottom profile from ILI tools
    - Iterative effective area calculations
    - Commercial software (e.g., RSTRENG+, Pipeline Toolbox)
    
    This simplified method:
    - Uses basic geometric approximations
    - May be non-conservative for complex corrosion patterns
    - Should NOT be used for critical integrity decisions
    - Is provided for preliminary screening only
    
    For pipeline integrity decisions requiring RSTRENG analysis, use:
    - Certified RSTRENG software
    - Actual ILI river-bottom profiles
    - Qualified pipeline integrity engineers
    
    Parameters:
    - defect_depth_pct: Defect depth as percentage of wall thickness
    - defect_length_mm: Defect length (mm)
    - defect_width_mm: Defect width (mm)
    - pipe_diameter_mm: Outside diameter of pipe (mm)
    - wall_thickness_mm: Nominal wall thickness (mm)
    - smys_mpa: Specified Minimum Yield Strength (MPa)
    - safety_factor: Safety factor (default 1.5)
    
    Returns:
    - Dict with results including failure pressure and safe pressure
    """
    # Convert depth from percentage to actual depth
    defect_depth_mm = (defect_depth_pct / 100.0) * wall_thickness_mm
    
    # Depth to thickness ratio
    d_t = defect_depth_mm / wall_thickness_mm
    
    # If depth is greater than 80% of wall thickness, it's invalid
    if d_t > 0.8:
        return {
            "method": "RSTRENG (Simplified)",
            "safe": False,
            "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0,
            "remaining_strength_pct": 0.0,
            "folias_factor_M": None,
            "note": "Defect depth exceeds 80% of wall thickness"
        }
    
    # Calculate effective area factor (simplified approach)
    # For true RSTRENG, we would need the river-bottom profile
    # We'll use a simplified approach based on width/circumference ratio
    width_factor = min(1.0, defect_width_mm / (np.pi * pipe_diameter_mm / 4))
    
    # Dimensionless length parameter z = L² / (D * t)
    z = (defect_length_mm ** 2) / (pipe_diameter_mm * wall_thickness_mm)
    
    # Applicability limit
    if z > 50:
        return {
            "method": "RSTRENG (Simplified)",
            "safe": False,
            "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0,
            "remaining_strength_pct": 0.0,
            "folias_factor_M": None,
            "note": f"Dimensionless defect parameter z ({z:.2f}) exceeds 50. Simplified RSTRENG may not be applicable."
        }
    
    # Folias factor (RSTRENG/Modified B31G form)
    m_term = 1 + 0.6275 * z - 0.003375 * (z**2)
    if m_term < 0:
        return {
            "method": "RSTRENG (Simplified)",
            "safe": False,
            "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0,
            "remaining_strength_pct": 0.0,
            "folias_factor_M": None,
            "note": f"Invalid Folias factor calculation"
        }
    M = math.sqrt(m_term)
    
    # Calculate effective depth using width factor adjustment
    # A more detailed approach would use the actual river-bottom profile
    # This is a simplified approximation
    effective_depth_factor = 0.85 + (0.15 * width_factor)
    effective_d_t = effective_depth_factor * d_t
    
    # Flow stress for RSTRENG (similar to Modified B31G)
    S_flow_mpa = smys_mpa + 69.0
    
    # Compute intact pipe hoop-burst pressure
    P_o_mpa = (2.0 * S_flow_mpa * wall_thickness_mm) / pipe_diameter_mm
    
    # Calculate failure pressure
    # This uses a similar formula to Modified B31G but with the effective depth
    A_A0_ratio = 0.85 * effective_d_t
    
    # Check denominator
    denominator = (1.0 - (A_A0_ratio / M))
    if denominator <= 0:
        return {
            "method": "RSTRENG (Simplified)",
            "safe": False,
            "failure_pressure_mpa": 0.0,
            "safe_pressure_mpa": 0.0,
            "remaining_strength_pct": 0.0,
            "folias_factor_M": M,
            "note": "Calculation error: denominator not positive"
        }
    
    # Calculate failure pressure
    failure_stress_mpa = S_flow_mpa * ((1.0 - A_A0_ratio) / denominator)
    failure_pressure_mpa = (2.0 * failure_stress_mpa * wall_thickness_mm) / pipe_diameter_mm
    
    # Apply safety factor
    safe_pressure_mpa = failure_pressure_mpa / safety_factor
    
    # Calculate remaining strength
    remaining_strength_pct = (failure_stress_mpa / S_flow_mpa) * 100.0
    
    return {
        "method": "RSTRENG (Simplified)",
        "safe": True,
        "failure_pressure_mpa": failure_pressure_mpa,
        "safe_pressure_mpa": safe_pressure_mpa,
        "remaining_strength_pct": remaining_strength_pct,
        "folias_factor_M": M,
        "effective_depth_factor": effective_depth_factor,
        "note": f"Calculation used simplified RSTRENG approach. Width factor: {width_factor:.2f}"
    }


def compute_corrosion_metrics_for_dataframe(defects_df, joints_df, pipe_diameter_mm, smys_mpa, safety_factor):
    """
    Compute B31G, Modified B31G, and Effective Area metrics for all defects in the dataframe.
    Wall thickness is extracted from joints_df based on each defect's joint number.
    NO DEFAULT VALUES ARE USED - missing wall thickness is a critical error.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - joints_df: DataFrame containing joint information with wall thickness
    - pipe_diameter_mm: Outside diameter of pipe in millimeters
    - smys_mpa: Specified Minimum Yield Strength in MPa
    - safety_factor: Safety factor to apply to all calculations
    
    Returns:
    - DataFrame with additional columns for corrosion assessment metrics
    
    Raises:
    - ValueError if required data is missing or invalid
    """

    # Check if we should use FFS combined defects
    if hasattr(st.session_state, 'use_ffs_combined') and st.session_state.use_ffs_combined:
        if hasattr(st.session_state, 'ffs_combined_defects'):
            defects_df = st.session_state.ffs_combined_defects
            st.info(f"Using FFS-combined defects ({len(defects_df)} defects after combination)")

    # Create a copy to avoid modifying the original dataframe
    enhanced_df = defects_df.copy()
    
    # Step 1: Validate joints_df has required columns
    if 'joint number' not in joints_df.columns:
        raise ValueError("joints_df must contain 'joint number' column")
    if 'wt nom [mm]' not in joints_df.columns:
        raise ValueError("joints_df must contain 'wt nom [mm]' column for wall thickness")
    
    # Step 2: Create wall thickness lookup and validate completeness
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    # Check for any missing wall thickness values in joints_df
    missing_wt_joints = joints_df[joints_df['wt nom [mm]'].isna()]
    if not missing_wt_joints.empty:
        missing_joint_numbers = missing_wt_joints['joint number'].tolist()
        raise ValueError(
            f"CRITICAL: {len(missing_wt_joints)} joints have missing wall thickness values. "
            f"Joint numbers: {missing_joint_numbers[:10]}{'...' if len(missing_joint_numbers) > 10 else ''}. "
            f"Please update your joints data with wall thickness values for ALL joints before proceeding."
        )
    
    # Check for invalid (zero or negative) wall thickness values
    invalid_wt_joints = joints_df[joints_df['wt nom [mm]'] <= 0]
    if not invalid_wt_joints.empty:
        invalid_joint_numbers = invalid_wt_joints['joint number'].tolist()
        raise ValueError(
            f"CRITICAL: {len(invalid_wt_joints)} joints have invalid (≤0) wall thickness values. "
            f"Joint numbers: {invalid_joint_numbers[:10]}{'...' if len(invalid_joint_numbers) > 10 else ''}"
        )
    
    # Step 3: Validate defects_df has required columns
    if 'joint number' not in defects_df.columns:
        raise ValueError("defects_df must contain 'joint number' column to link to wall thickness data")
    
    # Check that all defects have joint assignments
    missing_joint_defects = defects_df[defects_df['joint number'].isna()]
    if not missing_joint_defects.empty:
        raise ValueError(
            f"CRITICAL: {len(missing_joint_defects)} defects have no joint number assigned. "
            f"Cannot determine wall thickness for these defects. "
            f"Defect locations: {missing_joint_defects['log dist. [m]'].head(10).tolist()}"
        )
    
    # Check that all defect joint numbers exist in joints_df
    defect_joints = set(defects_df['joint number'].unique())
    available_joints = set(joints_df['joint number'].unique())
    missing_joints = defect_joints - available_joints
    if missing_joints:
        raise ValueError(
            f"CRITICAL: Defects reference joints that don't exist in joints_df: {list(missing_joints)[:10]}"
        )
    
    # Initialize new columns for each method
    methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
    
    for method in methods:
        enhanced_df[f'{method}_safe'] = True
        enhanced_df[f'{method}_failure_pressure_mpa'] = 0.0
        enhanced_df[f'{method}_safe_pressure_mpa'] = 0.0
        enhanced_df[f'{method}_remaining_strength_pct'] = 0.0
        enhanced_df[f'{method}_notes'] = ""
    
    # Add specific columns for certain methods
    enhanced_df['b31g_folias_factor'] = np.nan
    enhanced_df['b31g_flaw_type'] = ""
    enhanced_df['modified_b31g_folias_factor'] = np.nan
    enhanced_df['modified_b31g_z_parameter'] = np.nan
    enhanced_df['simplified_eff_area_folias_factor'] = np.nan
    enhanced_df['simplified_eff_area_effective_depth_factor'] = np.nan
    
    # Add a column to store the wall thickness used for each defect
    enhanced_df['wall_thickness_used_mm'] = np.nan
    
    # Process each defect
    for idx, row in defects_df.iterrows():
        # Get wall thickness for this defect's joint - NO DEFAULTS!
        joint_number = row['joint number']
        wall_thickness_mm = wt_lookup.get(joint_number)
        
        if wall_thickness_mm is None or pd.isna(wall_thickness_mm):
            # This should not happen given our validation above, but double-check
            raise ValueError(
                f"CRITICAL: Wall thickness not found for joint {joint_number} "
                f"(defect at {row.get('log dist. [m]', 'Unknown')}m). "
                f"This should have been caught in validation."
            )
        
        # Store the wall thickness used
        enhanced_df.loc[idx, 'wall_thickness_used_mm'] = wall_thickness_mm
        
        # Check if required defect dimension columns exist and have valid data
        required_cols = ['depth [%]', 'length [mm]', 'width [mm]']
        missing_data = []
        
        for col in required_cols:
            if col not in row or pd.isna(row[col]):
                missing_data.append(col)
        
        if missing_data:
            # Set all methods as invalid for this row
            for method in methods:
                enhanced_df.loc[idx, f'{method}_safe'] = False
                enhanced_df.loc[idx, f'{method}_notes'] = f"Missing required data: {', '.join(missing_data)}"
            continue
        
        depth_pct = float(row['depth [%]'])
        length_mm = float(row['length [mm]'])
        width_mm = float(row['width [mm]'])
        
        # Skip if any dimension is zero or negative
        if depth_pct <= 0 or length_mm <= 0 or width_mm <= 0:
            for method in methods:
                enhanced_df.loc[idx, f'{method}_safe'] = False
                enhanced_df.loc[idx, f'{method}_notes'] = "Invalid dimensional data (zero or negative values)"
            continue
        
        # Calculate B31G
        try:
            b31g_result = calculate_b31g(
                depth_pct, length_mm, pipe_diameter_mm, 
                wall_thickness_mm, smys_mpa, safety_factor
            )
            enhanced_df.loc[idx, 'b31g_safe'] = b31g_result['safe']
            enhanced_df.loc[idx, 'b31g_failure_pressure_mpa'] = b31g_result['failure_pressure_mpa']
            enhanced_df.loc[idx, 'b31g_safe_pressure_mpa'] = b31g_result['safe_pressure_mpa']
            enhanced_df.loc[idx, 'b31g_remaining_strength_pct'] = b31g_result['remaining_strength_pct']
            enhanced_df.loc[idx, 'b31g_folias_factor'] = b31g_result.get('folias_factor_M')
            enhanced_df.loc[idx, 'b31g_flaw_type'] = b31g_result.get('flaw_type_applied', '')
            enhanced_df.loc[idx, 'b31g_notes'] = b31g_result['note']
        except Exception as e:
            enhanced_df.loc[idx, 'b31g_safe'] = False
            enhanced_df.loc[idx, 'b31g_notes'] = f"Calculation error: {str(e)}"
        
        # Calculate Modified B31G
        try:
            mod_b31g_result = calculate_modified_b31g(
                depth_pct, length_mm, pipe_diameter_mm, 
                wall_thickness_mm, smys_mpa, safety_factor
            )
            enhanced_df.loc[idx, 'modified_b31g_safe'] = mod_b31g_result['safe']
            enhanced_df.loc[idx, 'modified_b31g_failure_pressure_mpa'] = mod_b31g_result['failure_pressure_mpa']
            enhanced_df.loc[idx, 'modified_b31g_safe_pressure_mpa'] = mod_b31g_result['safe_pressure_mpa']
            enhanced_df.loc[idx, 'modified_b31g_remaining_strength_pct'] = mod_b31g_result['remaining_strength_pct']
            enhanced_df.loc[idx, 'modified_b31g_folias_factor'] = mod_b31g_result.get('folias_factor_M')
            enhanced_df.loc[idx, 'modified_b31g_z_parameter'] = mod_b31g_result.get('z_parameter')
            enhanced_df.loc[idx, 'modified_b31g_notes'] = mod_b31g_result['note']
        except Exception as e:
            enhanced_df.loc[idx, 'modified_b31g_safe'] = False
            enhanced_df.loc[idx, 'modified_b31g_notes'] = f"Calculation error: {str(e)}"
        
        # Calculate Effective Area (Simplified RSTRENG)
        try:
            effective_area_result = calculate_simplified_effective_area_method(
                depth_pct, length_mm, width_mm, pipe_diameter_mm, 
                wall_thickness_mm, smys_mpa, safety_factor
            )
            enhanced_df.loc[idx, 'simplified_eff_area_safe'] = effective_area_result['safe']
            enhanced_df.loc[idx, 'simplified_eff_area_failure_pressure_mpa'] = effective_area_result['failure_pressure_mpa']
            enhanced_df.loc[idx, 'simplified_eff_area_safe_pressure_mpa'] = effective_area_result['safe_pressure_mpa']
            enhanced_df.loc[idx, 'simplified_eff_area_remaining_strength_pct'] = effective_area_result['remaining_strength_pct']
            enhanced_df.loc[idx, 'simplified_eff_area_folias_factor'] = effective_area_result.get('folias_factor_M')
            enhanced_df.loc[idx, 'simplified_eff_area_effective_depth_factor'] = effective_area_result.get('effective_depth_factor')
            enhanced_df.loc[idx, 'simplified_eff_area_notes'] = effective_area_result['note']
        except Exception as e:
            enhanced_df.loc[idx, 'simplified_eff_area_safe'] = False
            enhanced_df.loc[idx, 'simplified_eff_area_notes'] = f"Calculation error: {str(e)}"
    
    return enhanced_df


def get_wall_thickness_statistics(joints_df):
    """
    Get statistics about wall thickness distribution across joints.
    
    Parameters:
    - joints_df: DataFrame containing joint information
    
    Returns:
    - Dictionary with wall thickness statistics
    """
    if 'wt nom [mm]' not in joints_df.columns:
        return {"error": "Wall thickness column 'wt nom [mm]' not found"}
    
    wt_values = joints_df['wt nom [mm]'].dropna()
    
    if len(wt_values) == 0:
        return {"error": "No valid wall thickness data found"}
    
    return {
        "count": len(wt_values),
        "min": wt_values.min(),
        "max": wt_values.max(),
        "mean": wt_values.mean(),
        "std": wt_values.std(),
        "unique_values": sorted(wt_values.unique())
    }


def create_assessment_summary_stats(enhanced_df):
    """
    Create summary statistics for the corrosion assessment results.
    
    Parameters:
    - enhanced_df: DataFrame with computed corrosion metrics
    
    Returns:
    - Dictionary with summary statistics
    """
    methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
    summary = {}
    
    total_defects = len(enhanced_df)
    
    for method in methods:
        safe_col = f'{method}_safe'
        failure_pressure_col = f'{method}_failure_pressure_mpa'
        safe_pressure_col = f'{method}_safe_pressure_mpa'
        remaining_strength_col = f'{method}_remaining_strength_pct'
        
        # Count safe vs unsafe defects
        safe_count = enhanced_df[safe_col].sum()
        unsafe_count = total_defects - safe_count
        
        # Calculate statistics for safe defects only
        safe_defects = enhanced_df[enhanced_df[safe_col] == True]
        
        if len(safe_defects) > 0:
            avg_failure_pressure = safe_defects[failure_pressure_col].mean()
            min_failure_pressure = safe_defects[failure_pressure_col].min()
            avg_safe_pressure = safe_defects[safe_pressure_col].mean()
            min_safe_pressure = safe_defects[safe_pressure_col].min()
            avg_remaining_strength = safe_defects[remaining_strength_col].mean()
            min_remaining_strength = safe_defects[remaining_strength_col].min()
        else:
            avg_failure_pressure = min_failure_pressure = 0
            avg_safe_pressure = min_safe_pressure = 0
            avg_remaining_strength = min_remaining_strength = 0
        
        summary[method] = {
            'total_defects': total_defects,
            'safe_defects': safe_count,
            'unsafe_defects': unsafe_count,
            'safe_percentage': (safe_count / total_defects * 100) if total_defects > 0 else 0,
            'avg_failure_pressure_mpa': avg_failure_pressure,
            'min_failure_pressure_mpa': min_failure_pressure,
            'avg_safe_pressure_mpa': avg_safe_pressure,
            'min_safe_pressure_mpa': min_safe_pressure,
            'avg_remaining_strength_pct': avg_remaining_strength,
            'min_remaining_strength_pct': min_remaining_strength
        }
    
    return summary

def create_enhanced_csv_download_link(enhanced_df, year):
    """
    Create a download link for the enhanced dataframe with corrosion metrics.
    
    Parameters:
    - enhanced_df: DataFrame with computed corrosion metrics
    - year: Year for filename
    
    Returns:
    - HTML string with download link
    """
    csv = enhanced_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    filename = f"defects_with_corrosion_assessment_{year}.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="custom-button" style="display:inline-block;text-decoration:none;margin-top:10px;font-size:0.9em;padding:8px 15px;background-color:#27AE60;color:white;border-radius:5px;">📊 Download Enhanced CSV with Corrosion Metrics</a>'
    return href

def create_joint_assessment_visualization(joint_summary, method='b31g', metric='remaining_strength_pct', 
                                        aggregation='min', pipe_diameter=1.0):
    """
    Optimized pipeline visualization using bar traces
    """
    # Build column name
    color_col = f'{method}_{metric}_{aggregation}'
    defect_count_col = f'{method}_total_defects'
    safe_defects_col = f'{method}_safe_defects'
    
    # Check if column exists
    if color_col not in joint_summary.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Data not available: {color_col}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create a copy for processing
    viz_data = joint_summary.copy()
    
    # Calculate joint positions
    viz_data = viz_data.sort_values('log dist. [m]')
    viz_data['joint_start'] = viz_data['log dist. [m]']
    viz_data['joint_end'] = viz_data['joint_start'] + viz_data['joint length [m]']
    
    # Prepare color data
    color_values = viz_data[color_col]
    
    # Set up visualization parameters
    method_name = {'b31g': 'B31G Original', 'modified_b31g': 'Modified B31G', 'simplified_eff_area': 'Simplified Eff. Area'}[method]
    
    if 'pressure' in metric:
        colorscale = [
            [0.0, 'red'],      # low values → red
            [0.5, 'yellow'],   # mid values → yellow
            [1.0, 'green']     # high values → green
        ]
        if 'failure' in metric:
            metric_label = 'Failure Pressure'
            unit = 'MPa'
        else:
            metric_label = 'Safe Operating Pressure'
            unit = 'MPa'
    else:  # remaining strength
        colorscale = [
            [0.0, 'red'],      # low values → red
            [0.5, 'yellow'],   # mid values → yellow
            [1.0, 'green']     # high values → green
        ]
        metric_label = 'Remaining Strength'
        unit = '%'
    
    color_title = f"{metric_label} ({unit})<br>{method_name} - {aggregation.title()}"
    
    # Hover template
    hover_template = (
        "<b>Joint %{customdata[0]}</b><br>"
        "Distance: %{customdata[1]:.1f} m<br>"
        "Length: %{customdata[2]:.2f} m<br>"
        f"{metric_label}: %{{customdata[3]:.1f}} {unit}<br>"
        "Total Defects: %{customdata[4]:.0f}<br>"
        "Safe Defects: %{customdata[5]:.0f}<br>"
        "Wall Thickness: %{customdata[6]:.1f} mm<br>"
        "<extra></extra>"
    )
    
    # Prepare custom data
    custom_data = np.stack([
        viz_data['joint number'].astype(str),
        viz_data['joint_start'],
        viz_data['joint length [m]'],
        viz_data[color_col].fillna(0),
        viz_data.get(defect_count_col, 0).fillna(0),
        viz_data.get(safe_defects_col, 0).fillna(0),
        viz_data.get('wt nom [mm]', 10.0).fillna(10.0)
    ], axis=-1)
    
    # Create the figure
    fig = go.Figure()
    
    # Add pipeline segments as bars
    fig.add_trace(go.Bar(
        x=viz_data['joint_start'],
        y=[50] * len(viz_data),  # Fixed height
        width=viz_data['joint length [m]'],
        base=0,  # Start at y=0
        marker=dict(
            color=color_values,
            colorscale=colorscale,
            cmin=color_values.min(),
            cmax=color_values.max(),
            line=dict(color='black', width=1)
        ),
        customdata=custom_data,
        hovertemplate=hover_template,
        showlegend=False,
        name="Pipeline Joints",
        # CRITICAL: Enable hover on entire bar area
        hoverlabel=dict(namelength=0)
    ))
    
    # Add colorbar
    fig.update_traces(marker_showscale=True,
                      selector=dict(type='bar'),
                      marker_colorbar=dict(
                          title=color_title,
                          thickness=15,
                          len=0.7,
                          tickformat=".1f"
                      ))
    
    # Simple layout
    fig.update_layout(
        title=f"Pipeline Assessment - {method_name}<br><sub>{metric_label} ({aggregation.title()})</sub>",
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title=f"Pipeline Representation (Actual Ø: {pipe_diameter:.2f} m)",
        plot_bgcolor="white",
        height=400,
        hovermode='x unified',
        yaxis=dict(
            range=[0, 55],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)"
        ),
        bargap=0  # No gap between bars
    )
    return fig


def calculate_intact_pipe_values(pipe_diameter_mm, wall_thickness_mm, smys_mpa):
    """
    Calculate theoretical values for an intact pipe (no defects) using the same methods.
    
    Parameters:
    - pipe_diameter_mm: Pipe diameter in mm
    - wall_thickness_mm: Wall thickness in mm  
    - smys_mpa: SMYS in MPa
    
    Returns:
    - Dictionary with intact pipe values for each method
    """
    # For an intact pipe (0% defect depth), calculate theoretical values
    intact_values = {}
    
    try:
        # Calculate for each method using 0% depth (intact condition)
        for method_name, calc_func in [
            ('b31g', calculate_b31g),
            ('modified_b31g', calculate_modified_b31g),
            ('rstreng', calculate_simplified_effective_area_method)
        ]:
            if method_name == 'rstreng':
                # RSTRENG needs width parameter, use a small nominal value
                result = calc_func(0.01, 1.0, 1.0, pipe_diameter_mm, wall_thickness_mm, smys_mpa)
            else:
                # B31G methods only need depth and length
                result = calc_func(0.01, 1.0, pipe_diameter_mm, wall_thickness_mm, smys_mpa)
            
            if result['safe']:
                intact_values[method_name] = {
                    'failure_pressure_mpa': result['failure_pressure_mpa'],
                    'safe_pressure_mpa': result['safe_pressure_mpa'],
                    'remaining_strength_pct': 100.0  # Intact pipe has 100% strength
                }
            else:
                # Fallback to theoretical calculation if methods fail
                # Use Barlow's formula: P = 2*SMYS*t/D  
                theoretical_pressure = (2.0 * smys_mpa * wall_thickness_mm) / pipe_diameter_mm
                intact_values[method_name] = {
                    'failure_pressure_mpa': theoretical_pressure,
                    'safe_pressure_mpa': theoretical_pressure / 1.39,  # Apply safety factor
                    'remaining_strength_pct': 100.0
                }
                
    except Exception as e:
        # Ultimate fallback using Barlow's formula
        theoretical_pressure = (2.0 * smys_mpa * wall_thickness_mm) / pipe_diameter_mm
        for method_name in ['b31g', 'modified_b31g', 'rstreng']:
            intact_values[method_name] = {
                'failure_pressure_mpa': theoretical_pressure,
                'safe_pressure_mpa': theoretical_pressure / 1.39,
                'remaining_strength_pct': 100.0
            }
    
    return intact_values


def aggregate_assessment_results_by_joint(enhanced_df, joints_df, pipe_diameter_mm, smys_mpa, safety_factor=1.39):
    """
    Enhanced aggregation that includes wall thickness data for visualization.
    NEVER uses defaults for critical parameters - all data must be provided.
    
    Parameters:
    - enhanced_df: DataFrame with computed corrosion metrics
    - joints_df: DataFrame with joint information including wall thickness
    - pipe_diameter_mm: Pipe diameter in mm (REQUIRED - no defaults)
    - smys_mpa: SMYS in MPa (REQUIRED - no defaults)
    - safety_factor: Safety factor used in calculations
    
    Returns:
    - DataFrame with joint-level aggregated assessment results including wall thickness
    
    Raises:
    - ValueError if required parameters are missing or invalid
    """
    # Validate critical parameters
    if pipe_diameter_mm is None or pipe_diameter_mm <= 0:
        raise ValueError(f"Invalid pipe diameter: {pipe_diameter_mm}. Must be positive value in mm.")
    
    if smys_mpa is None or smys_mpa <= 0:
        raise ValueError(f"Invalid SMYS: {smys_mpa}. Must be positive value in MPa.")
    
    # Validate joints_df has required columns
    required_joint_cols = ['joint number', 'log dist. [m]', 'joint length [m]', 'wt nom [mm]']
    missing_cols = [col for col in required_joint_cols if col not in joints_df.columns]
    if missing_cols:
        raise ValueError(f"joints_df missing required columns: {missing_cols}")
    
    # Check for any missing wall thickness values
    missing_wt_joints = joints_df[joints_df['wt nom [mm]'].isna()]
    if not missing_wt_joints.empty:
        raise ValueError(
            f"CRITICAL: {len(missing_wt_joints)} joints have missing wall thickness values. "
            f"Joint numbers: {missing_wt_joints['joint number'].tolist()[:10]}... "
            f"Cannot proceed without complete wall thickness data."
        )
    
    # Check for invalid wall thickness values
    invalid_wt = joints_df[joints_df['wt nom [mm]'] <= 0]
    if not invalid_wt.empty:
        raise ValueError(
            f"CRITICAL: {len(invalid_wt)} joints have invalid wall thickness (≤0). "
            f"Joint numbers: {invalid_wt['joint number'].tolist()[:10]}"
        )
    
    # Start with joints dataframe to ensure all joints are included
    joint_summary = joints_df[required_joint_cols].copy()
    
    # Focus only on the 3 core metrics
    methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
    metrics = ['failure_pressure_mpa', 'safe_pressure_mpa', 'remaining_strength_pct']
    
    # Initialize columns for each method and metric
    for method in methods:
        # Initialize columns for the 3 core metrics
        for metric in metrics:
            joint_summary[f'{method}_{metric}_defect_count'] = 0
            joint_summary[f'{method}_{metric}_min'] = np.nan
            joint_summary[f'{method}_{metric}_max'] = np.nan
            joint_summary[f'{method}_{metric}_avg'] = np.nan
        
        # Initialize safety columns (needed for enhanced hover info)
        joint_summary[f'{method}_total_defects'] = 0
        joint_summary[f'{method}_safe_defects'] = 0
    
    # Process defects by joint if enhanced_df has data
    if len(enhanced_df) > 0:
        # Group enhanced_df by joint number
        for joint_num in joint_summary['joint number'].unique():
            joint_defects = enhanced_df[enhanced_df['joint number'] == joint_num]
            
            if len(joint_defects) == 0:
                continue  # No defects in this joint
            
            # Get the row index for this joint in joint_summary
            joint_idx = joint_summary[joint_summary['joint number'] == joint_num].index[0]
            
            # Calculate statistics for each method
            for method in methods:
                # Count total and safe defects
                safe_col = f'{method}_safe'
                if safe_col in joint_defects.columns:
                    total_defects = len(joint_defects)
                    safe_defects = joint_defects[safe_col].sum()
                    
                    joint_summary.loc[joint_idx, f'{method}_total_defects'] = total_defects
                    joint_summary.loc[joint_idx, f'{method}_safe_defects'] = safe_defects
                
                # Calculate statistics for each metric
                for metric in metrics:
                    col_name = f'{method}_{metric}'
                    if col_name in joint_defects.columns:
                        # Get valid (non-null) values
                        valid_values = joint_defects[col_name].dropna()
                        
                        if len(valid_values) > 0:
                            joint_summary.loc[joint_idx, f'{col_name}_defect_count'] = len(valid_values)
                            joint_summary.loc[joint_idx, f'{col_name}_min'] = valid_values.min()
                            joint_summary.loc[joint_idx, f'{col_name}_max'] = valid_values.max()
                            joint_summary.loc[joint_idx, f'{col_name}_avg'] = valid_values.mean()
    
    # Calculate intact pipe values for joints without defects
    # Each joint may have different wall thickness, so calculate per joint
    for idx, joint in joint_summary.iterrows():
        joint_wt = joint['wt nom [mm]']
        
        # Validate wall thickness for this specific joint
        if pd.isna(joint_wt) or joint_wt <= 0:
            raise ValueError(f"Invalid wall thickness {joint_wt} for joint {joint['joint number']}")
        
        # For joints with no defects, calculate intact pipe values
        for method in methods:
            if joint_summary.loc[idx, f'{method}_total_defects'] == 0:
                # Calculate theoretical intact pipe values for this specific joint
                # Using the actual wall thickness for THIS joint
                
                if method == 'b31g':
                    # B31G uses flow stress = 1.1 * SMYS
                    flow_stress = 1.1 * smys_mpa
                    theoretical_pressure = (2.0 * flow_stress * joint_wt) / pipe_diameter_mm
                elif method == 'modified_b31g':
                    # Modified B31G uses flow stress = SMYS + 69 MPa
                    flow_stress = smys_mpa + 69.0
                    theoretical_pressure = (2.0 * flow_stress * joint_wt) / pipe_diameter_mm
                elif method == 'effective_area':
                    # Effective area method (simplified RSTRENG) similar to Modified B31G
                    flow_stress = smys_mpa + 69.0
                    theoretical_pressure = (2.0 * flow_stress * joint_wt) / pipe_diameter_mm
                
                safe_pressure = theoretical_pressure / safety_factor
                
                # Set all statistics to intact pipe values
                for metric in metrics:
                    if metric == 'failure_pressure_mpa':
                        value = theoretical_pressure
                    elif metric == 'safe_pressure_mpa':
                        value = safe_pressure
                    elif metric == 'remaining_strength_pct':
                        value = 100.0
                    
                    joint_summary.loc[idx, f'{method}_{metric}_min'] = value
                    joint_summary.loc[idx, f'{method}_{metric}_max'] = value
                    joint_summary.loc[idx, f'{method}_{metric}_avg'] = value
    
    # Add a summary column for quick reference
    joint_summary['has_defects'] = joint_summary['b31g_total_defects'] > 0
    
    # Validate output
    # Ensure no NaN values in critical columns for joints without defects
    for method in methods:
        for metric in metrics:
            col_min = f'{method}_{metric}_min'
            intact_joints = joint_summary[~joint_summary['has_defects']]
            if not intact_joints.empty:
                nan_count = intact_joints[col_min].isna().sum()
                if nan_count > 0:
                    raise ValueError(
                        f"Internal error: {nan_count} intact joints have NaN values "
                        f"for {col_min}. This should not happen."
                    )
    
    return joint_summary

def render_corrosion_assessment_view():
    """Display the enhanced corrosion assessment view with pressure analysis."""
    st.markdown('<h2 class="section-header">Corrosion Assessment</h2>', unsafe_allow_html=True)
    
    # Check if datasets are available
    datasets = get_state('datasets', {})
    if not datasets:
        st.info("""
            **No datasets available**
            Please upload pipeline inspection data using the sidebar to enable corrosion assessment.
        """)
        return
    
    # Create main container
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    # Dataset selection
    st.markdown("<div class='section-header'>Select Dataset for Assessment</div>", unsafe_allow_html=True)
    
    available_years = sorted(datasets.keys())
    selected_year = st.selectbox(
        "Choose inspection year to assess",
        options=available_years,
        key="corrosion_assessment_year"
    )
    
    # Get selected dataset
    defects_df = datasets[selected_year]['defects_df']
    joints_df = datasets[selected_year]['joints_df']
    pipe_diameter_stored = datasets[selected_year].get('pipe_diameter', 1.0)
    
    # Display dataset info
    total_defects = len(defects_df)
    total_joints = len(joints_df)
    st.markdown(f"**Selected Dataset:** {selected_year} ({total_defects} defects, {total_joints} joints)")
    
    # Check for required columns in defects_df
    required_defect_cols = ['depth [%]', 'length [mm]', 'width [mm]', 'joint number']
    missing_defect_cols = [col for col in required_defect_cols if col not in defects_df.columns]
    
    # Check for required columns in joints_df
    required_joint_cols = ['joint number', 'wt nom [mm]']
    missing_joint_cols = [col for col in required_joint_cols if col not in joints_df.columns]
    
    if missing_defect_cols or missing_joint_cols:
        error_msg = "**Missing required columns for corrosion assessment:**\n"
        if missing_defect_cols:
            error_msg += f"- Defects data: {', '.join(missing_defect_cols)}\n"
        if missing_joint_cols:
            error_msg += f"- Joints data: {', '.join(missing_joint_cols)}\n"
        error_msg += "\nPlease ensure your dataset contains all required information."
        
        st.error(error_msg)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Check for valid data
    valid_defects = defects_df.dropna(subset=required_defect_cols)
    if len(valid_defects) == 0:
        st.warning("No defects with complete dimensional data found in the selected dataset.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if len(valid_defects) < total_defects:
        st.warning(f"Only {len(valid_defects)} out of {total_defects} defects have complete dimensional data for assessment.")
    
    # Display wall thickness statistics
    wt_stats = get_wall_thickness_statistics(joints_df)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close selection container
    
    # Enhanced Parameters input section
    st.markdown("<div class='section-header'>Pipeline Parameters</div>", unsafe_allow_html=True)
    
    # Create 5 columns for all parameters
    param_col1, param_col2, param_col3, param_col4, param_col5 = st.columns(5)
    
    with param_col1:
        pipe_diameter_m = st.number_input(
            "Pipe Outside Diameter (m)",
            min_value=0.1,
            max_value=3.0,
            value=pipe_diameter_stored,
            step=0.1,
            format="%.2f",
            key="pipe_diameter_corrosion",
            help="Enter the OUTSIDE diameter (OD) of the pipeline in meters"
        )
        st.caption(f"= {pipe_diameter_m * 1000:.0f} mm = {pipe_diameter_m * 39.37:.1f} inches")
    
    with param_col2:
        pipe_grade = st.selectbox(
            "Pipe Grade",
            options=["API 5L X42", "API 5L X52", "API 5L X60", "API 5L X65", "API 5L X70", "Custom"],
            index=1,
            key="pipe_grade_corrosion"
        )
        
        grade_to_smys = {
            "API 5L X42": 290,
            "API 5L X52": 358,
            "API 5L X60": 413,
            "API 5L X65": 448,
            "API 5L X70": 482
        }
        
        if pipe_grade != "Custom":
            smys_mpa = grade_to_smys[pipe_grade]
            st.caption(f"SMYS: {smys_mpa} MPa ({smys_mpa * 145.038:.0f} psi)")
        else:
            smys_mpa = st.number_input(
                "Custom SMYS (MPa)",
                min_value=200.0,
                max_value=800.0,
                value=358.0,
                step=1.0,
                format="%.0f",
                key="smys_custom"
            )

    with param_col3:
        safety_factor_type = st.selectbox(
            "Safety Factor Standard",
            options=["Gas Pipeline - Class 1 (1.39)", "Liquid Pipeline (1.25)", "Custom"],
            key="safety_factor_type",
            help="Select based on your pipeline type and regulatory requirements"
        )
        
        if safety_factor_type == "Gas Pipeline - Class 1 (1.39)":
            safety_factor = 1.39
            st.caption("Per ASME B31G for gas pipelines")
        elif safety_factor_type == "Liquid Pipeline (1.25)":
            safety_factor = 1.25
            st.caption("Common for liquid pipelines")
        else:
            safety_factor = st.number_input(
                "Custom Safety Factor",
                min_value=1.0,
                max_value=2.0,
                value=1.39,
                step=0.01,
                format="%.2f",
                key="safety_factor_custom"
            )

    # NEW: Analysis Pressure
    with param_col4:
        analysis_pressure_mpa = st.number_input(
            "Analysis Pressure (MPa)",
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.1,
            format="%.1f",
            key="analysis_pressure_corrosion",
            help="Current or proposed operating pressure for assessment"
        )
        st.caption(f"= {analysis_pressure_mpa * 145.038:.0f} psi")
    
    # NEW: Maximum Allowable Pressure  
    with param_col5:
        max_allowable_pressure_mpa = st.number_input(
            "Max Allowable Pressure (MPa)",
            min_value=0.1,
            max_value=20.0,
            value=8.0,
            step=0.1,
            format="%.1f",
            key="max_allowable_pressure_corrosion", 
            help="Maximum Allowable Operating Pressure (MAOP) - regulatory/design limit"
        )
        st.caption(f"= {max_allowable_pressure_mpa * 145.038:.0f} psi")

    # Convert diameter to mm for calculations
    pipe_diameter_mm = pipe_diameter_m * 1000
    
    st.markdown('</div>', unsafe_allow_html=True) 
    
    # Assessment button and results
    st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
 
    # Add option for handling missing wall thickness
    with st.expander("Advanced Options"):
        missing_wt_action = st.radio(
            "If wall thickness is missing for any joint:",
            options=[
                "Stop and show error (Recommended)",
                "Skip affected defects"
            ],
            index=0,
            key="missing_wt_action"
        )
    
    if st.button("Perform Enhanced Corrosion Assessment", key="perform_enhanced_assessment", use_container_width=True):
        with st.spinner("Computing enhanced corrosion metrics with pressure analysis..."):
            try:
                # Compute enhanced corrosion metrics
                enhanced_df = compute_enhanced_corrosion_metrics(
                    defects_df, joints_df, pipe_diameter_mm, smys_mpa, safety_factor,
                    analysis_pressure_mpa, max_allowable_pressure_mpa
                )

                # Create pressure-based summary
                pressure_summary = create_pressure_based_summary_stats(
                    enhanced_df, analysis_pressure_mpa, max_allowable_pressure_mpa
                )
                
                # Store results in session state
                st.session_state.enhanced_defects_df = enhanced_df
                st.session_state.pressure_summary = pressure_summary
                st.session_state.assessment_year = selected_year
                st.session_state.assessment_pipe_diameter = pipe_diameter_m
                
                st.success("✅ Enhanced corrosion assessment completed successfully!")
                st.rerun()  # Refresh to show results immediately
                
            except Exception as e:
                st.error(f"Error during enhanced corrosion assessment: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)  # Close assessment button container
    
    # Enhanced Results Display - Always show if results exist
    if (hasattr(st.session_state, 'enhanced_defects_df') and 
        hasattr(st.session_state, 'pressure_summary') and
        st.session_state.assessment_year == selected_year):
        
        enhanced_df = st.session_state.enhanced_defects_df
        pressure_summary = st.session_state.pressure_summary
        
        # Traditional Assessment Results Section
        st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 Traditional Assessment Results</div>", unsafe_allow_html=True)
        
        # Create summary statistics
        summary_stats = create_assessment_summary_stats(enhanced_df)
        
        # Create tabs for each method
        method_tabs = st.tabs(["B31G Original", "Modified B31G", "RSTRENG"])
        
        with method_tabs[0]:
            st.markdown("### B31G Original Level-1 Results")
            b31g_stats = summary_stats['b31g']
            
            metrics_data = [
                ("Total Defects", f"{b31g_stats['total_defects']}", None),
                ("Safe Defects", f"{b31g_stats['safe_defects']}", f"{b31g_stats['safe_percentage']:.1f}%"),
                ("Min Failure Pressure", f"{b31g_stats['min_failure_pressure_mpa']:.1f} MPa", None),
                ("Min Remaining Strength", f"{b31g_stats['min_remaining_strength_pct']:.1f}%", None)
            ]
            create_metrics_row(metrics_data)
        
        with method_tabs[1]:
            st.markdown("### Modified B31G (0.85dL) Results")
            mod_stats = summary_stats['modified_b31g']
            
            metrics_data = [
                ("Total Defects", f"{mod_stats['total_defects']}", None),
                ("Safe Defects", f"{mod_stats['safe_defects']}", f"{mod_stats['safe_percentage']:.1f}%"),
                ("Min Failure Pressure", f"{mod_stats['min_failure_pressure_mpa']:.1f} MPa", None),
                ("Min Remaining Strength", f"{mod_stats['min_remaining_strength_pct']:.1f}%", None)
            ]
            create_metrics_row(metrics_data)
        
        with method_tabs[2]:
            st.markdown("### RSTRENG (Simplified) Results")
            rstreng_stats = summary_stats['simplified_eff_area']
            
            metrics_data = [
                ("Total Defects", f"{rstreng_stats['total_defects']}", None),
                ("Safe Defects", f"{rstreng_stats['safe_defects']}", f"{rstreng_stats['safe_percentage']:.1f}%"),
                ("Min Failure Pressure", f"{rstreng_stats['min_failure_pressure_mpa']:.1f} MPa", None),
                ("Min Remaining Strength", f"{rstreng_stats['min_remaining_strength_pct']:.1f}%", None)
            ]
            create_metrics_row(metrics_data)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close traditional results container
        
        # NEW: Pressure Assessment Results Section
        st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎯 Pressure-Based Assessment Results</div>", unsafe_allow_html=True)
        
        # Summary metrics for each method
        pressure_method_tabs = st.tabs(["B31G Pressure Analysis", "Modified B31G Pressure Analysis", "RSTRENG Pressure Analysis"])
        
        methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
        method_names = ['B31G Original', 'Modified B31G', 'RSTRENG']
        
        for idx, (method_tab, method, method_name) in enumerate(zip(pressure_method_tabs, methods, method_names)):
            with method_tab:
                method_stats = pressure_summary[method]
                
                # Key metrics row
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric(
                        "Safe at Analysis Pressure", 
                        f"{method_stats['defects_safe_at_analysis_pressure']}/{method_stats['total_valid_defects']}",
                        f"{method_stats['pct_safe_at_analysis_pressure']:.1f}%"
                    )
                
                with metrics_col2:
                    st.metric(
                        "Within MAOP Limits",
                        f"{method_stats['defects_within_maop']}/{method_stats['total_valid_defects']}",
                        f"{method_stats['pct_within_maop']:.1f}%"
                    )
                
                with metrics_col3:
                    st.metric(
                        "Max Pipeline Pressure",
                        f"{method_stats['max_pipeline_operating_pressure_mpa']:.1f} MPa",
                        f"{method_stats['max_pipeline_operating_pressure_mpa'] * 145.038:.0f} psi"
                    )
                
                with metrics_col4:
                    pressure_margin = method_stats['max_pipeline_operating_pressure_mpa'] - analysis_pressure_mpa
                    margin_pct = (pressure_margin / analysis_pressure_mpa * 100) if analysis_pressure_mpa > 0 else 0
                    st.metric(
                        "Pressure Margin",
                        f"{pressure_margin:.1f} MPa",
                        f"{margin_pct:.1f}%"
                    )
                
                # Operational status breakdown
                st.markdown("#### Operational Status Breakdown")
                status_counts = method_stats['operational_status_counts']
                
                status_display = []
                for status, count in status_counts.items():
                    pct = (count / method_stats['total_valid_defects'] * 100) if method_stats['total_valid_defects'] > 0 else 0
                    
                    # Add color coding based on status
                    if status == 'ACCEPTABLE':
                        emoji = "✅"
                        color = "green"
                    elif status == 'PRESSURE_DERATION_REQUIRED':
                        emoji = "⚠️"
                        color = "orange"
                    elif status == 'IMMEDIATE_REPAIR_REQUIRED':
                        emoji = "❌"
                        color = "red"
                    else:
                        emoji = "❓"
                        color = "gray"
                    
                    status_display.append(f"{emoji} **{status.replace('_', ' ').title()}**: {count} defects ({pct:.1f}%)")
                
                if status_display:
                    for status_line in status_display:
                        st.markdown(status_line)
                else:
                    st.info("No operational status data available")
                
                # Visualization
                st.markdown("#### Pressure Assessment Visualization")
                pressure_viz = create_pressure_assessment_visualization(enhanced_df, method)
                st.plotly_chart(pressure_viz, use_container_width=True)
        
        # Overall recommendations
        st.markdown("#### 🎯 Operational Recommendations")
        
        # Find the most conservative method result
        min_pipeline_pressure = min(
            pressure_summary[method]['max_pipeline_operating_pressure_mpa'] 
            for method in methods
        )
        
        most_conservative_method = None
        for method in methods:
            if pressure_summary[method]['max_pipeline_operating_pressure_mpa'] == min_pipeline_pressure:
                most_conservative_method = method.replace('_', ' ').title()
                break
        
        # Analysis vs maximum pressure validation
        if analysis_pressure_mpa > max_allowable_pressure_mpa:
            st.error("❌ **Parameter Error**: Analysis pressure cannot exceed Maximum Allowable Operating Pressure (MAOP)")
        
        # Main recommendations
        if min_pipeline_pressure >= analysis_pressure_mpa:
            # Check if analysis pressure is within MAOP
            if analysis_pressure_mpa <= max_allowable_pressure_mpa:
                margin = min_pipeline_pressure - analysis_pressure_mpa
                st.success(f"""
                ✅ **SAFE TO OPERATE**
                - Pipeline can safely operate at {analysis_pressure_mpa:.1f} MPa ({analysis_pressure_mpa * 145.038:.0f} psi)
                - Safety margin: {margin:.1f} MPa ({margin * 145.038:.0f} psi)
                - Within MAOP limit of {max_allowable_pressure_mpa:.1f} MPa
                - Limiting method: {most_conservative_method}
                """)
            else:
                st.warning(f"""
                ⚠️ **MAOP LIMIT EXCEEDED**
                - Analysis pressure {analysis_pressure_mpa:.1f} MPa exceeds MAOP of {max_allowable_pressure_mpa:.1f} MPa
                - Maximum safe pressure from defects: {min_pipeline_pressure:.1f} MPa
                - Recommend operating at MAOP: {max_allowable_pressure_mpa:.1f} MPa
                """)
        elif min_pipeline_pressure >= max_allowable_pressure_mpa * 0.8:  # Within 80% of MAOP
            actual_limit = min(min_pipeline_pressure, max_allowable_pressure_mpa)
            st.warning(f"""
            ⚠️ **PRESSURE DERATION REQUIRED**
            - Current analysis pressure: {analysis_pressure_mpa:.1f} MPa is NOT safe
            - Recommended maximum pressure: {actual_limit:.1f} MPa ({actual_limit * 145.038:.0f} psi)
            - Constraint: {'Defect calculations' if min_pipeline_pressure < max_allowable_pressure_mpa else 'MAOP limit'}
            - Limiting method: {most_conservative_method}
            """)
        else:
            st.error(f"""
            ❌ **IMMEDIATE REPAIRS REQUIRED**
            - Pipeline cannot safely operate at reasonable pressures
            - Maximum safe pressure from defects: {min_pipeline_pressure:.1f} MPa ({min_pipeline_pressure * 145.038:.0f} psi)
            - MAOP: {max_allowable_pressure_mpa:.1f} MPa ({max_allowable_pressure_mpa * 145.038:.0f} psi)
            - Significant repairs needed before operations
            - Limiting method: {most_conservative_method}
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close pressure results container
        
        # Enhanced Dataset Preview Section
        st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Enhanced Dataset Preview</div>", unsafe_allow_html=True)
        st.markdown("The following shows the first 10 rows with the newly computed corrosion assessment and pressure analysis columns:")
        
        # Select key columns for display
        display_cols = ['log dist. [m]', 'joint number', 'depth [%]', 'length [mm]', 'width [mm]', 'wall_thickness_used_mm']
        
        # Add traditional assessment result columns
        assessment_cols = [
            'b31g_safe', 'b31g_failure_pressure_mpa', 'b31g_remaining_strength_pct',
            'modified_b31g_safe', 'modified_b31g_failure_pressure_mpa', 'modified_b31g_remaining_strength_pct',
            'simplified_eff_area_safe', 'simplified_eff_area_failure_pressure_mpa', 'simplified_eff_area_remaining_strength_pct'
        ]
        
        # Add pressure analysis columns
        pressure_cols = [
            'b31g_operational_status', 'b31g_max_safe_pressure_mpa',
            'modified_b31g_operational_status', 'modified_b31g_max_safe_pressure_mpa',
            'simplified_eff_area_operational_status', 'simplified_eff_area_max_safe_pressure_mpa'
        ]
        
        preview_cols = display_cols + assessment_cols + pressure_cols
        # Only include columns that actually exist
        available_preview_cols = [col for col in preview_cols if col in enhanced_df.columns]
        preview_df = enhanced_df[available_preview_cols].head(10)
        
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(preview_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close preview container
        
        # Export Section
        st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Export Results</div>", unsafe_allow_html=True)
        download_link = create_enhanced_csv_download_link(enhanced_df, selected_year)
        st.markdown(download_link, unsafe_allow_html=True)
        
        st.info(f"""
            **Export Information:**
            - Total columns in enhanced CSV: {len(enhanced_df.columns)}
            - Original defect data: {len(defects_df.columns)} columns
            - New assessment columns: {len(enhanced_df.columns) - len(defects_df.columns)} columns
            - Assessment methods: B31G Original, Modified B31G (0.85dL), RSTRENG (Simplified)
            - Pressure analysis: Analysis Pressure = {analysis_pressure_mpa:.1f} MPa, Min Allowable = {min_allowable_pressure_mpa:.1f} MPa
            - Wall thickness: Joint-specific values extracted from dataset
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close export container

    # Add FFS interaction analysis option (keeping existing functionality)
    st.subheader("Defect Interaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        apply_ffs_interaction = st.checkbox(
            "Apply FFS Defect Interaction Rules",
            value=True,
            help="Combine defects that are close enough to interact per ASME B31G/API 579 rules"
        )
    
    with col2:
        if apply_ffs_interaction:
            interaction_method = st.selectbox(
                "Circumferential Interaction Rule",
                options=['sqrt_dt', '3t', 'custom'],
                format_func=lambda x: {
                    'sqrt_dt': '√(D×t) - Standard',
                    '3t': '3×t - Conservative', 
                    'custom': 'Custom Distance'
                }[x]
            )
            
            if interaction_method == 'custom':
                custom_distance = st.number_input(
                    "Custom Circumferential Distance (mm)",
                    min_value=1.0,
                    max_value=100.0,
                    value=25.4
                )
            else:
                custom_distance = None
    
    if apply_ffs_interaction:
        with st.expander("ℹ️ FFS Defect Interaction Rules", expanded=False):
            st.markdown("""
            **Fitness-For-Service (FFS) Defect Interaction Criteria:**
            
            Per ASME B31G and API 579-1/ASME FFS-1, defects interact when:
            
            1. **Axial Spacing** < 1 inch (25.4 mm) - default
            2. **Circumferential Spacing** < √(D×t) or 3×t
            
            **Combined Defect Properties:**
            - Depth: Maximum of all interacting defects
            - Length: Total axial extent 
            - Width: Total circumferential extent
            
            This ensures conservative assessment of closely spaced defects.
            """)
    
    # Process with FFS rules if selected
    if apply_ffs_interaction and st.button("Analyze with FFS Interaction"):
        try:
            # Initialize FFS interaction analyzer
            ffs_analyzer = FFSDefectInteraction(
                axial_interaction_distance_mm=25.4,  # 1 inch
                circumferential_interaction_method=interaction_method,
                custom_circ_distance_mm=custom_distance
            )
            
            # Find and combine interacting defects
            original_count = len(defects_df)
            combined_defects_df = ffs_analyzer.combine_interacting_defects(
                defects_df, joints_df, pipe_diameter_mm
            )

            combined_count = len(combined_defects_df)            

            # Show summary
            st.success(f"""
            FFS Interaction Analysis Complete:
            - Original defects: {original_count}
            - After combining interactions: {combined_count}
            - Defects combined: {original_count - combined_count}
            """)

            # Show which defects were combined
            if 'is_combined' in combined_defects_df.columns:
                combined_only = combined_defects_df[combined_defects_df['is_combined'].fillna(False) == True]
            else:
                combined_only = pd.DataFrame()  # Empty dataframe if no combined defects

            if not combined_only.empty:
                st.markdown("### Combined Defects Summary")
                for idx, combined in combined_only.iterrows():
                    st.markdown(f"""
                    **Combined Defect {idx + 1}:**
                    - Number of original defects: {combined['num_original_defects']}
                    - Combined depth: {combined['depth [%]']:.1f}%
                    - Combined length: {combined['length [mm]']:.1f} mm
                    - Location: {combined['log dist. [m]']:.1f} m
                    """)
            
            # Store combined defects for assessment
            st.session_state.ffs_combined_defects = combined_defects_df
            st.session_state.use_ffs_combined = True
            
        except Exception as e:
            st.error(f"Error in FFS interaction analysis: {str(e)}")
