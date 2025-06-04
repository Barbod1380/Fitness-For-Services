"""
Corrosion assessment view for the Pipeline Analysis application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import base64
from app.ui_components import create_metrics_row
from app.services.state_manager import get_state


def calculate_b31g(defect_depth_pct, defect_length_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa):
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

    Returns:
    - dict containing:
        • method: "B31G Original Level-1"
        • safe (bool): False if invalid, True otherwise
        • failure_pressure_mpa: Predicted burst pressure in MPa (0 if invalid)
        • safe_pressure_mpa: failure_pressure_mpa / safety_factor (0 if invalid)
        • remaining_strength_pct: (1 - d/t)×100 (0 if invalid)
        • folias_factor_M: Folias factor M (None if not applicable or invalid)
        • flaw_type_applied: "Short Flaw" or "Long Flaw" or "N/A"
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

    # 11. Apply a safety factor to compute "safe pressure"
    # Using SF = 1.39 for gas pipelines (common for B31G Level 1, aligns with design factors)
    # or 1.25 for liquid. Let's make it an explicit choice or parameter in a real tool.
    # For this example, we'll stick to the user's previous 1.25.
    safety_factor = 1.25 # This should ideally be an input or configurable
    safe_pressure_mpa = failure_pressure_mpa / safety_factor

    # 12. Compute remaining strength as a percentage of original (based on remaining ligament)
    remaining_strength_pct = (1.0 - d_t) * 100.0

    # 13. Return the results
    return {
        "method": "B31G Original Level-1",
        "safe": True, # If we reach here, it passed initial checks
        "failure_pressure_mpa": failure_pressure_mpa,
        "safe_pressure_mpa": safe_pressure_mpa,
        "remaining_strength_pct": remaining_strength_pct,
        "folias_factor_M": M_calculated,
        "flaw_type_applied": flaw_type_note,
        "note": f"Calculation based on {flaw_type_note} criteria. z={z:.2f}, d/t={d_t:.3f}"
    }

def calculate_modified_b31g(defect_depth_pct, defect_length_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa):
    """
    Calculate remaining strength and failure pressure using the Modified B31G
    Level-1 method (often referred to as the 0.85dL method).

    Parameters:
    - defect_depth_pct: Defect depth (%). Must be an absolute value.
    - defect_length_mm: Longitudinal extent of corrosion (L) in millimeters.
    - pipe_diameter_mm: Outside diameter of pipe (D) in millimeters.
    - wall_thickness_mm: Nominal wall thickness (t) in millimeters.
    - smys_mpa: Specified Minimum Yield Strength (SMYS) in MPa.

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
    
    # Safe Operating Pressure (P_safe)
    # Safety factor can vary (e.g., 1.39 for gas Class 1, 1.25 for liquid)
    # Using 1.39 as a common example for Modified B31G application.
    safety_factor = 1.39 # This should ideally be an input or configurable based on code/operator requirements.
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

def calculate_rstreng(defect_depth_pct, defect_length_mm, defect_width_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa):
    """
    Calculate remaining strength using a simplified RSTRENG method.
    This is a simplified implementation as the actual method requires detailed river-bottom profile.
    
    Parameters:
    - defect_depth_pct: Defect depth as percentage of wall thickness
    - defect_length_mm: Defect length (mm)
    - defect_width_mm: Defect width (mm) - used as an additional parameter for simplified RSTRENG
    - pipe_diameter_mm: Outside diameter of pipe (mm)
    - wall_thickness_mm: Nominal wall thickness (mm)
    - smys_mpa: Specified Minimum Yield Strength (MPa)
    
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
    safety_factor = 1.5  # Common for RSTRENG
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



def compute_corrosion_metrics_for_dataframe(defects_df, joints_df, pipe_diameter_mm, smys_mpa):
    """
    Compute B31G, Modified B31G, and RSTRENG metrics for all defects in the dataframe.
    Wall thickness is extracted from joints_df based on each defect's joint number.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - joints_df: DataFrame containing joint information with wall thickness
    - pipe_diameter_mm: Outside diameter of pipe in millimeters
    - smys_mpa: Specified Minimum Yield Strength in MPa
    
    Returns:
    - DataFrame with additional columns for corrosion assessment metrics
    """
    # Create a copy to avoid modifying the original dataframe
    enhanced_df = defects_df.copy()
    
    # Create wall thickness lookup dictionary from joints_df
    wt_lookup = {}
    if 'joint number' in joints_df.columns and 'wt nom [mm]' in joints_df.columns:
        wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    else:
        raise ValueError("joints_df must contain 'joint number' and 'wt nom [mm]' columns")
    
    # Initialize new columns for each method
    methods = ['b31g', 'modified_b31g', 'rstreng']
    
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
    enhanced_df['rstreng_folias_factor'] = np.nan
    enhanced_df['rstreng_effective_depth_factor'] = np.nan
    
    # Add a column to store the wall thickness used for each defect
    enhanced_df['wall_thickness_used_mm'] = np.nan
    
    # Process each defect
    for idx, row in defects_df.iterrows():
        # Check if required columns exist and have valid data
        required_cols = ['depth [%]', 'length [mm]', 'width [mm]', 'joint number']
        missing_data = []
        
        for col in required_cols:
            if pd.isna(row.get(col)):
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
        joint_number = row['joint number']
        
        # Get wall thickness for this defect's joint
        wall_thickness_mm = wt_lookup.get(joint_number)
        if wall_thickness_mm is None or pd.isna(wall_thickness_mm):
            for method in methods:
                enhanced_df.loc[idx, f'{method}_safe'] = False
                enhanced_df.loc[idx, f'{method}_notes'] = f"Wall thickness not found for joint {joint_number}"
            continue
        
        # Store the wall thickness used
        enhanced_df.loc[idx, 'wall_thickness_used_mm'] = wall_thickness_mm
        
        # Skip if any dimension is zero or negative
        if depth_pct <= 0 or length_mm <= 0 or width_mm <= 0 or wall_thickness_mm <= 0:
            for method in methods:
                enhanced_df.loc[idx, f'{method}_safe'] = False
                enhanced_df.loc[idx, f'{method}_notes'] = "Invalid dimensional data (zero or negative values)"
            continue
        
        # Calculate B31G using existing function
        try:
            b31g_result = calculate_b31g(depth_pct, length_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa)
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
        
        # Calculate Modified B31G using existing function
        try:
            mod_b31g_result = calculate_modified_b31g(depth_pct, length_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa)
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
        
        # Calculate RSTRENG using existing function
        try:
            rstreng_result = calculate_rstreng(depth_pct, length_mm, width_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa)
            enhanced_df.loc[idx, 'rstreng_safe'] = rstreng_result['safe']
            enhanced_df.loc[idx, 'rstreng_failure_pressure_mpa'] = rstreng_result['failure_pressure_mpa']
            enhanced_df.loc[idx, 'rstreng_safe_pressure_mpa'] = rstreng_result['safe_pressure_mpa']
            enhanced_df.loc[idx, 'rstreng_remaining_strength_pct'] = rstreng_result['remaining_strength_pct']
            enhanced_df.loc[idx, 'rstreng_folias_factor'] = rstreng_result.get('folias_factor_M')
            enhanced_df.loc[idx, 'rstreng_effective_depth_factor'] = rstreng_result.get('effective_depth_factor')
            enhanced_df.loc[idx, 'rstreng_notes'] = rstreng_result['note']
        except Exception as e:
            enhanced_df.loc[idx, 'rstreng_safe'] = False
            enhanced_df.loc[idx, 'rstreng_notes'] = f"Calculation error: {str(e)}"
    
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
    methods = ['b31g', 'modified_b31g', 'rstreng']
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


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    method_name = {'b31g': 'B31G Original', 'modified_b31g': 'Modified B31G', 'rstreng': 'RSTRENG'}[method]
    
    if 'pressure' in metric:
        colorscale = 'Blues'
        if 'failure' in metric:
            metric_label = 'Failure Pressure'
            unit = 'MPa'
        else:
            metric_label = 'Safe Operating Pressure'
            unit = 'MPa'
    else:  # remaining strength
        colorscale = 'Greens'
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
            ('rstreng', calculate_rstreng)
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

def aggregate_assessment_results_by_joint(enhanced_df, joints_df, pipe_diameter_mm=None, smys_mpa=None):
    """
    Enhanced aggregation that includes wall thickness data for visualization.
    
    Parameters:
    - enhanced_df: DataFrame with computed corrosion metrics
    - joints_df: DataFrame with joint information
    - pipe_diameter_mm: Pipe diameter in mm (needed for intact pipe calculations)
    - smys_mpa: SMYS in MPa (needed for intact pipe calculations)
    
    Returns:
    - DataFrame with joint-level aggregated assessment results including wall thickness
    """
    # Focus only on the 3 core metrics
    methods = ['b31g', 'modified_b31g', 'rstreng']
    metrics = ['failure_pressure_mpa', 'safe_pressure_mpa', 'remaining_strength_pct']
    
    # Start with joints dataframe (include wall thickness)
    if 'wt nom [mm]' in joints_df.columns:
        joint_summary = joints_df[['joint number', 'log dist. [m]', 'joint length [m]', 'wt nom [mm]']].copy()
    else:
        joint_summary = joints_df[['joint number', 'log dist. [m]', 'joint length [m]']].copy()
        joint_summary['wt nom [mm]'] = 10.0  # Default fallback
    
    # Calculate intact pipe values if parameters provided
    intact_values = None
    if pipe_diameter_mm is not None and smys_mpa is not None:
        avg_wall_thickness = joint_summary['wt nom [mm]'].mean()
        intact_values = calculate_intact_pipe_values(pipe_diameter_mm, avg_wall_thickness, smys_mpa)
    
    # For each method and metric, calculate joint-level statistics
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
        
        # Process joints that have defects
        for metric in metrics:
            col_name = f'{method}_{metric}'
            if col_name in enhanced_df.columns:
                # Group by joint and calculate statistics for joints WITH defects
                joint_stats = enhanced_df.groupby('joint number')[col_name].agg([
                    'count', 'min', 'max', 'mean'
                ]).reset_index()
                
                # Update joints with defects
                for _, row in joint_stats.iterrows():
                    joint_num = row['joint number']
                    mask = joint_summary['joint number'] == joint_num
                    
                    joint_summary.loc[mask, f'{method}_{metric}_defect_count'] = row['count']
                    joint_summary.loc[mask, f'{method}_{metric}_min'] = row['min']
                    joint_summary.loc[mask, f'{method}_{metric}_max'] = row['max']
                    joint_summary.loc[mask, f'{method}_{metric}_avg'] = row['mean']
        
        # Calculate safety statistics for enhanced hover info
        safe_col = f'{method}_safe'
        if safe_col in enhanced_df.columns:
            safety_stats = enhanced_df.groupby('joint number')[safe_col].agg([
                'count', 'sum'
            ]).reset_index()
            
            # Update joints that have defects
            for _, row in safety_stats.iterrows():
                joint_num = row['joint number']
                mask = joint_summary['joint number'] == joint_num
                
                joint_summary.loc[mask, f'{method}_total_defects'] = row['count']
                joint_summary.loc[mask, f'{method}_safe_defects'] = row['sum']
    
    # Handle joints without defects using realistic calculated values
    for method in methods:
        # Find joints with no defects
        no_defect_mask = joint_summary[f'{method}_total_defects'] == 0
        
        if intact_values and method in intact_values:
            # Use calculated intact pipe values
            method_intact = intact_values[method]
            
            for metric in metrics:
                if metric in method_intact:
                    joint_summary.loc[no_defect_mask, f'{method}_{metric}_min'] = method_intact[metric]
                    joint_summary.loc[no_defect_mask, f'{method}_{metric}_max'] = method_intact[metric]
                    joint_summary.loc[no_defect_mask, f'{method}_{metric}_avg'] = method_intact[metric]
        else:
            # Fallback: use joint-specific wall thickness with Barlow's formula
            for idx, row in joint_summary[no_defect_mask].iterrows():
                wt = row['wt nom [mm]']
                if pipe_diameter_mm and smys_mpa and not pd.isna(wt):
                    # Calculate using Barlow's formula for this specific joint
                    theoretical_pressure = (2.0 * smys_mpa * wt) / pipe_diameter_mm
                    safe_pressure = theoretical_pressure / 1.39
                    
                    joint_summary.loc[idx, f'{method}_failure_pressure_mpa_min'] = theoretical_pressure
                    joint_summary.loc[idx, f'{method}_failure_pressure_mpa_max'] = theoretical_pressure
                    joint_summary.loc[idx, f'{method}_failure_pressure_mpa_avg'] = theoretical_pressure
                    
                    joint_summary.loc[idx, f'{method}_safe_pressure_mpa_min'] = safe_pressure
                    joint_summary.loc[idx, f'{method}_safe_pressure_mpa_max'] = safe_pressure
                    joint_summary.loc[idx, f'{method}_safe_pressure_mpa_avg'] = safe_pressure
                    
                    joint_summary.loc[idx, f'{method}_remaining_strength_pct_min'] = 100.0
                    joint_summary.loc[idx, f'{method}_remaining_strength_pct_max'] = 100.0
                    joint_summary.loc[idx, f'{method}_remaining_strength_pct_avg'] = 100.0
    
    return joint_summary


def render_corrosion_assessment_view():
    """Display the corrosion assessment view with B31G, Modified B31G, and RSTRENG calculations plus visualizations."""
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
    pipe_diameter_stored = datasets[selected_year].get('pipe_diameter', 1.0)  # Default from storage
    
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
    if "error" not in wt_stats:
        st.markdown("**Wall Thickness Distribution:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Joints", wt_stats['count'])
        with col2:
            st.metric("Min WT", f"{wt_stats['min']:.1f} mm")
        with col3:
            st.metric("Max WT", f"{wt_stats['max']:.1f} mm")
        with col4:
            st.metric("Avg WT", f"{wt_stats['mean']:.1f} mm")
        
        if len(wt_stats['unique_values']) <= 10:
            st.info(f"**Unique wall thickness values:** {', '.join([f'{v:.1f}mm' for v in wt_stats['unique_values']])}")
        else:
            st.info(f"**Wall thickness range:** {len(wt_stats['unique_values'])} different values from {wt_stats['min']:.1f}mm to {wt_stats['max']:.1f}mm")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close selection container
    
    # Parameters input section
    st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Pipeline Parameters</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pipe_diameter_m = st.number_input(
            "Pipe Diameter (m)",
            min_value=0.1,
            max_value=3.0,
            value=pipe_diameter_stored,
            step=0.1,
            format="%.2f",
            key="pipe_diameter_corrosion",
            help="Outer diameter of the pipeline"
        )
    
    with col2:
        smys_mpa = st.number_input(
            "SMYS (MPa)",
            min_value=200.0,
            max_value=800.0,
            value=358.0,  # Common value for API 5L X52
            step=1.0,
            format="%.0f",
            key="smys_corrosion",
            help="Specified Minimum Yield Strength"
        )
    
    st.info("ℹ️ **Wall thickness is automatically extracted from the joints data for each defect based on its joint number.**")
    
    # Convert diameter to mm for calculations
    pipe_diameter_mm = pipe_diameter_m * 1000
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close parameters container
    
    # Assessment button and results
    st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
    
    # FIX 2: Always show the assessment button, but also show results if they exist
    if st.button("Perform Corrosion Assessment", key="perform_assessment", use_container_width=True):
        with st.spinner("Computing B31G, Modified B31G, and RSTRENG metrics using joint-specific wall thickness..."):
            try:
                # Compute the enhanced dataframe with corrosion metrics
                enhanced_df = compute_corrosion_metrics_for_dataframe(
                    defects_df, joints_df, pipe_diameter_mm, smys_mpa
                )
                
                # Aggregate results by joint for visualization
                joint_summary = aggregate_assessment_results_by_joint(
                    enhanced_df, joints_df, pipe_diameter_mm, smys_mpa
                )
                
                # Store results in session state for download and visualization
                st.session_state.enhanced_defects_df = enhanced_df
                st.session_state.joint_summary_df = joint_summary
                st.session_state.assessment_year = selected_year
                st.session_state.assessment_pipe_diameter = pipe_diameter_m
                
                st.success("✅ Corrosion assessment completed successfully!")
                st.rerun()  # Refresh to show results immediately
                
            except Exception as e:
                st.error(f"Error during corrosion assessment: {str(e)}")
                st.info("Please check your input parameters and ensure the selected dataset has valid defect dimensions and joint data.")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close assessment button container
    
    # FIX 2: Always show results if they exist (persistent UI)
    if (hasattr(st.session_state, 'enhanced_defects_df') and 
        hasattr(st.session_state, 'joint_summary_df') and 
        st.session_state.assessment_year == selected_year):
        
        enhanced_df = st.session_state.enhanced_defects_df
        joint_summary = st.session_state.joint_summary_df
        
        # Assessment Results Section
        st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Assessment Results</div>", unsafe_allow_html=True)
        
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
                ("Avg Failure Pressure", f"{b31g_stats['avg_failure_pressure_mpa']:.1f} MPa", None),
                ("Min Remaining Strength", f"{b31g_stats['min_remaining_strength_pct']:.1f}%", None)
            ]
            create_metrics_row(metrics_data)
        
        with method_tabs[1]:
            st.markdown("### Modified B31G (0.85dL) Results")
            mod_stats = summary_stats['modified_b31g']
            
            metrics_data = [
                ("Total Defects", f"{mod_stats['total_defects']}", None),
                ("Safe Defects", f"{mod_stats['safe_defects']}", f"{mod_stats['safe_percentage']:.1f}%"),
                ("Avg Failure Pressure", f"{mod_stats['avg_failure_pressure_mpa']:.1f} MPa", None),
                ("Min Remaining Strength", f"{mod_stats['min_remaining_strength_pct']:.1f}%", None)
            ]
            create_metrics_row(metrics_data)
        
        with method_tabs[2]:
            st.markdown("### RSTRENG (Simplified) Results")
            rstreng_stats = summary_stats['rstreng']
            
            metrics_data = [
                ("Total Defects", f"{rstreng_stats['total_defects']}", None),
                ("Safe Defects", f"{rstreng_stats['safe_defects']}", f"{rstreng_stats['safe_percentage']:.1f}%"),
                ("Avg Failure Pressure", f"{rstreng_stats['avg_failure_pressure_mpa']:.1f} MPa", None),
                ("Min Remaining Strength", f"{rstreng_stats['min_remaining_strength_pct']:.1f}%", None)
            ]
            create_metrics_row(metrics_data)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close results container
        
        # Enhanced Dataset Preview Section (always visible)
        st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Enhanced Dataset Preview</div>", unsafe_allow_html=True)
        st.markdown("The following shows the first 10 rows with the newly computed corrosion assessment columns:")
        
        # Select key columns for display
        display_cols = ['log dist. [m]', 'joint number', 'depth [%]', 'length [mm]', 'width [mm]', 'wall_thickness_used_mm']
        
        # Add assessment result columns
        assessment_cols = [
            'b31g_safe', 'b31g_failure_pressure_mpa', 'b31g_remaining_strength_pct',
            'modified_b31g_safe', 'modified_b31g_failure_pressure_mpa', 'modified_b31g_remaining_strength_pct',
            'rstreng_safe', 'rstreng_failure_pressure_mpa', 'rstreng_remaining_strength_pct'
        ]
        
        preview_cols = display_cols + assessment_cols
        preview_df = enhanced_df[preview_cols].head(10)
        
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(preview_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close preview container
        
        # Export Section (always visible)
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
            - Wall thickness: Joint-specific values extracted from dataset
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close export container
        
        # Visualization Section (separate and additive)
        st.markdown('<div class="card-container" style="margin-top:30px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎨 Pipeline Assessment Visualization</div>", unsafe_allow_html=True)
        
        # Visualization controls
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            viz_method = st.selectbox(
                "Assessment Method",
                options=['b31g', 'modified_b31g', 'rstreng'],
                format_func=lambda x: {'b31g': 'B31G Original', 'modified_b31g': 'Modified B31G', 'rstreng': 'RSTRENG'}[x],
                key="viz_method"
            )
        
        with viz_col2:
            viz_metric = st.selectbox(
                "Metric to Visualize",
                options=['failure_pressure_mpa', 'safe_pressure_mpa', 'remaining_strength_pct'],
                format_func=lambda x: {
                    'failure_pressure_mpa': 'Failure Pressure',
                    'safe_pressure_mpa': 'Safe Operating Pressure',
                    'remaining_strength_pct': 'Remaining Strength'
                }[x],
                key="viz_metric"
            )
        
        with viz_col3:
            viz_aggregation = st.selectbox(
                "Aggregation Method",
                options=['min', 'max', 'avg'],
                format_func=lambda x: {'min': 'Minimum (Most Critical)', 'max': 'Maximum (Best Case)', 'avg': 'Average'}[x],
                key="viz_aggregation",
                help="How to aggregate multiple defects per joint"
            )
        
        # Generate visualization button
        if st.button("Generate Pipeline Visualization", key="generate_viz", use_container_width=True):
            with st.spinner("Creating pipeline assessment visualization..."):
                try:
                    pipe_diameter = st.session_state.assessment_pipe_diameter
                    
                    # Create the pipeline visualization
                    viz_fig = create_joint_assessment_visualization(
                        joint_summary, 
                        method=viz_method, 
                        metric=viz_metric, 
                        aggregation=viz_aggregation,
                        pipe_diameter=pipe_diameter
                    )
                    
                    # Store visualization in session state for persistence
                    st.session_state.pipeline_viz_fig = viz_fig
                    st.session_state.viz_settings = {
                        'method': viz_method,
                        'metric': viz_metric, 
                        'aggregation': viz_aggregation
                    }
                    
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
        
        # Display pipeline visualization if it exists (persistent)
        if hasattr(st.session_state, 'pipeline_viz_fig'):
            st.markdown("<div class='section-header' style='margin-top:20px;'>📊 Pipeline Visualization Results</div>", unsafe_allow_html=True)
            
            # Show current settings
            viz_settings = st.session_state.viz_settings
            method_name = {'b31g': 'B31G Original', 'modified_b31g': 'Modified B31G', 'rstreng': 'RSTRENG'}[viz_settings['method']]
            metric_name = {
                'safe_percentage': 'Safety Percentage',
                'failure_pressure_mpa': 'Failure Pressure',
                'safe_pressure_mpa': 'Safe Operating Pressure',
                'remaining_strength_pct': 'Remaining Strength'
            }[viz_settings['metric']]
            
            if viz_settings['metric'] != 'safe_percentage':
                st.caption(f"Showing: {method_name} - {metric_name} ({viz_settings['aggregation'].title()})")
            else:
                st.caption(f"Showing: {method_name} - {metric_name}")
            
            # Display the stored visualization
            st.plotly_chart(st.session_state.pipeline_viz_fig, use_container_width=True, config={'displayModeBar': True})
            
            # Add interpretation guide
            if 'pressure' in viz_settings['metric']:
                pressure_type = "Failure" if 'failure' in viz_settings['metric'] else "Safe Operating"
                st.info(f"""
                **Interpretation Guide - {pressure_type} Pressure:**
                - 🔵 **Darker blue joints**: Higher pressure capacity (better condition)
                - 🔷 **Lighter blue joints**: Lower pressure capacity (more critical condition)  
                - Higher values indicate the joint can withstand more pressure before failure
                - Hover over joints to see exact pressure values and defect counts
                """)
            else:  # remaining strength
                st.info("""
                **Interpretation Guide - Remaining Strength:**
                - 🟢 **Darker green joints**: Higher remaining strength (better condition)
                - 🟢 **Lighter green joints**: Lower remaining strength (more degraded condition)
                - Values close to 100% indicate minimal structural degradation
                - Hover over joints to see exact strength percentages and defect counts
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close visualization container
    
    # Show message if no assessment has been performed yet
    elif not hasattr(st.session_state, 'enhanced_defects_df'):
        st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
        st.info("👆 Click **'Perform Corrosion Assessment'** above to begin analysis and unlock interactive visualization features.")
        st.markdown('</div>', unsafe_allow_html=True)