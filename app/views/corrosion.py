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
import io

from app.ui_components import create_metrics_row, info_box, custom_metric
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