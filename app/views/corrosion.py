"""
Corrosion assessment view for the Pipeline Analysis application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import base64
from app.ui_components import create_metrics_row
from app.services.state_manager import get_state
from core.ffs_defect_interaction import *
from visualization.defect_assessment_viz import create_defect_assessment_scatter_plot, create_defect_assessment_summary_table, create_rstreng_envelope_plot
from visualization.pressure_assessment_viz import create_pressure_assessment_visualization
from typing import Optional, Sequence


def calculate_b31g(
    defect_depth_pct: float,
    defect_length_mm: float,
    pipe_diameter_mm: float,
    wall_thickness_mm: float,
    maop_mpa: float,
    smys_mpa: float,
    safety_factor: float = 1.39,
    smts_mpa: float | None = None,
    area_shape: str = "parabolic",        # “parabolic” is the official B31G shape
):
    """
    Original ASME B31G Level‑1 (2012) – corrected implementation.
    Follows the two‑segment Folias factor and the full flow‑stress rule set.
    """
    method = "B31G Original Level‑1"

    # ── 1  basic checks ─────────────────────────────────────────
    if defect_depth_pct <= 0:
        return dict(method=method, safe=False,
                    note="Defect depth must be positive")

    d_over_t = defect_depth_pct / 100.0
    if d_over_t > 0.80:
        return dict(method=method, safe=False,
                    note="d/t > 0.80 – Level‑1 not applicable")

    if pipe_diameter_mm <= 0 or wall_thickness_mm <= 0:
        return dict(method=method, safe=False,
                    note="Diameter and wall thickness must be positive")

    # ── 2  z‑parameter & Folias factor ─────────────────────────
    z = (defect_length_mm ** 2) / (pipe_diameter_mm * wall_thickness_mm)
    if z > 50.0:
        return dict(method=method, safe=False,
                    note=f"z = {z:.2f} > 50 – Level‑1 table required")

    if z <= 20.0:
        M = math.sqrt(1.0 + 0.8 * z)             # B31G eqn 1‑2
    else:                                        # 20 < z ≤ 50
        M = math.sqrt(1.6 + 0.16 * z)            # B31G eqn 1‑3

    # ── 3  flow stress (take the minimum of the three forms) ──
    s1 = 1.1 * smys_mpa
    s2 = smys_mpa + 68.95
    s3 = (smys_mpa + (smts_mpa or smys_mpa)) / 2
    S_flow = min(s1, s2, s3)
    if smts_mpa is not None:
        S_flow = min(S_flow, 0.9 * smts_mpa)

    # ── 4  corroded‑area ratio (B31G parabolic) ───────────────
    if area_shape.lower() != "parabolic":
        area_shape_used = "rectangular (non‑standard)"
        A_ratio = d_over_t                      # allow for studies
    else:
        area_shape_used = "parabolic"
        A_ratio = (2.0 / 3.0) * d_over_t       # B31G default

    # ── 5  failure stress & pressure ───────────────────────────
    denom = 1.0 - (A_ratio / M)
    if denom <= 0.0:
        return dict(method=method, safe=False,
                    note="Denominator ≤ 0 – check inputs")

    Sf = S_flow * (1.0 - A_ratio) / denom
    Pf = 2.0 * Sf * wall_thickness_mm / pipe_diameter_mm
    P_safe = Pf / safety_factor
    RSF_pct = 100.0 * (Sf / S_flow)

    # ── 6  results ─────────────────────────────────────────────
    return dict(
        method=method,
        safe=P_safe >= maop_mpa,
        failure_pressure_mpa=Pf,
        safe_pressure_mpa=P_safe,
        remaining_strength_pct=RSF_pct,
        folias_factor_M=M,
        z_parameter=z,
        depth_over_thickness=d_over_t,
        safety_factor_used=safety_factor,
        note=(f"area={area_shape_used}, z={z:.2f}, d/t={d_over_t:.3f}, "
              f"S_flow={S_flow:.2f} MPa, M={M:.3f}")
    )


def calculate_modified_b31g(
    defect_depth_pct: float,
    defect_length_mm: float,
    pipe_diameter_mm: float,
    wall_thickness_mm: float,
    maop_mpa: float,
    smys_mpa: float,
    safety_factor: float = 1.39,
    smts_mpa: float | None = None,  # optional – cap S_flow at 0.9 SMTS
):
    """
    ASME Modified B31G (0.85 dL) — Level‑1 corrosion assessment.
    Implements z‑limit (≤50), d/t‑limit (≤0.80), and 0.85 dL area rule.
    """
    method = "Modified B31G (0.85 dL)"

    # 1 ─ basic depth & geometry checks
    if defect_depth_pct <= 0.0:
        return dict(method=method, safe=False,
                    note="Defect depth must be positive")
    d_over_t = defect_depth_pct / 100.0
    if d_over_t > 0.80:
        return dict(method=method, safe=False,
                    note=f"d/t = {d_over_t:.3f} > 0.80 — outside Level‑1 scope")

    if pipe_diameter_mm <= 0 or wall_thickness_mm <= 0:
        return dict(method=method, safe=False,
                    note="Diameter and wall thickness must be positive")

    # 2 ─ dimensionless length & Folias factor
    z = (defect_length_mm ** 2) / (pipe_diameter_mm * wall_thickness_mm)
    if z > 50:
        return dict(method=method, safe=False,
                    note=f"z = {z:.2f} > 50 — use Level‑2")

    M = math.sqrt(1.0 + 0.6275 * z - 0.003375 * z ** 2)

    # 3 ─ flow stress
    S_flow = smys_mpa + 68.95            # +10 ksi
    if smts_mpa is not None:
        S_flow = min(S_flow, 0.9 * smts_mpa)

    # 4 ─ area ratio (0.85 dL rectangle)
    A_ratio = 0.85 * d_over_t
    denom = 1.0 - (A_ratio / M)
    if denom <= 0.0:
        return dict(method=method, safe=False,
                    note="Denominator ≤ 0 — check inputs (deep, long defect)")

    # 5 ─ stresses & pressures
    Sf = S_flow * (1.0 - A_ratio) / denom
    Pf = 2.0 * Sf * wall_thickness_mm / pipe_diameter_mm
    P_safe = Pf / safety_factor
    RSF_pct = 100.0 * Sf / S_flow

    return dict(
        method=method,
        safe=P_safe >= maop_mpa,
        failure_pressure_mpa=Pf,
        safe_pressure_mpa=P_safe,
        remaining_strength_pct=RSF_pct,
        folias_factor_M=M,
        z_parameter=z,
        depth_over_thickness=d_over_t,
        safety_factor_used=safety_factor,
        note=f"OK — z={z:.2f}, d/t={d_over_t:.3f}, M={M:.3f}"
    )


def calculate_rstreng_effective_area_single(
    defect_depth_pct: float,
    defect_length_mm: float,
    defect_width_mm: Optional[float],      
    pipe_diameter_mm: float,
    wall_thickness_mm: float,
    maop_mpa: float,
    smys_mpa: float,
    safety_factor: float = 1.39,
    smts_mpa: Optional[float] = None,      
) -> dict:
    """
    RSTRENG Level-1 (single defect, parabolic profile) — True implementation.

    Parameters:
        defect_depth_pct : float
            Maximum depth of defect as percent (%) of wall thickness (e.g. 40 for 40%)
        defect_length_mm : float
            Axial length of defect (mm)
        defect_width_mm  : float or None
            Unused in Level-1 RSTRENG for single defect, included for compatibility
        pipe_diameter_mm : float
            Outside diameter of pipe (mm)
        wall_thickness_mm: float
            Nominal wall thickness (mm)
        maop_mpa         : float
            Maximum allowable operating pressure (MPa)
        smys_mpa         : float
            Specified minimum yield strength (MPa)
        safety_factor    : float
            Safety factor (default 1.39)
        smts_mpa         : float or None
            Specified minimum tensile strength (MPa, optional for flow stress cap)

    Returns:
        dict: {
            'method': str,
            'safe': bool,
            'failure_pressure_mpa': float,
            'safe_pressure_mpa': float,
            'remaining_strength_pct': float,
            'folias_factor_Mt': float,
            'flow_stress_mpa': float,
            'psi_parameter': float,
            'note': str
        }
    """

    method = "RSTRENG Level-1 (single defect, parabolic profile)"

    # 1. Check input validity
    if pipe_diameter_mm <= 0 or wall_thickness_mm <= 0:
        return dict(method=method, safe=False, note="Diameter and wall thickness must be positive.")

    d = defect_depth_pct / 100.0
    if not (0.0 < d < 1.0):
        return dict(method=method, safe=False, note="Defect depth % (d/t) must be between 0 and 100.")

    if defect_length_mm <= 0:
        return dict(method=method, safe=False, note="Defect length must be positive.")

    if d >= 1.0:
        return dict(method=method, safe=False, failure_pressure_mpa=0.0, note="Through-wall defect.")

    # 2. Geometry & Folias factor
    D = pipe_diameter_mm
    t = wall_thickness_mm
    L = defect_length_mm

    L2DT = (L ** 2) / (D * t)
    if L2DT <= 50.0:
        M = math.sqrt(1.0 + 0.6275 * L2DT - 0.003375 * (L2DT ** 2))
    else:
        M = 0.032 * L2DT + 3.3
    psi = L / math.sqrt((D / 2.0) * t)

    # 3. Flow stress (capped if SMTS provided)
    S_flow = smys_mpa + 68.95
    if smts_mpa is not None:
        S_flow = min(S_flow, 0.9 * smts_mpa)

    # 4. RSTRENG shape factor for single defect
    alpha = 0.85

    numer = 1.0 - (alpha * d)
    denom = 1.0 - (alpha * d / M)
    if denom <= 0.0:
        return dict(method=method, safe=False, note="Denominator ≤ 0 — unphysical geometry/combination.")

    Sf = S_flow * (numer / denom)   # Failure stress (MPa)

    Pf = 2.0 * Sf * t / D           # Failure pressure (MPa)
    P_safe = Pf / safety_factor     # Allowable/safe pressure (MPa)
    RSF_pct = (Sf / S_flow) * 100.0

    return dict(
        method=method,
        safe=P_safe >= maop_mpa,
        failure_pressure_mpa=Pf,
        safe_pressure_mpa=P_safe,
        remaining_strength_pct=RSF_pct,
        folias_factor_Mt=M,
        flow_stress_mpa=S_flow,
        psi_parameter=psi,
        note=f"L2/DT={L2DT:.2f}, psi={psi:.2f}, Mt={M:.3f}, α={alpha}, d/t={d:.4f}"
    )


def calculate_rstreng_effective_area_cluster(
    depth_profile_mm: Sequence[float],
    axial_step_mm: float,
    pipe_diameter_mm: float,
    wall_thickness_mm: float,
    maop_mpa: float,
    smys_mpa: float,
    safety_factor: float = 1.39,
    smts_mpa: Optional[float] = None,
) -> dict:
    """
    RSTRENG (Effective Area) calculation for a corrosion cluster (using river-bottom profile).

    Parameters:
        depth_profile_mm: Sequence of defect depths at each axial position (mm)
        axial_step_mm: Axial interval (mm)
        pipe_diameter_mm: Pipe OD (mm)
        wall_thickness_mm: Wall thickness (mm)
        maop_mpa: Maximum allowable operating pressure (MPa)
        smys_mpa: Specified minimum yield strength (MPa)
        safety_factor: Safety factor for pressure (default: 1.39)
        smts_mpa: (Optional) Specified minimum tensile strength for flow stress cap

    Returns:
        dict with calculation results and metadata.
    """
    method = "RSTRENG (Cluster, river-bottom profile)"

    D = pipe_diameter_mm
    t = wall_thickness_mm
    L_total = len(depth_profile_mm) * axial_step_mm

    if D <= 0 or t <= 0 or L_total <= 0 or np.any(np.array(depth_profile_mm) < 0):
        return dict(method=method, safe=False, note="Invalid geometry or negative depth.")

    # Metal loss area (sq mm)
    area_lost = np.trapz(depth_profile_mm, dx=axial_step_mm)

    # Reference area (sq mm)
    area_ref = L_total * t

    # Area ratio (RSF)
    RSF = area_lost / area_ref

    if RSF >= 1.0:
        return dict(method=method, safe=False, failure_pressure_mpa=0.0, note="Full wall loss (RSF ≥ 1.0)")

    # Folias factor (using total cluster length)
    L2DT = (L_total ** 2) / (D * t)
    if L2DT <= 50.0:
        M = math.sqrt(1.0 + 0.6275 * L2DT - 0.003375 * (L2DT ** 2))
    else:
        M = 0.032 * L2DT + 3.3

    # Flow stress (MPa)
    S_flow = smys_mpa + 68.95
    if smts_mpa is not None:
        S_flow = min(S_flow, 0.9 * smts_mpa)

    numer = 1.0 - RSF
    denom = 1.0 - (RSF / M)
    if denom <= 0.0:
        return dict(method=method, safe=False, note="Denominator ≤ 0 — unphysical cluster geometry.")

    Sf = S_flow * (numer / denom)
    Pf = 2.0 * Sf * t / D           # Failure pressure (MPa)
    P_safe = Pf / safety_factor     # Safe/allowable pressure (MPa)
    RSF_pct = (Sf / S_flow) * 100.0

    return dict(
        method=method,
        safe=P_safe >= maop_mpa,
        failure_pressure_mpa=Pf,
        safe_pressure_mpa=P_safe,
        remaining_strength_pct=RSF_pct,
        folias_factor_Mt=M,
        flow_stress_mpa=S_flow,
        area_ratio_RSF=RSF,
        L2DT=L2DT,
        note=f"L2/DT={L2DT:.2f}, M={M:.3f}, RSF={RSF:.4f}, len={L_total:.2f}mm"
    )



def compute_enhanced_corrosion_metrics(defects_df, joints_df, pipe_diameter_mm, smys_mpa, safety_factor, analysis_pressure_mpa, max_allowable_pressure_mpa):
    """
    Enhanced version of compute_corrosion_metrics_for_dataframe with pressure-based assessment and ERF.
    """
    
    # First run the standard corrosion assessment
    enhanced_df = compute_corrosion_metrics_for_dataframe(defects_df, joints_df, pipe_diameter_mm, smys_mpa, safety_factor, max_allowable_pressure_mpa)
    # Add pressure analysis parameters
    enhanced_df['analysis_pressure_mpa'] = analysis_pressure_mpa
    enhanced_df['max_allowable_pressure_mpa'] = max_allowable_pressure_mpa
    
    # Initialize pressure-based assessment columns
    methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
    
    for method in methods:
        enhanced_df[f'{method}_pressure_status'] = 'UNKNOWN'
        enhanced_df[f'{method}_operational_status'] = 'UNKNOWN'
        enhanced_df[f'{method}_can_operate_at_analysis_pressure'] = False
        enhanced_df[f'{method}_can_operate_at_max_allowable_pressure'] = False
        enhanced_df[f'{method}_recommended_action'] = ''
        enhanced_df[f'{method}_max_safe_operating_pressure_mpa'] = 0.0
        enhanced_df[f'{method}_pressure_margin_pct'] = 0.0
        enhanced_df[f'{method}_erf'] = 0.0  # Add ERF column
    
    # Perform pressure-based assessment for each defect and method
    for idx, row in enhanced_df.iterrows():
        for method in methods:
            failure_pressure = row[f'{method}_failure_pressure_mpa']
            safe_pressure = row[f'{method}_safe_pressure_mpa']
            is_safe = row[f'{method}_safe']
            
            if is_safe and failure_pressure > 0 and safe_pressure > 0:
                pressure_assessment = classify_defect_pressure_status(
                    failure_pressure, safe_pressure, 
                    analysis_pressure_mpa, max_allowable_pressure_mpa
                )
                
                # Calculate ERF = MAOP  / Safe Working Pressure 
                erf_value = max_allowable_pressure_mpa / safe_pressure if max_allowable_pressure_mpa > 0 else float('inf')
                
                # Direct assignment of all assessment results
                enhanced_df.loc[idx, f'{method}_pressure_status'] = pressure_assessment['pressure_status']
                enhanced_df.loc[idx, f'{method}_operational_status'] = pressure_assessment['operational_status']
                enhanced_df.loc[idx, f'{method}_can_operate_at_analysis_pressure'] = pressure_assessment['can_operate_at_analysis_pressure']
                enhanced_df.loc[idx, f'{method}_can_operate_at_max_allowable_pressure'] = pressure_assessment['can_operate_at_max_allowable_pressure']
                enhanced_df.loc[idx, f'{method}_recommended_action'] = pressure_assessment['recommended_action']
                enhanced_df.loc[idx, f'{method}_max_safe_operating_pressure_mpa'] = pressure_assessment['max_safe_operating_pressure_mpa']
                enhanced_df.loc[idx, f'{method}_pressure_margin_pct'] = pressure_assessment['pressure_margin_pct']
                enhanced_df.loc[idx, f'{method}_erf'] = erf_value  # Add ERF value
            else:
                # Handle failed calculations
                enhanced_df.loc[idx, f'{method}_pressure_status'] = 'CALCULATION_ERROR'
                enhanced_df.loc[idx, f'{method}_operational_status'] = 'UNKNOWN'
                enhanced_df.loc[idx, f'{method}_recommended_action'] = 'Review defect parameters'
                enhanced_df.loc[idx, f'{method}_max_safe_operating_pressure_mpa'] = 0.0
                enhanced_df.loc[idx, f'{method}_pressure_margin_pct'] = 0.0
                enhanced_df.loc[idx, f'{method}_erf'] = float('inf')  # ERF is infinite for failed calculations
    
    return enhanced_df


def classify_defect_pressure_status(failure_pressure_mpa, safe_pressure_mpa, analysis_pressure_mpa, max_allowable_pressure_mpa):
    """
    Classify defect based on pressure-based criteria.
    
    Parameters:
    - failure_pressure_mpa: Calculated failure pressure
    - safe_pressure_mpa: Safe operating pressure (with safety factor)
    - analysis_pressure_mpa: Current/proposed operating pressure
    - max_allowable_pressure_mpa: Maximum allowable operating pressure (MAOP)
    
    Returns:
    - Dict with classification results
    """
    
    # Handle invalid calculations
    if failure_pressure_mpa <= 0 or safe_pressure_mpa <= 0:
        return {
            'pressure_status': 'CALCULATION_ERROR',
            'operational_status': 'UNKNOWN',
            'can_operate_at_analysis_pressure': False,
            'can_operate_at_max_allowable_pressure': False,
            'recommended_action': 'Review calculation inputs',
            'max_safe_operating_pressure_mpa': 0.0,
            'pressure_margin_pct': 0.0
        }
    
    # Determine if defect is safe at different pressures
    safe_at_analysis = safe_pressure_mpa >= analysis_pressure_mpa
    safe_at_max_allowable = safe_pressure_mpa >= max_allowable_pressure_mpa
    
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
    elif safe_at_max_allowable:
        operational_status = 'PRESSURE_DERATION_REQUIRED'
        max_safe_pressure = safe_pressure_mpa * 0.95  # 5% additional margin
        recommended_action = f'Reduce pressure to ≤{max_safe_pressure:.1f} MPa'
    else:
        operational_status = 'IMMEDIATE_REPAIR_REQUIRED'
        recommended_action = 'Repair defect before operations'
    
    # Calculate maximum safe operating pressure
    max_safe_operating_pressure = safe_pressure_mpa * 0.95  # 5% margin
    
    # Calculate pressure margin
    pressure_margin = ((safe_pressure_mpa - analysis_pressure_mpa) / analysis_pressure_mpa * 100) if analysis_pressure_mpa > 0 else 0
    
    return {
        'pressure_status': pressure_status,
        'operational_status': operational_status,
        'can_operate_at_analysis_pressure': safe_at_analysis,
        'can_operate_at_max_allowable_pressure': safe_at_max_allowable,
        'recommended_action': recommended_action,
        'max_safe_operating_pressure_mpa': max_safe_operating_pressure,
        'pressure_margin_pct': pressure_margin
    }


def create_pressure_based_summary_stats(enhanced_df, analysis_pressure_mpa, max_allowable_pressure_mpa):
    """
    Create summary statistics for pressure-based assessment including ERF metrics.
    """
    methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
    summary = {
        'analysis_pressure_mpa': analysis_pressure_mpa,
        'max_allowable_pressure_mpa': max_allowable_pressure_mpa,
        'total_defects': len(enhanced_df)
    }
    
    for method in methods:
        # Count defects by operational status
        operational_counts = enhanced_df[f'{method}_operational_status'].value_counts().to_dict()
        pressure_status_counts = enhanced_df[f'{method}_pressure_status'].value_counts().to_dict()
        
        # Calculate key metrics
        can_operate_analysis = enhanced_df[f'{method}_can_operate_at_analysis_pressure'].sum()
        can_operate_max_allowable = enhanced_df[f'{method}_can_operate_at_max_allowable_pressure'].sum()
        
        # Calculate percentages
        total_valid = enhanced_df[enhanced_df[f'{method}_safe'] == True].shape[0]
        pct_safe_analysis = (can_operate_analysis / total_valid * 100) if total_valid > 0 else 0
        pct_safe_max_allowable = (can_operate_max_allowable / total_valid * 100) if total_valid > 0 else 0
        
        # Find maximum safe operating pressure for the pipeline
        valid_pressures = enhanced_df[enhanced_df[f'{method}_safe'] == True][f'{method}_max_safe_operating_pressure_mpa']
        max_pipeline_pressure = valid_pressures.min() if not valid_pressures.empty else 0
        
        # ERF statistics - exclude infinite values for statistics
        valid_erf_data = enhanced_df[
            (enhanced_df[f'{method}_safe'] == True) & 
            (enhanced_df[f'{method}_erf'] != float('inf')) &
            (enhanced_df[f'{method}_erf'] > 0)
        ][f'{method}_erf']
        
        if not valid_erf_data.empty:
            erf_max = valid_erf_data.max()
            erf_min = valid_erf_data.min()
            erf_mean = valid_erf_data.mean()
            erf_greater_than_1 = (valid_erf_data > 1.0).sum()
            erf_less_than_1 = (valid_erf_data <= 1.0).sum()
            pct_erf_greater_than_1 = (erf_greater_than_1 / len(valid_erf_data) * 100) if len(valid_erf_data) > 0 else 0
        else:
            erf_max = erf_min = erf_mean = 0.0
            erf_greater_than_1 = erf_less_than_1 = 0
            pct_erf_greater_than_1 = 0.0
        
        summary[method] = {
            'operational_status_counts': operational_counts,
            'pressure_status_counts': pressure_status_counts,
            'defects_safe_at_analysis_pressure': can_operate_analysis,
            'defects_within_maop': can_operate_max_allowable,
            'pct_safe_at_analysis_pressure': pct_safe_analysis,
            'pct_within_maop': pct_safe_max_allowable,
            'max_pipeline_operating_pressure_mpa': max_pipeline_pressure,
            'total_valid_defects': total_valid,
            # ERF Statistics
            'erf_max': erf_max,
            'erf_min': erf_min,
            'erf_mean': erf_mean,
            'defects_erf_greater_than_1': erf_greater_than_1,
            'defects_erf_less_equal_1': erf_less_than_1,
            'pct_erf_requires_action': pct_erf_greater_than_1
        }
    return summary


def compute_corrosion_metrics_for_dataframe(defects_df, joints_df, pipe_diameter_mm, smys_mpa, safety_factor, maop_mpa=0.0):
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
    - maop_mpa: Maximum Allowable Operating Pressure in MPa (optional, default 0.0)
    
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
            f"CRITICAL: {len(missing_joint_defects)} defects have nnnnnnnnnno joint number assigned. "
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
            b31g_result = calculate_b31g(depth_pct, length_mm, pipe_diameter_mm, wall_thickness_mm, maop_mpa, smys_mpa, safety_factor) 
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
            mod_b31g_result = calculate_modified_b31g(depth_pct, length_mm, pipe_diameter_mm, wall_thickness_mm, maop_mpa, smys_mpa, safety_factor)
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
            effective_area_result = calculate_rstreng_effective_area_single(
                depth_pct, length_mm, width_mm, pipe_diameter_mm, 
                wall_thickness_mm, maop_mpa, smys_mpa, safety_factor
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
    Create a download link for the simplified corrosion assessment results.
    Removes unnecessary columns to reduce file size by 60-80%.
    """
    
    # Define essential columns only (remove calculation details, notes, etc.)
    essential_columns = [
        # Core defect identification
        'log dist. [m]',
        'joint number', 
        'depth [%]',
        'length [mm]',
        'width [mm]',
        'surface location',
        'component / anomaly identification',
        'wall_thickness_used_mm',
        
        # Assessment results - keep only the important ones
        'b31g_safe',
        'b31g_failure_pressure_mpa',
        'b31g_safe_pressure_mpa',
        'b31g_erf',
        'b31g_operational_status',
        'b31g_recommended_action',
        
        'modified_b31g_safe',
        'modified_b31g_failure_pressure_mpa', 
        'modified_b31g_safe_pressure_mpa',
        'modified_b31g_erf',
        'modified_b31g_operational_status',
        'modified_b31g_recommended_action',
        
        'simplified_eff_area_safe',
        'simplified_eff_area_failure_pressure_mpa',
        'simplified_eff_area_safe_pressure_mpa',
        'simplified_eff_area_erf',
        'simplified_eff_area_operational_status',
        'simplified_eff_area_recommended_action',
        
        # Pressure analysis (if available)
        'analysis_pressure_mpa',
        'max_allowable_pressure_mpa'
    ]
    
    # Filter to only existing columns
    available_columns = [col for col in essential_columns if col in enhanced_df.columns]
    
    # Create simplified DataFrame
    simplified_df = enhanced_df[available_columns].copy()
    
    # Round numerical columns to reduce file size
    numerical_columns = {
        'depth [%]': 2,
        'length [mm]': 1, 
        'width [mm]': 1,
        'wall_thickness_used_mm': 2,
        'b31g_failure_pressure_mpa': 2,
        'b31g_safe_pressure_mpa': 2,
        'b31g_erf': 3,
        'modified_b31g_failure_pressure_mpa': 2,
        'modified_b31g_safe_pressure_mpa': 2,
        'modified_b31g_erf': 3,
        'simplified_eff_area_failure_pressure_mpa': 2,
        'simplified_eff_area_safe_pressure_mpa': 2,
        'simplified_eff_area_erf': 3,
        'analysis_pressure_mpa': 3,
        'max_allowable_pressure_mpa': 3
    }
    
    for col, decimals in numerical_columns.items():
        if col in simplified_df.columns:
            simplified_df[col] = simplified_df[col].round(decimals)
    
    # Create CSV
    csv = simplified_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    filename = f"defects_corrosion_assessment_{year}.csv"
    
    # Calculate size reduction
    original_size = len(enhanced_df.to_csv(index=False)) / (1024 * 1024)
    simplified_size = len(csv) / (1024 * 1024)
    reduction_pct = (1 - simplified_size / original_size) * 100
    
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="custom-button" style="display:inline-block;text-decoration:none;margin-top:10px;font-size:0.9em;padding:8px 15px;background-color:#27AE60;color:white;border-radius:5px;">📊 Download Corrosion Assessment CSV</a>'
    
    return href



def render_corrosion_assessment_view():

    # Check if datasets are available
    datasets = get_state('datasets', {})
    if not datasets:
        st.info("""
            **No datasets available**
            Please upload pipeline inspection data using the sidebar to enable corrosion assessment.
        """)
        return
    
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
            format="%.4f",
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
                format="%.2f",
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
                format="%.3f",
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
            format="%.3f",
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
            format="%.3f",
            key="max_allowable_pressure_corrosion", 
            help="Maximum Allowable Operating Pressure (MAOP) - regulatory/design limit"
        )
        st.caption(f"= {max_allowable_pressure_mpa * 145.038:.0f} psi")

    # Convert diameter to mm for calculations
    pipe_diameter_mm = pipe_diameter_m * 1000
    
    st.markdown('</div>', unsafe_allow_html=True) 
    
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
        st.markdown("<div class='section-header'>📊 Traditional Assessment Results</div>", unsafe_allow_html=True)
        
        # Create summary statistics
        summary_stats = create_assessment_summary_stats(enhanced_df)
        
        # Create tabs for each method
        method_tabs = st.tabs(["B31G Original", "Modified B31G", "RSTRENG", "Defect Population Analysis"])
        
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

    
        with method_tabs[3]:
            st.markdown("### 🎯 Defect Population Assessment")
            
            # Create the defect assessment scatter plot
            assessment_plot = create_defect_assessment_scatter_plot(
                enhanced_df, 
                pipe_diameter_mm, 
                smys_mpa, 
                safety_factor
            )
            
            if assessment_plot:
                st.plotly_chart(assessment_plot, use_container_width=True, key="defect_assessment_plot")
                
                st.markdown("---")  # Add separator
                
                # Add explanation
                st.markdown("""
                **📊 Plot Interpretation:**
                - **X-axis (log scale)**: Defect axial length from 1mm to 10,000mm
                - **Y-axis**: Defect depth in millimeters (actual depth, not percentage)
                - **Color coding**: Blue = External, Red = Internal, Green = Combined (clustered), Gray = Unknown
                - **Reference lines**: 
                - Yellow dashed = ASME B31G allowable defect envelope
                - Orange solid = Modified B31G allowable defect envelope  
                - Black horizontal = Nominal wall thickness limit
                - **Engineering insight**: Defects above the curves require special attention per FFS standards
                """)
                
                # Create and display summary table
                assessment_summary = create_defect_assessment_summary_table(enhanced_df)
                if not assessment_summary.empty:
                    st.markdown("#### 📋 Defect Population Summary by Surface Location")
                    st.dataframe(
                        assessment_summary, 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Surface Location": st.column_config.TextColumn("🌍 Surface", width="medium"),
                            "Count": st.column_config.NumberColumn("🔢 Count", width="small"),
                            "Avg Depth (mm)": st.column_config.NumberColumn("📏 Avg Depth (mm)", format="%.2f", width="medium"),
                            "Max Depth (mm)": st.column_config.NumberColumn("⚠️ Max Depth (mm)", format="%.2f", width="medium"),
                            "Avg Length (mm)": st.column_config.NumberColumn("📐 Avg Length (mm)", format="%.1f", width="medium"),
                            "Max Length (mm)": st.column_config.NumberColumn("📏 Max Length (mm)", format="%.1f", width="medium"),
                            "Percentage": st.column_config.NumberColumn("📊 %", format="%.1f%%", width="small")
                        }
                    )
                    
                    # Add insights based on the data
                    external_count = assessment_summary[assessment_summary['Surface Location'] == 'NON-INT']['Count'].sum()
                    internal_count = assessment_summary[assessment_summary['Surface Location'] == 'INT']['Count'].sum()
                    total_count = assessment_summary['Count'].sum()
                    
                    if total_count > 0:
                        external_pct = (external_count / total_count * 100) if external_count > 0 else 0
                        internal_pct = (internal_count / total_count * 100) if internal_count > 0 else 0
                        
                        insights = []
                        if external_pct > 70:
                            insights.append("🔍 **External corrosion dominance** - typical for buried pipelines")
                        if internal_pct > 30:
                            insights.append("⚠️ **Significant internal corrosion** - consider product contamination or flow conditions")
                        if external_pct > 0 and internal_pct > 0:
                            insights.append("🔄 **Mixed corrosion pattern** - requires comprehensive integrity management")
                        
                        if insights:
                            st.markdown("#### 🔬 Engineering Insights")
                            for insight in insights:
                                st.markdown(insight)

            st.markdown("#### 📈 RSTRENG Fitness-for-Service Envelope")        
            rstreng_envelope_plot = create_rstreng_envelope_plot(
                enhanced_df,
                pipe_diameter_mm,
                smys_mpa,
                safety_factor,
                max_allowable_pressure_mpa
            )
            
            if rstreng_envelope_plot:
                st.plotly_chart(rstreng_envelope_plot, use_container_width=True, key="rstreng_envelope_plot")
                
                st.markdown("""
                **📊 RSTRENG Envelope Interpretation:**
                - **Red curve**: RSTRENG allowable defect envelope per Kiefner & Vieth methodology
                - **Above curve**: Defects requiring repair or pressure reduction
                - **Below curve**: Defects acceptable for continued operation at MAOP
                - **Color coding**: Green = External, Blue = Internal, Red = Unknown surface
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close traditional results container
        
        # NEW: Pressure Assessment Results Section
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
                pressure_plot = create_pressure_assessment_visualization(enhanced_df, method)
                if(pressure_plot):
                    st.plotly_chart(pressure_plot, use_container_width=True, key=f"pressure_plot_{method}_{idx}")


        st.markdown('</div>', unsafe_allow_html=True)  # Close pressure results container
        
 
        # Enhanced Dataset Preview Section
        st.markdown("<div class='section-header'>Enhanced Dataset Preview</div>", unsafe_allow_html=True)
        st.markdown("The following shows the first 10 rows with the newly computed corrosion assessment, pressure analysis, and ERF columns:")

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

        # NEW: Add ERF columns for each method
        erf_cols = [
            'b31g_erf',
            'modified_b31g_erf',
            'simplified_eff_area_erf'
        ]

        # Add ERF interpretation guide
        st.markdown("#### 📊 ERF (Estimated Repair Factor) Interpretation")
        st.markdown("""
        **ERF = Max Allowable Pressure / Safe Working Pressure **
        - **ERF ⩽ 0.99**: ✅ Defect acceptable for normal operations
        - **ERF > 0.99**: ⚠️ Repair required or pressure reduction needed
        - **Higher ERF values**: Better condition, more safety margin
        """)

        # Export Section
        st.markdown("<div class='section-header'>Export Results</div>", unsafe_allow_html=True)
        download_link = create_enhanced_csv_download_link(enhanced_df, selected_year)
        st.markdown(download_link, unsafe_allow_html=True)

        st.info(f"""
            **Export Information:**
            - Total columns in enhanced CSV: {len(enhanced_df.columns)}
            - Original defect data: {len(defects_df.columns)} columns
            - New assessment columns: {len(enhanced_df.columns) - len(defects_df.columns)} columns
            - Assessment methods: B31G Original, Modified B31G (0.85dL), RSTRENG (Simplified)
            - Pressure analysis: Analysis Pressure = {analysis_pressure_mpa:.1f} MPa, Max Allowable = {max_allowable_pressure_mpa:.1f} MPa
            - ERF analysis: Estimated Repair Factor calculated for each method
            - Wall thickness: Joint-specific values extracted from dataset
        """)

        st.markdown('</div>', unsafe_allow_html=True)  # Close export container