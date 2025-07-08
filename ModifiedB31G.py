def calculate_modified_b31g(defect_depth_pct, defect_length_mm, pipe_diameter_mm, wall_thickness_mm, maop_mpa, smys_mpa, safety_factor = 1.39):
    """
    Calculate remaining strength and failure pressure using the Modified B31G
    Level-1 method (often referred to as the 0.85dL method).

    Parameters:
    - defect_depth_pct: Defect depth (%). Must be an absolute value.
    - defect_length_mm: Longitudinal extent of corrosion (L) in millimeters.
    - pipe_diameter_mm: Outside diameter of pipe (D) in millimeters.
    - wall_thickness_mm: Nominal wall thickness (t) in millimeters.
    - maop_moa: Maximum allowed operating pressure
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
        "safe": safe_pressure_mpa >= maop_mpa,
        "failure_pressure_mpa": failure_pressure_mpa,
        "safe_pressure_mpa": safe_pressure_mpa,
        "remaining_strength_pct": remaining_strength_factor_pct,
        "folias_factor_M": M,
        "z_parameter": z,
        "d_over_t_ratio": d_t,
        "note": f"Calculation successful. z={z:.2f}, d/t={d_t:.3f}, A/A0_ratio={A_A0_ratio:.3f}"
    }