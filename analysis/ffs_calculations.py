"""
Core Fitness-for-Service (FFS) calculation functions.

This module contains the implementations of various industry-standard
corrosion assessment methods.
"""
import math
import numpy as np
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

    if z <= 20.0:
        M = math.sqrt(1.0 + 0.8 * z)             # B31G eqn 1‑2
    else:
        # For z > 20, the original B31G standard uses a different formulation.
        # M = 0.6 * sqrt(z) + 2.0. The previous implementation was non-standard.
        # This has been corrected to align with the published standard.
        M = 0.6 * math.sqrt(z) + 2.0

    # ── 3  flow stress (take the minimum of the three forms) ──
    S_flow = 1.1 * smys_mpa
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


def calculate_rstreng_level1(
    defect_depth_pct: float,
    defect_length_mm: float,
    defect_width_mm: Optional[float],
    pipe_diameter_mm: float,
    wall_thickness_mm: float,
    maop_mpa: float,
    smys_mpa: float,
    alpha: float = 0.85,
    safety_factor: float = 1.39,
    smts_mpa: Optional[float] = None,
) -> dict:
    """
    RSTRENG Level-1 / Modified B31G calculation for a single defect.

    This function implements the generalized Level-1 methodology that is the
    basis for both Modified B31G and RSTRENG. The behavior is controlled by
    the `alpha` parameter, which defines the effective area of the defect.

    Parameters:
        defect_depth_pct: Max defect depth as a percentage of wall thickness (e.g., 40 for 40%).
        defect_length_mm: Axial length of the defect (mm).
        defect_width_mm: Unused in Level-1, included for API compatibility.
        pipe_diameter_mm: Outside diameter of the pipe (mm).
        wall_thickness_mm: Nominal wall thickness (mm).
        maop_mpa: Maximum Allowable Operating Pressure (MPa).
        smys_mpa: Specified Minimum Yield Strength (MPa).
        alpha: Shape factor for the defect area.
               - 0.85: Standard for Modified B31G / RSTRENG (effective rectangular area).
               - 2/3 (~0.667): Parabolic area, as used in original B31G.
               Default is 0.85.
        safety_factor: Safety factor for pressure (default 1.39).
        smts_mpa: Optional SMTS (MPa) to cap the flow stress.

    Returns:
        A dictionary containing the detailed results of the assessment.
    """

    method = "RSTRENG Level-1"

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
        folias_factor_M=M,
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

    # ---- Basic checks
    D = float(pipe_diameter_mm)
    t = float(wall_thickness_mm)
    dx = float(axial_step_mm)

    depths = np.asarray(depth_profile_mm, dtype=float)
    N = int(len(depths))
    if D <= 0 or t <= 0 or dx <= 0 or N == 0:
        return dict(method=method, safe=False, note="Invalid geometry or sampling.")

    if np.any(depths < 0):
        return dict(method=method, safe=False, note="Negative depth in profile.")

    # ---- Flow stress (RSTRENG / Mod-B31G convention: SMYS + 10 ksi, capped at 0.9*SMTS if provided)
    S_flow = smys_mpa + 68.95
    if smts_mpa is not None:
        S_flow = min(S_flow, 0.9 * smts_mpa)

    # ---- Prefix sums for O(1) area queries
    # area(i:j) = dx * (prefix[j] - prefix[i]) , window length Lw = (j - i) * dx
    prefix = np.concatenate(([0.0], np.cumsum(depths)))

    best = {
        "P_safe": float("inf"),
        "Pf": float("inf"),
        "Sf": float("inf"),
        "RSF": None,
        "Mt": None,
        "L2DT": None,
        "i": None,
        "j": None,
    }

    # ---- Optimized Sliding-Window Search ----
    # Identify candidate start points to avoid iterating from every single point.
    # A candidate is a point where a defect region begins.
    candidate_starts = [i for i in range(N) if depths[i] > 0 and (i == 0 or depths[i-1] == 0)]

    # If no corrosion, the pipe is safe.
    if not candidate_starts:
        return dict(
            method=method, safe=True, failure_pressure_mpa=float('inf'),
            safe_pressure_mpa=float('inf'), remaining_strength_pct=100.0,
            note="No corrosion detected in profile."
        )

    for i in candidate_starts:
        for j in range(i + 1, N + 1):
            Lw = (j - i) * dx
            if Lw <= 0.0:
                continue

            area_lost = dx * (prefix[j] - prefix[i])         # mm^2
            area_ref = Lw * t                                 # mm^2
            RSF = area_lost / area_ref if area_ref > 0.0 else 0.0
            if RSF >= 1.0:
                # Through-wall over this window => denominator would be <= 0
                continue

            # Folias bulging factor Mt for this sub-length (RSTRENG / Mod-B31G form)
            z = (Lw * Lw) / (D * t)                           # L^2 / (D t)
            if z <= 50.0:
                Mt = math.sqrt(max(1.0 + 0.6275 * z - 0.003375 * z * z, 1.0))
            else:
                Mt = 0.032 * z + 3.3

            numer = 1.0 - RSF
            denom = 1.0 - (RSF / Mt)
            if denom <= 0.0:
                # Unphysical combo for this window; skip
                continue

            Sf = S_flow * (numer / denom)                     # MPa
            Pf = 2.0 * Sf * t / D                             # MPa
            P_safe = Pf / safety_factor                       # MPa

            # Keep the governing (lowest safe pressure) window
            if P_safe < best["P_safe"]:
                best.update(dict(P_safe=P_safe, Pf=Pf, Sf=Sf, RSF=RSF, Mt=Mt, L2DT=z, i=i, j=j))

    # ---- No valid window?
    if not np.isfinite(best["P_safe"]):
        return dict(method=method, safe=False, failure_pressure_mpa=0.0,
                    note="No valid sub-length found (denominator <= 0 over all windows).")

    # ---- Build result using governing window metrics
    note = (f"Gov. window: i={best['i']}, j={best['j']}, "
            f"L={((best['j'] - best['i']) * dx):.1f} mm, "
            f"L2/DT={best['L2DT']:.2f}, Mt={best['Mt']:.3f}, RSF={best['RSF']:.4f}")

    return dict(
        method=method,
        safe=(best["P_safe"] >= maop_mpa),
        failure_pressure_mpa=best["Pf"],
        safe_pressure_mpa=best["P_safe"],
        remaining_strength_pct=(best["Sf"] / S_flow) * 100.0,
        folias_factor_Mt=best["Mt"],
        flow_stress_mpa=S_flow,
        area_ratio_RSF=best["RSF"],
        L2DT=best["L2DT"],
        note=note
    )
