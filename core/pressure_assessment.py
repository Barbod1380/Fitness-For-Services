# Create new file: core/pressure_assessment.py

"""
Safe operating pressure calculations for different assessment methods.
Used in time-forward failure prediction simulation.
"""

import math
from dataclasses import dataclass

@dataclass
class PressureAssessmentParams:
    """Parameters for pressure assessment"""
    method: str  # 'B31G', 'Modified_B31G', 'RSTRENG'
    smys: float  # Specified Minimum Yield Strength (MPa)
    diameter: float  # Pipe diameter (mm)
    safety_factor: float  # Safety factor
    
class SafeOperatingPressureCalculator:
    """
    Calculate safe operating pressures using different industry methods.
    """
    
    def __init__(self, params: PressureAssessmentParams):
        self.params = params
    
    def calculate_safe_pressure(self, 
                              defect_depth_pct: float,
                              defect_length_mm: float,
                              wall_thickness_mm: float,
                              stress_concentration_factor: float = 1.0) -> float:
        """
        Calculate safe operating pressure for a defect.
        
        Parameters:
        - defect_depth_pct: Defect depth as percentage of wall thickness
        - defect_length_mm: Defect length in mm
        - wall_thickness_mm: Wall thickness in mm
        - stress_concentration_factor: Stress concentration due to clustering
        
        Returns:
        - Safe operating pressure in MPa
        """
        
        if self.params.method == 'B31G':
            return self._calculate_b31g_pressure(
                defect_depth_pct, defect_length_mm, wall_thickness_mm, stress_concentration_factor
            )
        elif self.params.method == 'Modified_B31G':
            return self._calculate_modified_b31g_pressure(
                defect_depth_pct, defect_length_mm, wall_thickness_mm, stress_concentration_factor
            )
        elif self.params.method == 'RSTRENG':
            return self._calculate_rstreng_pressure(
                defect_depth_pct, defect_length_mm, wall_thickness_mm, stress_concentration_factor
            )
        else:
            raise ValueError(f"Unknown assessment method: {self.params.method}")
    
    def _calculate_b31g_pressure(self, depth_pct: float, length_mm: float, 
                                wall_thickness_mm: float, stress_factor: float) -> float:
        """Calculate safe pressure using B31G method"""
        
        # B31G classic formula
        d_over_t = depth_pct / 100.0
        diameter_mm = self.params.diameter
        
        # Calculate length parameter
        L = length_mm
        sqrt_Dt = math.sqrt(diameter_mm * wall_thickness_mm)
        
        # B31G factor calculation
        if L <= sqrt_Dt:
            # Short defect
            M = math.sqrt(1 + 0.6275 * (L / sqrt_Dt)**2 - 0.003375 * (L / sqrt_Dt)**4)
        else:
            # Long defect
            M = 0.032 * (L / sqrt_Dt) + 3.3
        
        # Safe pressure calculation
        # Apply stress concentration factor to effective depth
        effective_d_over_t = d_over_t * stress_factor
        effective_d_over_t = min(effective_d_over_t, 0.8)  # Cap at 80%
        
        safe_pressure = (2 * self.params.smys * wall_thickness_mm / diameter_mm) * \
                       (1 - effective_d_over_t) / (1 - (effective_d_over_t / M)) / \
                       self.params.safety_factor
        
        return max(safe_pressure, 0.1)  # Minimum 0.1 MPa
    
    def _calculate_modified_b31g_pressure(self, depth_pct: float, length_mm: float,
                                        wall_thickness_mm: float, stress_factor: float) -> float:
        """Calculate safe pressure using Modified B31G method"""
        
        d_over_t = depth_pct / 100.0
        diameter_mm = self.params.diameter
        
        # Modified B31G uses effective area approach
        L = length_mm
        sqrt_Dt = math.sqrt(diameter_mm * wall_thickness_mm)
        
        # Modified factor calculation (more conservative)
        M = math.sqrt(1 + 0.6275 * (L / sqrt_Dt)**2 - 0.003375 * (L / sqrt_Dt)**4)
        
        # Apply stress concentration to both depth and length effects
        effective_d_over_t = d_over_t * stress_factor
        effective_length_factor = min(stress_factor, 1.5)  # Cap length effect
        
        effective_d_over_t = min(effective_d_over_t, 0.8)
        
        # Modified B31G safe pressure
        safe_pressure = (2 * self.params.smys * wall_thickness_mm / diameter_mm) * \
                       (1 - effective_d_over_t) / (1 - (effective_d_over_t / (M * effective_length_factor))) / \
                       self.params.safety_factor
        
        return max(safe_pressure, 0.1)
    
    def _calculate_rstreng_pressure(self, depth_pct: float, length_mm: float,
                                  wall_thickness_mm: float, stress_factor: float) -> float:
        """Calculate safe pressure using RSTRENG (effective area) method"""
        
        d_over_t = depth_pct / 100.0
        diameter_mm = self.params.diameter
        
        # RSTRENG effective area calculation
        defect_area = length_mm * (d_over_t * wall_thickness_mm)
        
        # Apply stress concentration to effective area
        effective_area = defect_area * stress_factor
        
        # Total cross-sectional area
        total_area = math.pi * diameter_mm * wall_thickness_mm
        
        # Effective area ratio
        area_ratio = effective_area / total_area
        area_ratio = min(area_ratio, 0.8)  # Cap at 80%
        
        # RSTRENG folias factor for effective area
        if length_mm > 0:
            Mt = math.sqrt(1 + 0.6275 * (length_mm / math.sqrt(diameter_mm * wall_thickness_mm))**2)
        else:
            Mt = 1.0
        
        # Safe pressure calculation
        safe_pressure = (2 * self.params.smys * wall_thickness_mm / diameter_mm) * \
                       (1 - area_ratio) / (1 - (area_ratio / Mt)) / \
                       self.params.safety_factor
        
        return max(safe_pressure, 0.1)