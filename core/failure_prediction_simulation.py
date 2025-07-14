# SIMPLIFIED: core/failure_prediction_simulation.py
# Uses existing corrosion assessment instead of duplicate pressure calculations

"""
Time-forward failure prediction simulation engine.
Uses existing corrosion assessment functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from app.views.corrosion import calculate_b31g, calculate_modified_b31g, calculate_true_rstreng_method


@dataclass
class SimulationParams:
    """Parameters for failure prediction simulation"""
    assessment_method: str  # 'b31g', 'modified_b31g', 'simplified_eff_area'
    max_operating_pressure: float  # MPa  
    simulation_years: int
    erf_threshold: float
    depth_threshold: float  # percentage
    
    def __post_init__(self):
        """Convert UI method names to function names"""
        method_mapping = {
            'B31G': 'b31g',
            'Modified_B31G': 'modified_b31g',
            'RSTRENG': 'simplified_eff_area'
        }
        
        if self.assessment_method in method_mapping:
            self.assessment_method = method_mapping[self.assessment_method]

@dataclass
class DefectState:
    """Current state of a defect during simulation"""
    defect_id: int
    joint_number: int
    current_depth_pct: float
    current_length_mm: float
    current_width_mm: float
    location_m: float
    growth_rate_pct_per_year: float
    length_growth_rate_mm_per_year: float
    wall_thickness_mm: float
    stress_concentration_factor: float
    is_clustered: bool
    cluster_id: Optional[int] = None


@dataclass
class DefectFailure:
    """Record of an individual defect failure - CHANGED FROM JOINT TO DEFECT"""
    defect_id: int
    joint_number: int  # Keep for reference
    failure_year: int
    failure_mode: str  # 'ERF_EXCEEDED' or 'DEPTH_EXCEEDED' or 'BOTH'
    final_erf: float
    final_depth_pct: float
    location_m: float
    was_clustered: bool
    stress_concentration_factor: float


class FailurePredictionSimulator:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.defect_states: List[DefectState] = []
        self.failure_history: List[DefectFailure] = []  # CHANGED: Now tracks individual defects
        self.annual_results: List[Dict] = []

    
    def initialize_simulation(self, defects_df, joints_df, growth_rates_df, clusters, pipe_diameter, smys, safety_factor, use_clustering=True):
        # Filter defects with valid depth data
        valid_defects = defects_df[
            (defects_df['depth [%]'].notna()) & 
            (defects_df['depth [%]'] > 0) & 
            (defects_df['depth [%]'] <= 100) &
            (defects_df['length [mm]'].notna()) & 
            (defects_df['length [mm]'] > 0) &
            (defects_df['width [mm]'].notna()) & 
            (defects_df['width [mm]'] > 0)
        ].copy()
        
        if len(valid_defects) < len(defects_df):
            print(f"Filtered {len(defects_df) - len(valid_defects)} defects with invalid dimensions")
        
        try:
            self.pipe_diameter = pipe_diameter
            self.smys = smys
            self.safety_factor = safety_factor
            
            # Create wall thickness lookup
            wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
            
            # Create growth rate lookup
            growth_lookup = {}
            if not growth_rates_df.empty:
                for idx, row in growth_rates_df.iterrows():
                    defect_id = row.get('new_defect_id') or row.get('defect_id') or idx
                    growth_lookup[defect_id] = {
                        'depth_growth_pct': row.get('growth_rate_pct_per_year', 0.5),
                        'length_growth_mm': row.get('length_growth_rate_mm_per_year', 0.1)
                    }            

            if use_clustering:
                # Use clustering (existing logic)
                cluster_lookup = {}
                for cluster in clusters:
                    stress_factor = cluster.stress_concentration_factor
                    for defect_idx in cluster.defect_indices:
                        cluster_lookup[defect_idx] = {
                            'stress_factor': stress_factor,
                            'cluster_id': id(cluster),
                            'is_clustered': True
                        }
            else:
                # No clustering - all defects individual
                cluster_lookup = {}
                for idx in defects_df.index:
                    cluster_lookup[idx] = {
                        'stress_factor': 1.0,  # No stress concentration
                        'cluster_id': None,
                        'is_clustered': False
                    }
            
            # Initialize defect states
            self.defect_states = []
            for idx, defect in defects_df.iterrows():
                growth_data = growth_lookup.get(idx, {
                    'depth_growth_pct': 0.5,
                    'length_growth_mm': 0.1
                })
                
                cluster_info = cluster_lookup.get(idx, {
                    'stress_factor': 1.0,
                    'cluster_id': None,
                    'is_clustered': False
                })
                
                wall_thickness = wt_lookup.get(defect['joint number'], 10.0)
                
                defect_state = DefectState(
                    defect_id=idx,
                    joint_number=defect['joint number'],
                    current_depth_pct=defect['depth [%]'],
                    current_length_mm=defect['length [mm]'],
                    current_width_mm=defect['width [mm]'],
                    location_m=defect['log dist. [m]'],
                    growth_rate_pct_per_year=growth_data['depth_growth_pct'],
                    length_growth_rate_mm_per_year=growth_data['length_growth_mm'],
                    wall_thickness_mm=wall_thickness,
                    stress_concentration_factor=cluster_info['stress_factor'],
                    is_clustered=cluster_info['is_clustered'],
                    cluster_id=cluster_info['cluster_id']
                )
                
                self.defect_states.append(defect_state)
            
            return True
            
        except Exception as e:
            print(f"Simulation initialization failed: {e}")
            return False
    

    def run_simulation(self) -> Dict:
        """Run the complete simulation over the specified timeframe."""
        self.failure_history = []
        self.annual_results = []
        failed_defects = set()  # CHANGED: Track failed defect IDs instead of joint numbers
        
        for year in range(self.params.simulation_years + 1):
            # Grow defects for this year (except year 0)
            if year > 0:
                self._grow_defects(year)
            
            # Check for failures - CHANGED to defect-level
            year_failures = self._check_defect_failures(year, failed_defects)
            
            # Record annual results - CHANGED metrics
            annual_result = {
                'year': year,
                'total_defects': len(self.defect_states),  # CHANGED: Count defects
                'failed_defects_this_year': len(year_failures),  # CHANGED
                'cumulative_failed_defects': len(failed_defects),  # CHANGED
                'surviving_defects': len(self.defect_states) - len(failed_defects),  # CHANGED
                'max_erf': self._calculate_max_erf(),
                'max_depth': max(defect.current_depth_pct for defect in self.defect_states),
                'avg_depth': np.mean([defect.current_depth_pct for defect in self.defect_states])
            }
            
            self.annual_results.append(annual_result)
        
        return self._compile_results()
    

    def calculate_stress_accelerated_growth(self, base_growth_rate, stress_concentration_factor, growth_type):
        """
        Calculate stress-accelerated growth using NACE SP0169 and API 579-1 principles
        """
        if stress_concentration_factor <= 1.0:
            return base_growth_rate
        
        if growth_type == 'depth':
            # NACE SP0169-2013 Section 6.3: Stress corrosion acceleration
            # Conservative power law with exponent 0.5 for general corrosion
            acceleration_factor = 1.0 + 0.2 * (stress_concentration_factor - 1.0) ** 0.5
            
        elif growth_type == 'length':
            # API 579-1 Part 9: Crack growth follows Paris Law da/dN = C(ΔK)^m
            # For m=2 (conservative): acceleration ∝ (K_ratio)²
            stress_intensity_ratio = stress_concentration_factor ** 0.5  # K ∝ σ√(πa)
            acceleration_factor = 1.0 + 0.15 * (stress_intensity_ratio - 1.0) ** 2.0
            
        else:  # width
            # Conservative linear relationship for lateral expansion
            acceleration_factor = 1.0 + 0.1 * (stress_concentration_factor - 1.0)
        
        # Cap at 2.5x to prevent unrealistic predictions
        return base_growth_rate * min(acceleration_factor, 2.5)
    

    def _grow_defects(self, year: int):
        """Grow all defects based on their growth rates and clustering effects."""
        
        for defect in self.defect_states:
            if defect.is_clustered:
                # USE CORRECT STRESS ACCELERATION
                depth_acceleration = self.calculate_stress_accelerated_growth(base_growth_rate = 1.0, stress_concentration_factor = defect.stress_concentration_factor, growth_type = 'depth')
                length_acceleration = self.calculate_stress_accelerated_growth(base_growth_rate = 1.0, stress_concentration_factor = defect.stress_concentration_factor, growth_type = 'length')
            else:
                depth_acceleration = 1.0
                length_acceleration = 1.0
            
            # Apply accelerated growth
            depth_growth = defect.growth_rate_pct_per_year * depth_acceleration
            defect.current_depth_pct += depth_growth
            defect.current_depth_pct = min(defect.current_depth_pct, 95.0)
            
            length_growth = defect.length_growth_rate_mm_per_year * length_acceleration
            defect.current_length_mm += length_growth
    

    def _check_defect_failures(self, year: int, failed_defects: set) -> List[DefectFailure]:
        """CHANGED: Check individual defect failures instead of joint failures."""
        
        year_failures = []
        
        # Check each defect individually
        for defect in self.defect_states:
            if defect.defect_id in failed_defects:
                continue  # Skip already failed defects
            
            # Calculate ERF for this individual defect
            defect_erf = self._calculate_defect_erf(defect)
            defect_depth = defect.current_depth_pct

            # Check failure criteria for this defect
            erf_failure = defect_erf >= self.params.erf_threshold
            depth_failure = defect_depth >= self.params.depth_threshold

            if erf_failure or depth_failure:
                # Determine failure mode
                if erf_failure and depth_failure:
                    failure_mode = 'BOTH'
                elif erf_failure:
                    failure_mode = 'ERF_EXCEEDED'
                else:
                    failure_mode = 'DEPTH_EXCEEDED'
                
                # Record defect failure
                failure = DefectFailure(
                    defect_id=defect.defect_id,
                    joint_number=defect.joint_number,  # Keep for reference
                    failure_year=year,
                    failure_mode=failure_mode,
                    final_erf=defect_erf,
                    final_depth_pct=defect_depth,
                    location_m=defect.location_m,
                    was_clustered=defect.is_clustered,
                    stress_concentration_factor=defect.stress_concentration_factor
                )
                
                year_failures.append(failure)
                failed_defects.add(defect.defect_id)
                self.failure_history.append(failure)
        
        return year_failures


    def _calculate_defect_erf(self, defect: DefectState) -> float:
        """Calculate ERF for an individual defect."""
        
        try:
            # Call appropriate assessment method for this defect
            if self.params.assessment_method == 'b31g':
                result = calculate_b31g(
                    defect_depth_pct=defect.current_depth_pct,
                    defect_length_mm=defect.current_length_mm,
                    pipe_diameter_mm=self.pipe_diameter * 1000,
                    wall_thickness_mm=defect.wall_thickness_mm,
                    maop_mpa=self.params.max_operating_pressure,
                    smys_mpa=self.smys,
                    safety_factor=self.safety_factor
                )
            
            elif self.params.assessment_method == 'modified_b31g':
                result = calculate_modified_b31g(
                    defect_depth_pct=defect.current_depth_pct,
                    defect_length_mm=defect.current_length_mm,
                    pipe_diameter_mm=self.pipe_diameter * 1000,
                    wall_thickness_mm=defect.wall_thickness_mm,
                    maop_mpa=self.params.max_operating_pressure,
                    smys_mpa=self.smys,
                    safety_factor=self.safety_factor
                )
            
            elif self.params.assessment_method == 'rstreng':
                result = calculate_true_rstreng_method(
                    defect_depth_pct=defect.current_depth_pct,
                    defect_length_mm=defect.current_length_mm,
                    defect_width_mm=defect.current_width_mm,
                    pipe_diameter_mm=self.pipe_diameter * 1000,
                    wall_thickness_mm=defect.wall_thickness_mm,
                    maop_mpa=self.params.max_operating_pressure,
                    smys_mpa=self.smys,
                    safety_factor=self.safety_factor
                )
            
            # Calculate ERF
            safe_pressure = result.get('safe_pressure_mpa', 0)
            calculation_safe = result.get('safe', False)
            
            if calculation_safe and safe_pressure > 0:
                # Apply stress concentration factor to ERF calculation
                base_erf = self.params.max_operating_pressure / safe_pressure
                # For clustered defects, stress concentration increases ERF (worse condition)
                final_erf = base_erf * defect.stress_concentration_factor
                return final_erf
            else:
                return 999.0  # Very high ERF for failed calculations (indicates failure)
                
        except Exception as e:
            print(f"Error calculating ERF for defect {defect.defect_id}: {str(e)}")
            return 999.0  # Conservative failure assumption


    def _calculate_max_erf(self) -> float:
        """Calculate maximum ERF across all surviving defects."""
        max_erf = 0.0
        
        for defect in self.defect_states:
            defect_erf = self._calculate_defect_erf(defect)
            max_erf = max(max_erf, defect_erf)
        
        return max_erf
    

    def _compile_results(self) -> Dict:
        """Compile final simulation results - CHANGED to defect-based."""
        
        return {
            'simulation_params': self.params,
            'annual_results': self.annual_results,
            'failure_history': self.failure_history,
            'total_failures': len(self.failure_history),  # CHANGED: Count defect failures
            'failure_timeline': self._create_failure_timeline(),
            'survival_statistics': self._calculate_survival_statistics()
        }
    
    def _create_failure_timeline(self) -> Dict[int, int]:
        """Create timeline of defect failures by year."""
        timeline = {}
        for year in range(self.params.simulation_years + 1):
            timeline[year] = len([f for f in self.failure_history if f.failure_year == year])
        return timeline
    
    def _calculate_survival_statistics(self) -> Dict:
        """Calculate survival statistics - CHANGED to defect-based."""
        total_defects = len(self.defect_states)  # CHANGED
        failed_defects = len(self.failure_history)  # CHANGED
        surviving_defects = total_defects - failed_defects  # CHANGED
        
        return {
            'total_defects': total_defects,  # CHANGED
            'failed_defects': failed_defects,  # CHANGED
            'surviving_defects': surviving_defects,  # CHANGED
            'failure_rate': (failed_defects / total_defects * 100) if total_defects > 0 else 0,
            'survival_rate': (surviving_defects / total_defects * 100) if total_defects > 0 else 100
        }