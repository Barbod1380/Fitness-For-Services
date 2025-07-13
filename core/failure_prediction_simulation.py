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
class JointFailure:
    """Record of a joint failure"""
    joint_number: int
    failure_year: int
    failure_mode: str  # 'ERF_EXCEEDED' or 'DEPTH_EXCEEDED' or 'BOTH'
    final_erf: float
    final_depth_pct: float
    defect_count: int



                
class FailurePredictionSimulator:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.defect_states: List[DefectState] = []
        self.failure_history: List[JointFailure] = []
        self.annual_results: List[Dict] = []

    
    def initialize_simulation(self, defects_df, joints_df, growth_rates_df, clusters, pipe_diameter, smys, safety_factor, use_clustering=True):
        # SOLUTION: Filter defects with valid depth data
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
            
            # Initialize defect states (rest unchanged)
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
        failed_joints = set()
        
        for year in range(self.params.simulation_years + 1):
            # Grow defects for this year (except year 0)
            if year > 0:
                self._grow_defects(year)
            
            # Check for failures
            year_failures = self._check_failures(year, failed_joints)
            
            # Record annual results
            annual_result = {
                'year': year,
                'total_joints': len(set(defect.joint_number for defect in self.defect_states)),
                'failed_joints_this_year': len(year_failures),
                'cumulative_failed_joints': len(failed_joints),
                'surviving_joints': len(set(defect.joint_number for defect in self.defect_states)) - len(failed_joints),
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
    


    # Integration: Replace in analysis/growth_analysis.py line 420
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
    

    def _check_failures(self, year: int, failed_joints: set) -> List[JointFailure]:
        """SIMPLIFIED: Check for joint failures using existing corrosion assessment."""
        
        year_failures = []
        
        # Group defects by joint
        joints_defects = {}
        for defect in self.defect_states:
            if defect.joint_number not in failed_joints:  # Skip already failed joints
                if defect.joint_number not in joints_defects:
                    joints_defects[defect.joint_number] = []
                joints_defects[defect.joint_number].append(defect)
        
        # Check each joint for failure
        for joint_num, joint_defects in joints_defects.items():
            
            # Create temporary DataFrame for this joint's defects
            joint_defects_data = []
            for defect in joint_defects:
                joint_defects_data.append({
                    'depth [%]': defect.current_depth_pct,
                    'length [mm]': defect.current_length_mm,
                    'width [mm]': defect.current_width_mm,
                    'wt nom [mm]': defect.wall_thickness_mm,
                    'stress_concentration_factor': 1 if year == 0 else defect.stress_concentration_factor
                })
            
            joint_df = pd.DataFrame(joint_defects_data)
            
            # USE EXISTING CORROSION ASSESSMENT to calculate ERF
            worst_erf = self._calculate_erf_using_existing_system(joint_df)
            worst_depth = max(defect.current_depth_pct for defect in joint_defects)

            # Check failure criteria
            erf_failure = worst_erf >= self.params.erf_threshold
            depth_failure = worst_depth >= self.params.depth_threshold

            if erf_failure or depth_failure:
                # Determine failure mode
                if erf_failure and depth_failure:
                    failure_mode = 'BOTH'
                elif erf_failure:
                    failure_mode = 'ERF_EXCEEDED'
                else:
                    failure_mode = 'DEPTH_EXCEEDED'
                
                # Record failure
                failure = JointFailure(
                    joint_number=joint_num,
                    failure_year=year,
                    failure_mode=failure_mode,
                    final_erf=worst_erf,
                    final_depth_pct=worst_depth,
                    defect_count=len(joint_defects)
                )
                
                year_failures.append(failure)
                failed_joints.add(joint_num)
                self.failure_history.append(failure)
        
        return year_failures


    def _calculate_erf_using_existing_system(self, joint_df: pd.DataFrame) -> float:
        max_erf = 0.0
        valid_calculations = 0

        # Single loop that processes each defect individually
        for _, defect in joint_df.iterrows():
            try:
                # Extract CURRENT defect parameters (already grown) - INSIDE the loop
                depth_pct = defect['depth [%]']
                length_mm = defect['length [mm]']
                width_mm = defect['width [mm]']
                wt_mm = defect['wt nom [mm]']

                # Call assessment method with CURRENT dimensions only
                if self.params.assessment_method == 'b31g':
                    result = calculate_b31g(
                        defect_depth_pct=depth_pct,  # Current defect's parameters
                        defect_length_mm=length_mm,
                        pipe_diameter_mm=self.pipe_diameter * 1000,
                        wall_thickness_mm=wt_mm,
                        maop_mpa=self.params.max_operating_pressure,
                        smys_mpa=self.smys,
                        safety_factor=self.safety_factor
                    )
                
                elif self.params.assessment_method == 'modified_b31g':
                    result = calculate_modified_b31g(
                        defect_depth_pct=depth_pct,
                        defect_length_mm=length_mm,
                        pipe_diameter_mm=self.pipe_diameter * 1000,
                        wall_thickness_mm=wt_mm,
                        maop_mpa=self.params.max_operating_pressure,
                        smys_mpa=self.smys,
                        safety_factor=self.safety_factor
                    )
                
                elif self.params.assessment_method == 'rstreng':
                    result = calculate_true_rstreng_method(
                        defect_depth_pct=depth_pct,
                        defect_length_mm=length_mm,
                        defect_width_mm=width_mm,
                        pipe_diameter_mm=self.pipe_diameter * 1000,
                        wall_thickness_mm=wt_mm,
                        maop_mpa=self.params.max_operating_pressure,
                        smys_mpa=self.smys,
                        safety_factor=self.safety_factor
                    )
                
                # SOLUTION: Robust ERF calculation
                safe_pressure = result.get('safe_pressure_mpa', 0)
                calculation_safe = result.get('safe', False)
                
                # Only calculate ERF for valid results
                if calculation_safe and safe_pressure > 0:
                    erf = self.params.max_operating_pressure / safe_pressure
                    max_erf = max(max_erf, erf)
                    valid_calculations += 1
                else:
                    # Log but don't use invalid calculations
                    print(f"Invalid calculation for defect: depth={depth_pct}%, method={self.params.assessment_method}")
                    
            except Exception as e:
                print(f"Error calculating ERF for defect: {str(e)}")
                # Continue processing other defects
                continue
        
        # If no valid calculations, return low ERF (safe)
        if valid_calculations == 0:
            print(f"Warning: No valid calculations for joint, assuming safe")
            return 0.5  # Conservative but not failure-triggering
        
        return max_erf


    def _calculate_max_erf(self) -> float:
        """Calculate maximum ERF across all surviving joints."""
        max_erf = 0.0
        
        # Group defects by joint first
        joints_defects = {}
        for defect in self.defect_states:
            if defect.joint_number not in joints_defects:
                joints_defects[defect.joint_number] = []
            joints_defects[defect.joint_number].append(defect)
        
        # Calculate ERF per joint, then take maximum
        for _, joint_defects in joints_defects.items():
            # Create DataFrame directly (like you do in _calculate_erf_using_existing_system)
            joint_defects_data = []
            for defect in joint_defects:
                joint_defects_data.append({
                    'depth [%]': defect.current_depth_pct,
                    'length [mm]': defect.current_length_mm,
                    'width [mm]': defect.current_width_mm,
                    'wt nom [mm]': defect.wall_thickness_mm,
                    'stress_concentration_factor': defect.stress_concentration_factor
                })
            
            joint_df = pd.DataFrame(joint_defects_data)
            joint_erf = self._calculate_erf_using_existing_system(joint_df)
            max_erf = max(max_erf, joint_erf)
        
        return max_erf
    

    def _compile_results(self) -> Dict:
        """Compile final simulation results."""
        
        return {
            'simulation_params': self.params,
            'annual_results': self.annual_results,
            'failure_history': self.failure_history,
            'total_failures': len(self.failure_history),
            'failure_timeline': self._create_failure_timeline(),
            'survival_statistics': self._calculate_survival_statistics()
        }
    
    def _create_failure_timeline(self) -> Dict[int, int]:
        """Create timeline of failures by year."""
        timeline = {}
        for year in range(self.params.simulation_years + 1):
            timeline[year] = len([f for f in self.failure_history if f.failure_year == year])
        return timeline
    
    def _calculate_survival_statistics(self) -> Dict:
        """Calculate survival statistics."""
        total_joints = len(set(defect.joint_number for defect in self.defect_states))
        failed_joints = len(self.failure_history)
        surviving_joints = total_joints - failed_joints
        
        return {
            'total_joints': total_joints,
            'failed_joints': failed_joints,
            'surviving_joints': surviving_joints,
            'failure_rate': (failed_joints / total_joints * 100) if total_joints > 0 else 0,
            'survival_rate': (surviving_joints / total_joints * 100) if total_joints > 0 else 100
        }