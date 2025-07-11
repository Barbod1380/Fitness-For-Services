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
from app.views.corrosion import (
    calculate_b31g, 
    calculate_modified_b31g, 
    calculate_simplified_effective_area_method
)


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
    """
    SIMPLIFIED: Main simulation engine using existing corrosion assessment.
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        
        # NO MORE PRESSURE CALCULATOR - we'll use existing corrosion assessment
        
        # Simulation state
        self.defect_states: List[DefectState] = []
        self.failure_history: List[JointFailure] = []
        self.annual_results: List[Dict] = []
    
    def initialize_simulation(self, 
                            defects_df: pd.DataFrame,
                            joints_df: pd.DataFrame,
                            growth_rates_df: pd.DataFrame,
                            clusters: List,
                            pipe_diameter: float,  # in meters
                            smys: float,           # MPa
                            safety_factor: float) -> bool:
        """
        SIMPLIFIED: Initialize simulation with current defect states and growth rates.
        """
        try:
            # Store additional parameters needed for ERF calculation
            self.pipe_diameter = pipe_diameter
            self.smys = smys
            self.safety_factor = safety_factor
            
            # Create wall thickness lookup
            wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
            
            # Create growth rate lookup
            growth_lookup = {}
            if not growth_rates_df.empty:
                for idx, row in growth_rates_df.iterrows():
                    # Try different possible column names for defect ID
                    defect_id = row.get('new_defect_id') or row.get('defect_id') or idx
                    growth_lookup[defect_id] = {
                        'depth_growth_pct': row.get('growth_rate_pct_per_year', 0.5),
                        'length_growth_mm': row.get('length_growth_rate_mm_per_year', 0.1)
                    }
            
            # Create cluster lookup for stress concentration factors
            cluster_lookup = {}
            for cluster in clusters:
                stress_factor = cluster.stress_concentration_factor
                for defect_idx in cluster.defect_indices:
                    cluster_lookup[defect_idx] = {
                        'stress_factor': stress_factor,
                        'cluster_id': id(cluster),
                        'is_clustered': True
                    }
            
            # Initialize defect states
            self.defect_states = []
            for idx, defect in defects_df.iterrows():
                
                # Get growth rates
                growth_data = growth_lookup.get(idx, {
                    'depth_growth_pct': 0.5,  # Conservative default
                    'length_growth_mm': 0.1
                })
                
                # Get clustering info
                cluster_info = cluster_lookup.get(idx, {
                    'stress_factor': 1.0,
                    'cluster_id': None,
                    'is_clustered': False
                })
                
                # Get wall thickness
                wall_thickness = wt_lookup.get(defect['joint number'], 10.0)
                
                # Create defect state
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
    
    def _grow_defects(self, year: int):
        """Grow all defects based on their growth rates and clustering effects."""
        
        for defect in self.defect_states:
            # Apply clustering acceleration to growth rates
            if defect.is_clustered:
                # Clustered defects grow faster due to stress concentration
                depth_acceleration = 1.0 + 0.1 * (defect.stress_concentration_factor - 1.0)
                length_acceleration = 1.0 + 0.05 * (defect.stress_concentration_factor - 1.0)
            else:
                depth_acceleration = 1.0
                length_acceleration = 1.0
            
            # Grow depth
            depth_growth = defect.growth_rate_pct_per_year * depth_acceleration
            defect.current_depth_pct += depth_growth
            defect.current_depth_pct = min(defect.current_depth_pct, 95.0)  # Cap at 95%
            
            # Grow length
            length_growth = defect.length_growth_rate_mm_per_year * length_acceleration
            defect.current_length_mm += length_growth
            defect.current_length_mm = max(defect.current_length_mm, 1.0)  # Minimum 1mm
    
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
                    'stress_concentration_factor': defect.stress_concentration_factor
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
        """
        FIXED: Use existing corrosion assessment functions to calculate ERF.
        """
        
        max_erf = 0.0
        
        for idx, defect in joint_df.iterrows():
            # Extract defect parameters
            depth_pct = defect['depth [%]']
            length_mm = defect['length [mm]']
            width_mm = defect['width [mm]']
            wt_mm = defect['wt nom [mm]']
            stress_factor = defect.get('stress_concentration_factor', 1.0)
            
            # Apply stress concentration to depth (conservative approach)
            effective_depth_pct = depth_pct * stress_factor
            effective_depth_pct = min(effective_depth_pct, 80.0)  # Cap at 80%
            
            # Apply stress concentration to length (moderate effect)
            effective_length_mm = length_mm * min(stress_factor, 1.2)  # Cap length effect

            print(self.params.assessment_method)
            
            try:
                # Call your existing assessment function based on method
                if self.params.assessment_method == 'b31g':
                    result = calculate_b31g(
                        defect_depth_pct=effective_depth_pct,
                        defect_length_mm=effective_length_mm,
                        pipe_diameter_mm=self.pipe_diameter * 1000,  # Convert to mm
                        wall_thickness_mm=wt_mm,
                        maop_mpa=self.params.max_operating_pressure,
                        smys_mpa=self.smys,
                        safety_factor=self.safety_factor
                    )
                
                elif self.params.assessment_method == 'modified_b31g':
                    result = calculate_modified_b31g(
                        defect_depth_pct=effective_depth_pct,
                        defect_length_mm=effective_length_mm,
                        pipe_diameter_mm=self.pipe_diameter * 1000,
                        wall_thickness_mm=wt_mm,
                        maop_mpa=self.params.max_operating_pressure,
                        smys_mpa=self.smys,
                        safety_factor=self.safety_factor
                    )
                
                elif self.params.assessment_method == 'simplified_eff_area':
                    result = calculate_simplified_effective_area_method(
                        defect_depth_pct=effective_depth_pct,
                        defect_length_mm=effective_length_mm,
                        defect_width_mm=width_mm,
                        pipe_diameter_mm=self.pipe_diameter * 1000,
                        wall_thickness_mm=wt_mm,
                        smys_mpa=self.smys,
                        safety_factor=self.safety_factor
                    )
                
                # Calculate ERF: safe_pressure / maop
                safe_pressure = result['safe_pressure_mpa']
                erf = self.params.max_operating_pressure / safe_pressure
                max_erf = max(max_erf, erf)
                
            except Exception as e:
                print(f"Error calculating ERF for defect: {e}")
        
        return max_erf


    def _calculate_max_erf(self) -> float:
        """Calculate maximum ERF across all surviving defects."""
        
        # Create DataFrame for all current defects
        all_defects_data = []
        for defect in self.defect_states:
            all_defects_data.append({
                'depth [%]': defect.current_depth_pct,
                'length [mm]': defect.current_length_mm,
                'width [mm]': defect.current_width_mm,
                'wt nom [mm]': defect.wall_thickness_mm,
                'stress_concentration_factor': defect.stress_concentration_factor
            })
        
        if not all_defects_data:
            return 0.0
        
        all_defects_df = pd.DataFrame(all_defects_data)
        return self._calculate_erf_using_existing_system(all_defects_df)
    
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


# INTEGRATION HELPER: Function to connect with your existing corrosion assessment
def integrate_with_existing_corrosion_assessment(simulation_engine, your_corrosion_function):
    """
    Helper function to integrate the simulation with your existing corrosion assessment.
    
    Parameters:
    - simulation_engine: The FailurePredictionSimulator instance
    - your_corrosion_function: Your existing ERF calculation function
    
    You would modify _calculate_erf_using_existing_system to call your function.
    """
    
    # Replace the placeholder ERF calculation with your actual function
    def enhanced_erf_calculation(joint_df):
        # Call your existing corrosion assessment
        results = your_corrosion_function(
            joint_df, 
            simulation_engine.pipe_diameter,
            simulation_engine.smys,
            simulation_engine.safety_factor,
            method=simulation_engine.params.assessment_method
        )
        
        # Extract ERF values from your results
        return results['erf_values'].max()  # Adjust based on your function's output format
    
    # Replace the simulation engine's ERF calculation
    simulation_engine._calculate_erf_using_existing_system = enhanced_erf_calculation
    
    return simulation_engine