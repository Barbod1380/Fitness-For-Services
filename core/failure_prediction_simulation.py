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
        """
        FIXED: Proper initialization without index mismatches
        """
        
        # STEP 1: Filter valid defects FIRST and track index mapping
        print("🔍 Filtering valid defects...")
        
        original_count = len(defects_df)
        
        valid_defects = defects_df[
            (defects_df['depth [%]'].notna()) & 
            (defects_df['depth [%]'] > 0) & 
            (defects_df['depth [%]'] <= 100) &
            (defects_df['length [mm]'].notna()) & 
            (defects_df['length [mm]'] > 0) &
            (defects_df['width [mm]'].notna()) & 
            (defects_df['width [mm]'] > 0)
        ].copy()
        
        valid_count = len(valid_defects)
        
        print(f"✅ Filtered: {valid_count} valid defects (from {original_count})")
        
        if valid_count == 0:
            print("❌ No valid defects found!")
            return False
        
        # STEP 2: Create index mapping from original to filtered
        original_to_filtered = {}
        filtered_to_original = {}
        
        for new_idx, (original_idx, _) in enumerate(valid_defects.iterrows()):
            original_to_filtered[original_idx] = new_idx
            filtered_to_original[new_idx] = original_idx
        
        # STEP 3: Store pipe parameters
        try:
            self.pipe_diameter = pipe_diameter
            self.smys = smys
            self.safety_factor = safety_factor
            print(f"📏 Pipe specs: D={pipe_diameter:.3f}m, SMYS={smys}MPa, SF={safety_factor}")
        except Exception as e:
            print(f"❌ Error setting pipe parameters: {e}")
            return False
        
        # STEP 4: Create wall thickness lookup
        try:
            wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
            print(f"🔧 Wall thickness lookup: {len(wt_lookup)} joints")
        except Exception as e:
            print(f"❌ Error creating wall thickness lookup: {e}")
            return False
        
        # STEP 5: Create growth rate lookup with proper index mapping
        growth_lookup = {}
        if not growth_rates_df.empty:
            print("📈 Processing growth rates...")
            
            for idx, row in growth_rates_df.iterrows():
                # Try multiple ways to get the defect ID
                defect_id = None
                
                # Method 1: Direct ID columns
                for id_col in ['new_defect_id', 'defect_id', 'old_defect_id']:
                    if id_col in row and pd.notna(row[id_col]):
                        defect_id = int(row[id_col])
                        break
                
                # Method 2: Use index if no ID found
                if defect_id is None:
                    defect_id = idx
                
                # Only store growth data for valid defects
                if defect_id in original_to_filtered:
                    mapped_id = original_to_filtered[defect_id]
                    growth_lookup[mapped_id] = {
                        'depth_growth_pct': row.get('growth_rate_pct_per_year', 0.5),
                        'length_growth_mm': row.get('length_growth_rate_mm_per_year', 0.1)
                    }
            
            print(f"📈 Growth rates mapped: {len(growth_lookup)} defects")
        else:
            print("⚠️ No growth rate data provided - using defaults")
        
        # STEP 6: Process clustering with proper index mapping
        cluster_lookup = {}
        
        if use_clustering and clusters:
            print("🔗 Processing clustering data...")
            processed_defects = set()
            
            for cluster_idx, cluster in enumerate(clusters):
                try:
                    stress_factor = getattr(cluster, 'stress_concentration_factor', 1.0)
                    defect_indices = getattr(cluster, 'defect_indices', [])
                    
                    # Map original cluster indices to filtered indices
                    mapped_indices = []
                    for orig_idx in defect_indices:
                        if orig_idx in original_to_filtered:
                            mapped_idx = original_to_filtered[orig_idx]
                            mapped_indices.append(mapped_idx)
                            processed_defects.add(mapped_idx)
                            
                            cluster_lookup[mapped_idx] = {
                                'stress_factor': stress_factor,
                                'cluster_id': cluster_idx,
                                'is_clustered': True
                            }
                    
                    if mapped_indices:
                        print(f"  Cluster {cluster_idx}: {len(mapped_indices)} defects, stress={stress_factor:.2f}")
                    
                except Exception as e:
                    print(f"⚠️ Error processing cluster {cluster_idx}: {e}")
                    continue
            
            # Add non-clustered defects
            for filtered_idx in range(len(valid_defects)):
                if filtered_idx not in processed_defects:
                    cluster_lookup[filtered_idx] = {
                        'stress_factor': 1.0,
                        'cluster_id': None,
                        'is_clustered': False
                    }
            
            print(f"🔗 Clustering complete: {len([v for v in cluster_lookup.values() if v['is_clustered']])} clustered, {len([v for v in cluster_lookup.values() if not v['is_clustered']])} individual")
        
        else:
            print("📍 No clustering - treating all defects as individual")
            # All defects individual
            for filtered_idx in range(len(valid_defects)):
                cluster_lookup[filtered_idx] = {
                    'stress_factor': 1.0,
                    'cluster_id': None,
                    'is_clustered': False
                }
        
        # STEP 7: Initialize defect states using FILTERED data
        print("🏗️ Initializing defect states...")
        self.defect_states = []
        
        for filtered_idx, (original_idx, defect) in enumerate(valid_defects.iterrows()):
            try:
                # Get growth data (use defaults if not found)
                growth_data = growth_lookup.get(filtered_idx, {
                    'depth_growth_pct': 0.5,  # 0.5% per year default
                    'length_growth_mm': 0.1   # 0.1 mm per year default
                })
                
                # Get clustering data
                cluster_info = cluster_lookup.get(filtered_idx, {
                    'stress_factor': 1.0,
                    'cluster_id': None,
                    'is_clustered': False
                })
                
                # Get wall thickness
                wall_thickness = wt_lookup.get(defect['joint number'], 10.0)
                if pd.isna(wall_thickness) or wall_thickness <= 0:
                    wall_thickness = 10.0
                    print(f"⚠️ Using default wall thickness for joint {defect['joint number']}")
                
                # Create defect state
                defect_state = DefectState(
                    defect_id=filtered_idx,  # Use filtered index as ID
                    joint_number=defect['joint number'],
                    current_depth_pct=float(defect['depth [%]']),
                    current_length_mm=float(defect['length [mm]']),
                    current_width_mm=float(defect['width [mm]']),
                    location_m=float(defect['log dist. [m]']),
                    growth_rate_pct_per_year=float(growth_data['depth_growth_pct']),
                    length_growth_rate_mm_per_year=float(growth_data['length_growth_mm']),
                    wall_thickness_mm=float(wall_thickness),
                    stress_concentration_factor=float(cluster_info['stress_factor']),
                    is_clustered=bool(cluster_info['is_clustered']),
                    cluster_id=cluster_info['cluster_id']
                )
                
                self.defect_states.append(defect_state)
                
            except Exception as e:
                print(f"❌ Error creating defect state for index {filtered_idx}: {e}")
                continue
        
        final_count = len(self.defect_states)
        print(f"✅ Initialization complete: {final_count} defect states created")
        
        # STEP 8: Validation
        if final_count == 0:
            print("❌ No defect states created!")
            return False
        
        # Quick sanity check
        severe_defects = sum(1 for ds in self.defect_states if ds.current_depth_pct > 80)
        clustered_defects = sum(1 for ds in self.defect_states if ds.is_clustered)
        
        print(f"📊 Validation: {severe_defects} severe defects (>80%), {clustered_defects} clustered")
        
        if severe_defects > final_count * 0.5:
            print("⚠️ WARNING: >50% of defects are severe (>80% depth)")
        
        return True
    

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
        
        if(year == 0):
            print("YEAR 0 Analysis")

        year_failures = []
        counter = 0

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
                counter += 1
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
        
        if( year == 0 ):
            print("COUNTER", counter)
            for fail in self.failure_history:
                print(fail)
                print('---------------')
        return year_failures
    

    def _calculate_defect_erf(self, defect: DefectState) -> float:
        """
        FIXED: Calculate ERF with proper stress concentration application.
        Addresses immediate failures from overly aggressive stress factors.
        """
        
        try:
            # Validation (unchanged)
            if defect.current_depth_pct >= 95.0:
                return 99.0
            
            if defect.current_length_mm <= 0 or defect.wall_thickness_mm <= 0:
                return 99.0
            
            # Calculate base assessment (unchanged)
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
            
            safe_pressure = result.get('safe_pressure_mpa', 0)
            failure_pressure = result.get('failure_pressure_mpa', 0)
            
            if safe_pressure > 0:
                base_erf = self.params.max_operating_pressure / safe_pressure
                
                # FIXED: Apply stress concentration more appropriately
                # Only apply to defects that are actually close to failure
                if base_erf > 0.7:  # Only apply to defects approaching limits
                    # Apply stress concentration factor
                    stress_multiplier = defect.stress_concentration_factor
                    
                    # FIXED: Use graduated approach instead of direct multiplication
                    if defect.is_clustered:
                        # For clustered defects, apply stress concentration more gradually
                        stress_effect = 1.0 + (stress_multiplier - 1.0) * 0.6  # Reduce impact by 40%
                        final_erf = base_erf * stress_effect
                    else:
                        # Individual defects - no stress concentration
                        final_erf = base_erf
                else:
                    # FIXED: For defects well below failure threshold, 
                    # stress concentration has minimal effect
                    stress_effect = 1.0 + (defect.stress_concentration_factor - 1.0) * 0.2  # Only 20% effect
                    final_erf = base_erf * stress_effect
                
                final_erf = min(final_erf, 99.0)
                return final_erf
                
            elif failure_pressure > 0:
                erf_from_failure = self.params.max_operating_pressure / (failure_pressure / self.safety_factor)
                erf_from_failure *= min(defect.stress_concentration_factor, 2.0)  # Cap stress effect
                return min(erf_from_failure, 99.0)
            else:
                return 50.0
                
        except Exception as e:
            print(f"Error calculating ERF for defect {defect.defect_id}: {str(e)}")
            return 50.0


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