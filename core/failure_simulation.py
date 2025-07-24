import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from app.views.corrosion import calculate_b31g, calculate_modified_b31g, calculate_rstreng_effective_area_single, calculate_rstreng_effective_area_cluster
from core.standards_compliant_clustering import create_standards_compliant_clusterer

"""
Time-forward failure prediction simulation engine.
Uses existing corrosion assessment functionality.
"""

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
    erf_rst_ea: Optional[float] = None  


@dataclass
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
    

    original_defect_count: int = 1             # How many original defects this represents
    represents_cluster: bool = False           # Is this a combined defect?
    original_defect_indices: List[int] = None  # Which original defects were combined


class FailurePredictionSimulator:
    def __init__(self, params: SimulationParams, clustering_config: Optional[Dict] = None):
        self.params = params
        self.clustering_config = clustering_config  
        self.defect_states: List[DefectState] = []
        self.failure_history: List[DefectFailure] = []
        self.annual_results: List[Dict] = []
        
        # NEW: Store current year's clustering info
        self.current_clusters = []
        self.clusterer = None
        self.joints_df = None
        self.defect_id_to_index = {}
        
        # Initialize clusterer if needed
        if clustering_config and clustering_config['enabled']:
            self.clusterer = create_standards_compliant_clusterer(
                clustering_config['standard'],
                clustering_config['pipe_diameter_mm']
            )

        self.failed_joints: Set[int] = set()
        self.joint_failure_timeline: Dict[int, int] = {}  # year -> number of joints that failed that year

        # Add cluster profiles storage
        self.cluster_profiles: Dict[int, Dict] = {}  # {cluster_id: profile_data}

    
    def initialize_simulation(self, defects_df, joints_df, growth_rates_df, clusters, pipe_diameter, smys, safety_factor, use_clustering=True):
        """
        FIXED: Proper initialization with robust index mapping and validation.
        
        Critical fixes:
        1. Validates index alignment between defects and growth rates
        2. Uses explicit defect ID mapping instead of assuming index correspondence
        3. Provides detailed logging for debugging index issues
        """
        
        if use_clustering and clusters and self.clustering_config and self.clustering_config.get('enabled'):        
            # Use clustering if requested (existing logic)
            if hasattr(self, 'clusterer') and self.clusterer is not None:
                from core.enhanced_ffs_clustering import EnhancedFFSClusterer
                
                enhanced_clusterer = EnhancedFFSClusterer(
                    standard=self.clustering_config['standard'],  
                    pipe_diameter_mm=self.clustering_config['pipe_diameter_mm'],  
                    include_stress_concentration=True
                )
                
                combined_defects_df = enhanced_clusterer.create_combined_defects_dataframe(defects_df, clusters)
                print(f"✅ Using combined defects: {len(combined_defects_df)} total ({len(defects_df)} original)")
                defects_df = combined_defects_df
        
        # STEP 1: Filter valid defects FIRST and create robust ID mapping
        print("🔍 Filtering valid defects with robust ID tracking...")
        
        self.joints_df = joints_df
        original_count = len(defects_df)
        
        # CRITICAL: Preserve original index information
        defects_df_with_original_idx = defects_df.reset_index()
        if 'index' not in defects_df_with_original_idx.columns:
            defects_df_with_original_idx['original_index'] = defects_df_with_original_idx.index
        else:
            defects_df_with_original_idx['original_index'] = defects_df_with_original_idx['index']
        
        # Filter valid defects while preserving original index
        valid_mask = (
            (defects_df_with_original_idx['depth [%]'].notna()) & 
            (defects_df_with_original_idx['depth [%]'] > 0) & 
            (defects_df_with_original_idx['depth [%]'] <= 100) &
            (defects_df_with_original_idx['length [mm]'].notna()) & 
            (defects_df_with_original_idx['length [mm]'] > 0) &
            (defects_df_with_original_idx['width [mm]'].notna()) & 
            (defects_df_with_original_idx['width [mm]'] > 0)
        )
        
        valid_defects = defects_df_with_original_idx[valid_mask].copy()
        valid_count = len(valid_defects)
        
        print(f"✅ Filtered: {valid_count} valid defects (from {original_count})")
        
        if valid_count == 0:
            print("❌ No valid defects found!")
            return False
        
        # STEP 2: FIXED - Create robust index mapping with validation
        print("🔧 Creating robust index mapping...")
        
        # Reset index for valid defects to create clean sequential IDs
        valid_defects = valid_defects.reset_index(drop=True)
        valid_defects['simulation_id'] = range(len(valid_defects))
        
        # Create bidirectional mapping
        original_to_simulation = dict(zip(valid_defects['original_index'], valid_defects['simulation_id']))
        simulation_to_original = dict(zip(valid_defects['simulation_id'], valid_defects['original_index']))
        
        print(f"📋 Index mapping created: {len(original_to_simulation)} defects mapped")
        
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
        
        # STEP 5: CRITICAL FIX - Create growth rate lookup with proper ID resolution
        growth_lookup = {}
        if not growth_rates_df.empty:
            print("📈 Processing growth rates with robust ID matching...")
            
            # Method 1: Try to match using explicit ID columns if available
            id_columns_to_try = ['new_defect_id', 'defect_id', 'old_defect_id']
            
            # Check if growth_rates_df has explicit defect ID columns
            available_id_cols = [col for col in id_columns_to_try if col in growth_rates_df.columns]
            
            if available_id_cols:
                print(f"📋 Using explicit ID columns: {available_id_cols}")
                
                for idx, row in growth_rates_df.iterrows():
                    # Try to get defect ID from explicit columns
                    defect_id = None
                    for id_col in available_id_cols:
                        if pd.notna(row[id_col]):
                            defect_id = int(row[id_col])
                            break
                    
                    if defect_id is not None:
                        # Check if this original defect ID exists in our valid defects
                        if defect_id in original_to_simulation:
                            sim_id = original_to_simulation[defect_id]
                            growth_lookup[sim_id] = {
                                'depth_growth_pct': row.get('growth_rate_pct_per_year', 0.5),
                                'length_growth_mm': row.get('length_growth_rate_mm_per_year', 0.1),
                                'original_defect_id': defect_id,
                                'growth_source': f'matched_via_{id_col}'
                            }
                        else:
                            print(f"⚠️ Growth rate for defect ID {defect_id} - defect not in valid set")
            
            else:
                print("📋 No explicit ID columns found, using index-based matching")
                
                # Method 2: Index-based matching with validation
                # Assume growth_rates_df index corresponds to original defects_df index
                for growth_idx, row in growth_rates_df.iterrows():
                    # Check if this growth index corresponds to a valid defect
                    if growth_idx in original_to_simulation:
                        sim_id = original_to_simulation[growth_idx]
                        growth_lookup[sim_id] = {
                            'depth_growth_pct': row.get('growth_rate_pct_per_year', 0.5),
                            'length_growth_mm': row.get('length_growth_rate_mm_per_year', 0.1),
                            'original_defect_id': growth_idx,
                            'growth_source': 'index_based'
                        }
                    else:
                        print(f"⚠️ Growth rate at index {growth_idx} - no corresponding valid defect")
            
            print(f"📈 Growth rates mapped: {len(growth_lookup)} defects")
            
            # VALIDATION: Check mapping success rate
            mapping_rate = len(growth_lookup) / len(valid_defects) * 100
            if mapping_rate < 50:
                print(f"⚠️ WARNING: Low growth rate mapping success ({mapping_rate:.1f}%)")
                print("   This may indicate index misalignment issues")
            else:
                print(f"✅ Good growth rate mapping: {mapping_rate:.1f}% of defects have growth data")
        
        else:
            print("⚠️ No growth rate data provided - using defaults")
        

        # STEP 6: Process clustering with proper index mapping
        cluster_lookup = {}
        
        if use_clustering and clusters:
            print("🔗 Processing clustering data with index validation...")
            processed_defects = set()
            
            for cluster_idx, cluster in enumerate(clusters):
                try:
                    stress_factor = getattr(cluster, 'stress_concentration_factor', 1.0)
                    defect_indices = getattr(cluster, 'defect_indices', [])
                    
                    # Map cluster indices to simulation IDs
                    mapped_indices = []
                    for orig_idx in defect_indices:
                        if orig_idx in original_to_simulation:
                            sim_id = original_to_simulation[orig_idx]
                            mapped_indices.append(sim_id)
                            processed_defects.add(sim_id)
                            
                            cluster_lookup[sim_id] = {
                                'stress_factor': stress_factor,
                                'cluster_id': cluster_idx,
                                'is_clustered': True
                            }
                    
                    if mapped_indices:
                        print(f"  Cluster {cluster_idx}: {len(mapped_indices)} defects, stress={stress_factor:.2f}")
                    else:
                        print(f"  Cluster {cluster_idx}: No valid defects found")
                    
                except Exception as e:
                    print(f"⚠️ Error processing cluster {cluster_idx}: {e}")
                    continue
            
            # Add non-clustered defects
            for sim_id in range(len(valid_defects)):
                if sim_id not in processed_defects:
                    cluster_lookup[sim_id] = {
                        'stress_factor': 1.0,
                        'cluster_id': None,
                        'is_clustered': False
                    }
            
            print(f"🔗 Clustering complete: {len([v for v in cluster_lookup.values() if v['is_clustered']])} clustered")
        
        else:
            print("📍 No clustering - treating all defects as individual")
            for sim_id in range(len(valid_defects)):
                cluster_lookup[sim_id] = {
                    'stress_factor': 1.0,
                    'cluster_id': None,
                    'is_clustered': False
                }
        
        # STEP 7: Initialize defect states with validated data
        print("🏗️ Initializing defect states with validated mappings...")
        self.defect_states = []
        
        for sim_id, (_, defect) in enumerate(valid_defects.iterrows()):
            try:
                # Get growth data with fallback
                growth_data = growth_lookup.get(sim_id, {
                    'depth_growth_pct': 0.5,
                    'length_growth_mm': 0.1,
                    'growth_source': 'default'
                })
                
                # Get clustering data
                cluster_info = cluster_lookup.get(sim_id, {
                    'stress_factor': 1.0,
                    'cluster_id': None,
                    'is_clustered': False
                })
                
                # Get wall thickness
                wall_thickness = wt_lookup.get(defect['joint number'], 10.0)
                if pd.isna(wall_thickness) or wall_thickness <= 0:
                    wall_thickness = 10.0
                    print(f"⚠️ Using default wall thickness for joint {defect['joint number']}")
                
                # Create defect state with simulation ID
                defect_state = DefectState(
                    defect_id=sim_id,  # Use simulation ID for consistency
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
                    cluster_id=cluster_info['cluster_id'],
                    erf_rst_ea = defect['ERF RST EA']
                )
                
                self.defect_states.append(defect_state)
                
            except Exception as e:
                print(f"❌ Error creating defect state for simulation ID {sim_id}: {e}")
                continue
        
        # Create final ID mapping for simulation use
        self.defect_id_to_index = {
            defect.defect_id: idx 
            for idx, defect in enumerate(self.defect_states)
        }
        
        final_count = len(self.defect_states)
        print(f"✅ Initialization complete: {final_count} defect states created")
        
        # FINAL VALIDATION
        if final_count == 0:
            print("❌ No defect states created!")
            return False
        
        # Detailed validation report
        growth_data_count = sum(1 for s in self.defect_states if hasattr(s, 'growth_rate_pct_per_year'))
        clustered_count = sum(1 for s in self.defect_states if s.is_clustered)
        severe_count = sum(1 for s in self.defect_states if s.current_depth_pct > 80)
        
        print(f"📊 Validation Summary:")
        print(f"   - Total defects: {final_count}")
        print(f"   - With growth data: {growth_data_count}")
        print(f"   - Clustered: {clustered_count}")
        print(f"   - Severe (>80%): {severe_count}")
        
        # Store mapping for debugging
        self.index_mapping = {
            'original_to_simulation': original_to_simulation,
            'simulation_to_original': simulation_to_original,
            'growth_mapping_count': len(growth_lookup),
            'cluster_mapping_count': len([v for v in cluster_lookup.values() if v['is_clustered']])
        }
        return True
    

    def run_simulation(self) -> Dict:
        """Run simulation with proper failed defect & joint tracking."""
        self.failure_history = []
        self.annual_results = []
        failed_defects = set()

        # Initialize cluster profiles dictionary for each simulation
        self.cluster_profiles = {}

        # Initialize joint failure timeline
        for year in range(self.params.simulation_years + 1):
            self.joint_failure_timeline[year] = 0
        
        for year in range(self.params.simulation_years + 1):
            # Check initial failures (year 0) before any growth
            if year == 0:
                year_failures, failed_defects = self._check_defect_failures(year, failed_defects)
                self.failure_history.extend(year_failures) 
                self._track_joint_failures(year, year_failures)

            else:       
                # Grow defects BEFORE clustering, passing failed_defects
                self._grow_defects(year, failed_defects)

                # Apply dynamic clustering (if enabled)
                if self.clusterer is not None:
                    self._apply_dynamic_clustering(year, failed_defects)

                # Check for failures after growth
                year_failures, failed_defects = self._check_defect_failures(year, failed_defects)
                self.failure_history.extend(year_failures) 
                self._track_joint_failures(year, year_failures)
            
            # Calculate annual statistics
            annual_result = self._calculate_annual_stats(year, year_failures, failed_defects)
            self.annual_results.append(annual_result)
        return self._compile_results()
    

    def _generate_river_bottom_profile(self, cluster_defects: List[DefectState], cluster_id: int) -> Dict:
        """
        Generate river-bottom profile for a cluster of defects.
        
        Parameters:
        - cluster_defects: List of DefectState objects in the cluster
        - cluster_id: Unique cluster identifier
        
        Returns:
        - Dictionary with profile data ready for RSTRENG calculation
        """
        if len(cluster_defects) <= 1:
            # Single defect - no need for complex profile
            return None
        
        # Step 1: Calculate defect boundaries (using START positions)
        defect_boundaries = []
        for defect in cluster_defects:
            start_pos_mm = defect.location_m * 1000  
            end_pos_mm = start_pos_mm + defect.current_length_mm
            
            defect_boundaries.append({
                'defect_id': defect.defect_id,
                'start_mm': start_pos_mm,
                'end_mm': end_pos_mm,
                'depth_mm': defect.current_depth_pct * defect.wall_thickness_mm / 100.0  
            })
        
        # Step 2: Find overall cluster boundaries
        cluster_start_mm = min(d['start_mm'] for d in defect_boundaries)
        cluster_end_mm = max(d['end_mm'] for d in defect_boundaries)
        cluster_length_mm = cluster_end_mm - cluster_start_mm
        
        # Step 3: Generate depth profile (1mm increments)
        axial_step_mm = 1.0
        num_points = int(cluster_length_mm) + 1  # +1 to include end point
        depth_profile_mm = []
        
        for i in range(num_points):
            current_position_mm = cluster_start_mm + i * axial_step_mm
            
            # Find maximum depth at this position from all overlapping defects
            max_depth_at_position = 0.0
            
            for boundary in defect_boundaries:
                if boundary['start_mm'] <= current_position_mm <= boundary['end_mm']:
                    # This defect overlaps at current position
                    # Using option C: constant max depth across defect
                    max_depth_at_position = max(max_depth_at_position, boundary['depth_mm'])
            
            depth_profile_mm.append(max_depth_at_position)
        
        # Step 4: Create profile data structure
        profile_data = {
            'depth_profile_mm': depth_profile_mm,
            'axial_step_mm': axial_step_mm,
            'cluster_start_mm': cluster_start_mm,
            'cluster_end_mm': cluster_end_mm,
            'cluster_length_mm': cluster_length_mm,
            'defect_ids': [d.defect_id for d in cluster_defects],
            'num_defects': len(cluster_defects),
            'profile_generation_timestamp': f"Year_{getattr(self, 'current_simulation_year', 0)}"
        }
        
        # Store in simulator
        self.cluster_profiles[cluster_id] = profile_data
        
        return profile_data


    def _track_joint_failures(self, year: int, year_failures: List[DefectFailure]):
        """Track which joints failed this year (simple implementation)."""
        joints_failed_this_year = 0
        
        for failure in year_failures:
            joint_number = failure.joint_number
            if joint_number not in self.failed_joints:
                # This joint is failing for the first time
                self.failed_joints.add(joint_number)
                joints_failed_this_year += 1
        
        self.joint_failure_timeline[year] = joints_failed_this_year


    def _calculate_annual_stats(self, year: int, year_failures: List[DefectFailure], failed_defects: set) -> Dict:
        """Calculate statistics for the current year - now includes joint stats."""
        
        # Existing defect stats
        clustered_count = sum(1 for d in self.defect_states if d.is_clustered and d.defect_id not in failed_defects)
        total_active = len([d for d in self.defect_states if d.defect_id not in failed_defects])
        
        # NEW: Calculate total joints and surviving joints
        all_joints = set(d.joint_number for d in self.defect_states)
        total_joints = len(all_joints)
        surviving_joints = total_joints - len(self.failed_joints)
        
        # Safe calculations for max ERF and depths
        try:
            max_erf = self._calculate_max_erf()
            if max_erf is None or not isinstance(max_erf, (int, float)):
                max_erf = 0.0
        except:
            max_erf = 0.0
        
        try:
            valid_depths = [d.current_depth_pct for d in self.defect_states 
                        if d.defect_id not in failed_defects and 
                        d.current_depth_pct is not None and 
                        isinstance(d.current_depth_pct, (int, float))]
            max_depth = max(valid_depths) if valid_depths else 0.0
            avg_depth = np.mean(valid_depths) if valid_depths else 0.0
        except:
            max_depth = avg_depth = 0.0
        
        return {
            'year': year,
            # Defect-level stats (existing)
            'total_defects': len(self.defect_states),
            'active_defects': total_active,
            'failed_defects_this_year': len(year_failures),
            'cumulative_failed_defects': len(failed_defects),
            'surviving_defects': len(self.defect_states) - len(failed_defects),
            'clustered_defects': clustered_count,
            'max_erf': float(max_erf),
            'max_depth': float(max_depth),
            'avg_depth': float(avg_depth),
            
            # NEW: Joint-level stats
            'total_joints': total_joints,
            'failed_joints_this_year': self.joint_failure_timeline[year],
            'cumulative_failed_joints': len(self.failed_joints),
            'surviving_joints': surviving_joints,
            'joint_survival_rate': (surviving_joints / total_joints * 100) if total_joints > 0 else 100.0
        }


    def _apply_dynamic_clustering(self, year: int, failed_defects: set):
        """Apply clustering with river-bottom profile generation"""
        
        # Store current year for profile metadata
        self.current_simulation_year = year
        
        # OPTIMIZATION: Only cluster every 2 years for performance
        if year % 2 != 0 and year > 1:
            return
        
        # Get only active (non-failed) defects
        active_defects = [defect for defect in self.defect_states if defect.defect_id not in failed_defects]

        if len(active_defects) < 2:
            return
        
        # Create current DataFrame with grown dimensions
        current_df = pd.DataFrame({
            'log dist. [m]': [d.location_m for d in active_defects],
            'joint number': [d.joint_number for d in active_defects],
            'depth [%]': [d.current_depth_pct for d in active_defects],
            'length [mm]': [d.current_length_mm for d in active_defects],
            'width [mm]': [d.current_width_mm for d in active_defects],
            'defect_id': [d.defect_id for d in active_defects]
        })
        
        # Find new clusters with current dimensions
        cluster_indices = self.clusterer.find_interacting_defects(
            current_df, 
            self.joints_df,
            show_progress=False
        )
        
        # Reset all clustering info and clear old profiles
        for defect in self.defect_states:
            defect.is_clustered = False
            defect.cluster_id = None
            defect.stress_concentration_factor = 1.0
        
        # Clear old cluster profiles
        self.cluster_profiles.clear()
        
        # Apply new clustering and generate profiles
        clusters_with_profiles = 0
        
        for cluster_idx, defect_indices in enumerate(cluster_indices):
            if len(defect_indices) > 1:  # Real clusters only
                
                # Get defect objects for this cluster
                cluster_defects = []
                for idx in defect_indices:
                    defect_id = current_df.iloc[idx]['defect_id']
                    defect_index = self.defect_id_to_index[defect_id]
                    cluster_defects.append(self.defect_states[defect_index])
                
                # Calculate stress concentration for this cluster
                cluster_defects_df = current_df.iloc[defect_indices]
                stress_factor = self._calculate_cluster_stress_factor(cluster_defects_df)
                
                # NEW: Generate river-bottom profile for multi-defect cluster
                try:
                    profile_data = self._generate_river_bottom_profile(cluster_defects, cluster_idx)
                    if profile_data is not None:
                        clusters_with_profiles += 1
                        print(f"  Generated profile for cluster {cluster_idx}: "
                            f"{len(cluster_defects)} defects, "
                            f"{profile_data['cluster_length_mm']:.1f}mm span")
        
                except Exception as e:
                    print(f"  WARNING: Failed to generate profile for cluster {cluster_idx}: {e}")
                    # Continue without profile - will fall back to single-defect method
                
                # Apply clustering info to defects
                for defect in cluster_defects:
                    defect.is_clustered = True
                    defect.cluster_id = cluster_idx
                    defect.stress_concentration_factor = stress_factor
        
        
        # Summary logging
        total_clusters = len([c for c in cluster_indices if len(c) > 1])
        if total_clusters > 0:
            print(f"Year {year}: {total_clusters} clusters formed, "
                f"{clusters_with_profiles} with river-bottom profiles")


    def _calculate_cluster_stress_factor(self, cluster_defects: pd.DataFrame) -> float:
        """Industry-realistic stress concentration factors"""
        n_defects = len(cluster_defects)
        
        # Much more conservative factors based on industry practice
        if n_defects == 2:
            return 1.15 
        elif n_defects <= 5:
            return 1.25  
        else:
            return 1.35


    def calculate_stress_accelerated_growth(self, base_growth_rate, stress_concentration_factor, growth_type):
        """Calculate stress-accelerated growth - FIXED to return total growth, not multiplier."""
        if stress_concentration_factor <= 1.0:
            return base_growth_rate  # Return the rate itself, not 1.0
        
        if growth_type == 'depth':
            acceleration_factor = 1.0 + 0.2 * (stress_concentration_factor - 1.0) ** 0.5
            max_acceleration = 1.8
        elif growth_type == 'length':
            stress_intensity_ratio = stress_concentration_factor ** 0.5
            acceleration_factor = 1.0 + 0.15 * (stress_intensity_ratio - 1.0) ** 2.0
            max_acceleration = 2.0
        else:  # width
            acceleration_factor = 1.0 + 0.1 * (stress_concentration_factor - 1.0)
            max_acceleration = 1.5
        
        # Return the accelerated growth rate, not just the multiplier
        return base_growth_rate * min(acceleration_factor, max_acceleration)
        

    def _grow_defects(self, year: int, failed_defects: set):
        """Grow only active (non-failed) defects."""
        
        for defect in self.defect_states:
            # CRITICAL FIX: Skip failed defects
            if defect.defect_id in failed_defects:
                continue
                
            if defect.is_clustered:
                # Fix the acceleration calculation
                depth_acceleration = self.calculate_stress_accelerated_growth(
                    base_growth_rate=defect.growth_rate_pct_per_year,  # Use actual rate
                    stress_concentration_factor=defect.stress_concentration_factor,
                    growth_type='depth'
                )
                length_acceleration = self.calculate_stress_accelerated_growth(
                    base_growth_rate=defect.length_growth_rate_mm_per_year,  # Use actual rate
                    stress_concentration_factor=defect.stress_concentration_factor,
                    growth_type='length'
                )
            else:
                depth_acceleration = defect.growth_rate_pct_per_year
                length_acceleration = defect.length_growth_rate_mm_per_year
            
            # Apply growth (not double multiplication)
            defect.current_depth_pct += depth_acceleration
            defect.current_depth_pct = min(defect.current_depth_pct, 95.0)
            
            defect.current_length_mm += length_acceleration
    

    def _check_defect_failures(self, year: int, failed_defects: set) -> List[DefectFailure]:
        """Check failures with all required DefectFailure fields"""
    
        year_failures = []

        for defect in self.defect_states:
            if defect.defect_id in failed_defects:
                continue            
            
            if(defect.erf_rst_ea == None or year > 0):
                defect_erf = self._calculate_defect_erf(defect)
            elif( year == 0 ):
                defect_erf = defect.erf_rst_ea

            defect_depth = defect.current_depth_pct
            
            # Check failure criteria
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
                
                # ✅ Create DefectFailure with ALL required fields
                failure = DefectFailure(
                    defect_id=defect.defect_id,
                    joint_number=defect.joint_number,
                    failure_year=year,
                    failure_mode=failure_mode,
                    final_erf=defect_erf,
                    final_depth_pct=defect_depth,
                    location_m=defect.location_m,
                    was_clustered=defect.is_clustered,
                    stress_concentration_factor=defect.stress_concentration_factor,
                    # ✅ Additional tracking fields
                    original_defect_count=getattr(defect, 'original_defect_count', 1),
                    represents_cluster=getattr(defect, 'is_combined', False),
                    original_defect_indices=getattr(defect, 'original_defect_indices', None)
                )
                                
                year_failures.append(failure)
                failed_defects.add(defect.defect_id)

        return year_failures, failed_defects


    def _calculate_defect_erf(self, defect: DefectState) -> float:
        """
        Calculate ERF using cluster profile for clustered defects, single-defect method otherwise.
        """
        
        try:
            # Validation (unchanged)
            if defect.current_depth_pct >= 95.0:
                return 99.0
            
            if defect.current_length_mm <= 0 or defect.wall_thickness_mm <= 0:
                return 99.0
            
            # Check if this defect is in a cluster with a profile
            if defect.is_clustered and defect.cluster_id is not None:

                cluster_profile = self.cluster_profiles.get(defect.cluster_id)
                
                if cluster_profile is not None:
                    # Use cluster-based RSTRENG calculation
                    result = self._calculate_cluster_erf(defect, cluster_profile)
                    
                    # Apply stress concentration (existing logic)
                    if defect.stress_concentration_factor > 1.0:
                        base_erf = result
                        
                        if base_erf < 0.5:
                            stress_effect = 1.0 + (defect.stress_concentration_factor - 1.0) * 0.1
                        elif base_erf < 0.7:
                            stress_effect = 1.0 + (defect.stress_concentration_factor - 1.0) * 0.3
                        elif base_erf < 0.9:
                            stress_effect = 1.0 + (defect.stress_concentration_factor - 1.0) * 0.5
                        else:
                            stress_effect = defect.stress_concentration_factor
                        
                        final_erf = base_erf * stress_effect
                    else:
                        final_erf = result
                    
                    return min(max(float(final_erf), 0.0), 1.5)
            
            # Fall back to single-defect calculation (existing code)
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
                result = calculate_rstreng_effective_area_single(
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
            if safe_pressure > 0:
                return min(max(float(self.params.max_operating_pressure / safe_pressure), 0.0), 1.5)
            else:
                return 50.0
                
        except Exception as e:
            print(f"Error calculating ERF for defect {defect.defect_id}: {str(e)}")
            return 50.0
        


    def _calculate_cluster_erf(self, defect: DefectState, cluster_profile: Dict) -> float:
        """
        Calculate ERF using cluster river-bottom profile with RSTRENG method.
        
        Parameters:
        - defect: The defect (representative of cluster)
        - cluster_profile: Profile data dictionary
        
        Returns:
        - ERF value for the cluster
        """
        
        try:
            # Extract profile data
            depth_profile_mm = cluster_profile['depth_profile_mm']
            axial_step_mm = cluster_profile['axial_step_mm']
            
            # Use defect's wall thickness (should be consistent within joint)
            wall_thickness_mm = defect.wall_thickness_mm
            
            # Call the new RSTRENG cluster function
            result = calculate_rstreng_effective_area_cluster(
                depth_profile_mm=depth_profile_mm,
                axial_step_mm=axial_step_mm,
                pipe_diameter_mm=self.pipe_diameter * 1000,  # Convert to mm
                wall_thickness_mm=wall_thickness_mm,
                maop_mpa=self.params.max_operating_pressure,
                smys_mpa=self.smys,
                safety_factor=self.safety_factor,
                smts_mpa=None  # Optional parameter
            )
            
            # Check if calculation was successful
            if not result.get('safe', True):
                print(f"WARNING: Cluster RSTRENG calculation failed for cluster {defect.cluster_id}: {result.get('note', 'Unknown error')}")
                return 50.0  # Conservative fallback
            
            # Extract safe pressure and calculate ERF
            safe_pressure_mpa = result.get('safe_pressure_mpa', 0)
            
            if safe_pressure_mpa > 0:
                cluster_erf = self.params.max_operating_pressure / safe_pressure_mpa
                return cluster_erf
            else:
                print(f"WARNING: Zero safe pressure for cluster {defect.cluster_id}")
                return 50.0  # Conservative fallback
                
        except Exception as e:
            print(f"ERROR: Cluster ERF calculation failed for defect {defect.defect_id}, cluster {defect.cluster_id}: {str(e)}")
            return 50.0  # Conservative fallback

    def _calculate_max_erf(self) -> float:
        """Calculate maximum ERF across all surviving defects."""
        max_erf = 0.0
        
        for defect in self.defect_states:
            defect_erf = self._calculate_defect_erf(defect)
            max_erf = max(max_erf, defect_erf)
        
        return max_erf
    

    def _compile_results(self) -> Dict:
        """Compile final simulation results - now includes joint data."""
        
        # Calculate joint survival statistics
        all_joints = set(d.joint_number for d in self.defect_states)
        total_joints = len(all_joints)
        failed_joints_count = len(self.failed_joints)
        surviving_joints_count = total_joints - failed_joints_count
        
        return {
            'simulation_params': self.params,
            'annual_results': self.annual_results,
            'failure_history': self.failure_history,
            'total_failures': len(self.failure_history),
            'failure_timeline': self._create_failure_timeline(),
            'survival_statistics': self._calculate_survival_statistics(),
            
            # Joint-specific results
            'joint_failure_timeline': self.joint_failure_timeline,
            'joint_survival_statistics': {
                'total_joints': total_joints,
                'failed_joints': failed_joints_count,
                'surviving_joints': surviving_joints_count,
                'joint_failure_rate': (failed_joints_count / total_joints * 100) if total_joints > 0 else 0,
                'joint_survival_rate': (surviving_joints_count / total_joints * 100) if total_joints > 0 else 100
            }
        }

    
    def _create_failure_timeline(self) -> Dict[int, int]:
        """Create timeline of defect failures by year."""
        timeline = {}
        for year in range(self.params.simulation_years + 1):
            timeline[year] = len([f for f in self.failure_history if f.failure_year == year])
        return timeline
    

    def _calculate_survival_statistics(self) -> Dict:
        """Calculate statistics accounting for combined defects"""
        
        total_physical_defects = 0
        failed_physical_defects = 0
        
        for defect_state in self.defect_states:
            # Count physical defects (original count for combined defects)
            physical_count = getattr(defect_state, 'original_defect_count', 1)
            total_physical_defects += physical_count
        
        for failure in self.failure_history:
            physical_count = getattr(failure, 'original_defect_count', 1)
            failed_physical_defects += physical_count
        
        surviving_physical_defects = total_physical_defects - failed_physical_defects
        
        return {
            'total_defects': total_physical_defects,  # ✅ Physical defect count
            'failed_defects': failed_physical_defects,
            'surviving_defects': surviving_physical_defects,
            'failure_rate': (failed_physical_defects / total_physical_defects * 100) if total_physical_defects > 0 else 0,
            'simulation_used_combined_defects': any(getattr(d, 'is_combined', False) for d in self.defect_states)
        }