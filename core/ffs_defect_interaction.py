# Create new file: core/ffs_defect_interaction.py
"""
Fitness-For-Service (FFS) compliant defect interaction analysis.
Based on ASME B31G-2012 and API 579-1/ASME FFS-1 interaction rules.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
import warnings

class FFSDefectInteraction:
    """
    Implements defect interaction rules per FFS standards.
    """
    
    def __init__(self, 
                 axial_interaction_distance_mm: float = 25.4,  # 1 inch default
                 circumferential_interaction_method: str = 'sqrt_dt',
                 custom_circ_distance_mm: float = None):
        """
        Initialize FFS defect interaction analyzer.
        
        Parameters:
        - axial_interaction_distance_mm: Maximum axial spacing for interaction (default 1")
        - circumferential_interaction_method: Method for circumferential interaction
          Options: 'sqrt_dt' (√(D×t)), '3t' (3×wall thickness), 'custom'
        - custom_circ_distance_mm: Custom circumferential interaction distance if method='custom'
        """
        self.axial_interaction_distance = axial_interaction_distance_mm
        self.circ_method = circumferential_interaction_method
        self.custom_circ_distance = custom_circ_distance_mm
        
        if self.circ_method == 'custom' and custom_circ_distance_mm is None:
            raise ValueError("Must specify custom_circ_distance_mm when using 'custom' method")
    
    def calculate_circumferential_interaction_distance(self, 
                                                     pipe_diameter_mm: float, 
                                                     wall_thickness_mm: float) -> float:
        """
        Calculate circumferential interaction distance based on selected method.
        """
        if self.circ_method == 'sqrt_dt':
            # Common rule: √(D×t)
            return np.sqrt(pipe_diameter_mm * wall_thickness_mm)
        elif self.circ_method == '3t':
            # Alternative rule: 3×wall thickness
            return 3.0 * wall_thickness_mm
        elif self.circ_method == 'custom':
            return self.custom_circ_distance
        else:
            raise ValueError(f"Unknown circumferential interaction method: {self.circ_method}")
        
    # Parse clock position string (e.g., '8:43' -> 8.717 hours)
    def parse_clock_to_decimal_hours(self, clock_str):
        """Convert clock string 'H:MM' or 'HH:MM' to decimal hours."""
        try:
            parts = clock_str.strip().split(':')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            
            # Convert to decimal hours
            decimal_hours = hours + minutes / 60.0
            
            # Validate range (1:00 to 12:59)
            if decimal_hours < 1.0 or decimal_hours >= 13.0:
                warnings.warn(f"Clock position {clock_str} outside expected range 1:00-12:59")
            
            return decimal_hours
        except (ValueError, IndexError) as e:
            warnings.warn(f"Invalid clock format '{clock_str}': {e}")
            return 12.0  # Default to 12:00 if parsing fails
    
    def convert_to_cartesian(self, defects_df: pd.DataFrame, joints_df: pd.DataFrame, 
                            pipe_diameter_mm: float) -> pd.DataFrame:
        """
        Convert defect positions to cartesian coordinates for interaction analysis.
        
        Adds columns: x_start, x_end, y_center (circumferential position)
        """
        df = defects_df.copy()
        
        # Ensure we have required columns
        required_cols = ['log dist. [m]', 'length [mm]', 'clock', 'joint number']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert to consistent units (mm)
        df['x_center_mm'] = df['log dist. [m]'] * 1000  # Convert m to mm
        df['x_start_mm'] = df['x_center_mm'] - df['length [mm]'] / 2
        df['x_end_mm'] = df['x_center_mm'] + df['length [mm]'] / 2
        
        # Convert clock strings to decimal hours
        df['decimal_hours'] = df['clock'].apply(self.parse_clock_to_decimal_hours)
        
        # Convert to angular position (degrees)
        # 12:00 = 0°, 3:00 = 90°, 6:00 = 180°, 9:00 = 270°
        df['angle_deg'] = (df['decimal_hours'] % 12) * 30
        
        # Calculate circumferential position (y) in mm
        # Using arc length = radius × angle (in radians)
        radius_mm = pipe_diameter_mm / 2
        df['y_center_mm'] = radius_mm * np.radians(df['angle_deg'])
        
        # Add circumferential extent (width)
        if 'width [mm]' in df.columns:
            df['y_start_mm'] = df['y_center_mm'] - df['width [mm]'] / 2
            df['y_end_mm'] = df['y_center_mm'] + df['width [mm]'] / 2
        else:
            # If no width data, treat as point defects circumferentially
            df['y_start_mm'] = df['y_center_mm']
            df['y_end_mm'] = df['y_center_mm']
            df['width [mm]'] = 0
        
        return df        

    def find_interacting_defects(self, defects_df: pd.DataFrame, joints_df: pd.DataFrame,
                            pipe_diameter_mm: float) -> List[List[int]]:
        """
        Find groups of interacting defects according to FFS rules.
        OPTIMIZED: Uses sorted nature of defects to skip distant comparisons.
        
        Returns:
        - List of defect groups, where each group is a list of defect indices
        """
        # Convert to cartesian coordinates
        df = self.convert_to_cartesian(defects_df, joints_df, pipe_diameter_mm)
        
        # Get wall thickness lookup
        wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
        
        # Initialize Union-Find structure for grouping
        n_defects = len(df)
        parent = list(range(n_defects))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # OPTIMIZATION: Pre-calculate the maximum possible interaction distance
        # This is the axial distance + maximum possible defect length
        max_defect_length = df['length [mm]'].max() if not df.empty else 0
        max_axial_search_distance = self.axial_interaction_distance + max_defect_length
        
        # Track statistics for debugging/info
        comparisons_made = 0
        comparisons_skipped = 0
        
        # Check pairs of defects for interaction - OPTIMIZED VERSION
        for i in range(n_defects):
            defect_i = df.iloc[i]
            
            # Only check defects ahead in the sorted list
            for j in range(i + 1, n_defects):
                defect_j = df.iloc[j]
                
                # OPTIMIZATION: Calculate axial distance between defect centers
                axial_center_distance = abs(defect_j['x_center_mm'] - defect_i['x_center_mm'])
                
                # If defects are too far apart axially (even considering their lengths),
                # all subsequent defects will also be too far
                if axial_center_distance > max_axial_search_distance:
                    comparisons_skipped += (n_defects - j)  # Count skipped comparisons
                    break  # Skip all remaining defects for this i
                
                comparisons_made += 1
                
                # Check if defects actually interact
                if self._defects_interact(defect_i, defect_j, pipe_diameter_mm, wt_lookup):
                    union(i, j)
        
        # Log optimization statistics (optional)
        total_possible = n_defects * (n_defects - 1) // 2
        print(f"Defect interaction optimization: {comparisons_made} comparisons made, "
            f"{comparisons_skipped} skipped (out of {total_possible} possible)")

        # Group defects by their root parent
        groups = {}
        for i in range(n_defects):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Return groups with more than one defect, plus individual defects
        interacting_groups = [group for group in groups.values() if len(group) > 1]
        single_defects = [[i] for group in groups.values() if len(group) == 1 for i in group]
        
        return interacting_groups + single_defects
    

    def _defects_interact(self, defect1: pd.Series, defect2: pd.Series, pipe_diameter_mm: float, wt_lookup: Dict) -> bool:
        """
        Check if two defects interact according to FFS rules.
        """
        # Get wall thickness for circumferential interaction calculation
        # Use the minimum wall thickness of the two joints (conservative)
        wt1 = wt_lookup.get(defect1['joint number'], None)
        wt2 = wt_lookup.get(defect2['joint number'], None)
        
        if wt1 is None or wt2 is None:
            warnings.warn(f"Missing wall thickness for defect interaction check")
            return False
        
        wall_thickness_mm = min(wt1, wt2)
        
        # Calculate interaction distances
        axial_limit = self.axial_interaction_distance
        circ_limit = self.calculate_circumferential_interaction_distance(
            pipe_diameter_mm, wall_thickness_mm
        )
        
        # Check axial separation
        axial_gap = max(defect1['x_start_mm'], defect2['x_start_mm']) - \
                   min(defect1['x_end_mm'], defect2['x_end_mm'])
        
        if axial_gap > axial_limit:
            return False  # Too far apart axially
        
        # Check circumferential separation
        # Need to account for wrap-around at 360 degrees
        circ_gap = self._calculate_circumferential_gap(
            defect1['y_start_mm'], defect1['y_end_mm'],
            defect2['y_start_mm'], defect2['y_end_mm'],
            pipe_diameter_mm
        )
        
        if circ_gap > circ_limit:
            return False  # Too far apart circumferentially
        
        return True  # Defects interact
    

    def _calculate_circumferential_gap(self, y1_start, y1_end, y2_start, y2_end, 
                                     pipe_diameter_mm):
        """
        Calculate minimum circumferential gap between two defects.
        Accounts for wrap-around at 360 degrees.
        """
        # Calculate circumference
        circumference = np.pi * pipe_diameter_mm
        
        # Simple gap calculation (can be enhanced for wrap-around)
        gap = max(y1_start, y2_start) - min(y1_end, y2_end)
        
        # Check wrap-around gap
        wrap_gap = circumference - max(y1_end, y2_end) + min(y1_start, y2_start)
        
        return min(max(0, gap), max(0, wrap_gap))
    
    def combine_interacting_defects(self, defects_df: pd.DataFrame, joints_df: pd.DataFrame, pipe_diameter_mm: float) -> pd.DataFrame:
        """
        Combine interacting defects according to FFS rules.
        
        Returns new dataframe with combined defects.
        """
        # Find interacting groups
        groups = self.find_interacting_defects(defects_df, joints_df, pipe_diameter_mm)
        combined_defects = []

        for group in groups:
            if len(group) == 1:
                # Single defect - keep as is
                combined_defects.append(defects_df.iloc[group[0]].to_dict())
            else:
                # Multiple interacting defects - combine them
                group_defects = defects_df.iloc[group]
                
                decimal_hours = group_defects['clock'].apply(self.parse_clock_to_decimal_hours)
                mean_decimal_hours = decimal_hours.mean()
            
                # Convert back to clock format (optional - you could keep decimal)
                mean_hours = int(mean_decimal_hours)
                mean_minutes = int((mean_decimal_hours - mean_hours) * 60)
                mean_clock_str = f"{mean_hours}:{mean_minutes:02d}"

                combined = {
                    # Take maximum depth (most conservative)
                    'depth [%]': group_defects['depth [%]'].max(),
                    
                    # Calculate total extent
                    'log dist. [m]': group_defects['log dist. [m]'].mean(),  # Center of combined defect
                    'length [mm]': (group_defects['log dist. [m]'].max() * 1000 + 
                                   group_defects['length [mm]'].iloc[group_defects['log dist. [m]'].argmax()] / 2) - \
                                  (group_defects['log dist. [m]'].min() * 1000 - 
                                   group_defects['length [mm]'].iloc[group_defects['log dist. [m]'].argmin()] / 2),
                    
                    # For width, take the maximum circumferential extent
                    'width [mm]': self._calculate_combined_width(group_defects, pipe_diameter_mm),
                    
                    # Preserve other important fields
                    'joint number': group_defects['joint number'].iloc[0],  # Assuming same joint
                    'clock': mean_clock_str,
                    
                    # Metadata about combination
                    'is_combined': True,
                    'num_original_defects': len(group),
                    'original_indices': list(group),
                    'combination_note': f"Combined {len(group)} interacting defects per FFS rules"
                }

                # Preserve other columns from the first defect
                first_defect = defects_df.iloc[group[0]]
                for col in defects_df.columns:
                    if col not in combined:
                        combined[col] = first_defect[col]
                
                combined_defects.append(combined)
        
        result_df = pd.DataFrame(combined_defects)
        
        # Add analysis metadata
        result_df['ffs_interaction_analysis'] = True
        result_df['axial_interaction_distance_mm'] = self.axial_interaction_distance
        result_df['circ_interaction_method'] = self.circ_method
        
        return result_df
    
    def _calculate_combined_width(self, group_defects: pd.DataFrame, pipe_diameter_mm: float) -> float:
        """
        Calculate combined circumferential width of interacting defects.
        """
        if 'width [mm]' not in group_defects.columns:
            return 0
        
        group_defects['decimal_hours'] = group_defects['clock'].apply(self.parse_clock_to_decimal_hours)
        
        # Convert to angular position (degrees)
        # 12:00 = 0°, 3:00 = 90°, 6:00 = 180°, 9:00 = 270°
        angles = (group_defects['decimal_hours'] % 12) * 30
        
        # Simple approach - can be enhanced for better accuracy
        min_angle = angles.min()
        max_angle = angles.max()
        
        # Account for individual defect widths
        radius = pipe_diameter_mm / 2
        
        # This is simplified - a more complex calculation would account for
        # exact circumferential positions and overlaps
        angle_extent = max_angle - min_angle
        arc_length = radius * np.radians(angle_extent)
        
        # Add the maximum width to account for defect size
        return arc_length + group_defects['width [mm]'].max()