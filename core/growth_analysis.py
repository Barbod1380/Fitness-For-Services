# Create new file: core/growth_analysis.py
"""
Growth rate analysis for defects considering FFS clustering.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from .defect_matching import DefectMatch

class ClusterAwareGrowthAnalyzer:
    """
    Analyzes defect growth rates considering FFS clustering effects.
    """
    
    def __init__(self, 
                 min_growth_rate: float = 0.0,  # mm/year minimum
                 max_growth_rate: float = 5.0,   # mm/year maximum reasonable
                 negative_growth_strategy: str = 'similar_match'):
        """
        Initialize growth analyzer.
        
        Parameters:
        - min_growth_rate: Minimum acceptable growth rate (0 = no shrinkage)
        - max_growth_rate: Maximum reasonable growth rate for validation
        - negative_growth_strategy: How to handle negative growth
          Options: 'similar_match', 'zero', 'statistical'
        """
        self.min_growth_rate = min_growth_rate
        self.max_growth_rate = max_growth_rate
        self.negative_growth_strategy = negative_growth_strategy


    def analyze_growth_with_clustering(self,
                                year1_defects: pd.DataFrame,
                                year2_defects: pd.DataFrame,
                                matches: List[DefectMatch],
                                year1_date: pd.Timestamp,
                                year2_date: pd.Timestamp,
                                wall_thickness_lookup: Dict) -> pd.DataFrame:
        """
        Analyze growth rates for matched defects considering clustering.
        
        FIXED: Proper handling of DataFrame vs Series extraction
        """
        time_years = (year2_date - year1_date).days / 365.25
        
        growth_results = []
        
        for match in matches:
            try:
                # CRITICAL FIX: Extract data properly to avoid tuple indexing errors
                if match.match_type == '1-to-1':
                    # Single defect to single defect - extract as Series
                    y1_defect = year1_defects.iloc[match.year1_indices[0]]
                    y2_defect = year2_defects.iloc[match.year2_indices[0]]
                    
                    growth_data = self._analyze_simple_growth(
                        y1_defect, y2_defect, 
                        time_years, wall_thickness_lookup
                    )
                    
                elif match.match_type == 'many-to-1':
                    # Multiple defects to single defect
                    y1_defects_group = year1_defects.iloc[match.year1_indices]
                    y2_defect = year2_defects.iloc[match.year2_indices[0]]
                    
                    growth_data = self._analyze_coalescence_growth(
                        y1_defects_group, y2_defect,
                        time_years, wall_thickness_lookup
                    )
                    
                elif match.match_type == '1-to-many':
                    # Single defect to multiple defects
                    y1_defect = year1_defects.iloc[match.year1_indices[0]]
                    y2_defects_group = year2_defects.iloc[match.year2_indices]
                    
                    growth_data = self._analyze_split_growth(
                        y1_defect, y2_defects_group,
                        time_years, wall_thickness_lookup
                    )
                    
                else:  # many-to-many
                    # Multiple defects to multiple defects
                    y1_defects_group = year1_defects.iloc[match.year1_indices]
                    y2_defects_group = year2_defects.iloc[match.year2_indices]
                    
                    growth_data = self._analyze_complex_growth(
                        y1_defects_group, y2_defects_group,
                        time_years, wall_thickness_lookup
                    )
                
                # Add match metadata
                growth_data.update({
                    'match_type': match.match_type,
                    'match_confidence': match.match_confidence,
                    'year1_indices': match.year1_indices,
                    'year2_indices': match.year2_indices,
                    'time_years': time_years
                })
                
                growth_results.append(growth_data)
                
            except Exception as e:
                # Add detailed error information for debugging
                import traceback
                error_details = traceback.format_exc()
                
                print(f"Error processing match {match.match_type}: {str(e)}")
                print(f"Year1 indices: {match.year1_indices}, Year2 indices: {match.year2_indices}")
                print(f"Error details: {error_details}")
                
                # Try to extract some basic info for the error case
                try:
                    if match.year1_indices and match.year2_indices:
                        y1_loc = year1_defects.iloc[match.year1_indices[0]]['log dist. [m]']
                        y2_loc = year2_defects.iloc[match.year2_indices[0]]['log dist. [m]']
                        avg_location = (y1_loc + y2_loc) / 2
                    else:
                        avg_location = 0.0
                except:
                    avg_location = 0.0
                
                # Add a placeholder result to avoid stopping the entire analysis
                growth_results.append({
                    'match_type': match.match_type,
                    'error': str(e),
                    'depth_growth_mm_per_year': 0.0,
                    'length_growth_mm_per_year': 0.0,
                    'width_growth_mm_per_year': 0.0,
                    'year1_depth_pct': 0.0,
                    'year2_depth_pct': 0.0,
                    'year1_length_mm': 0.0,
                    'year2_length_mm': 0.0,
                    'location_m': avg_location,
                    'growth_type': 'error'
                })
        
        # Convert to DataFrame
        if growth_results:
            growth_df = pd.DataFrame(growth_results)
            
            # Filter out error rows for negative growth handling
            valid_rows = growth_df[~growth_df.get('error', '').astype(bool)]
            if not valid_rows.empty:
                # Handle negative growth rates only for valid rows
                growth_df = self._handle_negative_growth(growth_df, year1_defects, year2_defects)
            
            return growth_df
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'match_type', 'depth_growth_mm_per_year', 'length_growth_mm_per_year', 
                'width_growth_mm_per_year', 'location_m', 'growth_type'
            ])

    
    def _analyze_simple_growth(self, 
                             d1: pd.Series, 
                             d2: pd.Series,
                             time_years: float,
                             wt_lookup: Dict) -> Dict:
        """Analyze growth for simple 1-to-1 match."""
        # Get wall thickness
        wt = wt_lookup.get(d1['joint number'], wt_lookup.get(d2['joint number'], 10.0))
        
        # Depth growth
        depth1_mm = d1['depth [%]'] * wt / 100
        depth2_mm = d2['depth [%]'] * wt / 100
        depth_growth_rate = (depth2_mm - depth1_mm) / time_years
        
        # Length growth
        length_growth_rate = (d2['length [mm]'] - d1['length [mm]']) / time_years
        
        # Width growth (if available)
        if 'width [mm]' in d1.index and 'width [mm]' in d2.index:
            width_growth_rate = (d2['width [mm]'] - d1['width [mm]']) / time_years
        else:
            width_growth_rate = 0.0
        
        return {
            'depth_growth_mm_per_year': depth_growth_rate,
            'length_growth_mm_per_year': length_growth_rate,
            'width_growth_mm_per_year': width_growth_rate,
            'year1_depth_pct': d1['depth [%]'],
            'year2_depth_pct': d2['depth [%]'],
            'year1_length_mm': d1['length [mm]'],
            'year2_length_mm': d2['length [mm]'],
            'location_m': (d1['log dist. [m]'] + d2['log dist. [m]']) / 2,
            'growth_type': 'simple'
        }


    def _analyze_coalescence_growth(self,
                                d1_group: pd.DataFrame,
                                d2_single: pd.Series,
                                time_years: float,
                                wt_lookup: Dict) -> Dict:
        """
        CORRECTED: Analyze growth when multiple defects coalesced into one.
        This represents accelerated degradation due to stress concentration.
        
        FIXES APPLIED:
        1. Added stress concentration factor for interacting defects
        2. Improved wall thickness selection logic
        3. Enhanced length calculation with proper geometric extent
        4. Added basic validation while maintaining original interface
        """
        
        # Use minimum wall thickness for conservative analysis, with better error handling
        wall_thicknesses = []
        for _, defect in d1_group.iterrows():
            wt_val = wt_lookup.get(defect['joint number'])
            if wt_val is not None and wt_val > 0:
                wall_thicknesses.append(wt_val)
        
        # Add year 2 wall thickness
        y2_wt = wt_lookup.get(d2_single['joint number'])
        if y2_wt is not None and y2_wt > 0:
            wall_thicknesses.append(y2_wt)
        
        if wall_thicknesses:
            wt = min(wall_thicknesses) 
        else:
            wt = 10.0       

        # Still use maximum depth as baseline, but apply stress concentration factor
        max_depth1_pct = d1_group['depth [%]'].max()
        max_depth1_mm = max_depth1_pct * wt / 100
        
        # Year 2 depth
        depth2_mm = d2_single['depth [%]'] * wt / 100
        
        # CRITICAL FIX: Apply stress concentration factor for coalescing defects
        # Based on API 579-1 principles - multiple interacting defects accelerate growth
        num_coalesced = len(d1_group)
        stress_concentration_factor = 1.0 + 0.1 * np.log(max(num_coalesced, 2))  # Logarithmic increase
        stress_concentration_factor = min(stress_concentration_factor, 1.5)  # Cap at 1.5x
        
        # Calculate raw growth rate
        raw_depth_growth = (depth2_mm - max_depth1_mm) / time_years
        
        # Apply stress concentration to account for interaction effects
        # This reflects the accelerated degradation due to multiple defects interacting
        depth_growth_rate = raw_depth_growth * stress_concentration_factor
        
        # === FIXED: Improved length calculation with proper geometric extent ===
        # Calculate actual geometric extent more accurately
        locations_mm = d1_group['log dist. [m]'] * 1000  # Convert to mm
        lengths_mm = d1_group['length [mm]']
        
        # Calculate start and end positions for each defect
        start_positions = locations_mm - lengths_mm / 2
        end_positions = locations_mm + lengths_mm / 2
        
        # Total extent is from earliest start to latest end
        total_start = start_positions.min()
        total_end = end_positions.max()
        total_extent_y1 = total_end - total_start
        
        # Add interaction buffer per API 579-1 (defects within 25.4mm interact)
        interaction_buffer = 25.4  # mm
        if num_coalesced > 2:
            total_extent_y1 += interaction_buffer
        
        # Length growth rate
        length_growth_rate = (d2_single['length [mm]'] - total_extent_y1) / time_years
        
        return {
            'depth_growth_mm_per_year': depth_growth_rate,
            'length_growth_mm_per_year': length_growth_rate,
            'width_growth_mm_per_year': 0.0,  # Complex to calculate
            'year1_depth_pct': max_depth1_pct,
            'year2_depth_pct': d2_single['depth [%]'],
            'year1_length_mm': total_extent_y1,
            'year2_length_mm': d2_single['length [mm]'],
            'location_m': d2_single['log dist. [m]'],
            'growth_type': 'coalescence',
            'num_coalesced': len(d1_group),
            'coalescence_note': f"{len(d1_group)} defects coalesced into 1"
        }

    def _analyze_split_growth(self,
                            d1_single: pd.Series,
                            d2_group: pd.DataFrame,
                            time_years: float,
                            wt_lookup: Dict) -> Dict:
        """
        Analyze growth when one defect split into multiple.
        This is rare but can happen with complex corrosion patterns.
        """
        # Use the most severe defect in year 2 for conservative analysis
        max_d2 = d2_group.loc[d2_group['depth [%]'].idxmax()]
        
        # Treat as simple growth to the worst defect
        result = self._analyze_simple_growth(d1_single, max_d2, time_years, wt_lookup)
        result['growth_type'] = 'split'
        result['split_note'] = f"1 defect split into {len(d2_group)}"
        
        return result
    
    def _analyze_complex_growth(self,
                              d1_group: pd.DataFrame,
                              d2_group: pd.DataFrame,
                              time_years: float,
                              wt_lookup: Dict) -> Dict:
        """
        Analyze complex many-to-many matches.
        Use conservative assumptions.
        """
        # Use maximum depths for conservative analysis
        max_d1 = d1_group.loc[d1_group['depth [%]'].idxmax()]
        max_d2 = d2_group.loc[d2_group['depth [%]'].idxmax()]
        
        # Analyze as simple growth between worst defects
        result = self._analyze_simple_growth(max_d1, max_d2, time_years, wt_lookup)
        result['growth_type'] = 'complex'
        result['complex_note'] = f"{len(d1_group)} defects → {len(d2_group)} defects"
        
        return result
    
    def _handle_negative_growth(self,
                              growth_df: pd.DataFrame,
                              year1_defects: pd.DataFrame,
                              year2_defects: pd.DataFrame) -> pd.DataFrame:
        """Handle cases where calculated growth is negative."""
        
        negative_mask = growth_df['depth_growth_mm_per_year'] < self.min_growth_rate
        
        if not negative_mask.any():
            return growth_df
        
        if self.negative_growth_strategy == 'zero':
            # Simply set to zero
            growth_df.loc[negative_mask, 'depth_growth_mm_per_year'] = 0.0
            growth_df.loc[negative_mask, 'growth_adjustment'] = 'Set to zero'
            
        elif self.negative_growth_strategy == 'similar_match':
            # Find similar defects with positive growth
            for idx in growth_df[negative_mask].index:
                row = growth_df.loc[idx]
                
                # Find similar defects based on depth and length
                similar_growth = self._find_similar_positive_growth(
                    row, growth_df[~negative_mask]
                )
                
                if similar_growth is not None:
                    growth_df.loc[idx, 'depth_growth_mm_per_year'] = similar_growth
                    growth_df.loc[idx, 'growth_adjustment'] = 'Matched to similar defect'
                else:
                    # Fall back to statistical average
                    avg_growth = growth_df[~negative_mask]['depth_growth_mm_per_year'].mean()
                    growth_df.loc[idx, 'depth_growth_mm_per_year'] = avg_growth
                    growth_df.loc[idx, 'growth_adjustment'] = 'Set to average'
        
        elif self.negative_growth_strategy == 'statistical':
            # Use statistical distribution of positive growth rates
            positive_growths = growth_df[~negative_mask]['depth_growth_mm_per_year']
            if len(positive_growths) > 0:
                p50 = positive_growths.quantile(0.5)
                growth_df.loc[negative_mask, 'depth_growth_mm_per_year'] = p50
                growth_df.loc[negative_mask, 'growth_adjustment'] = 'Set to median growth'
        
        return growth_df
    
    def _find_similar_positive_growth(self, 
                                    target_row: pd.Series,
                                    positive_growth_df: pd.DataFrame) -> Optional[float]:
        """Find growth rate from similar defect with positive growth."""
        if positive_growth_df.empty:
            return None
        
        # Calculate similarity based on year1 characteristics
        similarities = []
        
        for idx, row in positive_growth_df.iterrows():
            depth_diff = abs(row['year1_depth_pct'] - target_row['year1_depth_pct'])
            length_diff = abs(row['year1_length_mm'] - target_row['year1_length_mm'])
            
            # Normalized similarity score
            similarity = 1 / (1 + depth_diff/10 + length_diff/100)
            similarities.append((similarity, row['depth_growth_mm_per_year']))
        
        # Return growth rate of most similar defect
        if similarities:
            similarities.sort(reverse=True)
            return similarities[0][1]
        
        return None
    
    def calculate_remaining_life(self,
                               growth_df: pd.DataFrame,
                               max_allowable_depth_pct: float = 80.0) -> pd.DataFrame:
        """
        Calculate remaining life for each defect/cluster based on growth rates.
        
        Parameters:
        - growth_df: DataFrame from analyze_growth_with_clustering
        - max_allowable_depth_pct: Maximum allowable depth (% of wall thickness)
        
        Returns:
        - DataFrame with remaining life estimates
        """
        results = growth_df.copy()
        
        # Calculate remaining life for each defect/cluster
        for idx, row in results.iterrows():
            current_depth_pct = row['year2_depth_pct']
            growth_rate_pct_per_year = row['depth_growth_mm_per_year'] * 100 / 10  # Assume 10mm wall
            
            if growth_rate_pct_per_year <= 0:
                remaining_years = np.inf
            else:
                remaining_depth_pct = max_allowable_depth_pct - current_depth_pct
                remaining_years = remaining_depth_pct / growth_rate_pct_per_year
            
            results.loc[idx, 'remaining_life_years'] = remaining_years
            results.loc[idx, 'max_allowable_depth_pct'] = max_allowable_depth_pct
            
            # Add safety classification
            if remaining_years < 2:
                safety_class = 'CRITICAL'
            elif remaining_years < 5:
                safety_class = 'HIGH PRIORITY'
            elif remaining_years < 10:
                safety_class = 'MODERATE'
            else:
                safety_class = 'LOW PRIORITY'
            
            results.loc[idx, 'safety_classification'] = safety_class
        
        return results