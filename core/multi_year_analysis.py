"""
Core functions for comparing pipeline defect data across multiple years.
"""
import pandas as pd
import numpy as np

def compare_defects(old_defects_df, new_defects_df, old_joints_df=None, new_joints_df=None, 
                   old_year=None, new_year=None, distance_tolerance=0.1, clock_tolerance_minutes=20,
                   correct_negative_growth=False):
    """
    Compare defects between two inspection years with optimized algorithm.
    
    Parameters:
    - old_defects_df: DataFrame with defects from the earlier inspection
    - new_defects_df: DataFrame with defects from the newer inspection
    - old_joints_df: DataFrame with joints from the earlier inspection (for wall thickness)
    - new_joints_df: DataFrame with joints from the newer inspection (for wall thickness)
    - old_year: Year of the earlier inspection (optional, for growth rate calculation)
    - new_year: Year of the later inspection (optional, for growth rate calculation)
    - distance_tolerance: Maximum distance (in meters) to consider defects at the same location
    - clock_tolerance_minutes: Maximum difference in clock position (in minutes) to consider defects at the same position
    - correct_negative_growth: Whether to apply KNN correction to negative growth rates
    
    Returns:
    - results: Dictionary with comparison results and statistics
    """
    try:
        # Verify input data
        if old_defects_df is None or new_defects_df is None:
            raise ValueError("Missing defect dataframes")
            
        # Copy inputs to avoid modifying originals
        old_df = old_defects_df.copy()
        new_df = new_defects_df.copy()
        
        # Print column names for debugging
        print("Old defects columns:", old_df.columns.tolist())
        print("New defects columns:", new_df.columns.tolist())
        
        if old_joints_df is not None and new_joints_df is not None:
            print("Old joints columns:", old_joints_df.columns.tolist())
            print("New joints columns:", new_joints_df.columns.tolist())
        
        # Check if we can calculate growth rates
        calculate_growth = False
        if old_year is not None and new_year is not None and new_year > old_year:
            calculate_growth = True
            year_diff = new_year - old_year
        
        # Check if depth data is available for growth calculations
        has_depth_data = ('depth [%]' in old_df.columns and 'depth [%]' in new_df.columns)
        has_length_data = ('length [mm]' in old_df.columns and 'length [mm]' in new_df.columns)
        has_width_data = ('width [mm]' in old_df.columns and 'width [mm]' in new_df.columns)

        # Check if joint number column exists in both dataframes
        has_joint_num = False
        if 'joint number' in old_df.columns and 'joint number' in new_df.columns:
            has_joint_num = True
        else:
            print("Warning: 'joint number' column missing in one or both defect dataframes")
        
        # Check if wall thickness data is available in joints dataframes
        has_wt_data = False
        if has_joint_num and old_joints_df is not None and new_joints_df is not None:
            if 'wt nom [mm]' in old_joints_df.columns and 'wt nom [mm]' in new_joints_df.columns:
                has_wt_data = True
            else:
                print("Warning: 'wt nom [mm]' column missing in one or both joint dataframes")
        
        # Check if clock data is available for position matching
        has_clock_data = ('clock' in old_df.columns and 'clock' in new_df.columns)
        has_clock_float = ('clock_float' in old_df.columns and 'clock_float' in new_df.columns)
        
        # Convert clock tolerance from minutes to hours for comparison with clock_float
        clock_tolerance_hours = clock_tolerance_minutes / 60.0
        
        # Check required columns
        for col in ['log dist. [m]']:
            if col not in old_df.columns or col not in new_df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Sort both dataframes by distance for faster matching
        old_df = old_df.sort_values('log dist. [m]').reset_index(drop=True)
        new_df = new_df.sort_values('log dist. [m]').reset_index(drop=True)
        
        # Assign IDs
        old_df['defect_id'] = range(len(old_df))
        new_df['defect_id'] = range(len(new_df))
        
        # Create lookup dictionaries for wall thickness if available
        old_wt_lookup = {}
        new_wt_lookup = {}
        
        if has_wt_data:
            try:
                old_wt_lookup = dict(zip(old_joints_df['joint number'], old_joints_df['wt nom [mm]']))
                new_wt_lookup = dict(zip(new_joints_df['joint number'], new_joints_df['wt nom [mm]']))
            except Exception as e:
                print(f"Error creating wall thickness lookup: {e}")
                has_wt_data = False
        
        matches = []
        matched_old_indices = set()
        matched_new_indices = set()
        
        # Optimized matching algorithm using the sorted property
        old_index = 0
        old_max = len(old_df)
        
        for new_idx, new_defect in new_df.iterrows():
            new_dist = new_defect['log dist. [m]']
            
            # Get clock position for new defect if available
            new_clock_float = None
            if has_clock_float:
                new_clock_float = new_defect['clock_float']
            
            # Set the search window based on tolerance
            # Find the starting point in old_df where distance is within tolerance
            while old_index < old_max and old_df.loc[old_index, 'log dist. [m]'] < new_dist - distance_tolerance:
                old_index += 1
            
            # Check all old defects within tolerance window
            temp_old_index = old_index
            potential_matches = []
            
            # Only look at defects within the tolerance window
            while temp_old_index < old_max and old_df.loc[temp_old_index, 'log dist. [m]'] <= new_dist + distance_tolerance:
                # Skip if already matched
                if temp_old_index not in matched_old_indices:
                    old_defect = old_df.loc[temp_old_index]
                    dist_diff = abs(old_defect['log dist. [m]'] - new_dist)
                    
                    # Check if distance is within tolerance
                    if dist_diff <= distance_tolerance:
                        # Check clock position if available
                        clock_match = True
                        clock_diff = 0
                        
                        if has_clock_float and new_clock_float is not None:
                            old_clock_float = old_defect['clock_float']
                            
                            # Calculate the clock difference in hours
                            # Handle the case where clocks are near 12 and 1 (wrap around)
                            if old_clock_float is not None:
                                raw_clock_diff = abs(old_clock_float - new_clock_float)
                                # Handle wrap-around (e.g. 12 o'clock vs 1 o'clock should be 1 hour, not 11)
                                clock_diff = min(raw_clock_diff, 12 - raw_clock_diff)
                                
                                # Check if clock difference is within tolerance
                                clock_match = clock_diff <= clock_tolerance_hours
                        
                        if clock_match:
                            # Store both distance and clock differences for ranking
                            potential_matches.append((temp_old_index, dist_diff, clock_diff))
                
                temp_old_index += 1
            
            # Find the best match considering both distance and clock position
            if potential_matches:
                # Sort matches by a combined score (distance_diff + weight * clock_diff)
                # This gives more weight to distance but still considers clock position
                weight = 2.0  # Adjust this weight based on the relative importance of clock vs distance
                
                if has_clock_float:
                    # Sort by a combined score of distance and clock differences
                    potential_matches.sort(key=lambda x: x[1] + weight * x[2])
                else:
                    # If no clock data, sort by distance only
                    potential_matches.sort(key=lambda x: x[1])
                
                best_old_idx = potential_matches[0][0]
                distance_diff = potential_matches[0][1]
                closest_match = old_df.loc[best_old_idx]
                
                # Basic match data
                match_data = {
                    'new_defect_id': new_defect['defect_id'],
                    'old_defect_id': closest_match['defect_id'],
                    'distance_diff': distance_diff,
                    'log_dist': new_defect['log dist. [m]'],
                    'old_log_dist': closest_match['log dist. [m]'],
                    'defect_type': new_defect['component / anomaly identification'] if 'component / anomaly identification' in new_defect else 'Unknown'
                }
                
                # Add joint number if available (needed for KNN correction later)
                if has_joint_num:
                    match_data['joint number'] = new_defect['joint number']
                
                # Add dimensional data needed for KNN correction
                for dimension_col in ['length [mm]', 'width [mm]']:
                    if dimension_col in new_defect:
                        match_data[dimension_col] = new_defect[dimension_col]
                
                # Add clock difference information if available
                if has_clock_float:
                    match_data['clock_diff_hours'] = potential_matches[0][2]
                    match_data['clock_diff_minutes'] = potential_matches[0][2] * 60
                
                # Add growth data if available
                if calculate_growth:
                    # Depth growth (existing logic)
                    if has_depth_data:
                        old_depth = closest_match['depth [%]']
                        new_depth = new_defect['depth [%]']
                        
                        match_data.update({
                            'old_depth_pct': old_depth,
                            'new_depth_pct': new_depth,
                            'depth_change_pct': new_depth - old_depth,
                            'growth_rate_pct_per_year': (new_depth - old_depth) / year_diff,
                            'is_negative_growth': (new_depth - old_depth) < 0
                        })
                    
                    # Length growth (NEW)
                    if has_length_data:
                        old_length = closest_match['length [mm]']
                        new_length = new_defect['length [mm]']
                        
                        match_data.update({
                            'old_length_mm': old_length,
                            'new_length_mm': new_length,
                            'length_change_mm': new_length - old_length,
                            'length_growth_rate_mm_per_year': (new_length - old_length) / year_diff,
                            'is_negative_length_growth': (new_length - old_length) < 0
                        })
                    
                    # Width growth (NEW)
                    if has_width_data:
                        old_width = closest_match['width [mm]']
                        new_width = new_defect['width [mm]']
                        
                        match_data.update({
                            'old_width_mm': old_width,
                            'new_width_mm': new_width,
                            'width_change_mm': new_width - old_width,
                            'width_growth_rate_mm_per_year': (new_width - old_width) / year_diff,
                            'is_negative_width_growth': (new_width - old_width) < 0
                        })
                    
                    # If wall thickness data is available, convert to mm/year
                    if has_wt_data and has_joint_num:
                        try:
                            # Get joint numbers for both defects
                            old_joint = closest_match['joint number']
                            new_joint = new_defect['joint number']
                            
                            # Look up wall thickness values from the joint dataframes
                            old_wt = old_wt_lookup.get(old_joint)
                            new_wt = new_wt_lookup.get(new_joint)
                            
                            # Only proceed if both wall thicknesses are available
                            if old_wt is not None and new_wt is not None:
                                # Use the average wall thickness for conversion
                                avg_wt = (old_wt + new_wt) / 2
                                
                                old_depth_mm = old_depth * avg_wt / 100
                                new_depth_mm = new_depth * avg_wt / 100
                                
                                match_data.update({
                                    'old_wt_mm': old_wt,
                                    'new_wt_mm': new_wt,
                                    'old_depth_mm': old_depth_mm,
                                    'new_depth_mm': new_depth_mm,
                                    'depth_change_mm': new_depth_mm - old_depth_mm,
                                    'growth_rate_mm_per_year': (new_depth_mm - old_depth_mm) / year_diff
                                })
                        except Exception as e:
                            print(f"Error calculating mm-based growth for joint {new_joint}: {e}")
                
                matches.append(match_data)
                matched_old_indices.add(best_old_idx)
                matched_new_indices.add(new_idx)
        
        # Column list for empty dataframe handling
        columns = ['new_defect_id', 'old_defect_id', 'distance_diff', 
                   'log_dist', 'old_log_dist', 'defect_type']
        
        for dimension_col in ['length [mm]', 'width [mm]']:
            if dimension_col in new_defects_df.columns:
                columns.append(dimension_col)
        
        # Add joint number to columns if available
        if has_joint_num:
            columns.append('joint number')
        
        if has_clock_float:
            columns.extend(['clock_diff_hours', 'clock_diff_minutes'])
                   
        if calculate_growth and has_depth_data:
            columns.extend(['old_depth_pct', 'new_depth_pct', 'depth_change_pct', 
                           'growth_rate_pct_per_year', 'is_negative_growth'])
            if has_wt_data:
                columns.extend(['old_wt_mm', 'new_wt_mm', 'old_depth_mm', 'new_depth_mm', 
                               'depth_change_mm', 'growth_rate_mm_per_year'])
                
        if calculate_growth and has_length_data:
            columns.extend(['old_length_mm', 'new_length_mm', 'length_change_mm', 
                        'length_growth_rate_mm_per_year', 'is_negative_length_growth'])

        if calculate_growth and has_width_data:
            columns.extend(['old_width_mm', 'new_width_mm', 'width_change_mm', 
                        'width_growth_rate_mm_per_year', 'is_negative_width_growth'])
            
        # Build results
        matches_df = pd.DataFrame(matches, columns=columns) if matches else pd.DataFrame(columns=columns)
        
        # Store the year values for correction usage
        if calculate_growth:
            matches_df['old_year'] = old_year
            matches_df['new_year'] = new_year
        
        new_defects = new_df.loc[~new_df.index.isin(matched_new_indices)].copy()
        
        total = len(new_df)
        common = len(matches_df)
        new_cnt = len(new_defects)
        
        # Stats
        pct_common = common/total*100 if total else 0
        pct_new = new_cnt/total*100 if total else 0
        
        # Distribution of "truly new" types
        if new_cnt and 'component / anomaly identification' in new_defects.columns:
            dist = (new_defects['component / anomaly identification']
                    .value_counts()
                    .rename_axis('defect_type')
                    .reset_index(name='count'))
            dist['percentage'] = dist['count']/new_cnt*100
        else:
            dist = pd.DataFrame(columns=['defect_type', 'count', 'percentage'])
        
        # Calculate growth statistics if depth data is available
        growth_stats = None
        if calculate_growth and (has_depth_data or has_length_data or has_width_data) and not matches_df.empty:
            growth_stats = {
                'total_matched_defects': len(matches_df)
            }
            
            # Depth statistics (existing)
            if has_depth_data:
                negative_growth_count = matches_df['is_negative_growth'].sum()
                pct_negative_growth = (negative_growth_count / len(matches_df)) * 100 if len(matches_df) > 0 else 0
                positive_growth = matches_df[~matches_df['is_negative_growth']]
                
                growth_stats.update({
                    'depth_negative_growth_count': negative_growth_count,
                    'depth_pct_negative_growth': pct_negative_growth,
                    'avg_growth_rate_pct': matches_df['growth_rate_pct_per_year'].mean(),
                    'avg_positive_growth_rate_pct': positive_growth['growth_rate_pct_per_year'].mean() if len(positive_growth) > 0 else 0,
                    'max_growth_rate_pct': positive_growth['growth_rate_pct_per_year'].max() if len(positive_growth) > 0 else 0
                })
                
                # Add mm-based depth stats if available
                if has_wt_data and 'growth_rate_mm_per_year' in matches_df.columns:
                    growth_stats.update({
                        'avg_growth_rate_mm': matches_df['growth_rate_mm_per_year'].mean(),
                        'avg_positive_growth_rate_mm': positive_growth['growth_rate_mm_per_year'].mean() if len(positive_growth) > 0 else 0,
                        'max_growth_rate_mm': positive_growth['growth_rate_mm_per_year'].max() if len(positive_growth) > 0 else 0
                    })
            
            # Length statistics (NEW)
            if has_length_data:
                negative_length_count = matches_df['is_negative_length_growth'].sum()
                pct_negative_length = (negative_length_count / len(matches_df)) * 100 if len(matches_df) > 0 else 0
                positive_length_growth = matches_df[~matches_df['is_negative_length_growth']]
                
                growth_stats.update({
                    'length_negative_growth_count': negative_length_count,
                    'length_pct_negative_growth': pct_negative_length,
                    'avg_length_growth_rate_mm': matches_df['length_growth_rate_mm_per_year'].mean(),
                    'avg_positive_length_growth_rate_mm': positive_length_growth['length_growth_rate_mm_per_year'].mean() if len(positive_length_growth) > 0 else 0,
                    'max_length_growth_rate_mm': positive_length_growth['length_growth_rate_mm_per_year'].max() if len(positive_length_growth) > 0 else 0
                })
            
            # Width statistics (NEW)
            if has_width_data:
                negative_width_count = matches_df['is_negative_width_growth'].sum()
                pct_negative_width = (negative_width_count / len(matches_df)) * 100 if len(matches_df) > 0 else 0
                positive_width_growth = matches_df[~matches_df['is_negative_width_growth']]
                
                growth_stats.update({
                    'width_negative_growth_count': negative_width_count,
                    'width_pct_negative_growth': pct_negative_width,
                    'avg_width_growth_rate_mm': matches_df['width_growth_rate_mm_per_year'].mean(),
                    'avg_positive_width_growth_rate_mm': positive_width_growth['width_growth_rate_mm_per_year'].mean() if len(positive_width_growth) > 0 else 0,
                    'max_width_growth_rate_mm': positive_width_growth['width_growth_rate_mm_per_year'].max() if len(positive_width_growth) > 0 else 0
                })

        
        return {
            'matches_df': matches_df,
            'new_defects': new_defects,
            'common_defects_count': common,
            'new_defects_count': new_cnt,
            'total_defects': total,
            'pct_common': pct_common,
            'pct_new': pct_new,
            'defect_type_distribution': dist,
            'growth_stats': growth_stats,
            'has_depth_data': has_depth_data,
            'has_length_data': has_length_data,  # NEW
            'has_width_data': has_width_data,    # NEW
            'has_wt_data': has_wt_data,
            'has_joint_num': has_joint_num,
            'calculate_growth': calculate_growth
        }
        
    except KeyError as ke:
        # Catch specific KeyError for 'joint number'
        if str(ke).strip("'") == "joint number":
            raise ValueError("Missing 'joint number' column in the defects data. Please ensure both datasets have this column.")
        else:
            raise ValueError(f"KeyError: {str(ke)}")
    except Exception as e:
        # Catch any other exceptions
        raise ValueError(f"Error in compare_defects: {str(e)}")


def create_matching_debug_view(old_defects_df, new_defects_df, distance_tolerance=0.1):
    """
    Create a diagnostic view to debug defect matching issues
    
    Parameters:
    - old_defects_df: DataFrame with defects from the earlier inspection
    - new_defects_df: DataFrame with defects from the newer inspection
    - distance_tolerance: Maximum distance used for matching
    
    Returns:
    - Pandas DataFrame with matching diagnostics
    """
    # Common columns to compare
    columns = ['log dist. [m]', 'component / anomaly identification']
    extra_cols = ['depth [%]', 'clock', 'length [mm]', 'width [mm]'] 
    
    # Add any extra columns that are available in both datasets
    for col in extra_cols:
        if col in old_defects_df.columns and col in new_defects_df.columns:
            columns.append(col)
    
    # Create a merged view of close defects
    merged_view = []
    
    # Loop through each defect in the new dataset
    for _, new_defect in new_defects_df.iterrows():
        # Find old defects of the same type within tolerance
        nearby_old = old_defects_df[
            #(old_defects_df['component / anomaly identification'] == new_defect['component / anomaly identification']) &
            (abs(old_defects_df['log dist. [m]'] - new_defect['log dist. [m]']) <= distance_tolerance * 2)  # Using 2x tolerance for this view
        ]
        
        if not nearby_old.empty:
            for _, old_defect in nearby_old.iterrows():
                row = {
                    'new_dist': new_defect['log dist. [m]'],
                    'old_dist': old_defect['log dist. [m]'],
                    'distance_diff': abs(new_defect['log dist. [m]'] - old_defect['log dist. [m]']),
                    'defect_type': new_defect['component / anomaly identification'],
                    'would_match': abs(new_defect['log dist. [m]'] - old_defect['log dist. [m]']) <= distance_tolerance
                }
                
                # Add additional columns
                for col in columns:
                    if col not in ['log dist. [m]', 'component / anomaly identification']:
                        if col in new_defect and col in old_defect:
                            row[f'new_{col}'] = new_defect[col]
                            row[f'old_{col}'] = old_defect[col]
                
                merged_view.append(row)
    
    return pd.DataFrame(merged_view)