"""
Core functions for comparing pipeline defect data across multiple years.

This module provides functionality to:
- Match defects between inspection years based on location and clock position
- Calculate growth rates for matched defects (depth, length, width)
- Identify new defects that appeared between inspections
- Generate comprehensive statistics about defect evolution
"""


import pandas as pd
import numpy as np
import logging


# Set up logging for debugging (can be configured externally)
logger = logging.getLogger(__name__)



def compare_defects(old_defects_df, new_defects_df, old_joints_df = None, new_joints_df = None, old_year = None, new_year = None, 
                    distance_tolerance = 0.1, clock_tolerance_minutes = 20):
    """
    Compare defects between two pipeline inspection years to track defect evolution.
    
    This function matches defects from an older inspection to a newer inspection based on:
    - Physical location (distance along pipeline)
    - Clock position (circumferential position around pipe)
    
    For matched defects, it calculates growth rates in depth, length, and width.
    
    Parameters:
    -----------
    old_defects_df : pandas.DataFrame
        Defects from the earlier inspection. Must contain 'log dist. [m]' column.
        
    new_defects_df : pandas.DataFrame
        Defects from the newer inspection. Must contain 'log dist. [m]' column.
        
    old_joints_df : pandas.DataFrame, optional
        Joint data from earlier inspection. Used to get wall thickness for depth calculations.
        Should contain 'joint number' and 'wt nom [mm]' columns.
        
    new_joints_df : pandas.DataFrame, optional
        Joint data from newer inspection. Used to get wall thickness for depth calculations.
        
    old_year : int, optional
        Year of the earlier inspection. Required for growth rate calculations.
        
    new_year : int, optional
        Year of the newer inspection. Required for growth rate calculations.
        
    distance_tolerance : float, default=0.1
        Maximum distance difference (in meters) to consider defects as the same location.
        
    clock_tolerance_minutes : float, default=20
        Maximum clock position difference (in minutes) to consider defects at same position.
        Clock positions are on a 12-hour dial (e.g., 3 o'clock = 3.0 hours).
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'matches_df': DataFrame of matched defects with growth calculations
        - 'new_defects': DataFrame of defects that couldn't be matched (truly new)
        - 'common_defects_count': Number of matched defects
        - 'new_defects_count': Number of unmatched defects
        - 'total_defects': Total defects in new inspection
        - 'pct_common': Percentage of defects that were matched
        - 'pct_new': Percentage of defects that are new
        - 'defect_type_distribution': Distribution of defect types for new defects
        - 'growth_stats': Statistics about growth rates (if years provided)
        - Various flags indicating data availability
        
    Raises:
    -------
    ValueError
        If required columns are missing or input data is invalid
    """

    try:
        # === STEP 1: Validate inputs ===
        # Ensure we have the minimum required data
        if old_defects_df is None or new_defects_df is None:
            raise ValueError("Both old and new defect dataframes are required")
            
        # Create working copies to avoid modifying original data
        old_df = old_defects_df.copy()
        new_df = new_defects_df.copy()
        
        # Log column information for debugging
        logger.debug(f"Old defects columns: {old_df.columns.tolist()}")
        logger.debug(f"New defects columns: {new_df.columns.tolist()}")
        
        if old_joints_df is not None and new_joints_df is not None:
            logger.debug(f"Old joints columns: {old_joints_df.columns.tolist()}")
            logger.debug(f"New joints columns: {new_joints_df.columns.tolist()}")
        

        # === STEP 2: Check data availability for different calculations ===
        # Determine if we can calculate growth rates (need both years and valid time difference)
        calculate_growth = False
        year_diff = None
        if old_year is not None and new_year is not None and new_year > old_year:
            calculate_growth = True
            year_diff = new_year - old_year
            logger.info(f"Growth calculation enabled: {year_diff} years between inspections")
        
        # Check which dimensional data is available for growth calculations
        # These flags help us avoid KeyErrors and provide appropriate outputs
        has_depth_data = all(col in df.columns for df in [old_df, new_df] for col in ['depth [%]'])
        has_length_data = all(col in df.columns for df in [old_df, new_df] for col in ['length [mm]'])
        has_width_data = all(col in df.columns for df in [old_df, new_df] for col in ['width [mm]'])
        
        # Check if we can link defects to joints (needed for wall thickness calculations)
        has_joint_num = 'joint number' in old_df.columns and 'joint number' in new_df.columns
        if not has_joint_num:
            logger.warning("'joint number' column missing - cannot calculate mm-based depth growth")
        
        # Check if wall thickness data is available in joint dataframes
        has_wt_data = False
        if (has_joint_num and old_joints_df is not None and new_joints_df is not None and 'wt nom [mm]' in old_joints_df.columns and 'wt nom [mm]' in new_joints_df.columns):
            has_wt_data = True
        else:
            logger.warning("Wall thickness data unavailable - depth growth will be in % only")
        
        # Check clock position data availability (for circumferential matching)
        has_clock_float = 'clock_float' in old_df.columns and 'clock_float' in new_df.columns
        
        # Convert clock tolerance from minutes to hours for comparison
        # (clock_float is in hours, e.g., 3.5 = 3:30 position)
        clock_tolerance_hours = clock_tolerance_minutes / 60.0
        

        # === STEP 3: Validate required columns ===
        required_columns = ['log dist. [m]']
        for col in required_columns:
            if col not in old_df.columns or col not in new_df.columns:
                raise ValueError(f"Required column missing: '{col}'")
        
        # === STEP 4: Prepare data for efficient matching ===
        # Sort by distance for O(n) matching algorithm instead of O(n²)
        old_df = old_df.sort_values('log dist. [m]').reset_index(drop=True)
        new_df = new_df.sort_values('log dist. [m]').reset_index(drop=True)
        
        # Assign unique IDs for tracking
        old_df['defect_id'] = range(len(old_df))
        new_df['defect_id'] = range(len(new_df))
        

        # === STEP 5: Create wall thickness lookup tables ===
        old_wt_lookup = {}
        new_wt_lookup = {}
        
        if has_wt_data:
            try:
                # Create dictionaries mapping joint number to wall thickness
                old_wt_lookup = dict(zip(old_joints_df['joint number'], old_joints_df['wt nom [mm]']))
                new_wt_lookup = dict(zip(new_joints_df['joint number'], new_joints_df['wt nom [mm]']))
                logger.info(f"Created wall thickness lookups: {len(old_wt_lookup)} old joints, {len(new_wt_lookup)} new joints")
            except Exception as e:
                logger.error(f"Failed to create wall thickness lookup: {e}")
                has_wt_data = False
        

        # === STEP 6: Main matching algorithm ===
        # This uses a sliding window approach for O(n) complexity
        matches = []  # List to store all matched defect pairs
        matched_old_indices = set()  # Track which old defects have been matched
        matched_new_indices = set()  # Track which new defects have been matched
        
        # Initialize sliding window pointer
        old_index = 0
        old_max = len(old_df)
        
        # Process each defect in the new inspection
        for new_idx, new_defect in new_df.iterrows():
            new_dist = new_defect['log dist. [m]']
            
            # Get clock position if available
            new_clock_float = new_defect.get('clock_float') if has_clock_float else None
            
            # === SLIDING WINDOW OPTIMIZATION ===
            # Move the window start to the first old defect within distance tolerance
            # This works because both dataframes are sorted by distance
            while old_index < old_max and old_df.loc[old_index, 'log dist. [m]'] < new_dist - distance_tolerance:
                old_index += 1
            
            # Collect all potential matches within the distance tolerance window
            temp_old_index = old_index
            potential_matches = []
            
            # Check all old defects that could possibly match based on distance
            while temp_old_index < old_max and old_df.loc[temp_old_index, 'log dist. [m]'] <= new_dist + distance_tolerance:
                # Skip if this old defect was already matched to another new defect
                if temp_old_index not in matched_old_indices:
                    old_defect = old_df.loc[temp_old_index]
                    dist_diff = abs(old_defect['log dist. [m]'] - new_dist)
                    
                    # First filter: distance must be within tolerance
                    if dist_diff <= distance_tolerance:
                        # Second filter: check clock position if available
                        clock_match = True
                        clock_diff = 0
                        
                        if has_clock_float and new_clock_float is not None:
                            old_clock_float = old_defect.get('clock_float')
                            
                            if old_clock_float is not None:
                                # Calculate clock difference handling wrap-around
                                # E.g., 11.5 and 0.5 should be 1 hour apart, not 11
                                raw_clock_diff = abs(old_clock_float - new_clock_float)
                                clock_diff = min(raw_clock_diff, 12 - raw_clock_diff)
                                
                                # Check if within tolerance
                                clock_match = clock_diff <= clock_tolerance_hours
                        
                        if clock_match:
                            # This is a valid potential match
                            potential_matches.append((temp_old_index, dist_diff, clock_diff))
                
                temp_old_index += 1


            # === SELECT BEST MATCH ===
            if potential_matches:
                # Sort by combined score: prioritize distance but consider clock position (Weight determines relative importance (higher = clock matters more))
                weight = 1.5
                
                if has_clock_float:
                    # Combined score considers both distance and clock position
                    potential_matches.sort(key=lambda x: x[1] + weight * x[2])
                else:
                    # Only distance matters if no clock data
                    potential_matches.sort(key=lambda x: x[1])
                
                # Take the best match
                best_old_idx, distance_diff, clock_diff = potential_matches[0]
                closest_match = old_df.loc[best_old_idx]
                
                # === BUILD MATCH RECORD ===
                match_data = _build_match_record(
                    new_defect, closest_match, distance_diff, clock_diff,
                    has_clock_float, has_joint_num, calculate_growth, year_diff,
                    has_depth_data, has_length_data, has_width_data, has_wt_data,
                    old_wt_lookup, new_wt_lookup
                )
                
                matches.append(match_data)
                matched_old_indices.add(best_old_idx)
                matched_new_indices.add(new_idx)
        

        # === STEP 7: Create results dataframes ===
        # Build column list based on available data
        columns = _get_result_columns(new_defects_df, has_joint_num, has_clock_float, calculate_growth, has_depth_data, has_length_data, has_width_data, has_wt_data)
        
        # Create matches dataframe
        matches_df = pd.DataFrame(matches, columns=columns) if matches else pd.DataFrame(columns=columns)
        
        # Add year information if growth was calculated
        if calculate_growth and not matches_df.empty:
            matches_df['old_year'] = old_year
            matches_df['new_year'] = new_year
        
        # Identify truly new defects (those that couldn't be matched)
        new_defects = new_df.loc[~new_df.index.isin(matched_new_indices)].copy()
        
        # === STEP 8: Calculate summary statistics ===
        total = len(new_df)
        common = len(matches_df)
        new_cnt = len(new_defects)
        
        # Calculate percentages
        pct_common = (common / total * 100) if total > 0 else 0
        pct_new = (new_cnt / total * 100) if total > 0 else 0
        
        # Analyze distribution of new defect types
        if new_cnt > 0 and 'component / anomaly identification' in new_defects.columns:
            dist = (new_defects['component / anomaly identification']
                    .value_counts()
                    .rename_axis('defect_type')
                    .reset_index(name='count'))
            dist['percentage'] = dist['count'] / new_cnt * 100
        else:
            dist = pd.DataFrame(columns=['defect_type', 'count', 'percentage'])
        
        
        # === STEP 9: Calculate growth statistics ===
        growth_stats = None
        if calculate_growth and not matches_df.empty and any([has_depth_data, has_length_data, has_width_data]):
            growth_stats = _calculate_growth_statistics(
                matches_df, has_depth_data, has_length_data, has_width_data, has_wt_data
            )
        

        # === STEP 10: Return comprehensive results ===
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
            'has_length_data': has_length_data,
            'has_width_data': has_width_data,
            'has_wt_data': has_wt_data,
            'has_joint_num': has_joint_num,
            'calculate_growth': calculate_growth
        }
        
    except KeyError as ke:
        # Provide specific error message for missing columns
        column_name = str(ke).strip("'")
        if column_name == "joint number":
            raise ValueError("Missing 'joint number' column in the defects data. "
                           "Please ensure both datasets have this column for wall thickness calculations.")
        else:
            raise ValueError(f"Missing required column: {column_name}")
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Error in compare_defects: {str(e)}") from e


def _build_match_record(new_defect, old_defect, distance_diff, clock_diff, has_clock_float, has_joint_num, calculate_growth, year_diff,
                       has_depth_data, has_length_data, has_width_data, has_wt_data, old_wt_lookup, new_wt_lookup):
    """
    Build a complete match record for a matched defect pair.
    
    This helper function consolidates all the match data creation logic,
    making the main function cleaner and more maintainable.
    """

    # Start with basic match information
    match_data = {
        'new_defect_id': new_defect['defect_id'],
        'old_defect_id': old_defect['defect_id'],
        'distance_diff': distance_diff,
        'log_dist': new_defect['log dist. [m]'],
        'old_log_dist': old_defect['log dist. [m]'],
        'defect_type': new_defect.get('component / anomaly identification', 'Unknown')
    }
    
    # Add joint number if available (needed for wall thickness lookups)
    if has_joint_num:
        match_data['joint number'] = new_defect['joint number']
    
    # Add dimensional data (needed for KNN correction if implemented)
    for dimension_col in ['length [mm]', 'width [mm]']:
        if dimension_col in new_defect:
            match_data[dimension_col] = new_defect[dimension_col]
    
    # Add clock difference information
    if has_clock_float:
        match_data['clock_diff_hours'] = clock_diff
        match_data['clock_diff_minutes'] = clock_diff * 60
    
    # Calculate growth rates if we have year information
    if calculate_growth and year_diff:
        # === DEPTH GROWTH ===
        if has_depth_data:
            old_depth = old_defect['depth [%]']
            new_depth = new_defect['depth [%]']
            depth_change = new_depth - old_depth
            
            match_data.update({
                'old_depth_pct': old_depth,
                'new_depth_pct': new_depth,
                'depth_change_pct': depth_change,
                'growth_rate_pct_per_year': depth_change / year_diff,
                'is_negative_growth': depth_change < 0
            })
            
            # Convert to mm if wall thickness is available
            if has_wt_data and has_joint_num:
                try:
                    # Look up wall thicknesses for both joints
                    old_joint = old_defect['joint number']
                    new_joint = new_defect['joint number']
                    
                    old_wt = old_wt_lookup.get(old_joint)
                    new_wt = new_wt_lookup.get(new_joint)
                    
                    if old_wt is not None and new_wt is not None:
                        # Use average wall thickness for conversion
                        avg_wt = (old_wt + new_wt) / 2
                        
                        # Convert percentage depths to millimeters
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
                    logger.warning(f"Could not calculate mm-based depth growth for joint {new_joint}: {e}")
        
        # === LENGTH GROWTH ===
        if has_length_data:
            old_length = old_defect['length [mm]']
            new_length = new_defect['length [mm]']
            length_change = new_length - old_length
            
            match_data.update({
                'old_length_mm': old_length,
                'new_length_mm': new_length,
                'length_change_mm': length_change,
                'length_growth_rate_mm_per_year': length_change / year_diff,
                'is_negative_length_growth': length_change < 0
            })
        
        # === WIDTH GROWTH ===
        if has_width_data:
            old_width = old_defect['width [mm]']
            new_width = new_defect['width [mm]']
            width_change = new_width - old_width
            
            match_data.update({
                'old_width_mm': old_width,
                'new_width_mm': new_width,
                'width_change_mm': width_change,
                'width_growth_rate_mm_per_year': width_change / year_diff,
                'is_negative_width_growth': width_change < 0
            })
    
    return match_data


def _get_result_columns(new_defects_df, has_joint_num, has_clock_float, 
                       calculate_growth, has_depth_data, has_length_data, 
                       has_width_data, has_wt_data):
    """
    Build the list of columns for the results DataFrame based on available data.
    
    This ensures we only include columns for data that actually exists,
    preventing empty columns in the output.
    """
    # Always include basic columns
    columns = ['new_defect_id', 'old_defect_id', 'distance_diff', 
               'log_dist', 'old_log_dist', 'defect_type']
    
    # Add dimensional columns if present in input
    for dimension_col in ['length [mm]', 'width [mm]']:
        if dimension_col in new_defects_df.columns:
            columns.append(dimension_col)
    
    # Add joint number if available
    if has_joint_num:
        columns.append('joint number')
    
    # Add clock difference columns if clock data exists
    if has_clock_float:
        columns.extend(['clock_diff_hours', 'clock_diff_minutes'])
    
    # Add growth-related columns if growth calculation is enabled
    if calculate_growth:
        # Depth growth columns
        if has_depth_data:
            columns.extend(['old_depth_pct', 'new_depth_pct', 'depth_change_pct', 
                           'growth_rate_pct_per_year', 'is_negative_growth'])
            # Add mm-based depth columns if wall thickness is available
            if has_wt_data:
                columns.extend(['old_wt_mm', 'new_wt_mm', 'old_depth_mm', 'new_depth_mm', 
                               'depth_change_mm', 'growth_rate_mm_per_year'])
        
        # Length growth columns
        if has_length_data:
            columns.extend(['old_length_mm', 'new_length_mm', 'length_change_mm', 
                           'length_growth_rate_mm_per_year', 'is_negative_length_growth'])
        
        # Width growth columns
        if has_width_data:
            columns.extend(['old_width_mm', 'new_width_mm', 'width_change_mm', 
                           'width_growth_rate_mm_per_year', 'is_negative_width_growth'])
    
    return columns


def _calculate_growth_statistics(matches_df, has_depth_data, has_length_data, 
                               has_width_data, has_wt_data):
    """
    Calculate comprehensive growth statistics from matched defects.
    
    This function computes averages, maximums, and counts of negative growth
    for each dimension (depth, length, width) where data is available.
    """
    growth_stats = {
        'total_matched_defects': len(matches_df)
    }
    
    # === DEPTH STATISTICS ===
    if has_depth_data:
        # Count and percentage of defects with negative depth growth
        negative_growth_count = matches_df['is_negative_growth'].sum()
        pct_negative_growth = (negative_growth_count / len(matches_df)) * 100
        
        # Filter to only positive growth for certain statistics
        positive_growth = matches_df[~matches_df['is_negative_growth']]
        
        growth_stats.update({
            'depth_negative_growth_count': int(negative_growth_count),
            'depth_pct_negative_growth': pct_negative_growth,
            'avg_growth_rate_pct': matches_df['growth_rate_pct_per_year'].mean(),
            'avg_positive_growth_rate_pct': positive_growth['growth_rate_pct_per_year'].mean() if len(positive_growth) > 0 else 0,
            'max_growth_rate_pct': positive_growth['growth_rate_pct_per_year'].max() if len(positive_growth) > 0 else 0
        })
        
        # Add mm-based statistics if wall thickness data was available
        if has_wt_data and 'growth_rate_mm_per_year' in matches_df.columns:
            # Only calculate for rows where mm conversion was successful
            mm_data = matches_df.dropna(subset=['growth_rate_mm_per_year'])
            positive_mm = mm_data[mm_data['growth_rate_mm_per_year'] > 0]
            
            growth_stats.update({
                'avg_growth_rate_mm': mm_data['growth_rate_mm_per_year'].mean() if len(mm_data) > 0 else np.nan,
                'avg_positive_growth_rate_mm': positive_mm['growth_rate_mm_per_year'].mean() if len(positive_mm) > 0 else 0,
                'max_growth_rate_mm': positive_mm['growth_rate_mm_per_year'].max() if len(positive_mm) > 0 else 0
            })
    
    # === LENGTH STATISTICS ===
    if has_length_data:
        negative_length_count = matches_df['is_negative_length_growth'].sum()
        pct_negative_length = (negative_length_count / len(matches_df)) * 100
        positive_length_growth = matches_df[~matches_df['is_negative_length_growth']]
        
        growth_stats.update({
            'length_negative_growth_count': int(negative_length_count),
            'length_pct_negative_growth': pct_negative_length,
            'avg_length_growth_rate_mm': matches_df['length_growth_rate_mm_per_year'].mean(),
            'avg_positive_length_growth_rate_mm': positive_length_growth['length_growth_rate_mm_per_year'].mean() if len(positive_length_growth) > 0 else 0,
            'max_length_growth_rate_mm': positive_length_growth['length_growth_rate_mm_per_year'].max() if len(positive_length_growth) > 0 else 0
        })
    
    # === WIDTH STATISTICS ===
    if has_width_data:
        negative_width_count = matches_df['is_negative_width_growth'].sum()
        pct_negative_width = (negative_width_count / len(matches_df)) * 100
        positive_width_growth = matches_df[~matches_df['is_negative_width_growth']]
        
        growth_stats.update({
            'width_negative_growth_count': int(negative_width_count),
            'width_pct_negative_growth': pct_negative_width,
            'avg_width_growth_rate_mm': matches_df['width_growth_rate_mm_per_year'].mean(),
            'avg_positive_width_growth_rate_mm': positive_width_growth['width_growth_rate_mm_per_year'].mean() if len(positive_width_growth) > 0 else 0,
            'max_width_growth_rate_mm': positive_width_growth['width_growth_rate_mm_per_year'].max() if len(positive_width_growth) > 0 else 0
        })
    
    return growth_stats


def create_matching_debug_view(old_defects_df, new_defects_df, distance_tolerance=0.1):
    """
    Create a diagnostic view to debug defect matching issues.
    
    This function helps identify why certain defects may not be matching by showing
    all potential matches within 2x the normal distance tolerance. It's useful for:
    - Verifying the matching algorithm is working correctly
    - Identifying edge cases where defects just miss the tolerance
    - Understanding the distribution of defects near the tolerance boundary
    
    Parameters:
    -----------
    old_defects_df : pandas.DataFrame
        Defects from the earlier inspection
        
    new_defects_df : pandas.DataFrame
        Defects from the newer inspection
        
    distance_tolerance : float, default=0.1
        The distance tolerance used in the main matching algorithm
        (this function uses 2x this value for the debug view)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame showing potential matches with columns:
        - new_dist: Distance of defect in new inspection
        - old_dist: Distance of defect in old inspection
        - distance_diff: Absolute difference in distances
        - defect_type: Type of defect
        - would_match: Boolean indicating if this would match with normal tolerance
        - Additional columns for any shared attributes (depth, clock, etc.)
    """
    # Define columns we want to compare if they exist in both datasets
    base_columns = ['log dist. [m]', 'component / anomaly identification']
    optional_columns = ['depth [%]', 'clock', 'clock_float', 'length [mm]', 'width [mm]'] 
    
    # Build list of columns that exist in both datasets
    columns_to_compare = base_columns.copy()
    for col in optional_columns:
        if col in old_defects_df.columns and col in new_defects_df.columns:
            columns_to_compare.append(col)
    
    # Use 2x tolerance for debug view to see near-misses
    debug_tolerance = distance_tolerance * 2
    
    merged_view = []
    
    # For each defect in the new dataset, find all nearby defects in old dataset
    for _, new_defect in new_defects_df.iterrows():
        new_dist = new_defect['log dist. [m]']
        
        # Find all old defects within the expanded tolerance
        # This helps identify defects that are close but don't match
        nearby_old = old_defects_df[
            (abs(old_defects_df['log dist. [m]'] - new_dist) <= debug_tolerance)
        ]
        
        # Create a row for each potential match
        for _, old_defect in nearby_old.iterrows():
            distance_diff = abs(new_defect['log dist. [m]'] - old_defect['log dist. [m]'])
            
            row = {
                'new_dist': new_dist,
                'old_dist': old_defect['log dist. [m]'],
                'distance_diff': distance_diff,
                'defect_type': new_defect.get('component / anomaly identification', 'Unknown'),
                'would_match': distance_diff <= distance_tolerance  # Would this match normally?
            }
            
            # Add additional comparison columns
            for col in columns_to_compare:
                if col not in ['log dist. [m]', 'component / anomaly identification']:
                    if col in new_defect and col in old_defect:
                        row[f'new_{col}'] = new_defect[col]
                        row[f'old_{col}'] = old_defect[col]
                        
                        # Calculate differences for numeric columns
                        if pd.api.types.is_numeric_dtype(type(new_defect[col])):
                            try:
                                row[f'{col}_diff'] = abs(new_defect[col] - old_defect[col])
                            except:
                                pass  # Skip if subtraction fails
            
            merged_view.append(row)
    
    # Convert to DataFrame and sort by distance difference
    debug_df = pd.DataFrame(merged_view)
    if not debug_df.empty:
        debug_df = debug_df.sort_values(['new_dist', 'distance_diff'])
    
    return debug_df