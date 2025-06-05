"""
## Purpose:
    Processes the standardized pipeline data into two DataFrames: one for joints, one for defects.

## Key Steps:
    - Replace empty strings with NaN for easier missing data handling.
    - Convert numeric columns to appropriate types (with error coercion).
    - Sort by log distance to ensure correct order for forward-filling.
    - Extract joints_df:
        1. All rows with a non-null joint number, selecting key joint columns.
        2. Deduplicates by joint number.

    - Forward-fill joint number in the main DataFrame to associate defects with their joints.
    - Extract defects_df:
        1. All rows with both length and width present.

    - Selects relevant defect columns.
    - Standardizes surface location using a utility function.
    - Return both DataFrames.

If standardize_surface_location can fail or has edge cases, wrap in a try/except or validate its output.
"""


import pandas as pd
import numpy as np
from utils.format_utils import standardize_surface_location


def process_pipeline_data(df):
    """
    Process the pipeline inspection data into two separate tables:
    1. joints_df: Contains unique joint information
    2. defects_df: Contains defect information with joint associations
    
    Parameters:
    - df: DataFrame with the raw pipeline data
    
    Returns:
    - joints_df: DataFrame with joint information
    - defects_df: DataFrame with defect information
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # 1. Replace empty strings with NaN for proper handling
    df_copy = df_copy.replace(r'^\s*$', np.nan, regex=True)
    
    # 2. Convert numeric columns to appropriate types
    numeric_columns = [
        'joint number', 
        'joint length [m]', 
        'wt nom [mm]', 
        'up weld dist. [m]', 
        'depth [%]', 
        'length [mm]', 
        'width [mm]'
    ]
    
    for col in numeric_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # 3. Sort by log distance to ensure proper order for forward fill
    if 'log dist. [m]' in df_copy.columns:
        df_copy = df_copy.sort_values('log dist. [m]')
    
    # 4. Create joints_df with only the specified columns
    joints_df = df_copy[df_copy['joint number'].notna()][['log dist. [m]', 'joint number', 'joint length [m]', 'wt nom [mm]']].copy()
    
    # 5. Drop duplicate joint numbers if any
    joints_df = joints_df.drop_duplicates(subset=['joint number'])
    joints_df = joints_df.reset_index().drop(columns = ['index'])
    
    # 6. Create defects_df - records with length and width values
    # First, forward fill joint number to associate defects with joints
    df_copy['joint number'] = df_copy['joint number'].ffill()
    
    # Filter for records that have both length and width values
    defects_df = df_copy[
        df_copy['length [mm]'].notna() & 
        df_copy['width [mm]'].notna()
    ].copy()
    
    # Select only the specified columns
    defect_columns = [
        'log dist. [m]',
        'component / anomaly identification',
        'joint number',
        'up weld dist. [m]',
        'clock',
        'depth [%]',
        'length [mm]',
        'width [mm]',
        'surface location'
    ]
    
    # Check which columns exist in the data
    available_columns = [col for col in defect_columns if col in df_copy.columns]
    
    # Select only available columns
    defects_df = defects_df[available_columns]
    defects_df = defects_df.reset_index().drop(columns = ['index'])

    # Standardize surface location if the column exists
    if 'surface location' in defects_df.columns:
        defects_df['surface location'] = defects_df['surface location'].apply(standardize_surface_location)
    
    return joints_df, defects_df