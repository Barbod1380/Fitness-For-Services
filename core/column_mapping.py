"""
Functions and constants for mapping columns from CSV files to standard formats.
"""
import pandas as pd
from fuzzywuzzy import process  # For fuzzy string matching

# Define standard columns from the first format
STANDARD_COLUMNS = [
    'log dist. [m]',
    'component / anomaly identification',
    'joint number',
    'joint length [m]',
    'wt nom [mm]',
    'up weld dist. [m]',
    'clock',
    'depth [%]',
    'ERF B31G',
    'length [mm]',
    'width [mm]',
    'surface location'
]

# Define common variants for each standard column
COLUMN_VARIANTS = {
    'log dist. [m]': ['log distance', 'distance', 'chainage', 'position'],
    'component / anomaly identification': ['event', 'anomaly', 'defect type', 'feature', 'feature type'],
    'joint number': ['J. no.', 'J.no.', 'joint #', 'Joint No', 'joint_number'],
    'joint length [m]': ['J. len [m]', 'joint length', 'length [m]', 'Joint Length', 'pipe length'],
    'wt nom [mm]': ['t [mm]', 'thickness', 'wall thickness', 'nominal thickness', 'WT'],
    'up weld dist. [m]': ['to u/s w. [m]', 'upstream weld distance', 'distance to upstream weld', 'up weld'],
    'clock': ['o\'clock', 'oclock', 'clock position', 'angular position'],
    'depth [%]': ['depth percent', 'depth percentage', 'defect depth'],
    'ERF B31G': ['ERF_AS2885', 'ERF', 'B31G', 'expansion_ratio_factor'],
    'length [mm]': ['defect length', 'anomaly length', 'feature length'],
    'width [mm]': ['defect width', 'anomaly width', 'feature width'],
    'surface location': ['internal', 'external', 'location', 'orientation']
}

# Known exact mappings from format 2 to format 1 (standard)
KNOWN_MAPPINGS = {
    'event': 'component / anomaly identification',
    'J. no.': 'joint number',
    'J. len [m]': 'joint length [m]',
    't [mm]': 'wt nom [mm]',
    'to u/s w. [m]': 'up weld dist. [m]',
    'o\'clock': 'clock',
    'ERF_AS2885': 'ERF B31G',
    'internal': 'surface location'
}

# Required columns for processing (subset of standard columns)
REQUIRED_COLUMNS = [
    'log dist. [m]',
    'joint number',
    'joint length [m]',
    'wt nom [mm]',
    'component / anomaly identification',
    'up weld dist. [m]',
    'clock',
    'depth [%]',
    'length [mm]',
    'width [mm]',
    'surface location'
]

def suggest_column_mapping(df):
    """
    Suggest mapping between uploaded columns and standard columns.
    
    Parameters:
    - df: DataFrame with columns to be mapped
    
    Returns:
    - A dictionary of {standard_column: suggested_match}
    """
    file_columns = df.columns.tolist()
    mapping = {}
    
    for std_col in STANDARD_COLUMNS:
        # Check if the standard column exists in the file (exact match)
        if std_col in file_columns:
            mapping[std_col] = std_col
            continue
        
        # Check if there's a known mapping from the second format
        for file_col, standard in KNOWN_MAPPINGS.items():
            if standard == std_col and file_col in file_columns:
                mapping[std_col] = file_col
                break
                
        # If still no match, check for variants
        if std_col not in mapping and std_col in COLUMN_VARIANTS:
            for variant in COLUMN_VARIANTS[std_col]:
                if variant in file_columns:
                    mapping[std_col] = variant
                    break
                    
        # If still no match, use fuzzy matching as a fallback
        if std_col not in mapping:
            # Get the best match with score
            match, score = process.extractOne(std_col, file_columns)
            if score > 70:  # Threshold for acceptable match
                mapping[std_col] = match
            else:
                mapping[std_col] = None  # No good match found
                
    return mapping

def apply_column_mapping(df, mapping):
    """
    Apply the confirmed mapping to rename columns.
    
    Parameters:
    - df: Original DataFrame
    - mapping: Dictionary of {standard_column: file_column}
    
    Returns:
    - DataFrame with standardized column names
    """
    # Create a new DataFrame to avoid modifying the original
    renamed_df = df.copy()
    
    # Add columns based on the mapping
    for std_col, file_col in mapping.items():
        if file_col is not None and file_col in df.columns:
            renamed_df[std_col] = df[file_col]
    
    return renamed_df

def get_missing_required_columns(mapping):
    """
    Get a list of required columns that don't have a mapping.
    
    Parameters:
    - mapping: Dictionary of {standard_column: file_column}
    
    Returns:
    - List of standard column names that are required but don't have a mapping
    """
    missing = []
    for col in REQUIRED_COLUMNS:
        if col not in mapping or mapping[col] is None:
            missing.append(col)
    return missing