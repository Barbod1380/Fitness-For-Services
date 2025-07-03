"""
Validation utilities for critical pipeline parameters.
"""
import pandas as pd

def create_wall_thickness_lookup(joints_df, validate=True):
    """
    Create a wall thickness lookup dictionary from joints data.
    
    Parameters:
    - joints_df: DataFrame with joint information
    - validate: Whether to validate data completeness
    
    Returns:
    - Dictionary mapping joint number to wall thickness
    
    Raises:
    - ValueError if required columns missing or data incomplete
    """
    # Check required columns
    required_cols = ['joint number', 'wt nom [mm]']
    missing_cols = [col for col in required_cols if col not in joints_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in joints data: {missing_cols}")
    
    # Check for any missing wall thickness values
    missing_wt = joints_df[joints_df['wt nom [mm]'].isna()]
    if not missing_wt.empty and validate:
        missing_joints = missing_wt['joint number'].tolist()
        raise ValueError(
            f"CRITICAL: {len(missing_wt)} joints have missing wall thickness values. "
            f"Joint numbers: {missing_joints[:10]}{'...' if len(missing_joints) > 10 else ''}. "
            f"Cannot proceed with analysis without complete wall thickness data."
        )
    
    # Create lookup dictionary
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    # Validate all values are positive
    if validate:
        for joint, wt in wt_lookup.items():
            if pd.notna(wt) and wt <= 0:
                raise ValueError(f"Invalid wall thickness {wt} for joint {joint}. Must be positive.")
    
    return wt_lookup


def get_wall_thickness_for_defect(defect_row, wt_lookup):
    """
    Get wall thickness for a specific defect.
    
    Parameters:
    - defect_row: Series or dict containing defect data with 'joint number'
    - wt_lookup: Wall thickness lookup dictionary
    
    Returns:
    - Wall thickness in mm
    
    Raises:
    - ValueError if wall thickness cannot be determined
    """
    if 'joint number' not in defect_row:
        raise ValueError("Defect missing 'joint number' - cannot determine wall thickness")
    
    joint_number = defect_row['joint number']
    if pd.isna(joint_number):
        raise ValueError(f"Defect at {defect_row.get('log dist. [m]', 'unknown')}m has no joint number")
    
    wall_thickness = wt_lookup.get(joint_number)
    if wall_thickness is None or pd.isna(wall_thickness):
        raise ValueError(
            f"No wall thickness found for joint {joint_number} "
            f"(defect at {defect_row.get('log dist. [m]', 'unknown')}m)"
        )
    
    return wall_thickness