import pandas as pd
import numpy as np

def process_pipeline_data(df):
    """
    Process the pipeline inspection data into two separate tables:
    1. joints_df: Contains unique joint information
    2. defects_df: Contains defect information with joint associations

    Parameters:
    - df: pandas.DataFrame with the raw pipeline data

    Returns:
    - joints_df: pandas.DataFrame with joint information
    - defects_df: pandas.DataFrame with defect information
    """    

    # Use view instead of copy for initial processing
    df_view = df.copy()     

    # Replace empty strings efficiently
    string_cols = df_view.select_dtypes(include=['object']).columns
    df_view[string_cols] = df_view[string_cols].replace(r'^\s*$', np.nan, regex=True)
    
    # === Batch numeric conversion ===
    numeric_columns = [
        "joint number", "joint length [m]", "wt nom [mm]", 
        "up weld dist. [m]", "depth [%]", "length [mm]", "width [mm]"
    ]
    
    # Convert multiple columns at once
    existing_numeric_cols = [col for col in numeric_columns if col in df_view.columns]
    if existing_numeric_cols:
        df_view[existing_numeric_cols] = df_view[existing_numeric_cols].apply(pd.to_numeric, errors = 'coerce')
    
    # === Efficient sorting ===
    if "log dist. [m]" in df_view.columns:
        df_view.sort_values("log dist. [m]", inplace = True)
        df_view.reset_index(drop = True, inplace = True)
    
    # === Optimized joints DataFrame creation ===
    joint_columns = ["log dist. [m]", "joint number", "joint length [m]", "wt nom [mm]"]
    existing_joint_cols = [col for col in joint_columns if col in df_view.columns]
    
    if existing_joint_cols and "joint number" in df_view.columns:
        # Use boolean indexing for efficiency
        has_joint_num = df_view["joint number"].notna()
        joints_df = df_view.loc[has_joint_num, existing_joint_cols].copy()
        
        # Remove duplicates efficiently
        joints_df.drop_duplicates(subset=["joint number"], inplace=True)
        joints_df.reset_index(drop=True, inplace=True)
    else:
        joints_df = pd.DataFrame(columns=joint_columns)
    
    # === Efficient forward fill ===
    if "joint number" in df_view.columns:
        df_view = df_view.copy()
        df_view["joint number"] = df_view["joint number"].ffill()
    
    # === Optimized defects DataFrame creation ===
    length_width_cols = ["length [mm]", "width [mm]"]
    has_dimensions = all(col in df_view.columns for col in length_width_cols)
    
    if has_dimensions:
        # Boolean indexing for defects
        has_both_dims = (df_view["length [mm]"].notna() & df_view["width [mm]"].notna())
        
        defect_columns = [
            "log dist. [m]", "component / anomaly identification", "joint number",
            "up weld dist. [m]", "clock", "depth [%]", "length [mm]", "width [mm]", "surface location"
        ]
        existing_defect_cols = [col for col in defect_columns if col in df_view.columns]
        
        defects_df = df_view.loc[has_both_dims, existing_defect_cols].copy()
        defects_df.reset_index(drop=True, inplace=True)
    else:
        defects_df = pd.DataFrame()
    
    # === Vectorized surface location standardization ===
    if "surface location" in defects_df.columns:
        # Use map for efficient categorical conversion
        surface_mapping = {
            'INT': 'INT', 'I': 'INT', 'INTERNAL': 'INT', 'YES': 'INT', 'INTERNE': 'INT',
            'NON-INT': 'NON-INT', 'E': 'NON-INT', 'EXTERNAL': 'NON-INT', 
            'NO': 'NON-INT', 'NON INT': 'NON-INT', 'EXTERNE': 'NON-INT'
        }
        
        defects_df['surface location'] = (
            defects_df['surface location']
            .str.strip().str.upper()
            .map(surface_mapping)
            .fillna(defects_df['surface location'])
        )
    
    return joints_df, defects_df


# Add to core/data_processing.py
def validate_pipeline_data(joints_df, defects_df):
    """
    Validate that pipeline data is complete and consistent.
    """
    errors = []
    
    # Check joints have wall thickness
    if 'wt nom [mm]' not in joints_df.columns:
        errors.append("Joints data missing 'wt nom [mm]' column")
    else:
        missing_wt = joints_df[joints_df['wt nom [mm]'].isna()]
        if not missing_wt.empty:
            errors.append(f"{len(missing_wt)} joints have missing wall thickness")
    
    # Check all defects have joint assignments
    if 'joint number' not in defects_df.columns:
        errors.append("Defects data missing 'joint number' column")
    else:
        missing_joints = defects_df[defects_df['joint number'].isna()]
        if not missing_joints.empty:
            errors.append(f"{len(missing_joints)} defects have no joint number")
        
        # Check all defect joints exist in joints_df
        defect_joints = set(defects_df['joint number'].dropna().unique())
        joint_numbers = set(joints_df['joint number'].unique())
        orphan_joints = defect_joints - joint_numbers
        if orphan_joints:
            errors.append(f"Defects reference non-existent joints: {list(orphan_joints)[:5]}")
    
    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(errors))
    
    return True