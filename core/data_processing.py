# data_processing.py


import pandas as pd
import numpy as np
from utils.format_utils import standardize_surface_location


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
    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # 1. Replace empty strings with NaN for proper handling
    df_copy = df_copy.replace(r'^\s*$', np.nan, regex=True)

    # 2. Convert numeric columns to appropriate types
    numeric_columns = [
        "joint number",
        "joint length [m]",
        "wt nom [mm]",
        "up weld dist. [m]",
        "depth [%]",
        "length [mm]",
        "width [mm]",
    ]
    for col in numeric_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    # 3. Sort by log distance to ensure proper order for forward-fill
    if "log dist. [m]" in df_copy.columns:
        df_copy = df_copy.sort_values("log dist. [m]").reset_index(drop=True)

    # 4. Create joints_df with only the specified columns
    if {"log dist. [m]", "joint number", "joint length [m]", "wt nom [mm]"}.issubset(df_copy.columns):
        joints_df = df_copy.loc[
            df_copy["joint number"].notna(),
            ["log dist. [m]", "joint number", "joint length [m]", "wt nom [mm]"],
        ].copy()
    else:
        joints_df = pd.DataFrame(
            columns=["log dist. [m]", "joint number", "joint length [m]", "wt nom [mm]"]
        )

    # 5. Drop duplicate joint numbers if any
    joints_df = joints_df.drop_duplicates(subset=["joint number"]).reset_index(drop=True)

    # 6. Forward-fill joint number to associate defects with joints
    if "joint number" in df_copy.columns:
        df_copy["joint number"] = df_copy["joint number"].ffill()

    # 7. Create defects_df - records with both length and width values
    if {"length [mm]", "width [mm]"}.issubset(df_copy.columns):
        defects_df = df_copy[
            df_copy["length [mm]"].notna() & df_copy["width [mm]"].notna()
        ].copy()
    else:
        defects_df = pd.DataFrame(columns=[])

    # Select only the specified columns (if they exist)
    defect_columns = [
        "log dist. [m]",
        "component / anomaly identification",
        "joint number",
        "up weld dist. [m]",
        "clock",
        "depth [%]",
        "length [mm]",
        "width [mm]",
        "surface location",
    ]
    available_columns = [col for col in defect_columns if col in df_copy.columns]
    defects_df = defects_df[available_columns].reset_index(drop=True)

    # Standardize surface location if the column exists
    if "surface location" in defects_df.columns:
        defects_df["surface location"] = defects_df["surface location"].apply(
            standardize_surface_location
        )

    return joints_df, defects_df