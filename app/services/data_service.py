"""
Data service for managing pipeline data.
"""
import pandas as pd
import re
import streamlit as st
from app.services.state_manager import get_state, update_state, add_dataset
from app.ui_components import info_box
from core.column_mapping import apply_column_mapping
from core.data_processing import process_pipeline_data
from utils.format_utils import float_to_clock, parse_clock

def process_dataset(df, column_mapping, pipe_diameter, year):
    """
    Process a dataset with column mapping and store in session state.
    
    Parameters:
    - df: DataFrame with the raw data
    - column_mapping: Dictionary mapping file columns to standard columns
    - pipe_diameter: Pipe diameter value
    - year: Year for the dataset
    
    Returns:
    - Boolean indicating success
    """
    try:
        # Apply the mapping to rename columns
        standardized_df = apply_column_mapping(df, column_mapping)
        
        # Process the pipeline data
        joints_df, defects_df = process_pipeline_data(standardized_df)
        
        # Process clock and area data
        if 'clock' in defects_df.columns:
            # First ensure all clock values are in string format
            defects_df['clock'] = defects_df['clock'].astype(str)
            
            # Check if string values don't match the expected format
            clock_pattern = re.compile(r'^\d{1,2}:\d{2}$')
            non_standard = defects_df['clock'].apply(
                lambda x: pd.notna(x) and not clock_pattern.match(x) and x != 'nan'
            ).any()
            
            if non_standard:
                info_box("Some clock values may not be in standard HH:MM format. These will be handled as NaN.", "warning")
                # Try to fix non-standard formats
                defects_df['clock'] = defects_df['clock'].apply(
                    lambda x: float_to_clock(float(x)) if pd.notna(x) and x != 'nan' and not clock_pattern.match(x) else x
                )
            
            # Now convert to float for visualization
            defects_df["clock_float"] = defects_df["clock"].apply(parse_clock)
        
        if 'length [mm]' in defects_df.columns and 'width [mm]' in defects_df.columns:
            defects_df["area_mm2"] = defects_df["length [mm]"] * defects_df["width [mm]"]
        
        if 'joint number' in defects_df.columns:
            defects_df["joint number"] = defects_df["joint number"].astype("Int64")
        
        # Store in session state
        add_dataset(year, joints_df, defects_df, pipe_diameter)
        
        # Update active step
        update_state('active_step', 3)
        
        return True
        
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        return False

def get_dataset(year):
    """
    Get a dataset for a specific year.
    
    Parameters:
    - year: Year to retrieve
    
    Returns:
    - Dictionary with joints_df, defects_df, and pipe_diameter
    """
    datasets = get_state('datasets', {})
    return datasets.get(year, None) # type: ignore

def get_available_years():
    """
    Get a list of years with available datasets.
    
    Returns:
    - List of years sorted in ascending order
    """
    datasets = get_state('datasets', {})
    return sorted(datasets.keys()) # type: ignore

def get_defect_dimensions(defects_df):
    """
    Get available dimension columns in a defects DataFrame.
    
    Parameters:
    - defects_df: DataFrame with defect data
    
    Returns:
    - Dictionary with dimension availability
    """
    dimensions = {
        'depth': 'depth [%]' in defects_df.columns,
        'length': 'length [mm]' in defects_df.columns,
        'width': 'width [mm]' in defects_df.columns,
        'clock': 'clock' in defects_df.columns
    }
    return dimensions