"""
Data upload and column mapping view for the Pipeline Analysis application.
"""
import streamlit as st
import pandas as pd
import re
import time

from app.config import ENCODING_OPTIONS
from core.column_mapping import *
from core.data_processing import process_pipeline_data, validate_pipeline_data
from utils.format_utils import float_to_clock
from app.ui_components import show_step_indicator, info_box, create_column_mapping_form
from utils.format_utils import float_to_clock, parse_clock

def load_csv_with_encoding(file):
    """
    Try to load a CSV file with different encodings.
    
    Parameters:
    - file: Uploaded file object
    
    Returns:
    - Tuple of (DataFrame, encoding)
    
    Raises:
    - ValueError if the file cannot be loaded with any encoding
    """
    # This function is copied from main.py
    for encoding in ENCODING_OPTIONS:
        try:
            # Try to read with current encoding
            df = pd.read_csv(
                file, 
                encoding=encoding,
                engine='python',  # More flexible engine
                on_bad_lines='warn'  # Continue despite bad lines
            )
            
            # Check and convert clock column if needed
            if 'clock' in df.columns:
                # Check if any values are numeric (floating point)
                if df['clock'].dtype.kind in 'fi' or any(isinstance(x, (int, float)) for x in df['clock'].dropna()):
                    st.info("Converting numeric clock values to HH:MM format")
                    # Convert numeric values to clock format
                    df['clock'] = df['clock'].apply(
                        lambda x: float_to_clock(float(x)) if pd.notna(x) and isinstance(x, (int, float)) else x
                    )
                
                # For string values that don't look like clock format (HH:MM)
                clock_pattern = re.compile(r'^\d{1,2}:\d{2}$')
                non_standard = df['clock'].apply(
                    lambda x: pd.notna(x) and isinstance(x, str) and not clock_pattern.match(x)
                ).any()
                
                if non_standard:
                    info_box("Some clock values may not be in standard HH:MM format", box_type="warning")
            return df, encoding
            
        except Exception as e:
            continue  # Try next encoding
    
    # If all encodings fail
    raise ValueError(f"Failed to load the file with any of the encodings: {', '.join(ENCODING_OPTIONS)}")

def render_upload_view(uploaded_file, selected_year):
    """
    Display the data upload and processing view.
    
    Parameters:
    - uploaded_file: Uploaded file from Streamlit
    - selected_year: Selected year for the data
    """
    # Extract the upload page content from main.py
    if uploaded_file is None:
        return

    # Create progress indicator for the workflow
    show_step_indicator(st.session_state.active_step)

    # Create a container for the column mapping process
    with st.container():
        # Load the data with robust encoding handling
        try:
            df, successful_encoding = load_csv_with_encoding(uploaded_file)
            if successful_encoding != 'utf-8':
                info_box(f"File loaded with {successful_encoding} encoding. Some special characters may display differently.", "info")
        except ValueError as e:
            info_box(str(e), "warning")
            st.stop()
        
        # Display file info in a card-like container
        with st.expander("File Preview", expanded=True):

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Filename:** {uploaded_file.name}")
            with col2:
                st.markdown(f"**Rows:** {df.shape[0]}")
            with col3:
                st.markdown(f"**Columns:** {df.shape[1]}")
            
            st.dataframe(df.head(100), height=200, use_container_width=True)
        
        # Process data (extracted from main.py's process_data_section function)
        # Column mapping process in a collapsible section
        with st.expander("Column Mapping", expanded=True):
            st.markdown('<div class="section-header">Map Columns for Standardization</div>', unsafe_allow_html=True)
            
            # Update active step
            st.session_state.active_step = 2
            
            # Get suggested column mapping
            suggested_mapping = suggest_column_mapping(df)

            # Create column mapping form
            confirmed_mapping = create_column_mapping_form(df, selected_year, suggested_mapping)

        # Add this after the column mapping expander
        with st.expander("Pipeline Specifications", expanded=True):
            st.markdown('<div class="section-header">Enter Pipeline Parameters</div>', unsafe_allow_html=True)
            
            # Pipe diameter input with a reasonable default and validation
            pipe_diameter = st.number_input(
                "Pipe Diameter (m)",
                min_value=0.1,
                max_value=3.0,  # Reasonable range for pipeline diameters
                step=0.1,
                value=None,
                format="%.3f",
                help="Enter the pipeline diameter in meters"
            )

        # Process and add button
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        process_col1, process_col2 = st.columns([1, 3])
        with process_col1:
            process_button = st.button(
                f"Process {selected_year} Data", 
                key=f"process_data_{selected_year}",
                use_container_width=True
            )
        
        if process_button:
            if(pipe_diameter != None):
                # Update active step
                st.session_state.active_step = 3
                
                with st.spinner(f"Processing {selected_year} data..."):
                    # Apply the mapping to rename columns
                    standardized_df = apply_column_mapping(df, confirmed_mapping)
                    
                    # Process the pipeline data
                    joints_df, defects_df = process_pipeline_data(standardized_df)
                    
                    # Add progress bar for processing
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Simulating work
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
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
                    
                    try:
                        validate_pipeline_data(joints_df, defects_df)
                    except ValueError as e:
                        st.error(f"Data validation failed:\n{e}")
                        st.stop()

                    # Store in session state
                    st.session_state.datasets[selected_year] = {
                        'joints_df': joints_df,
                        'defects_df': defects_df,
                        'pipe_diameter': pipe_diameter  # Store the pipe diameter
                    }
                    st.session_state.current_year = selected_year
                    
                    # Force the file uploader to reset
                    st.session_state.file_upload_key += 1
                    
                    # Show success message and then rerun
                    st.success(f"Successfully processed {selected_year} data")
                    st.rerun()
            else:
                st.warning("Please enter a valid number for pipe diameter.")