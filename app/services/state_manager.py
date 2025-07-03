"""
Centralized session state management for the Pipeline Analysis application.
"""
import streamlit as st
import json

# Define default state values
DEFAULT_STATE = {
    'datasets': {},
    'current_year': None,
    'file_upload_key': 0,
    'active_step': 1,
    'comparison_results': None,
    'corrected_results': None,
    'comparison_years': None,
    'form_submitted': False,
    'comparison_viz_tab': 0,
    'correction_dimension': 'depth',
    'growth_analysis_dimension': 'depth',
    'current_page': 'home',
    'analysis_tabs': {
        'single_year': 0,
        'multi_year': 0
    },
    'filter_settings': {
        'depth': {'apply': False, 'min': 0, 'max': 100},
        'length': {'apply': False, 'min': 0, 'max': 1000},
        'width': {'apply': False, 'min': 0, 'max': 1000},
        'defect_type': 'All Types'
    },
    'visualization_settings': {
        'colorscale': 'Turbo',
        'show_joint_markers': True,
        'plot_height': 600
    }
}

def initialize_session_state():
    """Initialize or update session state variables with defaults."""
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_state(key, default=None):
    """
    Get a value from session state with a default fallback.
    
    Parameters:
    - key: The state key to retrieve
    - default: Default value if key doesn't exist
    
    Returns:
    - The state value or default
    """
    # If the key exists in session state, return it
    if key in st.session_state:
        return st.session_state[key]
    # If a default is provided, return it
    if default is not None:
        return default
    # Otherwise, return the DEFAULT_STATE value if it exists
    if key in DEFAULT_STATE:
        return DEFAULT_STATE[key]
    # As a last resort, return None
    return None

def update_state(key, value, validate=True):
    """
    Update a value in session state with optional validation.
    
    Parameters:
    - key: The state key to update
    - value: The new value
    - validate: Whether to validate the update
    
    Returns:
    - True if update was successful, False otherwise
    """
    # Validate the update if requested
    if validate:
        # Perform basic type validation
        if key in DEFAULT_STATE and not isinstance(value, type(DEFAULT_STATE[key])):
            # Special case for None values
            if value is not None or DEFAULT_STATE[key] is not None:
                print(f"Warning: Type mismatch for key '{key}'. Expected {type(DEFAULT_STATE[key])}, got {type(value)}")
                return False
    
    # Update the state
    st.session_state[key] = value
    return True

def reset_state():
    """Reset session state to defaults."""
    for key, default_value in DEFAULT_STATE.items():
        st.session_state[key] = default_value

def get_nested_state(path, default=None):
    """
    Get a nested state value using a dot-notation path.
    
    Parameters:
    - path: Dot-notation path (e.g., 'filter_settings.depth.min')
    - default: Default value if path doesn't resolve
    
    Returns:
    - The resolved value or default
    """
    parts = path.split('.')
    current = st.session_state
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    
    return current

def update_nested_state(path, value):
    """
    Update a nested state value using a dot-notation path.
    
    Parameters:
    - path: Dot-notation path (e.g., 'filter_settings.depth.min')
    - value: The new value
    
    Returns:
    - True if update was successful, False otherwise
    """
    parts = path.split('.')
    target_key = parts[-1]
    
    # Navigate to the parent object
    current = st.session_state
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    
    # Update the value
    if isinstance(current, dict):
        current[target_key] = value
        return True
    
    return False

def add_dataset(year, joints_df, defects_df, pipe_diameter):
    """
    Add a dataset to the session state.
    
    Parameters:
    - year: Dataset year
    - joints_df: Joints DataFrame
    - defects_df: Defects DataFrame
    - pipe_diameter: Pipe diameter value
    """
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    
    st.session_state.datasets[year] = {
        'joints_df': joints_df,
        'defects_df': defects_df,
        'pipe_diameter': pipe_diameter
    }
    st.session_state.current_year = year
    st.session_state.file_upload_key += 1  # Force file uploader to reset

def clear_datasets():
    """Clear all datasets from the session state."""
    st.session_state.datasets = {}
    st.session_state.current_year = None
    st.session_state.file_upload_key += 1  # Force file uploader to reset
    st.session_state.active_step = 1
    st.session_state.comparison_results = None
    st.session_state.corrected_results = None
    st.session_state.comparison_years = None

def export_state():
    """
    Export the current state as a JSON string (excluding large datasets).
    
    Returns:
    - JSON string representation of the exportable state
    """
    # Create a copy of the state with only the keys we want to export
    exportable_state = {
        'active_step': st.session_state.active_step,
        'current_page': st.session_state.current_page,
        'analysis_tabs': st.session_state.analysis_tabs,
        'filter_settings': st.session_state.filter_settings,
        'visualization_settings': st.session_state.visualization_settings,
        'dataset_years': list(st.session_state.datasets.keys()) if hasattr(st.session_state, 'datasets') else []
    }
    
    return json.dumps(exportable_state)