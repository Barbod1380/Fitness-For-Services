"""
UI components for the Pipeline Analysis application.
"""
import streamlit as st
import base64
from datetime import datetime
from app.services.navigation_service import *

def custom_metric(label, value, description=None):
    """
    Create a custom metric display with a value and label.
    
    Parameters:
    - label: Metric name
    - value: Metric value
    - description: Optional description text
    
    Returns:
    - HTML string for the metric
    """
    metric_html = f"""
    <div class="custom-metric">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {f"<div style='font-size:12px;color:#95a5a6;'>{description}</div>" if description else ""}
    </div>
    """
    return metric_html

def status_badge(text, status):
    """
    Create a colored status badge.
    
    Parameters:
    - text: Badge text
    - status: Badge color/status (green, yellow, red)
    
    Returns:
    - HTML string for the badge
    """
    badge_html = f"""
    <span class="status-badge {status}">{text}</span>
    """
    return badge_html

def info_box(text, box_type="info"):
    """
    Create an info, warning, or success box.
    
    Parameters:
    - text: Box content
    - box_type: Box style (info, warning, success)
    
    Returns:
    - Streamlit markdown element
    """
    box_class = f"{box_type}-box"
    box_html = f"""
    <div class="{box_class}">
        {text}
    </div>
    """
    return st.markdown(box_html, unsafe_allow_html=True)

def show_step_indicator(active_step):
    """
    Display a step progress indicator.
    
    Parameters:
    - active_step: Current active step (1-based index)
    """
    steps = ["Upload File", "Map Columns", "Process Data"]
    cols = st.columns(len(steps))
    
    for i, (col, step_label) in enumerate(zip(cols, steps), 1):
        with col:
            if i < active_step:
                emoji = "✅"  # Completed
                color = "green"
            elif i == active_step:
                emoji = "🔵"  # Active
                color = "blue"
            else:
                emoji = "⚪"  # Not started
                color = "gray"
            
            st.markdown(f"### {emoji} **Step {i}**", unsafe_allow_html=True)
            st.caption(step_label)

def get_logo_base64():
    """
    Get a base64-encoded SVG logo for the application.
    
    Returns:
    - Base64-encoded SVG string
    """
    # Pipeline logo placeholder
    return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgIDxjaXJjbGUgZmlsbD0iIzM0OThkYiIgY3g9IjEwMCIgY3k9IjEwMCIgcj0iMTAwIi8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iNzAiIHdpZHRoPSIxMjAiIGhlaWdodD0iMjAiIHJ4PSI1Ii8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iMTEwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjIwIiByeD0iNSIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iNTUiIGN5PSI4MCIgcj0iOCIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iMTQwIiBjeT0iMTIwIiByPSI4Ii8+CiAgPC9nPgo8L3N2Zz4K"

def create_welcome_screen():
    """
    Create a welcome screen for when no data is loaded.
    
    Returns:
    - Streamlit markdown element
    """
    welcome_html = f"""
    <div class="card-container" style="text-align:center;padding:40px;background-color:#f8f9fa;">
        <img src="{get_logo_base64()}" style="width:120px;margin-bottom:20px;">
        <h2 style="color:#7f8c8d;margin-bottom:20px;">Welcome to Pipeline Inspection Analysis</h2>
        <p style="color:#95a5a6;margin-bottom:30px;">Upload at least one dataset using the sidebar to begin analysis.</p>
        <div style="color:#3498db;font-size:2em;"><i class="fas fa-arrow-left"></i> Start by uploading a CSV file</div>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)
    
    # Add a quick guide
    guide_html = """
    <div class="card-container" style="margin-top:20px;">
        <div class="section-header">Quick Guide</div>
        <ol style="padding-left:20px;">
            <li><strong>Upload Data:</strong> Use the sidebar to upload pipeline inspection CSV files</li>
            <li><strong>Map Columns:</strong> Match your file's columns to standard names</li>
            <li><strong>Analyze:</strong> View statistics and visualizations for your pipeline data</li>
            <li><strong>Compare:</strong> Upload multiple years to track defect growth over time</li>
        </ol>
        <div class="section-header" style="margin-top:20px;">Supported Features</div>
        <ul style="padding-left:20px;">
            <li>Statistical analysis of defect dimensions</li>
            <li>Unwrapped pipeline visualizations</li>
            <li>Joint-by-joint defect analysis</li>
            <li>Multi-year comparison with growth rate calculation</li>
            <li>New defect identification</li>
        </ul>
    </div>
    """
    st.markdown(guide_html, unsafe_allow_html=True)

def create_sidebar(session_state):
    """
    Create the application sidebar for data management.
    
    Parameters:
    - session_state: Streamlit session state
    """
    with st.sidebar:
        st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)
        
        # Display currently loaded datasets
        if session_state.datasets:
            st.markdown("<strong>Loaded Datasets</strong>", unsafe_allow_html=True)
            for year in sorted(session_state.datasets.keys()):
                st.markdown(
                    f'<div style="padding:8px 10px;margin-bottom:5px;background-color:#e8f8ef;border-radius:5px;">'
                    f'<span style="color:#2ecc71;margin-right:8px;">✓</span>{year} data loaded'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        
        st.markdown('<div style="margin-top:20px;"><strong>Add New Dataset</strong></div>', unsafe_allow_html=True)
        
        # Year selection for new data
        current_year = datetime.now().year
        year_options = list(range(current_year - 30, current_year + 1))
        selected_year = st.selectbox(
            "Select Inspection Year", 
            options=year_options,
            index=len(year_options) - 1,  # Default to current year
            key="year_selector_sidebar"
        )
        
        # File uploader with improved styling
        # File uploader with improved styling
        st.markdown(f'<div style="margin:10px 0 5px 0;"><strong>Upload {selected_year} Inspection CSV</strong></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type="csv",
            key=f"file_uploader_{get_state('file_upload_key', 0)}",
            label_visibility="collapsed"
        )        

        if uploaded_file is not None:
            # Only change page if it's not already on the upload page
            if get_current_page() != 'upload':
                set_current_page('upload')
        
        # Button to clear all data with better styling
        st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
        if st.button("Clear All Datasets", key="clear_all_datasets_btn", use_container_width=True):
            session_state.datasets = {}
            session_state.current_year = None
            session_state.file_upload_key += 1  # Force file uploader to reset
            session_state.active_step = 1
            session_state.comparison_results = None
            session_state.corrected_results = None
            session_state.comparison_years = None
            st.rerun()
        
        # Add a footer with app info
        st.markdown(
            """
            <div class="footer">
                <p>Pipeline Inspection Analysis Tool<br>Version 1.0</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        return uploaded_file, selected_year

def create_data_download_links(df, prefix, year):
    """
    Create download links for dataframes.
    
    Parameters:
    - df: DataFrame to download
    - prefix: Prefix for the filename
    - year: Year to include in the filename
    
    Returns:
    - HTML string with the download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{prefix}_{year}.csv" class="custom-button" style="display:inline-block;text-decoration:none;margin-top:10px;font-size:0.8em;padding:5px 10px;">Download {prefix} CSV</a>'
    return href