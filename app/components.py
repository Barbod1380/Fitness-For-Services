"""
UI components for the Pipeline Analysis application.
"""

import streamlit as st
import base64
from datetime import datetime
from app.services.navigation_service import *
from app.ui_components import *

# --- UI COMPONENTS ---

def custom_metric(label, value, description=None):
    """
    Create a custom metric display with value, label, and optional description.
    """
    description_html = (
        f"<div style='font-size:12px;color:#95a5a6;'>{description}</div>" if description else ""
    )
    return f"""
    <div class="custom-metric">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {description_html}
    </div>
    """


def status_badge(text, status):
    """
    Create a colored status badge.
    """
    return f'<span class="status-badge {status}">{text}</span>'


def info_box(text, box_type="info"):
    """
    Create an info/warning/success message box using Streamlit markdown.
    """
    return st.markdown(f'<div class="{box_type}-box">{text}</div>', unsafe_allow_html=True)


def show_step_indicator(active_step):
    """
    Display step progress indicator with visual status.
    """
    steps = ["Upload File", "Map Columns", "Process Data"]
    cols = st.columns(len(steps))

    for i, (col, step_label) in enumerate(zip(cols, steps), start=1):
        with col:
            if i < active_step:
                emoji, color = "✅", "green"
            elif i == active_step:
                emoji, color = "🔵", "blue"
            else:
                emoji, color = "⚪", "gray"

            st.markdown(f"### {emoji} **Step {i}**", unsafe_allow_html=True)
            st.caption(step_label)


# --- LAYOUTS ---

def create_welcome_screen():
    """
    Create the welcome screen shown when no dataset is loaded.
    """
    welcome_html = f"""
    <div class="card-container" style="text-align:center;padding:40px;background-color:#f8f9fa;">
        <h2 style="color:#7f8c8d;margin-bottom:20px;">Welcome to Pipeline Inspection Analysis</h2>
        <p style="color:#95a5a6;margin-bottom:30px;">
            Upload at least one dataset using the sidebar to begin analysis.
        </p>
        <div style="color:#3498db;font-size:2em;">
            <i class="fas fa-arrow-left"></i> Start by uploading a CSV file
        </div>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)

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
    Create the sidebar UI for uploading and managing data.
    
    Returns:
    - uploaded_file: Uploaded CSV file
    - selected_year: Year selected by the user
    """
    with st.sidebar:
        st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)

        if session_state.datasets:
            st.markdown("<strong>Loaded Datasets</strong>", unsafe_allow_html=True)
            for year in sorted(session_state.datasets.keys()):
                st.markdown(
                    f"""
                    <div style="padding:8px 10px;margin-bottom:5px;background-color:#e8f8ef;border-radius:5px;">
                        <span style="color:#2ecc71;margin-right:8px;">✓</span>{year} data loaded
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown('<div style="margin-top:20px;"><strong>Add New Dataset</strong></div>', unsafe_allow_html=True)

        current_year = datetime.now().year
        year_options = list(range(current_year - 30, current_year + 1))
        selected_year = st.selectbox(
            "Select Inspection Year",
            options=year_options,
            index=len(year_options) - 1,
            key="year_selector_sidebar"
        )

        st.markdown(
            f'<div style="margin:10px 0 5px 0;"><strong>Upload {selected_year} Inspection CSV</strong></div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type="csv",
            key=f"file_uploader_{get_state('file_upload_key', 0)}",
            label_visibility="collapsed"
        )

        if uploaded_file and get_current_page() != 'upload':
            set_current_page('upload')

        st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)

        if st.button("Clear All Datasets", key="clear_all_datasets_btn", use_container_width=True):
            session_state.datasets = {}
            session_state.current_year = None
            session_state.file_upload_key += 1
            session_state.active_step = 1
            session_state.comparison_results = None
            session_state.corrected_results = None
            session_state.comparison_years = None
            st.rerun()

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
    Generate a CSV download link for a given DataFrame.
    
    Parameters:
    - df: DataFrame to export
    - prefix: Filename prefix
    - year: Year for filename
    
    Returns:
    - HTML anchor tag with embedded base64 CSV data
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="{prefix}_{year}.csv" '
        f'class="custom-button" '
        f'style="display:inline-block;text-decoration:none;margin-top:10px;'
        f'font-size:0.8em;padding:5px 10px;">Download {prefix} CSV</a>'
    )