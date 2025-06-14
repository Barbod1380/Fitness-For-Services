"""
Welcome screen components for the Pipeline Analysis application.
"""
import streamlit as st

def create_welcome_screen():
    """
    Create a welcome screen for when no data is loaded.
    
    Returns:
    - Streamlit markdown element
    """
    welcome_html = f"""
    <div class="card-container" style="text-align:center;padding:40px;background-color:#f8f9fa;">
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