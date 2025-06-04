"""
Router service for the Pipeline Analysis application.
"""
import streamlit as st
from app.services.state_manager import get_state
from app.services.navigation_service import get_current_page
from app.views import *

def route_to_current_page(uploaded_file=None, selected_year=None):
    """
    Route to the current page based on session state.
    
    Parameters:
    - uploaded_file: Uploaded file from sidebar
    - selected_year: Selected year from sidebar
    """
    current_page = get_current_page()
    
    # If a file is uploaded, automatically go to the upload page
    if uploaded_file is not None:
        render_upload_view(uploaded_file, selected_year)
        return
    
    # Otherwise, route based on current page
    if current_page == 'home':
        render_home_view()
    elif current_page == 'upload':
        st.info("Please upload a file to continue.")
    elif current_page == 'single_analysis':
        render_single_analysis_view()
    elif current_page == 'comparison':
        render_comparison_view()
    elif current_page == 'corrosion_assessment':  # Add this condition
        render_corrosion_assessment_view()
    elif current_page == 'settings':
        render_settings_view()
    else:
        # Fallback to home if unknown page
        render_home_view()


def render_settings_view():
    """Render the settings page."""
    st.title("Application Settings")
    
    # Create tabs for different settings categories
    settings_tabs = st.tabs(["Visualization", "Analysis", "Interface"])
    
    with settings_tabs[0]:
        st.header("Visualization Settings")
        
        # Colorscale selection
        colorscale = st.selectbox(
            "Default Colorscale",
            options=["Turbo", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "YlOrRd", "YlGnBu"],
            index=0,
            key="settings_colorscale"
        )
        
        # Plot height
        plot_height = st.slider(
            "Default Plot Height",
            min_value=400,
            max_value=800,
            value=600,
            step=50,
            key="settings_plot_height"
        )
        
        # Joint markers
        show_joint_markers = st.checkbox(
            "Show Joint Markers on Pipeline Visualization",
            value=True,
            key="settings_show_joint_markers"
        )
        
        # Save button
        if st.button("Save Visualization Settings", use_container_width=True):
            st.session_state.visualization_settings = {
                'colorscale': colorscale,
                'show_joint_markers': show_joint_markers,
                'plot_height': plot_height
            }
            st.success("Visualization settings saved!")
    
    with settings_tabs[1]:
        st.header("Analysis Settings")
        
        # Analysis settings
        st.subheader("Multi-Year Comparison")
        
        # Default tolerance settings
        distance_tolerance = st.number_input(
            "Default Distance Tolerance (m)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            format="%.3f",
            key="settings_distance_tolerance"
        )
        
        clock_tolerance = st.number_input(
            "Default Clock Position Tolerance (minutes)",
            min_value=0,
            max_value=60,
            value=20,
            step=5,
            key="settings_clock_tolerance"
        )
        
        # KNN settings
        knn_neighbors = st.number_input(
            "Default KNN Neighbors",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="settings_knn_neighbors"
        )
        
        # Save button
        if st.button("Save Analysis Settings", use_container_width=True):
            st.session_state.analysis_settings = {
                'distance_tolerance': distance_tolerance,
                'clock_tolerance': clock_tolerance,
                'knn_neighbors': knn_neighbors
            }
            st.success("Analysis settings saved!")
    
    with settings_tabs[2]:
        st.header("Interface Settings")
        
        # Interface settings
        st.subheader("General")
        
        # Display theme
        theme = st.radio(
            "Display Theme",
            options=["Default", "Light", "Dark"],
            horizontal=True,
            key="settings_theme"
        )
        
        # Data preview size
        preview_rows = st.slider(
            "Data Preview Rows",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            key="settings_preview_rows"
        )
        
        # Save button
        if st.button("Save Interface Settings", use_container_width=True):
            st.session_state.interface_settings = {
                'theme': theme,
                'preview_rows': preview_rows
            }
            st.success("Interface settings saved!")