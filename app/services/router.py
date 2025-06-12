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
    else:
        # Fallback to home if unknown page
        render_home_view()