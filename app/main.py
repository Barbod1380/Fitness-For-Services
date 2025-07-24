"""
Updated main module for the Professional Pipeline Analysis Streamlit application.
"""
import streamlit as st

# Import the new professional styling
from app.styles import load_css, apply_navigation_styles
from app.ui_components.navigation import (
    create_professional_header, 
    create_professional_sidebar, 
    create_professional_breadcrumb
)
from app.services.state_manager import initialize_session_state
from app.services.router import route_to_current_page
from app.config import APP_TITLE, APP_SUBTITLE

def run_app():
    """Main function to run the Professional Pipeline FFS Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Professional page configuration
    st.set_page_config(
        page_title="Pipeline Integrity FFS - Professional Assessment Platform", 
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Professional Pipeline Integrity Assessment Platform"
        }
    )
    
    # Apply professional CSS styling
    load_css()
    apply_navigation_styles()
    
    # Create professional header
    create_professional_header()
    
    # Create professional sidebar and get uploaded file
    uploaded_file, selected_year = create_professional_sidebar(st.session_state)
    
    # Add professional breadcrumb navigation
    create_professional_breadcrumb()
    
    # Route to the current page
    route_to_current_page(uploaded_file, selected_year)

if __name__ == "__main__":
    run_app()