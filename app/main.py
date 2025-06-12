"""
Main module for the Pipeline Analysis Streamlit application.
"""
import streamlit as st

from app.styles import load_css
from app.ui_components import get_logo_base64, create_sidebar, create_breadcrumb
from app.services.state_manager import initialize_session_state
from app.services.router import route_to_current_page
from app.config import APP_TITLE, APP_SUBTITLE

def run_app():
    """Main function to run the Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="Pipeline Inspection Analysis", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    load_css()
    
    # Logo and title
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="{get_logo_base64()}" style="width:210px;margin-bottom:20px;">
        </div>
        <h1 class="custom-title">{APP_TITLE}</h1>
        <p class="custom-subtitle">{APP_SUBTITLE}</p>
        """, 
        unsafe_allow_html=True
    )
    
    # Create sidebar and get uploaded file
    uploaded_file, selected_year = create_sidebar(st.session_state)
    
    # Add breadcrumb navigation
    create_breadcrumb()
    
    # Route to the current page
    route_to_current_page(uploaded_file, selected_year)

if __name__ == "__main__":
    run_app()