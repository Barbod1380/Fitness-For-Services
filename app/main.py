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


# Additional utility functions for professional UI
def create_professional_container(content_func, title=None, status=None):
    """Wrapper for creating professional content containers."""
    
    container_html = '<div class="card-container animate-in">'
    
    if title:
        status_indicator = ""
        if status:
            from app.styles import get_status_indicator
            status_indicator = f' {get_status_indicator(status)}'
        
        container_html += f'<h3 class="section-header">{title}{status_indicator}</h3>'
    
    st.markdown(container_html, unsafe_allow_html=True)
    
    # Execute the content function
    content_func()
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_professional_warning(message, title="Important Notice"):
    """Show professional warning message."""
    from app.styles import professional_alert
    alert_html = professional_alert(f"<strong>{title}:</strong> {message}", "warning")
    st.markdown(alert_html, unsafe_allow_html=True)


def show_professional_success(message, title="Success"):
    """Show professional success message."""
    from app.styles import professional_alert
    alert_html = professional_alert(f"<strong>{title}:</strong> {message}", "success")
    st.markdown(alert_html, unsafe_allow_html=True)


def show_professional_error(message, title="Error"):
    """Show professional error message."""
    from app.styles import professional_alert
    alert_html = professional_alert(f"<strong>{title}:</strong> {message}", "error")
    st.markdown(alert_html, unsafe_allow_html=True)


def show_professional_info(message, title="Information"):
    """Show professional info message."""
    from app.styles import professional_alert
    alert_html = professional_alert(f"<strong>{title}:</strong> {message}", "info")
    st.markdown(alert_html, unsafe_allow_html=True)