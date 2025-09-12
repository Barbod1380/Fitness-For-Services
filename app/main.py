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

from app.s3_utils import load_csv_from_s3
from app.services.navigation_service import set_current_page

from app.auth import start_login, complete_new_password

def show_login_page():
    st.title("Welcome to the Pipeline Integrity Assessment Platform")

    # If Cognito returned NEW_PASSWORD_REQUIRED
    if st.session_state.get("auth_challenge") == "NEW_PASSWORD_REQUIRED":
        st.info("First login detected. Please set a new password to continue.")
        with st.form("new_password_form"):
            new_pw = st.text_input("New password", type="password")
            confirm_pw = st.text_input("Confirm new password", type="password")
            submitted = st.form_submit_button("Update password")
            if submitted:
                if not new_pw or not confirm_pw:
                    st.warning("Please fill both fields.")
                elif new_pw != confirm_pw:
                    st.error("Passwords do not match.")
                else:
                    # Provide required attributes if your pool enforces them at first login
                    complete_new_password(new_pw, user_attributes=None)
        return

    # Normal login form
    st.header("Please Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                start_login(username, password)


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

    if not st.session_state.get('logged_in', False):
        # If not logged in, show the login page
        show_login_page()
    else:
        # If logged in, show the main application
        # Create professional header
        create_professional_header()

        # Create professional sidebar
        selected_year, selected_file = create_professional_sidebar(st.session_state)

        # This is where the data loading will be triggered.
        if selected_year and selected_file:
            current_selection = (selected_year, selected_file)
            # Load data only if the selection has changed
            if st.session_state.get('selection_details') != current_selection:
                with st.spinner(f"Loading {selected_file} from S3..."):
                    username = st.session_state.get('username')
                    df = load_csv_from_s3(username, selected_year, selected_file)
                    if df is not None:
                        st.session_state.raw_df_to_process = df
                        st.session_state.selection_details = current_selection
                        # Navigate to the processing page
                        set_current_page('upload')
                        st.rerun()

        # Add professional breadcrumb navigation
        create_professional_breadcrumb()

        # Route to the current page. The router no longer handles file uploads directly.
        route_to_current_page(None, selected_year)

if __name__ == "__main__":
    run_app()