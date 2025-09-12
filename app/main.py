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
from app.s3_utils import load_csv_from_s3
from app.services.navigation_service import set_current_page
from app.auth import start_login, complete_new_password


def show_login_page():
    """
    Centered login card with navy/orange decorative accents.
    - No logo
    - Constrained card width so inputs don't span full window
    - Two blurred decorative circles: navy (top-left) and orange (bottom-right)
    - Smooth navy->orange header band and orange login button with hover glow
    - Preserves NEW_PASSWORD_REQUIRED flow and calls to start_login / complete_new_password
    """
    # Page header
    st.markdown(
        "<h2 style='text-align:center; color:var(--navy); margin-bottom:6px'>Pipeline Integrity Assessment</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center; color:#6b7280; margin-bottom:18px'>Secure access for pipeline data analysts</div>",
        unsafe_allow_html=True,
    )

    # NEW_PASSWORD_REQUIRED flow (unchanged)
    if st.session_state.get("auth_challenge") == "NEW_PASSWORD_REQUIRED":
        st.info("First login: set a new password.")
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
                    complete_new_password(new_pw, user_attributes=None)
        return

    # CSS: page theme, decorative accents, card styling
    css = """
    <style>
    :root{
      --navy: #072033;
      --navy-2: #0b2540;
      --orange: #ff7a18;
      --orange-soft: rgba(255,122,24,0.12);
      --muted: #6b7280;
      --card-bg: #ffffff;
      --page-bg: linear-gradient(180deg, #fbfcfd 0%, #fff8f3 100%);
    }
    html, body, .stApp {
      background: var(--page-bg) !important;
    }

    /* Decorative blurred circles behind the card */
    .decorative-bg {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: -1;
      overflow: hidden;
    }

    /* top-left navy */
    .decorative-bg .navy-circle {
      width: 360px;
      height: 360px;
      background: linear-gradient(140deg, var(--navy) 0%, var(--navy-2) 100%);
      top: -140px;
      left: -120px;
      opacity: 0.14;
    }

    /* bottom-right soft orange */
    .decorative-bg .orange-circle {
      width: 420px;
      height: 420px;
      background: linear-gradient(140deg, rgba(255,122,24,0.96) 0%, rgba(255,152,72,0.85) 100%);
      bottom: -160px;
      right: -140px;
      opacity: 0.10;
    }

    /* header band */
    .card-header {
      background: linear-gradient(90deg, rgba(7,32,51,1) 0%, rgba(11,37,64,1) 28%, rgba(255,122,24,0.85) 100%);
      border-top-left-radius:14px;
      border-top-right-radius:14px;
      padding:14px 18px 12px 18px;
      color: white;
      text-align:center;
      font-weight:700;
      font-size:15px;
      letter-spacing:0.3px;
      position: relative;
    }
    /* delicate accent line under header */
    .card-header::after {
      content: "";
      position: absolute;
      left: 8%;
      bottom: -8px;
      width: 84%;
      height: 6px;
      background: rgba(255,255,255,0.06);
      border-radius: 6px;
      filter: blur(4px);
    }

    .card-body {
      padding: 20px 22px 22px 22px;
    }

    .login-tag {
      text-align:center;
      color:var(--muted);
      font-size:13px;
      margin-bottom:14px;
    }

    /* ensure Streamlit inputs fit nicely inside the constrained card */
    .login-card .stTextInput, .login-card .stPasswordInput, .login-card .stCheckbox {
      width: 100% !important;
      max-width: 320px;
    }

    /* Professional button styling */
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, var(--orange) 0%, #ff8c3d 100%);
        color: white;
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
        border: none;
        box-shadow: 0 4px 10px rgba(255, 122, 24, 0.2);
        font-weight: 600;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 122, 24, 0.3);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 5px rgba(255, 122, 24, 0.2);
    }

    .card-footer {
      text-align:center;
      color:#94a3b8;
      font-size:12px;
      margin-top:10px;
      padding-bottom:12px;
    }

    /* responsive tweak */
    @media (max-width: 560px) {
      .login-card { width: 92% !important; margin: 0 4%; }
      .decorative-bg .navy-circle { display: none; }
      .decorative-bg .orange-circle { display: none; }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # decorative circles: navy top-left, soft orange bottom-right
    st.markdown(
        """
        <div class="decorative-bg" aria-hidden="true">
          <div class="circle navy-circle"></div>
          <div class="circle orange-circle"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card wrapper and content
    st.markdown("<div class='login-center'><div class='login-card'>", unsafe_allow_html=True)

    # header band
    st.markdown("<div class='card-header'>Pipeline Analytics — Secure Login</div>", unsafe_allow_html=True)

    # body
    st.markdown("<div class='card-body'>", unsafe_allow_html=True)
    st.markdown("<div class='login-tag'>Calm and modern — access limited to authorized analysts</div>", unsafe_allow_html=True)

    # show auth error if present
    auth_error = st.session_state.get("auth_error")
    if auth_error:
        st.error(auth_error)

    # compact input group to create subtle inner background
    with st.form("login_form"):
        username = st.text_input("Username", value=st.session_state.get("username", ""), placeholder="your.company.username")
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state["show_password"] = False
        pw_type = "password"
        password = st.text_input("Password", type=pw_type, placeholder="Enter your password")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        cols = st.columns([1, 2])
        with cols[0]:
            submitted = st.form_submit_button("Log in")


    # submission handling (preserve your start_login behavior)
    if submitted:
        if not username or not password:
            st.warning("Please provide both username and password.")
        else:
            st.session_state["username"] = username.strip()
            if "auth_error" in st.session_state:
                del st.session_state["auth_error"]
            try:
                start_login(username.strip(), password)
            except Exception:
                st.session_state["auth_error"] = "Authentication service unavailable. Please try again later."

    st.markdown("</div>", unsafe_allow_html=True)  # close card-body
    st.markdown("<div class='card-footer'>© Pipeline Analytics — secure access</div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)  # close card & wrapper

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