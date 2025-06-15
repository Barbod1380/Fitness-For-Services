"""
Professional navigation components for Pipeline FFS application.
"""
import streamlit as st
from datetime import datetime
from app.services.state_manager import get_state, clear_datasets
from app.services.navigation_service import (
    get_navigation_items, set_current_page, get_breadcrumb_items
)
from app.config import APP_VERSION

def get_logo_base64():
    """Get base64 encoded logo or return SVG fallback."""
    import base64
    try:
        with open("assets/logo-pica.png", "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
            return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        # Professional SVG fallback for oil & gas industry
        return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGRlZnM+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImdyYWQiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMEExNjI4IiBzdG9wLW9wYWNpdHk9IjEiIC8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzFFNDBBRiIgc3RvcC1vcGFjaXR5PSIxIiAvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICA8L2RlZnM+CiAgPGNpcmNsZSBjeD0iMTAwIiBjeT0iMTAwIiByPSI5MCIgZmlsbD0idXJsKCNncmFkKSIgc3Ryb2tlPSIjRUE1ODBDIiBzdHJva2Utd2lkdGg9IjQiLz4KICA8IS0tIFBpcGVsaW5lIHJlcHJlc2VudGF0aW9uIC0tPgogIDxyZWN0IHg9IjMwIiB5PSI4NSIgd2lkdGg9IjE0MCIgaGVpZ2h0PSIzMCIgZmlsbD0iI0Y4RkFGQyIgcng9IjE1Ii8+CiAgPHJlY3QgeD0iMzAiIHk9Ijg1IiB3aWR0aD0iMTQwIiBoZWlnaHQ9IjMwIiBmaWxsPSJub25lIiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMyIgcng9IjE1Ii8+CiAgPCEtLSBEZWZlY3QgaW5kaWNhdG9ycyAtLT4KICA8Y2lyY2xlIGN4PSI2MCIgY3k9IjEwMCIgcj0iNiIgZmlsbD0iI0VBNTgwQyIvPgogIDxjaXJjbGUgY3g9IjEyMCIgY3k9IjEwMCIgcj0iNiIgZmlsbD0iI0VBNTgwQyIvPgogIDxjaXJjbGUgY3g9IjE1MCIgY3k9IjEwMCIgcj0iNiIgZmlsbD0iI0VBNTgwQyIvPgo8L3N2Zz4K"


def create_professional_header():
    """Create professional header with branding."""
    logo_b64 = get_logo_base64()
    
    header_html = f"""
    <div class="main-header">
        <div class="logo-container">
            <img src="{logo_b64}" style="width:60px;height:60px;margin-right:20px;border-radius:8px;">
            <div>
                <h1 class="custom-title">Pipeline Integrity FFS</h1>
                <p class="custom-subtitle">Fitness-for-Service Assessment Platform</p>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def create_professional_sidebar(session_state):
    """Create professional sidebar with enhanced navigation."""
    
    with st.sidebar:
        # Apply navigation-specific styles
        from app.styles import apply_navigation_styles
        apply_navigation_styles()
        
        # Company branding in sidebar
        st.markdown("""
        <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid var(--accent-orange); margin-bottom: 20px;">
            <div style="color: var(--accent-gold); font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">
                Pipeline FFS
            </div>
            <div style="color: var(--neutral-300); font-size: 0.8rem; margin-top: 5px;">
                Enterprise Assessment Suite
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation Section
        st.markdown('<div class="sidebar-section-header">🧭 Navigation</div>', unsafe_allow_html=True)

        nav_items = get_navigation_items()
        for item in nav_items:
            # Create professional navigation item
            icon = item['icon']
            title = item['title']
            item_id = item['id']
            available = item['available']
            active = item['active']

            if active:
                st.markdown(f"""
                <div class="nav-item active">
                    {icon} {title}
                </div>
                """, unsafe_allow_html=True)
            elif available:
                # Use columns to create clickable navigation
                if st.button(f"{icon} {title}", key=f"nav_{item_id}", use_container_width=True):
                    set_current_page(item_id)
                    st.rerun()
            else:
                st.markdown(f"""
                <div class="nav-item disabled">
                    {icon} {title}
                </div>
                """, unsafe_allow_html=True)

        # Data Management Section
        st.markdown('<div class="sidebar-section-header">📊 Data Management</div>', unsafe_allow_html=True)

        datasets = get_state('datasets', {})
        if datasets:
            st.markdown("**📈 Active Datasets**")
            for year in sorted(datasets.keys()):
                defect_count = len(datasets[year]['defects_df'])
                joint_count = len(datasets[year]['joints_df'])
                
                st.markdown(f"""
                <div class="dataset-status">
                    {year} Dataset: {defect_count} defects, {joint_count} joints
                </div>
                """, unsafe_allow_html=True)

        # File Upload Section
        st.markdown('<div class="sidebar-section-header">📤 Data Upload</div>', unsafe_allow_html=True)
        
        current_year = datetime.now().year
        year_options = list(range(current_year - 30, current_year + 1))
        selected_year = st.selectbox(
            "🗓️ Inspection Year",
            options=year_options,
            index=len(year_options) - 1,
            key="year_selector_sidebar",
            help="Select the year for the inspection data you're uploading"
        )

        uploaded_file = st.file_uploader(
            "📁 Upload CSV Data",
            type="csv",
            key=f"file_uploader_{get_state('file_upload_key', 0)}",
            help="Upload pipeline inspection data in CSV format"
        )

        # System Actions
        st.markdown('<div class="sidebar-section-header">⚙️ System Actions</div>', unsafe_allow_html=True)
        
        if datasets:
            if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    clear_datasets()
                    set_current_page('home')
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm data deletion")
        
        # System Status Footer
        st.markdown('<div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--neutral-400);">', unsafe_allow_html=True)
        
        # Memory usage indicator (simplified)
        dataset_memory = len(datasets) * 100  # Rough estimate
        memory_status = "🟢 Optimal" if dataset_memory < 500 else "🟡 Moderate" if dataset_memory < 1000 else "🔴 High"
        
        st.markdown(f"""
        <div style="text-align: center; color: var(--neutral-300); font-size: 0.75rem;">
            <div><strong>System Status</strong></div>
            <div>Memory Usage: {memory_status}</div>
            <div>Version: {APP_VERSION}</div>
            <div style="margin-top: 10px; color: var(--accent-gold);">
                Pipeline FFS Suite
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        return uploaded_file, selected_year


def create_professional_breadcrumb(items=None):
    """Create professional breadcrumb navigation."""
    if items is None:
        items = get_breadcrumb_items()

    if len(items) <= 1:
        return  # Don't show breadcrumb for single items

    breadcrumb_items = []
    for i, (label, active) in enumerate(items):
        if active:
            breadcrumb_items.append(f'<span class="breadcrumb-item" style="color:var(--primary-blue);font-weight:600;">{label}</span>')
        else:
            breadcrumb_items.append(f'<span class="breadcrumb-item">{label}</span>')

    breadcrumb_html = f"""
    <div class="breadcrumb animate-in">
        🏠 {' <span class="breadcrumb-separator">›</span> '.join(breadcrumb_items)}
    </div>
    """
    
    st.markdown(breadcrumb_html, unsafe_allow_html=True)


def create_page_header(title, subtitle=None, status=None):
    """Create consistent page headers with status indicators."""
    
    status_html = ""
    if status:
        from app.styles import get_status_indicator
        status_html = f'<div style="margin-top: 10px;">{get_status_indicator(status)}</div>'
    
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<p style="color: var(--neutral-600); font-size: 1.1rem; margin-top: 10px; font-weight: 400;">{subtitle}</p>'
    
    header_html = f"""
    <div class="card-container animate-in">
        <h2 class="section-header">{title}</h2>
        {subtitle_html}
        {status_html}
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)


def show_professional_metrics_row(metrics_data):
    """Display a row of professional metrics."""
    if not metrics_data:
        return
    
    cols = st.columns(len(metrics_data))
    
    for i, (title, value, status, description) in enumerate(metrics_data):
        with cols[i]:
            from app.styles import professional_metric_card
            metric_html = professional_metric_card(title, value, status, description)
            st.markdown(metric_html, unsafe_allow_html=True)


def create_action_toolbar(actions):
    """Create a professional action toolbar with buttons."""
    if not actions:
        return
    
    toolbar_html = """
    <div style="display: flex; gap: 10px; align-items: center; padding: 15px; 
                background: var(--neutral-50); border-radius: var(--radius-md); 
                border: 1px solid var(--neutral-200); margin: 20px 0;">
    """
    
    for action in actions:
        if action.get('type') == 'button':
            toolbar_html += f"""
            <button class="custom-button" onclick="{action.get('onclick', '')}">
                {action.get('icon', '')} {action.get('label', 'Action')}
            </button>
            """
        elif action.get('type') == 'separator':
            toolbar_html += '<div style="width: 1px; height: 20px; background: var(--neutral-300);"></div>'
    
    toolbar_html += "</div>"
    st.markdown(toolbar_html, unsafe_allow_html=True)


def show_loading_state(message="Processing..."):
    """Show professional loading state."""
    loading_html = f"""
    <div style="text-align: center; padding: 40px; color: var(--neutral-600);">
        <div class="loading" style="font-size: 1.2rem; margin-bottom: 10px;">
            ⚙️ {message}
        </div>
        <div style="font-size: 0.9rem; color: var(--neutral-500);">
            Please wait while we process your request...
        </div>
    </div>
    """
    st.markdown(loading_html, unsafe_allow_html=True)