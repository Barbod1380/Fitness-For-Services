"""
Navigation components for the Pipeline Analysis application.
"""
import streamlit as st
from datetime import datetime
from app.services.state_manager import get_state, update_state, clear_datasets
from app.services.navigation_service import (
    get_navigation_items, set_current_page, get_breadcrumb_items
)
from app.config import APP_VERSION

def get_logo_base64():
    """
    Get a base64-encoded SVG logo for the application.
    
    Returns:
    - Base64-encoded SVG string
    """
    # Pipeline logo placeholder
    return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgIDxjaXJjbGUgZmlsbD0iIzM0OThkYiIgY3g9IjEwMCIgY3k9IjEwMCIgcj0iMTAwIi8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iNzAiIHdpZHRoPSIxMjAiIGhlaWdodD0iMjAiIHJ4PSI1Ii8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iMTEwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjIwIiByeD0iNSIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iNTUiIGN5PSI4MCIgcj0iOCIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iMTQwIiBjeT0iMTIwIiByPSI4Ii8+CiAgPC9nPgo8L3N2Zz4K"

def create_sidebar(session_state):
    """
    Create the application sidebar for data management.
    
    Parameters:
    - session_state: Streamlit session state
    
    Returns:
    - Tuple of (uploaded_file, selected_year)
    """
    with st.sidebar:
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        
        # Display navigation items
        nav_items = get_navigation_items()
        for item in nav_items:
    
            # Create a clickable navigation item if available
            if item['available']:
                if st.sidebar.button(
                    f"{item['icon']} {item['title']}", 
                    key=f"nav_{item['id']}",
                    use_container_width=True,
                    type="primary" if item['active'] else "secondary"
                ):
                    set_current_page(item['id'])
                    st.rerun()
            else:
                # Create a disabled navigation item
                st.sidebar.button(
                    f"{item['icon']} {item['title']}", 
                    key=f"nav_{item['id']}_disabled",
                    use_container_width=True,
                    disabled=True
                )
        
        st.markdown('<hr style="margin: 15px 0;">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)
        
        # Display currently loaded datasets
        datasets = get_state('datasets', {})
        if datasets:
            st.markdown("<strong>Loaded Datasets</strong>", unsafe_allow_html=True)
            for year in sorted(datasets.keys()):
                st.markdown(
                    f'<div style="padding:8px 10px;margin-bottom:5px;background-color:#e8f8ef;border-radius:5px;">'
                    f'<span style="color:#2ecc71;margin-right:8px;">✓</span>{year} data loaded'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        
        st.markdown('<div style="margin-top:20px;"><strong>Add New Dataset</strong></div>', unsafe_allow_html=True)
        
        # Year selection for new data
        current_year = datetime.now().year
        year_options = list(range(current_year - 30, current_year + 1))
        selected_year = st.selectbox(
            "Select Inspection Year", 
            options=year_options,
            index=len(year_options) - 1,  # Default to current year
            key="year_selector_sidebar"
        )
        
        # File uploader with improved styling
        st.markdown(f'<div style="margin:10px 0 5px 0;"><strong>Upload {selected_year} Inspection CSV</strong></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type="csv",
            key=f"file_uploader_{get_state('file_upload_key', 0)}",
            label_visibility="collapsed"
        )
        
        # Button to clear all data with better styling
        st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
        if st.button("Clear All Datasets", key="clear_all_datasets_btn", use_container_width=True):
            clear_datasets()
            set_current_page('home')
            st.rerun()
        
        # Add a footer with app info
        st.markdown(
            f"""
            <div class="footer">
                <p>Pipeline Inspection Analysis Tool<br>Version {APP_VERSION}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        return uploaded_file, selected_year

def create_breadcrumb(items=None):
    """
    Create a breadcrumb navigation trail.
    
    Parameters:
    - items: List of (label, active) tuples, where active is a boolean
             If None, use current page breadcrumb
    
    Returns:
    - Streamlit element
    """
    if items is None:
        items = get_breadcrumb_items()
    
    html = '<div class="breadcrumb">'
    
    for i, (label, active) in enumerate(items):
        if active:
            html += f'<span class="breadcrumb-item active">{label}</span>'
        else:
            html += f'<span class="breadcrumb-item">{label}</span>'
        
        # Add separator except for the last item
        if i < len(items) - 1:
            html += '<span class="breadcrumb-separator">›</span>'
    
    html += '</div>'
    
    return st.markdown(html, unsafe_allow_html=True)