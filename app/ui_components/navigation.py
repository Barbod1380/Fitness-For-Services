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
    import base64
    try:
        with open("assets/logo-pica.png", "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
            return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgIDxjaXJjbGUgZmlsbD0iIzM0OThkYiIgY3g9IjEwMCIgY3k9IjEwMCIgcj0iMTAwIi8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iNzAiIHdpZHRoPSIxMjAiIGhlaWdodD0iMjAiIHJ4PSI1Ii8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iMTEwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjIwIiByeD0iNSIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iNTUiIGN5PSI4MCIgcj0iOCIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iMTQwIiBjeT0iMTIwIiByPSI4Ii8+CiAgPC9nPgo8L3N2Zz4K"


def create_sidebar(session_state):
    st.markdown('''
        <style>
            .section-header {
                font-size: 18px;
                font-weight: 700;
                color: #333;
                margin: 20px 0 10px 0;
                border-left: 5px solid #4A90E2;
                padding-left: 10px;
            }
            .stButton>button:hover {
                background-color: #e0f0ff !important;
                color: #0056b3 !important;
            }
        </style>
    ''', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)

        nav_items = get_navigation_items()
        for item in nav_items:
            label = f"{item['icon']} {item['title']}"

            if item['active']:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #4A90E2;
                        color: white;
                        padding: 10px 16px;
                        margin-bottom: 8px;
                        border-radius: 8px;
                        font-weight: 600;
                        cursor: default;">
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif item['available']:
                if st.button(label, key=f"nav_{item['id']}", use_container_width=True):
                    set_current_page(item['id'])
                    st.rerun()
            else:
                st.markdown(
                    f"""
                    <div style="
                        color: #bbb;
                        background-color: #f2f2f2;
                        padding: 10px 16px;
                        margin-bottom: 8px;
                        border-radius: 8px;
                        cursor: not-allowed;">
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown('<hr style="margin: 20px 0;">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)

        datasets = get_state('datasets', {})
        if datasets:
            st.markdown("<strong>Loaded Datasets</strong>", unsafe_allow_html=True)
            for year in sorted(datasets.keys()):
                st.markdown(
                    f"""
                    <div style="
                        display:inline-block;
                        background-color:#eafaf1;
                        color:#27ae60;
                        padding:6px 12px;
                        border-radius:50px;
                        margin:4px 0;
                        font-size: 14px;">
                        ✅ {year} data loaded
                    </div>
                    """, unsafe_allow_html=True
                )

        st.markdown('<div style="margin-top:20px;"><strong>➕ Add New Dataset</strong></div>', unsafe_allow_html=True)

        current_year = datetime.now().year
        year_options = list(range(current_year - 30, current_year + 1))
        selected_year = st.selectbox(
            "Select Inspection Year",
            options=year_options,
            index=len(year_options) - 1,
            key="year_selector_sidebar"
        )

        st.markdown(f'<div style="margin:10px 0 5px 0;"><strong>📤 Upload {selected_year} Inspection CSV</strong></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type="csv",
            key=f"file_uploader_{get_state('file_upload_key', 0)}",
            label_visibility="collapsed"
        )

        st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear All Datasets", key="clear_all_datasets_btn", use_container_width=True):
            clear_datasets()
            set_current_page('home')
            st.rerun()

        st.markdown(
            f"""
            <div style="
                margin-top: 40px;
                font-size: 12px;
                color: #aaa;
                text-align: center;">
                Pipeline Inspection Tool<br>v{APP_VERSION}
            </div>
            """,
            unsafe_allow_html=True
        )

        return uploaded_file, selected_year


def create_breadcrumb(items=None):
    if items is None:
        items = get_breadcrumb_items()

    html = '<div class="breadcrumb" style="margin-bottom: 15px;">'

    for i, (label, active) in enumerate(items):
        if active:
            html += f'<span class="breadcrumb-item" style="font-weight:bold;color:#4A90E2;">{label}</span>'
        else:
            html += f'<span class="breadcrumb-item" style="color:#666;">{label}</span>'

        if i < len(items) - 1:
            html += '<span class="breadcrumb-separator" style="margin: 0 5px;">›</span>'

    html += '</div>'

    return st.markdown(html, unsafe_allow_html=True)