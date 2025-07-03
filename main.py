# pipeline_analysis/run.py

"""
Entry point for the Pipeline Analysis application.
"""

import streamlit as st
from app.main import run_app

if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        st.error(f"Application failed to start: {str(e)}")
