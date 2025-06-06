"""
Home view for the Pipeline Analysis application.
"""
import streamlit as st
from app.components import create_welcome_screen

def render_home_view():
    """Render the welcome/home page when no data is loaded."""
    create_welcome_screen()