"""
Home view for the Pipeline Analysis application.
"""
from app.ui_components.welcome import create_welcome_screen

def render_home_view():
    """Render the welcome/home page when no data is loaded."""
    create_welcome_screen()