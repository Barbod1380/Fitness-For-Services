"""
Service modules for the Pipeline Analysis application.
"""
from .state_manager import initialize_session_state, get_state, update_state
from .navigation_service import get_current_page, set_current_page