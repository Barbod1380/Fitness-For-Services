"""
Configuration settings and constants for the Pipeline Analysis application.
"""

# Application settings
APP_TITLE = "Pipeline Inspection Analysis"
APP_SUBTITLE = "Upload inspection data to analyze pipeline condition and track defects over time"
APP_VERSION = "1.0"

# Data processing settings
ENCODING_OPTIONS = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']

# Visualization settings
DEFAULT_PLOT_HEIGHT = 600
DEFAULT_COLORSCALE = "Turbo"  # For depth visualization
DEFAULT_JOINT_COLORSCALE = "YlOrRd"  # For joint defect visualization

# Analysis settings
DEFAULT_KNN_NEIGHBORS = 3  # Default number of neighbors for KNN algorithm
DEFAULT_DISTANCE_TOLERANCE = 0.01  # Default distance tolerance for matching defects (meters)
DEFAULT_CLOCK_TOLERANCE = 20  # Default clock position tolerance for matching defects (minutes)

# UI settings
SINGLE_YEAR_TAB_LABEL = "Single Year Analysis"
MULTI_YEAR_TAB_LABEL = "Multi-Year Comparison"
WORKFLOW_STEPS = ["Upload File", "Map Columns", "Process Data"]