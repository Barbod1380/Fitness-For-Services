"""
View modules for the Pipeline Analysis application.
"""
from .home import render_home_view
from .upload import render_upload_view, load_csv_with_encoding
from .single_analysis import render_single_analysis_view
from .comparison import render_comparison_view
from .corrosion import *
from .failure_prediction import render_failure_prediction_view
