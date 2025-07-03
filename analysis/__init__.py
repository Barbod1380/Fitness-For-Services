"""
Analysis modules for the Pipeline Analysis application.
"""
from .defect_analysis import (
    create_dimension_distribution_plots,
    create_dimension_statistics_table,
    create_combined_dimensions_plot,
    create_joint_summary
)
from .growth_analysis import (
    correct_negative_growth_rates,
    create_growth_summary_table,
    create_highest_growth_table
)

from .remaining_life_analysis import *

from .failure_prediction import predict_joint_failures_over_time