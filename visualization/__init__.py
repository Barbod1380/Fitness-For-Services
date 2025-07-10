"""
Visualization modules for the Pipeline Analysis application.
"""
from .pipeline_viz import create_unwrapped_pipeline_visualization
from .joint_viz import create_joint_defect_visualization
from .comparison_viz import (
    create_comparison_stats_plot,
    create_new_defect_types_plot,
    create_negative_growth_plot,
    create_growth_rate_histogram
)

from .failure_prediction_viz import (
    create_failure_prediction_chart,
    create_failure_summary_metrics,
    create_failure_details_table
)

from .defect_assessment_viz import (
    create_defect_assessment_scatter_plot,
    create_defect_assessment_summary_table
)

from .pressure_assessment_viz import create_pressure_assessment_visualization