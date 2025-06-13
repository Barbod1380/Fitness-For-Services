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
from .clustering_analysis import ( 
    prepare_clustering_features,
    perform_clustering,
    optimize_clustering_parameters,
    create_cluster_summary,
    create_cluster_visualization_2d,
    create_cluster_pipeline_visualization,
    create_cluster_characteristics_plot
)

from .remaining_life_analysis import *