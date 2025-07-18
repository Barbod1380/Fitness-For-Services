"""
Core functionality for the Pipeline Analysis application.
"""
from .column_mapping import (
    suggest_column_mapping, 
    apply_column_mapping, 
    get_missing_required_columns,
    STANDARD_COLUMNS,
    REQUIRED_COLUMNS
)
from .data_processing import process_pipeline_data, validate_pipeline_data
from .ffs_defect_interaction import *
from .defect_matching import ClusterAwareDefectMatcher, DefectMatch

from .standards_compliant_clustering import (
    StandardsCompliantClusterer,
    create_standards_compliant_clusterer
)
from .enhanced_ffs_clustering import (
    EnhancedFFSClusterer,
    enhance_existing_assessment
)
from .failure_aware_clustering import (
    FailureAwareClusterer,
    integrate_failure_aware_clustering
)