# Create new file: core/defect_matching.py
"""
Advanced defect matching system that handles FFS clustering
for multi-year growth analysis.
"""

from typing import List
from dataclasses import dataclass

@dataclass
class DefectMatch:
    """Represents a match between defects across inspection years."""
    year1_indices: List[int]  # Can be multiple if clustered
    year2_indices: List[int]  # Can be multiple if clustered
    match_type: str  # '1-to-1', 'many-to-1', '1-to-many', 'many-to-many'
    match_confidence: float  # 0-1 confidence score
    match_distance: float  # Spatial distance used for matching