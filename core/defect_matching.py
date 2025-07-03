# Create new file: core/defect_matching.py
"""
Advanced defect matching system that handles FFS clustering
for multi-year growth analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DefectMatch:
    """Represents a match between defects across inspection years."""
    year1_indices: List[int]  # Can be multiple if clustered
    year2_indices: List[int]  # Can be multiple if clustered
    match_type: str  # '1-to-1', 'many-to-1', '1-to-many', 'many-to-many'
    match_confidence: float  # 0-1 confidence score
    match_distance: float  # Spatial distance used for matching
    
class ClusterAwareDefectMatcher:
    """
    Matches defects between inspection years considering FFS clustering.
    """
    
    def __init__(self, 
                 max_axial_distance_mm: float = 300.0,  # 300mm = 30cm tolerance
                 max_clock_difference_hours: float = 1.0,  # 1 hour clock tolerance
                 pipe_diameter_mm: float = None):
        self.max_axial_distance = max_axial_distance_mm
        self.max_clock_difference = max_clock_difference_hours
        self.pipe_diameter = pipe_diameter_mm
    
    def match_defects_with_clustering(self,
                                    year1_defects: pd.DataFrame,
                                    year2_defects: pd.DataFrame,
                                    year1_clusters: List[List[int]],
                                    year2_clusters: List[List[int]]) -> List[DefectMatch]:
        """
        Match defects between years considering clustering.
        
        Parameters:
        - year1_defects: Earlier inspection defects
        - year2_defects: Later inspection defects  
        - year1_clusters: Cluster groups from FFS analysis for year1
        - year2_clusters: Cluster groups from FFS analysis for year2
        
        Returns:
        - List of DefectMatch objects
        """
        matches = []
        
        # Convert defects to arrays for easier manipulation
        y1_positions = self._get_defect_positions(year1_defects)
        y2_positions = self._get_defect_positions(year2_defects)
        
        # Track which defects have been matched
        y1_matched = set()
        y2_matched = set()
        
        # Phase 1: Match clusters to clusters
        cluster_matches = self._match_clusters(
            year1_clusters, year2_clusters, 
            y1_positions, y2_positions,
            year1_defects, year2_defects
        )
        
        for match in cluster_matches:
            matches.append(match)
            y1_matched.update(match.year1_indices)
            y2_matched.update(match.year2_indices)
        
        # Phase 2: Match remaining individual defects
        y1_unmatched = [i for i in range(len(year1_defects)) if i not in y1_matched]
        y2_unmatched = [i for i in range(len(year2_defects)) if i not in y2_matched]
        
        individual_matches = self._match_individual_defects(
            y1_unmatched, y2_unmatched,
            year1_defects, year2_defects
        )
        
        matches.extend(individual_matches)
        
        return matches
    
    def _get_defect_positions(self, defects_df: pd.DataFrame) -> np.ndarray:
        """Extract position data for matching."""
        positions = np.zeros((len(defects_df), 3))
        
        # Axial position in mm
        positions[:, 0] = defects_df['log dist. [m]'].values * 1000
        
        # Clock position in decimal hours
        positions[:, 1] = defects_df['clock'].apply(self._parse_clock_to_hours).values
        
        # Length for size-aware matching
        positions[:, 2] = defects_df['length [mm]'].values
        
        return positions
    
    def _parse_clock_to_hours(self, clock_str: str) -> float:
        """Convert clock string to decimal hours."""
        try:
            parts = clock_str.strip().split(':')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            return hours + minutes / 60.0
        except:
            return 12.0
    
    def _match_clusters(self, 
                       year1_clusters: List[List[int]],
                       year2_clusters: List[List[int]],
                       y1_positions: np.ndarray,
                       y2_positions: np.ndarray,
                       y1_defects: pd.DataFrame,
                       y2_defects: pd.DataFrame) -> List[DefectMatch]:
        """Match cluster groups between years."""
        matches = []
        
        # Calculate cluster centroids and extents
        y1_cluster_info = [self._get_cluster_info(cluster, y1_positions, y1_defects) 
                          for cluster in year1_clusters]
        y2_cluster_info = [self._get_cluster_info(cluster, y2_positions, y2_defects)
                          for cluster in year2_clusters]
        
        # Match clusters based on spatial overlap
        for i, (c1_indices, c1_info) in enumerate(zip(year1_clusters, y1_cluster_info)):
            best_match = None
            best_score = 0
            
            for j, (c2_indices, c2_info) in enumerate(zip(year2_clusters, y2_cluster_info)):
                score = self._calculate_cluster_match_score(c1_info, c2_info)
                
                if score > best_score and score > 0.5:  # Minimum threshold
                    best_score = score
                    best_match = j
            
            if best_match is not None:
                # Determine match type
                c2_indices = year2_clusters[best_match]
                match_type = self._determine_match_type(len(c1_indices), len(c2_indices))
                
                matches.append(DefectMatch(
                    year1_indices=c1_indices,
                    year2_indices=c2_indices,
                    match_type=match_type,
                    match_confidence=best_score,
                    match_distance=self._calculate_cluster_distance(c1_info, c2_info)
                ))
        
        return matches
    
    def _get_cluster_info(self, cluster_indices: List[int], 
                         positions: np.ndarray,
                         defects_df: pd.DataFrame) -> Dict:
        """Calculate cluster centroid and extent."""
        cluster_positions = positions[cluster_indices]
        
        return {
            'indices': cluster_indices,
            'centroid_axial': np.mean(cluster_positions[:, 0]),
            'centroid_clock': np.mean(cluster_positions[:, 1]),
            'min_axial': np.min(cluster_positions[:, 0]),
            'max_axial': np.max(cluster_positions[:, 0]),
            'total_length': np.sum(cluster_positions[:, 2]),
            'max_depth': defects_df.iloc[cluster_indices]['depth [%]'].max(),
            'num_defects': len(cluster_indices)
        }
    
    def _calculate_cluster_match_score(self, c1_info: Dict, c2_info: Dict) -> float:
        """Calculate match score between two clusters."""
        # Axial distance component
        axial_distance = abs(c1_info['centroid_axial'] - c2_info['centroid_axial'])
        axial_score = max(0, 1 - axial_distance / self.max_axial_distance)
        
        # Clock distance component
        clock_distance = self._circular_distance(c1_info['centroid_clock'], 
                                               c2_info['centroid_clock'], 12)
        clock_score = max(0, 1 - clock_distance / self.max_clock_difference)
        
        # Size similarity component
        size_ratio = min(c1_info['total_length'], c2_info['total_length']) / \
                    max(c1_info['total_length'], c2_info['total_length'])
        
        # Overlap component - do the clusters spatially overlap?
        overlap = self._calculate_axial_overlap(
            c1_info['min_axial'], c1_info['max_axial'],
            c2_info['min_axial'], c2_info['max_axial']
        )
        
        # Weighted score
        score = (0.3 * axial_score + 
                0.2 * clock_score + 
                0.2 * size_ratio + 
                0.3 * overlap)
        
        return score
    
    def _calculate_axial_overlap(self, min1, max1, min2, max2) -> float:
        """Calculate overlap fraction between two axial ranges."""
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        total_length = max(max1, max2) - min(min1, min2)
        
        return overlap_length / total_length if total_length > 0 else 0
    
    def _circular_distance(self, a1: float, a2: float, period: float) -> float:
        """Calculate minimum distance on a circular scale (e.g., clock positions)."""
        direct = abs(a2 - a1)
        wrapped = period - direct
        return min(direct, wrapped)
    
    def _calculate_cluster_distance(self, c1_info: Dict, c2_info: Dict) -> float:
        """Calculate spatial distance between cluster centroids."""
        axial_dist = abs(c1_info['centroid_axial'] - c2_info['centroid_axial'])
        
        if self.pipe_diameter:
            # Convert clock distance to arc length
            clock_dist = self._circular_distance(
                c1_info['centroid_clock'], 
                c2_info['centroid_clock'], 12
            )
            arc_length = (clock_dist / 12) * np.pi * self.pipe_diameter
            
            # Euclidean distance
            return np.sqrt(axial_dist**2 + arc_length**2)
        else:
            return axial_dist
    
    def _determine_match_type(self, n_year1: int, n_year2: int) -> str:
        """Determine the type of match based on defect counts."""
        if n_year1 == 1 and n_year2 == 1:
            return '1-to-1'
        elif n_year1 > 1 and n_year2 == 1:
            return 'many-to-1'
        elif n_year1 == 1 and n_year2 > 1:
            return '1-to-many'
        else:
            return 'many-to-many'
    
    def _match_individual_defects(self,
                                 y1_unmatched: List[int],
                                 y2_unmatched: List[int],
                                 y1_defects: pd.DataFrame,
                                 y2_defects: pd.DataFrame) -> List[DefectMatch]:
        """Match individual defects that weren't part of matched clusters."""
        matches = []
        
        if not y1_unmatched or not y2_unmatched:
            return matches
        
        # Create distance matrix
        distance_matrix = self._compute_distance_matrix(
            y1_defects.iloc[y1_unmatched],
            y2_defects.iloc[y2_unmatched]
        )
        
        # Use Hungarian algorithm or greedy matching
        used_y2 = set()
        
        for i, y1_idx in enumerate(y1_unmatched):
            if not y2_unmatched:
                break
                
            # Find best match from remaining year2 defects
            best_j = None
            best_distance = float('inf')
            
            for j, y2_idx in enumerate(y2_unmatched):
                if y2_idx in used_y2:
                    continue
                    
                if distance_matrix[i, j] < best_distance and \
                   distance_matrix[i, j] < self.max_axial_distance:
                    best_distance = distance_matrix[i, j]
                    best_j = j
            
            if best_j is not None:
                y2_idx = y2_unmatched[best_j]
                used_y2.add(y2_idx)
                
                confidence = max(0, 1 - best_distance / self.max_axial_distance)
                
                matches.append(DefectMatch(
                    year1_indices=[y1_idx],
                    year2_indices=[y2_idx],
                    match_type='1-to-1',
                    match_confidence=confidence,
                    match_distance=best_distance
                ))
        
        return matches
    
    def _compute_distance_matrix(self, 
                                defects1: pd.DataFrame,
                                defects2: pd.DataFrame) -> np.ndarray:
        """Compute pairwise distances between defects."""
        n1, n2 = len(defects1), len(defects2)
        distances = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                distances[i, j] = self._compute_defect_distance(
                    defects1.iloc[i], defects2.iloc[j]
                )
        
        return distances
    
    def _compute_defect_distance(self, d1: pd.Series, d2: pd.Series) -> float:
        """Compute distance between two individual defects."""
        # Axial distance
        axial_dist = abs(d1['log dist. [m]'] - d2['log dist. [m]']) * 1000  # in mm
        
        # Clock distance
        clock1 = self._parse_clock_to_hours(d1['clock'])
        clock2 = self._parse_clock_to_hours(d2['clock'])
        clock_dist = self._circular_distance(clock1, clock2, 12)
        
        if self.pipe_diameter:
            # Convert to arc length
            arc_length = (clock_dist / 12) * np.pi * self.pipe_diameter
            return np.sqrt(axial_dist**2 + arc_length**2)
        else:
            # Simple weighted distance
            return axial_dist + clock_dist * 100  # Weight clock distance