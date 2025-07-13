# core/failure_aware_clustering.py
"""
Failure-aware clustering system for pipeline integrity simulation.
Directly integrates clustering with failure criteria for accurate joint failure prediction.

This module implements:
1. ERF-threshold aware clustering
2. Joint-level failure prediction
3. Time-forward simulation foundation
4. Integration with existing corrosion assessment
"""

import numpy as np
import pandas as pd
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import existing modules
from .enhanced_ffs_clustering import EnhancedFFSClusterer, ClusterProperties

class FailureCriterion(Enum):
    """Supported failure criteria for joint assessment"""
    DEPTH_THRESHOLD = "depth_threshold"
    ERF_B31G = "erf_b31g"
    ERF_MODIFIED_B31G = "erf_modified_b31g"
    ERF_RSTRENG = "erf_simplified_eff_area"
    COMBINED_CRITERIA = "combined_criteria"

@dataclass
class FailureRisk:
    """Risk assessment for a defect or cluster"""
    current_risk_level: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    failure_probability: float  # 0.0 to 1.0
    time_to_failure_years: Optional[float]  # None if no failure predicted
    critical_parameter: str  # Which parameter drives the failure
    failure_margin: float  # How close to failure (negative = already failed)

@dataclass
class JointFailureAssessment:
    """Complete failure assessment for a pipeline joint"""
    joint_number: int
    defect_indices: List[int]
    cluster_properties: Optional[ClusterProperties]
    current_failure_risk: FailureRisk
    failure_criteria_status: Dict[str, bool]
    recommended_action: str
    inspection_priority: int  # 1 (highest) to 5 (lowest)

class FailureAwareClusterer:
    """
    Advanced clusterer that considers failure criteria in clustering decisions.
    Designed for time-forward failure simulation and integrity management.
    """
    
    def __init__(self,
                 depth_threshold_pct: float = 80.0,
                 erf_threshold_b31g: float = 0.90,
                 erf_threshold_modified: float = 0.90,
                 erf_threshold_rstreng: float = 0.90,
                 clustering_standard: str = "RSTRENG",
                 pipe_diameter_mm: float = 1000.0,
                 conservative_factor: float = 1.0):
        """
        Initialize failure-aware clusterer.
        
        Parameters:
        - depth_threshold_pct: Depth threshold for failure (default 80%)
        - erf_threshold_*: ERF thresholds for different methods (default 0.90)
        - clustering_standard: Industry standard for clustering
        - pipe_diameter_mm: Pipeline diameter
        - conservative_factor: Additional conservatism
        """
        
        self.depth_threshold = depth_threshold_pct
        self.erf_thresholds = {
            'b31g': erf_threshold_b31g,
            'modified_b31g': erf_threshold_modified,
            'simplified_eff_area': erf_threshold_rstreng
        }
        
        # Initialize enhanced clusterer
        self.enhanced_clusterer = EnhancedFFSClusterer(
            standard=clustering_standard,
            pipe_diameter_mm=pipe_diameter_mm,
            conservative_factor=conservative_factor,
            include_stress_concentration=True
        )
        
        self.clustering_standard = clustering_standard
        self.pipe_diameter_mm = pipe_diameter_mm
    
    def assess_joint_failures(self,
                            defects_df: pd.DataFrame,
                            joints_df: pd.DataFrame,
                            enhanced_assessment_df: Optional[pd.DataFrame] = None) -> List[JointFailureAssessment]:
        """
        Assess failure risk for all pipeline joints considering clustering effects.
        
        Parameters:
        - defects_df: Original defects DataFrame
        - joints_df: Joints DataFrame with specifications
        - enhanced_assessment_df: DataFrame with computed corrosion metrics (optional)
        
        Returns:
        - List of JointFailureAssessment objects for each joint with defects
        """
        
        # Find enhanced clusters with stress concentration
        clusters = self.enhanced_clusterer.find_enhanced_clusters(defects_df, joints_df)
        
        # Create combined defects DataFrame if needed
        if enhanced_assessment_df is None:
            # Need to compute basic assessment
            combined_df = self.enhanced_clusterer.create_combined_defects_dataframe(defects_df, clusters)
        else:
            combined_df = enhanced_assessment_df
        
        # Group defects by joint
        joint_assessments = []
        processed_joints = set()
        
        # Process clustered defects first
        for cluster in clusters:
            # Get joint number from first defect in cluster
            first_defect_idx = cluster.defect_indices[0]
            joint_number = defects_df.iloc[first_defect_idx]['joint number']
            
            if joint_number in processed_joints:
                continue
            
            # Assess this joint with clustering considerations
            joint_assessment = self._assess_joint_with_cluster(
                joint_number, cluster, defects_df, joints_df, combined_df
            )
            
            joint_assessments.append(joint_assessment)
            processed_joints.add(joint_number)
        
        # Process remaining individual defects
        remaining_defects = defects_df[~defects_df.index.isin(
            [idx for cluster in clusters for idx in cluster.defect_indices]
        )]
        
        for joint_number in remaining_defects['joint number'].unique():
            if joint_number in processed_joints:
                continue
                
            joint_defects = remaining_defects[remaining_defects['joint number'] == joint_number]
            
            # Assess joint with individual defects
            joint_assessment = self._assess_joint_individual(
                joint_number, joint_defects, joints_df, combined_df
            )
            
            joint_assessments.append(joint_assessment)
            processed_joints.add(joint_number)
        
        # Sort by failure risk (highest risk first)
        joint_assessments.sort(key=lambda x: (
            x.current_failure_risk.failure_probability,
            -x.current_failure_risk.failure_margin
        ), reverse=True)
        
        return joint_assessments
    
    def _assess_joint_with_cluster(self,
                                 joint_number: int,
                                 cluster: ClusterProperties,
                                 defects_df: pd.DataFrame,
                                 joints_df: pd.DataFrame,
                                 assessment_df: pd.DataFrame) -> JointFailureAssessment:
        """
        Assess failure risk for a joint with clustered defects.
        """
        
        # Calculate failure risk with stress concentration effects
        failure_risk = self._calculate_cluster_failure_risk(cluster, assessment_df)
        
        # Check multiple failure criteria
        criteria_status = self._check_failure_criteria(cluster, assessment_df)
        
        # Determine recommended action
        recommended_action = self._determine_recommended_action(failure_risk, criteria_status)
        
        # Assign inspection priority
        inspection_priority = self._assign_inspection_priority(failure_risk)
        
        return JointFailureAssessment(
            joint_number=joint_number,
            defect_indices=cluster.defect_indices,
            cluster_properties=cluster,
            current_failure_risk=failure_risk,
            failure_criteria_status=criteria_status,
            recommended_action=recommended_action,
            inspection_priority=inspection_priority
        )
    
    def _assess_joint_individual(self,
                               joint_number: int,
                               joint_defects: pd.DataFrame,
                               joints_df: pd.DataFrame,
                               assessment_df: pd.DataFrame) -> JointFailureAssessment:
        """
        Assess failure risk for a joint with individual (non-clustered) defects.
        """
        
        # Find worst defect in joint
        worst_defect_idx = joint_defects['depth [%]'].idxmax()
        defect_indices = joint_defects.index.tolist()
        
        # Calculate failure risk for individual defects
        failure_risk = self._calculate_individual_failure_risk(joint_defects, assessment_df)
        
        # Check failure criteria for worst defect
        criteria_status = self._check_individual_failure_criteria(worst_defect_idx, assessment_df)
        
        # Determine recommended action
        recommended_action = self._determine_recommended_action(failure_risk, criteria_status)
        
        # Assign inspection priority
        inspection_priority = self._assign_inspection_priority(failure_risk)
        
        return JointFailureAssessment(
            joint_number=joint_number,
            defect_indices=defect_indices,
            cluster_properties=None,
            current_failure_risk=failure_risk,
            failure_criteria_status=criteria_status,
            recommended_action=recommended_action,
            inspection_priority=inspection_priority
        )
    
    def _calculate_cluster_failure_risk(self,
                                      cluster: ClusterProperties,
                                      assessment_df: pd.DataFrame) -> FailureRisk:
        """
        Calculate failure risk for a clustered defect with stress concentration.
        """
        
        # Find the assessment row for this cluster
        cluster_row = None
        for idx, row in assessment_df.iterrows():
            if (row.get('is_combined', False) and 
                row.get('original_defect_indices') == cluster.defect_indices):
                cluster_row = row
                break
        
        if cluster_row is None:
            # Fallback: estimate risk from cluster properties
            return self._estimate_risk_from_cluster(cluster)
        
        # Calculate risk based on multiple criteria
        depth_risk = cluster.max_depth_pct / self.depth_threshold
        
        # ERF-based risks with stress concentration
        erf_risks = []
        for method in ['b31g', 'modified_b31g', 'simplified_eff_area']:
            erf_col = f'{method}_erf_stress_corrected'
            if erf_col in cluster_row:
                erf_value = cluster_row[erf_col]
                erf_threshold = self.erf_thresholds.get(method, 0.90)
                erf_risk = erf_value / erf_threshold
                erf_risks.append(erf_risk)
        
        # Overall risk is maximum of individual risks
        max_risk = max([depth_risk] + erf_risks)
        
        # Classify risk level
        if max_risk >= 1.0:
            risk_level = "CRITICAL"
            failure_probability = 0.95
        elif max_risk >= 0.9:
            risk_level = "HIGH"
            failure_probability = 0.75
        elif max_risk >= 0.7:
            risk_level = "MODERATE"
            failure_probability = 0.40
        else:
            risk_level = "LOW"
            failure_probability = 0.10
        
        # Determine critical parameter
        if depth_risk == max_risk:
            critical_parameter = "Defect Depth"
        else:
            # Find which ERF method has highest risk
            method_names = ['B31G', 'Modified B31G', 'RSTRENG']
            max_erf_idx = erf_risks.index(max(erf_risks)) if erf_risks else 0
            critical_parameter = f"ERF ({method_names[max_erf_idx]})"
        
        # Calculate failure margin (how close to failure)
        failure_margin = 1.0 - max_risk
        
        return FailureRisk(
            current_risk_level=risk_level,
            failure_probability=failure_probability,
            time_to_failure_years=None,  # Will be calculated in simulation
            critical_parameter=critical_parameter,
            failure_margin=failure_margin
        )
    
    def _calculate_individual_failure_risk(self,
                                         joint_defects: pd.DataFrame,
                                         assessment_df: pd.DataFrame) -> FailureRisk:
        """
        Calculate failure risk for individual defects in a joint.
        """
        
        # Find worst defect
        max_depth = joint_defects['depth [%]'].max()
        worst_defect_idx = joint_defects['depth [%]'].idxmax()
        
        # Get assessment data for worst defect
        assessment_row = assessment_df.loc[worst_defect_idx] if worst_defect_idx in assessment_df.index else None
        
        if assessment_row is None:
            # Fallback estimation
            depth_risk = max_depth / self.depth_threshold
            risk_level = "MODERATE" if depth_risk > 0.5 else "LOW"
            return FailureRisk(
                current_risk_level=risk_level,
                failure_probability=depth_risk * 0.5,
                time_to_failure_years=None,
                critical_parameter="Defect Depth",
                failure_margin=1.0 - depth_risk
            )
        
        # Calculate risk similar to cluster method but without stress concentration
        depth_risk = max_depth / self.depth_threshold
        
        erf_risks = []
        for method in ['b31g', 'modified_b31g', 'simplified_eff_area']:
            erf_col = f'{method}_erf'
            if erf_col in assessment_row:
                erf_value = assessment_row[erf_col]
                erf_threshold = self.erf_thresholds.get(method, 0.90)
                erf_risk = erf_value / erf_threshold
                erf_risks.append(erf_risk)
        
        max_risk = max([depth_risk] + erf_risks)
        
        # Classify risk level (same logic as cluster)
        if max_risk >= 1.0:
            risk_level = "CRITICAL"
            failure_probability = 0.90  # Slightly lower than clustered
        elif max_risk >= 0.9:
            risk_level = "HIGH"
            failure_probability = 0.70
        elif max_risk >= 0.7:
            risk_level = "MODERATE"
            failure_probability = 0.35
        else:
            risk_level = "LOW"
            failure_probability = 0.05
        
        # Determine critical parameter
        if depth_risk == max_risk:
            critical_parameter = "Defect Depth"
        else:
            method_names = ['B31G', 'Modified B31G', 'RSTRENG']
            max_erf_idx = erf_risks.index(max(erf_risks)) if erf_risks else 0
            critical_parameter = f"ERF ({method_names[max_erf_idx]})"
        
        failure_margin = 1.0 - max_risk
        
        return FailureRisk(
            current_risk_level=risk_level,
            failure_probability=failure_probability,
            time_to_failure_years=None,
            critical_parameter=critical_parameter,
            failure_margin=failure_margin
        )
    
    def _check_failure_criteria(self, 
                              cluster: ClusterProperties, 
                              assessment_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Check specific failure criteria for a cluster.
        """
        
        criteria_status = {
            'depth_exceeded': cluster.max_depth_pct > self.depth_threshold,
            'erf_b31g_exceeded': False,
            'erf_modified_b31g_exceeded': False,
            'erf_rstreng_exceeded': False
        }
        
        # Find assessment row for cluster
        for idx, row in assessment_df.iterrows():
            if (row.get('is_combined', False) and 
                row.get('original_defect_indices') == cluster.defect_indices):
                
                # Check ERF criteria with stress concentration
                for method in ['b31g', 'modified_b31g', 'simplified_eff_area']:
                    erf_col = f'{method}_erf_stress_corrected'
                    if erf_col in row:
                        erf_value = row[erf_col]
                        erf_threshold = self.erf_thresholds.get(method, 0.90)
                        criteria_status[f'erf_{method}_exceeded'] = erf_value > erf_threshold
                
                break
        
        return criteria_status
    
    
    def _check_individual_failure_criteria(self, 
                                         defect_idx: int, 
                                         assessment_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Check failure criteria for an individual defect.
        """
        
        criteria_status = {
            'depth_exceeded': False,
            'erf_b31g_exceeded': False,
            'erf_modified_b31g_exceeded': False,
            'erf_rstreng_exceeded': False
        }
        
        if defect_idx in assessment_df.index:
            row = assessment_df.loc[defect_idx]
            
            # Check depth criterion
            if 'depth [%]' in row:
                criteria_status['depth_exceeded'] = row['depth [%]'] > self.depth_threshold
            
            # Check ERF criteria
            for method in ['b31g', 'modified_b31g', 'simplified_eff_area']:
                erf_col = f'{method}_erf'
                if erf_col in row:
                    erf_value = row[erf_col]
                    erf_threshold = self.erf_thresholds.get(method, 0.90)
                    criteria_status[f'erf_{method}_exceeded'] = erf_value > erf_threshold
        
        return criteria_status
    
    def _determine_recommended_action(self, 
                                    failure_risk: FailureRisk, 
                                    criteria_status: Dict[str, bool]) -> str:
        """
        Determine recommended action based on failure risk and criteria.
        """
        
        # Check if any critical criteria are exceeded
        critical_criteria = any(criteria_status.values())
        
        if critical_criteria or failure_risk.current_risk_level == "CRITICAL":
            return "IMMEDIATE REPAIR REQUIRED"
        elif failure_risk.current_risk_level == "HIGH":
            return "PRIORITY REPAIR - Schedule within 6 months"
        elif failure_risk.current_risk_level == "MODERATE":
            return "MONITOR - Increase inspection frequency"
        else:
            return "NORMAL OPERATIONS - Continue routine monitoring"
    
    def _assign_inspection_priority(self, failure_risk: FailureRisk) -> int:
        """
        Assign inspection priority (1 = highest, 5 = lowest).
        """
        
        if failure_risk.current_risk_level == "CRITICAL":
            return 1
        elif failure_risk.current_risk_level == "HIGH":
            return 2
        elif failure_risk.current_risk_level == "MODERATE":
            return 3
        else:
            return 4
    
    def _estimate_risk_from_cluster(self, cluster: ClusterProperties) -> FailureRisk:
        """
        Fallback risk estimation when assessment data is not available.
        """
        
        depth_risk = cluster.max_depth_pct / self.depth_threshold
        
        # Apply stress concentration to risk
        adjusted_risk = depth_risk * cluster.stress_concentration_factor
        
        if adjusted_risk >= 1.0:
            risk_level = "CRITICAL"
            failure_probability = 0.85
        elif adjusted_risk >= 0.8:
            risk_level = "HIGH"
            failure_probability = 0.65
        else:
            risk_level = "MODERATE"
            failure_probability = 0.30
        
        return FailureRisk(
            current_risk_level=risk_level,
            failure_probability=failure_probability,
            time_to_failure_years=None,
            critical_parameter="Defect Depth (Estimated)",
            failure_margin=1.0 - adjusted_risk
        )
    
    def create_failure_summary_report(self, 
                                    joint_assessments: List[JointFailureAssessment]) -> pd.DataFrame:
        """
        Create a summary report of failure assessments.
        """
        
        report_data = []
        
        for assessment in joint_assessments:
            risk = assessment.current_failure_risk
            
            report_data.append({
                'Joint Number': assessment.joint_number,
                'Risk Level': risk.current_risk_level,
                'Failure Probability': f"{risk.failure_probability:.1%}",
                'Critical Parameter': risk.critical_parameter,
                'Failure Margin': f"{risk.failure_margin:.2f}",
                'Defect Count': len(assessment.defect_indices),
                'Is Clustered': assessment.cluster_properties is not None,
                'Stress Factor': assessment.cluster_properties.stress_concentration_factor if assessment.cluster_properties else 1.0,
                'Recommended Action': assessment.recommended_action,
                'Inspection Priority': assessment.inspection_priority
            })
        
        return pd.DataFrame(report_data)
    
    def predict_failure_timeline(self, 
                               joint_assessments: List[JointFailureAssessment],
                               growth_rates: Dict,
                               simulation_years: int = 15) -> Dict[int, List[int]]:
        """
        Predict failure timeline for joints based on growth rates.
        This is the foundation for the time-forward simulation.
        
        Parameters:
        - joint_assessments: List of joint failure assessments
        - growth_rates: Dictionary of growth rates for defects
        - simulation_years: Number of years to simulate
        
        Returns:
        - Dictionary mapping year to list of joint numbers that fail
        """
        
        failure_timeline = {}
        
        for assessment in joint_assessments:
            # Estimate time to failure based on current risk and growth rates
            time_to_failure = self._estimate_time_to_failure(assessment, growth_rates)
            
            if time_to_failure and time_to_failure <= simulation_years:
                failure_year = int(math.ceil(time_to_failure))
                if failure_year not in failure_timeline:
                    failure_timeline[failure_year] = []
                failure_timeline[failure_year].append(assessment.joint_number)
        
        return failure_timeline
    
    def _estimate_time_to_failure(self, 
                                assessment: JointFailureAssessment, 
                                growth_rates: Dict) -> Optional[float]:
        """
        Estimate time to failure for a joint based on current state and growth rates.
        """
        
        if assessment.current_failure_risk.current_risk_level == "CRITICAL":
            return 0.0  # Already at failure
        
        # Use failure margin and estimated growth to predict time
        failure_margin = assessment.current_failure_risk.failure_margin
        
        if failure_margin <= 0:
            return 0.0
        
        # Estimate average growth rate for defects in this joint
        avg_growth_rate = 0.02  # Default 2% per year
        
        # Try to get actual growth rates if available
        if growth_rates:
            defect_growth_rates = []
            for defect_idx in assessment.defect_indices:
                if defect_idx in growth_rates:
                    # Assume depth growth rate in % per year
                    rate = growth_rates[defect_idx].get('depth_growth_pct_per_year', 2.0)
                    defect_growth_rates.append(rate / 100.0)  # Convert to fraction
            
            if defect_growth_rates:
                avg_growth_rate = np.mean(defect_growth_rates)
        
        # Apply stress concentration acceleration if clustered
        if assessment.cluster_properties:
            stress_acceleration = 1.0 + 0.1 * (assessment.cluster_properties.stress_concentration_factor - 1.0)
            avg_growth_rate *= stress_acceleration
        
        # Calculate time to failure
        if avg_growth_rate > 0:
            time_to_failure = failure_margin / avg_growth_rate
            return min(time_to_failure, 50.0)  # Cap at 50 years
        
        return None  # No growth, no failure predicted


# Integration functions for existing system
def integrate_failure_aware_clustering(defects_df: pd.DataFrame,
                                     joints_df: pd.DataFrame,
                                     enhanced_assessment_df: pd.DataFrame,
                                     pipe_diameter_mm: float,
                                     erf_threshold: float = 1.0,
                                     depth_threshold: float = 80.0) -> Tuple[List[JointFailureAssessment], pd.DataFrame]:
    """
    Main integration function for existing corrosion assessment system.
    
    Parameters:
    - defects_df: Original defects DataFrame
    - joints_df: Joints DataFrame
    - enhanced_assessment_df: Result from compute_enhanced_corrosion_metrics
    - pipe_diameter_mm: Pipeline diameter
    - erf_threshold: ERF threshold for failure
    - depth_threshold: Depth threshold for failure
    
    Returns:
    - Tuple of (joint_assessments, failure_summary_report)
    """
    
    # Create failure-aware clusterer
    clusterer = FailureAwareClusterer(
        depth_threshold_pct=depth_threshold,
        erf_threshold_b31g=erf_threshold,
        erf_threshold_modified=erf_threshold,
        erf_threshold_rstreng=erf_threshold,
        pipe_diameter_mm=pipe_diameter_mm
    )
    
    # Assess joint failures
    joint_assessments = clusterer.assess_joint_failures(
        defects_df, joints_df, enhanced_assessment_df
    )
    
    # Create summary report
    summary_report = clusterer.create_failure_summary_report(joint_assessments)
    
    return joint_assessments, summary_report