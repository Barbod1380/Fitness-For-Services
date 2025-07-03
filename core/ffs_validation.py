# core/ffs_validation.py
"""
Validation methods for FFS assessment results against industry standards.
"""
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple

class FFSValidation:
    """
    Validation suite for FFS assessment results per ASME B31G and API 579 standards.
    """
    
    def __init__(self):
        # ASME B31G Appendix C test cases (simplified subset for validation)
        self.b31g_test_cases = [
            {
                'name': 'B31G_Case_1_Short_Flaw',
                'depth_pct': 30.0,
                'length_mm': 50.0,
                'diameter_mm': 508.0,  # 20 inch
                'wall_thickness_mm': 12.7,  # 0.5 inch
                'smys_mpa': 358.0,  # X52
                'expected_pf_range': (15.0, 20.0),  # Expected failure pressure range MPa
                'notes': 'Short flaw, z < 20'
            },
            {
                'name': 'B31G_Case_2_Long_Flaw',
                'depth_pct': 40.0,
                'length_mm': 200.0,
                'diameter_mm': 508.0,
                'wall_thickness_mm': 12.7,
                'smys_mpa': 358.0,
                'expected_pf_range': (12.0, 16.0),
                'notes': 'Long flaw, z > 20'
            }
        ]
        
        # Modified B31G validation cases
        self.modified_b31g_test_cases = [
            {
                'name': 'ModB31G_Case_1',
                'depth_pct': 35.0,
                'length_mm': 75.0,
                'diameter_mm': 508.0,
                'wall_thickness_mm': 12.7,
                'smys_mpa': 358.0,
                'expected_pf_range': (16.0, 22.0),
                'notes': 'Modified B31G comparison'
            }
        ]
    
    def validate_b31g_implementation(self, calculate_b31g_func) -> Dict:
        """
        Validate B31G implementation against known test cases.
        
        Parameters:
        - calculate_b31g_func: The B31G calculation function to test
        
        Returns:
        - Dict with validation results
        """
        results = {
            'passed_tests': 0,
            'total_tests': len(self.b31g_test_cases),
            'test_details': [],
            'overall_status': 'UNKNOWN'
        }
        
        for test_case in self.b31g_test_cases:
            try:
                # Run the calculation
                calc_result = calculate_b31g_func(
                    defect_depth_pct=test_case['depth_pct'],
                    defect_length_mm=test_case['length_mm'],
                    pipe_diameter_mm=test_case['diameter_mm'],
                    wall_thickness_mm=test_case['wall_thickness_mm'],
                    smys_mpa=test_case['smys_mpa'],
                    safety_factor=1.0  # Use 1.0 for validation to get raw failure pressure
                )
                
                if calc_result['safe']:
                    failure_pressure = calc_result['failure_pressure_mpa']
                    expected_min, expected_max = test_case['expected_pf_range']
                    
                    # Check if result is within expected range
                    within_range = expected_min <= failure_pressure <= expected_max
                    percent_error = abs(failure_pressure - (expected_min + expected_max) / 2) / ((expected_min + expected_max) / 2) * 100
                    
                    test_result = {
                        'test_name': test_case['name'],
                        'calculated_pf': failure_pressure,
                        'expected_range': test_case['expected_pf_range'],
                        'within_range': within_range,
                        'percent_error': percent_error,
                        'status': 'PASS' if within_range else 'FAIL',
                        'notes': test_case['notes']
                    }
                    
                    if within_range:
                        results['passed_tests'] += 1
                else:
                    test_result = {
                        'test_name': test_case['name'],
                        'status': 'FAIL',
                        'error': 'Calculation returned unsafe/invalid result',
                        'notes': test_case['notes']
                    }
                
                results['test_details'].append(test_result)
                
            except Exception as e:
                test_result = {
                    'test_name': test_case['name'],
                    'status': 'ERROR',
                    'error': str(e),
                    'notes': test_case['notes']
                }
                results['test_details'].append(test_result)
        
        # Determine overall status
        pass_rate = results['passed_tests'] / results['total_tests']
        if pass_rate >= 0.8:
            results['overall_status'] = 'PASS'
        elif pass_rate >= 0.6:
            results['overall_status'] = 'WARNING'
        else:
            results['overall_status'] = 'FAIL'
        
        results['pass_rate'] = pass_rate
        
        return results
    
    def validate_modified_b31g_implementation(self, calculate_modified_b31g_func) -> Dict:
        """
        Validate Modified B31G implementation.
        """
        results = {
            'passed_tests': 0,
            'total_tests': len(self.modified_b31g_test_cases),
            'test_details': [],
            'overall_status': 'UNKNOWN'
        }
        
        for test_case in self.modified_b31g_test_cases:
            try:
                calc_result = calculate_modified_b31g_func(
                    defect_depth_pct=test_case['depth_pct'],
                    defect_length_mm=test_case['length_mm'],
                    pipe_diameter_mm=test_case['diameter_mm'],
                    wall_thickness_mm=test_case['wall_thickness_mm'],
                    smys_mpa=test_case['smys_mpa'],
                    safety_factor=1.0
                )
                
                if calc_result['safe']:
                    failure_pressure = calc_result['failure_pressure_mpa']
                    expected_min, expected_max = test_case['expected_pf_range']
                    
                    within_range = expected_min <= failure_pressure <= expected_max
                    percent_error = abs(failure_pressure - (expected_min + expected_max) / 2) / ((expected_min + expected_max) / 2) * 100
                    
                    test_result = {
                        'test_name': test_case['name'],
                        'calculated_pf': failure_pressure,
                        'expected_range': test_case['expected_pf_range'],
                        'within_range': within_range,
                        'percent_error': percent_error,
                        'status': 'PASS' if within_range else 'FAIL',
                        'notes': test_case['notes']
                    }
                    
                    if within_range:
                        results['passed_tests'] += 1
                else:
                    test_result = {
                        'test_name': test_case['name'],
                        'status': 'FAIL',
                        'error': 'Calculation returned unsafe/invalid result',
                        'notes': test_case['notes']
                    }
                
                results['test_details'].append(test_result)
                
            except Exception as e:
                test_result = {
                    'test_name': test_case['name'],
                    'status': 'ERROR',
                    'error': str(e),
                    'notes': test_case['notes']
                }
                results['test_details'].append(test_result)
        
        # Determine overall status
        pass_rate = results['passed_tests'] / results['total_tests']
        if pass_rate >= 0.8:
            results['overall_status'] = 'PASS'
        elif pass_rate >= 0.6:
            results['overall_status'] = 'WARNING'
        else:
            results['overall_status'] = 'FAIL'
        
        results['pass_rate'] = pass_rate
        
        return results
    
    def cross_validate_assessment_methods(self, 
                                        enhanced_df: pd.DataFrame,
                                        tolerance_pct: float = 20.0) -> Dict:
        """
        Cross-validate different assessment methods for consistency.
        
        Parameters:
        - enhanced_df: DataFrame with B31G, Modified B31G, and RSTRENG results
        - tolerance_pct: Acceptable percentage difference between methods
        
        Returns:
        - Dict with cross-validation results
        """
        if enhanced_df.empty:
            return {'status': 'NO_DATA', 'message': 'No data to validate'}
        
        methods = ['b31g', 'modified_b31g', 'simplified_eff_area']
        comparison_results = {
            'total_defects': len(enhanced_df),
            'method_comparisons': [],
            'outliers': [],
            'overall_consistency': 'UNKNOWN'
        }
        
        # Compare each pair of methods
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                col1 = f'{method1}_failure_pressure_mpa'
                col2 = f'{method2}_failure_pressure_mpa'
                
                if col1 in enhanced_df.columns and col2 in enhanced_df.columns:
                    # Filter out invalid results
                    valid_data = enhanced_df[
                        (enhanced_df[f'{method1}_safe'] == True) & 
                        (enhanced_df[f'{method2}_safe'] == True) &
                        (enhanced_df[col1] > 0) & 
                        (enhanced_df[col2] > 0)
                    ]
                    
                    if len(valid_data) > 0:
                        # Calculate percentage differences
                        pct_diffs = abs(valid_data[col1] - valid_data[col2]) / \
                                   ((valid_data[col1] + valid_data[col2]) / 2) * 100
                        
                        within_tolerance = (pct_diffs <= tolerance_pct).sum()
                        consistency_rate = within_tolerance / len(valid_data)
                        
                        comparison = {
                            'method1': method1,
                            'method2': method2,
                            'valid_comparisons': len(valid_data),
                            'within_tolerance': within_tolerance,
                            'consistency_rate': consistency_rate,
                            'avg_percent_diff': pct_diffs.mean(),
                            'max_percent_diff': pct_diffs.max(),
                            'status': 'GOOD' if consistency_rate >= 0.8 else 
                                     'ACCEPTABLE' if consistency_rate >= 0.6 else 'POOR'
                        }
                        
                        comparison_results['method_comparisons'].append(comparison)
                        
                        # Identify outliers (large differences)
                        outlier_indices = valid_data[pct_diffs > tolerance_pct * 2].index
                        for idx in outlier_indices:
                            outlier = {
                                'defect_index': idx,
                                'location_m': enhanced_df.loc[idx, 'log dist. [m]'],
                                'depth_pct': enhanced_df.loc[idx, 'depth [%]'],
                                'length_mm': enhanced_df.loc[idx, 'length [mm]'],
                                'method1': method1,
                                'method2': method2,
                                'value1': enhanced_df.loc[idx, col1],
                                'value2': enhanced_df.loc[idx, col2],
                                'percent_diff': pct_diffs.loc[idx]
                            }
                            comparison_results['outliers'].append(outlier)
        
        # Determine overall consistency
        if comparison_results['method_comparisons']:
            avg_consistency = np.mean([comp['consistency_rate'] for comp in comparison_results['method_comparisons']])
            if avg_consistency >= 0.8:
                comparison_results['overall_consistency'] = 'GOOD'
            elif avg_consistency >= 0.6:
                comparison_results['overall_consistency'] = 'ACCEPTABLE'
            else:
                comparison_results['overall_consistency'] = 'POOR'
            
            comparison_results['avg_consistency_rate'] = avg_consistency
        
        return comparison_results
    
    def validate_clustering_results(self, 
                                  original_defects: pd.DataFrame,
                                  clustered_defects: pd.DataFrame) -> Dict:
        """
        Validate defect clustering results for physical reasonableness.
        
        Parameters:
        - original_defects: Original individual defects
        - clustered_defects: Defects after clustering combination
        
        Returns:
        - Dict with clustering validation results
        """
        validation_results = {
            'original_count': len(original_defects),
            'clustered_count': len(clustered_defects),
            'reduction_ratio': 1 - len(clustered_defects) / len(original_defects) if len(original_defects) > 0 else 0,
            'validation_checks': [],
            'overall_status': 'UNKNOWN'
        }
        
        issues = []
        
        # Check 1: Reasonable reduction in defect count
        if validation_results['reduction_ratio'] > 0.5:
            issues.append({
                'check': 'Defect Count Reduction',
                'status': 'WARNING',
                'message': f'Large reduction in defect count ({validation_results["reduction_ratio"]:.1%}). Verify clustering criteria.'
            })
        elif validation_results['reduction_ratio'] < 0.05:
            issues.append({
                'check': 'Defect Count Reduction',
                'status': 'INFO',
                'message': f'Minimal clustering occurred ({validation_results["reduction_ratio"]:.1%}). Check if interaction distances are appropriate.'
            })
        else:
            issues.append({
                'check': 'Defect Count Reduction',
                'status': 'PASS',
                'message': f'Reasonable clustering reduction ({validation_results["reduction_ratio"]:.1%})'
            })
        
        # Check 2: Combined defect dimensions are physically reasonable
        combined_defects = clustered_defects[clustered_defects.get('is_combined', False) == True]
        
        for idx, defect in combined_defects.iterrows():
            if 'num_original_defects' in defect and defect['num_original_defects'] > 1:
                # Check if combined length is reasonable
                if defect['length [mm]'] > 2000:  # > 2 meters
                    issues.append({
                        'check': 'Combined Length Check',
                        'status': 'WARNING',
                        'message': f'Very large combined defect length ({defect["length [mm]"]:.1f}mm) at {defect["log dist. [m]"]:.1f}m'
                    })
                
                # Check if combined width exceeds half circumference
                if 'pipe_diameter_mm' in locals():
                    max_reasonable_width = np.pi * pipe_diameter_mm / 2  # Half circumference
                    if defect['width [mm]'] > max_reasonable_width:
                        issues.append({
                            'check': 'Combined Width Check',
                            'status': 'WARNING',
                            'message': f'Combined width ({defect["width [mm]"]:.1f}mm) exceeds half circumference'
                        })
        
        # Check 3: Mass/volume conservation
        if 'depth [%]' in original_defects.columns and 'length [mm]' in original_defects.columns and 'width [mm]' in original_defects.columns:
            original_volume = (original_defects['depth [%]'] * original_defects['length [mm]'] * original_defects['width [mm]']).sum()
            clustered_volume = (clustered_defects['depth [%]'] * clustered_defects['length [mm]'] * clustered_defects['width [mm]']).sum()
            
            volume_change = abs(clustered_volume - original_volume) / original_volume if original_volume > 0 else 0
            
            if volume_change > 0.2:  # 20% change
                issues.append({
                    'check': 'Volume Conservation',
                    'status': 'WARNING',
                    'message': f'Significant volume change during clustering ({volume_change:.1%}). Check vector summation logic.'
                })
            else:
                issues.append({
                    'check': 'Volume Conservation',
                    'status': 'PASS',
                    'message': f'Volume reasonably conserved ({volume_change:.1%} change)'
                })
        
        validation_results['validation_checks'] = issues
        
        # Determine overall status
        warning_count = sum(1 for issue in issues if issue['status'] == 'WARNING')
        error_count = sum(1 for issue in issues if issue['status'] == 'ERROR')
        
        if error_count > 0:
            validation_results['overall_status'] = 'FAIL'
        elif warning_count > 2:
            validation_results['overall_status'] = 'WARNING'
        else:
            validation_results['overall_status'] = 'PASS'
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """
        Generate a human-readable validation report.
        
        Parameters:
        - validation_results: Combined results from all validation methods
        
        Returns:
        - Formatted string report
        """
        report_lines = [
            "=" * 60,
            "FFS ASSESSMENT VALIDATION REPORT",
            "=" * 60,
            ""
        ]
        
        # B31G Validation
        if 'b31g_validation' in validation_results:
            b31g = validation_results['b31g_validation']
            report_lines.extend([
                "B31G IMPLEMENTATION VALIDATION:",
                f"  Status: {b31g['overall_status']}",
                f"  Tests Passed: {b31g['passed_tests']}/{b31g['total_tests']} ({b31g['pass_rate']:.1%})",
                ""
            ])
            
            for test in b31g['test_details']:
                if test['status'] != 'PASS':
                    report_lines.append(f"  ⚠️  {test['test_name']}: {test['status']}")
                    if 'error' in test:
                        report_lines.append(f"      Error: {test['error']}")
            
            report_lines.append("")
        
        # Cross-validation
        if 'cross_validation' in validation_results:
            cross_val = validation_results['cross_validation']
            report_lines.extend([
                "CROSS-METHOD VALIDATION:",
                f"  Overall Consistency: {cross_val['overall_consistency']}",
                f"  Average Consistency Rate: {cross_val.get('avg_consistency_rate', 0):.1%}",
                ""
            ])
            
            if cross_val['outliers']:
                report_lines.append(f"  Outliers Found: {len(cross_val['outliers'])}")
                for outlier in cross_val['outliers'][:3]:  # Show first 3 outliers
                    report_lines.append(f"    Location {outlier['location_m']:.1f}m: {outlier['percent_diff']:.1f}% difference")
                report_lines.append("")
        
        # Clustering validation
        if 'clustering_validation' in validation_results:
            cluster_val = validation_results['clustering_validation']
            report_lines.extend([
                "CLUSTERING VALIDATION:",
                f"  Status: {cluster_val['overall_status']}",
                f"  Defects: {cluster_val['original_count']} → {cluster_val['clustered_count']} ({cluster_val['reduction_ratio']:.1%} reduction)",
                ""
            ])
            
            for check in cluster_val['validation_checks']:
                status_icon = "✅" if check['status'] == 'PASS' else "⚠️" if check['status'] == 'WARNING' else "❌"
                report_lines.append(f"  {status_icon} {check['check']}: {check['message']}")
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 60,
            "END OF VALIDATION REPORT",
            "=" * 60
        ])
        
        return "\n".join(report_lines)