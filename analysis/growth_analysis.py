"""
Functions for analyzing defect growth rates.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def correct_negative_growth_rates(matches_df, k=3):
    """
    Correct negative depth growth rates using K-Nearest Neighbors from similar defects in the same joint.
    
    Parameters:
    - matches_df: DataFrame with matched defects from two inspections
    - k: Number of neighbors to consider (default: 3)
    
    Returns:
    - Tuple of (corrected_df, correction_info) where correction_info contains statistics
    """
    # Make a copy to avoid modifying the original
    df = matches_df.copy()
    
    # Check if we have the necessary columns for depth correction
    required_cols = ['joint number', 'old_depth_pct', 'new_depth_pct', 'is_negative_growth']
    dimension_cols = ['length [mm]', 'width [mm]']

    # Check required columns
    for col in required_cols:
        if col not in df.columns:
            return df, {"error": f"Missing required column: {col}", "success": False}    

    available_dimension_cols = [col for col in dimension_cols if col in df.columns]
    if not available_dimension_cols:
        return df, {"error": f"Missing at least one dimension column (length [mm] or width [mm]) for features", "success": False}
    
    has_mm_data = 'growth_rate_mm_per_year' in df.columns
    
    # Identify defects with negative depth growth only
    negative_growth = df[df['is_negative_growth']].copy()
    positive_growth = df[~df['is_negative_growth']].copy()
    
    if negative_growth.empty or positive_growth.empty:
        return df, {"error": "No negative or positive depth growth defects found", "success": False}
    
    # Add a column to track corrections
    df['is_corrected'] = False
    
    # Statistics for reporting
    total_negative = len(negative_growth)
    corrected_count = 0
    uncorrected_count = 0
    uncorrected_joints = set()
    
    # Dictionary to store corrections
    corrections = {}
    
    try:
        # Process each joint separately
        for joint_num in df['joint number'].unique():
            # Get defects in this joint
            joint_negative = negative_growth[negative_growth['joint number'] == joint_num]
            joint_positive = positive_growth[positive_growth['joint number'] == joint_num]
            
            # Skip if no negative defects to correct in this joint
            if len(joint_negative) == 0:
                continue
                
            # Check if enough positive defects for comparison
            if len(joint_positive) < k:
                # Not enough neighbors in this joint
                uncorrected_count += len(joint_negative)
                uncorrected_joints.add(joint_num)
                continue
            
            # Features for KNN - use available dimensions plus old depth
            features = available_dimension_cols + ['old_depth_pct']
            
            # Standardize the features (important for KNN)
            scaler = StandardScaler()
            
            # Fit scaler on positive growth defects
            X_positive = scaler.fit_transform(joint_positive[features])
            
            # Transform negative growth defects
            X_negative = scaler.transform(joint_negative[features])
            
            # Create and fit KNN model
            k_value = min(k, len(joint_positive))  # Ensure k is not larger than available samples
            knn = NearestNeighbors(n_neighbors=k_value)
            knn.fit(X_positive)
            
            # Find k nearest neighbors for each negative growth defect
            distances, indices = knn.kneighbors(X_negative)
            
            # Calculate and apply corrections for DEPTH only
            for i, idx in enumerate(joint_negative.index):
                # Get indices of nearest neighbors in the positive growth dataset
                nn_indices = indices[i]
                pos_indices = joint_positive.index[nn_indices]
                
                # Calculate average growth rate of nearest neighbors
                avg_growth_pct = joint_positive.loc[pos_indices, 'growth_rate_pct_per_year'].mean()
                year_diff = df.loc[idx, 'new_year'] - df.loc[idx, 'old_year']
                
                corrections[idx] = {
                    'growth_rate_pct_per_year': avg_growth_pct,
                    'depth_change_pct': avg_growth_pct * year_diff,
                    'new_depth_pct': df.loc[idx, 'old_depth_pct'] + (avg_growth_pct * year_diff),
                    'is_negative_growth': False,
                    'is_corrected': True
                }
                
                # Handle mm-based corrections if available
                if has_mm_data:
                    avg_growth_mm = joint_positive.loc[pos_indices, 'growth_rate_mm_per_year'].mean()
                    corrections[idx].update({
                        'growth_rate_mm_per_year': avg_growth_mm,
                        'depth_change_mm': avg_growth_mm * year_diff,
                        'new_depth_mm': df.loc[idx, 'old_depth_mm'] + (avg_growth_mm * year_diff)
                    })
                
                corrected_count += 1
    
    except Exception as e:
        return df, {"error": f"Error during correction: {str(e)}", "success": False}
    
    # Apply corrections
    for idx, correction in corrections.items():
        for col, value in correction.items():
            df.loc[idx, col] = value
    
    # Calculate updated growth statistics after correction
    updated_growth_stats = None
    if corrected_count > 0:
        # Growth statistics for depth only
        negative_growth_count = df['is_negative_growth'].sum()
        pct_negative_growth = (negative_growth_count / len(df)) * 100 if len(df) > 0 else 0
        
        # Filter out negative growth for positive growth stats
        positive_growth = df[~df['is_negative_growth']]
        
        updated_growth_stats = {
            'total_matched_defects': len(df),
            'negative_growth_count': negative_growth_count,
            'pct_negative_growth': pct_negative_growth,
            'avg_growth_rate_pct': df['growth_rate_pct_per_year'].mean(),
            'avg_positive_growth_rate_pct': positive_growth['growth_rate_pct_per_year'].mean() if len(positive_growth) > 0 else 0,
            'max_growth_rate_pct': positive_growth['growth_rate_pct_per_year'].max() if len(positive_growth) > 0 else 0
        }
        
        # Add mm-based stats if available
        if has_mm_data:
            updated_growth_stats.update({
                'avg_growth_rate_mm': df['growth_rate_mm_per_year'].mean(),
                'avg_positive_growth_rate_mm': positive_growth['growth_rate_mm_per_year'].mean() if len(positive_growth) > 0 else 0,
                'max_growth_rate_mm': positive_growth['growth_rate_mm_per_year'].max() if len(positive_growth) > 0 else 0
            })
    
    # Prepare correction information
    correction_info = {
        "total_negative": total_negative,
        "corrected_count": corrected_count,
        "uncorrected_count": uncorrected_count,
        "uncorrected_joints": list(uncorrected_joints),
        "success": corrected_count > 0
    }
    
    # Add updated growth stats if calculated
    if updated_growth_stats:
        correction_info["updated_growth_stats"] = updated_growth_stats
    
    return df, correction_info


def create_growth_summary_table(comparison_results):
    """
    Create a summary table of growth statistics
    
    Parameters:
    - comparison_results: Results dictionary from compare_defects function
    
    Returns:
    - Pandas DataFrame with growth statistics
    """
    if (not comparison_results.get('has_depth_data', False) or 
        comparison_results['matches_df'].empty or
        not comparison_results.get('calculate_growth', False) or
        comparison_results.get('growth_stats') is None):
        return pd.DataFrame()
    
    growth_stats = comparison_results['growth_stats']
    has_wt_data = comparison_results.get('has_wt_data', False)
    
    # Create table rows
    rows = []
    
    # Total matched defects
    rows.append({
        'Statistic': 'Total Matched Defects',
        'Value': growth_stats['total_matched_defects']
    })
    
    # Negative growth anomalies
    rows.append({
        'Statistic': 'Negative Growth Anomalies',
        'Value': f"{growth_stats['negative_growth_count']} ({growth_stats['pct_negative_growth']:.1f}%)"
    })
    
    # Average growth rate
    if has_wt_data:
        rows.append({
            'Statistic': 'Average Growth Rate (All)',
            'Value': f"{growth_stats['avg_growth_rate_mm']:.3f} mm/year"
        })
        rows.append({
            'Statistic': 'Average Positive Growth Rate',
            'Value': f"{growth_stats['avg_positive_growth_rate_mm']:.3f} mm/year"
        })
        rows.append({
            'Statistic': 'Maximum Growth Rate',
            'Value': f"{growth_stats['max_growth_rate_mm']:.3f} mm/year"
        })
    else:
        rows.append({
            'Statistic': 'Average Growth Rate (All)',
            'Value': f"{growth_stats['avg_growth_rate_pct']:.3f} %/year"
        })
        rows.append({
            'Statistic': 'Average Positive Growth Rate',
            'Value': f"{growth_stats['avg_positive_growth_rate_pct']:.3f} %/year"
        })
        rows.append({
            'Statistic': 'Maximum Growth Rate',
            'Value': f"{growth_stats['max_growth_rate_pct']:.3f} %/year"
        })
    
    return pd.DataFrame(rows)


def create_highest_growth_table(comparison_results, top_n=10):
    """
    Create a table showing the defects with the highest growth rates
    
    Parameters:
    - comparison_results: Results dictionary from compare_defects function
    - top_n: Number of top defects to include
    
    Returns:
    - Pandas DataFrame with top growing defects
    """
    if (not comparison_results.get('has_depth_data', False) or 
        comparison_results['matches_df'].empty or
        not comparison_results.get('calculate_growth', False)):
        return pd.DataFrame()
    
    matches_df = comparison_results['matches_df']
    has_wt_data = comparison_results.get('has_wt_data', False)
    
    # Filter out negative growth
    positive_growth = matches_df[~matches_df['is_negative_growth']]
    
    if positive_growth.empty:
        return pd.DataFrame()
    
    # Sort by growth rate
    if has_wt_data:
        sorted_df = positive_growth.sort_values('growth_rate_mm_per_year', ascending=False)
        growth_col = 'growth_rate_mm_per_year'
        unit = 'mm/year'
    else:
        sorted_df = positive_growth.sort_values('growth_rate_pct_per_year', ascending=False)
        growth_col = 'growth_rate_pct_per_year'
        unit = '%/year'
    
    # Take top N
    top_defects = sorted_df.head(top_n)
    
    # Select columns for display
    display_cols = ['log_dist', 'defect_type']
    
    if has_wt_data:
        depth_cols = ['old_depth_mm', 'new_depth_mm']
    else:
        depth_cols = ['old_depth_pct', 'new_depth_pct']
    
    display_cols.extend(depth_cols)
    display_cols.append(growth_col)
    
    # Create display dataframe
    display_df = top_defects[display_cols].copy()
    
    # Rename columns for clarity
    column_map = {
        'log_dist': 'Location (m)',
        'defect_type': 'Defect Type',
        'old_depth_pct': 'Old Depth (%)',
        'new_depth_pct': 'New Depth (%)',
        'old_depth_mm': 'Old Depth (mm)',
        'new_depth_mm': 'New Depth (mm)',
        'growth_rate_pct_per_year': f'Growth Rate ({unit})',
        'growth_rate_mm_per_year': f'Growth Rate ({unit})'
    }
    display_df = display_df.rename(columns=column_map)
    
    # Format numeric columns
    for col in display_df.columns:
        if 'Depth' in col or 'Growth' in col or 'Location' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    
    # Reset index for display
    display_df = display_df.reset_index(drop=True)
    
    return display_df