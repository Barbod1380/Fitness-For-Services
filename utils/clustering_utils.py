"""
Utility functions for clustering analysis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score


def validate_clustering_data(defects_df, feature_config):
    """
    Validate that the data is suitable for clustering analysis.
    
    Parameters:
    - defects_df: DataFrame with defect data
    - feature_config: Dictionary with feature configuration
    
    Returns:
    - is_valid: Boolean indicating if data is valid
    - issues: List of validation issues
    """
    issues = []
    
    # Check minimum number of defects
    if len(defects_df) < 5:
        issues.append("At least 5 defects required for meaningful clustering")
    
    # Check feature availability
    feature_mapping = {
        'include_location': 'log dist. [m]',
        'include_clock': 'clock_float',
        'include_depth': 'depth [%]',
        'include_length': 'length [mm]',
        'include_width': 'width [mm]',
        'include_defect_type': 'component / anomaly identification',
        'include_joint': 'joint number'
    }
    
    selected_features = []
    for config_key, column_name in feature_mapping.items():
        if feature_config.get(config_key, False):
            if column_name not in defects_df.columns:
                issues.append(f"Selected feature '{column_name}' not available in data")
            else:
                # Check for sufficient non-null values
                non_null_count = defects_df[column_name].count()
                if non_null_count < len(defects_df) * 0.7:  # At least 70% non-null
                    issues.append(f"Feature '{column_name}' has too many missing values ({non_null_count}/{len(defects_df)})")
                else:
                    selected_features.append(column_name)
    
    if not selected_features:
        issues.append("No valid features selected for clustering")
    
    # Check for feature variance
    for feature in selected_features:
        if defects_df[feature].dtype in ['float64', 'int64']:
            if defects_df[feature].var() == 0:
                issues.append(f"Feature '{feature}' has zero variance (all values are identical)")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def interpret_silhouette_score(score):
    """
    Provide interpretation of silhouette score.
    
    Parameters:
    - score: Silhouette score
    
    Returns:
    - interpretation: String interpretation
    """
    if score >= 0.7:
        return "Excellent clustering structure"
    elif score >= 0.5:
        return "Good clustering structure"
    elif score >= 0.3:
        return "Weak clustering structure"
    elif score >= 0.0:
        return "Poor clustering structure"
    else:
        return "Very poor clustering (overlapping clusters)"


def get_cluster_quality_metrics(X, labels):
    """
    Calculate comprehensive clustering quality metrics.
    
    Parameters:
    - X: Feature matrix
    - labels: Cluster labels
    
    Returns:
    - metrics: Dictionary with quality metrics
    """
    metrics = {}
    
    # Remove noise points for some calculations
    valid_mask = labels != -1
    if np.sum(valid_mask) > 0:
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        unique_labels = np.unique(labels_valid)
        metrics['n_clusters'] = len(unique_labels)
        metrics['n_noise'] = np.sum(~valid_mask)
        metrics['noise_ratio'] = metrics['n_noise'] / len(labels)
        
        if len(unique_labels) > 1:
            metrics['silhouette_score'] = silhouette_score(X_valid, labels_valid)
            metrics['silhouette_interpretation'] = interpret_silhouette_score(
                metrics['silhouette_score']
            )
            
            # Calculate within-cluster sum of squares
            wcss = 0
            for label in unique_labels:
                cluster_points = X_valid[labels_valid == label]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    wcss += np.sum((cluster_points - centroid) ** 2)
            metrics['wcss'] = wcss
            
            # Calculate cluster sizes
            cluster_sizes = [np.sum(labels_valid == label) for label in unique_labels]
            metrics['min_cluster_size'] = min(cluster_sizes)
            metrics['max_cluster_size'] = max(cluster_sizes)
            metrics['avg_cluster_size'] = np.mean(cluster_sizes)
            metrics['cluster_size_std'] = np.std(cluster_sizes)
    
    return metrics


def suggest_optimal_clusters(X, max_clusters=10):
    """
    Suggest optimal number of clusters using multiple methods.
    
    Parameters:
    - X: Feature matrix
    - max_clusters: Maximum number of clusters to test
    
    Returns:
    - suggestions: Dictionary with suggestions from different methods
    """
    from sklearn.cluster import KMeans
    
    suggestions = {}
    
    # Elbow method
    wcss = []
    k_range = range(2, min(max_clusters + 1, len(X) // 2))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Simple elbow detection (largest decrease)
    if len(wcss) > 1:
        decreases = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
        elbow_idx = np.argmax(decreases)
        suggestions['elbow_method'] = list(k_range)[elbow_idx]
    
    # Silhouette method
    silhouette_scores = []
    for k in k_range:
        if k <= len(X):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)
    
    if silhouette_scores:
        best_silhouette_idx = np.argmax(silhouette_scores)
        suggestions['silhouette_method'] = list(k_range)[best_silhouette_idx]
        suggestions['best_silhouette_score'] = max(silhouette_scores)
    
    return suggestions