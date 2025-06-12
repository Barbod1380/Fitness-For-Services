# analysis/clustering_analysis.py

"""
Clustering analysis for pipeline defect data.
Provides multiple clustering algorithms and visualizations for defect pattern analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


def prepare_clustering_features(defects_df, feature_config):
    """
    Prepare feature matrix for clustering based on user configuration.
    
    Parameters:
    - defects_df: DataFrame with defect data
    - feature_config: Dict specifying which features to include
    
    Returns:
    - features_df: DataFrame with prepared features
    - feature_info: Dict with feature preparation info
    """
    features = []
    feature_names = []
    feature_info = {
        'numeric_features': [],
        'categorical_features': [],
        'scaled_features': [],
        'original_shape': defects_df.shape[0]
    }
    
    # Spatial features
    if feature_config.get('include_location', False):
        if 'log dist. [m]' in defects_df.columns:
            features.append(defects_df['log dist. [m]'].values.reshape(-1, 1))
            feature_names.append('location')
            feature_info['numeric_features'].append('location')
    
    if feature_config.get('include_clock', False):
        if 'clock_float' in defects_df.columns:
            # Convert clock to x,y coordinates for proper distance calculation
            clock_radians = defects_df['clock_float'] * np.pi / 6  # Convert to radians
            clock_x = np.cos(clock_radians).values.reshape(-1, 1)
            clock_y = np.sin(clock_radians).values.reshape(-1, 1)
            features.extend([clock_x, clock_y])
            feature_names.extend(['clock_x', 'clock_y'])
            feature_info['numeric_features'].extend(['clock_x', 'clock_y'])
    
    # Dimensional features
    dimension_cols = {
        'include_depth': 'depth [%]',
        'include_length': 'length [mm]',
        'include_width': 'width [mm]'
    }
    
    for config_key, col_name in dimension_cols.items():
        if feature_config.get(config_key, False) and col_name in defects_df.columns:
            values = pd.to_numeric(defects_df[col_name], errors='coerce')
            if not values.isna().all():
                features.append(values.fillna(values.median()).values.reshape(-1, 1))
                feature_names.append(col_name.split(' [')[0])  # Clean name
                feature_info['numeric_features'].append(col_name.split(' [')[0])
    
    # Categorical features
    if feature_config.get('include_defect_type', False):
        if 'component / anomaly identification' in defects_df.columns:
            le = LabelEncoder()
            defect_types = defects_df['component / anomaly identification'].fillna('Unknown')
            encoded_types = le.fit_transform(defect_types).reshape(-1, 1)
            features.append(encoded_types)
            feature_names.append('defect_type')
            feature_info['categorical_features'].append('defect_type')
            feature_info['defect_type_mapping'] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    if feature_config.get('include_joint', False):
        if 'joint number' in defects_df.columns:
            joint_nums = pd.to_numeric(defects_df['joint number'], errors='coerce')
            if not joint_nums.isna().all():
                features.append(joint_nums.fillna(0).values.reshape(-1, 1))
                feature_names.append('joint_number')
                feature_info['numeric_features'].append('joint_number')
    
    if not features:
        raise ValueError("No valid features selected for clustering")
    
    # Combine features
    feature_matrix = np.hstack(features)
    features_df = pd.DataFrame(feature_matrix, columns=feature_names)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_indices = [i for i, name in enumerate(feature_names) 
                      if name in feature_info['numeric_features']]
    
    if numeric_indices:
        features_df.iloc[:, numeric_indices] = scaler.fit_transform(
            features_df.iloc[:, numeric_indices]
        )
        feature_info['scaler'] = scaler
        feature_info['scaled_features'] = [feature_names[i] for i in numeric_indices]
    
    return features_df, feature_info


def perform_clustering(features_df, algorithm='kmeans', algorithm_params=None):
    """
    Perform clustering using specified algorithm.
    
    Parameters:
    - features_df: DataFrame with prepared features
    - algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
    - algorithm_params: Dict with algorithm-specific parameters
    
    Returns:
    - cluster_labels: Array of cluster labels
    - clustering_info: Dict with clustering performance metrics
    """
    if algorithm_params is None:
        algorithm_params = {}
    
    X = features_df.values
    
    if algorithm == 'kmeans':
        n_clusters = int(algorithm_params.get('n_clusters', 5))
        random_state = algorithm_params.get('random_state', 42)
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        
    elif algorithm == 'dbscan':
        eps = algorithm_params.get('eps', 0.5)
        min_samples = algorithm_params.get('min_samples', 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        
    elif algorithm == 'hierarchical':
        n_clusters = int(algorithm_params.get('n_clusters', 5))
        linkage = algorithm_params.get('linkage', 'ward')
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    cluster_labels = clusterer.fit_predict(X)
    
    # Calculate clustering metrics
    clustering_info = {
        'algorithm': algorithm,
        'parameters': algorithm_params,
        'n_clusters_found': len(np.unique(cluster_labels)),
        'n_noise_points': np.sum(cluster_labels == -1) if -1 in cluster_labels else 0
    }
    
    # Calculate silhouette score (if more than 1 cluster and not all noise)
    valid_labels = cluster_labels[cluster_labels != -1]
    if len(np.unique(valid_labels)) > 1 and len(valid_labels) > 1:
        # Filter out noise points for silhouette calculation
        valid_indices = cluster_labels != -1
        if np.sum(valid_indices) > 1:
            silhouette = silhouette_score(X[valid_indices], valid_labels)
            clustering_info['silhouette_score'] = silhouette
            
            # Calinski-Harabasz score
            ch_score = calinski_harabasz_score(X[valid_indices], valid_labels)
            clustering_info['calinski_harabasz_score'] = ch_score
    
    return cluster_labels, clustering_info


def optimize_clustering_parameters(features_df, algorithm='kmeans', param_ranges=None):
    """
    Find optimal clustering parameters using silhouette analysis.
    
    Parameters:
    - features_df: DataFrame with prepared features
    - algorithm: Clustering algorithm to optimize
    - param_ranges: Dict with parameter ranges to test
    
    Returns:
    - best_params: Dict with optimal parameters
    - optimization_results: DataFrame with all tested combinations
    """
    if param_ranges is None:
        if algorithm == 'kmeans':
            param_ranges = {'n_clusters': range(2, 11)}
        elif algorithm == 'dbscan':
            param_ranges = {
                'eps': np.arange(0.1, 2.0, 0.2),
                'min_samples': range(3, 11)
            }
        elif algorithm == 'hierarchical':
            param_ranges = {'n_clusters': range(2, 11)}
    
    X = features_df.values
    results = []
    
    if algorithm == 'kmeans':
        for n_clusters in param_ranges['n_clusters']:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X)
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
                inertia = clusterer.inertia_
                
                results.append({
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': ch_score,
                    'inertia': inertia
                })
    
    elif algorithm == 'dbscan':
        for eps in param_ranges['eps']:
            for min_samples in param_ranges['min_samples']:
                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                labels = clusterer.fit_predict(X)
                
                valid_labels = labels[labels != -1]
                n_clusters = int(len(np.unique(valid_labels)))
                n_noise = np.sum(labels == -1)
                
                if n_clusters > 1 and len(valid_labels) > 1:
                    valid_indices = labels != -1
                    silhouette = silhouette_score(X[valid_indices], valid_labels)
                    ch_score = calinski_harabasz_score(X[valid_indices], valid_labels)
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': ch_score
                    })
    
    elif algorithm == 'hierarchical':
        for n_clusters in param_ranges['n_clusters']:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clusterer.fit_predict(X)
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
                
                results.append({
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': ch_score
                })
    
    if not results:
        raise ValueError("No valid clustering results found")
    
    results_df = pd.DataFrame(results)
    
    # Find best parameters based on silhouette score
    best_idx = results_df['silhouette_score'].idxmax()
    best_params = results_df.iloc[best_idx].to_dict()
    
    # Remove scores from best_params
    param_keys = list(param_ranges.keys())
    best_params = {k: best_params[k] for k in param_keys if k in best_params}
    
    return best_params, results_df


def create_cluster_summary(defects_df, cluster_labels, features_used):
    """
    Create summary statistics for each cluster.
    
    Parameters:
    - defects_df: Original defects DataFrame
    - cluster_labels: Array of cluster labels
    - features_used: List of features used in clustering
    
    Returns:
    - summary_df: DataFrame with cluster summaries
    """
    # Add cluster labels to defects data
    df_with_clusters = defects_df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    summary_data = []
    
    for cluster_id in np.unique(cluster_labels):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        summary = {
            'cluster_id': cluster_id,
            'size': cluster_size,
            'percentage': (cluster_size / len(defects_df)) * 100
        }
        
        # Add statistics for each feature
        if 'depth [%]' in cluster_data.columns:
            depths = pd.to_numeric(cluster_data['depth [%]'], errors='coerce')
            summary['avg_depth'] = depths.mean()
            summary['max_depth'] = depths.max()
        
        if 'length [mm]' in cluster_data.columns:
            lengths = pd.to_numeric(cluster_data['length [mm]'], errors='coerce')
            summary['avg_length'] = lengths.mean()
        
        if 'width [mm]' in cluster_data.columns:
            widths = pd.to_numeric(cluster_data['width [mm]'], errors='coerce')
            summary['avg_width'] = widths.mean()
        
        if 'log dist. [m]' in cluster_data.columns:
            summary['location_range'] = (
                f"{cluster_data['log dist. [m]'].min():.1f} - "
                f"{cluster_data['log dist. [m]'].max():.1f} m"
            )
        
        # Most common defect type
        if 'component / anomaly identification' in cluster_data.columns:
            defect_types = cluster_data['component / anomaly identification'].value_counts()
            if not defect_types.empty:
                summary['primary_defect_type'] = defect_types.index[0]
                summary['type_percentage'] = (defect_types.iloc[0] / cluster_size) * 100
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def create_cluster_visualization_2d(defects_df, cluster_labels, feature_config):
    """
    Create 2D visualization of clusters using PCA or selected features.
    
    Parameters:
    - defects_df: Original defects DataFrame
    - cluster_labels: Array of cluster labels
    - feature_config: Dict with feature configuration
    
    Returns:
    - Plotly figure object
    """
    # Prepare features for visualization
    features_df, _ = prepare_clustering_features(defects_df, feature_config)
    
    # If more than 2 features, use PCA
    if features_df.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features_df)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
        x_vals, y_vals = coords[:, 0], coords[:, 1]
    else:
        x_vals = features_df.iloc[:, 0]
        y_vals = features_df.iloc[:, 1] if features_df.shape[1] > 1 else np.zeros(len(features_df))
        x_label = features_df.columns[0] if features_df.shape[1] > 0 else "Feature 1"
        y_label = features_df.columns[1] if features_df.shape[1] > 1 else "Feature 2"
    
    # Create scatter plot
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    unique_clusters = np.unique(cluster_labels)
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_name = f"Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        color = "gray" if cluster_id == -1 else colors[i % len(colors)]
        
        # Prepare hover text
        hover_text = []
        cluster_defects = defects_df[cluster_mask]
        for _, defect in cluster_defects.iterrows():
            text = f"<b>{cluster_name}</b><br>"
            text += f"Location: {defect.get('log dist. [m]', 'N/A'):.2f}m<br>"
            if 'depth [%]' in defect:
                text += f"Depth: {defect['depth [%]']:.1f}%<br>"
            if 'component / anomaly identification' in defect:
                text += f"Type: {defect['component / anomaly identification']}"
            hover_text.append(text)
        
        fig.add_trace(go.Scatter(
            x=x_vals[cluster_mask],
            y=y_vals[cluster_mask],
            mode='markers',
            marker=dict(
                size=8,
                color=color,
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            name=cluster_name,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Defect Clusters Visualization",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_cluster_pipeline_visualization(defects_df, cluster_labels):
    """
    Create pipeline visualization with clusters color-coded.
    
    Parameters:
    - defects_df: Original defects DataFrame
    - cluster_labels: Array of cluster labels
    
    Returns:
    - Plotly figure object
    """
    if 'log dist. [m]' not in defects_df.columns or 'clock_float' not in defects_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Pipeline visualization requires location and clock data",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    unique_clusters = np.unique(cluster_labels)
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_defects = defects_df[cluster_mask]
        cluster_name = f"Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        color = "gray" if cluster_id == -1 else colors[i % len(colors)]
        
        # Prepare hover text
        hover_text = []
        for _, defect in cluster_defects.iterrows():
            text = f"<b>{cluster_name}</b><br>"
            text += f"Location: {defect['log dist. [m]']:.2f}m<br>"
            text += f"Clock: {defect.get('clock', 'N/A')}<br>"
            if 'depth [%]' in defect:
                text += f"Depth: {defect['depth [%]']:.1f}%<br>"
            if 'component / anomaly identification' in defect:
                text += f"Type: {defect['component / anomaly identification']}"
            hover_text.append(text)
        
        fig.add_trace(go.Scatter(
            x=cluster_defects['log dist. [m]'],
            y=cluster_defects['clock_float'],
            mode='markers',
            marker=dict(
                size=10,
                color=color,
                opacity=0.8,
                line=dict(width=2, color='black')
            ),
            name=cluster_name,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add clock hour grid
    min_dist = defects_df['log dist. [m]'].min() - 1
    max_dist = defects_df['log dist. [m]'].max() + 1
    for hour in range(1, 13):
        fig.add_shape(
            type="line", x0=min_dist, x1=max_dist, y0=hour, y1=hour,
            line=dict(color="lightgray", dash="dot", width=1), layer="below"
        )
    
    fig.update_layout(
        title="Pipeline Defect Clusters",
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title="Clock Position (hr)",
        yaxis=dict(
            tickmode="array", tickvals=list(range(1, 13)),
            ticktext=[f"{h}:00" for h in range(1, 13)],
            range=[0.5, 12.5]
        ),
        height=600,
        plot_bgcolor="white",
        hovermode='closest'
    )
    
    return fig


def create_cluster_characteristics_plot(summary_df):
    """
    Create radar/spider chart showing cluster characteristics.
    
    Parameters:
    - summary_df: DataFrame with cluster summaries
    
    Returns:
    - Plotly figure object
    """
    if summary_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No cluster data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Select numeric columns for radar chart
    numeric_cols = ['avg_depth', 'avg_length', 'avg_width', 'size']
    available_cols = [col for col in numeric_cols if col in summary_df.columns]
    
    if len(available_cols) < 2:
        # Fallback to bar chart if not enough dimensions
        return create_cluster_bar_chart(summary_df)
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    for i, (_, cluster) in enumerate(summary_df.iterrows()):
        cluster_id = cluster['cluster_id']
        if cluster_id == -1:
            continue  # Skip noise cluster for radar chart
        
        # Normalize values to 0-1 range for radar chart
        values = []
        labels = []
        for col in available_cols:
            if pd.notna(cluster[col]):
                values.append(cluster[col])
                labels.append(col.replace('avg_', '').replace('_', ' ').title())
        
        if values:
            # Normalize values
            max_vals = summary_df[available_cols].max()
            normalized_values = [val / max_vals[col] for val, col in zip(values, available_cols)]
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],  # Close the polygon
                theta=labels + [labels[0]],
                fill='toself',
                name=f'Cluster {cluster_id}',
                line_color=colors[i % len(colors)],
                opacity=0.6
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Cluster Characteristics Comparison",
        height=500
    )
    
    return fig


def create_cluster_bar_chart(summary_df):
    """
    Create bar chart of cluster characteristics.
    
    Parameters:
    - summary_df: DataFrame with cluster summaries
    
    Returns:
    - Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Cluster Sizes', 'Average Depth', 'Average Length', 'Average Width'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    clusters = summary_df[summary_df['cluster_id'] != -1]  # Exclude noise
    
    # Cluster sizes
    fig.add_trace(go.Bar(
        x=clusters['cluster_id'],
        y=clusters['size'],
        name='Size',
        showlegend=False
    ), row=1, col=1)
    
    # Average depth
    if 'avg_depth' in clusters.columns:
        fig.add_trace(go.Bar(
            x=clusters['cluster_id'],
            y=clusters['avg_depth'],
            name='Avg Depth',
            showlegend=False
        ), row=1, col=2)
    
    # Average length
    if 'avg_length' in clusters.columns:
        fig.add_trace(go.Bar(
            x=clusters['cluster_id'],
            y=clusters['avg_length'],
            name='Avg Length',
            showlegend=False
        ), row=2, col=1)
    
    # Average width
    if 'avg_width' in clusters.columns:
        fig.add_trace(go.Bar(
            x=clusters['cluster_id'],
            y=clusters['avg_width'],
            name='Avg Width',
            showlegend=False
        ), row=2, col=2)
    
    fig.update_layout(height=600, title="Cluster Characteristics")
    return fig