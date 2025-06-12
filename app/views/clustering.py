# app/views/clustering.py

"""
Clustering analysis view for the Pipeline Analysis application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.ui_components import info_box, custom_metric
from app.services.state_manager import get_state
from analysis.clustering_analysis import (
    prepare_clustering_features,
    perform_clustering,
    optimize_clustering_parameters,
    create_cluster_summary,
    create_cluster_visualization_2d,
    create_cluster_pipeline_visualization,
    create_cluster_characteristics_plot
)


def render_clustering_view():
    """Display clustering analysis view with defect pattern identification."""
    st.markdown('<h2 class="section-header">Defect Clustering Analysis</h2>', unsafe_allow_html=True)
    
    # Check if datasets are available
    datasets = get_state('datasets', {})
    if not datasets:
        st.info("""
            **No datasets available**
            Please upload pipeline inspection data using the sidebar to enable clustering analysis.
        """)
        return
    
    # Dataset selection
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Select Dataset for Clustering</div>", unsafe_allow_html=True)
    
    available_years = sorted(datasets.keys())
    selected_year = st.selectbox(
        "Choose inspection year to analyze",
        options=available_years,
        key="clustering_year"
    )
    
    # Get selected dataset
    defects_df = datasets[selected_year]['defects_df']
    
    # Display dataset info
    total_defects = len(defects_df)
    st.markdown(f"**Selected Dataset:** {selected_year} ({total_defects} defects)")
    
    if total_defects < 5:
        st.warning("Clustering requires at least 5 defects for meaningful results.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature selection section
    st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Feature Selection</div>", unsafe_allow_html=True)
    
    st.info("""
        **Select which defect characteristics to use for clustering:**
        Different combinations will reveal different patterns in your data.
    """)
    
    # Create feature selection interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Spatial Features**")
        include_location = st.checkbox(
            "Pipeline Location (log distance)", 
            value=True,
            key="cluster_location",
            help="Group defects by their position along the pipeline"
        )
        
        include_clock = st.checkbox(
            "Clock Position", 
            value='clock_float' in defects_df.columns,
            disabled='clock_float' not in defects_df.columns,
            key="cluster_clock",
            help="Group defects by their circumferential position"
        )
        
        include_joint = st.checkbox(
            "Joint Number",
            value=False,
            disabled='joint number' not in defects_df.columns,
            key="cluster_joint",
            help="Consider joint boundaries in clustering"
        )
    
    with col2:
        st.markdown("**Defect Characteristics**")
        include_depth = st.checkbox(
            "Defect Depth (%)", 
            value='depth [%]' in defects_df.columns,
            disabled='depth [%]' not in defects_df.columns,
            key="cluster_depth",
            help="Group defects by severity (depth)"
        )
        
        include_length = st.checkbox(
            "Defect Length (mm)", 
            value='length [mm]' in defects_df.columns,
            disabled='length [mm]' not in defects_df.columns,
            key="cluster_length",
            help="Group defects by longitudinal extent"
        )
        
        include_width = st.checkbox(
            "Defect Width (mm)", 
            value='width [mm]' in defects_df.columns,
            disabled='width [mm]' not in defects_df.columns,
            key="cluster_width",
            help="Group defects by circumferential extent"
        )
        
        include_defect_type = st.checkbox(
            "Defect Type",
            value='component / anomaly identification' in defects_df.columns,
            disabled='component / anomaly identification' not in defects_df.columns,
            key="cluster_type",
            help="Group defects by type (e.g., corrosion, pitting)"
        )
    
    feature_config = {
        'include_location': include_location,
        'include_clock': include_clock,
        'include_joint': include_joint,
        'include_depth': include_depth,
        'include_length': include_length,
        'include_width': include_width,
        'include_defect_type': include_defect_type
    }
    
    # Check if at least one feature is selected
    features_selected = any(feature_config.values())
    if not features_selected:
        st.warning("Please select at least one feature for clustering.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Algorithm selection section
    st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Clustering Algorithm</div>", unsafe_allow_html=True)
    
    algorithm_col1, algorithm_col2 = st.columns(2)
    
    with algorithm_col1:
        algorithm = st.selectbox(
            "Select Clustering Algorithm",
            options=['kmeans', 'dbscan', 'hierarchical'],
            format_func=lambda x: {
                'kmeans': 'K-Means (specify number of clusters)',
                'dbscan': 'DBSCAN (density-based, finds clusters automatically)',
                'hierarchical': 'Hierarchical (builds cluster tree)'
            }[x],
            key="clustering_algorithm"
        )
        
        auto_optimize = st.checkbox(
            "Auto-optimize parameters",
            value=True,
            key="auto_optimize",
            help="Automatically find the best parameters using silhouette analysis"
        )
    
    with algorithm_col2:
        st.markdown("**Algorithm Parameters**")
        
        if algorithm == 'kmeans':
            if not auto_optimize:
                n_clusters = st.slider(
                    "Number of clusters", 
                    min_value=2, max_value=min(15, total_defects//2),
                    value=5, key="kmeans_clusters"
                )
                n_clusters = int(n_clusters)
            else:
                st.info("Number of clusters will be optimized automatically (2-10)")
                n_clusters = None
                
        elif algorithm == 'dbscan':
            if not auto_optimize:
                eps = st.slider(
                    "Epsilon (neighborhood distance)", 
                    min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                    key="dbscan_eps"
                )
                min_samples = st.slider(
                    "Minimum samples per cluster", 
                    min_value=2, max_value=20, value=5,
                    key="dbscan_min_samples"
                )
            else:
                st.info("Epsilon and min_samples will be optimized automatically")
                eps, min_samples = None, None
                
        elif algorithm == 'hierarchical':
            if not auto_optimize:
                n_clusters = st.slider(
                    "Number of clusters", 
                    min_value=2, max_value=min(15, total_defects//2),
                    value=5, key="hierarchical_clusters"
                )
                n_clusters = int(n_clusters)
                linkage = st.selectbox(
                    "Linkage method", 
                    options=['ward', 'complete', 'average', 'single'],
                    key="hierarchical_linkage"
                )
            else:
                st.info("Number of clusters will be optimized automatically")
                n_clusters, linkage = None, 'ward'
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clustering execution
    st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
    
    if st.button("Perform Clustering Analysis", key="perform_clustering", use_container_width=True):
        with st.spinner("Performing clustering analysis..."):
            try:
                # Prepare features
                features_df, feature_info = prepare_clustering_features(defects_df, feature_config)
                
                st.success(f"✅ Prepared {features_df.shape[1]} features from {features_df.shape[0]} defects")
                
                # Optimize parameters if requested
                if auto_optimize:
                    with st.spinner("Optimizing clustering parameters..."):
                        best_params, optimization_results = optimize_clustering_parameters(
                            features_df, algorithm
                        )
                        st.info(f"🎯 Optimal parameters found: {best_params}")
                        
                        # Store optimization results
                        st.session_state.clustering_optimization = optimization_results
                else:
                    # Use manual parameters
                    if algorithm == 'kmeans':
                        best_params = {'n_clusters': n_clusters}
                    elif algorithm == 'dbscan':
                        best_params = {'eps': eps, 'min_samples': min_samples}
                    elif algorithm == 'hierarchical':
                        best_params = {'n_clusters': n_clusters, 'linkage': linkage}
                
                # Perform clustering
                cluster_labels, clustering_info = perform_clustering(
                    features_df, algorithm, best_params
                )
                
                # Create cluster summary
                summary_df = create_cluster_summary(defects_df, cluster_labels, list(feature_config.keys()))
                
                # Store results in session state
                st.session_state.clustering_results = {
                    'cluster_labels': cluster_labels,
                    'clustering_info': clustering_info,
                    'summary_df': summary_df,
                    'features_df': features_df,
                    'feature_info': feature_info,
                    'feature_config': feature_config,
                    'algorithm': algorithm,
                    'best_params': best_params,
                    'defects_df': defects_df.copy()
                }
                
                st.success("✅ Clustering analysis completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during clustering analysis: {str(e)}")
                st.info("Try adjusting the feature selection or algorithm parameters.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results if available
    if hasattr(st.session_state, 'clustering_results'):
        results = st.session_state.clustering_results
        
        # Results summary section
        st.markdown('<div class="card-container" style="margin-top:30px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Clustering Results</div>", unsafe_allow_html=True)
        
        # Display key metrics
        clustering_info = results['clustering_info']
        summary_df = results['summary_df']
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(
                custom_metric("Clusters Found", clustering_info['n_clusters_found']),
                unsafe_allow_html=True
            )
        
        with metric_col2:
            noise_points = clustering_info.get('n_noise_points', 0)
            st.markdown(
                custom_metric("Noise Points", f"{noise_points}"),
                unsafe_allow_html=True
            )
        
        with metric_col3:
            silhouette = clustering_info.get('silhouette_score', 0)
            if silhouette > 0:
                st.markdown(
                    custom_metric("Silhouette Score", f"{silhouette:.3f}"),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    custom_metric("Silhouette Score", "N/A"),
                    unsafe_allow_html=True
                )
        
        with metric_col4:
            algorithm_name = clustering_info['algorithm'].upper()
            st.markdown(
                custom_metric("Algorithm", algorithm_name),
                unsafe_allow_html=True
            )
        
        # Display cluster summary table
        if not summary_df.empty:
            st.markdown("#### Cluster Summary")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            # Format summary for display
            display_summary = summary_df.copy()
            
            # Round numeric columns
            numeric_cols = ['avg_depth', 'avg_length', 'avg_width', 'percentage', 'type_percentage']
            for col in numeric_cols:
                if col in display_summary.columns:
                    display_summary[col] = display_summary[col].round(2)
            
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization tabs
        st.markdown('<div class="card-container" style="margin-top:30px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎨 Cluster Visualizations</div>", unsafe_allow_html=True)
        
        viz_tabs = st.tabs([
            "2D Cluster View", 
            "Pipeline View", 
            "Cluster Characteristics",
            "Parameter Optimization"
        ])
        
        with viz_tabs[0]:
            st.markdown("#### 2D Cluster Visualization")
            st.info("This plot shows clusters in a reduced 2D space. If more than 2 features were used, PCA is applied for dimensionality reduction.")
            
            try:
                fig_2d = create_cluster_visualization_2d(
                    results['defects_df'], 
                    results['cluster_labels'], 
                    results['feature_config']
                )
                st.plotly_chart(fig_2d, use_container_width=True, config={'displayModeBar': True})
            except Exception as e:
                st.error(f"Error creating 2D visualization: {str(e)}")
        
        with viz_tabs[1]:
            st.markdown("#### Pipeline Cluster View")
            st.info("This shows clusters mapped onto the pipeline, with each cluster color-coded by location and clock position.")
            
            try:
                fig_pipeline = create_cluster_pipeline_visualization(
                    results['defects_df'], 
                    results['cluster_labels']
                )
                st.plotly_chart(fig_pipeline, use_container_width=True, config={'displayModeBar': True})
            except Exception as e:
                st.error(f"Error creating pipeline visualization: {str(e)}")
        
        with viz_tabs[2]:
            st.markdown("#### Cluster Characteristics")
            st.info("This chart compares the average characteristics of each cluster.")
            
            try:
                fig_chars = create_cluster_characteristics_plot(results['summary_df'])
                st.plotly_chart(fig_chars, use_container_width=True, config={'displayModeBar': True})
            except Exception as e:
                st.error(f"Error creating characteristics plot: {str(e)}")
        
        with viz_tabs[3]:
            st.markdown("#### Parameter Optimization Results")
            
            if hasattr(st.session_state, 'clustering_optimization'):
                opt_results = st.session_state.clustering_optimization
                st.markdown("Optimization explored different parameter combinations to find the best clustering:")
                st.dataframe(opt_results, use_container_width=True)
                
                # Plot optimization results
                if 'silhouette_score' in opt_results.columns:
                    import plotly.express as px
                    
                    if algorithm == 'kmeans':
                        fig_opt = px.line(
                            opt_results, x='n_clusters', y='silhouette_score',
                            title='Silhouette Score vs Number of Clusters',
                            markers=True
                        )
                        fig_opt.update_layout(
                            xaxis_title='Number of Clusters',
                            yaxis_title='Silhouette Score'
                        )
                        st.plotly_chart(fig_opt, use_container_width=True)
                        
                    elif algorithm == 'dbscan':
                        # Create 3D surface plot for DBSCAN optimization
                        fig_opt = px.scatter_3d(
                            opt_results, x='eps', y='min_samples', z='silhouette_score',
                            color='silhouette_score', 
                            title='DBSCAN Parameter Optimization',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_opt, use_container_width=True)
                        
            else:
                st.info("Parameter optimization results not available. Enable 'Auto-optimize parameters' to see optimization analysis.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export section
        st.markdown('<div class="card-container" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Export Results</div>", unsafe_allow_html=True)
        
        # Create enhanced dataset with cluster labels
        enhanced_defects = results['defects_df'].copy()
        enhanced_defects['cluster_id'] = results['cluster_labels']
        enhanced_defects['cluster_size'] = enhanced_defects['cluster_id'].map(
            results['summary_df'].set_index('cluster_id')['size']
        )
        
        # Create download link
        csv = enhanced_defects.to_csv(index=False)
        import base64
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="defects_with_clusters_{selected_year}.csv" class="custom-button" style="display:inline-block;text-decoration:none;margin-top:10px;font-size:0.9em;padding:8px 15px;background-color:#27AE60;color:white;border-radius:5px;">📊 Download Enhanced CSV with Cluster Labels</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.info(f"""
            **Export Information:**
            - Original defects: {len(results['defects_df'])} records
            - Features used: {', '.join([k.replace('include_', '') for k, v in results['feature_config'].items() if v])}
            - Algorithm: {results['algorithm'].upper()}
            - New columns added: cluster_id, cluster_size
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Help section
    with st.expander("📖 Clustering Analysis Guide", expanded=False):
        st.markdown("""
        ### Understanding Clustering Analysis
        
        **What is clustering?**
        Clustering groups similar defects together based on their characteristics. This helps identify:
        - Areas with similar defect patterns
        - Defect families with common characteristics
        - Anomalous defects that don't fit typical patterns
        
        **Algorithm Comparison:**
        
        **K-Means:**
        - Best for: Spherical clusters, known number of groups
        - Requires: Specifying number of clusters
        - Good for: General pattern identification
        
        **DBSCAN:**
        - Best for: Arbitrary cluster shapes, automatic cluster detection
        - Handles: Noise points and outliers well
        - Good for: Finding defect hotspots and anomalies
        
        **Hierarchical:**
        - Best for: Understanding cluster relationships
        - Creates: Tree structure of clusters
        - Good for: Multi-level analysis (joints → sections → pipeline)
        
        **Feature Selection Tips:**
        
        - **Spatial clustering**: Use location + clock position
        - **Severity clustering**: Use depth + length + width
        - **Type-based clustering**: Include defect type
        - **Comprehensive analysis**: Use all available features
        
        **Interpreting Results:**
        
        - **Silhouette Score**: 0.7+ excellent, 0.5+ good, <0.3 poor clustering
        - **Cluster size**: Very small clusters may indicate outliers
        - **Noise points**: Defects that don't fit any cluster pattern
        
        **Practical Applications:**
        
        1. **Inspection Planning**: Focus on cluster hotspots
        2. **Maintenance Scheduling**: Prioritize clusters with severe defects
        3. **Root Cause Analysis**: Investigate why certain areas cluster
        4. **Growth Monitoring**: Track how clusters evolve over time
        """)