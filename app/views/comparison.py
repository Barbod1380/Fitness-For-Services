"""
Multi-year comparison view for the Pipeline Analysis application.
"""

import pandas as pd
import streamlit as st
import datetime
from visualization.comparison_viz import *
from app.services.state_manager import *
from core.multi_year_analysis import compare_defects
from visualization.prediction_viz import (
    create_failure_timeline_histogram,
    create_survival_curve,
    create_erf_evolution_plot
)


def render_enhanced_clustering_analysis(earlier_data, later_data):
    """Enhanced clustering analysis for multi-year comparison"""
    
    st.markdown("---")
    st.markdown("### 🔬 Enhanced Clustering Analysis")
    
    if st.button("🚀 Analyze Defect Clustering", key="clustering_analysis"):
        
        with st.spinner("Performing enhanced clustering analysis..."):
            try:
                # Get data
                defects_df = later_data['defects_df']
                joints_df = later_data['joints_df']
                pipe_diameter_mm = later_data['pipe_diameter'] * 1000
                
                # Import and use new clustering
                from core.enhanced_ffs_clustering import enhance_existing_assessment
                
                combined_df, clusters = enhance_existing_assessment(
                    defects_df, joints_df, pipe_diameter_mm, "RSTRENG"
                )
                
                # Display results
                st.success(f"✅ Found {len(clusters)} clusters with enhanced analysis")
                
                # Show cluster details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters Found", len(clusters))
                with col2:
                    defects_clustered = sum(len(c.defect_indices) for c in clusters)
                    st.metric("Defects Clustered", defects_clustered)
                with col3:
                    max_stress = max([c.stress_concentration_factor for c in clusters], default=1.0)
                    st.metric("Max Stress Factor", f"{max_stress:.2f}x")
                
                # Store results for future use
                st.session_state.enhanced_clusters = clusters
                st.session_state.combined_defects = combined_df
                
                st.info("💡 **Ready for simulation**: Clustering results stored for 15-year failure prediction")
                
            except Exception as e:
                st.error(f"Error in clustering analysis: {str(e)}")


def test_new_clustering():
    """Test that your new clustering files work"""
    
    st.markdown("## 🧪 Test New Clustering System")
    
    if st.button("Test Clustering Integration"):
        try:
            # Test 1: Check imports work
            from core.standards_compliant_clustering import create_standards_compliant_clusterer
            from core.enhanced_ffs_clustering import enhance_existing_assessment  
            from core.failure_aware_clustering import integrate_failure_aware_clustering
            st.success("✅ All imports successful!")
            
            # Test 2: Check if we have data
            datasets = st.session_state.get('datasets', {})
            if not datasets:
                st.warning("⚠️ No datasets available. Upload data first to test fully.")
                return
            
            # Test 3: Try basic clustering
            test_year = list(datasets.keys())[0]
            defects_df = datasets[test_year]['defects_df']
            joints_df = datasets[test_year]['joints_df']
            pipe_diameter_mm = datasets[test_year].get('pipe_diameter', 1.0) * 1000
            
            # Test clustering
            clusterer = create_standards_compliant_clusterer(
                standard_name="RSTRENG",
                pipe_diameter_mm=pipe_diameter_mm
            )
            
            clusters = clusterer.find_interacting_defects(defects_df, joints_df)
            
            st.success(f"✅ Clustering works! Found {len(clusters)} clusters")
            st.info("🎉 Integration successful! Your new clustering system is ready to use.")
            
        except ImportError as e:
            st.error(f"❌ Import error: {e}")
            st.error("Check that all 4 files are in the correct directories")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.error("Check your data structure and file contents")


def _perform_advanced_comparison_analysis(datasets, earlier_year, later_year, distance_tolerance, clock_tolerance):
    """
    Perform comprehensive multi-year comparison analysis with automatic growth correction.
    
    FIXED: Now accepts tolerance parameters directly from user input
    """
    st.success(f"✅ Analysis initialized for {earlier_year} vs {later_year}")
    
    # Store user parameters in session state for future use
    analysis_params = {
        'distance_tolerance': distance_tolerance,
        'clock_tolerance': clock_tolerance,
        'analysis_timestamp': pd.Timestamp.now()
    }
    update_state('analysis_parameters', analysis_params)
    
    with st.spinner("🔄 Processing advanced multi-year comparison..."):
        try:
            # Step 1: Basic defect matching
            progress_bar = st.progress(0)
            st.write("Step 1/4: Matching defects between years...")
            
            earlier_data = datasets[earlier_year]
            later_data = datasets[later_year]
            
            comparison_results = compare_defects(
                earlier_data['defects_df'], later_data['defects_df'],
                earlier_data['joints_df'], later_data['joints_df'],
                earlier_year, later_year,
                distance_tolerance,  # User input
                clock_tolerance      # User input
            )
            progress_bar.progress(25)
            
            # Step 2: Automatic Growth Correction
            st.write("Step 2/4: Correcting unrealistic growth rates...")
            correction_results = None
            
            if (comparison_results.get('has_depth_data', False) and 
                not comparison_results['matches_df'].empty and 
                comparison_results.get('calculate_growth', False)):
                
                # Apply our correction methodology
                corrected_df, correction_info = _apply_growth_correction(
                    comparison_results['matches_df'],
                    earlier_data['joints_df'],
                    analysis_params  # Pass the user parameters
                )
                
                # Update comparison results with corrected data
                comparison_results['matches_df'] = corrected_df
                comparison_results['correction_info'] = correction_info
                correction_results = correction_info
                
            progress_bar.progress(50)
            
            # Step 3: Advanced analysis (clustering, etc.)
            st.write("Step 3/4: Advanced analysis...")
            # Future: Add clustering analysis here
            progress_bar.progress(75)
            
            # Step 4: Store results
            st.write("Step 4/4: Finalizing results...")
            update_state('comparison_results', comparison_results, validate=False)
            update_state('correction_results', correction_results, validate=False)
            update_state('comparison_years', (earlier_year, later_year), validate=False)
            
            progress_bar.progress(100)
            
            # Show correction summary
            if correction_results:
                _display_correction_summary(correction_results)
            else:
                st.success("✅ Analysis complete - no growth corrections needed")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 **Troubleshooting Tips:**\n- Ensure both datasets have required columns\n- Check that defects have valid location and depth data")
            return
        
        st.rerun()



def _apply_growth_correction(matches_df, joints_df, params):
    """
    Apply automatic growth correction based on engineering methodology.
    
    Returns:
    - corrected_df: DataFrame with corrected growth rates
    - correction_info: Dictionary with correction statistics and details
    """
    from analysis.growth_analysis import correct_negative_growth_rates
    
    # Create wall thickness lookup for 15% threshold calculation
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    corrected_df = matches_df.copy()
    correction_info = {
        'total_defects': len(matches_df),
        'flagged_for_review': 0,
        'auto_corrected': 0,
        'valid_growth': 0,
        'flagged_defects': [],
        'corrected_defects': [],
        'correction_method': 'weighted_nearby_5_joints'
    }
    
    # Step 1: Flag defects exceeding 15% wall thickness measurement uncertainty
    flagged_indices = []
    
    for idx, row in corrected_df.iterrows():
        if pd.isna(row.get('joint number')):
            continue
            
        joint_wt = wt_lookup.get(row['joint number'], 10.0)  # Default 10mm if missing
        uncertainty_threshold = joint_wt * 0.15  # 15% of wall thickness
        
        # Check if depth change exceeds measurement uncertainty
        depth_change_mm = abs(row.get('depth_change_mm', 0))
        
        if depth_change_mm > uncertainty_threshold and row.get('is_negative_growth', False):
            # Flag for review and set to zero growth
            corrected_df.loc[idx, 'growth_rate_pct_per_year'] = 0.0
            corrected_df.loc[idx, 'growth_rate_mm_per_year'] = 0.0
            corrected_df.loc[idx, 'is_negative_growth'] = False
            corrected_df.loc[idx, 'correction_applied'] = 'flagged_zero_growth'
            corrected_df.loc[idx, 'correction_reason'] = f'Exceeds 15% WT uncertainty ({uncertainty_threshold:.2f}mm)'
            
            flagged_indices.append(idx)
            correction_info['flagged_defects'].append({
                'index': idx,
                'location': row.get('log_dist', 0),
                'joint': row.get('joint number', 0),
                'original_rate': row.get('growth_rate_mm_per_year', 0),
                'threshold': uncertainty_threshold
            })
    
    correction_info['flagged_for_review'] = len(flagged_indices)
    
    # Step 2: Apply weighted correction for remaining negative growth using existing function
    # but with our specific parameters
    if corrected_df['is_negative_growth'].any():
        corrected_df_final, correction_details = correct_negative_growth_rates(
            corrected_df,
            k=3,  # Use 3 nearest neighbors
            joint_tolerance=5  # Within ±5 joints as discussed
        )
        
        # Track auto-corrected defects
        if correction_details.get('success', False):
            correction_info['auto_corrected'] = correction_details.get('corrected_count', 0)
            correction_info['correction_details'] = correction_details
        
        corrected_df = corrected_df_final
    
    # Step 3: Count valid growth defects
    correction_info['valid_growth'] = len(corrected_df[~corrected_df.get('is_negative_growth', True)])
    
    # Add correction metadata to dataframe
    corrected_df['correction_applied'] = corrected_df.get('correction_applied', 'none')
    
    return corrected_df, correction_info


def _display_correction_summary(correction_info):
    """Display automatic correction summary with quick stats."""
    
    st.markdown("---")
    st.subheader("🔧 Growth Correction Summary")
    
    # Summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Defects", 
            correction_info['total_defects']
        )
    
    with col2:
        st.metric(
            "🔴 Flagged for Review", 
            correction_info['flagged_for_review'],
            help="Defects exceeding 15% wall thickness measurement uncertainty"
        )
    
    with col3:
        st.metric(
            "🟡 Auto-Corrected", 
            correction_info['auto_corrected'],
            help="Negative growth rates corrected using nearby similar defects"
        )
    
    with col4:
        st.metric(
            "🟢 Valid Growth", 
            correction_info['valid_growth'],
            help="Defects with realistic positive growth rates"
        )
    
    # Correction quality indicator
    total_corrections = correction_info['flagged_for_review'] + correction_info['auto_corrected']
    if total_corrections > 0:
        correction_rate = (total_corrections / correction_info['total_defects']) * 100
        
        if correction_rate <= 10:
            st.success(f"✅ **Good data quality**: {correction_rate:.1f}% of defects required correction")
        elif correction_rate <= 25:
            st.warning(f"⚠️ **Moderate data quality**: {correction_rate:.1f}% of defects required correction")
        else:
            st.error(f"❌ **Poor data quality**: {correction_rate:.1f}% of defects required correction - consider data review")
    else:
        st.success("✅ **Excellent data quality**: No corrections needed!")



def display_data_preview_and_results(earlier_data, later_data):
    """Display comprehensive analysis results with correction review."""
    
    # Show correction summary first (if corrections were made)
    correction_results = get_state('correction_results')
    if correction_results:
        _display_correction_summary(correction_results)
    
    st.markdown("---")
    st.subheader("📈 Analysis Results")
    
    # Create tabs for different analysis views
    tabs = st.tabs([
        "📊 Overview", 
        "📈 Growth Analysis",
        "🔧 Correction Review",  # New tab for detailed correction review
        "📋 Data Export"
    ])
    
    with tabs[0]:
        # Existing overview charts
        col1, col2 = st.columns(2)
        
        with col1:
            donut_chart = create_defect_status_donut(earlier_data, later_data)
            st.plotly_chart(donut_chart, use_container_width=True)
        
        with col2:
            bar_chart = create_new_defects_by_type_bar(later_data)
            st.plotly_chart(bar_chart, use_container_width=True)
    
    with tabs[1]:
        # Growth analysis visualizations (existing function)
        _render_growth_analysis_visualizations()
    
    with tabs[2]:
        # New correction review section
        _render_correction_review_section()
    
    with tabs[3]:
        # Data export capabilities
        _render_data_export_section()

    render_enhanced_clustering_analysis(earlier_data, later_data)


def _render_growth_analysis_visualizations():
    """Render comprehensive growth analysis visualizations."""
    comparison_results = get_state('comparison_results')
    
    if not comparison_results or comparison_results['matches_df'].empty:
        st.warning("No comparison results available for visualization.")
        return
    
    st.markdown("---")
    st.subheader("📈 Growth Rate Analysis")
    
    # Growth dimension selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        available_dimensions = []
        if comparison_results.get('has_depth_data', False):
            available_dimensions.append('depth')
        if comparison_results.get('has_length_data', False):
            available_dimensions.append('length') 
        if comparison_results.get('has_width_data', False):
            available_dimensions.append('width')
        
        if not available_dimensions:
            st.error("No dimensional growth data available.")
            return
            
        selected_dimension = st.selectbox(
            "**Select Dimension for Analysis**",
            options=available_dimensions,
            format_func=lambda x: x.title() + " Growth",
            key="growth_dimension_selector"
        )
    
    with col2:
        show_negative = st.checkbox(
            "Include Negative Growth",
            value=False,
            help="Show negative growth rates (may indicate measurement errors)"
        )
    
    # Create three columns for the histograms
    st.markdown("### Growth Rate Distribution")
    
    if len(available_dimensions) >= 3:
        # Show all three dimensions side by side
        col1, col2, col3 = st.columns(3)
        dimensions_cols = [col1, col2, col3]
    elif len(available_dimensions) == 2:
        col1, col2 = st.columns(2)
        dimensions_cols = [col1, col2]
    else:
        dimensions_cols = [st.container()]
    
    # Generate histogram for each available dimension
    for i, dimension in enumerate(available_dimensions):
        with dimensions_cols[i]:
            fig = create_growth_rate_histogram(comparison_results, dimension)
            st.plotly_chart(fig, use_container_width=True, key=f"growth_hist_{dimension}")
    
    # Detailed analysis for selected dimension
    st.markdown(f"### Detailed {selected_dimension.title()} Growth Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Growth rate statistics
        _render_growth_statistics_table(comparison_results, selected_dimension)
    
    with col2:
        # Scatter plot of growth vs location
        fig_scatter = create_negative_growth_plot(comparison_results, selected_dimension)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Multi-dimensional comparison if multiple dimensions available
    if len(available_dimensions) > 1:
        st.markdown("### Multi-Dimensional Growth Comparison")
        fig_multi = create_multi_dimensional_growth_plot(comparison_results)
        st.plotly_chart(fig_multi, use_container_width=True)


def _render_growth_statistics_table(comparison_results, dimension):
    """Render statistics table for selected dimension."""
    matches_df = comparison_results['matches_df']
    
    if dimension == 'depth':
        if comparison_results.get('has_wt_data', False):
            growth_col = 'growth_rate_mm_per_year'
            unit = 'mm/year'
        else:
            growth_col = 'growth_rate_pct_per_year' 
            unit = '%/year'
        negative_col = 'is_negative_growth'
    elif dimension == 'length':
        growth_col = 'length_growth_rate_mm_per_year'
        unit = 'mm/year'
        negative_col = 'is_negative_length_growth'
    else:  # width
        growth_col = 'width_growth_rate_mm_per_year'
        unit = 'mm/year'
        negative_col = 'is_negative_width_growth'
    
    if growth_col not in matches_df.columns:
        st.error(f"No {dimension} growth data available")
        return
    
    # Calculate statistics
    all_growth = matches_df[growth_col].dropna()
    positive_growth = matches_df[~matches_df[negative_col]][growth_col].dropna()
    negative_count = matches_df[negative_col].sum()

    # Create separate numeric and text columns to avoid PyArrow issues
    stats_data = []

    stats_data.append({'Metric': 'Total Defects', 'Count': len(matches_df), 'Unit': 'defects'})
    stats_data.append({'Metric': 'Positive Growth', 'Count': len(positive_growth), 'Unit': 'defects'})
    stats_data.append({'Metric': 'Negative Growth', 'Count': int(negative_count), 'Unit': 'defects'})

    if not all_growth.empty:
        stats_data.append({'Metric': 'Mean Growth (All)', 'Count': round(all_growth.mean(), 4), 'Unit': unit})
        stats_data.append({'Metric': 'Min Growth Rate', 'Count': round(all_growth.min(), 4), 'Unit': unit})

    if not positive_growth.empty:
        stats_data.append({'Metric': 'Mean Growth (Positive)', 'Count': round(positive_growth.mean(), 4), 'Unit': unit})
        stats_data.append({'Metric': 'Median Growth (Positive)', 'Count': round(positive_growth.median(), 4), 'Unit': unit})
        stats_data.append({'Metric': 'Max Growth Rate', 'Count': round(positive_growth.max(), 4), 'Unit': unit})
        stats_data.append({'Metric': 'Std Deviation', 'Count': round(positive_growth.std(), 4), 'Unit': unit})

    stats_df = pd.DataFrame(stats_data)

    # Format for display
    stats_df['Value'] = stats_df.apply(
        lambda row: f"{row['Count']} {row['Unit']}" if row['Unit'] != 'defects' 
        else str(int(row['Count']) if row['Metric'] in ['Total Defects', 'Positive Growth', 'Negative Growth'] 
                else row['Count']), 
        axis=1
    )

    # Display only Metric and Value columns
    display_df = stats_df[['Metric', 'Value']].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)



# New tab structure visualization and helper functions

# Add these imports to the top of comparison.py
import streamlit as st
import pandas as pd
from app.services.state_manager import get_state, update_state

def render_comparison_view():
    """
    UPDATED: Main function to render the multi-year comparison view with tabbed interface.
    """
    st.title("🔄 Multi-Year Comparison")
    st.markdown("**Compare defect growth patterns between inspection years**")
    
    # Check if we have enough datasets
    datasets = get_state('datasets', {})
    available_years = sorted(datasets.keys())
    
    if len(available_years) < 2:
        st.error("**⚠️ Insufficient Data for Multi-Year Analysis**")
        
        with st.container():
            st.info(f"""
            📋 **Requirements Check:**
            - **Current datasets:** {len(available_years)}
            - **Required:** 2 or more
            - **Status:** {'✅ Ready' if len(available_years) >= 2 else '❌ Need more data'}
            
            **Next Steps:**
            1. Upload additional inspection data from different years
            2. Ensure data contains matching defects with location information
            3. Return to this page to start the comparison
            """)
        
        if available_years:
            st.markdown("**📅 Currently loaded years:**")
            cols = st.columns(min(len(available_years), 4))
            for i, year in enumerate(available_years):
                with cols[i % 4]:
                    defect_count = len(datasets[year]['defects_df'])
                    st.metric(f"Year {year}", f"{defect_count} defects")
        return

    # NEW: Create main tabs for different analysis types
    main_tabs = st.tabs([
        "📈 Growth Analysis", 
        "🔧 Advanced Clustering",
        "📊 Results & Visualization", 
        "📥 Export Data"
    ])
    
    with main_tabs[0]:
        # Growth Rate Analysis Tab
        render_growth_analysis_tab(datasets, available_years)
    
    with main_tabs[1]:
        # Advanced Clustering Tab (NEW)
        render_clustering_analysis_tab(datasets, available_years)
    
    with main_tabs[2]:
        # Results and Visualization Tab
        render_results_visualization_tab(datasets)
    
    with main_tabs[3]:
        # Export Tab
        render_export_tab()


def render_growth_analysis_tab(datasets, available_years):
    """
    Render the growth analysis tab - FOCUSED ONLY ON GROWTH ANALYSIS
    """

    # Year Selection Section
    with st.container():
        st.markdown("#### 📅 Year Selection")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            earlier_year = st.selectbox(
                "**Baseline Year** (Earlier)",
                options=available_years[:-1],
                key="earlier_year_select",
                help="Select the baseline inspection year for comparison"
            )
        
        with col2:
            later_years = [year for year in available_years if year > earlier_year]
            if not later_years:
                st.error("No later years available")
                return
                
            later_year = st.selectbox(
                "**Comparison Year** (Later)", 
                options=later_years,
                key="later_year_select",
                help="Select the year to compare against baseline"
            )
        
        with col3:
            year_gap = later_year - earlier_year
            st.metric("Time Gap", f"{year_gap} years")
    
    # Dataset Overview Section
    with st.container():
        st.markdown("#### 📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{earlier_year} Dataset (Baseline)**")
            earlier_data = datasets[earlier_year]
            earlier_defects = len(earlier_data['defects_df'])
            earlier_joints = len(earlier_data['joints_df'])
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Defects", earlier_defects)
            with metrics_col2:
                st.metric("Joints", earlier_joints)
        
        with col2:
            st.markdown(f"**{later_year} Dataset (Comparison)**")
            later_data = datasets[later_year]
            later_defects = len(later_data['defects_df'])
            later_joints = len(later_data['joints_df'])
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Defects", later_defects, delta=later_defects - earlier_defects)
            with metrics_col2:
                st.metric("Joints", later_joints, delta=later_joints - earlier_joints)
    
    # Analysis Parameters Section
    with st.container():
        st.markdown("#### 🎯 Analysis Parameters")
        
        # Get previous parameters from session state or use defaults
        previous_params = get_state('analysis_parameters', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            distance_tolerance = st.number_input(
                "**Distance Tolerance (meters)**",
                min_value=0.01,
                max_value=1.0,
                value=previous_params.get('distance_tolerance', 0.1),
                step=0.01,
                key="distance_tolerance_input",
                help="Maximum distance difference to consider defects as the same location"
            )
        
        with col2:
            clock_tolerance = st.number_input(
                "**Clock Tolerance (minutes)**",
                min_value=1,
                max_value=120,
                value=previous_params.get('clock_tolerance', 20),
                step=5,
                key="clock_tolerance_input",
                help="Maximum clock position difference to consider defects at same position"
            )
        
        # Parameter confirmation
        with st.expander("🔍 Current Parameters", expanded=False):
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                st.metric("Distance Tolerance", f"{distance_tolerance} m")
            with param_col2:
                st.metric("Clock Tolerance", f"{clock_tolerance} min")
    
    # Analysis Execution
    st.markdown("#### 🔍 Run Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Growth Rate Analysis", type="primary", use_container_width=True):
            _perform_advanced_comparison_analysis(
                datasets, earlier_year, later_year, 
                distance_tolerance, clock_tolerance
            )
    
    # Show analysis status
    comparison_years = get_state('comparison_years')
    comparison_results = get_state('comparison_results')
    
    if comparison_years and comparison_results:
        if comparison_years == (earlier_year, later_year):
            st.success("✅ Growth analysis completed! Check other tabs for results and clustering.")
        else:
            st.warning(f"⚠️ Results available for {comparison_years[0]} vs {comparison_years[1]}. Run analysis again for current selection.")


def render_clustering_analysis_tab(datasets, available_years):
    """
    FIXED: Clustering tab with proper function calls
    """
    st.markdown("### 🔧 Advanced Clustering Analysis")
    st.markdown("Industry-standards compliant clustering with stress concentration and failure prediction.")
    
    # Check if growth analysis has been performed
    comparison_results = get_state('comparison_results')
    comparison_years = get_state('comparison_years')
    
    if not comparison_results or not comparison_years:
        st.warning("⚠️ **Growth Analysis Required**")
        st.info("""
        Please complete the Growth Analysis first:
        1. Go to the **Growth Analysis** tab
        2. Select your years and parameters  
        3. Run the growth rate analysis
        4. Return here for clustering analysis
        """)
        return
    
    # Show which years are being analyzed
    earlier_year, later_year = comparison_years
    st.info(f"🔍 **Clustering Analysis for**: {earlier_year} → {later_year}")
    
    # Get data
    earlier_data = datasets[earlier_year]
    later_data = datasets[later_year]
    
    # Create sub-tabs for clustering and prediction
    cluster_tabs = st.tabs([
        "🔧 Clustering Analysis",
        "🔮 Future Prediction"
    ])
    
    with cluster_tabs[0]:
        render_clustering_analysis_section(later_data)
    
    with cluster_tabs[1]:
        render_future_prediction_section(later_data, comparison_results)


def render_clustering_analysis_section(later_data):
    """
    Render the clustering analysis section (extracted from original code)
    """
    st.markdown("#### 🔧 Clustering Configuration")
    
    # Configuration section
    with st.expander("🔧 Advanced Clustering Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            clustering_standard = st.selectbox(
                "Industry Standard",
                options=["RSTRENG", "BS7910", "API579", "DNV"],
                index=0,
                help="Select industry standard for clustering methodology"
            )
        
        with col2:
            conservative_factor = st.slider(
                "Conservative Factor", 
                min_value=1.0, max_value=2.0, value=1.0, step=0.1,
                help="Additional conservatism factor (1.0 = standard, >1.0 = more conservative)"
            )
        
        with col3:
            erf_threshold = st.slider(
                "ERF Failure Threshold",
                min_value=0.90, max_value=1.00, value=0.99, step=0.01,
                help="ERF threshold for failure assessment"
            )
    
    # Analysis Execution
    st.markdown("#### 🚀 Run Clustering Analysis")
    
    if st.button("🔬 Perform Advanced Clustering Analysis", type="primary", use_container_width=True, key="clustering_analysis_main"):
        
        # Pre-analysis setup
        defects_df = later_data['defects_df']
        joints_df = later_data['joints_df']
        pipe_diameter_mm = later_data['pipe_diameter'] * 1000
        
        # Validation checks
        if defects_df.empty:
            st.error("❌ No defects found in the selected dataset")
            return
        
        if joints_df.empty:
            st.error("❌ No joints found in the selected dataset")
            return
        
        # Check required columns
        required_defect_cols = ['log dist. [m]', 'joint number', 'depth [%]', 'length [mm]', 'width [mm]']
        missing_cols = [col for col in required_defect_cols if col not in defects_df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns in defects data: {missing_cols}")
            return
        
        st.info(f"🔍 **Starting clustering analysis**: {len(defects_df)} defects, {len(joints_df)} joints")
        

        try:
            import time
            start_time = time.time()
            
            # More robust import handling
            try:
                from core.enhanced_ffs_clustering import enhance_existing_assessment
            except ImportError as e:
                st.error(f"❌ Import error: {e}")
                st.info("💡 Make sure all clustering modules are properly installed")
                return
            
            # Run with progress tracking
            combined_df, clusters = enhance_existing_assessment(
                defects_df, joints_df, pipe_diameter_mm, clustering_standard
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Better validation of results
            if clusters is None:
                st.warning("⚠️ No clusters were generated")
                return
            
            if not isinstance(clusters, list):
                st.error("❌ Invalid cluster results format")
                return
            
            # Store results with performance metadata
            st.session_state.enhanced_clusters = clusters
            st.session_state.combined_defects = combined_df
            st.session_state.clustering_config = {
                'standard': clustering_standard,
                'conservative_factor': conservative_factor,
                'erf_threshold': erf_threshold,
                'analysis_timestamp': pd.Timestamp.now(),
                'processing_time_seconds': processing_time,
                'defects_processed': len(defects_df)
            }
            
            # Performance summary
            st.success(f"✅ Clustering analysis completed in {processing_time:.2f} seconds!")
            
            if processing_time > 0:
                defects_per_second = len(defects_df) / processing_time
                st.caption(f"⚡ Performance: {defects_per_second:.1f} defects/second")
            
            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Clusters Found", len(clusters))
            with col2:
                defects_clustered = sum(len(c.defect_indices) for c in clusters)
                st.metric("Defects Clustered", defects_clustered)
            with col3:
                max_stress = max([c.stress_concentration_factor for c in clusters], default=1.0)
                st.metric("Max Stress Factor", f"{max_stress:.2f}x")
            with col4:
                total_defects = len(defects_df)
                clustering_rate = (defects_clustered / total_defects) * 100 if total_defects > 0 else 0
                st.metric("Clustering Rate", f"{clustering_rate:.1f}%")
            
        except AttributeError as e:
            st.error(f"❌ Method missing: {str(e)}")
            st.info("💡 **Solution**: Make sure all required methods are implemented in the clustering classes")
        except ImportError as e:
            st.error(f"❌ Import error: {str(e)}")
            st.info("💡 **Solution**: Check that all required modules are available")
        except Exception as e:
            st.error(f"❌ Clustering analysis failed: {str(e)}")
            st.info("💡 **Troubleshooting**: Ensure defects have valid location and dimension data")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write("**Dataset Info:**")
                st.write(f"- Defects shape: {defects_df.shape}")
                st.write(f"- Joints shape: {joints_df.shape}")
                st.write(f"- Defects columns: {defects_df.columns.tolist()}")
                st.write(f"- Joints columns: {joints_df.columns.tolist()}")
                
                st.write("**Sample Data:**")
                st.write("First defect:", defects_df.iloc[0].to_dict() if not defects_df.empty else "No data")
    
    # Display clustering results if available
    if hasattr(st.session_state, 'enhanced_clusters') and st.session_state.enhanced_clusters:
        display_clustering_results()


def render_future_prediction_section(later_data, comparison_results):
    """
    Future prediction section with time-forward simulation - UPDATED for defect-based failures
    """
    st.markdown("### 🔮 Future Defect Failure Prediction")  # CHANGED title
    st.markdown("Simulate defect growth and predict individual defect failures over time")  # CHANGED description
    
    # Check if clustering has been performed
    if not hasattr(st.session_state, 'enhanced_clusters'):
        st.warning("⚠️ **Clustering Analysis Required**")
        st.info("Please run the Clustering Analysis first to enable failure prediction.")
        return
    
    # Parameter input form
    with st.expander("⚙️ Simulation Parameters", expanded=True):
        
        col1, col2 = st.columns(2)
        
        with col1:
            assessment_method = st.selectbox(
                "**Assessment Method**",
                options=['B31G', 'Modified_B31G', 'RSTRENG'],
                index=2,  # Default to RSTRENG
                help="Method for calculating safe operating pressure"
            )
            
            max_operating_pressure = st.number_input(
                "**Maximum Allowed Operating Pressure (MPa)**",
                min_value=1.0,
                max_value=20.0,
                value=8.0,
                step=0.1,
                help="Maximum allowed operating pressure for the pipeline"
            )
            
            safety_factor = st.number_input(
                "**Safety Factor**",
                min_value=1.0,
                max_value=5.0,
                value=1.39,
                step=0.01,
                help="Design safety factor"
            )
            
            smys = st.number_input(
                "**Pipe Grade (SMYS) (MPa)**",
                min_value=200.0,
                max_value=800.0,
                value=415.0,
                step=5.0,
                help="Specified Minimum Yield Strength"
            )
        
        with col2:
            pipe_diameter = st.number_input(
                "**Pipe Diameter (mm)**",
                min_value=100.0,
                max_value=2000.0,
                value=later_data['pipe_diameter'] * 1000,  # Convert to mm
                step=1.0,
                help="Outside diameter of the pipeline"
            )
            
            simulation_years = st.number_input(
                "**Simulation Years**",
                min_value=1,
                max_value=50,
                value=15,
                step=1,
                help="Number of years to simulate into the future"
            )
            
            erf_threshold = st.number_input(
                "**ERF Failure Threshold**",
                min_value=0.5,
                max_value=1.0,
                value=0.90,
                step=0.01,
                help="ERF threshold above which defect fails"  # CHANGED
            )
            
            depth_threshold = st.number_input(
                "**Depth Failure Threshold (%)**",
                min_value=50.0,
                max_value=95.0,
                value=80.0,
                step=1.0,
                help="Depth threshold above which defect fails"  # CHANGED
            )
    
    # TEMPORARY: Simple test implementation
    if st.button("🚀 Run Defect Failure Prediction Simulation", type="primary", use_container_width=True, key="prediction_simulation"):
        
        # Validation
        if not hasattr(st.session_state, 'enhanced_clusters'):
            st.error("❌ No clustering results available. Run clustering analysis first.")
            return
        
        with st.spinner("🔮 Running defect failure prediction simulation..."):  # CHANGED text
            try:
                import time
                start_time = time.time()
                
                # Import simulation modules
                from core.failure_prediction_simulation import (
                    FailurePredictionSimulator, SimulationParams
                )
                
                # Create simulation parameters
                sim_params = SimulationParams(
                    assessment_method=assessment_method.lower().replace(' ', '_'),  # Convert to function format
                    max_operating_pressure=max_operating_pressure,
                    simulation_years=simulation_years,
                    erf_threshold=erf_threshold,
                    depth_threshold=depth_threshold
                )                

                # Initialize simulator
                simulator = FailurePredictionSimulator(sim_params)
                
                # Initialize with current data
                success = simulator.initialize_simulation(
                    defects_df=later_data['defects_df'],
                    joints_df=later_data['joints_df'],
                    growth_rates_df=comparison_results['matches_df'],
                    clusters=st.session_state.enhanced_clusters,
                    pipe_diameter=pipe_diameter / 1000,  # Convert mm to meters
                    smys=smys,
                    safety_factor=safety_factor
                )

                if not success:
                    st.error("❌ Failed to initialize simulation")
                    return
                
                # Run simulation
                results = simulator.run_simulation()
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Store results
                st.session_state.prediction_results = results
                
                st.success(f"✅ Simulation completed in {processing_time:.2f} seconds!")
                st.success(f"📊 Predicted {results['total_failures']} defect failures over {simulation_years} years")  # CHANGED
                
                display_prediction_results_simple()

            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write(f"Error details: {str(e)}")
                    st.write(f"Assessment method: {assessment_method}")
                    st.write(f"Available clusters: {len(st.session_state.enhanced_clusters) if hasattr(st.session_state, 'enhanced_clusters') else 0}")


def display_prediction_results_simple():
    """
    FIXED: Display simulation results with simple visualizations - handles pandas arrays properly.
    """
    
    if not hasattr(st.session_state, 'prediction_results'):
        return
    
    results = st.session_state.prediction_results
    
    st.markdown("---")
    st.markdown("### 📊 Simulation Results")
    
    # Summary metrics - FIXED: Safe access to nested dictionaries
    try:
        survival_stats = results.get('survival_statistics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_defects = survival_stats.get('total_defects', 0)
            st.metric("Total Defects", total_defects)
        
        with col2:
            failed_defects = survival_stats.get('failed_defects', 0)
            failure_rate = survival_stats.get('failure_rate', 0.0)
            st.metric(
                "Failed Defects", 
                failed_defects,
                f"{failure_rate:.1f}% of total"
            )

        with col3:
            surviving_defects = survival_stats.get('surviving_defects', 0)
            survival_rate = survival_stats.get('survival_rate', 100.0)
            st.metric(
                "Surviving Defects", 
                surviving_defects,
                f"{survival_rate:.1f}% survival"
            )
        
        with col4:
            # FIXED: Safe check for failure history
            failure_history = results.get('failure_history', [])
            if failure_history and len(failure_history) > 0:
                try:
                    first_failure_year = min(f.failure_year for f in failure_history)
                    st.metric("First Failure", f"Year {first_failure_year}")
                except (AttributeError, ValueError, TypeError):
                    st.metric("First Failure", "Error calculating")
            else:
                st.metric("First Failure", "None predicted")
        
    except Exception as e:
        st.error(f"Error displaying summary metrics: {str(e)}")
        return
    
    # Failure timeline chart - FIXED: Better error handling
    try:
        # Try to import and use the visualization
        from visualization.prediction_viz import create_failure_timeline_histogram

        fig = create_failure_timeline_histogram(results)
        st.plotly_chart(fig, use_container_width=True)   

    except Exception as e:
        st.error(f"Error creating timeline chart: {str(e)}")
    
    # Failure details - FIXED: Much safer handling
    try:
        failure_history = results.get('failure_history', [])
        
        # FIXED: Safe check for failure history existence
        has_failures = False
        if failure_history:
            if hasattr(failure_history, '__len__'):
                has_failures = len(failure_history) > 0
            else:
                # If it's not a normal list/array, try to convert
                try:
                    failure_list = list(failure_history)
                    has_failures = len(failure_list) > 0
                    failure_history = failure_list
                except:
                    has_failures = False
        
        if has_failures:
            with st.expander("📋 Defect Failure Details"):
                try:
                    failure_data = []
                    for failure in failure_history:
                        try:
                            # FIXED: Safe attribute access with getattr
                            failure_data.append({
                                'Defect ID': getattr(failure, 'defect_id', 'Unknown'),
                                'Joint Number': getattr(failure, 'joint_number', 'Unknown'),
                                'Location (m)': f"{getattr(failure, 'location_m', 0.0):.2f}",
                                'Failure Year': getattr(failure, 'failure_year', 'Unknown'),
                                'Failure Mode': getattr(failure, 'failure_mode', 'Unknown'),
                                'Final ERF': f"{getattr(failure, 'final_erf', 0.0):.3f}",
                                'Final Depth (%)': f"{getattr(failure, 'final_depth_pct', 0.0):.1f}%",
                                'Was Clustered': "Yes" if getattr(failure, 'was_clustered', False) else "No",
                                'Stress Factor': f"{getattr(failure, 'stress_concentration_factor', 1.0):.2f}x"
                            })
                        except Exception as e:
                            st.warning(f"Could not process failure record: {str(e)}")
                            continue
                    
                    if failure_data:
                        failure_df = pd.DataFrame(failure_data)
                        st.dataframe(failure_df, use_container_width=True)
                    else:
                        st.warning("No valid failure data could be processed")
                        
                except Exception as e:
                    st.error(f"Error processing failure details: {str(e)}")
        
    except Exception as e:
        st.error(f"Error in failure details section: {str(e)}")
    
    # Risk assessment - FIXED: Safe access to stats
    try:
        st.markdown("### 💡 Risk Assessment")
        
        survival_stats = results.get('survival_statistics', {})
        failure_rate = survival_stats.get('failure_rate', 0.0)
        
        # FIXED: Ensure failure_rate is a number, not an array
        if hasattr(failure_rate, '__iter__') and not isinstance(failure_rate, str):
            failure_rate = float(failure_rate[0]) if len(failure_rate) > 0 else 0.0
        else:
            failure_rate = float(failure_rate)
        
        if failure_rate > 20:
            st.error(f"🚨 **HIGH RISK**: {failure_rate:.1f}% defect failure rate predicted")
        elif failure_rate > 10:
            st.warning(f"⚠️ **MODERATE RISK**: {failure_rate:.1f}% defect failure rate predicted")
        elif failure_rate > 0:
            st.success(f"✅ **LOW RISK**: {failure_rate:.1f}% defect failure rate predicted")
        else:
            st.success("🎉 **EXCELLENT**: No defect failures predicted!")
            
    except Exception as e:
        st.error(f"Error in risk assessment: {str(e)}")


def render_results_visualization_tab(datasets):
    """
    NEW: Combined results and visualization tab
    """
    st.markdown("### 📊 Analysis Results & Visualization")
    
    comparison_results = get_state('comparison_results')
    comparison_years = get_state('comparison_years')
    
    if not comparison_results or not comparison_years:
        st.info("📊 No analysis results available. Please run Growth Analysis first.")
        return
    
    earlier_year, later_year = comparison_years
    earlier_data = datasets[earlier_year]
    later_data = datasets[later_year]
    
    # Use the existing results display function
    display_data_preview_and_results(earlier_data, later_data)


def display_clustering_results():
    """
    FIXED: Display detailed clustering results without styling errors
    """
    clusters = st.session_state.enhanced_clusters
    config = st.session_state.clustering_config
    
    st.markdown("#### 🔍 Clustering Results Details")
    
    # Configuration summary
    with st.expander("📋 Analysis Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Standard", config['standard'])
        with col2:
            st.metric("Conservative Factor", f"{config['conservative_factor']:.1f}x")
        with col3:
            st.metric("ERF Threshold", f"{config['erf_threshold']:.2f}")
        
        st.caption(f"Analysis performed: {config['analysis_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Cluster details table
    if clusters:
        st.markdown("#### 📋 Cluster Details")
        
        cluster_data = []
        for i, cluster in enumerate(clusters):
            # Format stress factor with emoji indicators instead of complex styling
            stress_factor = cluster.stress_concentration_factor
            if stress_factor >= 2.0:
                stress_display = f"{stress_factor:.2f}x 🔴"
            elif stress_factor >= 1.5:
                stress_display = f"{stress_factor:.2f}x 🟡"
            elif stress_factor > 1.0:
                stress_display = f"{stress_factor:.2f}x 🟢"
            else:
                stress_display = f"{stress_factor:.2f}x"
            
            cluster_data.append({
                'Cluster ID': f'Cluster {i+1}',
                'Defect Count': len(cluster.defect_indices),
                'Max Depth (%)': f"{cluster.max_depth_pct:.1f}%",
                'Combined Length (mm)': f"{cluster.combined_length_mm:.1f}mm",
                'Combined Width (mm)': f"{cluster.combined_width_mm:.1f}mm",
                'Stress Factor': stress_display, 
                'Center Location (m)': f"{cluster.center_location_m:.2f}m",
                'Interaction Type': cluster.interaction_type,
                'Standard Used': cluster.standard_used
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # Simple dataframe display without problematic styling
        st.dataframe(cluster_df, use_container_width=True)
        
        # Add simple legend
        st.markdown("**Stress Factor Guide:** 🟢 Low (1.0-1.5x) | 🟡 Moderate (1.5-2.0x) | 🔴 High (>2.0x)")
        
        # Download cluster results
        cluster_csv = cluster_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cluster Results",
            data=cluster_csv,
            file_name=f"clustering_results_{config['analysis_timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="clustering_results_download_button"
        )
        
        # Additional summary metrics
        st.markdown("#### 📊 Cluster Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stress_factors = [cluster.stress_concentration_factor for cluster in clusters]
            avg_stress = sum(stress_factors) / len(stress_factors)
            st.metric("Average Stress Factor", f"{avg_stress:.2f}x")
        
        with col2:
            cluster_sizes = [len(cluster.defect_indices) for cluster in clusters]
            avg_size = sum(cluster_sizes) / len(cluster_sizes)
            st.metric("Average Cluster Size", f"{avg_size:.1f} defects")
        
        with col3:
            total_defects_clustered = sum(cluster_sizes)
            st.metric("Total Defects Clustered", total_defects_clustered)
    
    else:
        st.info("No clusters found with current parameters.")
        st.markdown("""
        **Possible reasons:**
        - Defects are too far apart to interact
        - Conservative clustering parameters
        - Limited defect data
        
        **Try:**
        - Increasing the conservative factor
        - Using a different clustering standard
        - Checking defect location data quality
        """)
        

def render_export_tab():
    """
    FIXED: Enhanced export tab with unique keys for all download buttons
    """
    st.markdown("### 📥 Data Export & Documentation")
    
    comparison_results = get_state('comparison_results')
    correction_results = get_state('correction_results')
    
    if not comparison_results:
        st.info("📁 No analysis results available for export.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Growth Analysis Data")
        
        # Export matched defects - FIXED: Added unique key
        if not comparison_results['matches_df'].empty:
            matches_csv = comparison_results['matches_df'].to_csv(index=False)
            st.download_button(
                label="📈 Download Matched Defects (CSV)",
                data=matches_csv,
                file_name=f"matched_defects_{st.session_state.get('earlier_year_select')}_{st.session_state.get('later_year_select')}.csv",
                mime="text/csv",
                key="export_tab_matched_defects_download"  # UNIQUE KEY
            )
        
        # Export new defects - FIXED: Added unique key
        if not comparison_results['new_defects'].empty:
            new_defects_csv = comparison_results['new_defects'].to_csv(index=False)
            st.download_button(
                label="🆕 Download New Defects (CSV)",
                data=new_defects_csv,
                file_name=f"new_defects_{st.session_state.get('later_year_select')}.csv",
                mime="text/csv",
                key="export_tab_new_defects_download"  # UNIQUE KEY
            )
    
    with col2:
        st.markdown("#### 🔧 Clustering Data")
        
        # Export clustering results if available - FIXED: Added unique key
        if hasattr(st.session_state, 'enhanced_clusters') and st.session_state.enhanced_clusters:
            # Combined defects export
            if hasattr(st.session_state, 'combined_defects'):
                combined_csv = st.session_state.combined_defects.to_csv(index=False)
                st.download_button(
                    label="🔗 Download Combined Defects (CSV)",
                    data=combined_csv,
                    file_name=f"combined_defects_{st.session_state.clustering_config['analysis_timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="export_tab_combined_defects_download"  # UNIQUE KEY
                )
        else:
            st.info("🔧 Run clustering analysis to enable clustering data export.")
    
    # Documentation section
    st.markdown("#### 📋 Documentation & Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Growth correction report - FIXED: Added unique key
        if correction_results:
            report_content = _generate_correction_report(correction_results, comparison_results)
            st.download_button(
                label="📋 Download Correction Report",
                data=report_content,
                file_name=f"growth_correction_report_{st.session_state.get('earlier_year_select')}_{st.session_state.get('later_year_select')}.txt",
                mime="text/plain",
                key="export_tab_correction_report_download"  # UNIQUE KEY
            )
    
    with col2:
        # Methodology documentation - FIXED: Added unique key
        methodology_doc = _generate_methodology_documentation()
        st.download_button(
            label="📖 Download Methodology Documentation",
            data=methodology_doc,
            file_name="analysis_methodology.txt",
            mime="text/plain",
            key="export_tab_methodology_download"  # UNIQUE KEY
        )



def _render_correction_review_section():
    """Render detailed correction review and override capabilities."""
    
    correction_results = get_state('correction_results')
    comparison_results = get_state('comparison_results')
    
    if not correction_results or not comparison_results:
        st.info("No correction data available for review.")
        return
    
    st.markdown("### 🔍 Detailed Correction Review")
    
    # Correction methodology summary
    with st.expander("📋 Correction Methodology Applied", expanded=False):
        st.markdown("""
        **Automatic Correction Process:**
        1. **Measurement Uncertainty Check**: Defects with growth changes >15% of wall thickness flagged for review
        2. **Zero Growth Assignment**: Flagged defects set to zero growth (conservative approach)
        3. **Weighted Correction**: Remaining negative growth corrected using similar defects within ±5 joints
        4. **Similarity Weighting**: Based on joint distance, depth similarity, and defect size
        """)
        
        if 'correction_details' in correction_results:
            details = correction_results['correction_details']
            st.write(f"**Joint Search Tolerance**: ±{details.get('joint_tolerance_used', 5)} joints")
            st.write(f"**Correction Success Rate**: {details.get('corrected_count', 0)}/{details.get('total_negative', 0)} negative growth defects corrected")
    
    # Flagged defects table
    if correction_results['flagged_defects']:
        st.markdown("#### 🔴 Defects Flagged for Engineering Review")
        
        flagged_df = pd.DataFrame(correction_results['flagged_defects'])
        flagged_df['Original Rate (mm/year)'] = flagged_df['original_rate'].round(4)
        flagged_df['Uncertainty Threshold (mm)'] = flagged_df['threshold'].round(2)
        flagged_df['Location (m)'] = flagged_df['location'].round(2)
        flagged_df['Joint Number'] = flagged_df['joint'].astype(int)
        
        display_flagged = flagged_df[['Location (m)', 'Joint Number', 'Original Rate (mm/year)', 'Uncertainty Threshold (mm)']]
        st.dataframe(display_flagged, use_container_width=True, hide_index=True)
        
        st.warning("⚠️ **Engineering Action Required**: These defects exceed measurement uncertainty and should be reviewed by qualified personnel.")
    
    # Show corrected defects summary
    if correction_results['auto_corrected'] > 0:
        st.markdown("#### 🟡 Auto-Corrected Defects Summary")
        
        matches_df = comparison_results['matches_df']
        corrected_defects = matches_df[matches_df.get('is_corrected', False)]
        
        if not corrected_defects.empty:
            correction_stats = {
                'Count': len(corrected_defects),
                'Average Original Rate (mm/year)': corrected_defects['growth_rate_mm_per_year'].mean(),
                'Average Corrected Rate (mm/year)': corrected_defects.get('corrected_growth_rate_mm_per_year', corrected_defects['growth_rate_mm_per_year']).mean(),
                'Location Range (m)': f"{corrected_defects['log_dist'].min():.1f} - {corrected_defects['log_dist'].max():.1f}"
            }
            
            for key, value in correction_stats.items():
                if 'Rate' in key and isinstance(value, (int, float)):
                    st.write(f"**{key}**: {value:.4f}")
                else:
                    st.write(f"**{key}**: {value}")

def _render_data_export_section():
    """
    FIXED: Render data export capabilities with unique keys for download buttons
    """
    
    comparison_results = get_state('comparison_results')
    correction_results = get_state('correction_results')
    
    if not comparison_results:
        st.info("No analysis results available for export.")
        return
    
    st.markdown("### 📥 Data Export & Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Analysis Data")
        
        # Export matched defects with corrections - FIXED: Added unique key
        if not comparison_results['matches_df'].empty:
            matches_csv = comparison_results['matches_df'].to_csv(index=False)
            st.download_button(
                label="📊 Download Matched Defects (CSV)",
                data=matches_csv,
                file_name=f"matched_defects_corrected_{st.session_state.get('earlier_year_select')}_{st.session_state.get('later_year_select')}.csv",
                mime="text/csv",
                key="results_tab_matched_defects_download"  # UNIQUE KEY
            )
        
        # Export new defects - FIXED: Added unique key
        if not comparison_results['new_defects'].empty:
            new_defects_csv = comparison_results['new_defects'].to_csv(index=False)
            st.download_button(
                label="🆕 Download New Defects (CSV)",
                data=new_defects_csv,
                file_name=f"new_defects_{st.session_state.get('later_year_select')}.csv",
                mime="text/csv",
                key="results_tab_new_defects_download"  # UNIQUE KEY
            )
    
    with col2:
        st.markdown("#### Documentation")
        
        # Generate correction report - FIXED: Added unique key
        if correction_results:
            report_content = _generate_correction_report(correction_results, comparison_results)
            st.download_button(
                label="📋 Download Correction Report",
                data=report_content,
                file_name=f"growth_correction_report_{st.session_state.get('earlier_year_select')}_{st.session_state.get('later_year_select')}.txt",
                mime="text/plain",
                key="results_tab_correction_report_download"  # UNIQUE KEY
            )
        
        # Export methodology documentation - FIXED: Added unique key
        methodology_doc = _generate_methodology_documentation()
        st.download_button(
            label="📖 Download Methodology Documentation",
            data=methodology_doc,
            file_name="growth_correction_methodology.txt",
            mime="text/plain",
            key="results_tab_methodology_download"  # UNIQUE KEY
        )
        
def _generate_correction_report(correction_results, comparison_results):
    """Generate a detailed correction report for documentation."""
    
    from datetime import datetime
    
    report = f"""
PIPELINE GROWTH RATE CORRECTION REPORT
======================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Years: {st.session_state.get('earlier_year_select')} to {st.session_state.get('later_year_select')}

CORRECTION SUMMARY
------------------
Total Defects Analyzed: {correction_results['total_defects']}
Defects Flagged for Review: {correction_results['flagged_for_review']}
Defects Auto-Corrected: {correction_results['auto_corrected']}
Valid Growth Defects: {correction_results['valid_growth']}

METHODOLOGY APPLIED
-------------------
1. Measurement Uncertainty Threshold: 15% of wall thickness
2. Flagged Defect Treatment: Zero growth assignment
3. Negative Growth Correction: Weighted average from ±5 joints
4. Similarity Factors: Joint distance, depth, defect size

FLAGGED DEFECTS (Require Engineering Review)
--------------------------------------------
"""
    
    if correction_results['flagged_defects']:
        for defect in correction_results['flagged_defects']:
            report += f"Location: {defect['location']:.2f}m, Joint: {defect['joint']}, Original Rate: {defect['original_rate']:.4f} mm/year\n"
    else:
        report += "None\n"
    
    report += f"""

CORRECTION QUALITY
------------------
Correction Success: {'Yes' if correction_results.get('auto_corrected', 0) > 0 else 'No'}
Data Quality Assessment: {"Good" if (correction_results['flagged_for_review'] + correction_results['auto_corrected']) / correction_results['total_defects'] < 0.1 else "Requires Review"}

This report documents automatic corrections applied to pipeline defect growth rates
in accordance with industry best practices for measurement uncertainty handling.
"""
    
    return report

def _generate_methodology_documentation():
    """Generate methodology documentation for regulatory compliance."""
    
    return """
PIPELINE DEFECT GROWTH RATE CORRECTION METHODOLOGY
==================================================

PURPOSE
-------
This methodology addresses measurement uncertainty in inline inspection (ILI) 
data when calculating defect growth rates between inspection periods.

TECHNICAL BASIS
---------------
- API 579-1: Fitness-for-Service standard
- DNV-RP-F101: Pipeline integrity assessment guidelines  
- Industry measurement uncertainty: ±10-15% of wall thickness

CORRECTION ALGORITHM
--------------------
1. MEASUREMENT UNCERTAINTY CHECK
   - Calculate 15% of nominal wall thickness for each defect location
   - Flag defects where |growth change| > 15% wall thickness
   - Apply zero growth to flagged defects (conservative approach)

2. NEGATIVE GROWTH CORRECTION
   - Identify remaining defects with negative growth rates
   - Search for similar defects within ±5 joints of target location
   - Calculate weighted average based on:
     * Joint distance (closer = higher weight)
     * Depth similarity (similar depths corrode similarly)
     * Defect size (larger defects = more reliable measurements)

3. QUALITY ASSURANCE
   - Document all corrections applied
   - Flag cases requiring engineering review
   - Maintain audit trail for regulatory compliance

ENGINEERING JUSTIFICATION
--------------------------
Physical reality: Pipeline steel cannot heal itself, therefore negative
growth rates indicate measurement uncertainty rather than actual improvement.

Conservative approach: When in doubt, apply zero growth or use similar
defects from the local environment to estimate realistic growth rates.

Regulatory compliance: All corrections documented and available for
engineering review and regulatory audit.
"""
