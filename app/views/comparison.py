"""
Multi-year comparison view for the Pipeline Analysis application.
"""

import pandas as pd
import streamlit as st
from visualization.comparison_viz import *
from app.services.state_manager import *
from core.matching_defects import compare_defects
from analysis.growth_correction import correct_negative_growth_rates
from visualization.prediction_viz import create_failure_timeline_histogram
from core.failure_simulation import FailurePredictionSimulator, SimulationParams
from datetime import datetime


def _perform_advanced_comparison_analysis(datasets, earlier_year, later_year, distance_tolerance, clock_tolerance):
    """
    Perform comprehensive multi-year comparison analysis with automatic growth correction.
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
                    analysis_params  
                )

                if correction_info.get('auto_corrected', 0) > 0:
                    st.write(f"✅ Applied KNN corrections to {correction_info['auto_corrected']} defects")
                else:
                    st.write("ℹ️ No KNN corrections needed - all growth rates realistic")
                
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
        "🔧 Correction Review", 
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
        "🔮 Failure Prediction",  # Renamed and simplified
        "📊 Results & Visualization", 
        "📥 Export Data"
    ])
    
    with main_tabs[0]:
        render_growth_analysis_tab(datasets, available_years)
    
    with main_tabs[1]:
        # Direct failure prediction - no sub-tabs!
        render_failure_prediction_section(datasets, get_state('comparison_results'))    
    

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
                min_value=0.001,
                max_value=1.0,
                value=previous_params.get('distance_tolerance', 0.01),
                step=0.001,
                key="distance_tolerance_input",
                help="Maximum distance difference to consider defects as the same location",
                format="%.3f"
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


def render_failure_prediction_section(datasets, comparison_results):
    """Integrated failure prediction with dynamic clustering"""
    st.markdown("### 🔮 Failure Prediction with Dynamic Clustering")
    
    comparison_years = get_state('comparison_years')

    # Check prerequisites
    if not comparison_results or not comparison_years:
        st.warning("⚠️ Please complete Growth Analysis first")
        return
    
    earlier_year, later_year = comparison_years
    later_data = datasets[later_year]
    
    # Parameter input form
    with st.expander("⚙️ Simulation Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            assessment_method = st.selectbox(
                "Assessment Method",
                options=['B31G', 'Modified_B31G', 'RSTRENG'],
                index=2
            )
            
            max_operating_pressure = st.number_input(
                "Max Operating Pressure (MPa)",
                min_value=1.0, max_value=20.0, value=8.0, step=0.1
            )
            
            simulation_years = st.number_input(
                "Simulation Years",
                min_value=1, max_value=50, value=15, step=1
            )
        
        with col2:
            # Clustering configuration
            use_dynamic_clustering = st.checkbox(
                "Enable Dynamic Clustering",
                value=True,
                help="Re-evaluate defect interactions each year"
            )
            
            if use_dynamic_clustering:
                clustering_standard = st.selectbox(
                    "Clustering Standard",
                    options=["DNV", "ROSEN#15"],
                    index=0
                )
            else:
                clustering_standard = None
            
            safety_factor = st.number_input(
                "Safety Factor",
                min_value=1.0, max_value=5.0, value=1.39, step=0.01
            )
        
        with col3:
            erf_threshold = st.number_input(
                "ERF Failure Threshold",
                min_value=0.5, max_value=1.0, value=0.90, step=0.01
            )
            
            depth_threshold = st.number_input(
                "Depth Failure Threshold (%)",
                min_value=50.0, max_value=95.0, value=80.0, step=1.0
            )
            
            smys = st.number_input(
                "SMYS (MPa)",
                min_value=200.0, max_value=800.0, value=415.0, step=5.0
            )
    
    # Run simulation
    if st.button("🚀 Run Failure Prediction", type="primary", use_container_width=True):
        with st.spinner("Running failure prediction with dynamic clustering..."):
            # Create simulation parameters
            sim_params = SimulationParams(
                assessment_method=assessment_method.lower().replace('_', '_'),
                max_operating_pressure=max_operating_pressure,
                simulation_years=simulation_years,
                erf_threshold=erf_threshold,
                depth_threshold=depth_threshold
            )
            
            # Create clustering config
            clustering_config = {
                'enabled': use_dynamic_clustering,
                'standard': clustering_standard,
                'pipe_diameter_mm': later_data['pipe_diameter'] * 1000
            } if use_dynamic_clustering else None

            # Run integrated simulation
            results = run_integrated_simulation(
                sim_params,
                later_data,
                comparison_results,
                clustering_config,
                smys,
                safety_factor
            )
            
    display_prediction_results_simple()


def run_integrated_simulation(sim_params, data, growth_results, clustering_config, smys, safety_factor):
    
    """Run failure simulation with CORRECTED growth rates from KNN analysis. with optional dynamic clustering"""
    

    # Verify we're using corrected growth rates
    if 'matches_df' not in growth_results:
        raise ValueError("Growth results missing matches_df - cannot run simulation")
    
    matches_df = growth_results['matches_df']
    
    # VERIFICATION: Check if growth corrections were applied
    has_corrections = False
    correction_columns = ['is_corrected', 'correction_applied']
    
    for col in correction_columns:
        if col in matches_df.columns:
            corrected_count = matches_df[col].sum() if matches_df[col].dtype == bool else len(matches_df[matches_df[col] != 'none'])
            if corrected_count > 0:
                has_corrections = True
                print(f"✅ Using corrected growth rates: {corrected_count} defects corrected")
                break
    
    if not has_corrections:
        print("⚠️ Warning: No growth corrections detected - ensure KNN correction was applied")


    # Initialize simulator with clustering config
    simulator = FailurePredictionSimulator(sim_params, clustering_config)
    
    # Initialize with data
    success = simulator.initialize_simulation(
        defects_df=data['defects_df'],
        joints_df=data['joints_df'],
        growth_rates_df=growth_results['matches_df'],
        clusters=[],  # No initial clusters!
        pipe_diameter=data['pipe_diameter'],
        smys=smys,
        safety_factor=safety_factor,
        use_clustering=False  # Disable initial clustering
    )
    
    if not success:
        st.error("Failed to initialize simulation")
        return None
    
    # Run simulation
    results = simulator.run_simulation()
    st.session_state.prediction_results = results

    return results


def display_prediction_results_simple():
    """Display simulation results with joint AND defect statistics."""
    
    if not hasattr(st.session_state, 'prediction_results'):
        return
    
    results = st.session_state.prediction_results
    
    st.markdown("---")
    st.markdown("### 📊 Simulation Results")
    
    # Get both defect and joint statistics
    try:
        survival_stats = results.get('survival_statistics', {})
        joint_stats = results.get('joint_survival_statistics', {})
        
        # Display metrics in two rows: Defects and Joints
        
        # DEFECT STATISTICS (existing)
        st.markdown("#### 🔧 Defect-Level Results")
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
            failure_history = results.get('failure_history', [])
            if failure_history and len(failure_history) > 0:
                try:
                    first_failure_year = min(f.failure_year for f in failure_history)
                    st.metric("First Defect Failure", f"Year {first_failure_year}")
                except:
                    st.metric("First Defect Failure", "Error calculating")
            else:
                st.metric("First Defect Failure", "None predicted")
        
        # NEW: JOINT STATISTICS
        st.markdown("#### 🏗️ Joint-Level Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_joints = joint_stats.get('total_joints', 0)
            st.metric("Total Joints", total_joints)
        
        with col2:
            failed_joints = joint_stats.get('failed_joints', 0)
            joint_failure_rate = joint_stats.get('joint_failure_rate', 0.0)
            st.metric(
                "Failed Joints", 
                failed_joints,
                f"{joint_failure_rate:.1f}% of total"
            )

        with col3:
            surviving_joints = joint_stats.get('surviving_joints', 0)
            joint_survival_rate = joint_stats.get('joint_survival_rate', 100.0)
            st.metric(
                "Surviving Joints", 
                surviving_joints,
                f"{joint_survival_rate:.1f}% survival"
            )
        
        with col4:
            # Find first joint failure year
            joint_timeline = results.get('joint_failure_timeline', {})
            first_joint_failure = None
            for year in sorted(joint_timeline.keys()):
                if joint_timeline[year] > 0:
                    first_joint_failure = year
                    break
            
            if first_joint_failure is not None:
                st.metric("First Joint Failure", f"Year {first_joint_failure}")
            else:
                st.metric("First Joint Failure", "None predicted")
        
    except Exception as e:
        st.error(f"Error displaying summary metrics: {str(e)}")
        return
    
    # Enhanced timeline visualization (now includes joints)
    try:
        fig = create_failure_timeline_histogram(results)
        st.plotly_chart(fig, use_container_width=True)   

    except Exception as e:
        st.error(f"Error creating timeline chart: {str(e)}")
    
    # Risk assessment based on JOINT failures (more critical)
    try:
        st.markdown("### 💡 Risk Assessment")
        
        joint_stats = results.get('joint_survival_statistics', {})
        joint_failure_rate = joint_stats.get('joint_failure_rate', 0.0)
        
        # Ensure failure_rate is a number
        if hasattr(joint_failure_rate, '__iter__') and not isinstance(joint_failure_rate, str):
            joint_failure_rate = float(joint_failure_rate[0]) if len(joint_failure_rate) > 0 else 0.0
        else:
            joint_failure_rate = float(joint_failure_rate)
        
        # Risk assessment based on JOINT failures (more critical than defect failures)
        if joint_failure_rate > 15:
            st.error(f"🚨 **CRITICAL RISK**: {joint_failure_rate:.1f}% joint failure rate predicted")
            st.error("⚠️ **Immediate Action Required**: Multiple pipeline segments at risk")
        elif joint_failure_rate > 5:
            st.warning(f"⚠️ **HIGH RISK**: {joint_failure_rate:.1f}% joint failure rate predicted")
            st.warning("🔧 **Targeted Maintenance Required**: Focus on failing joints")
        elif joint_failure_rate > 0:
            st.info(f"ℹ️ **MODERATE RISK**: {joint_failure_rate:.1f}% joint failure rate predicted")
            st.info("👀 **Monitor Closely**: Some joints may require attention")
        else:
            st.success("🎉 **EXCELLENT**: No joint failures predicted!")
            
    except Exception as e:
        st.error(f"Error in risk assessment: {str(e)}")

    # Joint failure details (if any)
    joint_timeline = results.get('joint_failure_timeline', {})
    total_joint_failures = sum(joint_timeline.values()) if joint_timeline else 0
    
    if total_joint_failures > 0:
        with st.expander("🏗️ Joint Failure Timeline Details"):
            timeline_data = []
            for year in sorted(joint_timeline.keys()):
                if joint_timeline[year] > 0:
                    timeline_data.append({
                        'Year': year,
                        'Joints Failed': joint_timeline[year],
                        'Impact': 'High' if joint_timeline[year] > 2 else 'Moderate'
                    })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                st.dataframe(timeline_df, use_container_width=True, hide_index=True)
            else:
                st.info("No joint failures predicted during simulation period")


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


def render_export_tab():
    st.markdown("### 📥 Data Export & Documentation")
    
    comparison_results = get_state('comparison_results')
    correction_results = get_state('correction_results')
    
    if not comparison_results:
        st.info("📁 No analysis results available for export.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Growth Analysis Data")
        
        # Export matched defects - Added unique key
        if not comparison_results['matches_df'].empty:
            matches_csv = comparison_results['matches_df'].to_csv(index=False)
            st.download_button(
                label="📈 Download Matched Defects (CSV)",
                data=matches_csv,
                file_name=f"matched_defects_{st.session_state.get('earlier_year_select')}_{st.session_state.get('later_year_select')}.csv",
                mime="text/csv",
                key="export_tab_matched_defects_download"  # UNIQUE KEY
            )
        
        # Export new defects - Added unique key
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
        
        # Export clustering results if available - Added unique key
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
        # Growth correction report - Added unique key
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
        # Methodology documentation -  Added unique key
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
    Render data export capabilities with unique keys for download buttons
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
        
        # Export matched defects with corrections - Added unique key
        if not comparison_results['matches_df'].empty:
            matches_csv = comparison_results['matches_df'].to_csv(index=False)
            st.download_button(
                label="📊 Download Matched Defects (CSV)",
                data=matches_csv,
                file_name=f"matched_defects_corrected_{st.session_state.get('earlier_year_select')}_{st.session_state.get('later_year_select')}.csv",
                mime="text/csv",
                key="results_tab_matched_defects_download"  # UNIQUE KEY
            )
        
        # Export new defects - Added unique key
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
        
        # Generate correction report - Added unique key
        if correction_results:
            report_content = _generate_correction_report(correction_results, comparison_results)
            st.download_button(
                label="📋 Download Correction Report",
                data=report_content,
                file_name=f"growth_correction_report_{st.session_state.get('earlier_year_select')}_{st.session_state.get('later_year_select')}.txt",
                mime="text/plain",
                key="results_tab_correction_report_download"  # UNIQUE KEY
            )
        
        # Export methodology documentation - Added unique key
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
