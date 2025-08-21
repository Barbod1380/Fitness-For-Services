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
    Performs a comprehensive multi-year comparison, including defect matching and growth rate correction.
    """
    st.success(f"Analysis initialized for {earlier_year} vs. {later_year}.")
    analysis_params = {
        'distance_tolerance': distance_tolerance,
        'clock_tolerance': clock_tolerance,
        'analysis_timestamp': pd.Timestamp.now()
    }
    update_state('analysis_parameters', analysis_params)

    with st.spinner("Processing multi-year comparison..."):
        try:
            progress_bar = st.progress(0, "Step 1/4: Matching defects...")
            earlier_data, later_data = datasets[earlier_year], datasets[later_year]
            
            comparison_results = compare_defects(
                earlier_data['defects_df'], later_data['defects_df'],
                earlier_data['joints_df'], later_data['joints_df'],
                earlier_year, later_year,
                distance_tolerance, clock_tolerance
            )
            progress_bar.progress(25, "Step 2/4: Correcting growth rates...")

            correction_results = None
            if comparison_results.get('calculate_growth', False) and not comparison_results['matches_df'].empty:
                corrected_df, correction_info = _apply_growth_correction(
                    comparison_results['matches_df'],
                    earlier_data['joints_df'],
                    analysis_params
                )
                if correction_info.get('auto_corrected', 0) > 0:
                    st.write(f"Applied KNN corrections to {correction_info['auto_corrected']} defects.")
                else:
                    st.write("No KNN corrections were needed; all growth rates are realistic.")
                
                comparison_results.update({
                    'matches_df': corrected_df,
                    'correction_info': correction_info
                })
                correction_results = correction_info
            
            progress_bar.progress(50, "Step 3/4: Performing advanced analysis...")
            # Placeholder for future clustering or other advanced analyses
            
            progress_bar.progress(75, "Step 4/4: Finalizing results...")
            update_state('comparison_results', comparison_results, validate=False)
            update_state('correction_results', correction_results, validate=False)
            update_state('comparison_years', (earlier_year, later_year), validate=False)
            
            progress_bar.progress(100, "Analysis complete.")
            
            if correction_results:
                _display_correction_summary(correction_results)
            else:
                st.success("Analysis completed successfully; no growth corrections were necessary.")
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Troubleshooting: Ensure both datasets have the required columns and valid defect data.")
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
    """
    Displays a summary of the growth correction analysis, including metrics and data quality assessment.
    """
    with st.container(border=True):
        st.subheader("Growth Correction Summary")

        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Defects", correction_info['total_defects'])
        col2.metric(
            "Flagged for Review",
            correction_info['flagged_for_review'],
            help="Defects with growth changes exceeding 15% of wall thickness."
        )
        col3.metric(
            "Auto-Corrected",
            correction_info['auto_corrected'],
            help="Defects with negative growth rates corrected via KNN."
        )
        col4.metric(
            "Valid Growth",
            correction_info['valid_growth'],
            help="Defects with realistic, positive growth rates."
        )

        # Assess and display data quality
        total_corrections = correction_info['flagged_for_review'] + correction_info['auto_corrected']
        if total_corrections > 0:
            correction_percentage = (total_corrections / correction_info['total_defects']) * 100
            if correction_percentage <= 10:
                st.success(f"Good Data Quality: {correction_percentage:.1f}% of defects required correction.")
            elif correction_percentage <= 25:
                st.warning(f"Moderate Data Quality: {correction_percentage:.1f}% of defects required correction.")
            else:
                st.error(f"Poor Data Quality: {correction_percentage:.1f}% of defects required correction. A data review is recommended.")
        else:
            st.success("Excellent Data Quality: No growth corrections were needed.")





def _render_growth_analysis_visualizations():
    """
    Renders a set of visualizations for growth analysis, including histograms and scatter plots.
    """
    comparison_results = get_state('comparison_results')
    if not comparison_results or comparison_results['matches_df'].empty:
        st.warning("No growth analysis results available to visualize.")
        return

    st.subheader("Growth Rate Analysis")
    
    # Selector for choosing the growth dimension to analyze
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
    Main function to render the multi-year comparison view.
    """
    st.title("Multi-Year Comparison")
    st.markdown("**Compare defect growth patterns between inspection years**")

    datasets = get_state('datasets', {})
    available_years = sorted(datasets.keys())

    if len(available_years) < 2:
        st.error("Insufficient Data for Multi-Year Analysis")
        with st.container(border=True):
            st.info(f"""
            **Requirements:**
            - **Uploaded Datasets:** {len(available_years)}
            - **Needed for Comparison:** 2 or more
            """)
            st.warning("Please upload at least two datasets from different years to proceed.")
        return

    # Main navigation tabs
    main_tabs = st.tabs([
        "Growth Analysis",
        "Failure Prediction",
        "Results"
    ])

    with main_tabs[0]:
        render_growth_analysis_tab(datasets, available_years)

    with main_tabs[1]:
        # Direct failure prediction - no sub-tabs!
        render_failure_prediction_section(datasets, get_state('comparison_results'))

    with main_tabs[2]:
        # Results and Visualization Tab
        render_results_visualization_tab(datasets)


def render_growth_analysis_tab(datasets, available_years):
    """
    Renders the Growth Analysis tab, including year selection, parameters, and execution.
    """
    st.header("Growth Analysis Setup")
    st.markdown("Configure and run the defect growth analysis between two inspection years.")

    # Section for selecting years
    with st.container(border=True):
        st.subheader("Year Selection")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            earlier_year = st.selectbox(
                "Baseline Year (Earlier)",
                options=available_years[:-1],
                key="earlier_year_select",
                help="The starting year for the growth comparison."
            )
        with col2:
            later_years = [y for y in available_years if y > earlier_year]
            if not later_years:
                st.error("No subsequent years available for comparison.")
                return
            later_year = st.selectbox(
                "Comparison Year (Later)",
                options=later_years,
                key="later_year_select",
                help="The year to compare against the baseline."
            )
        with col3:
            st.metric("Time Gap", f"{later_year - earlier_year} years")

    # Section for analysis parameters
    with st.container(border=True):
        st.subheader("Analysis Parameters")
        previous_params = get_state('analysis_parameters', {})
        col1, col2 = st.columns(2)
        with col1:
            distance_tolerance = st.number_input(
                "Distance Tolerance (m)",
                min_value=0.001, max_value=1.0,
                value=previous_params.get('distance_tolerance', 0.01),
                step=0.001, format="%.3f",
                help="Maximum axial distance to match defects."
            )
        with col2:
            clock_tolerance = st.number_input(
                "Clock Tolerance (min)",
                min_value=1, max_value=120,
                value=previous_params.get('clock_tolerance', 20),
                step=5,
                help="Maximum clock position difference for matching."
            )

    # Expander for methodology details
    with st.expander("About the Growth Analysis and Correction Methodology"):
        st.markdown("""
        This process matches defects across years and calculates their growth, with automated corrections for measurement anomalies.
        - **Defect Matching**: Uses axial distance and clock position to identify corresponding defects.
        - **Negative Growth Correction**: Addresses physically unrealistic "negative growth" readings:
            - **K-Nearest Neighbors (KNN)**: For minor negative growth, a KNN algorithm identifies similar defects within a local radius (±5 joints) to impute a realistic, weighted-average growth rate.
            - **Zero-Growth Flagging**: For significant negative growth (exceeding 15% of wall thickness), growth is set to zero and flagged for engineering review.
        """)

    # Execution and status display
    if st.button("Run Growth Analysis", type="primary", use_container_width=True):
        _perform_advanced_comparison_analysis(
            datasets, earlier_year, later_year,
            distance_tolerance, clock_tolerance
        )

    comparison_years = get_state('comparison_years')
    if comparison_years and get_state('comparison_results'):
        if comparison_years == (earlier_year, later_year):
            st.success("Growth analysis is complete. View results in the other tabs.")
        else:
            st.warning(f"Results for {comparison_years[0]}-{comparison_years[1]} are loaded. Run analysis for the current selection if needed.")


def render_failure_prediction_section(datasets, comparison_results):
    """
    Renders the failure prediction section, allowing users to configure and run simulations.
    """
    st.header("Failure Prediction with Dynamic Clustering")
    st.markdown("Simulate future defect growth and predict potential failures over time.")

    comparison_years = get_state('comparison_years')
    if not comparison_results or not comparison_years:
        st.warning("Please complete the Growth Analysis before running a failure prediction.")
        return

    _, later_year = comparison_years
    later_data = datasets[later_year]

    # Group simulation parameters in a bordered container
    with st.container(border=True):
        st.subheader("Simulation Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            assessment_method = st.selectbox("Assessment Method", ['B31G', 'Modified_B31G', 'RSTRENG'], index=2)
            max_operating_pressure = st.number_input("Max Operating Pressure (MPa)", 1.0, 20.0, 8.0, 0.1)
            simulation_years = st.number_input("Simulation Years", 1, 50, 15, 1)
        with col2:
            use_dynamic_clustering = st.checkbox("Enable Dynamic Clustering", True, help="Re-evaluate defect interactions each year.")
            clustering_standard = st.selectbox("Clustering Standard", ["DNV", "ROSEN#15"], 0) if use_dynamic_clustering else None
            safety_factor = st.number_input("Safety Factor", 1.0, 5.0, 1.39, 0.01)
        with col3:
            erf_threshold = st.number_input("ERF Failure Threshold", 0.5, 1.0, 0.90, 0.01)
            depth_threshold = st.number_input("Depth Failure Threshold (%)", 50.0, 95.0, 80.0, 1.0)
            smys = st.number_input("SMYS (MPa)", 200.0, 800.0, 415.0, 5.0)

    # Expander for explaining the simulation methodology
    with st.expander("About the Failure Prediction Methodology"):
        st.markdown("""
        This simulation projects future defect failures using the calculated growth rates.
        - **Growth Model**: Assumes linear growth (constant mm/year) for defect depth and length.
        - **Dynamic Clustering**: Re-evaluates defect interactions each simulated year, which is crucial as defects can grow into proximity over time.
        - **Failure Criteria**: A defect fails if its depth exceeds a threshold (e.g., 80%) or if its safe operating pressure falls below the specified maximum.
        - **Assessment Standard**: Failure pressure for clusters is calculated using the RSTRENG Effective Area method.
        """)

    if st.button("Run Failure Prediction", type="primary", use_container_width=True):
        with st.spinner("Running failure prediction with dynamic clustering..."):
            sim_params = SimulationParams(
                assessment_method=assessment_method.lower().replace('_', '_'),
                max_operating_pressure=max_operating_pressure,
                simulation_years=simulation_years,
                erf_threshold=erf_threshold,
                depth_threshold=depth_threshold
            )
            clustering_config = {
                'enabled': use_dynamic_clustering,
                'standard': clustering_standard,
                'pipe_diameter_mm': later_data['pipe_diameter'] * 1000
            } if use_dynamic_clustering else None

            run_integrated_simulation(
                sim_params, later_data, comparison_results,
                clustering_config, smys, safety_factor
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
    """
    Displays the results of the failure prediction simulation, including metrics, charts, and risk assessment.
    """
    results = get_state('prediction_results')
    if not results:
        return

    with st.container(border=True):
        st.header("Simulation Results")

        # Display defect and joint-level statistics
        survival_stats = results.get('survival_statistics', {})
        joint_stats = results.get('joint_survival_statistics', {})

        st.subheader("Defect-Level Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Defects", survival_stats.get('total_defects', 0))
        col2.metric("Failed Defects", survival_stats.get('failed_defects', 0), f"{survival_stats.get('failure_rate', 0.0):.1f}% of total")
        col3.metric("Surviving Defects", survival_stats.get('surviving_defects', 0), f"{survival_stats.get('survival_rate', 100.0):.1f}% survival")
        first_failure_year = min((f.failure_year for f in results.get('failure_history', [])), default=None)
        col4.metric("First Defect Failure", f"Year {first_failure_year}" if first_failure_year else "None Predicted")

        st.subheader("Joint-Level Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Joints", joint_stats.get('total_joints', 0))
        col2.metric("Failed Joints", joint_stats.get('failed_joints', 0), f"{joint_stats.get('joint_failure_rate', 0.0):.1f}% of total")
        col3.metric("Surviving Joints", joint_stats.get('surviving_joints', 0), f"{joint_stats.get('joint_survival_rate', 100.0):.1f}% survival")
        first_joint_failure = next((year for year, count in sorted(results.get('joint_failure_timeline', {}).items()) if count > 0), None)
        col4.metric("First Joint Failure", f"Year {first_joint_failure}" if first_joint_failure else "None Predicted")

        # Display failure timeline chart
        st.plotly_chart(create_failure_timeline_histogram(results), use_container_width=True)

    # Risk assessment section
    with st.container(border=True):
        st.header("Risk Assessment")
        joint_failure_rate = joint_stats.get('joint_failure_rate', 0.0)
        if joint_failure_rate > 15:
            st.error(f"Critical Risk: {joint_failure_rate:.1f}% joint failure rate predicted. Immediate action is required.")
        elif joint_failure_rate > 5:
            st.warning(f"High Risk: {joint_failure_rate:.1f}% joint failure rate predicted. Targeted maintenance should be considered.")
        elif joint_failure_rate > 0:
            st.info(f"Moderate Risk: {joint_failure_rate:.1f}% joint failure rate predicted. Continued monitoring is advised.")
        else:
            st.success("Excellent: No joint failures are predicted under the simulated conditions.")

    # Detailed timeline in an expander
    joint_timeline = results.get('joint_failure_timeline', {})
    if any(joint_timeline.values()):
        with st.expander("View Joint Failure Timeline Details"):
            timeline_df = pd.DataFrame([
                {'Year': year, 'Joints Failed': count}
                for year, count in sorted(joint_timeline.items()) if count > 0
            ])
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)


def render_results_visualization_tab(datasets):
    """
    Renders the 'Results' tab, providing a comprehensive display of the analysis outcomes.
    This tab includes overviews, growth analysis, correction reviews, and data export functionalities.
    """
    st.header("Analysis Results")
    st.markdown("Explore the detailed outcomes of the growth analysis, including charts, data summaries, and export options.")

    comparison_results = get_state('comparison_results')
    comparison_years = get_state('comparison_years')

    if not comparison_results or not comparison_years:
        st.info("No analysis results are available. Please run the Growth Analysis to generate results.")
        return

    earlier_year, later_year = comparison_years
    earlier_data = datasets[earlier_year]
    later_data = datasets[later_year]

    # Correction Summary
    correction_results = get_state('correction_results')
    if correction_results:
        _display_correction_summary(correction_results)

    # Overview Charts
    st.subheader("Analysis Overview")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_defect_status_donut(earlier_data, later_data), use_container_width=True)
        with col2:
            st.plotly_chart(create_new_defects_by_type_bar(later_data), use_container_width=True)

    # Detailed Analysis Sections
    with st.expander("Growth Analysis Details", expanded=True):
        _render_growth_analysis_visualizations()

    with st.expander("Correction Review Details"):
        _render_correction_review_section()

    # Data Export Section
    with st.container(border=True):
        st.subheader("Data Export and Documentation")
        _render_data_export_section()





def _render_correction_review_section():
    """
    Renders the detailed correction review section, showing flagged defects and summaries of auto-corrections.
    """
    correction_results = get_state('correction_results')
    comparison_results = get_state('comparison_results')
    if not correction_results or not comparison_results:
        st.info("No correction data is available for review.")
        return

    st.header("Detailed Correction Review")
    
    # Expander for methodology details
    with st.expander("Correction Methodology Summary"):
        st.markdown("""
        **Automated Correction Process:**
        1. **Uncertainty Check**: Flags defects with growth changes exceeding 15% of the wall thickness.
        2. **Zero-Growth Assignment**: Assigns zero growth to flagged defects as a conservative measure.
        3. **Weighted KNN Correction**: Corrects remaining negative growth using a weighted average from similar defects within a ±5 joint radius.
        """)
        if 'correction_details' in correction_results:
            details = correction_results['correction_details']
            st.write(f"**Joint Search Tolerance Used**: ±{details.get('joint_tolerance_used', 5)} joints")
            st.write(f"**Correction Success Rate**: {details.get('corrected_count', 0)} of {details.get('total_negative', 0)} negative growth defects corrected.")

    # Display table of flagged defects
    if correction_results['flagged_defects']:
        st.subheader("Defects Flagged for Engineering Review")
        flagged_df = pd.DataFrame(correction_results['flagged_defects'])
        display_cols = {
            'Location (m)': flagged_df['location'].round(2),
            'Joint Number': flagged_df['joint'].astype(int),
            'Original Rate (mm/year)': flagged_df['original_rate'].round(4),
            'Uncertainty Threshold (mm)': flagged_df['threshold'].round(2)
        }
        st.dataframe(pd.DataFrame(display_cols), use_container_width=True, hide_index=True)
        st.warning("Action Required: These defects exceeded measurement uncertainty and have been assigned zero growth. A manual review is recommended.")

    # Display summary of auto-corrected defects
    if correction_results['auto_corrected'] > 0:
        st.subheader("Auto-Corrected Defects Summary")
        matches_df = comparison_results['matches_df']
        corrected_defects = matches_df[matches_df.get('is_corrected', False)]
        if not corrected_defects.empty:
            avg_original_rate = corrected_defects['growth_rate_mm_per_year'].mean()
            avg_corrected_rate = corrected_defects.get('corrected_growth_rate_mm_per_year', avg_original_rate).mean()
            st.metric("Defects Auto-Corrected", len(corrected_defects))
            st.metric("Average Original Rate", f"{avg_original_rate:.4f} mm/year")
            st.metric("Average Corrected Rate", f"{avg_corrected_rate:.4f} mm/year")

def _render_data_export_section():
    """
    Renders data export capabilities with unique keys for download buttons.
    """
    comparison_results = get_state('comparison_results')
    correction_results = get_state('correction_results')
    if not comparison_results:
        st.info("No analysis results are available for export.")
        return

    st.header("Data Export and Documentation")

    # Section for data downloads
    with st.container(border=True):
        st.subheader("Analysis Data")
        col1, col2 = st.columns(2)
        with col1:
            if not comparison_results['matches_df'].empty:
                st.download_button(
                    label="Download Matched Defects (CSV)",
                    data=comparison_results['matches_df'].to_csv(index=False),
                    file_name=f"matched_defects_corrected_{get_state('comparison_years')[0]}_{get_state('comparison_years')[1]}.csv",
                    mime="text/csv",
                    key="export_matched_defects_corrected"
                )
        with col2:
            if not comparison_results['new_defects'].empty:
                st.download_button(
                    label="Download New Defects (CSV)",
                    data=comparison_results['new_defects'].to_csv(index=False),
                    file_name=f"new_defects_{get_state('comparison_years')[1]}.csv",
                    mime="text/csv",
                    key="export_new_defects_details"
                )

    # Section for documentation
    with st.container(border=True):
        st.subheader("Documentation")
        col1, col2 = st.columns(2)
        with col1:
            if correction_results:
                st.download_button(
                    label="Download Correction Report",
                    data=_generate_correction_report(correction_results, comparison_results),
                    file_name=f"growth_correction_report_{get_state('comparison_years')[0]}_{get_state('comparison_years')[1]}.txt",
                    mime="text/plain",
                    key="export_correction_report_details"
                )
        with col2:
            st.download_button(
                label="Download Methodology Document",
                data=_generate_methodology_documentation(),
                file_name="growth_correction_methodology.txt",
                mime="text/plain",
                key="export_methodology_details"
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
