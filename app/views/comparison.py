"""
Multi-year comparison view for the Pipeline Analysis application.
"""
import streamlit as st
import pandas as pd
import numpy as np

from app.ui_components import (
    custom_metric, info_box, create_data_download_links, create_comparison_metrics
)
from core.multi_year_analysis import compare_defects
from analysis.growth_analysis import correct_negative_growth_rates
from visualization.comparison_viz import *
from app.services.state_manager import *

def display_comparison_visualization_tabs(comparison_results, earlier_year, later_year):
    """Display the consolidated visualization tabs for comparison results."""
    
    # Create visualization tabs
    viz_tabs = st.tabs([
        "New vs Common", "New Defect Types", "Negative Growth Correction", "Growth Rate Analysis", "Remaining Life Analysis"
    ])
    
    with viz_tabs[0]:
        # Pie chart of common vs new defects
        pie_fig = create_comparison_stats_plot(comparison_results)
        st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar': False})
    
    with viz_tabs[1]:
        # Bar chart of new defect types
        bar_fig = create_new_defect_types_plot(comparison_results)
        st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Negative Growth Correction tab
    with viz_tabs[2]:
        st.subheader("Growth Analysis with Correction")
        
        # Add dimension selection to this tab
        st.markdown("#### Select Dimension for Analysis")
        
        # Get available dimensions
        available_dimensions = []
        if comparison_results.get('has_depth_data', False):
            available_dimensions.append('depth')
        if comparison_results.get('has_length_data', False):
            available_dimensions.append('length')
        if comparison_results.get('has_width_data', False):
            available_dimensions.append('width')
        
        if not available_dimensions:
            st.warning("No growth data available for any dimension.")
        else:
            # Get current dimension from session state
            current_dimension = get_state('correction_dimension', 'depth')
            
            # If current dimension isn't in available dimensions, default to the first available
            if current_dimension not in available_dimensions:
                current_dimension = available_dimensions[0]
                update_state('correction_dimension', current_dimension)
            
            # Create a unique key for the selectbox
            select_key = f"correction_dimension_{earlier_year}_{later_year}"
            
            # Create the selectbox with default value from session state
            selected_dimension = st.selectbox(
                "Choose dimension to analyze",
                options=available_dimensions,
                index=available_dimensions.index(current_dimension),
                key=select_key,
                help="Select which defect dimension to analyze for growth patterns"
            )
            
            # Check if selection changed
            if selected_dimension != current_dimension:
                # Update session state
                update_state('correction_dimension', selected_dimension)
                # Force a rerun to ensure UI updates immediately
                st.rerun()
                
            # Update session state
            st.session_state.correction_dimension = selected_dimension
            
            # Show growth plot for selected dimension
            st.markdown(f"#### {selected_dimension.title()} Growth Data")
            
            # Show the selected dimension plot
            original_plot = create_negative_growth_plot(comparison_results, dimension=selected_dimension)
            st.plotly_chart(original_plot, use_container_width=True, config={'displayModeBar': False})
            
            # Only show correction controls for depth dimension
            if selected_dimension == 'depth':
                # Check if depth data is available for correction
                if not (comparison_results.get('has_depth_data', False) and 'is_negative_growth' in comparison_results['matches_df'].columns):
                    st.warning("No depth growth data available for correction. Make sure both datasets have depth measurements.")
                else:
                    # Display negative depth growth summary
                    neg_count = comparison_results['matches_df']['is_negative_growth'].sum()
                    total_count = len(comparison_results['matches_df'])
                    pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                    
                    st.markdown("#### Negative Depth Growth Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(custom_metric("Total Defects", f"{total_count}"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(custom_metric("Negative Growth", f"{neg_count}"), unsafe_allow_html=True)
                    with col3:
                        st.markdown(custom_metric("Percentage", f"{pct:.1f}%"), unsafe_allow_html=True)
                    
                    if neg_count > 0:
                        st.info("Negative depth growth rates are likely measurement errors and can be corrected using similar defects in the same joint.")
                    else:
                        st.success("No negative depth growth detected - no correction needed!")
                    
                    # Show corrected results if available
                    if st.session_state.corrected_results is not None:
                        corrected_results = st.session_state.corrected_results
                        correction_info = corrected_results.get('correction_info', {})
                        
                        if correction_info.get("success", False):
                            st.markdown("#### Correction Results")
                            st.success(f"Successfully corrected {correction_info['corrected_count']} out of {correction_info['total_negative']} negative depth growth defects.")
                            
                            if correction_info['uncorrected_count'] > 0:
                                st.warning(f"Could not correct {correction_info['uncorrected_count']} defects in {len(correction_info['uncorrected_joints'])} joints due to insufficient similar defects.")
                            
                            # Show corrected growth plot
                            st.markdown("#### Corrected Depth Growth Data")
                            corrected_plot = create_negative_growth_plot(corrected_results, dimension='depth')
                            st.plotly_chart(corrected_plot, use_container_width=True, config={'displayModeBar': False})
                            
                            # Legend
                            st.markdown("""
                            **Legend:**
                            - Blue circles: Positive growth (unchanged)
                            - Red triangles: Negative growth (uncorrected)
                            - Green diamonds: Corrected growth (formerly negative)
                            """)
                    
                    # Show KNN correction controls only if there are negative growth defects
                    if neg_count > 0:
                        # Check if joint numbers are available for KNN correction
                        has_joint_num = comparison_results.get('has_joint_num', False)
                        if not has_joint_num:
                            st.warning("""
                            **Joint numbers not available for correction**
                            
                            The KNN correction requires the 'joint number' column to be present in your defect data.
                            Please ensure both datasets have this column properly mapped.
                            """)
                        else:
                            # KNN correction controls
                            st.markdown("#### Apply KNN Correction to Depth")
                            
                            k_neighbors = st.slider(
                                "Number of Similar Defects (K) for Correction",
                                min_value=1,
                                max_value=5,
                                value=3,  # Default value
                                key=f"k_neighbors_{earlier_year}_{later_year}",
                                help="Number of similar defects with positive growth to use for estimating corrected depth growth rates"
                            )
                            
                            # Correction form
                            with st.form(key=f"depth_correction_form_{earlier_year}_{later_year}"):
                                st.write("Click the button below to apply KNN correction to negative depth growth:")
                                submit_correction = st.form_submit_button("Apply Depth Correction", use_container_width=True)
                                
                                if submit_correction:
                                    with st.spinner("Correcting negative depth growth rates using KNN..."):
                                        try:
                                            corrected_results = st.session_state.comparison_results.copy()
                                            
                                            # Apply the correction
                                            corrected_df, correction_info = correct_negative_growth_rates(
                                                st.session_state.comparison_results['matches_df'], 
                                                k=k_neighbors
                                            )
                                            
                                            corrected_results['matches_df'] = corrected_df
                                            corrected_results['correction_info'] = correction_info
                                            
                                            # Update growth stats if correction was successful
                                            if correction_info.get("updated_growth_stats"):
                                                corrected_results['growth_stats'] = correction_info['updated_growth_stats']
                                            
                                            st.session_state.corrected_results = corrected_results
                                            
                                            if correction_info.get("success", False):
                                                st.success(f"Successfully corrected {correction_info['corrected_count']} out of {correction_info['total_negative']} negative depth growth defects.")
                                                
                                                if correction_info['uncorrected_count'] > 0:
                                                    st.warning(f"Could not correct {correction_info['uncorrected_count']} defects in {len(correction_info['uncorrected_joints'])} joints due to insufficient similar defects.")
                                                
                                                st.rerun()
                                            else:
                                                st.error(f"Could not apply correction: {correction_info.get('error', 'Unknown error')}")
                                        except Exception as e:
                                            st.error(f"Error during correction: {str(e)}")
                                            st.info("This could be due to missing sklearn library or incompatible data. Please check that your data has all required fields: joint number, length, width, and depth.")
            else:
                # For length and width dimensions, show analysis but no correction
                st.info(f"""
                **{selected_dimension.title()} Growth Analysis**
                
                You are viewing {selected_dimension} growth analysis. The plot above shows how {selected_dimension} measurements 
                changed between inspections. 
                
                **Note**: KNN correction is only available for depth measurements. Switch to 'depth' dimension 
                to access correction features.
                """)
                
                # Show basic stats for length/width
                matches_df = comparison_results['matches_df']
                
                if selected_dimension == 'length' and comparison_results.get('has_length_data', False):
                    if 'is_negative_length_growth' in matches_df.columns:
                        neg_count = matches_df['is_negative_length_growth'].sum()
                        total_count = len(matches_df)
                        pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                        
                        st.markdown("#### Length Growth Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(custom_metric("Total Defects", f"{total_count}"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(custom_metric("Negative Growth", f"{neg_count}"), unsafe_allow_html=True)
                        with col3:
                            st.markdown(custom_metric("Percentage", f"{pct:.1f}%"), unsafe_allow_html=True)
                
                elif selected_dimension == 'width' and comparison_results.get('has_width_data', False):
                    if 'is_negative_width_growth' in matches_df.columns:
                        neg_count = matches_df['is_negative_width_growth'].sum()
                        total_count = len(matches_df)
                        pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                        
                        st.markdown("#### Width Growth Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(custom_metric("Total Defects", f"{total_count}"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(custom_metric("Negative Growth", f"{neg_count}"), unsafe_allow_html=True)
                        with col3:
                            st.markdown(custom_metric("Percentage", f"{pct:.1f}%"), unsafe_allow_html=True)
    
    # Growth Rate Analysis tab
    with viz_tabs[3]:
        st.subheader("Growth Rate Analysis")
        
        # Add dimension selection to this tab
        st.markdown("#### Select Dimension for Growth Rate Analysis")        

        # Get current dimension from session state
        current_dimension = get_state('growth_analysis_dimension', 'depth')
        
        # All available dimensions
        dimensions = ['depth', 'length', 'width']
        
        # Create a unique key
        select_key = f"growth_dimension_{earlier_year}_{later_year}"
        
        # Create the selectbox with default value from session state
        growth_dimension = st.selectbox(
            "Choose dimension for growth rate analysis",
            options=dimensions,
            index=dimensions.index(current_dimension),
            key=select_key,
            help="Select which defect dimension to analyze for growth rate statistics"
        )
        
        # Check if selection changed
        if growth_dimension != current_dimension:
            # Update session state
            update_state('growth_analysis_dimension', growth_dimension)
            # Force a rerun to ensure UI updates immediately
            st.rerun()

        st.session_state.growth_analysis_dimension = growth_dimension
        
        # Use the comparison_results parameter directly, check for corrected results in session state
        results_to_use = (
            st.session_state.corrected_results 
            if st.session_state.get("corrected_results") is not None 
            else st.session_state.get("comparison_results")
        )
        
        if results_to_use is None:
            st.info("No comparison data available.")
        else:
            matches_df = results_to_use.get('matches_df', pd.DataFrame())
            
            if matches_df.empty:
                st.warning("No comparison data available in the results.")
            else:
                # Define dimension-specific column names and check if they exist in the dataframe
                dimension_columns = {
                    'depth': {
                        'negative_flag': 'is_negative_growth',
                        'growth_rate_cols': ['growth_rate_mm_per_year', 'growth_rate_pct_per_year']
                    },
                    'length': {
                        'negative_flag': 'is_negative_length_growth', 
                        'growth_rate_cols': ['length_growth_rate_mm_per_year']
                    },
                    'width': {
                        'negative_flag': 'is_negative_width_growth',
                        'growth_rate_cols': ['width_growth_rate_mm_per_year']
                    }
                }
                
                # Check if the selected dimension has the required columns
                dim_config = dimension_columns.get(growth_dimension)
                if not dim_config:
                    st.warning(f"Invalid dimension selected: {growth_dimension}")
                else:
                    negative_flag = dim_config['negative_flag']
                    growth_rate_cols = dim_config['growth_rate_cols']
                    
                    # Find which growth rate column exists in the dataframe
                    available_growth_col = None
                    for col in growth_rate_cols:
                        if col in matches_df.columns:
                            available_growth_col = col
                            break
                    
                    # Check if we have the minimum required columns
                    if negative_flag not in matches_df.columns or available_growth_col is None:
                        st.warning(f"""
                        **No {growth_dimension} growth data available**
                        
                        Required columns missing from comparison results:
                        - Negative flag: {'✅' if negative_flag in matches_df.columns else '❌'} `{negative_flag}`
                        - Growth rate: {'✅' if available_growth_col else '❌'} `{' or '.join(growth_rate_cols)}`
                        
                        Make sure both datasets have {growth_dimension} measurements and valid year values.
                        
                        Available columns in matches_df: {list(matches_df.columns)}
                        """)
                    else:
                        # Show correction status if applicable
                        if growth_dimension == 'depth' and 'correction_info' in results_to_use and results_to_use['correction_info'].get('success', False):
                            st.success("Showing analysis with corrected depth growth rates. The negative growth defects have been adjusted based on similar defects.")
                        
                        # Display growth rate statistics
                        st.markdown(f"#### {growth_dimension.title()} Growth Statistics")
                        
                        # Determine the unit based on the column name
                        if 'mm_per_year' in available_growth_col:
                            unit = 'mm/year'
                        elif 'pct_per_year' in available_growth_col:
                            unit = '%/year'
                        else:
                            unit = 'units/year'
                        
                        # Calculate statistics dynamically (ensures they show immediately after comparison)
                        negative_count = matches_df[negative_flag].sum()
                        total_count = len(matches_df)
                        pct_negative = (negative_count / total_count) * 100 if total_count > 0 else 0
                        
                        # Calculate positive growth statistics
                        positive_growth = matches_df[~matches_df[negative_flag]]
                        avg_growth = positive_growth[available_growth_col].mean() if len(positive_growth) > 0 else 0
                        max_growth = positive_growth[available_growth_col].max() if len(positive_growth) > 0 else 0
                        
                        # Display statistics
                        stats_cols = st.columns(3)
                        
                        with stats_cols[0]:
                            st.markdown(
                                custom_metric(
                                    f"Avg {growth_dimension.title()} Growth Rate", 
                                    f"{avg_growth:.3f} {unit}"
                                ), 
                                unsafe_allow_html=True
                            )
                        
                        with stats_cols[1]:
                            st.markdown(
                                custom_metric(
                                    f"Max {growth_dimension.title()} Growth Rate", 
                                    f"{max_growth:.3f} {unit}"
                                ), 
                                unsafe_allow_html=True
                            )
                        
                        with stats_cols[2]:
                            st.markdown(
                                custom_metric(
                                    "Negative Growth", 
                                    f"{negative_count} ({pct_negative:.1f}%)"
                                ), 
                                unsafe_allow_html=True
                            )
                        
                        # Show histogram for selected dimension
                        st.markdown(f"#### {growth_dimension.title()} Growth Rate Distribution")
                        try:
                            growth_hist_fig = create_growth_rate_histogram(results_to_use, dimension=growth_dimension)
                            st.plotly_chart(growth_hist_fig, use_container_width=True, config={'displayModeBar': False})
                        except Exception as e:
                            st.warning(f"Could not generate histogram: {str(e)}. Data is available but visualization failed.")

    # Remaining Life Analysis tab
    with viz_tabs[4]:
        st.subheader("Remaining Life Analysis")
        
        # Import remaining life functions
        try:
            from analysis.remaining_life_analysis import calculate_remaining_life_analysis
            from visualization.remaining_life_viz import (
                create_remaining_life_pipeline_visualization,
                create_remaining_life_histogram,
                create_remaining_life_summary_table,
                create_remaining_life_risk_matrix
            )
        except ImportError as e:
            st.error(f"Could not import remaining life analysis modules: {str(e)}")
            st.info("Please ensure the remaining life analysis files are properly installed.")
            return
        
        # Check if we have the required data for remaining life analysis
        if not comparison_results.get('has_depth_data', False):
            st.warning("**Remaining life analysis requires depth data**")
            st.info("Please ensure both datasets have depth measurements to enable remaining life predictions.")
        else:
            # Get joints data for wall thickness lookup
            earlier_joints = st.session_state.datasets[earlier_year]['joints_df']
            later_joints = st.session_state.datasets[later_year]['joints_df']
            
            # Use later joints for wall thickness (most recent data)
            joints_for_analysis = later_joints
            
            # Perform remaining life analysis
            with st.spinner("Calculating remaining life for all defects..."):
                try:
                    remaining_life_results = calculate_remaining_life_analysis(
                        comparison_results, 
                        joints_for_analysis
                    )
                    
                    if remaining_life_results.get('analysis_possible', False):
                        # Display summary statistics
                        st.markdown("#### Analysis Summary")
                        summary_table = create_remaining_life_summary_table(remaining_life_results)
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(summary_table, use_container_width=True, hide_index=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Create sub-tabs for different visualizations
                        remaining_life_subtabs = st.tabs([
                            "Pipeline Overview", "Risk Matrix", "Distribution", "Detailed Results"
                        ])
                        
                        with remaining_life_subtabs[0]:
                            st.markdown("#### Pipeline Remaining Life Overview")
                            st.info("""
                            **Color Legend:**
                            - 🔴 **Critical**: Already at 80% depth or above
                            - 🟠 **High Risk**: <2 years remaining
                            - 🟡 **Medium Risk**: 2-10 years remaining  
                            - 🟢 **Low Risk**: >10 years remaining
                            - 🔵 **Stable**: Zero/negative growth (>100 years)
                            """)
                            
                            pipeline_viz = create_remaining_life_pipeline_visualization(remaining_life_results)
                            st.plotly_chart(pipeline_viz, use_container_width=True, config={'displayModeBar': True})
                        
                        with remaining_life_subtabs[1]:
                            st.markdown("#### Risk Matrix")
                            st.info("This matrix shows the relationship between current defect depth and predicted remaining life.")
                            
                            risk_matrix = create_remaining_life_risk_matrix(remaining_life_results)
                            st.plotly_chart(risk_matrix, use_container_width=True, config={'displayModeBar': True})
                        
                        with remaining_life_subtabs[2]:
                            st.markdown("#### Remaining Life Distribution")
                            st.info("Distribution of predicted remaining life across all defects. Red/orange lines show risk thresholds.")
                            
                            histogram = create_remaining_life_histogram(remaining_life_results)
                            st.plotly_chart(histogram, use_container_width=True, config={'displayModeBar': True})
                        
                        with remaining_life_subtabs[3]:
                            st.markdown("#### Detailed Results")
                            
                            # Show results for matched defects (measured growth)
                            matched_results = remaining_life_results['matched_defects_analysis']
                            if matched_results:
                                st.markdown("##### Defects with Measured Growth Rates")
                                matched_df = pd.DataFrame(matched_results)
                                
                                # Select key columns for display
                                display_cols = [
                                    'log_dist', 'defect_type', 'current_depth_pct', 
                                    'growth_rate_pct_per_year', 'remaining_life_years', 'status'
                                ]
                                available_cols = [col for col in display_cols if col in matched_df.columns]
                                
                                # Format the display
                                display_matched = matched_df[available_cols].copy()
                                if 'remaining_life_years' in display_matched.columns:
                                    display_matched['remaining_life_years'] = display_matched['remaining_life_years'].apply(
                                        lambda x: f"{x:.1f}" if np.isfinite(x) else "Stable"
                                    )
                                
                                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                                st.dataframe(display_matched, use_container_width=True, hide_index=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show results for new defects (estimated growth)
                            new_results = remaining_life_results['new_defects_analysis']
                            if new_results:
                                st.markdown("##### New Defects with Estimated Growth Rates")
                                new_df = pd.DataFrame(new_results)
                                
                                # Select key columns for display
                                display_cols = [
                                    'log_dist', 'defect_type', 'current_depth_pct', 
                                    'growth_rate_pct_per_year', 'remaining_life_years', 'status',
                                    'estimation_confidence', 'similar_defects_count'
                                ]
                                available_cols = [col for col in display_cols if col in new_df.columns]
                                
                                # Format the display
                                display_new = new_df[available_cols].copy()
                                if 'remaining_life_years' in display_new.columns:
                                    display_new['remaining_life_years'] = display_new['remaining_life_years'].apply(
                                        lambda x: f"{x:.1f}" if np.isfinite(x) else "Stable"
                                    )
                                
                                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                                st.dataframe(display_new, use_container_width=True, hide_index=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Show estimation details
                                st.markdown("##### Growth Rate Estimation Details")
                                st.info("""
                                **Estimation Method**: Statistical inference from similar defects
                                
                                **Similarity Criteria**:
                                - Same defect type (exact match)
                                - Depth within ±10% of current defect
                                - Joint location within ±5 joints
                                
                                **Confidence Levels**:
                                - **HIGH**: Based on ≥10 similar defects
                                - **MEDIUM**: Based on 5-9 similar defects  
                                - **LOW**: Based on 3-4 similar defects or industry defaults
                                """)
                    else:
                        st.error("Remaining life analysis could not be performed.")
                        if 'error' in remaining_life_results:
                            st.error(f"Error: {remaining_life_results['error']}")
                            
                except Exception as e:
                    st.error(f"Error during remaining life analysis: {str(e)}")
                    st.info("Please check that your data has the required columns and format.")
                    
        # Add methodology explanation
        with st.expander("📖 Methodology & Assumptions", expanded=False):
            st.markdown("""
            ### Remaining Life Analysis Methodology
            
            #### **Objective**
            Predict when defects will reach the critical threshold of 80% wall thickness depth (B31G limit).
            
            #### **Growth Rate Sources**
            1. **Matched Defects**: Use measured growth rates from multi-year comparison
            2. **New Defects**: Estimate growth rates using statistical inference from similar defects
            
            #### **Similarity Criteria for New Defects**
            - **Defect Type**: Exact match (e.g., external corrosion, pitting)
            - **Current Depth**: Within ±10% of target defect's depth
            - **Location**: Within ±5 joints of target defect
            
            #### **Calculation Formula**
            ```
            Remaining Life = (80% - Current Depth%) / Growth Rate (%/year)
            ```
            
            #### **Risk Categories**
            - **🔴 Critical**: Already ≥80% depth
            - **🟠 High Risk**: <2 years remaining  
            - **🟡 Medium Risk**: 2-10 years remaining
            - **🟢 Low Risk**: >10 years remaining
            - **🔵 Stable**: Zero/negative growth
            
            #### **Limitations & Assumptions**
            - Assumes **linear growth** (constant rate over time)
            - Uses **80% depth** as failure criterion (industry standard)
            - **Conservative estimates** for new defects with limited data
            - Does not account for **environmental changes** or **mitigation measures**
            - Growth rates based on **historical data** may not reflect future conditions
            
            #### **Recommendations**
            - Use results for **prioritization** and **planning** purposes
            - **Validate estimates** with additional inspections
            - Consider **environmental factors** and **operational changes**
            - **Update analysis** regularly with new inspection data
            """)

    

def render_comparison_view():
    """Display the multi-year comparison view with analysis across different years."""
    st.markdown('<h2 class="section-header">Multi-Year Comparison</h2>', unsafe_allow_html=True)
    
    if len(st.session_state.datasets) < 2:
        st.info("""
            **Multiple datasets required**
            Please upload at least two datasets from different years to enable comparison.
            Use the sidebar to add more inspection data.
        """
        )
    else:
        # Year selection for comparison with improved UI
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        available_years = sorted(st.session_state.datasets.keys())
        
        st.markdown("<div class='section-header'>Select Years to Compare</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            earlier_year = st.selectbox(
                "Earlier Inspection Year", 
                options=available_years[:-1],  # All but the last year
                index=0,
                key="earlier_year_comparison"
            )
        
        with col2:
            # Filter for years after the selected earlier year
            later_years = [year for year in available_years if year > earlier_year]
            later_year = st.selectbox(
                "Later Inspection Year", 
                options=later_years,
                index=0,
                key="later_year_comparison"
            )
        
        # Get the datasets
        earlier_defects = st.session_state.datasets[earlier_year]['defects_df']
        later_defects = st.session_state.datasets[later_year]['defects_df']
        earlier_joints = st.session_state.datasets[earlier_year]['joints_df']
        later_joints = st.session_state.datasets[later_year]['joints_df']
        
        # Add parameter settings with better UI
        st.markdown("<div class='section-header' style='margin-top:20px;'>Comparison Parameters</div>", unsafe_allow_html=True)
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            # Distance tolerance for matching defects with tooltip
            tolerance = st.slider(
                "Distance Tolerance (m)", 
                min_value=0.001, 
                max_value=0.1, 
                value=0.01,  # Default value 
                step=0.001,
                format="%.3f",
                key="distance_tolerance_slider"
            )
            st.markdown(
                """
                <div style="font-size:0.8em;color:#7f8c8d;margin-top:-10px;">
                Maximum distance between defects to consider them the same feature
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with param_col2:
            # Clock tolerance for matching defects
            clock_tolerance = st.slider(
                "Clock Position Tolerance (minutes)",
                min_value=0,
                max_value=60,
                value=20,  # Default value
                step=5,
                key="clock_tolerance_slider"
            )
            st.markdown(
                """
                <div style="font-size:0.8em;color:#7f8c8d;margin-top:-10px;">
                Maximum difference in clock position to consider defects the same feature
                </div>
                """, 
                unsafe_allow_html=True
            )

        
        # Button to perform comparison
        if st.button("Compare Defects", key="compare_defects_button", use_container_width=True):
            with st.spinner(f"Comparing defects between {earlier_year} and {later_year}..."):
                try:
                    # Store the years for later reference
                    st.session_state.comparison_years = (earlier_year, later_year)
                    
                    # Perform the comparison
                    comparison_results = compare_defects(
                        earlier_defects, 
                        later_defects,
                        old_joints_df=earlier_joints,
                        new_joints_df=later_joints,
                        old_year=int(earlier_year),
                        new_year=int(later_year),
                        distance_tolerance=tolerance,
                        clock_tolerance_minutes=clock_tolerance,
                    )
                    
                    # Store the comparison results in session state for other tabs
                    st.session_state.comparison_results = comparison_results
                    # Reset corrected results when new comparison is made
                    st.session_state.corrected_results = None
                    
                    # Display summary statistics
                    st.markdown("<div class='section-header' style='margin-top:20px;'>Comparison Summary</div>", unsafe_allow_html=True)
                    
                    # Create metrics for comparison results
                    create_comparison_metrics(comparison_results)
                    
                    # Call the consolidated visualization function
                    display_comparison_visualization_tabs(comparison_results, earlier_year, later_year)
                    
                    # Display tables of common and new defects in an expander
                    with st.expander("Detailed Defect Lists", expanded=False):
                        if not comparison_results['matches_df'].empty:
                            st.markdown("<div class='section-header'>Common Defects</div>", unsafe_allow_html=True)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(comparison_results['matches_df'], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        if not comparison_results['new_defects'].empty:
                            st.markdown("<div class='section-header' style='margin-top:20px;'>New Defects</div>", unsafe_allow_html=True)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(comparison_results['new_defects'], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    info_box(
                        f"""
                        <strong>Error comparing defects:</strong> {str(e)}<br><br>
                        Make sure both datasets have the required columns and compatible data formats.
                        """, 
                        "warning"
                    )
        
        # Show stored comparison results if available
        elif st.session_state.comparison_results is not None:
            comparison_results = st.session_state.comparison_results
            # Check if the years match our current selection
            if st.session_state.comparison_years == (earlier_year, later_year):
                # Display summary statistics
                st.markdown("<div class='section-header' style='margin-top:20px;'>Comparison Summary</div>", unsafe_allow_html=True)
                
                # Create metrics for comparison results
                create_comparison_metrics(comparison_results)
                
                # Call the consolidated visualization function
                display_comparison_visualization_tabs(comparison_results, earlier_year, later_year)
                
                # Display tables of common and new defects in an expander
                with st.expander("Detailed Defect Lists", expanded=False):
                    if not comparison_results['matches_df'].empty:
                        st.markdown("<div class='section-header'>Common Defects</div>", unsafe_allow_html=True)
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(comparison_results['matches_df'], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if not comparison_results['new_defects'].empty:
                        st.markdown("<div class='section-header' style='margin-top:20px;'>New Defects</div>", unsafe_allow_html=True)
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(comparison_results['new_defects'], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Years don't match, ask user to re-run comparison
                st.info("You've changed the years for comparison. Please click 'Compare Defects' to analyze the new year combination.")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close the card container