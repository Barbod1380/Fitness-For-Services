"""
Multi-year comparison view for the Pipeline Analysis application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from app.ui_components import custom_metric, info_box, create_comparison_metrics
from core.multi_year_analysis import compare_defects
from analysis.remaining_life_analysis import enhanced_calculate_remaining_life_analysis
from analysis.growth_analysis import correct_negative_growth_rates
from visualization.comparison_viz import *
from app.services.state_manager import *

def display_comparison_visualization_tabs(comparison_results, earlier_year, later_year):
    """Display the consolidated visualization tabs for comparison results."""
    
    # Create visualization tabs
    viz_tabs = st.tabs([
        "New vs Common", "New Defect Types", "Negative Growth Correction", "Growth Rate Analysis", "Remaining Life Analysis", "Dynamic Clustering Simulation"
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
            # Create a unique key for the selectbox
            select_key = f"correction_dimension_{earlier_year}_{later_year}"

            # Initialize the selectbox state if it doesn't exist
            if select_key not in st.session_state:
                st.session_state[select_key] = 'depth' if 'depth' in available_dimensions else available_dimensions[0]

            # Ensure the stored value is valid
            if st.session_state[select_key] not in available_dimensions:
                st.session_state[select_key] = available_dimensions[0]

            # Create the selectbox - Streamlit automatically syncs with st.session_state[select_key]
            selected_dimension = st.selectbox(
                "Choose dimension to analyze",
                options=available_dimensions,
                key=select_key,
                help="Select which defect dimension to analyze for growth patterns"
            )
            
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

        # Initialize if needed
        if select_key not in st.session_state:
            st.session_state[select_key] = 'depth'

        # Ensure valid
        if st.session_state[select_key] not in ['depth', 'length', 'width']:
            st.session_state[select_key] = 'depth'

        # Create the selectbox
        growth_dimension = st.selectbox(
            "Choose dimension for growth rate analysis",
            options=['depth', 'length', 'width'],
            key=select_key,
            help="Select which defect dimension to analyze for growth rate statistics"
        )
        
        # Check if selection changed
        if growth_dimension != current_dimension:
            # Update session state
            update_state('growth_analysis_dimension', growth_dimension)

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
            
            # === NEW: Operating Pressure and Pipeline Parameters Input ===
            st.markdown("#### Pipeline Parameters for Pressure-Based Analysis")
            
            # Create three columns for parameters
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                operating_pressure_mpa = st.number_input(
                    "Operating Pressure (MPa)",
                    min_value=0.1,
                    max_value=20.0,
                    value=5.0,
                    step=0.1,
                    format="%.2f",
                    key="operating_pressure_remaining_life",
                    help="Current operating pressure of the pipeline"
                )
                st.caption(f"= {operating_pressure_mpa * 145.038:.0f} psi")
            
            with param_col2:
                pipe_diameter_m = st.number_input(
                    "Pipe Diameter (m)",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    format="%.3f",
                    key="pipe_diameter_remaining_life",
                    help="Outside diameter of the pipeline"
                )
                st.caption(f"= {pipe_diameter_m * 1000:.0f} mm")
            
            with param_col3:
                # Pipe grade selector
                pipe_grade = st.selectbox(
                    "Pipe Grade",
                    options=["API 5L X42", "API 5L X52", "API 5L X60", "API 5L X65", "API 5L X70", "Custom"],
                    index=1,
                    key="pipe_grade_remaining_life"
                )
                
                grade_to_smys = {
                    "API 5L X42": 290,
                    "API 5L X52": 358,
                    "API 5L X60": 413,
                    "API 5L X65": 448,
                    "API 5L X70": 482
                }
                
                if pipe_grade != "Custom":
                    smys_mpa = grade_to_smys[pipe_grade]
                    st.caption(f"SMYS: {smys_mpa} MPa")
                else:
                    smys_mpa = st.number_input(
                        "Custom SMYS (MPa)",
                        min_value=200.0,
                        max_value=800.0,
                        value=358.0,
                        step=1.0,
                        format="%.2f",
                        key="smys_custom_remaining_life"
                    )
            
            # Convert diameter to mm for calculations
            pipe_diameter_mm = pipe_diameter_m * 1000
            
            if st.button("Run Remaining Life Analysis"):
                # Perform enhanced remaining life analysis
                with st.spinner("Calculating enhanced remaining life for all defects..."):
                    enhanced_remaining_life_results = enhanced_calculate_remaining_life_analysis(
                        comparison_results, 
                        joints_for_analysis,
                        operating_pressure_mpa,
                        pipe_diameter_mm,
                        smys_mpa
                    )
                    st.session_state.remaining_life_results = enhanced_remaining_life_results
            
            if hasattr(st.session_state, 'remaining_life_results'):
                if enhanced_remaining_life_results.get('analysis_possible', False):
                    # Display enhanced summary statistics
                    st.markdown("#### Enhanced Analysis Summary")
                    
                    # Create enhanced summary table
                    summary_stats = enhanced_remaining_life_results.get('summary_statistics', {})
                    
                    # Display key metrics
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    with summary_col1:
                        st.metric("Total Defects", summary_stats.get('total_defects_analyzed', 0))
                    with summary_col2:
                        st.metric("Measured Growth", summary_stats.get('defects_with_measured_growth', 0))
                    with summary_col3:
                        st.metric("Estimated Growth", summary_stats.get('defects_with_estimated_growth', 0))
                    with summary_col4:
                        st.metric("Operating Pressure", f"{operating_pressure_mpa:.1f} MPa")
                    
                    # Warning about assumptions
                    st.warning("""
                        ⚠️ **Enhanced Analysis Assumptions**:
                        - **Depth-based**: Failure at 80% wall thickness depth
                        - **Pressure-based**: Failure when operating pressure ≥ failure pressure
                        - **Growth rates**: Linear (constant over time)
                        - **Negative growth**: Replaced with average of similar defects
                        - **Assessment methods**: B31G, Modified B31G, and RSTRENG calculated yearly
                    """)
                    
                    # Create enhanced sub-tabs for different visualizations and results
                    enhanced_subtabs = st.tabs([
                        "Summary Comparison", "Detailed Results", "Pipeline Overview"
                    ])
                    
                    with enhanced_subtabs[0]:
                        st.markdown("#### Failure Criteria Comparison")
                        st.info("""
                        **Failure Criteria:**
                        - 🟦 **Depth-based**: Time until defect reaches 80% wall thickness
                        - 🟥 **B31G Pressure**: Time until operating pressure ≥ B31G failure pressure  
                        - 🟨 **Modified B31G Pressure**: Time until operating pressure ≥ Modified B31G failure pressure
                        - 🟩 **RSTRENG Pressure**: Time until operating pressure ≥ RSTRENG failure pressure
                        """)
                        
                        # Create comparison summary table
                        methods = ['depth_based', 'b31g_pressure', 'modified_b31g_pressure', 'rstreng_pressure']
                        method_names = {
                            'depth_based': 'Depth-Based (80%)',
                            'b31g_pressure': 'B31G Pressure-Based', 
                            'modified_b31g_pressure': 'Modified B31G Pressure-Based',
                            'rstreng_pressure': 'RSTRENG Pressure-Based'
                        }
                        
                        comparison_rows = []
                        for method in methods:
                            avg_life = summary_stats.get(f'{method}_avg_remaining_life', np.nan)
                            min_life = summary_stats.get(f'{method}_min_remaining_life', np.nan)
                            status_dist = summary_stats.get(f'{method}_status_distribution', {})
                            
                            critical_count = status_dist.get('CRITICAL', 0) + status_dist.get('ERROR', 0)
                            high_risk_count = status_dist.get('HIGH_RISK', 0)
                            
                            comparison_rows.append({
                                'Failure Criterion': method_names[method],
                                'Avg Life (years)': f"{avg_life:.1f}" if not np.isnan(avg_life) else "N/A",
                                'Min Life (years)': f"{min_life:.1f}" if not np.isnan(min_life) else "N/A", 
                                'Critical/Error': critical_count,
                                'High Risk': high_risk_count
                            })
                        
                        comparison_df = pd.DataFrame(comparison_rows)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    with enhanced_subtabs[1]:
                        st.markdown("#### Detailed Results by Defect")
                        
                        # Show results for matched defects (measured growth)
                        matched_results = enhanced_remaining_life_results['matched_defects_analysis']
                        if matched_results:
                            st.markdown("##### Defects with Measured Growth Rates")
                            matched_df = pd.DataFrame(matched_results)
                            
                            # Select key columns for display
                            display_cols = [
                                'log_dist', 'defect_type', 'joint_number',
                                'depth_based_remaining_life', 'depth_based_status',
                                'b31g_pressure_remaining_life', 'b31g_pressure_status',
                                'modified_b31g_pressure_remaining_life', 'modified_b31g_pressure_status', 
                                'rstreng_pressure_remaining_life', 'rstreng_pressure_status'
                            ]
                            available_cols = [col for col in display_cols if col in matched_df.columns]
                            
                            # Format the display
                            display_matched = matched_df[available_cols].copy()
                            
                            # Format remaining life columns
                            life_cols = [col for col in display_matched.columns if 'remaining_life' in col]
                            for col in life_cols:
                                display_matched[col] = display_matched[col].apply(
                                    lambda x: f"{x:.1f}" if np.isfinite(x) else ("∞" if x == float('inf') else "Error")
                                )
                            
                            # Rename columns for better display
                            column_rename = {
                                'log_dist': 'Location (m)',
                                'defect_type': 'Type',
                                'joint_number': 'Joint',
                                'depth_based_remaining_life': 'Depth Life (yrs)',
                                'depth_based_status': 'Depth Status',
                                'b31g_pressure_remaining_life': 'B31G Life (yrs)',
                                'b31g_pressure_status': 'B31G Status',
                                'modified_b31g_pressure_remaining_life': 'Mod-B31G Life (yrs)',
                                'modified_b31g_pressure_status': 'Mod-B31G Status',
                                'rstreng_pressure_remaining_life': 'RSTRENG Life (yrs)',
                                'rstreng_pressure_status': 'RSTRENG Status'
                            }
                            display_matched = display_matched.rename(columns=column_rename)                            

                            st.dataframe(display_matched, use_container_width=True, hide_index=True)
                        
                        # Show results for new defects (estimated growth)
                        new_results = enhanced_remaining_life_results['new_defects_analysis']
                        if new_results:
                            st.markdown("##### New Defects with Estimated Growth Rates")
                            new_df = pd.DataFrame(new_results)
                            
                            # Same column processing as above
                            display_cols = [
                                'log_dist', 'defect_type', 'joint_number', 'estimation_confidence',
                                'depth_based_remaining_life', 'depth_based_status',
                                'b31g_pressure_remaining_life', 'b31g_pressure_status',
                                'modified_b31g_pressure_remaining_life', 'modified_b31g_pressure_status',
                                'rstreng_pressure_remaining_life', 'rstreng_pressure_status'
                            ]
                            available_cols = [col for col in display_cols if col in new_df.columns]
                            
                            display_new = new_df[available_cols].copy()
                            
                            # Format remaining life columns
                            life_cols = [col for col in display_new.columns if 'remaining_life' in col]
                            for col in life_cols:
                                display_new[col] = display_new[col].apply(
                                    lambda x: f"{x:.1f}" if np.isfinite(x) else ("∞" if x == float('inf') else "Error")
                                )
                            
                            # Rename columns
                            column_rename = {
                                'log_dist': 'Location (m)',
                                'defect_type': 'Type', 
                                'joint_number': 'Joint',
                                'estimation_confidence': 'Confidence',
                                'depth_based_remaining_life': 'Depth Life (yrs)',
                                'depth_based_status': 'Depth Status',
                                'b31g_pressure_remaining_life': 'B31G Life (yrs)',
                                'b31g_pressure_status': 'B31G Status',
                                'modified_b31g_pressure_remaining_life': 'Mod-B31G Life (yrs)',
                                'modified_b31g_pressure_status': 'Mod-B31G Status',
                                'rstreng_pressure_remaining_life': 'RSTRENG Life (yrs)', 
                                'rstreng_pressure_status': 'RSTRENG Status'
                            }
                            display_new = display_new.rename(columns=column_rename)
                            
                            st.dataframe(display_new, use_container_width=True, hide_index=True)
                    
                    with enhanced_subtabs[2]:
                        st.markdown("#### Pipeline Overview")
                        st.info("Enhanced pipeline visualization showing depth-based remaining life (80% criterion)")
                        
                        try:
                            # Create a simple enhanced pipeline visualization
                            all_analyses = (enhanced_remaining_life_results['matched_defects_analysis'] + 
                                        enhanced_remaining_life_results['new_defects_analysis'])
                            
                            if all_analyses:
                                df = pd.DataFrame(all_analyses)
                                
                                # Create color mapping based on depth-based status
                                df_simp = df[df['depth_based_remaining_life'] < 100]

                                # Add pressure-based comparison option
                                st.markdown("##### Compare with Pressure-Based Analysis")
                                
                                pressure_method = st.selectbox(
                                    "Select pressure-based method to compare:",
                                    options=["b31g_pressure", "modified_b31g_pressure", "rstreng_pressure"],
                                    format_func=lambda x: {
                                        "b31g_pressure": "B31G Pressure-Based",
                                        "modified_b31g_pressure": "Modified B31G Pressure-Based", 
                                        "rstreng_pressure": "RSTRENG Pressure-Based"
                                    }[x],
                                    key="pressure_method_comparison"
                                )
                                
                                # Create comparison figure
                                fig_compare = go.Figure()
                                
                                # Add depth-based data
                                fig_compare.add_trace(go.Scatter(
                                    x=df_simp['log_dist'],
                                    y=df_simp['depth_based_remaining_life'].replace([np.inf], 100),  # Cap infinite at 100
                                    mode='markers',
                                    marker=dict(size=8, color='blue', opacity=0.7),
                                    name='Depth-Based (80%)',
                                    hovertemplate="<b>Location:</b> %{x:.2f}m<br><b>Depth Life:</b> %{y:.1f} years<extra></extra>"
                                ))
                                
                                # Add pressure-based data
                                pressure_col = f"{pressure_method}_remaining_life"
                                pressure_status_col = f"{pressure_method}_status"
                                
                                if pressure_col in df_simp.columns:
                                    pressure_data = df_simp[pressure_col].replace([np.inf], 100)  # Cap infinite at 100
                                    
                                    fig_compare.add_trace(go.Scatter(
                                        x=df_simp['log_dist'],
                                        y=pressure_data,
                                        mode='markers',
                                        marker=dict(size=8, color='red', opacity=0.7),
                                        name=pressure_method.replace('_', ' ').title(),
                                        hovertemplate="<b>Location:</b> %{x:.2f}m<br><b>Pressure Life:</b> %{y:.1f} years<extra></extra>"
                                    ))
                                
                                fig_compare.update_layout(
                                    title=f"Comparison: Depth-Based vs {pressure_method.replace('_', ' ').title()} Analysis",
                                    xaxis_title="Distance Along Pipeline (m)",
                                    yaxis_title="Remaining Life (Years, capped at 100)",
                                    height=500,
                                    hovermode='closest'
                                )
                                
                                st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': True})
                                
                                st.caption("Note: Infinite remaining life values are capped at 100 years for visualization.")
                            else:
                                st.info("No data available for pipeline visualization")
                                
                        except Exception as e:
                            st.error(f"Error creating pipeline visualization: {str(e)}")
                            st.info("Trying to display available data structure for debugging...")
                            if 'enhanced_remaining_life_results' in locals():
                                all_analyses = (enhanced_remaining_life_results.get('matched_defects_analysis', []) + 
                                            enhanced_remaining_life_results.get('new_defects_analysis', []))
                                if all_analyses:
                                    st.write("Available columns in analysis results:")
                                    st.write(list(all_analyses[0].keys()))

                else:
                    st.error("Enhanced remaining life analysis could not be performed.")
                    if 'error' in enhanced_remaining_life_results:
                        st.error(f"Error: {enhanced_remaining_life_results['error']}")
    
                    
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


    with viz_tabs[5]:
        st.subheader("🔮 Dynamic Clustering Simulation")
        st.info("""
        **Advanced Analysis**: This simulation projects defect growth forward in time and detects when 
        defects will start clustering according to FFS rules, potentially causing earlier failures 
        than individual analysis would predict.
        """)
        
        # Check if we have the required data
        if not comparison_results.get('has_depth_data', False):
            st.warning("Dynamic clustering simulation requires depth data from both years")
            return
        
        if comparison_results['matches_df'].empty:
            st.warning("No matched defects found for growth rate estimation")
            return
        
        # Initialize session state for dynamic clustering if not exists
        if 'dynamic_clustering_params' not in st.session_state:
            st.session_state.dynamic_clustering_params = {
                'max_sim_years': 20,
                'time_resolution': 0.5,
                'clustering_method': 'sqrt_dt',
                'earlier_year': earlier_year,
                'later_year': later_year
            }
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_sim_years = st.slider(
                "Simulation Years",
                min_value=5, max_value=50, value=st.session_state.dynamic_clustering_params['max_sim_years'],
                help="How many years into the future to simulate"
            )
        
        with col2:
            time_resolution = st.selectbox(
                "Time Resolution",
                options=[0.25, 0.5, 1.0],
                index=[0.25, 0.5, 1.0].index(st.session_state.dynamic_clustering_params['time_resolution']),
                format_func=lambda x: f"{x} years ({int(x*12)} months)",
                help="How often to check for clustering events"
            )
        
        with col3:
            clustering_method = st.selectbox(
                "Clustering Rule",
                options=['sqrt_dt', '3t'],
                index=['sqrt_dt', '3t'].index(st.session_state.dynamic_clustering_params['clustering_method']),
                format_func=lambda x: {
                    'sqrt_dt': '√(D×t) - Standard',
                    '3t': '3×t - Conservative'
                }[x]
            )
        
        # Check if parameters have changed
        params_changed = (
            max_sim_years != st.session_state.dynamic_clustering_params['max_sim_years'] or
            time_resolution != st.session_state.dynamic_clustering_params['time_resolution'] or
            clustering_method != st.session_state.dynamic_clustering_params['clustering_method'] or
            earlier_year != st.session_state.dynamic_clustering_params['earlier_year'] or
            later_year != st.session_state.dynamic_clustering_params['later_year']
        )
        
        # Update stored parameters
        new_params = {
            'max_sim_years': max_sim_years,
            'time_resolution': time_resolution,
            'clustering_method': clustering_method,
            'earlier_year': earlier_year,
            'later_year': later_year
        }
        
        # Show current status
        if hasattr(st.session_state, 'dynamic_clustering_results') and not params_changed:
            st.success("✅ Simulation results available (parameters unchanged)")
        elif params_changed and hasattr(st.session_state, 'dynamic_clustering_results'):
            st.warning("⚠️ Parameters changed - simulation needs to be re-run")
        
        # Button layout
        button_col1, button_col2, button_col3 = st.columns([2, 1, 1])
        
        with button_col1:
            run_simulation = st.button(
                "🚀 Run Dynamic Clustering Simulation" if not hasattr(st.session_state, 'dynamic_clustering_results') 
                else "🔄 Re-run Simulation with New Parameters" if params_changed 
                else "✅ Simulation Complete",
                use_container_width=True,
                disabled=(hasattr(st.session_state, 'dynamic_clustering_results') and not params_changed),
                type="primary" if not hasattr(st.session_state, 'dynamic_clustering_results') or params_changed else "secondary"
            )
        
        with button_col2:
            if hasattr(st.session_state, 'dynamic_clustering_results'):
                if st.button("📊 Refresh View", use_container_width=True):
                    # Just trigger a rerun to refresh the display
                    st.rerun()
        
        with button_col3:
            if hasattr(st.session_state, 'dynamic_clustering_results'):
                if st.button("🗑️ Clear Results", use_container_width=True):
                    del st.session_state.dynamic_clustering_results
                    st.rerun()
        
        # Run simulation if button pressed
        if run_simulation:
            # Update parameters in session state
            st.session_state.dynamic_clustering_params = new_params
            
            with st.spinner("Running time-forward simulation..."):
                try:
                    # Import the dynamic clustering analyzer
                    from core.dynamic_clustering_analysis import (
                        DynamicClusteringAnalyzer, FFSDefectInteraction
                    )
                    
                    # Get required data
                    current_defects = st.session_state.datasets[later_year]['defects_df'].copy()
                    joints_df = st.session_state.datasets[later_year]['joints_df']
                    pipe_diameter_mm = st.session_state.datasets[later_year]['pipe_diameter'] * 1000
                    
                    # Extract growth rates from comparison results
                    growth_rates_dict = {}
                    matches_df = comparison_results['matches_df']
                    
                    for idx, match in matches_df.iterrows():
                        defect_id = idx  # Use index as defect ID
                        growth_rates_dict[defect_id] = {
                            'depth_growth_pct_per_year': match.get('growth_rate_pct_per_year', 2.0),
                            'length_growth_mm_per_year': match.get('length_growth_rate_mm_per_year', 0.0),
                            'width_growth_mm_per_year': match.get('width_growth_rate_mm_per_year', 0.0)
                        }
                    
                    # For defects without matches, use conservative estimates
                    for i in range(len(current_defects)):
                        if i not in growth_rates_dict:
                            growth_rates_dict[i] = {
                                'depth_growth_pct_per_year': 2.0,  # Conservative default
                                'length_growth_mm_per_year': 3.0,
                                'width_growth_mm_per_year': 2.0
                            }
                    
                    # Add defect IDs to current defects
                    current_defects['defect_id'] = range(len(current_defects))
                    
                    # Initialize dynamic analyzer
                    ffs_rules = FFSDefectInteraction(
                        axial_interaction_distance_mm=25.4,
                        circumferential_interaction_method=clustering_method
                    )
                    
                    analyzer = DynamicClusteringAnalyzer(
                        ffs_rules=ffs_rules,
                        max_simulation_years=max_sim_years,
                        time_step_years=time_resolution,
                        depth_failure_threshold=80.0
                    )
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run simulation with progress updates
                    status_text.text("🔄 Initializing simulation...")
                    progress_bar.progress(10)
                    
                    status_text.text("🔄 Projecting defect growth...")
                    progress_bar.progress(30)
                    
                    simulation_results = analyzer.simulate_dynamic_clustering_failure(
                        current_defects, joints_df, growth_rates_dict, pipe_diameter_mm
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("🔄 Finalizing results...")
                    
                    # Store results with parameters
                    st.session_state.dynamic_clustering_results = {
                        'results': simulation_results,
                        'parameters': new_params,
                        'timestamp': datetime.now()
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Simulation completed!")
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Simulation completed successfully!")
                    st.rerun()  # Refresh to show results
                    
                except Exception as e:
                    st.error(f"Error in dynamic clustering simulation: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        # Display results if available
        if hasattr(st.session_state, 'dynamic_clustering_results'):
            # Show parameters used
            with st.expander("📋 Simulation Parameters", expanded=False):
                params = st.session_state.dynamic_clustering_params
                param_cols = st.columns(3)
                with param_cols[0]:
                    st.metric("Simulation Years", f"{params['max_sim_years']} years")
                with param_cols[1]:
                    st.metric("Time Resolution", f"{params['time_resolution']} years")
                with param_cols[2]:
                    st.metric("Clustering Method", params['clustering_method'].replace('_', ' ').upper())
                
                if 'timestamp' in st.session_state.dynamic_clustering_results:
                    st.caption(f"Last run: {st.session_state.dynamic_clustering_results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display the results
            display_dynamic_clustering_results(
                st.session_state.dynamic_clustering_results['results'], 
                earlier_year, 
                later_year
            )

def display_dynamic_clustering_results(simulation_results, earlier_year, later_year):
    """Display the results of the dynamic clustering simulation."""
    
    st.markdown("---")
    st.subheader("📊 Simulation Results")
    
    # Summary metrics
    summary = simulation_results['analysis_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        failure_time = simulation_results['earliest_failure_time']
        if failure_time == float('inf'):
            st.metric("Earliest Failure", "No failure predicted")
        else:
            st.metric("Earliest Failure", f"{failure_time:.1f} years")
    
    with col2:
        failure_mode = simulation_results['earliest_failure_mode']
        color = "🔴" if failure_mode == "clustering" else "🟡" if failure_mode == "individual" else "🟢"
        st.metric("Failure Mode", f"{color} {failure_mode.title()}")
    
    with col3:
        clustering_events = len(simulation_results['clustering_events'])
        st.metric("Clustering Events", clustering_events)
    
    with col4:
        benefit = summary.get('individual_vs_clustering_benefit', 0)
        if benefit > 0:
            st.metric("⚠️ Earlier Failure", f"{benefit:.1f} years", delta=f"-{benefit:.1f}")
        else:
            st.metric("Risk Assessment", "✅ No early failure")
    
    # Key insights
    if 'risk_insight' in summary:
        if simulation_results['earliest_failure_mode'] == 'clustering':
            st.error(f"🚨 **Critical Finding**: {summary['risk_insight']}")
        else:
            st.success(f"✅ **Good News**: {summary['risk_insight']}")
    
    # Detailed results in tabs
    result_tabs = st.tabs(["Timeline", "Clustering Events", "Individual vs Clustering", "Export"])
    
    with result_tabs[0]:
        st.subheader("📈 Simulation Timeline")
        
        if simulation_results['simulation_timeline']:
            timeline_df = pd.DataFrame(simulation_results['simulation_timeline'])
            
            # Create timeline plot
            import plotly.graph_objects as go
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timeline_df['time'],
                y=timeline_df['active_clusters'],
                mode='lines+markers',
                name='Active Clusters',
                line=dict(color='blue')
            ))
            
            # Add clustering events as vertical lines
            for event in simulation_results['clustering_events']:
                fig.add_vline(
                    x=event.year,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Cluster Event"
                )
            
            # Add failure time
            if simulation_results['earliest_failure_time'] != float('inf'):
                fig.add_vline(
                    x=simulation_results['earliest_failure_time'],
                    line_dash="solid",
                    line_color="red",
                    annotation_text="Predicted Failure"
                )
            
            fig.update_layout(
                title="Clustering Events Over Time",
                xaxis_title="Years from Now",
                yaxis_title="Number of Active Clusters",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available")
    
    with result_tabs[1]:
        st.subheader("🔗 Clustering Events")
        
        if simulation_results['clustering_events']:
            events_data = []
            for i, event in enumerate(simulation_results['clustering_events']):
                events_data.append({
                    'Event #': i + 1,
                    'Year': f"{event.year:.1f}",
                    'Defects Involved': len(event.defect_indices),
                    'Combined Depth (%)': f"{event.combined_defect_props['combined_depth_pct']:.1f}",
                    'Combined Length (mm)': f"{event.combined_defect_props['combined_length_mm']:.1f}",
                    'Estimated Failure Time': f"{event.failure_time:.1f} years after clustering"
                })
            
            events_df = pd.DataFrame(events_data)
            st.dataframe(events_df, use_container_width=True)
            
            # Detailed view of first event
            if len(simulation_results['clustering_events']) > 0:
                with st.expander("Detailed View: First Clustering Event"):
                    first_event = simulation_results['clustering_events'][0]
                    st.json(first_event.combined_defect_props)
        else:
            st.success("✅ No clustering events detected - defects remain individual")
    
    with result_tabs[2]:
        display_enhanced_clustering_analysis(simulation_results, earlier_year, later_year)
        
        # Compare individual failure times with clustering prediction
        individual_failures = simulation_results['individual_failure_times']
        
        if individual_failures:
            comparison_data = []
            earliest_individual = min(individual_failures.values()) if individual_failures else float('inf')
            clustering_failure = simulation_results['earliest_failure_time']
            
            for defect_id, individual_time in individual_failures.items():
                # Determine clustering impact
                if clustering_failure < individual_time:
                    impact = f"Fails {individual_time - clustering_failure:.1f} years earlier due to clustering"
                elif clustering_failure > individual_time:
                    impact = "Individual failure occurs first"
                else:
                    impact = "Same failure time"
                    
                comparison_data.append({
                    'Defect ID': defect_id,
                    'Individual Failure (years)': f"{individual_time:.1f}" if individual_time != float('inf') else "No failure",
                    'Clustering Impact': impact
                })
            
            # Fix the summary comparison metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Standard Analysis Prediction", 
                        f"{earliest_individual:.1f} years" if earliest_individual != float('inf') else "No failure")
            with col2:
                st.metric("Dynamic Clustering Prediction", 
                        f"{clustering_failure:.1f} years" if clustering_failure != float('inf') else "No failure")
    with result_tabs[3]:
        st.subheader("📥 Export Results")
        
        if st.button("Export Simulation Results"):
            # Prepare export data
            export_data = {
                'simulation_parameters': {
                    'max_years': simulation_results['analysis_summary']['total_simulation_years'],
                    'earlier_year': earlier_year,
                    'later_year': later_year
                },
                'results': simulation_results
            }
            
            import json
            json_str = json.dumps(export_data, default=str, indent=2)
            
            st.download_button(
                label="Download Simulation Results (JSON)",
                data=json_str,
                file_name=f"dynamic_clustering_simulation_{earlier_year}_{later_year}.json",
                mime="application/json"
            )
        
        st.info("""
        **Export includes:**
        - All clustering events with timing and properties
        - Individual vs clustering failure predictions  
        - Complete simulation timeline
        - Analysis parameters and settings
        """)

def render_comparison_view():
    """Display the multi-year comparison view with analysis across different years."""

    if len(st.session_state.datasets) < 2:
        st.info("""
            **Multiple datasets required**
            Please upload at least two datasets from different years to enable comparison.
            Use the sidebar to add more inspection data.
        """
        )
    else:
        # Year selection for comparison with improved UI
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
                            st.dataframe(comparison_results['matches_df'], use_container_width=True)
                        
                        if not comparison_results['new_defects'].empty:
                            st.markdown("<div class='section-header' style='margin-top:20px;'>New Defects</div>", unsafe_allow_html=True)
                            st.dataframe(comparison_results['new_defects'], use_container_width=True)

                
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
                        st.dataframe(comparison_results['matches_df'], use_container_width=True)
                    
                    if not comparison_results['new_defects'].empty:
                        st.markdown("<div class='section-header' style='margin-top:20px;'>New Defects</div>", unsafe_allow_html=True)
                        st.dataframe(comparison_results['new_defects'], use_container_width=True)
            else:
                # Years don't match, ask user to re-run comparison
                st.info("You've changed the years for comparison. Please click 'Compare Defects' to analyze the new year combination.")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close the card container


from core.ffs_defect_interaction import FFSDefectInteraction
from core.defect_matching import ClusterAwareDefectMatcher
from core.growth_analysis import ClusterAwareGrowthAnalyzer

def render_remaining_life_analysis_integrated(comparison_results, earlier_year, later_year):
    """
    Wrapper function with enhanced debugging and error handling.
    """
    try:
        # Check if we have the required data
        if not comparison_results.get('has_depth_data', False):
            st.warning("**Clustering-aware analysis requires depth data**")
            st.info("Please ensure both datasets have depth measurements.")
            return
        
        # Extract DataFrames from session state
        earlier_defects = st.session_state.datasets[earlier_year]['defects_df']
        later_defects = st.session_state.datasets[later_year]['defects_df']
        joints_df = st.session_state.datasets[later_year]['joints_df']
        
        # Get pipe diameter
        pipe_diameter_m = st.session_state.datasets[later_year]['pipe_diameter']
        pipe_diameter_mm = pipe_diameter_m * 1000
        
        # Add comprehensive debugging information
        st.write("🔍 **Debug Information:**")
        with st.expander("Data Validation Details", expanded=False):
            st.write(f"**Earlier year ({earlier_year}) data:**")
            st.write(f"  - Shape: {earlier_defects.shape}")
            st.write(f"  - Columns: {list(earlier_defects.columns)}")
            st.write(f"  - Sample data types:")
            for col in ['log dist. [m]', 'joint number', 'depth [%]', 'length [mm]', 'clock']:
                if col in earlier_defects.columns:
                    st.write(f"    - {col}: {earlier_defects[col].dtype}")
            
            st.write(f"**Later year ({later_year}) data:**")
            st.write(f"  - Shape: {later_defects.shape}")
            st.write(f"  - Columns: {list(later_defects.columns)}")
            
            st.write(f"**Joints data:**")
            st.write(f"  - Shape: {joints_df.shape}")
            st.write(f"  - Pipe diameter: {pipe_diameter_mm} mm")
        
        # Validate required columns
        required_cols = ['log dist. [m]', 'length [mm]', 'clock', 'joint number', 'depth [%]']
        missing_cols_early = [col for col in required_cols if col not in earlier_defects.columns]
        missing_cols_later = [col for col in required_cols if col not in later_defects.columns]
        
        if missing_cols_early or missing_cols_later:
            st.error(f"Missing required columns for clustering analysis:")
            if missing_cols_early:
                st.error(f"  {earlier_year} data: {missing_cols_early}")
            if missing_cols_later:
                st.error(f"  {later_year} data: {missing_cols_later}")
            return
        
        # Check for sufficient data
        if len(earlier_defects) == 0 or len(later_defects) == 0:
            st.warning("Insufficient defect data for clustering analysis")
            return
        
        # Step 1: User options for clustering
        st.subheader("📊 Clustering-Aware Remaining Life Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_clustering = st.checkbox(
                "Apply FFS Defect Clustering",
                value=True,
                help="Apply FFS interaction rules to both inspection years"
            )
        
        with col2:
            if use_clustering:
                clustering_method = st.selectbox(
                    "Clustering Method",
                    options=['sqrt_dt', '3t'],
                    format_func=lambda x: {
                        'sqrt_dt': '√(D×t) - Standard',
                        '3t': '3×t - Conservative'
                    }[x]
                )
            else:
                clustering_method = None
        
        # Step 2: Apply FFS clustering if selected
        if use_clustering:
            with st.spinner("Applying FFS clustering to inspection data..."):
                try:
                    from core.ffs_defect_interaction import FFSDefectInteraction
                    
                    # Initialize FFS analyzer
                    ffs_analyzer = FFSDefectInteraction(
                        axial_interaction_distance_mm=25.4,  # 1 inch
                        circumferential_interaction_method=clustering_method #  type: ignore
                    )
                    
                    # Apply clustering to both years
                    st.write(f"🔄 Clustering {earlier_year} defects...")
                    year1_clusters = ffs_analyzer.find_interacting_defects(
                        earlier_defects, joints_df, pipe_diameter_mm
                    )
                    
                    st.write(f"🔄 Clustering {later_year} defects...")
                    year2_clusters = ffs_analyzer.find_interacting_defects(
                        later_defects, joints_df, pipe_diameter_mm
                    )
                    
                    # Show clustering summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            f"{earlier_year} Clustering",
                            f"{len(year1_clusters)} groups",
                            f"from {len(earlier_defects)} defects"
                        )
                    with col2:
                        st.metric(
                            f"{later_year} Clustering", 
                            f"{len(year2_clusters)} groups",
                            f"from {len(later_defects)} defects"
                        )
                    
                    # Debug: Show cluster sizes
                    with st.expander("Cluster Details", expanded=False):
                        st.write(f"**{earlier_year} cluster sizes:** {[len(cluster) for cluster in year1_clusters]}")
                        st.write(f"**{later_year} cluster sizes:** {[len(cluster) for cluster in year2_clusters]}")
                        
                except Exception as e:
                    st.error(f"Error in FFS clustering: {str(e)}")
                    import traceback
                    with st.expander("Clustering Error Details"):
                        st.code(traceback.format_exc())
                    return
        else:
            # No clustering - each defect is its own group
            year1_clusters = [[i] for i in range(len(earlier_defects))]
            year2_clusters = [[i] for i in range(len(later_defects))]
        
        # Step 3: Match defects between years
        with st.spinner("Matching defects between inspection years..."):
            try:
                from core.defect_matching import ClusterAwareDefectMatcher
                
                matcher = ClusterAwareDefectMatcher(
                    max_axial_distance_mm=300.0,  # 30cm tolerance
                    max_clock_difference_hours=1.0,
                    pipe_diameter_mm=pipe_diameter_mm
                )
                
                st.write("🔄 Matching clusters between years...")
                matches = matcher.match_defects_with_clustering(
                    earlier_defects, later_defects,
                    year1_clusters, year2_clusters
                )
                
                # Show matching summary
                if matches:
                    match_types = {}
                    for m in matches:
                        match_types[m.match_type] = match_types.get(m.match_type, 0) + 1
                    
                    st.success(f"✅ Found {len(matches)} matches between years")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("1-to-1", match_types.get('1-to-1', 0))
                    with col2:
                        st.metric("Many-to-1", match_types.get('many-to-1', 0))
                    with col3:
                        st.metric("1-to-Many", match_types.get('1-to-many', 0))
                    with col4:
                        st.metric("Many-to-Many", match_types.get('many-to-many', 0))
                    
                    # Debug: Show first few matches
                    with st.expander("Match Details", expanded=False):
                        for i, match in enumerate(matches[:5]):
                            st.write(f"**Match {i+1}:** {match.match_type}")
                            st.write(f"  - Year1 indices: {match.year1_indices}")
                            st.write(f"  - Year2 indices: {match.year2_indices}")
                            st.write(f"  - Confidence: {match.match_confidence:.3f}")
                            
                else:
                    st.warning("No matches found between the two inspection years")
                    return
                    
            except Exception as e:
                st.error(f"Error in defect matching: {str(e)}")
                import traceback
                with st.expander("Matching Error Details"):
                    st.code(traceback.format_exc())
                return
        
        # Step 4: Analyze growth rates
        with st.spinner("Analyzing defect growth rates..."):
            try:
                from core.growth_analysis import ClusterAwareGrowthAnalyzer
                
                # Get wall thickness lookup
                wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
                
                # Initialize growth analyzer
                growth_analyzer = ClusterAwareGrowthAnalyzer(
                    negative_growth_strategy='similar_match'
                )
                
                # Parse dates
                import pandas as pd
                year1_date = pd.Timestamp(f'{earlier_year}-01-01')
                year2_date = pd.Timestamp(f'{later_year}-01-01')
                
                st.write("🔄 Starting growth analysis...")
                st.write(f"  - Time period: {(year2_date - year1_date).days / 365.25:.1f} years")
                st.write(f"  - Processing {len(matches)} matches...")
                
                # This is where the error was occurring - now fixed with better data extraction
                growth_df = growth_analyzer.analyze_growth_with_clustering(
                    earlier_defects, later_defects,
                    matches,
                    year1_date, year2_date,
                    wt_lookup
                )

                if growth_df.empty:
                    st.warning("No growth data could be calculated")
                    return
                
                # Check for errors in the results
                error_rows = growth_df[
                    growth_df.get('error', '').notna() & #  type: ignore
                    (growth_df.get('error', '') != '') & 
                    (growth_df.get('error', '').astype(str) != 'nan') # type: ignore
                ]
                if not error_rows.empty:
                    st.warning(f"⚠️ {len(error_rows)} matches had errors during analysis")
                    with st.expander("Growth Analysis Errors"):
                        st.dataframe(error_rows[['match_type', 'error']])
                
                # Filter out error rows for remaining life calculation
                valid_growth_df = growth_df[
                    growth_df.get('error', '').isna() |  # type: ignore
                    (growth_df.get('error', '') == '') |
                    (growth_df.get('error', '').astype(str) == 'nan') #  type: ignore
                ]
                if valid_growth_df.empty:
                    st.error("No valid growth data available after error filtering")
                    return
                    
                st.success(f"✅ Growth analysis completed. {len(valid_growth_df)} valid results.")
                
                # Calculate remaining life
                st.write("🔄 Calculating remaining life...")
                remaining_life_df = growth_analyzer.calculate_remaining_life(
                    valid_growth_df,
                    max_allowable_depth_pct=80.0
                )
                
            except Exception as e:
                st.error(f"Error in growth analysis: {str(e)}")
                import traceback
                with st.expander("Growth Analysis Error Details"):
                    st.code(traceback.format_exc())
                    
                    # Show debugging information
                    if 'matches' in locals() and matches:
                        st.write("**First match for debugging:**")
                        match = matches[0]
                        st.write(f"Match type: {match.match_type}")
                        st.write(f"Year1 indices: {match.year1_indices}")
                        st.write(f"Year2 indices: {match.year2_indices}")
                        
                        # Try to show the actual defect data
                        try:
                            if match.year1_indices:
                                y1_defect = earlier_defects.iloc[match.year1_indices[0]]
                                st.write("**Year1 defect data:**")
                                st.write(f"Type: {type(y1_defect)}")
                                if isinstance(y1_defect, pd.Series):
                                    st.write(f"Index: {y1_defect.index.tolist()}")
                                    st.write(f"Values: {y1_defect.to_dict()}")
                        except Exception as debug_e:
                            st.write(f"Could not show defect data: {debug_e}")
                return
        
        # Step 5: Display results
        st.subheader("📊 Growth Analysis Results")
        
        # Show summary statistics
        total_analyzed = len(remaining_life_df)
        critical_defects = remaining_life_df[
            remaining_life_df['safety_classification'].isin(['CRITICAL', 'HIGH PRIORITY'])
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyzed", total_analyzed)
        with col2:
            st.metric("Critical Defects", len(critical_defects))
        with col3:
            pct_critical = (len(critical_defects) / total_analyzed * 100) if total_analyzed > 0 else 0
            st.metric("% Critical", f"{pct_critical:.1f}%")
        
        if not critical_defects.empty:
            st.warning(f"⚠️ {len(critical_defects)} defects require immediate attention!")
            
            # Show critical defects table
            display_cols = ['location_m', 'year2_depth_pct', 'depth_growth_mm_per_year',
                           'remaining_life_years', 'safety_classification', 'growth_type']
            
            st.dataframe(
                critical_defects[display_cols].round(2),
                use_container_width=True
            )
        else:
            st.success("✅ No critical defects identified in this analysis")
        
        # Show full results table
        with st.expander("All Results", expanded=False):
            st.dataframe(remaining_life_df.round(3), use_container_width=True)
        
        # Export functionality
        if st.button("📥 Export Analysis Results"):
            csv = remaining_life_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"clustering_growth_analysis_{earlier_year}_{later_year}.csv",
                mime="text/csv"
            )
        
        return remaining_life_df
        
    except Exception as e:
        st.error(f"Error in clustering-aware analysis: {str(e)}")
        import traceback
        with st.expander("Main Error Details"):
            st.code(traceback.format_exc())


def render_remaining_life_analysis(defects_df_year1, defects_df_year2, 
                                  joints_df, pipe_params, assessment_params):
    """
    Perform remaining life analysis with FFS clustering consideration.
    
    FIXED: Add proper validation and error handling
    """
    
    # Validate inputs first
    if defects_df_year1 is None or defects_df_year2 is None:
        st.error("Missing defect data for analysis")
        return None
        
    if joints_df is None or joints_df.empty:
        st.error("Missing joint data for wall thickness lookup")
        return None
        
    if 'diameter_mm' not in pipe_params:
        st.error("Missing pipe diameter parameter")
        return None
    
    pipe_diameter_mm = pipe_params['diameter_mm']
    
    st.subheader("📊 Remaining Life Analysis with Clustering")
    
    # Step 1: User options for clustering
    col1, col2 = st.columns(2)
    
    with col1:
        use_clustering = st.checkbox(
            "Consider FFS Defect Clustering",
            value=True,
            help="Apply FFS interaction rules to both inspection years"
        )
    
    with col2:
        if use_clustering:
            clustering_method = st.selectbox(
                "Clustering Method",
                options=['sqrt_dt', '3t'],
                format_func=lambda x: {
                    'sqrt_dt': '√(D×t) - Standard',
                    '3t': '3×t - Conservative'
                }[x]
            )
        else:
            clustering_method = None
    
    # Step 2: Apply FFS clustering if selected
    if use_clustering:
        with st.spinner("Applying FFS clustering to inspection data..."):
            try:
                # Initialize FFS analyzer
                ffs_analyzer = FFSDefectInteraction(
                    axial_interaction_distance_mm=25.4,  # 1 inch
                    circumferential_interaction_method=clustering_method # type: ignore
                )
                
                # Apply clustering to both years
                year1_clusters = ffs_analyzer.find_interacting_defects(
                    defects_df_year1, joints_df, pipe_diameter_mm
                )
                year2_clusters = ffs_analyzer.find_interacting_defects(
                    defects_df_year2, joints_df, pipe_diameter_mm
                )
                
                # Show clustering summary
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        f"Year 1 Clustering",
                        f"{len(year1_clusters)} groups",
                        f"from {len(defects_df_year1)} defects"
                    )
                with col2:
                    st.metric(
                        f"Year 2 Clustering", 
                        f"{len(year2_clusters)} groups",
                        f"from {len(defects_df_year2)} defects"
                    )
                    
            except Exception as e:
                st.error(f"Error in FFS clustering: {str(e)}")
                return None
    else:
        # No clustering - each defect is its own group
        year1_clusters = [[i] for i in range(len(defects_df_year1))]
        year2_clusters = [[i] for i in range(len(defects_df_year2))]
    
    # Step 3: Match defects between years
    with st.spinner("Matching defects between inspection years..."):
        try:
            matcher = ClusterAwareDefectMatcher(
                max_axial_distance_mm=300.0,  # 30cm tolerance
                max_clock_difference_hours=1.0,
                pipe_diameter_mm=pipe_diameter_mm
            )
            
            matches = matcher.match_defects_with_clustering(
                defects_df_year1, defects_df_year2,
                year1_clusters, year2_clusters
            )
            
            # Show matching summary
            if matches:
                match_types = {}
                for m in matches:
                    match_types[m.match_type] = match_types.get(m.match_type, 0) + 1
                
                st.info(f"""
                **Defect Matching Results:**
                - Total matches found: {len(matches)}
                - Simple (1-to-1): {match_types.get('1-to-1', 0)}
                - Coalescence (many-to-1): {match_types.get('many-to-1', 0)}
                - Split (1-to-many): {match_types.get('1-to-many', 0)}
                - Complex (many-to-many): {match_types.get('many-to-many', 0)}
                """)
            else:
                st.warning("No matches found between the two inspection years")
                return None
                
        except Exception as e:
            st.error(f"Error in defect matching: {str(e)}")
            return None
    
    # Step 4: Analyze growth rates
    with st.spinner("Analyzing defect growth rates..."):
        try:
            # Get wall thickness lookup
            wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))

            # Initialize growth analyzer
            growth_analyzer = ClusterAwareGrowthAnalyzer(
                negative_growth_strategy='similar_match'
            )

            # Parse dates
            year1_date = pd.Timestamp(assessment_params['year1_date'])
            year2_date = pd.Timestamp(assessment_params['year2_date'])

            # Analyze growth
            growth_df = growth_analyzer.analyze_growth_with_clustering(
                defects_df_year1, defects_df_year2,
                matches,
                year1_date, year2_date,
                wt_lookup
            )

            if growth_df.empty:
                st.warning("No growth data could be calculated")
                return None

            # Calculate remaining life
            remaining_life_df = growth_analyzer.calculate_remaining_life(
                growth_df,
                max_allowable_depth_pct=80.0  # Standard limit
            )
            
        except Exception as e:
            st.error(f"Error in growth analysis: {str(e)}")
            return None
    
    # Step 5: Display results
    st.subheader("Growth Analysis Results")
    
    # Critical defects summary
    critical_defects = remaining_life_df[
        remaining_life_df['safety_classification'].isin(['CRITICAL', 'HIGH PRIORITY'])
    ]
    
    if not critical_defects.empty:
        st.warning(f"⚠️ {len(critical_defects)} defects require immediate attention!")
        
        # Show critical defects table
        st.dataframe(
            critical_defects[[
                'location_m', 'year2_depth_pct', 'depth_growth_mm_per_year',
                'remaining_life_years', 'safety_classification', 'growth_type'
            ]].round(2),
            use_container_width=True
        )
    else:
        st.success("✅ No critical defects identified in this analysis")
    
    # Export results
    if st.button("Export Detailed Growth Analysis"):
        # Prepare export data
        export_df = remaining_life_df.copy()
        
        # Add additional context
        export_df['clustering_applied'] = use_clustering
        export_df['clustering_method'] = clustering_method if use_clustering else 'None'
        
        # Convert to CSV
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Download Growth Analysis CSV",
            data=csv,
            file_name=f"growth_analysis_clustered_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    return remaining_life_df


def create_enhanced_clustering_analysis_table(simulation_results, defects_df):
    """
    Create an enhanced table showing clustering events and their impacts.
    
    Parameters:
    - simulation_results: Results from dynamic clustering simulation
    - defects_df: Original defects dataframe for additional context
    
    Returns:
    - pandas.DataFrame with enhanced clustering analysis
    """
    clustering_events = simulation_results['clustering_events']
    individual_failures = simulation_results['individual_failure_times']
    
    if not clustering_events:
        return pd.DataFrame([{
            'Analysis': 'No clustering events detected',
            'Details': 'All defects remain individual throughout simulation period'
        }])
    
    enhanced_data = []
    
    for i, event in enumerate(clustering_events):
        # Get involved defects information
        involved_defects = defects_df.iloc[event.defect_indices] if hasattr(defects_df, 'iloc') else defects_df.loc[event.defect_indices]
        
        # Location information
        locations = involved_defects['log dist. [m]'].values
        location_range = f"{locations.min():.1f} - {locations.max():.1f}m"
        center_location = f"{locations.mean():.1f}m"
        
        # Defect types involved
        if 'component / anomaly identification' in involved_defects.columns:
            defect_types = involved_defects['component / anomaly identification'].value_counts()
            # Show top 2 most common types
            main_types = defect_types.head(2)
            type_summary = ", ".join([f"{count}x {dtype}" for dtype, count in main_types.items()])
        else:
            type_summary = "Type info not available"
        
        # Depth information
        if 'depth [%]' in involved_defects.columns:
            depths = involved_defects['depth [%]'].values
            depth_info = f"{depths.min():.1f}% - {depths.max():.1f}% (max: {depths.max():.1f}%)"
        else:
            depth_info = "Depth info not available"
        
        # Individual failure times for involved defects
        individual_times = []
        for defect_idx in event.defect_indices:
            defect_id = defect_idx  # Assuming defect_id matches index
            individual_time = individual_failures.get(defect_id, float('inf'))
            if individual_time != float('inf'):
                individual_times.append(individual_time)
        
        # Calculate impact
        if individual_times:
            earliest_individual = min(individual_times)
            cluster_failure_time = event.year + event.failure_time
            
            if cluster_failure_time < earliest_individual:
                impact = f"⚠️ {earliest_individual - cluster_failure_time:.1f} years earlier"
                impact_type = "Accelerated Failure"
            else:
                impact = f"✅ {cluster_failure_time - earliest_individual:.1f} years later"
                impact_type = "Delayed Failure"
        else:
            impact = "No individual failures predicted"
            impact_type = "New Failure Mode"
        
        # Combined defect properties
        combined_props = event.combined_defect_props
        
        enhanced_data.append({
            'Cluster #': f"C{i+1}",
            'Formation Time': f"Year {event.year:.1f}",
            'Defects Involved': f"{len(event.defect_indices)} defects",
            'Location Range': location_range,
            'Center Location': center_location,
            'Primary Types': type_summary,
            'Depth Range': depth_info,
            'Combined Depth': f"{combined_props.get('combined_depth_pct', 0):.1f}%",
            'Combined Length': f"{combined_props.get('combined_length_mm', 0):.1f}mm",
            'Individual Failure': f"{min(individual_times):.1f}y" if individual_times else "None",
            'Cluster Failure': f"{event.year + event.failure_time:.1f}y",
            'Impact': impact,
            'Impact Type': impact_type
        })
    
    return pd.DataFrame(enhanced_data)


def create_cluster_defect_mapping_table(simulation_results, defects_df):
    """
    Create a detailed mapping showing which specific defects are in each cluster.
    
    Parameters:
    - simulation_results: Results from dynamic clustering simulation
    - defects_df: Original defects dataframe
    
    Returns:
    - pandas.DataFrame showing defect-to-cluster mapping
    """
    clustering_events = simulation_results['clustering_events']
    
    if not clustering_events:
        return pd.DataFrame()
    
    mapping_data = []
    
    for i, event in enumerate(clustering_events):
        cluster_id = f"C{i+1}"
        
        for defect_idx in event.defect_indices:
            if hasattr(defects_df, 'iloc'):
                defect = defects_df.iloc[defect_idx]
            else:
                defect = defects_df.loc[defect_idx]
            
            mapping_data.append({
                'Cluster ID': cluster_id,
                'Defect Index': defect_idx,
                'Location (m)': f"{defect['log dist. [m]']:.2f}",
                'Depth (%)': f"{defect.get('depth [%]', 0):.1f}",
                'Length (mm)': f"{defect.get('length [mm]', 0):.1f}",
                'Width (mm)': f"{defect.get('width [mm]', 0):.1f}",
                'Type': defect.get('component / anomaly identification', 'Unknown'),
                'Joint': defect.get('joint number', 'Unknown')
            })
    
    return pd.DataFrame(mapping_data)


# Modified section for display_dynamic_clustering_results function
def display_enhanced_clustering_analysis(simulation_results, earlier_year, later_year):
    """
    Enhanced version of the Individual vs Clustering Analysis tab.
    """
    
    st.subheader("🔗 Enhanced Clustering Impact Analysis")
    
    # Get the current defects dataframe for context
    current_defects = st.session_state.datasets[later_year]['defects_df']
    
    # Create enhanced analysis table
    enhanced_table = create_enhanced_clustering_analysis_table(simulation_results, current_defects)
    
    if not enhanced_table.empty and 'Cluster #' in enhanced_table.columns:
        st.markdown("#### 📊 Clustering Events Summary")
        
        # Key metrics
        total_clusters = len(enhanced_table)
        accelerated_failures = len(enhanced_table[enhanced_table['Impact Type'] == 'Accelerated Failure'])
        total_defects_clustered = enhanced_table['Defects Involved'].str.extract('(\d+)').astype(int).sum().iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clustering Events", total_clusters)
        with col2:
            st.metric("Accelerated Failures", accelerated_failures)
        with col3:
            st.metric("Total Defects Clustered", total_defects_clustered)
        
        # Main analysis table
        st.markdown("#### 📋 Detailed Clustering Analysis")
        st.dataframe(enhanced_table, use_container_width=True)
        
        # Detailed defect mapping (expandable)
        with st.expander("🔍 Detailed Defect-to-Cluster Mapping", expanded=False):
            mapping_table = create_cluster_defect_mapping_table(simulation_results, current_defects)
            if not mapping_table.empty:
                st.dataframe(mapping_table, use_container_width=True)
            else:
                st.info("No clustering events to map")
        
        # Analysis insights
        if accelerated_failures > 0:
            st.warning(f"⚠️ **Critical Finding**: {accelerated_failures} clustering event(s) cause earlier failures than individual analysis predicted!")
            
            # Show worst case
            worst_case = enhanced_table[enhanced_table['Impact Type'] == 'Accelerated Failure'].copy()
            if not worst_case.empty:
                # Extract years from impact string
                worst_case['Impact_Years'] = worst_case['Impact'].str.extract('(\d+\.?\d*)').astype(float)
                worst_cluster = worst_case.loc[worst_case['Impact_Years'].idxmax()]
                
                st.error(f"""
                **Most Critical Cluster**: {worst_cluster['Cluster #']}
                - **Location**: {worst_cluster['Center Location']}
                - **Formation Time**: {worst_cluster['Formation Time']}
                - **Impact**: Failure occurs {worst_cluster['Impact']}
                - **Defects Involved**: {worst_cluster['Defects Involved']}
                """)
        else:
            st.success("✅ **Good News**: No clustering events cause earlier failures than individual analysis")
            
    else:
        st.info("No clustering events detected during the simulation period")
        
        # Still show summary of individual analysis
        individual_failures = simulation_results['individual_failure_times']
        if individual_failures:
            earliest_individual = min(individual_failures.values())
            st.info(f"**Individual Analysis Result**: Earliest predicted failure in {earliest_individual:.1f} years")