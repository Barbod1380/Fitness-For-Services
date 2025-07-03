# app/views/failure_prediction.py - Enhanced with joint failure visualization

import streamlit as st
import pandas as pd
from app.ui_components.charts import create_metrics_row
from app.services.state_manager import get_state
from app.ui_components.ui_elements import info_box

# Import the analysis and visualization functions
from analysis.failure_prediction import predict_joint_failures_over_time
from visualization.failure_prediction_viz import (
    create_failure_prediction_chart,
    create_failure_summary_metrics, 
    create_failure_details_table,
    create_failure_comparison_chart
)

# NEW: Import joint failure visualization functions
from visualization.joint_failure_viz import (
    create_joint_failure_visualization,
    create_failure_summary_card,
    create_joint_failure_timeline_chart
)

def render_failure_prediction_view():
    """Display the failure prediction view with timeline analysis and joint visualization."""

    # Check if datasets are available
    datasets = get_state('datasets', {})
    if not datasets:
        st.info("""
            **No datasets available**
            Please upload pipeline inspection data using the sidebar to enable failure prediction.
        """)
        return
    
    # Dataset selection and analysis mode
    st.markdown("<div class='section-header'>Analysis Configuration</div>", unsafe_allow_html=True)
    
    # Determine analysis mode based on available datasets
    available_years = sorted(datasets.keys())
    
    if len(available_years) == 1:
        analysis_mode = "single_file"
        selected_year = available_years[0]
        st.info(f"**Single File Mode**: Using {selected_year} data with estimated growth rates")
    else:
        analysis_mode = st.radio(
            "Analysis Mode",
            ["single_file", "multi_year"],
            format_func=lambda x: {
                "single_file": "Single File (Estimate Growth Rates)",
                "multi_year": "Multi-Year (Use Measured Growth Rates)"
            }[x],
            help="Choose whether to estimate growth rates or use measured rates from comparison"
        )
        
        if analysis_mode == "single_file":
            selected_year = st.selectbox(
                "Select Dataset for Analysis",
                options=available_years,
                key="failure_prediction_year"
            )
        else:
            # For multi-year, check if comparison results exist
            comparison_results = get_state('comparison_results')
            if comparison_results is None:
                st.warning("""
                    **Multi-year analysis requires comparison results**
                    Please run Multi-Year Comparison first to calculate growth rates.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                return
            else:
                comparison_years = get_state('comparison_years')
                if comparison_years:
                    selected_year = comparison_years[1]  # Use later year
                    st.success(f"Using growth rates from {comparison_years[0]} → {comparison_years[1]} comparison")
                else:
                    st.error("Invalid comparison results. Please re-run Multi-Year Comparison.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close config container
    
    # Parameter input section
    st.markdown("<div class='section-header'>Prediction Parameters</div>", unsafe_allow_html=True)
    
    # Create parameter input columns
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    
    with param_col1:
        window_years = st.number_input(
            "Prediction Window (years)",
            min_value=1,
            max_value=50,
            value=15,
            step=1,
            help="Number of years to predict into the future"
        )
    
    with param_col2:
        operating_pressure_mpa = st.number_input(
            "Operating Pressure (MPa)",
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.1,
            format="%.3f",
            help="Current/planned operating pressure for ERF calculation"
        )
        st.caption(f"= {operating_pressure_mpa * 145.038:.0f} psi")
    
    with param_col3:
        assessment_method = st.selectbox(
            "Assessment Method",
            options=["b31g", "modified_b31g", "simplified_eff_area"],
            format_func=lambda x: {
                "b31g": "B31G Original",
                "modified_b31g": "Modified B31G",
                "simplified_eff_area": "RSTRENG (Simplified)"
            }[x],
            help="Method for calculating failure pressure"
        )
    
    with param_col4:
        if analysis_mode == "single_file":
            pipe_creation_year = st.number_input(
                "Pipe Creation Year",
                min_value=1950,
                max_value=2030,
                value=max(2000, selected_year - 10),
                step=1,
                help="Year the pipeline was installed"
            )
            current_year = selected_year
        else:
            pipe_creation_year = None
            current_year = None
    
    # Additional parameters in expandable section
    with st.expander("Advanced Parameters"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            # Get pipe diameter from stored data
            stored_diameter = datasets[selected_year].get('pipe_diameter', 1.0)
            pipe_diameter_m = st.number_input(
                "Pipe Diameter (m)",
                min_value=0.1,
                max_value=3.0,
                value=stored_diameter,
                step=0.1,
                format="%.3f"
            )
        
        with adv_col2:
            pipe_grade = st.selectbox(
                "Pipe Grade",
                options=["API 5L X42", "API 5L X52", "API 5L X60", "API 5L X65", "API 5L X70", "Custom"],
                index=1
            )
            
            grade_to_smys = {
                "API 5L X42": 290, "API 5L X52": 358, "API 5L X60": 413,
                "API 5L X65": 448, "API 5L X70": 482
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
                    step=1.0
                )
        
        with adv_col3:
            safety_factor = st.number_input(
                "Safety Factor",
                min_value=1.0,
                max_value=2.0,
                value=1.39,
                step=0.01,
                format="%.3f",
                help="Safety factor for failure pressure calculation"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close parameters container    

    # Validation and run button
    valid_config = True
    validation_messages = []
    
    if analysis_mode == "single_file" and pipe_creation_year >= selected_year:
        valid_config = False
        validation_messages.append("Pipe creation year must be before inspection year")
    
    if operating_pressure_mpa <= 0:
        valid_config = False
        validation_messages.append("Operating pressure must be positive")
    
    if validation_messages:
        for msg in validation_messages:
            st.error(msg)
    
    # Run analysis button
    run_analysis = st.button(
        "🚀 Run Failure Prediction Analysis",
        disabled=not valid_config,
        use_container_width=True,
        type="primary"
    )
    
    if run_analysis:
        with st.spinner("Analyzing future joint failures..."):
            try:
                # Get required data
                defects_df = datasets[selected_year]['defects_df']
                joints_df = datasets[selected_year]['joints_df']
                pipe_diameter_mm = pipe_diameter_m * 1000
                
                # Prepare growth rates
                growth_rates_dict = None
                if analysis_mode == "multi_year":
                    comparison_results = get_state('comparison_results')
                    if comparison_results and 'matches_df' in comparison_results:
                        matches_df = comparison_results['matches_df']
                        growth_rates_dict = {}
                        
                        for idx, match in matches_df.iterrows():
                            # Map matches to defect indices in the later year dataset
                            new_defect_id = match.get('new_defect_id', idx)
                            growth_rates_dict[new_defect_id] = {
                                'depth_growth_pct_per_year': match.get('growth_rate_pct_per_year', 2.0),
                                'length_growth_mm_per_year': match.get('length_growth_rate_mm_per_year', 3.0),
                                'width_growth_mm_per_year': match.get('width_growth_rate_mm_per_year', 2.0)
                            }
                
                # Run enhanced prediction analysis
                results = predict_joint_failures_over_time(
                    defects_df=defects_df,
                    joints_df=joints_df,
                    pipe_diameter_mm=pipe_diameter_mm,
                    smys_mpa=smys_mpa,
                    operating_pressure_mpa=operating_pressure_mpa,
                    assessment_method=assessment_method,
                    window_years=window_years,
                    safety_factor=safety_factor,
                    growth_rates_dict=growth_rates_dict,    # type: ignore
                    pipe_creation_year=pipe_creation_year,  # type: ignore
                    current_year=current_year               # type: ignore
                )
                
                # Store results in session state
                st.session_state.failure_prediction_results = results
                st.session_state.failure_prediction_config = {
                    'analysis_mode': analysis_mode,
                    'selected_year': selected_year,
                    'assessment_method': assessment_method,
                    'operating_pressure_mpa': operating_pressure_mpa,
                    'window_years': window_years,
                    'pipe_diameter_m': pipe_diameter_m
                }
                
                st.success("✅ Failure prediction analysis completed!")
                st.rerun()  # Refresh to show results
                
            except Exception as e:
                st.error(f"Error during failure prediction analysis: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close analysis container
    
    # Results display
    if hasattr(st.session_state, 'failure_prediction_results'):
        display_failure_prediction_results()


def display_failure_prediction_results():
    """Display the failure prediction results with charts, tables, and joint visualization."""
    
    results = st.session_state.failure_prediction_results
    config = st.session_state.failure_prediction_config
    
    # Summary metrics
    st.markdown("<div class='section-header'>📊 Prediction Summary</div>", unsafe_allow_html=True)
    
    # Create and display metrics
    metrics_data = create_failure_summary_metrics(results)
    metrics_list = [
        (metrics_data['total_joints']['label'], metrics_data['total_joints']['value'], 
         metrics_data['total_joints']['description']),
        (metrics_data['joints_with_defects']['label'], metrics_data['joints_with_defects']['value'],
         metrics_data['joints_with_defects']['description']),
        (metrics_data['max_erf_failures']['label'], metrics_data['max_erf_failures']['value'],
         metrics_data['max_erf_failures']['description']),
        (metrics_data['first_failure_year']['label'], metrics_data['first_failure_year']['value'],
         metrics_data['first_failure_year']['description'])
    ]
    
    create_metrics_row(metrics_list)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main visualization
    st.markdown("<div class='section-header'>📈 Failure Timeline</div>", unsafe_allow_html=True)
    
    # Create and display main chart
    main_chart = create_failure_prediction_chart(results)
    st.plotly_chart(main_chart, use_container_width=True, config={'displayModeBar': True})
    
    # Chart explanation
    st.markdown("""
    **Chart Explanation:**
    - **Bars**: Annual failures by type (ERF < 1.0 vs Depth > 80%)
    - **Lines**: Cumulative failures over time
    - **ERF Failures**: Joints where operating pressure exceeds safe capacity
    - **Depth Failures**: Joints where any defect exceeds 80% wall thickness
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # NEW: Joint Failure Visualization Section
    failing_joints = results.get('failing_joints_summary', [])
    
    if failing_joints:
        st.markdown("<div class='section-header'>🔍 Joint Failure Analysis</div>", unsafe_allow_html=True)
        
        # Joint failure timeline chart
        if len(failing_joints) > 1:
            st.markdown("#### Failure Timeline Overview")
            timeline_chart = create_joint_failure_timeline_chart(results['joint_failure_timeline'])
            st.plotly_chart(timeline_chart, use_container_width=True)
        
        # Joint selection interface
        st.markdown("#### Detailed Joint Visualization")
        
        # Create selection columns
        select_col1, select_col2 = st.columns([3, 1])
        
        with select_col1:
            # Prepare joint options sorted by failure year
            joint_options = []
            for joint_info in failing_joints:
                joint_num = joint_info['joint_number']
                failure_year = joint_info['failure_year']
                failure_mode = joint_info['failure_mode']
                location = joint_info['location_m']
                defect_count = joint_info['defect_count']
                
                label = (f"Joint {joint_num} (Year {failure_year}) - "
                        f"{failure_mode} @ {location:.1f}m - "
                        f"{defect_count} defects")
                joint_options.append((label, joint_num))
            
            if joint_options:
                selected_joint_label, selected_joint_num = st.selectbox(
                    "Select a failing joint to visualize:",
                    options=joint_options,
                    format_func=lambda x: x[0],
                    key="joint_failure_selector"
                )
        
        with select_col2:
            show_visualization = st.button(
                "🔍 Visualize Joint",
                use_container_width=True,
                type="secondary"
            )
        
        # Display joint visualization if requested
        if show_visualization and 'selected_joint_num' in locals():
            # Get joint failure information
            joint_timeline = results['joint_failure_timeline'].get(selected_joint_num, {})
            
            if joint_timeline:
                # Get the failure information (first year it fails)
                first_failure_year = min(joint_timeline.keys())
                joint_failure_info = joint_timeline[first_failure_year]                

                # Create and display the before/after visualization
                st.markdown("#### Before/After Comparison")
                
                try:
                    joint_viz = create_joint_failure_visualization(
                        joint_failure_info, 
                        config['pipe_diameter_m'] * 1000  # Convert to mm
                    )
                    st.plotly_chart(joint_viz, use_container_width=True)
                    
                    # Add explanation
                    st.info("""
                    **Visualization Guide:**
                    - 🔴 **Red defects** are the ones that cause failure
                    - **Color intensity** indicates defect depth severity
                    - **Blue arrows** show growth direction and magnitude
                    - **Left side** shows current state, **right side** shows projected state at failure
                    """)
                    
                except Exception as e:
                    st.error(f"Error creating joint visualization: {str(e)}")
                    with st.expander("Debug Information"):
                        st.write("Joint failure info:", joint_failure_info)
            else:
                st.warning(f"No detailed failure information available for Joint {selected_joint_num}")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close joint analysis container
    
    # Detailed results tabs (existing functionality)
    result_tabs = st.tabs(["Failure Details", "Analysis Settings", "Export Results"])
    
    with result_tabs[0]:
        st.markdown("#### First 5 Years - Detailed Failures")
        
        details_table = create_failure_details_table(results, max_year=5)
        if not details_table.empty:
            st.dataframe(details_table, use_container_width=True, hide_index=True)
        else:
            st.info("No failures predicted in the first 5 years.")
        
        # Show summary statistics
        summary = results['summary']
        
        if summary['first_erf_failure_year'] or summary['first_depth_failure_year']:
            st.markdown("#### Key Insights")
            
            insights = []
            if summary['first_erf_failure_year']:
                insights.append(f"🔴 **First ERF failure** predicted in year {summary['first_erf_failure_year']}")
            if summary['first_depth_failure_year']:
                insights.append(f"🔵 **First depth failure** predicted in year {summary['first_depth_failure_year']}")
            
            insights.append(f"📈 **Peak ERF failures**: {summary['max_erf_failures']} joints ({summary['pct_erf_failures']:.1f}%)")
            insights.append(f"📈 **Peak depth failures**: {summary['max_depth_failures']} joints ({summary['pct_depth_failures']:.1f}%)")
            
            # NEW: Add failing joints insight
            total_failing = len(results.get('failing_joints_summary', []))
            insights.append(f"⚠️ **Total joints that will fail**: {total_failing} out of {summary['total_joints_analyzed']} ({total_failing/summary['total_joints_analyzed']*100:.1f}%)")
            
            for insight in insights:
                st.markdown(insight)
        else:
            st.success("🎉 **No failures predicted** within the analysis window!")
    
    with result_tabs[1]:
        st.markdown("#### Analysis Configuration")
        
        config_data = {
            'Parameter': [
                'Analysis Mode',
                'Dataset Year', 
                'Assessment Method',
                'Operating Pressure',
                'Prediction Window',
                'Pipe Creation Year',
                'Total Joints Analyzed',
                'Joints with Defects',
                'Pipe Diameter'  # NEW
            ],
            'Value': [
                config['analysis_mode'].replace('_', ' ').title(),
                str(config['selected_year']),
                config['assessment_method'].replace('_', ' ').title(),
                f"{config['operating_pressure_mpa']:.1f} MPa ({config['operating_pressure_mpa'] * 145.038:.0f} psi)",
                f"{config['window_years']} years",
                str(st.session_state.failure_prediction_config.get('pipe_creation_year', 'N/A')),
                f"{results['total_joints']:,}",
                f"{results['joints_with_defects']:,}",
                f"{config['pipe_diameter_m']:.2f} m"  # NEW
            ]
        }
        
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        # Methodology notes
        st.markdown("#### Methodology Notes")
        st.markdown("""
        **Failure Criteria:**
        - **ERF < 1.0**: Operating pressure exceeds failure pressure capacity
        - **Depth > 80%**: Any defect exceeds 80% of wall thickness
        - **Joint failure**: Occurs when ANY defect in the joint fails
        
        **Growth Rate Estimation** (Single File Mode):
        - Assumes defects started at minimal detectable size
        - Estimates growth based on pipe age and current defect size
        - Applies conservative minimum growth rates
        
        **Dimensional Growth**:
        - All three dimensions (depth, length, width) grow over time
        - Larger defects result in lower failure pressures
        - Conservative approach uses worst-case scenarios
        
        **Joint Visualization**:
        - Shows before/after states of failing joints
        - Highlights specific defects that cause failure
        - Displays growth progression with arrows and metrics
        """)
    
    with result_tabs[2]:
        st.markdown("#### Export Options")
        
        # Prepare export data
        export_data = {
            'analysis_config': config,
            'summary_metrics': results['summary'],
            'annual_failures': {
                'years': results['years'],
                'erf_failures': results['erf_failures_by_year'],
                'depth_failures': results['depth_failures_by_year'],
                'cumulative_erf': results['cumulative_erf_failures'],
                'cumulative_depth': results['cumulative_depth_failures']
            }
        }
        
        # Convert to CSV format
        export_df = pd.DataFrame({
            'Year': results['years'],
            'ERF_Failures_Annual': results['erf_failures_by_year'],
            'Depth_Failures_Annual': results['depth_failures_by_year'],
            'ERF_Failures_Cumulative': results['cumulative_erf_failures'],
            'Depth_Failures_Cumulative': results['cumulative_depth_failures']
        })
        
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Failure Timeline CSV",
            data=csv,
            file_name=f"failure_prediction_{config['selected_year']}_{config['assessment_method']}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Also export detailed failures
        details_df = create_failure_details_table(results, max_year=config['window_years'])
        if not details_df.empty:
            details_csv = details_df.to_csv(index=False)
            st.download_button(
                label="📋 Download Detailed Failures CSV", 
                data=details_csv,
                file_name=f"detailed_failures_{config['selected_year']}_{config['assessment_method']}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # NEW: Export failing joints summary
        if results.get('failing_joints_summary'):
            failing_joints_df = pd.DataFrame(results['failing_joints_summary'])
            failing_joints_csv = failing_joints_df.to_csv(index=False)
            st.download_button(
                label="🔍 Download Failing Joints Summary CSV",
                data=failing_joints_csv,
                file_name=f"failing_joints_{config['selected_year']}_{config['assessment_method']}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.info("""
        **Export Information:**
        - Timeline CSV contains annual and cumulative failure counts
        - Detailed CSV contains specific joint and defect failure information
        - Failing Joints CSV contains summary of all joints that will fail
        - Data can be used for further analysis or reporting
        """)