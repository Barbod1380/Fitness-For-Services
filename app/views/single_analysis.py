"""
Single year analysis view for the Pipeline Analysis application.
"""
import streamlit as st
import pandas as pd

from app.components import custom_metric, info_box, create_data_download_links
from app.ui_components.charts import create_metrics_row

from analysis.defect_analysis import (
    create_dimension_distribution_plots, create_dimension_statistics_table,
    create_combined_dimensions_plot, create_joint_summary
)


from visualization.pipeline_viz import create_unwrapped_pipeline_visualization
from visualization.joint_viz import create_joint_defect_visualization

def render_single_analysis_view():
    """Display the single year analysis view with various analysis options."""
    st.markdown('<h2 class="section-header">Single Year Analysis</h2>', unsafe_allow_html=True)
    
    # Select year to analyze
    years = sorted(st.session_state.datasets.keys())
    col1, col2 = st.columns([2, 2])
    
    with col1:
        selected_analysis_year = st.selectbox(
            "Select Year to Analyze",
            options=years,
            index=years.index(st.session_state.current_year) if st.session_state.current_year in years else 0,
            key="year_selector_single_analysis"
        )
    
    # Get the selected dataset
    joints_df = st.session_state.datasets[selected_analysis_year]['joints_df']
    defects_df = st.session_state.datasets[selected_analysis_year]['defects_df']
    
    # Display dataset summary in custom metrics
    with col2:
        metrics_data = [
            ("Joints", f"{len(joints_df)}", None),
            ("Defects", f"{len(defects_df)}", None),
            ("Max Depth", f"{defects_df['depth [%]'].max():.1f}%" if 'depth [%]' in defects_df.columns else "N/A", None)
        ]
        create_metrics_row(metrics_data)

    
    # Create tabs for different analysis types within a card container
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    analysis_tabs = st.tabs(["Data Preview", "Defect Dimensions", "Visualizations"])
    
    # Tab 1: Data Preview
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_analysis_year} Joints")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(joints_df.head(5), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add download buttons
            href = create_data_download_links(joints_df, "joints", selected_analysis_year)
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            st.subheader(f"{selected_analysis_year} Defects")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(defects_df.head(5), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add download buttons 
            href = create_data_download_links(defects_df, "defects", selected_analysis_year)
            st.markdown(href, unsafe_allow_html=True)
    
    # Tab 2: Defect Dimensions Analysis
    with analysis_tabs[1]:
        st.subheader("Defect Dimension Analysis")
        
        # Display dimension statistics table
        st.markdown("<div class='section-header'>Dimension Statistics</div>", unsafe_allow_html=True)
        stats_df = create_dimension_statistics_table(defects_df)
        if not stats_df.empty:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(stats_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            info_box("No dimension data available for analysis.", "warning")
        
        # Create distribution plots
        dimension_figs = create_dimension_distribution_plots(defects_df)
        
        if dimension_figs:
            st.markdown("<div class='section-header' style='margin-top:20px;'>Dimension Distributions</div>", unsafe_allow_html=True)
            # Create columns for the plots
            cols = st.columns(min(len(dimension_figs), 3))
            
            # Display each dimension distribution
            for i, (col_name, fig) in enumerate(dimension_figs.items()):
                with cols[i % len(cols)]:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Display combined dimensions plot
            st.markdown("<div class='section-header' style='margin-top:20px;'>Defect Dimensions Relationship</div>", unsafe_allow_html=True)
            combined_fig = create_combined_dimensions_plot(defects_df)
            st.plotly_chart(combined_fig, use_container_width=True)
        else:
            info_box("No dimension data available for plotting distributions.", "warning")
    
    # Tab 3: Pipeline Visualizations
    with analysis_tabs[2]:
        st.subheader("Pipeline Visualization")
        
        # Visualization type selection with improved UI
        viz_col1, viz_col2 = st.columns([2, 2])
        
        with viz_col1:
            viz_type = st.radio(
                "Select Visualization Type",
                ["Complete Pipeline", "Joint-by-Joint"],
                horizontal=True,
                key="viz_type_single_analysis"
            )

        # Complete Pipeline visualization section
        if viz_type == "Complete Pipeline":
            # Add filtering options
            with st.expander("Filter Defects", expanded=False):
                st.subheader("Filter Defects by Properties")
                
                # Create columns for different filter types
                filter_col1, filter_col2 = st.columns(2)
                
                # Initialize filter variables
                apply_depth_filter = False
                apply_length_filter = False
                apply_width_filter = False
                
                with filter_col1:
                    # Depth filter
                    if 'depth [%]' in defects_df.columns:
                        # Get min/max values with safety checks for non-numeric values
                        depth_values = pd.to_numeric(defects_df['depth [%]'], errors='coerce')
                        depth_min = float(depth_values.min())
                        depth_max = float(depth_values.max())
                        
                        apply_depth_filter = st.checkbox("Filter by Depth", key="filter_depth")
                        if apply_depth_filter:
                            min_depth, max_depth = st.slider(
                                "Depth Range (%)",
                                min_value=depth_min,
                                max_value=depth_max,
                                value=(depth_min, depth_max),
                                step=0.5,
                                key="depth_range"
                            )
                    
                    # Length filter
                    if 'length [mm]' in defects_df.columns:
                        # Get min/max values with safety checks
                        length_values = pd.to_numeric(defects_df['length [mm]'], errors='coerce')
                        length_min = float(length_values.min())
                        length_max = float(length_values.max() + 10)  # Add small margin
                        
                        apply_length_filter = st.checkbox("Filter by Length", key="filter_length")
                        if apply_length_filter:
                            min_length, max_length = st.slider(
                                "Length Range (mm)",
                                min_value=length_min,
                                max_value=length_max,
                                value=(length_min, length_max),
                                step=5.0,
                                key="length_range"
                            )
                
                with filter_col2:
                    # Width filter
                    if 'width [mm]' in defects_df.columns:
                        # Get min/max values with safety checks
                        width_values = pd.to_numeric(defects_df['width [mm]'], errors='coerce')
                        width_min = float(width_values.min())
                        width_max = float(width_values.max() + 10)  # Add small margin
                        
                        apply_width_filter = st.checkbox("Filter by Width", key="filter_width")
                        if apply_width_filter:
                            min_width, max_width = st.slider(
                                "Width Range (mm)",
                                min_value=width_min,
                                max_value=width_max,
                                value=(width_min, width_max),
                                step=5.0,
                                key="width_range"
                            )
                    
                    # Add defect type filter if available
                    if 'component / anomaly identification' in defects_df.columns:
                        defect_types = ['All Types'] + sorted(defects_df['component / anomaly identification'].unique().tolist())
                        selected_defect_type = st.selectbox(
                            "Filter by Defect Type",
                            options=defect_types,
                            key="defect_type_filter"
                        )
            
            # Button to show visualization with filtering
            if st.button("Generate Pipeline Visualization", key="show_pipeline_single_analysis", use_container_width=True):
                st.markdown("<div class='section-header'>Pipeline Defect Map</div>", unsafe_allow_html=True)
                
                # Show a spinner during calculation
                with st.spinner("Generating pipeline visualization..."):
                    # Apply filters to the defects dataframe
                    filtered_defects = defects_df.copy()
                    
                    filter_applied = False
                    filter_description = []
                    
                    # Apply depth filter if checked
                    if apply_depth_filter and 'depth [%]' in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects['depth [%]'], errors='coerce') >= min_depth) & 
                            (pd.to_numeric(filtered_defects['depth [%]'], errors='coerce') <= max_depth)
                        ]
                        filter_applied = True
                        filter_description.append(f"Depth: {min_depth}% to {max_depth}%")
                    
                    # Apply length filter if checked
                    if apply_length_filter and 'length [mm]' in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects['length [mm]'], errors='coerce') >= min_length) & 
                            (pd.to_numeric(filtered_defects['length [mm]'], errors='coerce') <= max_length)
                        ]
                        filter_applied = True
                        filter_description.append(f"Length: {min_length}mm to {max_length}mm")
                    
                    # Apply width filter if checked
                    if apply_width_filter and 'width [mm]' in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects['width [mm]'], errors='coerce') >= min_width) & 
                            (pd.to_numeric(filtered_defects['width [mm]'], errors='coerce') <= max_width)
                        ]
                        filter_applied = True
                        filter_description.append(f"Width: {min_width}mm to {max_width}mm")
                    
                    # Apply defect type filter if selected
                    if 'component / anomaly identification' in filtered_defects.columns and selected_defect_type != 'All Types':
                        filtered_defects = filtered_defects[
                            filtered_defects['component / anomaly identification'] == selected_defect_type
                        ]
                        filter_applied = True
                        filter_description.append(f"Type: {selected_defect_type}")
                    
                    # Create and display statistics about filtered data
                    if filter_applied:
                        original_count = len(defects_df)
                        filtered_count = len(filtered_defects)
                        
                        # Join filter descriptions
                        filter_text = ", ".join(filter_description)
                        
                        # Show filter summary
                        st.info(f"Showing {filtered_count} defects out of {original_count} total defects ({filtered_count/original_count*100:.1f}%)\nFilters applied: {filter_text}")
                    
                    # Generate the visualization with filtered data
                    fig = create_unwrapped_pipeline_visualization(filtered_defects, joints_df)
                    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
                    
                    # Display guide
                    st.info("""
                    **Visualization Guide:**
                    - Each point represents a defect
                    - X-axis shows distance along pipeline in meters
                    - Y-axis shows clock position
                    - Color indicates defect depth percentage
                    """)

        else:  # Joint-by-joint visualization
            # Joint selection with improved UI
            available_joints = sorted(joints_df["joint number"].unique())
            
            # Format joint numbers with distance
            joint_options = {}
            for joint in available_joints:
                joint_row = joints_df[joints_df["joint number"] == joint].iloc[0]
                distance = joint_row["log dist. [m]"]
                joint_options[f"Joint {joint} (at {distance:.1f}m)"] = joint
            
            joint_col1, joint_col2 = st.columns([3, 1])

            with joint_col1:
                selected_joint_label = st.selectbox(
                    "Select Joint to Visualize",
                    options=list(joint_options.keys()),
                    key="joint_selector_single_analysis"
                )

            with joint_col2:
                view_mode = st.radio(
                    "View Mode",
                    ["2D Unwrapped"],
                    key="joint_view_mode"
                )
            
            selected_joint = joint_options[selected_joint_label]

            # Button to show joint visualization
            if st.button("Generate Joint Visualization", key="show_joint_single_analysis", use_container_width=True):
                st.markdown(f"<div class='section-header'>Defect Map for {selected_joint_label}</div>", unsafe_allow_html=True)
                
                # Get joint summary
                joint_summary = create_joint_summary(defects_df, joints_df, selected_joint)
                
                # Create summary panel with metrics
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    st.markdown(custom_metric("Defect Count", joint_summary["defect_count"]), unsafe_allow_html=True)
                
                with summary_cols[1]:
                    # Format defect types as a string
                    if joint_summary["defect_types"]:
                        defect_types_str = ", ".join([f"{count} {type_}" for type_, count in joint_summary["defect_types"].items()])
                        if len(defect_types_str) < 30:
                            st.markdown(custom_metric("Defect Types", defect_types_str), unsafe_allow_html=True)
                        else:
                            st.markdown(custom_metric("Defect Types", f"{len(joint_summary['defect_types'])} types"), unsafe_allow_html=True)
                            st.write(defect_types_str)
                    else:
                        st.markdown(custom_metric("Defect Types", "None"), unsafe_allow_html=True)
                
                with summary_cols[2]:
                    length_value = joint_summary["joint_length"]
                    if length_value != "N/A":
                        length_value = f"{length_value:.2f}m"
                    st.markdown(custom_metric("Joint Length", length_value), unsafe_allow_html=True)
                
                with summary_cols[3]:
                    st.markdown(custom_metric("Severity Rank", joint_summary["severity_rank"]), unsafe_allow_html=True)
                
                # Add a divider for clarity
                st.markdown("<hr style='margin:20px 0;border-color:#e0e0e0;'>", unsafe_allow_html=True)
                
                # Show the visualization with better handling
                with st.spinner("Generating joint visualization..."):
                    fig = create_joint_defect_visualization(defects_df, selected_joint)
                    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close the card container