"""
Single year analysis view for the Pipeline Analysis application.
"""

import streamlit as st
import pandas as pd
from app.ui_components.ui_elements import custom_metric, info_box, create_data_download_links
from app.ui_components.charts import create_metrics_row
from analysis.defect_analysis import (
    create_dimension_distribution_plots,
    create_dimension_statistics_table,
    create_combined_dimensions_plot,
    create_joint_summary,
    create_clean_combined_defect_plot,  
    create_defect_categorization_summary_table
)
from visualization.pipeline_viz import create_unwrapped_pipeline_visualization
from visualization.joint_viz import create_joint_defect_visualization


def render_single_analysis_view():
    """Display single‐year analysis view with Data Preview, Defect Dimensions, and Visualizations."""

    # Enhanced page header
    st.markdown("""
    <div class="page-section">
        <div class="page-title-area">
            <h2 class="page-title">Single Year Analysis</h2>
            <p class="page-description">Comprehensive analysis of individual inspection datasets</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Year Selection with better spacing ---
    st.markdown('<div class="content-spacing">', unsafe_allow_html=True)
    years = sorted(st.session_state.datasets.keys())
    col1, col2 = st.columns([2, 2])
    with col1:
        default_index = (
            years.index(st.session_state.current_year)
            if st.session_state.current_year in years
            else 0
        )
        selected_year = st.selectbox(
            "Select Year to Analyze",
            options=years,
            index=default_index,
            key="year_selector_single_analysis"
        )


    # --- Load Datasets ---
    joints_df = st.session_state.datasets[selected_year]["joints_df"]
    defects_df = st.session_state.datasets[selected_year]["defects_df"]


    # --- Summary Metrics ---
    with col2:
        max_depth = (
            f"{defects_df['depth [%]'].max():.1f}%"
            if "depth [%]" in defects_df.columns
            else "N/A"
        )
        metrics_data = [
            ("Joints", len(joints_df), None),
            ("Defects", len(defects_df), None),
            ("Max Depth", max_depth, None),
        ]
        create_metrics_row(metrics_data)


    # --- Analysis Tabs ---
    st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
    tabs = st.tabs(["Data Preview", "Defect Dimensions", "Visualizations"])
    st.markdown('</div>', unsafe_allow_html=True)


    # Tab 1: Data Preview
    with tabs[0]:
        left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(f"""
        <div class="table-section">
            <div class="table-header">
                <h3 class="table-title">{selected_year} Joints</h3>
                <div class="table-stats">{len(joints_df):,} records</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            joints_df.head(10), 
            use_container_width=True,
            height=320,
            column_config={
                "joint number": st.column_config.NumberColumn("Joint", format="%d"),
                "log dist. [m]": st.column_config.NumberColumn("Distance (m)", format="%.2f"),
                "wt nom [mm]": st.column_config.NumberColumn("Wall Thickness (mm)", format="%.1f"),
                "joint length [m]": st.column_config.NumberColumn("Length (m)", format="%.2f")
            }
        )
        st.markdown(create_data_download_links(joints_df, "joints", selected_year), unsafe_allow_html=True)

    with right_col:
        st.markdown(f"""
        <div class="table-section">
            <div class="table-header">
                <h3 class="table-title">{selected_year} Defects</h3>
                <div class="table-stats">{len(defects_df):,} records</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            defects_df.head(10),
            use_container_width=True, 
            height=320,
            column_config={
                "depth [%]": st.column_config.ProgressColumn(
                    "Depth (%)", 
                    min_value=0, 
                    max_value=100, 
                    format="%.1f%%"
                ),
                "length [mm]": st.column_config.NumberColumn("Length (mm)", format="%.0f"),
                "width [mm]": st.column_config.NumberColumn("Width (mm)", format="%.0f"),
                "joint number": st.column_config.NumberColumn("Joint", format="%d"),
                "log dist. [m]": st.column_config.NumberColumn("Distance (m)", format="%.2f")
            }
        )
        st.markdown(create_data_download_links(defects_df, "defects", selected_year), unsafe_allow_html=True)


    # Tab 2: Defect Dimensions Analysis
    with tabs[1]:
        st.markdown("<div class='section-header'>Dimension Statistics</div>", unsafe_allow_html = True)
        stats_df = create_dimension_statistics_table(defects_df)
        
        if stats_df.empty:
            info_box("No dimension data available for analysis.", "warning")
        else:
            st.dataframe(stats_df, use_container_width=True)

        dimension_figs = create_dimension_distribution_plots(defects_df)
        if dimension_figs:
            st.markdown(
                "<div class='section-header' style='margin-top:20px;'>"
                "Dimension Distributions</div>",
                unsafe_allow_html = True
            )
            
            # Since we now have a single combined figure, just display it once (Get combined figures)
            combined_fig = list(dimension_figs.values())[0] 
            st.plotly_chart(
                combined_fig, 
                use_container_width=True, 
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["toImage"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "defect_dimension_distribution",
                        "height": 900,
                        "width": 900,
                        "scale": 2
                    }
                }
            )

            st.markdown(
                "<div class='section-header' style='margin-top:20px;'>"
                "Defect Dimensions Relationship</div>",
                unsafe_allow_html = True
            )
            
            combined_fig = create_combined_dimensions_plot(defects_df, joints_df)
            st.plotly_chart(combined_fig, use_container_width = True)
        
        else:
            info_box("No dimension data available for plotting distributions.", "warning")


        st.markdown(
            "<div class='section-header' style='margin-top:30px;'>"
            "🔍 FFS Defect Categorization Analysis</div>",
            unsafe_allow_html = True
        )
        
        # Check if surface location data is available
        has_surface_location = ('surface location' in defects_df.columns and not defects_df['surface location'].isna().all())
        
        if has_surface_location:
            # Count defects by surface location
            surface_counts = defects_df['surface location'].value_counts()
            int_count = surface_counts.get('INT', 0)
            nonint_count = surface_counts.get('NON-INT', 0)
            total_count = len(defects_df)
            
            # Create selection options with counts
            plot_options = [
                f"All Defects ({total_count} total)",
                f"Internal (INT) Only ({int_count} defects)",
                f"External (NON-INT) Only ({nonint_count} defects)"
            ]
            
            # User selection
            categorization_col1, _ = st.columns([3, 1])
            
            with categorization_col1:
                selected_surface_filter = st.selectbox(
                    "Select defect surface location to analyze:",
                    options = plot_options,
                    index = 0,
                    key = "categorization_surface_filter",
                    help = "Choose which defects to include in the FFS categorization analysis"
                )
            
            # Filter defects based on selection
            if "Internal (INT)" in selected_surface_filter:
                filtered_defects_cat = defects_df[defects_df['surface location'] == 'INT'].copy()
                plot_title_suffix = " - Internal (INT) Defects Only"
                analysis_note = f"Analysis limited to {len(filtered_defects_cat)} internal defects"
            elif "External (NON-INT)" in selected_surface_filter:
                filtered_defects_cat = defects_df[defects_df['surface location'] == 'NON-INT'].copy()
                plot_title_suffix = " - External (NON-INT) Defects Only"
                analysis_note = f"Analysis limited to {len(filtered_defects_cat)} external defects"
            else:
                filtered_defects_cat = defects_df.copy()
                plot_title_suffix = " - All Defects"
                analysis_note = f"Analysis includes all {len(filtered_defects_cat)} defects"
            
            # Show analysis info
            if len(filtered_defects_cat) == 0:
                st.warning(f"No defects found for the selected filter: {selected_surface_filter}")
            else:
                st.caption(analysis_note)

        else:
            # No surface location data available - use all defects
            filtered_defects_cat = defects_df.copy()
            plot_title_suffix = ""
            st.info("Surface location data not available - showing analysis for all defects")
        
        # Create the categorization plot with filtered data
        if len(filtered_defects_cat) > 0:
            categorization_fig = create_clean_combined_defect_plot(
                filtered_defects_cat, 
                joints_df, 
                title_suffix=plot_title_suffix # type: ignore
            )
            if categorization_fig:
                st.plotly_chart(categorization_fig, use_container_width=True, config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["toImage"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": f"defect_categorization_analysis_{selected_surface_filter.split()[0].lower() if has_surface_location else 'all'}",
                        "height": 1200,
                        "width": 1200,
                        "scale": 2
                    }
                })
            else:
                info_box("Unable to create defect categorization analysis - check data requirements.", "warning")


        # Enhanced explanation section
        with st.expander("📖 Understanding FFS Defect Categorization", expanded=False):
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 10px; margin: 10px 0;'>
            
            ### 🎯 **Purpose**
            This analysis classifies pipeline defects according to **Fitness-for-Service (FFS)** standards, 
            helping engineers understand defect patterns and prioritize maintenance actions.
            
            ### 📏 **Classification Rules**
            
            **Parameter A = max(wall_thickness, 10mm)** - normalizes defect dimensions relative to pipe structure.
            
            | Category | Criteria | Engineering Significance |
            |----------|----------|------------------------|
            | 🔹 **PinHole** | Length < A, Width < A | Small localized defects |
            | ↕️ **AxialSlot** | Width < A, Length ≥ A | Narrow defects along pipe axis |
            | ↔️ **CircSlot** | Width ≥ A, Length < A | Narrow circumferential defects |
            | 📏 **AxialGroove** | A ≤ Width ≤ 3A, L/W ≥ 2 | Elongated axial corrosion |
            | 🔄 **CircGroove** | 1 < Length < 3A, L/W ≤ 0.5 | Elongated circumferential corrosion |
            | ⚪ **General** | Length ≥ 3A, Width ≥ 3A | Large area general corrosion |
            | 🔴 **Pitting** | *All other patterns* | Random or complex corrosion |
            
            ### 🔬 **How to Use This Analysis**
            
            1. **Left Plot**: Categorization map showing where your defects fall in the classification space
            2. **Right Chart**: Frequency distribution showing count of each category
            3. **Background Regions**: Different colored areas show theoretical category boundaries
            4. **Data Points**: Your actual defects plotted with category-specific colors
            5. **Legend**: Shows category names for easy identification
            6. **Engineering Actions**: Use categories to prioritize inspection and repair strategies
            
            ### ⚡ **Performance Features**
            
            - Clean, fast rendering optimized for large datasets
            - No hover functionality for better performance
            - Side-by-side layout for comprehensive view
            - Color-coded for immediate visual understanding
            
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced summary table
        summary_df = create_defect_categorization_summary_table(defects_df, joints_df)
        if not summary_df.empty:
            st.markdown("#### 📋 Detailed Category Summary")
            
            # Enhanced table display
            st.dataframe(
                summary_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("🏆 Rank", width="small"),
                    "Category": st.column_config.TextColumn("📂 Category", width="medium"),
                    "Count": st.column_config.NumberColumn("🔢 Count", width="small"),
                    "Percentage": st.column_config.NumberColumn("📊 %", format="%.1f%%", width="small")
                }
            )
        else:
            info_box("Unable to create category summary - insufficient data.", "warning")


    # Tab 3: Pipeline Visualizations
    with tabs[2]:
        st.subheader("Pipeline Visualization")
        viz_col1, _ = st.columns([2, 2])

        with viz_col1:
            viz_type = st.radio(
                "Select Visualization Type",
                ["Complete Pipeline", "Joint-by-Joint"],
                horizontal = True,
                key = "viz_type_single_analysis"
            )

        if viz_type == "Complete Pipeline":
            # --- Filtering Options ---
            with st.expander("Filter Defects", expanded=False):
                st.subheader("Filter Defects by Properties")
                fcol1, fcol2 = st.columns(2)

                apply_depth = False
                apply_length = False
                apply_width = False

                if "depth [%]" in defects_df.columns:
                    depth_vals = pd.to_numeric(defects_df["depth [%]"], errors="coerce")
                    depth_min, depth_max = float(depth_vals.min()), float(depth_vals.max())
                    with fcol1:
                        apply_depth = st.checkbox("Filter by Depth", key="filter_depth")
                        if apply_depth:
                            min_depth, max_depth = st.slider(
                                "Depth Range (%)",
                                min_value=depth_min,
                                max_value=depth_max,
                                value=(depth_min, depth_max),
                                step=0.5,
                                key="depth_range"
                            )

                if "length [mm]" in defects_df.columns:
                    length_vals = pd.to_numeric(defects_df["length [mm]"], errors="coerce")
                    length_min, length_max = float(length_vals.min()), float(length_vals.max() + 10)
                    with fcol1:
                        apply_length = st.checkbox("Filter by Length", key="filter_length")
                        if apply_length:
                            min_length, max_length = st.slider(
                                "Length Range (mm)",
                                min_value=length_min,
                                max_value=length_max,
                                value=(length_min, length_max),
                                step=5.0,
                                key="length_range"
                            )

                if "width [mm]" in defects_df.columns:
                    width_vals = pd.to_numeric(defects_df["width [mm]"], errors="coerce")
                    width_min, width_max = float(width_vals.min()), float(width_vals.max() + 10)
                    with fcol2:
                        apply_width = st.checkbox("Filter by Width", key="filter_width")
                        if apply_width:
                            min_width, max_width = st.slider(
                                "Width Range (mm)",
                                min_value=width_min,
                                max_value=width_max,
                                value=(width_min, width_max),
                                step=5.0,
                                key="width_range"
                            )

                if "component / anomaly identification" in defects_df.columns:
                    defect_types = ["All Types"] + sorted(
                        defects_df["component / anomaly identification"].unique().tolist()
                    )
                    with fcol2:
                        selected_defect_type = st.selectbox(
                            "Filter by Defect Type",
                            options=defect_types,
                            key="defect_type_filter"
                        )

            with st.expander("Visualization Options", expanded=True):
                color_col1, color_col2 = st.columns(2)
                
                with color_col1:
                    # Y-axis selector - NEW ADDITION
                    y_axis_options = {
                        "Clock Position": "Clock Position",
                        "Defect Depth (%)": "Depth [%]", 
                        "Defect Length (mm)": "Length [mm]",
                        "Defect Width (mm)": "Width [mm]",
                        "Distance from Upstream Weld (m)": "Distance from USGW [m]"
                    }
                    
                    selected_y_display = st.selectbox(
                        "Y-Axis Parameter:",
                        options=list(y_axis_options.keys()),
                        index=0,  # Default to Clock Position
                        key="y_axis_selector_single",
                        help="Choose what parameter to display on the Y-axis"
                    )
                    
                    y_axis_column = y_axis_options[selected_y_display]
                
                with color_col2:
                    # Check if surface location data is available
                    has_surface_location = 'surface location' in defects_df.columns and not defects_df['surface location'].isna().all()
                    
                    if has_surface_location:
                        color_options = ["Depth (%)", "Surface Location (Internal/External)"]
                    else:
                        color_options = ["Depth (%)"]
                        
                    color_by = st.selectbox(
                        "Color/Shape Coding:",
                        options=color_options,
                        index=0,
                        key="pipeline_color_method",
                        help="Choose how to color-code the defects"
                    )

            required_columns = ["log dist. [m]"]
            
            if y_axis_column == "Clock Position":
                required_columns.append("clock_float")
            elif y_axis_column == "Depth [%]":
                required_columns.append("depth [%]")
            elif y_axis_column == "Length [mm]":
                required_columns.append("length [mm]")
            elif y_axis_column == "Width [mm]":
                required_columns.append("width [mm]")
            elif y_axis_column == "Distance from USGW [m]":
                required_columns.append("up weld dist. [m]")
            
            if color_by == "Surface Location (Internal/External)":
                required_columns.append("surface location")
            else:
                required_columns.append("depth [%]")
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in defects_df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns for this visualization: {missing_columns}")
                st.info("Please ensure your dataset contains all necessary columns.")
            else:
                # Show info about current selection
                st.info(f"📊 **Displaying**: {selected_y_display} vs Axial Distance | **Coded by**: {color_by}")


            if st.button(
                "Generate Pipeline Visualization",
                key="show_pipeline_single_analysis",
                use_container_width=True
            ):
                st.markdown(
                    "<div class='section-header'>Pipeline Defect Map</div>",
                    unsafe_allow_html=True
                )
                with st.spinner("Generating pipeline visualization..."):
                    filtered_defects = defects_df.copy()
                    filters_applied = []

                    if apply_depth and "depth [%]" in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects["depth [%]"], errors="coerce") >= min_depth)
                            & (pd.to_numeric(filtered_defects["depth [%]"], errors="coerce") <= max_depth)
                        ]
                        filters_applied.append(f"Depth: {min_depth}% to {max_depth}%")

                    if apply_length and "length [mm]" in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects["length [mm]"], errors="coerce") >= min_length)
                            & (pd.to_numeric(filtered_defects["length [mm]"], errors="coerce") <= max_length)
                        ]
                        filters_applied.append(f"Length: {min_length}mm to {max_length}mm")

                    if apply_width and "width [mm]" in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects["width [mm]"], errors="coerce") >= min_width)
                            & (pd.to_numeric(filtered_defects["width [mm]"], errors="coerce") <= max_width)
                        ]
                        filters_applied.append(f"Width: {min_width}mm to {max_width}mm")

                    if ("component / anomaly identification" in defects_df.columns and selected_defect_type != "All Types"):
                        filtered_defects = filtered_defects[
                            filtered_defects["component / anomaly identification"] == selected_defect_type
                        ]
                        filters_applied.append(f"Type: {selected_defect_type}")

                    if filters_applied:
                        orig_count = len(defects_df)
                        filt_count = len(filtered_defects)
                        filter_text = ", ".join(filters_applied)
                        st.info(
                            f"Showing {filt_count} defects out of {orig_count} "
                            f"total ({filt_count/orig_count*100:.1f}%) — Filters: {filter_text}"
                        )

                    pipe_diameter = st.session_state.datasets[selected_year]['pipe_diameter']
                    
                    # Check if data is valid for selected parameters
                    if not missing_columns:
                        # Filter out rows with missing data for selected parameters
                        plot_data = filtered_defects #.dropna(subset=required_columns).copy()
                        
                        if plot_data.empty:
                            st.warning("No valid data available for the selected parameters.")
                        else:
                            # Show data summary if filtered
                            if len(plot_data) < len(filtered_defects):
                                st.warning(f"Data filtered: Showing {len(plot_data):,} of {len(filtered_defects):,} defects (missing data excluded)")
                            
                            # Create enhanced visualization
                            fig = create_unwrapped_pipeline_visualization(
                                defects_df=plot_data,
                                pipe_diameter=pipe_diameter,
                                color_by=color_by,
                                y_axis_column=y_axis_column
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
                            
                            # Enhanced visualization guide
                            if color_by == "Surface Location (Internal/External)":
                                st.info(
                                    f"**Visualization Guide:**\n"
                                    f"- Each symbol represents a defect\n"
                                    f"- X-axis: Distance along pipeline (m)\n"
                                    f"- Y-axis: {selected_y_display}\n"
                                    f"- ❌ Red X: Internal defects | 🔵 Blue Circle: External defects | 💎 Gray Diamond: Unknown"
                                )
                            else:
                                st.info(
                                    f"**Visualization Guide:**\n"
                                    f"- Each point represents a defect\n"
                                    f"- X-axis: Distance along pipeline (m)\n"
                                    f"- Y-axis: {selected_y_display}\n"
                                    f"- Color: Defect depth percentage"
                                )
                    else:
                        st.error("Cannot generate visualization due to missing required columns.")

        else:
            # --- Joint‐by‐Joint Visualization ---
            available_joints = sorted(joints_df["joint number"].unique())
            joint_labels = {
                f"Joint {j} (at {joints_df[joints_df['joint number'] == j].iloc[0]['log dist. [m]']:.1f}m)": j
                for j in available_joints
            }

            jcol1, _ = st.columns([3, 1])
            with jcol1:
                selected_label = st.selectbox(
                    "Select Joint to Visualize",
                    options = list(joint_labels.keys()),
                    key="joint_selector_single_analysis"
                )

            if st.button("Generate Joint Visualization", key="show_joint_single_analysis", use_container_width=True):
                joint_id = joint_labels[selected_label]
            
                st.markdown(
                    f"<div class='section-header'>Defect Map for {selected_label}</div>",
                    unsafe_allow_html=True
                )
                joint_summary = create_joint_summary(defects_df, joints_df, joint_id)
                summary_cols = st.columns(4)

                with summary_cols[0]:
                    st.markdown(
                        custom_metric("Defect Count", joint_summary["defect_count"]),
                        unsafe_allow_html=True
                    )

                with summary_cols[1]:
                    defect_types = joint_summary["defect_types"]
                    if defect_types:
                        types_str = ", ".join(
                            [f"{cnt} {typ}" for typ, cnt in defect_types.items()]
                        )
                        display_label = (
                            types_str if len(types_str) < 30 else f"{len(defect_types)} types"
                        )
                        st.markdown(
                            custom_metric("Defect Types", display_label),
                            unsafe_allow_html=True
                        )
                        if len(types_str) >= 30:
                            st.write(types_str)
                    else:
                        st.markdown(
                            custom_metric("Defect Types", "None"),
                            unsafe_allow_html=True
                        )

                with summary_cols[2]:
                    jl = joint_summary["joint_length"]
                    jl_display = f"{jl:.2f}m" if jl != "N/A" else jl
                    st.markdown(
                        custom_metric("Joint Length", jl_display),
                        unsafe_allow_html=True
                    )

                with summary_cols[3]:
                    st.markdown(
                        custom_metric("Severity Rank", joint_summary["severity_rank"]),
                        unsafe_allow_html=True
                    )

                pipe_diameter = st.session_state.datasets[selected_year]['pipe_diameter']
                st.markdown("<hr style='margin:20px 0;border-color:#e0e0e0;'>", unsafe_allow_html=True)
                with st.spinner("Generating joint visualization..."):
                    fig = create_joint_defect_visualization(defects_df, joint_id, pipe_diameter)
                    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})


    st.markdown('</div>', unsafe_allow_html=True)