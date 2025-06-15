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
    create_joint_summary
)
from visualization.pipeline_viz import create_unwrapped_pipeline_visualization
from visualization.joint_viz import create_joint_defect_visualization


def render_single_analysis_view():
    """Display single‐year analysis view with Data Preview, Defect Dimensions, and Visualizations."""
    st.markdown('<h2 class="section-header">Single Year Analysis</h2>', unsafe_allow_html=True)

    # --- Year Selection ---
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
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    tabs = st.tabs(["Data Preview", "Defect Dimensions", "Visualizations"])

    # Tab 1: Data Preview
    with tabs[0]:
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader(f"{selected_year} Joints")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(joints_df.head(5), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                create_data_download_links(joints_df, "joints", selected_year),
                unsafe_allow_html=True
            )

        with right_col:
            st.subheader(f"{selected_year} Defects")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(defects_df.head(5), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                create_data_download_links(defects_df, "defects", selected_year),
                unsafe_allow_html=True
            )

    # Tab 2: Defect Dimensions Analysis
    with tabs[1]:
        st.subheader("Defect Dimension Analysis")

        st.markdown("<div class='section-header'>Dimension Statistics</div>", unsafe_allow_html=True)
        stats_df = create_dimension_statistics_table(defects_df)
        if not stats_df.empty:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(stats_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            info_box("No dimension data available for analysis.", "warning")

        dimension_figs = create_dimension_distribution_plots(defects_df)
        if dimension_figs:
            st.markdown(
                "<div class='section-header' style='margin-top:20px;'>"
                "Dimension Distributions</div>",
                unsafe_allow_html=True
            )
            
            # Since we now have a single combined figure, just display it once
            combined_fig = list(dimension_figs.values())[0]  # Get the combined figure
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
                        "height": 500,
                        "width": 900,  # Adjust based on your needs
                        "scale": 2
                    }
                }
            )

            st.markdown(
                "<div class='section-header' style='margin-top:20px;'>"
                "Defect Dimensions Relationship</div>",
                unsafe_allow_html=True
            )
            combined_fig = create_combined_dimensions_plot(defects_df)
            st.plotly_chart(combined_fig, use_container_width=True)
        else:
            info_box("No dimension data available for plotting distributions.", "warning")

    # Tab 3: Pipeline Visualizations
    with tabs[2]:
        st.subheader("Pipeline Visualization")
        viz_col1, viz_col2 = st.columns([2, 2])

        with viz_col1:
            viz_type = st.radio(
                "Select Visualization Type",
                ["Complete Pipeline", "Joint-by-Joint"],
                horizontal=True,
                key="viz_type_single_analysis"
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

                    if (
                        "component / anomaly identification" in defects_df.columns
                        and selected_defect_type != "All Types"
                    ):
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
                    fig = create_unwrapped_pipeline_visualization(filtered_defects, pipe_diameter)
                    
                    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
                    st.info(
                        "**Visualization Guide:**\n"
                        "- Each point represents a defect\n"
                        "- X-axis: distance along pipeline (m)\n"
                        "- Y-axis: clock position\n"
                        "- Color: defect depth percentage"
                    )

        else:
            # --- Joint‐by‐Joint Visualization ---
            available_joints = sorted(joints_df["joint number"].unique())
            joint_labels = {
                f"Joint {j} (at {joints_df[joints_df['joint number'] == j].iloc[0]['log dist. [m]']:.1f}m)": j
                for j in available_joints
            }

            jcol1, jcol2 = st.columns([3, 1])
            with jcol1:
                selected_label = st.selectbox(
                    "Select Joint to Visualize",
                    options=list(joint_labels.keys()),
                    key="joint_selector_single_analysis"
                )
            with jcol2:
                _ = st.radio("View Mode", ["2D Unwrapped"], key="joint_view_mode")

            joint_id = joint_labels[selected_label]
            if st.button(
                "Generate Joint Visualization",
                key="show_joint_single_analysis",
                use_container_width=True
            ):
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

                st.markdown("<hr style='margin:20px 0;border-color:#e0e0e0;'>", unsafe_allow_html=True)
                with st.spinner("Generating joint visualization..."):
                    fig = create_joint_defect_visualization(defects_df, joint_id)
                    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

    st.markdown('</div>', unsafe_allow_html=True)