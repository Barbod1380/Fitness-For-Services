"""
Functions for creating joint-specific visualizations.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def create_joint_defect_visualization(defects_df, joint_number, pipe_diameter_m):
    """
    Create a visualization of defects for a specific joint, representing defects
    as rectangles whose fill color maps to depth (%) between the joint's min & max,
    plus an interactive hover and a matching colorbar.

    Parameters:
    - defects_df: DataFrame containing defect information
    - joint_number: The joint number to visualize
    - pipe_diameter_m: The pipe diameter in meter unit

    Returns:
    - Plotly figure object
    """
    # Filter to this joint
    joint_defects = defects_df[defects_df["joint number"] == joint_number].copy()
    if joint_defects.empty:
        return (
            go.Figure()
            .update_layout(
                title=f"No defects found for Joint {joint_number}",
                xaxis_title="Distance (m)",
                yaxis_title="Clock Position",
                plot_bgcolor="white",
            )
        )

    # Determine depth range for color mapping
    depths = joint_defects["depth [%]"].astype(float)
    min_depth, max_depth = depths.min(), depths.max()

    # Prevent zero range
    if min_depth == max_depth:
        min_depth = max(0.0, min_depth - 1.0)
        max_depth = max_depth + 1.0

    # Geometry constants
    min_dist = joint_defects["log dist. [m]"].min()
    max_dist = joint_defects["log dist. [m]"].max()
    meters_per_clock_unit = np.pi * (pipe_diameter_m) / 12

    fig = go.Figure()
    colorscale = "YlOrRd"

    # Draw each defect as a filled rectangle
    for _, defect in joint_defects.iterrows():
        x_center = defect["log dist. [m]"]
        clock_pos = defect["clock_float"]
        length_m = defect["length [mm]"] / 1000
        width_m = defect["width [mm]"] / 1000
        depth_pct = float(defect["depth [%]"])

        # Calculate rectangle corners in x and y (clock) space
        half_len = length_m / 2
        w_clock = width_m / meters_per_clock_unit
        x0, x1 = x_center - half_len, x_center + half_len
        y0, y1 = clock_pos - w_clock / 2, clock_pos + w_clock / 2

        # Normalize depth into [0, 1] for color sampling
        norm_depth = (depth_pct - min_depth) / (max_depth - min_depth)
        fill_color = px.colors.sample_colorscale(colorscale, [norm_depth])[0]

        # Hover information
        custom_data = [
            defect["clock"],
            depth_pct,
            defect["length [mm]"],
            defect["width [mm]"],
            defect.get("component / anomaly identification", "Unknown"),
        ]

        # Add rectangle trace (closed loop of 5 points)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="black", width=1),
                hoveron="fills+points",
                hoverinfo="text",
                customdata=[custom_data] * 5,
                hovertemplate=(
                    "<b>Defect Information</b><br>"
                    "Distance: %{x:.3f} m<br>"
                    "Clock: %{customdata[0]}<br>"
                    "Depth: %{customdata[1]:.1f}%<br>"
                    "Length: %{customdata[2]:.1f} mm<br>"
                    "Width: %{customdata[3]:.1f} mm<br>"
                    "Type: %{customdata[4]}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Invisible scatter trace to create shared colorbar
    fig.add_trace(
        go.Scatter(
            x=[None] * len(depths),
            y=[None] * len(depths),
            mode="markers",
            marker=dict(
                color=depths,
                colorscale=colorscale,
                cmin=min_depth,
                cmax=max_depth,
                showscale=True,
                colorbar=dict(title="Depth (%)", thickness=15, len=0.5, tickformat=".1f"),
                opacity=0,
            ),
            showlegend=False,
        )
    )

    # Draw clock‐hour grid lines across the full x‐range
    for hour in range(1, 13):
        fig.add_shape(
            type="line",
            x0=min_dist - 0.2,
            x1=max_dist + 0.2,
            y0=hour,
            y1=hour,
            line=dict(color="lightgray", dash="dot", width=1),
            layer="below",
        )

    # Final layout adjustments
    fig.update_layout(
        title=f"Defect Map for Joint {joint_number}",
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title="Clock Position (hr)",
        plot_bgcolor="white",
        xaxis=dict(
            range=[min_dist - 0.2, max_dist + 0.2],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.2)",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=[f"{h}:00" for h in range(1, 13)],
            range=[0.5, 12.5],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.2)",
        ),
        height=600,
        width=1200,
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig