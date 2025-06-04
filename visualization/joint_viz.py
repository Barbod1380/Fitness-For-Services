"""
Functions for creating joint-specific visualizations.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_joint_defect_visualization(defects_df, joint_number):
    """
    Create a visualization of defects for a specific joint, representing defects
    as rectangles whose fill color maps to depth (%) between the joint's min & max,
    plus an interactive hover and a matching colorbar.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - joint_number: The joint number to visualize
    
    Returns:
    - Plotly figure object
    """
    # Filter
    joint_defects = defects_df[defects_df['joint number'] == joint_number].copy()
    if joint_defects.empty:
        return go.Figure().update_layout(
            title=f"No defects found for Joint {joint_number}",
            xaxis_title="Distance (m)",
            yaxis_title="Clock Position",
            plot_bgcolor="white"
        )
    
    # Depth range for this joint
    depths = joint_defects['depth [%]'].astype(float)
    min_depth, max_depth = depths.min(), depths.max()
    
    # Ensure we have a valid range (avoid division by zero)
    if min_depth == max_depth:
        min_depth = max(0, min_depth - 1)
        max_depth = max_depth + 1

    # Geometry constants
    min_dist = joint_defects['log dist. [m]'].min()
    max_dist = joint_defects['log dist. [m]'].max()
    pipe_diameter = 1.0  # m
    meters_per_clock_unit = np.pi * pipe_diameter / 12

    fig = go.Figure()
    colorscale_name = "YlOrRd"

    # Draw each defect
    for _, defect in joint_defects.iterrows():
        x_center = defect['log dist. [m]']
        clock_pos = defect['clock_float']
        length_m = defect['length [mm]'] / 1000
        width_m = defect['width [mm]'] / 1000
        depth_pct = float(defect['depth [%]'])

        # rectangle corners
        w_clock = width_m / meters_per_clock_unit
        x0, x1 = x_center - length_m/2, x_center + length_m/2
        y0, y1 = clock_pos - w_clock/2, clock_pos + w_clock/2

        # Calculate normalized depth (0-1) for color mapping
        norm_depth = (depth_pct - min_depth) / (max_depth - min_depth)
        
        # Get color from colorscale using plotly's helper
        color = px.colors.sample_colorscale(colorscale_name, [norm_depth])[0]

        # Create custom data for hover info
        custom_data = [
            defect['clock'],
            depth_pct,
            defect['length [mm]'],
            defect['width [mm]'],
            defect.get('component / anomaly identification', 'Unknown')
        ]
        
        # Add rectangle for each defect with proper hover template
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            mode='lines',
            fill='toself',
            fillcolor=color,  # Apply the color from the colorscale
            line=dict(color='black', width=1),
            hoveron='fills+points',
            hoverinfo='text',
            customdata=[custom_data] * 5,  # Same data for all 5 points
            hovertemplate="<b>Defect Information</b><br>" +
                          "Distance: %{x:.3f} m<br>" +
                          "Clock: %{customdata[0]}<br>" +
                          "Depth: %{customdata[1]:.1f}%<br>" +
                          "Length: %{customdata[2]:.1f} mm<br>" +
                          "Width: %{customdata[3]:.1f} mm<br>" +
                          "Type: %{customdata[4]}<extra></extra>",
            showlegend=False
        ))

    # Invisible scatter for shared colorbar
    fig.add_trace(go.Scatter(
        x=[None]*len(depths),
        y=[None]*len(depths),
        mode='markers',
        marker=dict(
            color=depths,
            colorscale=colorscale_name,
            cmin=min_depth,
            cmax=max_depth,
            showscale=True,
            colorbar=dict(
                title="Depth (%)",
                thickness=15,
                len=0.5,
                tickformat=".1f"
            ),
            opacity=0
        ),
        showlegend=False
    ))

    # Clock‐hour grid lines
    for hr in range(1,13):
        fig.add_shape(
            type="line",
            x0=min_dist - 0.2, x1=max_dist + 0.2,
            y0=hr, y1=hr,
            line=dict(color="lightgray", dash="dot", width=1),
            layer="below"
        )

    # Layout
    fig.update_layout(
        title=f"Defect Map for Joint {joint_number}",
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title="Clock Position (hr)",
        plot_bgcolor="white",
        xaxis=dict(
            range=[min_dist - 0.2, max_dist + 0.2],
            showgrid=True, gridcolor="rgba(200,200,200,0.2)"
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(1,13)),
            ticktext=[f"{h}:00" for h in range(1,13)],
            range=[0.5,12.5],
            showgrid=True, gridcolor="rgba(200,200,200,0.2)"
        ),
        height=600, width=1200,
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig