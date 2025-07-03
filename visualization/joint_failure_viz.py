# visualization/joint_failure_viz.py

"""
Visualization components for joint failure analysis showing before/after states.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils.format_utils import parse_clock


def create_joint_failure_visualization(joint_failure_info: dict, pipe_diameter_mm: float) -> go.Figure:
    """
    Create a before/after visualization of a failing joint showing defect growth.
    
    Parameters:
    - joint_failure_info: Dictionary with joint failure details from failure prediction
    - pipe_diameter_mm: Pipe diameter for proper scaling
    
    Returns:
    - Plotly figure with side-by-side before/after comparison
    """
    
    joint_num = joint_failure_info['joint_number']
    failure_year = joint_failure_info['failure_year']
    current_defects = joint_failure_info['current_defects_df']
    projected_defects = joint_failure_info['projected_defects_df']
    failure_causes = joint_failure_info['failure_causing_defects']
    
    # Create subplot with two columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Current State - Joint {joint_num}",
            f"Projected State (Year {failure_year}) - Joint {joint_num}"
        ],
        horizontal_spacing=0.1,
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Get failure-causing defect indices for highlighting
    failure_defect_indices = [f['defect_idx'] for f in failure_causes]
    
    # Create visualizations for both states
    _add_joint_defects_to_subplot(
        fig, current_defects, pipe_diameter_mm, 
        col=1, title="Current", 
        failure_indices=failure_defect_indices
    )
    
    _add_joint_defects_to_subplot(
        fig, projected_defects, pipe_diameter_mm, 
        col=2, title="Projected", 
        failure_indices=failure_defect_indices
    )
    
    # Add growth arrows between the two states
    _add_growth_arrows(fig, current_defects, projected_defects, pipe_diameter_mm)
    
    # Update layout
    _update_joint_failure_layout(fig, joint_num, failure_year, failure_causes, pipe_diameter_mm)
    
    return fig


def _add_joint_defects_to_subplot(fig, defects_df, pipe_diameter_mm, col, title, failure_indices):
    """
    Add defects visualization to a specific subplot column.
    """
    if defects_df.empty:
        fig.add_annotation(
            text=f"No defects in {title.lower()} state",
            xref=f"x{col}", yref=f"y{col}",
            x=0.5, y=6, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return
    
    # Determine depth range for color mapping
    depths = defects_df["depth [%]"].astype(float)
    min_depth, max_depth = depths.min(), depths.max()
    
    # Prevent zero range
    if min_depth == max_depth:
        min_depth = max(0.0, min_depth - 1.0)
        max_depth = max_depth + 1.0
    
    # Geometry constants
    min_dist = defects_df["log dist. [m]"].min()
    max_dist = defects_df["log dist. [m]"].max()
    pipe_diameter_m = pipe_diameter_mm / 1000  # Convert to meters
    meters_per_clock_unit = np.pi * pipe_diameter_m / 12
    
    # Color scheme
    colorscale = "YlOrRd"
    failure_color = "red"
    normal_edge_color = "black"
    failure_edge_color = "darkred"
    
    # Draw each defect as a filled rectangle
    for idx, defect in defects_df.iterrows():
        x_center = defect["log dist. [m]"]
        
        # Parse clock position
        clock_str = defect.get("clock", "12:00")
        if isinstance(clock_str, str):
            clock_pos = parse_clock(clock_str)
        else:
            clock_pos = float(clock_str) if not pd.isna(clock_str) else 12.0
        
        length_m = defect["length [mm]"] / 1000
        width_m = defect["width [mm]"] / 1000
        depth_pct = float(defect["depth [%]"])
        
        # Calculate rectangle corners
        half_len = length_m / 2
        w_clock = width_m / meters_per_clock_unit
        x0, x1 = x_center - half_len, x_center + half_len
        y0, y1 = clock_pos - w_clock / 2, clock_pos + w_clock / 2
        
        # Determine if this defect causes failure
        is_failure_cause = idx in failure_indices
        
        # Color and styling
        if is_failure_cause:
            # Failure-causing defects are highlighted
            fill_color = failure_color
            line_color = failure_edge_color
            line_width = 3
            opacity = 0.8
        else:
            # Normal color mapping by depth
            norm_depth = (depth_pct - min_depth) / (max_depth - min_depth)
            fill_color = px.colors.sample_colorscale(colorscale, [norm_depth])[0]
            line_color = normal_edge_color
            line_width = 1
            opacity = 0.6
        
        # Hover information
        defect_type = defect.get("component / anomaly identification", "Unknown")
        custom_data = [
            clock_str,
            depth_pct,
            defect["length [mm]"],
            defect["width [mm]"],
            defect_type,
            "⚠️ FAILURE CAUSE" if is_failure_cause else "Normal"
        ]
        
        # Add rectangle trace
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=line_width),
                opacity=opacity,
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
                    "Type: %{customdata[4]}<br>"
                    "Status: %{customdata[5]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=col
        )
    
    # Add clock hour grid lines
    x_range = [min_dist - 0.1, max_dist + 0.1]
    for hour in range(1, 13):
        fig.add_shape(
            type="line",
            x0=x_range[0], x1=x_range[1],
            y0=hour, y1=hour,
            line=dict(color="lightgray", dash="dot", width=1),
            row=1, col=col
        )


def _add_growth_arrows(fig, current_defects, projected_defects, pipe_diameter_mm):
    """
    Add arrows showing defect growth between current and projected states.
    """
    if current_defects.empty or projected_defects.empty:
        return
    
    pipe_diameter_m = pipe_diameter_mm / 1000
    meters_per_clock_unit = np.pi * pipe_diameter_m / 12
    
    # Find common defects between current and projected
    for idx in current_defects.index:
        if idx not in projected_defects.index:
            continue
            
        current = current_defects.loc[idx]
        projected = projected_defects.loc[idx]
        
        # Calculate centers
        current_x = current["log dist. [m]"]
        projected_x = projected["log dist. [m]"]
        
        # Parse clock positions
        current_clock = parse_clock(current.get("clock", "12:00"))
        projected_clock = parse_clock(projected.get("clock", "12:00"))
        
        # Only add arrow if there's significant growth
        depth_growth = projected["depth [%]"] - current["depth [%]"]
        length_growth = projected["length [mm]"] - current["length [mm]"]
        
        if depth_growth > 1.0 or length_growth > 5.0:  # Thresholds for showing arrows
            # Add annotation arrow showing growth
            fig.add_annotation(
                x=projected_x, y=projected_clock,
                ax=current_x, ay=current_clock,
                xref="x2", yref="y2",
                axref="x1", ayref="y1",
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor="blue",
                showarrow=True,
                text=f"Δd: +{depth_growth:.1f}%<br>Δl: +{length_growth:.0f}mm",
                font=dict(size=8, color="blue"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="blue",
                borderwidth=1
            )


def _update_joint_failure_layout(fig, joint_num, failure_year, failure_causes, pipe_diameter_mm):
    """
    Update the layout for the joint failure visualization.
    """
    
    # Determine failure mode
    failure_types = [f['failure_type'] for f in failure_causes]
    if 'erf' in failure_types and 'depth' in failure_types:
        failure_mode = "ERF & Depth Failure"
    elif 'erf' in failure_types:
        failure_mode = "ERF Failure (Operating Pressure Exceeded)"
    else:
        failure_mode = "Depth Failure (>80% Wall Thickness)"
    
    fig.update_layout(
        title={
            'text': f"Joint {joint_num} Failure Analysis<br><sub>{failure_mode} in Year {failure_year}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=600,
        width=1400,
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=False,
        # Add legend for failure indicators
        annotations=[
            dict(
                x=0.5, y=-0.1,
                xref="paper", yref="paper",
                text="🔴 Red = Failure-causing defects | Color intensity = Depth severity | Blue arrows = Growth direction",
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="center"
            )
        ]
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Distance Along Pipeline (m)",
        showgrid=True,
        gridcolor="rgba(200,200,200,0.2)",
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Distance Along Pipeline (m)",
        showgrid=True,
        gridcolor="rgba(200,200,200,0.2)",
        row=1, col=2
    )
    
    # Update y-axes
    for col in [1, 2]:
        fig.update_yaxes(
            title_text="Clock Position (hr)",
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=[f"{h}:00" for h in range(1, 13)],
            range=[0.5, 12.5],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.2)",
            row=1, col=col
        )


def create_failure_summary_card(joint_failure_info: dict) -> dict:
    """
    Create a summary card with key information about the joint failure.
    
    Parameters:
    - joint_failure_info: Dictionary with joint failure details
    
    Returns:
    - Dictionary with summary information for display
    """
    
    joint_num = joint_failure_info['joint_number']
    failure_year = joint_failure_info['failure_year']
    failure_causes = joint_failure_info['failure_causing_defects']
    defect_growth = joint_failure_info['defect_growth_info']
    
    # Analyze failure causes
    failure_types = [f['failure_type'] for f in failure_causes]
    erf_failures = [f for f in failure_causes if f['failure_type'] == 'erf']
    depth_failures = [f for f in failure_causes if f['failure_type'] == 'depth']
    
    # Find worst-case failure criteria
    worst_erf = min([f['failure_criteria'] for f in erf_failures]) if erf_failures else None
    worst_depth = max([f['failure_criteria'] for f in depth_failures]) if depth_failures else None
    
    # Calculate total growth statistics
    total_defects = len(defect_growth)
    failure_causing_defects = len([d for d in defect_growth if d['is_failure_cause']])
    
    avg_depth_growth = np.mean([d['projected_depth'] - d['current_depth'] for d in defect_growth])
    max_depth_growth = max([d['projected_depth'] - d['current_depth'] for d in defect_growth])
    
    avg_length_growth = np.mean([d['projected_length'] - d['current_length'] for d in defect_growth])
    max_length_growth = max([d['projected_length'] - d['current_length'] for d in defect_growth])
    
    return {
        'joint_number': joint_num,
        'failure_year': failure_year,
        'failure_mode': _determine_failure_mode(failure_types),
        'worst_erf': worst_erf,
        'worst_depth_pct': worst_depth,
        'total_defects': total_defects,
        'failure_causing_defects': failure_causing_defects,
        'avg_depth_growth': avg_depth_growth,
        'max_depth_growth': max_depth_growth,
        'avg_length_growth': avg_length_growth,
        'max_length_growth': max_length_growth
    }


def _determine_failure_mode(failure_types):
    """Determine the primary failure mode from failure types."""
    if 'erf' in failure_types and 'depth' in failure_types:
        return "Combined ERF & Depth"
    elif 'erf' in failure_types:
        return "ERF (Pressure)"
    elif 'depth' in failure_types:
        return "Depth (80% Threshold)"
    else:
        return "Unknown"


def create_joint_failure_timeline_chart(joint_timeline_data: dict) -> go.Figure:
    """
    Create a timeline chart showing when different joints fail.
    
    Parameters:
    - joint_timeline_data: Dictionary with joint failure timeline from results
    
    Returns:
    - Plotly figure showing failure timeline
    """
    
    # Prepare data for timeline
    timeline_data = []
    for joint_num, years_data in joint_timeline_data.items():
        for year, failure_info in years_data.items():
            failure_causes = failure_info['failure_causing_defects']
            failure_types = [f['failure_type'] for f in failure_causes]
            
            timeline_data.append({
                'joint_number': joint_num,
                'failure_year': year,
                'failure_mode': _determine_failure_mode(failure_types),
                'defect_count': len(failure_info['defect_growth_info']),
                'location_m': failure_causes[0]['location_m'] if failure_causes else 0
            })
    
    if not timeline_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No joint failures predicted in the analysis window",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    df = pd.DataFrame(timeline_data)
    
    # Color mapping for failure modes
    color_map = {
        'ERF (Pressure)': 'red',
        'Depth (80% Threshold)': 'orange',
        'Combined ERF & Depth': 'darkred'
    }
    
    fig = go.Figure()
    
    for mode, color in color_map.items():
        mode_data = df[df['failure_mode'] == mode]
        if mode_data.empty:
            continue
        
        fig.add_trace(
            go.Scatter(
                x=mode_data['failure_year'],
                y=mode_data['joint_number'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=color,
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name=f"{mode} ({len(mode_data)})",
                text=[
                    f"Joint {row['joint_number']}<br>"
                    f"Year {row['failure_year']}<br>"
                    f"Location: {row['location_m']:.1f}m<br>"
                    f"Defects: {row['defect_count']}<br>"
                    f"Mode: {row['failure_mode']}"
                    for _, row in mode_data.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>"
            )
        )
    
    fig.update_layout(
        title="Joint Failure Timeline",
        xaxis_title="Failure Year",
        yaxis_title="Joint Number",
        height=500,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig