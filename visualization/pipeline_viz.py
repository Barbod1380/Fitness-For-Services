"""
Functions for creating pipeline visualizations.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def create_unwrapped_pipeline_visualization(defects_df, joints_df, pipe_diameter=1.0):
    """
    Create an enhanced unwrapped cylinder visualization of pipeline defects,
    optimized for performance with large datasets.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - joints_df: DataFrame containing joint information
    - pipe_diameter: Pipeline diameter in meters (default: 1.0)
    
    Returns:
    - Plotly figure object
    """
    # Performance optimization - use Scattergl instead of Scatter for large datasets
    use_webgl = defects_df.shape[0] > 1000
    scatter_type = go.Scattergl if use_webgl else go.Scatter
    
    # Simplify customdata to reduce memory usage
    if 'component / anomaly identification' in defects_df.columns:
        # Simplified custom data with just the essentials
        custom_data = np.stack([
            defects_df["joint number"].astype(str),
            defects_df["component / anomaly identification"],
            defects_df["depth [%]"].fillna(0),
        ], axis=-1)
    else:
        custom_data = np.stack([
            defects_df["joint number"].astype(str),
            defects_df["depth [%]"].fillna(0),
        ], axis=-1)
    
    # Create figure with a single trace using depth for color
    fig = go.Figure()
    
    # Simplified hover template
    hover_template = (
        "<b>Distance:</b> %{x:.2f} m<br>"
        "<b>Depth:</b> %{customdata[2]:.1f}%<br>"
        "<b>Joint:</b> %{customdata[0]}<extra></extra>"
    )
    
    # Create a single trace for all defects, colored by depth
    if "depth [%]" in defects_df.columns:
        fig.add_trace(scatter_type(
            x=defects_df["log dist. [m]"],
            y=defects_df["clock_float"],
            mode="markers",
            marker=dict(
                size=5,  # Smaller markers for better performance
                color=defects_df["depth [%]"],
                colorscale="Turbo",
                cmin=0,
                cmax=defects_df["depth [%]"].max(),
                colorbar=dict(
                    title="Depth (%)",
                    thickness=15,
                    len=0.6,
                ),
                opacity=0.7,  # Slight transparency for better visibility when overlapping
            ),
            customdata=custom_data,
            hovertemplate=hover_template,
            name="Defects"
        ))
    else:
        # Fallback if no depth data
        fig.add_trace(scatter_type(
            x=defects_df["log dist. [m]"],
            y=defects_df["clock_float"],
            mode="markers",
            marker=dict(
                size=5,
                color="blue",
                opacity=0.7,
            ),
            customdata=custom_data,
            hovertemplate=hover_template,
            name="Defects"
        ))
    
    # Simplified joint markers - just add vertical lines instead of annotations
    for _, row in joints_df.iterrows():
        x0 = row["log dist. [m]"]
        joint_num = row["joint number"]
    
    # Add a simplified clock position grid (fewer lines)
    for hour in [3, 6, 9, 12]:
        fig.add_shape(
            type="line",
            x0=defects_df["log dist. [m]"].min() - 1,
            x1=defects_df["log dist. [m]"].max() + 1,
            y0=hour,
            y1=hour,
            line=dict(color="lightgray", width=1, dash="dot"),
            layer="below"
        )
    
    # Simplified layout
    fig.update_layout(
        title="Pipeline Defect Map",
        xaxis=dict(
            title="Distance Along Pipeline (m)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.2)",
        ),
        yaxis=dict(
            title="Clock Position (hr)",
            tickmode="array",
            tickvals=[3, 6, 9, 12],
            ticktext=["3:00", "6:00", "9:00", "12:00"],
            range=[0.5, 12.5],
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.2)",
        ),
        height=600,
        plot_bgcolor="white",
        hovermode="closest",
    )
    
    # Static color key instead of interactive buttons
    fig.add_annotation(
        text="Color indicates defect depth (%)",
        x=0.01,
        y=-0.1,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left"
    )
    
    # Remove buttons and interactive elements
    fig.update_layout(
        updatemenus=[],  # No updatemenus
        sliders=[]      # No sliders
    )
    
    return fig