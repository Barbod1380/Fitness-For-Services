# pipeline_viz.py

"""
Functions for creating pipeline visualizations.
"""
import numpy as np
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
    # Decide whether to use WebGL-based Scatter for large datasets
    use_webgl = defects_df.shape[0] > 1000
    scatter_class = go.Scattergl if use_webgl else go.Scatter

    # Build a simplified customdata array (joint number, [component], depth)
    has_component = 'component / anomaly identification' in defects_df.columns
    if has_component:
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

    # Extract x/y values once
    x_vals = defects_df["log dist. [m]"]
    y_vals = defects_df["clock_float"]

    # Determine marker properties based on presence of depth data
    if "depth [%]" in defects_df.columns:
        marker_props = dict(
            size=5,
            color=defects_df["depth [%]"],
            colorscale="Turbo",
            cmin=0,
            cmax=defects_df["depth [%]"].max(),
            colorbar=dict(title="Depth (%)", thickness=15, len=0.6),
            opacity=0.7,
        )
    else:
        marker_props = dict(size=5, color="blue", opacity=0.7)

    # Build the figure and add a single trace for all defects
    fig = go.Figure()
    hover_template = (
        "<b>Distance:</b> %{x:.2f} m<br>"
        "<b>Depth:</b> %{customdata[2]:.1f}%<br>"
        "<b>Joint:</b> %{customdata[0]}<extra></extra>"
    )
    fig.add_trace(
        scatter_class(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=marker_props,
            customdata=custom_data,
            hovertemplate=hover_template,
            name="Defects",
        )
    )

    # Add a simplified clock-position grid (horizontal lines at 3,6,9,12)
    min_dist = x_vals.min() - 1
    max_dist = x_vals.max() + 1
    for hour in [3, 6, 9, 12]:
        fig.add_shape(
            type="line",
            x0=min_dist,
            x1=max_dist,
            y0=hour,
            y1=hour,
            line=dict(color="lightgray", width=1, dash="dot"),
            layer="below",
        )

    # Final layout adjustments
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
        updatemenus=[],  # remove any interactive buttons
        sliders=[],
    )

    # Static color key annotation
    fig.add_annotation(
        text="Color indicates defect depth (%)",
        x=0.01,
        y=-0.1,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left",
    )

    return fig