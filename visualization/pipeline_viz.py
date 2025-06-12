import numpy as np
import plotly.graph_objects as go

def create_unwrapped_pipeline_visualization(defects_df, pipe_diameter = 1.0):
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
    
    # === OPTIMIZATION 1: Intelligent data sampling ===
    max_points = 7000  # WebGL performance threshold
    
    if len(defects_df) > max_points:
        # Priority-based sampling: keep critical defects + representative sample
        
        # Always keep high-severity defects
        critical_mask = defects_df['depth [%]'] > 70
        critical_defects = defects_df[critical_mask]
        
        # Sample remaining defects spatially distributed
        remaining_defects = defects_df[~critical_mask]
        
        if len(remaining_defects) > 0:
            # Spatial binning for representative sampling
            n_bins = min(50, len(remaining_defects) // 10)
            remaining_defects['spatial_bin'] = pd.cut(remaining_defects['log dist. [m]'], 
                                                    bins=n_bins, labels=False)
            
            # Sample from each bin
            samples_per_bin = max(1, (max_points - len(critical_defects)) // n_bins)
            sampled_remaining = (remaining_defects.groupby('spatial_bin', group_keys=False)
                               .apply(lambda x: x.sample(min(len(x), samples_per_bin))))
            
            plot_data = pd.concat([critical_defects, sampled_remaining], ignore_index=True)
        else:
            plot_data = critical_defects
    else:
        plot_data = defects_df
    
    # Extract values once to avoid repeated DataFrame access
    x_vals = plot_data["log dist. [m]"].values
    y_vals = plot_data["clock_float"].values
    
    # === Efficient marker properties ===
    if "depth [%]" in plot_data.columns:
        depth_values = plot_data["depth [%]"].values
        marker_props = dict(
            size=6,  # Slightly smaller for better performance
            color=depth_values,
            colorscale="Turbo",
            cmin=0,
            cmax=depth_values.max(),
            colorbar=dict(title="Depth (%)", thickness=15, len=0.6),
            opacity=0.8,
        )
    else:
        marker_props = dict(size=6, color="blue", opacity=0.8)
    
    # Minimize data in customdata to reduce memory usage
    has_component = 'component / anomaly identification' in plot_data.columns
    
    if has_component:
        custom_data = np.column_stack([
            plot_data["joint number"].astype(str).values,
            plot_data["component / anomaly identification"].values,
            plot_data["depth [%]"].fillna(0).values,
        ])
    else:
        custom_data = np.column_stack([
            plot_data["joint number"].astype(str).values,
            plot_data["depth [%]"].fillna(0).values,
        ])
    
    # === Use WebGL for large datasets ===
    use_webgl = len(plot_data) > 1000
    scatter_class = go.Scattergl if use_webgl else go.Scatter
    
    # === OPTIMIZATION 6: Streamlined hover template ===
    hover_template = (
        "<b>Distance:</b> %{x:.2f} m<br>"
        "<b>Depth:</b> %{customdata[2]:.1f}%<br>"
        "<b>Joint:</b> %{customdata[0]}<extra></extra>"
    )
    
    fig = go.Figure()
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
    
    # === Minimal grid lines for performance ===
    x_range = [x_vals.min() - 1, x_vals.max() + 1]
    for hour in [3, 6, 9, 12]: 
        fig.add_shape(
            type="line",
            x0=x_range[0], x1=x_range[1],
            y0=hour, y1=hour,
            line=dict(color="lightgray", width=1, dash="dot"),
            layer="below",
        )
    
    # === Efficient layout configuration ===
    fig.update_layout(
        title="Pipeline Defect Map" + (f" (Showing {len(plot_data):,} of {len(defects_df):,} defects)" if len(plot_data) != len(defects_df) else ""),
        xaxis=dict(
            title="Distance Along Pipeline (m)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            range=x_range
        ),
        yaxis=dict(
            title="Clock Position (hr)",
            tickmode="array",
            tickvals=[3, 6, 9, 12],
            ticktext=["3:00", "6:00", "9:00", "12:00"],
            range=[0.5, 12.5],
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
        ),
        height = 600,
        plot_bgcolor = "white",
        hovermode = "closest",
        uirevision = "constant",  # Prevent unnecessary re-renders
        dragmode = "pan",  # Better for large datasets
    )
    
    return fig