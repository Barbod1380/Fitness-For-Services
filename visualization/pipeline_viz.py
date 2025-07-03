import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_unwrapped_pipeline_visualization(defects_df, pipe_diameter=None,  color_by="Depth (%)"):
    """
    Create an enhanced unwrapped cylinder visualization of pipeline defects,
    with Y-axis showing clock positions for intuitive interpretation.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - pipe_diameter: Pipeline diameter in meters (REQUIRED)
    
    Returns:Fif
    - Plotly figure object
    """
    
    if pipe_diameter is None:
        raise ValueError("pipe_diameter is required. Pass the actual pipe diameter from your dataset.")
    
    if pipe_diameter <= 0:
        raise ValueError(f"pipe_diameter must be positive, got {pipe_diameter}")

    max_points = 7000  # WebGL performance threshold
    
    if len(defects_df) > max_points:
        # Priority-based sampling: keep critical defects + representative sample
        
        # Always keep high-severity defects
        critical_mask = defects_df['depth [%]'] > 80
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

    # Extract axial values (X-axis) - unchanged
    x_vals = plot_data["log dist. [m]"].values
    
    # Convert clock positions to actual circumferential distances (for proper spacing)
    clock_hours = plot_data["clock_float"].values
    angles_radians = (clock_hours / 12.0) * 2 * np.pi # type: ignore
    circumferential_distance_m = (pipe_diameter / 2) * angles_radians
    
    # Y-axis uses circumferential distances for proportional spacing
    y_vals = circumferential_distance_m
    
    # === NEW: Handle different coloring methods ===
    if color_by == "Surface Location (Internal/External)": # type: ignore
        # Color by surface location
        if "surface location" not in plot_data.columns:
            # Fallback to depth if surface location not available
            color_by = "Depth (%)"
        else:
            surface_locations = plot_data["surface location"].fillna("Unknown")
            
            # Create color mapping
            color_map = {"INT": "red", "NON-INT": "deepskyblue"}
            colors = [color_map.get(loc, "gray") for loc in surface_locations]
            
            marker_props = dict(
                size=6,
                color=colors,
                opacity=0.8,
            )
    
    if color_by == "Depth (%)" or color_by not in ["Surface Location (Internal/External)"]:
        # Original depth-based coloring
        if "depth [%]" in plot_data.columns:
            depth_values = plot_data["depth [%]"].values
            marker_props = dict(
                size=6,
                color=depth_values,
                colorscale="Turbo",
                cmin=0,
                cmax=depth_values.max(), # type: ignore
                colorbar=dict(title="Depth (%)", thickness=15, len=0.6),
                opacity=0.8,
            )
        else:
            marker_props = dict(size=6, color="blue", opacity=0.8)
    
    # === Update hover template and custom data based on coloring method ===
    has_component = 'component / anomaly identification' in plot_data.columns
    
    if color_by == "Surface Location (Internal/External)":
        # Include surface location in hover
        if has_component:
            custom_data = np.column_stack([
                plot_data["joint number"].astype(str).values,
                plot_data["component / anomaly identification"].values,
                plot_data["depth [%]"].fillna(0).values,
                clock_hours,
                plot_data["surface location"].fillna("Unknown").values,
            ]) # type: ignore
            hover_template = (
                "<b>Distance:</b> %{x:.2f} m<br>"
                "<b>Clock Position:</b> %{customdata[3]:.1f}:00<br>"
                "<b>Depth:</b> %{customdata[2]:.1f}%<br>"
                "<b>Surface:</b> %{customdata[4]}<br>"
                "<b>Type:</b> %{customdata[1]}<br>"
                "<b>Joint:</b> %{customdata[0]}<extra></extra>"
            )
        else:
            custom_data = np.column_stack([
                plot_data["joint number"].astype(str).values,
                plot_data["depth [%]"].fillna(0).values,
                clock_hours,
                plot_data["surface location"].fillna("Unknown").values,
            ])  # type: ignore
            hover_template = (
                "<b>Distance:</b> %{x:.2f} m<br>"
                "<b>Clock Position:</b> %{customdata[2]:.1f}:00<br>"
                "<b>Depth:</b> %{customdata[1]:.1f}%<br>"
                "<b>Surface:</b> %{customdata[3]}<br>"
                "<b>Joint:</b> %{customdata[0]}<extra></extra>"
            )
    else:
        # Original hover template for depth coloring
        if has_component:
            custom_data = np.column_stack([
                plot_data["joint number"].astype(str).values,
                plot_data["component / anomaly identification"].values,
                plot_data["depth [%]"].fillna(0).values,
                clock_hours,
            ])  # type: ignore
            hover_template = (
                "<b>Distance:</b> %{x:.2f} m<br>"
                "<b>Clock Position:</b> %{customdata[3]:.1f}:00<br>"
                "<b>Depth:</b> %{customdata[2]:.1f}%<br>"
                "<b>Type:</b> %{customdata[1]}<br>"
                "<b>Joint:</b> %{customdata[0]}<extra></extra>"
            )
        else:
            custom_data = np.column_stack([
                plot_data["joint number"].astype(str).values,
                plot_data["depth [%]"].fillna(0).values,
                clock_hours,
            ])  # type: ignore
            hover_template = (
                "<b>Distance:</b> %{x:.2f} m<br>"
                "<b>Clock Position:</b> %{customdata[2]:.1f}:00<br>"
                "<b>Depth:</b> %{customdata[1]:.1f}%<br>"
                "<b>Joint:</b> %{customdata[0]}<extra></extra>"
            )
    
    # === Use WebGL for large datasets ===
    use_webgl = len(plot_data) > 1000
    scatter_class = go.Scattergl if use_webgl else go.Scatter
    
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
    
    # === Rest of the function remains the same ===
    # Calculate major clock positions and their circumferential equivalents
    major_clock_positions = [12, 3, 6, 9, 12]
    tick_positions = []
    tick_labels = []
    
    for clock_pos in major_clock_positions:
        angle_rad = (clock_pos / 12.0) * 2 * np.pi
        circ_distance = (pipe_diameter / 2) * angle_rad
        tick_positions.append(circ_distance)
        tick_labels.append(f"{clock_pos}:00")
    
    # Enhanced grid lines at major clock positions
    x_range = [x_vals.min() - 1, x_vals.max() + 1] # type: ignore
    
    for clock_pos, circ_distance in zip(major_clock_positions, tick_positions):
        fig.add_shape(
            type="line",
            x0=x_range[0], x1=x_range[1],
            y0=circ_distance, y1=circ_distance,
            line=dict(color="lightgray", width=1, dash="dot"),
            layer="below",
        )
    
    # Calculate total circumference for Y-axis range
    total_circumference = np.pi * pipe_diameter
    
    # === Update title based on coloring method ===
    title_suffix = ""
    if len(plot_data) != len(defects_df):
        title_suffix = f" - Showing {len(plot_data):,} of {len(defects_df):,} defects"
    
    if color_by == "Surface Location (Internal/External)":
        title = f"Unwrapped Pipeline Defect Map - Colored by Surface Location (Ø{pipe_diameter:.2f}m){title_suffix}"
    else:
        title = f"Unwrapped Pipeline Defect Map (Ø{pipe_diameter:.2f}m){title_suffix}"
    

    if color_by == "Surface Location (Internal/External)":
        for label, color in color_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=8, color=color),
                name=label,
                legendgroup=label,
                showlegend=True
            ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Axial Distance Along Pipeline (m)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            range=x_range
        ),
        yaxis=dict(
            title="Clock Position Around Pipe<br><sub>(Pipeline circumferential position)</sub>",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            range=[0, total_circumference],
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            dtick=None
        ),
        height=600,
        plot_bgcolor="white",
        hovermode="closest",
        uirevision="constant",
        dragmode="pan",
    )
    return fig