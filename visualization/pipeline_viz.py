import numpy as np
import plotly.graph_objects as go


def create_unwrapped_pipeline_visualization(defects_df, pipe_diameter=None, color_by="Depth (%)", y_axis_column="Clock Position"):
    """
    Create an enhanced unwrapped cylinder visualization of pipeline defects,
    with selectable Y-axis and color coding options.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - pipe_diameter: Pipeline diameter in meters (REQUIRED)
    - color_by: "Depth (%)" or "Surface Location (Internal/External)"
    - y_axis_column: "Clock Position", "Depth [%]", "Length [mm]", "Width [mm]", or "Distance from USGW [m]"
    
    Returns:
    - Plotly figure object
    """
    
    if pipe_diameter is None:
        raise ValueError("pipe_diameter is required. Pass the actual pipe diameter from your dataset.")
    
    if pipe_diameter <= 0:
        raise ValueError(f"pipe_diameter must be positive, got {pipe_diameter}")

    plot_data = defects_df.copy()

    # Extract axial values (X-axis) - unchanged
    x_vals = plot_data["log dist. [m]"].values
    
    # === Y-axis Selection Logic ===
    y_axis_info = get_y_axis_data(plot_data, y_axis_column, pipe_diameter)
    y_vals = y_axis_info['values']
    y_title = y_axis_info['title'] 
    y_unit = y_axis_info['unit']
    
    # === Color and Shape Selection Logic ===
    if color_by == "Surface Location (Internal/External)":
        # Color and shape by surface location
        if "surface location" not in plot_data.columns:
            # Fallback to depth if surface location not available
            color_by = "Depth (%)"
        else:
            surface_locations = plot_data["surface location"].fillna("Unknown")
            
            # Create separate traces for Internal and External with different shapes
            marker_props = None  # Will be handled per trace
            use_separate_traces = True
    else:
        use_separate_traces = False
    
    if not use_separate_traces:
        # Original depth-based coloring
        if "depth [%]" in plot_data.columns:
            depth_values = plot_data["depth [%]"].values
            marker_props = dict(
                size=6,
                color=depth_values,
                colorscale="Turbo",
                cmin=0,
                cmax=depth_values.max(),
                colorbar=dict(title="Depth (%)", thickness=15, len=0.6),
                opacity=0.8,
            )
        else:
            marker_props = dict(size=6, color="blue", opacity=0.8)
    
    # === Create Figure ===
    fig = go.Figure()
    
    # Use WebGL for large datasets
    use_webgl = len(plot_data) > 1000
    scatter_class = go.Scattergl if use_webgl else go.Scattergl
    
    # === Create Hover Template ===
    hover_template = create_hover_template(plot_data, y_axis_column, color_by)
    custom_data = create_custom_data(plot_data, y_axis_column, color_by)
    
    if use_separate_traces:
        # Create separate traces for Internal/External with different shapes
        surface_locations = plot_data["surface location"].fillna("Unknown")

        external_mask = surface_locations == "NON-INT"
        internal_mask = surface_locations == "INT"
        
        # Unknown/Other surface locations - Gray dots
        other_mask = ~(internal_mask | external_mask)
        if other_mask.any():
            fig.add_trace(
                scatter_class(
                    x=x_vals[other_mask],
                    y=y_vals[other_mask],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color="gray",
                        symbol="diamond",
                        opacity=0.6,
                    ),
                    customdata=custom_data[other_mask] if custom_data is not None else None,
                    hovertemplate=hover_template,
                    name="Unknown",
                    legendgroup="Unknown",
                    showlegend=True
                )
            )
        
        # External defects - Blue circles  
        if external_mask.any():
            fig.add_trace(
                scatter_class(
                    x=x_vals[external_mask],
                    y=y_vals[external_mask],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color="deepskyblue", 
                        symbol="circle",
                        opacity=0.6,
                    ),
                    customdata=custom_data[external_mask] if custom_data is not None else None,
                    hovertemplate=hover_template,
                    name="External", 
                    legendgroup="External",
                    showlegend=True
                )
            )

        # Internal defects - Red crosses
        if internal_mask.any():
            fig.add_trace(
                scatter_class(
                    x=x_vals[internal_mask],
                    y=y_vals[internal_mask],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="red",
                        symbol="x",
                        opacity=0.6,
                        line=dict(width=2)
                    ),
                    customdata=custom_data[internal_mask] if custom_data is not None else None,
                    hovertemplate=hover_template,
                    name="Internal",
                    legendgroup="Internal",
                    showlegend=True
                )
            )
    
    else:
        # Single trace with depth coloring
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
    
    # === Add Reference Lines (for Clock Position only) ===
    if y_axis_column == "Clock Position":
        add_clock_reference_lines(fig, x_vals, pipe_diameter)
    
    # === Update Layout ===
    update_figure_layout(fig, y_title, y_unit, color_by, y_axis_column, pipe_diameter, len(plot_data), len(defects_df))

    fig.update_xaxes(
        tickfont=dict(size=20),     
        title_font=dict(size=20, family='Arial', color='black')
    )
    fig.update_yaxes(
        tickfont=dict(size=20),       # Increase y-axis tick label size
        title_font=dict(size=20, family='Arial', color='black')
    )

    fig.update_layout(
        legend=dict(
            font=dict(size=16, family='Arial', color='black'),
            x=1,  # Move to left
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        margin=dict(l=80, r=40, t=80, b=80),
        font=dict(family='Arial', size=16),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    return fig


def get_y_axis_data(plot_data, y_axis_column, pipe_diameter):
    """
    Extract and process Y-axis data based on selection.
    
    Returns:
    - dict with 'values', 'title', 'unit'
    """
    if y_axis_column == "Clock Position":
        # Original clock position logic
        clock_hours = plot_data["clock_float"].values
        angles_radians = (clock_hours / 12.0) * 2 * np.pi
        circumferential_distance_m = (pipe_diameter / 2) * angles_radians
        
        return {
            'values': circumferential_distance_m,
            'title': 'Clock Position Around Pipe',
            'unit': '(Pipeline circumferential position)'
        }
    
    elif y_axis_column == "Depth [%]":
        return {
            'values': plot_data["depth [%]"].values,
            'title': 'Defect Depth',
            'unit': '(%)'
        }
    
    elif y_axis_column == "Length [mm]":
        return {
            'values': plot_data["length [mm]"].values,
            'title': 'Defect Length', 
            'unit': '(mm)'
        }
    
    elif y_axis_column == "Width [mm]":
        return {
            'values': plot_data["width [mm]"].values,
            'title': 'Defect Width',
            'unit': '(mm)'
        }
    
    elif y_axis_column == "Distance from USGW [m]":
        # Handle up weld distance with negative value conversion
        weld_distances = plot_data["up weld dist. [m]"].values
        
        # Check if all values are negative and convert to positive
        weld_distances = np.abs(weld_distances)
        
        return {
            'values': weld_distances,
            'title': 'Distance from Upstream Weld',
            'unit': '(m)'
        }
    
    else:
        raise ValueError(f"Unknown y_axis_column: {y_axis_column}")



def create_hover_template(plot_data, y_axis_column, color_by):
    """Create appropriate hover template based on selections."""
    
    has_component = 'component / anomaly identification' in plot_data.columns
    
    # Base template
    template_parts = [
        "<b>Distance:</b> %{x:.2f} m<br>",
    ]
    
    # Y-axis specific hover info
    if y_axis_column == "Clock Position":
        template_parts.append("<b>Clock Position:</b> %{customdata[0]:.1f}:00<br>")
        data_index = 1
    else:
        y_info = get_y_axis_info_for_hover(y_axis_column)
        template_parts.append(f"<b>{y_info['label']}:</b> %{{y:.{y_info['precision']}f}} {y_info['unit']}<br>")
        data_index = 0
    
    # Add depth info (always useful)
    if y_axis_column != "Depth [%]":  # Don't duplicate if Y-axis is already depth
        template_parts.append(f"<b>Depth:</b> %{{customdata[{data_index}]:.1f}}%<br>")
        data_index += 1
    
    # Add surface location for depth coloring mode
    if color_by == "Depth (%)":
        template_parts.append(f"<b>Surface:</b> %{{customdata[{data_index}]}}<br>")
        data_index += 1
    
    # Add component type if available
    if has_component:
        template_parts.append(f"<b>Type:</b> %{{customdata[{data_index}]}}<br>")
        data_index += 1
    
    # Add joint number
    template_parts.append(f"<b>Joint:</b> %{{customdata[{data_index}]}}<extra></extra>")
    
    return "".join(template_parts)



def get_y_axis_info_for_hover(y_axis_column):
    """Get formatting info for Y-axis values in hover."""
    info_map = {
        "Depth [%]": {"label": "Depth", "unit": "%", "precision": 1},
        "Length [mm]": {"label": "Length", "unit": "mm", "precision": 1}, 
        "Width [mm]": {"label": "Width", "unit": "mm", "precision": 1},
        "Distance from USGW [m]": {"label": "Distance from Weld", "unit": "m", "precision": 3}
    }
    return info_map.get(y_axis_column, {"label": "Value", "unit": "", "precision": 2})



def create_custom_data(plot_data, y_axis_column, color_by):
    """Create custom data array for hover information."""
    
    has_component = 'component / anomaly identification' in plot_data.columns
    data_columns = []
    
    # Add clock hours for clock position mode
    if y_axis_column == "Clock Position":
        data_columns.append(plot_data["clock_float"].values)
    
    # Add depth (unless Y-axis is already depth)
    if y_axis_column != "Depth [%]":
        data_columns.append(plot_data["depth [%]"].fillna(0).values)
    
    # Add surface location for depth coloring mode
    if color_by == "Depth (%)":
        data_columns.append(plot_data["surface location"].fillna("Unknown").values)
    
    # Add component type if available
    if has_component:
        data_columns.append(plot_data["component / anomaly identification"].values)
    
    # Add joint number
    data_columns.append(plot_data["joint number"].astype(str).values)
    
    if data_columns:
        return np.column_stack(data_columns)
    else:
        return None


def add_clock_reference_lines(fig, x_vals, pipe_diameter):
    """Add reference lines for clock position visualization."""
    
    major_clock_positions = [12, 3, 6, 9, 12] 
    x_range = [x_vals.min() - 1, x_vals.max() + 1]
    
    for clock_pos in major_clock_positions:
        angle_rad = (clock_pos / 12.0) * 2 * np.pi
        circ_distance = (pipe_diameter / 2) * angle_rad
        
        fig.add_shape(
            type="line",
            x0=x_range[0], x1=x_range[1],
            y0=circ_distance, y1=circ_distance,
            line=dict(color="lightgray", width=1, dash="dot"),
            layer="below",
        )


def update_figure_layout(fig, y_title, y_unit, color_by, y_axis_column, pipe_diameter, plot_data_len, total_defects_len):
    """Update figure layout with appropriate titles and formatting."""
    
    # Create title with context
    title_suffix = ""
    if plot_data_len != total_defects_len:
        title_suffix = f" - Showing {plot_data_len:,} of {total_defects_len:,} defects"
    
    if color_by == "Surface Location (Internal/External)":
        color_context = "Shaped by Surface Location" 
    else:
        color_context = "Colored by Depth"
    
    title = f"Pipeline Defect Map - {y_title} vs Axial Distance ({color_context}){title_suffix}"
    
    # Y-axis formatting based on type
    y_axis_config = {
        "title": f"{y_title}<br><sub>{y_unit}</sub>",
        "showgrid": True,
        "gridcolor": "rgba(200, 200, 200, 0.3)"
    }
    
    # Special formatting for clock position
    if y_axis_column == "Clock Position":
        major_clock_positions = [12, 3, 6, 9, 12]
        tick_positions = []
        tick_labels = []
        
        for clock_pos in major_clock_positions:
            angle_rad = (clock_pos / 12.0) * 2 * np.pi
            circ_distance = (pipe_diameter / 2) * angle_rad
            tick_positions.append(circ_distance)
            tick_labels.append(f"{clock_pos}:00")
        
        total_circumference = np.pi * pipe_diameter
        
        y_axis_config.update({
            "range": [0, total_circumference],
            "tickmode": 'array',
            "tickvals": tick_positions,
            "ticktext": tick_labels,
            "dtick": None
        })
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Axial Distance Along Pipeline (m)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
        ),
        yaxis=y_axis_config,
        height=600,
        plot_bgcolor="white",
        hovermode="closest",
        uirevision="constant",
        dragmode="pan",
    )