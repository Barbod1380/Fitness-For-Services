# comparison_viz.py

"""
Functions for creating comparison visualizations between different inspection datasets.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def create_comparison_stats_plot(comparison_results):
    """
    Create a pie chart showing new vs. common defects.

    Parameters:
    - comparison_results: Results dictionary from compare_defects function

    Returns:
    - Plotly figure object
    """
    labels = ["Common Defects", "New Defects"]
    values = [
        comparison_results["common_defects_count"],
        comparison_results["new_defects_count"],
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                textinfo="label+percent",
                marker=dict(colors=["#2E86C1", "#EC7063"]),
            )
        ]
    )

    fig.update_layout(
        title="Distribution of Common vs. New Defects",
        font=dict(size=14),
        height=400,
    )

    return fig


def create_new_defect_types_plot(comparison_results):
    """
    Create a bar chart showing distribution of new defect types.

    Parameters:
    - comparison_results: Results dictionary from compare_defects function

    Returns:
    - Plotly figure object
    """
    type_dist = comparison_results["defect_type_distribution"]

    # If there are no new defects, show a notice
    if type_dist.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No new defects found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    # Sort by count descending
    type_dist = type_dist.sort_values("count", ascending=False)

    fig = go.Figure(
        data=[
            go.Bar(
                x=type_dist["defect_type"],
                y=type_dist["count"],
                text=type_dist["percentage"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto",
                marker_color="#EC7063",
            )
        ]
    )

    fig.update_layout(
        title="Distribution of New Defect Types",
        xaxis_title="Defect Type",
        yaxis_title="Count",
        font=dict(size=14),
        height=500,
        xaxis=dict(tickangle=-45),  # Rotate x labels for readability
    )

    return fig


def create_growth_rate_histogram(comparison_results, dimension="depth"):
    """
    Create a histogram showing the distribution of positive growth rates.

    Parameters:
    - comparison_results: Results dictionary from compare_defects function
    - dimension: Which dimension to plot ('depth', 'length', 'width')

    Returns:
    - Plotly figure object
    """
    # Dimension-specific flags and column names
    dim_config = {
        "depth": {
            "has_data_flag": "has_depth_data",
            "negative_flag": "is_negative_growth",
            "corrected_flag": "is_corrected_depth",
        },
        "length": {
            "has_data_flag": "has_length_data",
            "negative_flag": "is_negative_length_growth",
            "corrected_flag": "is_corrected_length",
        },
        "width": {
            "has_data_flag": "has_width_data",
            "negative_flag": "is_negative_width_growth",
            "corrected_flag": "is_corrected_width",
        },
    }
    config = dim_config.get(dimension, dim_config["depth"])

    # If no data or no growth calculation requested, show notice
    if (
        not comparison_results.get(config["has_data_flag"], False)
        or comparison_results["matches_df"].empty
        or not comparison_results.get("calculate_growth", False)
    ):
        fig = go.Figure()
        fig.add_annotation(
            text=f"No {dimension} growth rate data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    matches_df = comparison_results["matches_df"]

    # Determine which growth column and x-axis title to use
    if dimension == "depth":
        if comparison_results.get("has_wt_data", False):
            growth_col = "growth_rate_mm_per_year"
            x_title = "Depth Growth Rate (mm/year)"
        else:
            growth_col = "growth_rate_pct_per_year"
            x_title = "Depth Growth Rate (% points/year)"
    elif dimension == "length":
        growth_col = "length_growth_rate_mm_per_year"
        x_title = "Length Growth Rate (mm/year)"
    else:  # dimension == "width"
        growth_col = "width_growth_rate_mm_per_year"
        x_title = "Width Growth Rate (mm/year)"

    # Filter out negative-growth rows; include corrected if flagged
    if config["corrected_flag"] in matches_df.columns:
        positive_growth = matches_df[~matches_df[config["negative_flag"]]]
        natural_positive = positive_growth[
            ~positive_growth.get(config["corrected_flag"], False)
        ]
        corrected_growth = positive_growth[
            positive_growth.get(config["corrected_flag"], False)
        ]
        has_corrected = not corrected_growth.empty
    else:
        positive_growth = matches_df[~matches_df[config["negative_flag"]]]
        has_corrected = False

    if positive_growth.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No positive {dimension} growth data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    # Build histogram
    fig = go.Figure()

    if has_corrected:
        if not natural_positive.empty:
            fig.add_trace(
                go.Histogram(
                    x=natural_positive[growth_col],
                    nbinsx=20,
                    name="Natural Positive Growth",
                    marker=dict(
                        color="rgba(0, 0, 255, 0.6)",
                        line=dict(color="rgba(0, 0, 255, 1)", width=1),
                    ),
                )
            )
        if not corrected_growth.empty:
            fig.add_trace(
                go.Histogram(
                    x=corrected_growth[growth_col],
                    nbinsx=20,
                    name="Corrected Growth (formerly negative)",
                    marker=dict(
                        color="rgba(0, 180, 0, 0.6)",
                        line=dict(color="rgba(0, 180, 0, 1)", width=1),
                    ),
                )
            )
        fig.update_layout(barmode="overlay")
        fig.update_traces(opacity=0.7)
    else:
        fig.add_trace(
            go.Histogram(
                x=positive_growth[growth_col],
                nbinsx=20,
                name=f"Positive {dimension.title()} Growth Rates",
                marker=dict(
                    color="rgba(0, 0, 255, 0.7)",
                    line=dict(color="rgba(0, 0, 255, 1)", width=1),
                ),
            )
        )

    # Add vertical line at average growth rate
    mean_growth = positive_growth[growth_col].mean()

    fig.add_shape(
        type="line",
        x0=mean_growth,
        x1=mean_growth,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=mean_growth,
        y=1,
        yref="paper",
        text=f"Mean: {mean_growth:.3f}",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-30,
    )

    if has_corrected:
        title = (
            f"Distribution of {dimension.title()} Growth Rates "
            "(Including Corrected Negative Growth)"
        )
    else:
        title = f"Distribution of Positive {dimension.title()} Growth Rates"

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Count",
        bargap=0.1,
        height=500,
    )

    return fig


def create_negative_growth_plot(comparison_results, dimension="depth"):
    """
    Create a scatter plot highlighting negative growth defects and corrected points.

    Parameters:
    - comparison_results: Results dictionary from compare_defects function
    - dimension: Which dimension to plot ('depth', 'length', 'width')

    Returns:
    - Plotly figure object
    """
    # Set dimension-specific flags, columns, and titles
    if dimension == "depth":
        has_data_flag = "has_depth_data"
        negative_flag = "is_negative_growth"
        corrected_flag = "is_corrected"
        title = "Defect Depth Growth Rate vs Location"

        if comparison_results.get("has_wt_data", False):
            old_col = "old_depth_mm"
            new_col = "new_depth_mm"
            growth_col = "growth_rate_mm_per_year"
            y_title = "Depth Growth Rate (mm/year)"
            unit = " mm"
        else:
            old_col = "old_depth_pct"
            new_col = "new_depth_pct"
            growth_col = "growth_rate_pct_per_year"
            y_title = "Depth Growth Rate (% points/year)"
            unit = " %"
    elif dimension == "length":
        has_data_flag = "has_length_data"
        negative_flag = "is_negative_length_growth"
        corrected_flag = "is_corrected_length"
        old_col = "old_length_mm"
        new_col = "new_length_mm"
        growth_col = "length_growth_rate_mm_per_year"
        y_title = "Length Growth Rate (mm/year)"
        unit = " mm"
        title = "Defect Length Growth Rate vs Location"
    else:  # dimension == "width"
        has_data_flag = "has_width_data"
        negative_flag = "is_negative_width_growth"
        corrected_flag = "is_corrected_width"
        old_col = "old_width_mm"
        new_col = "new_width_mm"
        growth_col = "width_growth_rate_mm_per_year"
        y_title = "Width Growth Rate (mm/year)"
        unit = " mm"
        title = "Defect Width Growth Rate vs Location"

    # If no data or no growth calculation requested, show notice
    if (
        not comparison_results.get(has_data_flag, False)
        or comparison_results["matches_df"].empty
        or not comparison_results.get("calculate_growth", False)
    ):
        fig = go.Figure()
        fig.add_annotation(
            text=f"No {dimension} growth rate data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    matches_df = comparison_results["matches_df"]

    # If required flags/columns not present, show notice
    if negative_flag not in matches_df.columns or growth_col not in matches_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Required columns for {dimension} analysis not found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    # Determine if any corrected points exist (only relevant for depth)
    has_corrected = (
        dimension == "depth"
        and corrected_flag in matches_df.columns
        and matches_df[corrected_flag].any()
    )

    # Split matches into positive, negative, and corrected growth subsets
    if has_corrected:
        positive_growth = matches_df[
            ~matches_df[negative_flag] & ~matches_df[corrected_flag]
        ]
        negative_growth = matches_df[matches_df[negative_flag]]
        corrected_growth = matches_df[matches_df[corrected_flag]]
    else:
        positive_growth = matches_df[~matches_df[negative_flag]]
        negative_growth = matches_df[matches_df[negative_flag]]
        corrected_growth = pd.DataFrame()  # empty DataFrame

    if negative_growth.empty and positive_growth.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No {dimension} growth data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    fig = go.Figure()
    has_joint_num = "joint number" in matches_df.columns

    # Helper to add a trace for a given subset
    def add_scatter_trace(subset, name, color, symbol):
        if subset.empty:
            return

        trace_kwargs = {
            "x": subset["log_dist"],
            "y": subset[growth_col],
            "mode": "markers",
            "marker": dict(size=10, color=color, opacity=0.7, symbol=symbol),
            "name": name,
        }

        # Build hovertemplate & customdata depending on joint number presence
        if has_joint_num:
            hovertemplate = (
                "<b>Location:</b> %{x:.2f} m<br>"
                f"<b>Growth Rate:</b> %{{y:.3f}} {y_title.split()[-1]}<br>"
                f"<b>Old {dimension.title()}:</b> %{{customdata[0]:.2f}}{unit}<br>"
                f"<b>New {dimension.title()}:</b> %{{customdata[1]:.2f}}{unit}<br>"
                "<b>Type:</b> %{customdata[2]}<br>"
                "<b>Joint:</b> %{customdata[3]}<extra></extra>"
            )
            customdata = np.column_stack(
                (
                    subset[old_col],
                    subset[new_col],
                    subset["defect_type"],
                    subset["joint number"],
                )
            )
        else:
            hovertemplate = (
                "<b>Location:</b> %{x:.2f} m<br>"
                f"<b>Growth Rate:</b> %{{y:.3f}} {y_title.split()[-1]}<br>"
                f"<b>Old {dimension.title()}:</b> %{{customdata[0]:.2f}}{unit}<br>"
                f"<b>New {dimension.title()}:</b> %{{customdata[1]:.2f}}{unit}<br>"
                "<b>Type:</b> %{customdata[2]}<extra></extra>"
            )
            customdata = np.column_stack(
                (
                    subset[old_col],
                    subset[new_col],
                    subset["defect_type"],
                )
            )

        fig.add_trace(
            go.Scatter(
                **trace_kwargs,
                hovertemplate=hovertemplate,
                customdata=customdata,
                showlegend=True,
            )
        )

    # Add positive, negative, and corrected traces
    add_scatter_trace(positive_growth, "Positive Growth", "blue", "circle")
    add_scatter_trace(negative_growth, "Negative Growth (Anomaly)", "red", "triangle-down")
    add_scatter_trace(corrected_growth, "Corrected Growth", "green", "diamond")

    # Add a zero line across the full x-range (if matches_df has values)
    if not matches_df.empty:
        fig.add_shape(
            type="line",
            x0=matches_df["log_dist"].min(),
            x1=matches_df["log_dist"].max(),
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title=y_title,
        height=500,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_multi_dimensional_growth_plot(comparison_results):
    """
    Create a combined plot showing negative growth for all dimensions.

    Parameters:
    - comparison_results: Results dictionary from compare_defects function

    Returns:
    - Plotly figure object with subplots for each dimension
    """
    from plotly.subplots import make_subplots

    # Determine which dimensions have data
    dimensions = []
    if comparison_results.get("has_depth_data", False):
        dimensions.append("depth")
    if comparison_results.get("has_length_data", False):
        dimensions.append("length")
    if comparison_results.get("has_width_data", False):
        dimensions.append("width")

    if not dimensions:
        fig = go.Figure()
        fig.add_annotation(
            text="No growth rate data available for any dimension",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    # Create one row per dimension
    fig = make_subplots(
        rows=len(dimensions),
        cols=1,
        subplot_titles=[f"{dim.title()} Growth Rate Analysis" for dim in dimensions],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    matches_df = comparison_results["matches_df"]

    for idx, dim in enumerate(dimensions, start=1):
        # Map flags and column names per dimension
        if dim == "depth":
            negative_flag = "is_negative_growth"
            corrected_flag = "is_corrected_depth"
            if comparison_results.get("has_wt_data", False):
                growth_col = "growth_rate_mm_per_year"
                y_title = "Growth Rate (mm/year)"
            else:
                growth_col = "growth_rate_pct_per_year"
                y_title = "Growth Rate (%/year)"
        elif dim == "length":
            negative_flag = "is_negative_length_growth"
            corrected_flag = "is_corrected_length"
            growth_col = "length_growth_rate_mm_per_year"
            y_title = "Length Growth Rate (mm/year)"
        else:  # dim == "width"
            negative_flag = "is_negative_width_growth"
            corrected_flag = "is_corrected_width"
            growth_col = "width_growth_rate_mm_per_year"
            y_title = "Width Growth Rate (mm/year)"

        # Split matches into positive, negative, and corrected
        has_corr = (
            corrected_flag in matches_df.columns and matches_df[corrected_flag].any()
        )
        if has_corr:
            positive_growth = matches_df[
                ~matches_df[negative_flag] & ~matches_df[corrected_flag]
            ]
            negative_growth = matches_df[matches_df[negative_flag]]
            corrected_growth = matches_df[matches_df[corrected_flag]]
        else:
            positive_growth = matches_df[~matches_df[negative_flag]]
            negative_growth = matches_df[matches_df[negative_flag]]
            corrected_growth = pd.DataFrame()

        # Add scatter traces to the subplot (row=idx)
        if not positive_growth.empty:
            fig.add_trace(
                go.Scatter(
                    x=positive_growth["log_dist"],
                    y=positive_growth[growth_col],
                    mode="markers",
                    marker=dict(size=8, color="blue", opacity=0.6),
                    name=f"Positive {dim.title()}" if idx == 1 else None,
                    showlegend=(idx == 1),
                    legendgroup="positive",
                ),
                row=idx,
                col=1,
            )
        if not negative_growth.empty:
            fig.add_trace(
                go.Scatter(
                    x=negative_growth["log_dist"],
                    y=negative_growth[growth_col],
                    mode="markers",
                    marker=dict(
                        size=10, color="red", opacity=0.7, symbol="triangle-down"
                    ),
                    name=f"Negative {dim.title()}" if idx == 1 else None,
                    showlegend=(idx == 1),
                    legendgroup="negative",
                ),
                row=idx,
                col=1,
            )
        if not corrected_growth.empty:
            fig.add_trace(
                go.Scatter(
                    x=corrected_growth["log_dist"],
                    y=corrected_growth[growth_col],
                    mode="markers",
                    marker=dict(
                        size=10, color="green", opacity=0.7, symbol="diamond"
                    ),
                    name=f"Corrected {dim.title()}" if idx == 1 else None,
                    showlegend=(idx == 1),
                    legendgroup="corrected",
                ),
                row=idx,
                col=1,
            )

        # Add a zero line
        fig.add_shape(
            type="line",
            x0=matches_df["log_dist"].min(),
            x1=matches_df["log_dist"].max(),
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=idx,
            col=1,
        )

        # Y-axis label for this subplot
        fig.update_yaxes(title_text=y_title, row=idx, col=1)

    # Final layout tweaks
    fig.update_xaxes(
        title_text="Distance Along Pipeline (m)", row=len(dimensions), col=1
    )
    fig.update_layout(
        title="Multi-Dimensional Growth Rate Analysis",
        height=400 * len(dimensions),
        hovermode="closest",
    )

    return fig