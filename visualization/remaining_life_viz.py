# remaining_life_viz.py

"""
Visualization functions for remaining life analysis of pipeline defects.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _empty_fig(message):
    """
    Return a Plotly figure with a centered annotation (used for 'no data' cases).
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20),
    )
    return fig


def create_remaining_life_pipeline_visualization(remaining_life_results):
    """
    Create an interactive pipeline visualization colored by remaining life years.
    
    Parameters:
    - remaining_life_results: Dictionary from calculate_remaining_life_analysis
    
    Returns:
    - Plotly figure object
    """
    if not remaining_life_results.get("analysis_possible", False):
        return _empty_fig("Remaining life analysis not possible with current data")

    # Combine matched and new defect analyses
    analyses = (
        remaining_life_results.get("matched_defects_analysis", [])
        + remaining_life_results.get("new_defects_analysis", [])
    )
    if not analyses:
        return _empty_fig("No defects available for remaining life analysis")

    df = pd.DataFrame(analyses)
    # Cap infinite remaining life at 100 years for display
    df["remaining_life_display"] = df["remaining_life_years"].replace([np.inf], 100)

    color_map = {
        "CRITICAL": "red",
        "HIGH_RISK": "orange",
        "MEDIUM_RISK": "yellow",
        "LOW_RISK": "green",
        "STABLE": "blue",
        "ERROR": "gray",
    }

    fig = go.Figure()

    # Add one scatter trace per status category
    for status, color in color_map.items():
        status_data = df[df["status"] == status]
        if status_data.empty:
            continue

        hover_text = []
        for _, row in status_data.iterrows():
            life_val = row["remaining_life_years"]
            life_text = "Stable (>100 years)" if np.isinf(life_val) else f"{life_val:.1f} years"
            growth_src = "📊 Measured" if row["growth_rate_source"] == "MEASURED" else "📈 Estimated"
            hover_text.append(
                f"<b>Location:</b> {row['log_dist']:.2f}m<br>"
                f"<b>Remaining Life:</b> {life_text}<br>"
                f"<b>Current Depth:</b> {row['current_depth_pct']:.1f}%<br>"
                f"<b>Growth Rate:</b> {row['growth_rate_pct_per_year']:.2f}%/year<br>"
                f"<b>Defect Type:</b> {row['defect_type']}<br>"
                f"<b>Joint:</b> {row['joint_number']}<br>"
                f"<b>Growth Data:</b> {growth_src}<br>"
                f"<b>Status:</b> {status.replace('_', ' ').title()}"
            )

        fig.add_trace(
            go.Scatter(
                x=status_data["log_dist"],
                y=[1] * len(status_data),  # draw all points at y=1
                mode="markers",
                marker=dict(size=12, color=color, opacity=0.8, line=dict(width=2, color="black")),
                name=f"{status.replace('_', ' ').title()} ({len(status_data)})",
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Pipeline Remaining Life Analysis<br><sub>Hover over points to see detailed information</sub>",
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title="Pipeline Representation",
        height=500,
        hovermode="closest",
        yaxis=dict(range=[0.5, 1.5], showticklabels=False, showgrid=False, zeroline=False),
        xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
        plot_bgcolor="white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )

    return fig


def create_remaining_life_histogram(remaining_life_results):
    """
    Create a histogram showing distribution of remaining life years.
    
    Parameters:
    - remaining_life_results: Dictionary from calculate_remaining_life_analysis
    
    Returns:
    - Plotly figure object
    """
    if not remaining_life_results.get("analysis_possible", False):
        return _empty_fig("No data available for histogram")

    analyses = (
        remaining_life_results.get("matched_defects_analysis", [])
        + remaining_life_results.get("new_defects_analysis", [])
    )
    if not analyses:
        return _empty_fig("No data available for histogram")

    df = pd.DataFrame(analyses)
    finite_data = df[np.isfinite(df["remaining_life_years"]) & (df["status"] != "ERROR")]

    if finite_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No finite remaining life values to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    fig = go.Figure()

    # Histogram for measured growth rates
    measured = finite_data[finite_data["growth_rate_source"] == "MEASURED"]
    if not measured.empty:
        fig.add_trace(
            go.Histogram(
                x=measured["remaining_life_years"],
                nbinsx=20,
                name="Measured Growth Rates",
                marker=dict(color="rgba(0, 100, 200, 0.7)", line=dict(color="rgba(0, 100, 200, 1)", width=1)),
                opacity=0.7,
            )
        )

    # Histogram for estimated growth rates
    estimated = finite_data[finite_data["growth_rate_source"] == "ESTIMATED"]
    if not estimated.empty:
        fig.add_trace(
            go.Histogram(
                x=estimated["remaining_life_years"],
                nbinsx=20,
                name="Estimated Growth Rates",
                marker=dict(color="rgba(200, 100, 0, 0.7)", line=dict(color="rgba(200, 100, 0, 1)", width=1)),
                opacity=0.7,
            )
        )

    # Add vertical lines for risk thresholds
    fig.add_shape(type="line", x0=2, x1=2, y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=10, x1=10, y0=0, y1=1, yref="paper", line=dict(color="orange", width=2, dash="dash"))

    fig.add_annotation(
        x=2,
        y=1,
        yref="paper",
        text="High Risk<br>Threshold",
        showarrow=True,
        arrowhead=1,
        ax=-30,
        ay=-30,
        font=dict(color="red", size=10),
    )
    fig.add_annotation(
        x=10,
        y=1,
        yref="paper",
        text="Medium Risk<br>Threshold",
        showarrow=True,
        arrowhead=1,
        ax=30,
        ay=-30,
        font=dict(color="orange", size=10),
    )

    fig.update_layout(
        title="Distribution of Remaining Life Until Critical Depth (80%)",
        xaxis_title="Remaining Life (Years)",
        yaxis_title="Number of Defects",
        barmode="overlay",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_remaining_life_summary_table(remaining_life_results):
    """
    Create a summary table of remaining life analysis results.
    
    Parameters:
    - remaining_life_results: Dictionary from calculate_remaining_life_analysis
    
    Returns:
    - Pandas DataFrame for display
    """
    if not remaining_life_results.get("analysis_possible", False):
        return pd.DataFrame([{"Metric": "Analysis Status", "Value": "Not Possible - Missing Data"}])

    summary_stats = remaining_life_results.get("summary_statistics", {})
    if not summary_stats:
        return pd.DataFrame([{"Metric": "Analysis Status", "Value": "No Data Available"}])

    rows = [
        {
            "Metric": "Total Defects Analyzed",
            "Value": f"{summary_stats.get('total_defects_analyzed', 0)}",
        },
        {
            "Metric": "Defects with Measured Growth",
            "Value": f"{summary_stats.get('defects_with_measured_growth', 0)}",
        },
        {
            "Metric": "Defects with Estimated Growth",
            "Value": f"{summary_stats.get('defects_with_estimated_growth', 0)}",
        },
    ]

    avg = summary_stats.get("average_remaining_life_years", float("nan"))
    if not np.isnan(avg):
        rows.append({"Metric": "Average Remaining Life", "Value": f"{avg:.1f} years"})

    median = summary_stats.get("median_remaining_life_years", float("nan"))
    if not np.isnan(median):
        rows.append({"Metric": "Median Remaining Life", "Value": f"{median:.1f} years"})

    min_life = summary_stats.get("min_remaining_life_years", float("nan"))
    if not np.isnan(min_life):
        rows.append({"Metric": "Shortest Remaining Life", "Value": f"{min_life:.1f} years"})

    for status, count in summary_stats.get("status_distribution", {}).items():
        rows.append(
            {"Metric": f"{status.replace('_', ' ').title()} Defects", "Value": f"{count}"}
        )

    return pd.DataFrame(rows)


def create_remaining_life_risk_matrix(remaining_life_results):
    """
    Create a risk matrix visualization showing current condition vs remaining life.
    
    Parameters:
    - remaining_life_results: Dictionary from calculate_remaining_life_analysis
    
    Returns:
    - Plotly figure object
    """
    if not remaining_life_results.get("analysis_possible", False):
        return _empty_fig("Risk matrix not available with current data")

    analyses = (
        remaining_life_results.get("matched_defects_analysis", [])
        + remaining_life_results.get("new_defects_analysis", [])
    )
    if not analyses:
        return go.Figure()

    df = pd.DataFrame(analyses)
    valid = df[(df["status"] != "ERROR") & np.isfinite(df["remaining_life_years"])]
    if valid.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data for risk matrix",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    # Cap remaining life at 50 years for display
    valid["remaining_life_capped"] = valid["remaining_life_years"].clip(upper=50)

    color_map = {
        "CRITICAL": "red",
        "HIGH_RISK": "orange",
        "MEDIUM_RISK": "yellow",
        "LOW_RISK": "green",
        "STABLE": "blue",
    }

    fig = go.Figure()
    for status, color in color_map.items():
        subset = valid[valid["status"] == status]
        if subset.empty:
            continue

        hover_text = [
            f"<b>Location:</b> {row['log_dist']:.2f}m<br>"
            f"<b>Current Depth:</b> {row['current_depth_pct']:.1f}%<br>"
            f"<b>Remaining Life:</b> {row['remaining_life_years']:.1f} years<br>"
            f"<b>Defect Type:</b> {row['defect_type']}<br>"
            f"<b>Growth Source:</b> {row['growth_rate_source']}"
            for _, row in subset.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=subset["current_depth_pct"],
                y=subset["remaining_life_capped"],
                mode="markers",
                marker=dict(size=8, color=color, opacity=0.7, line=dict(width=1, color="black")),
                name=status.replace("_", " ").title(),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Risk-zone rectangles (red for <=2 years, orange for 2–10 years)
    fig.add_shape(
        type="rect",
        x0=0,
        x1=100,
        y0=0,
        y1=2,
        fillcolor="rgba(255, 0, 0, 0.1)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=0,
        x1=100,
        y0=2,
        y1=10,
        fillcolor="rgba(255, 165, 0, 0.1)",
        line=dict(width=0),
        layer="below",
    )

    fig.update_layout(
        title="Risk Matrix: Current Condition vs Remaining Life",
        xaxis_title="Current Depth (% of Wall Thickness)",
        yaxis_title="Remaining Life (Years, capped at 50)",
        height=500,
        hovermode="closest",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 50]),
        annotations=[
            dict(
                x=50,
                y=1,
                text="HIGH RISK ZONE",
                showarrow=False,
                font=dict(color="red", size=12, family="Arial Black"),
            ),
            dict(
                x=50,
                y=6,
                text="MEDIUM RISK ZONE",
                showarrow=False,
                font=dict(color="orange", size=12, family="Arial Black"),
            ),
        ],
    )

    return fig