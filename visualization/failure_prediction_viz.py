"""
Visualization components for failure prediction analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_failure_prediction_chart(results: dict) -> go.Figure:
    """
    Create a dual-bar chart showing ERF and depth failures by year.
    
    Parameters:
    - results: Results dictionary from predict_joint_failures_over_time
    
    Returns:
    - Plotly figure with dual bar chart
    """
    
    years = results['years']
    erf_failures = results['erf_failures_by_year']
    depth_failures = results['depth_failures_by_year']
    method_name = results['assessment_method'].replace('_', ' ').title()
    
    # Create figure with secondary y-axis for percentages
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]] ,
        subplot_titles=[f"Joint Failure Prediction - {method_name} Method"]
    )
    
    # Add ERF failures bar
    fig.add_trace(
        go.Bar(
            x=years,
            y=erf_failures,
            name=f'ERF < 1.0 Failures',
            marker_color='#E74C3C',  # Red
            opacity=0.8,
            hovertemplate=(
                "<b>Year %{x}</b><br>"
                "ERF Failures: %{y} joints<br>"
                f"Operating Pressure: {results['operating_pressure_mpa']:.1f} MPa<br>"
                f"Method: {method_name}"
                "<extra></extra>"
            )
        ),
        secondary_y=False
    )
    
    # Add depth failures bar
    fig.add_trace(
        go.Bar(
            x=years,
            y=depth_failures,
            name='Depth > 80% Failures',
            marker_color='#3498DB',  # Blue
            opacity=0.8,
            hovertemplate=(
                "<b>Year %{x}</b><br>"
                "Depth Failures: %{y} joints<br>"
                "Criteria: >80% wall thickness<br>"
                "<extra></extra>"
            )
        ),
        secondary_y=False
    )
    
    # Add cumulative failure lines
    cumulative_erf = results['cumulative_erf_failures']
    cumulative_depth = results['cumulative_depth_failures']
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=cumulative_erf,
            mode='lines+markers',
            name='Cumulative ERF Failures',
            line=dict(color='#C0392B', width=3, dash='dash'),
            marker=dict(size=6),
            yaxis='y2',
            hovertemplate=(
                "<b>Year %{x}</b><br>"
                "Total ERF Failures: %{y} joints<br>"
                "<extra></extra>"
            )
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=cumulative_depth,
            mode='lines+markers', 
            name='Cumulative Depth Failures',
            line=dict(color='#2E86AB', width=3, dash='dash'),
            marker=dict(size=6),
            yaxis='y2',
            hovertemplate=(
                "<b>Year %{x}</b><br>"
                "Total Depth Failures: %{y} joints<br>"
                "<extra></extra>"
            )
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Pipeline Joint Failure Prediction<br><sub>Assessment Method: {method_name} | Operating Pressure: {results['operating_pressure_mpa']:.1f} MPa</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Years from Now",
        barmode='group',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white'
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="Number of Failed Joints (Annual)",
        secondary_y=False,
        gridcolor='rgba(128,128,128,0.2)',
        showgrid=True
    )
    
    fig.update_yaxes(
        title_text="Cumulative Failed Joints",
        secondary_y=True,
        gridcolor='rgba(128,128,128,0.1)',
        showgrid=False
    )
    
    fig.update_xaxes(
        gridcolor='rgba(128,128,128,0.2)',
        showgrid=True,
        tick0=1,
        dtick=1
    )
    
    return fig


def create_failure_summary_metrics(results: dict) -> dict:
    """
    Create summary metrics for display in the UI.
    
    Parameters:
    - results: Results dictionary from predict_joint_failures_over_time
    
    Returns:
    - Dictionary with formatted metrics
    """
    
    summary = results['summary']
    
    metrics = {
        'total_joints': {
            'label': 'Total Joints',
            'value': f"{summary['total_joints_analyzed']:,}",
            'description': 'Joints in pipeline'
        },
        'joints_with_defects': {
            'label': 'Joints with Defects', 
            'value': f"{summary['joints_with_defects']:,}",
            'description': 'Joints containing defects'
        },
        'max_erf_failures': {
            'label': 'Max ERF Failures',
            'value': f"{summary['max_erf_failures']:,}",
            'description': f"Over {summary['prediction_window_years']} years",
            'status': 'critical' if summary['pct_erf_failures'] > 10 else 'warning' if summary['pct_erf_failures'] > 5 else 'safe'
        },
        'max_depth_failures': {
            'label': 'Max Depth Failures',
            'value': f"{summary['max_depth_failures']:,}",
            'description': f"Over {summary['prediction_window_years']} years",
            'status': 'critical' if summary['pct_depth_failures'] > 10 else 'warning' if summary['pct_depth_failures'] > 5 else 'safe'
        },
        'first_failure_year': {
            'label': 'First Predicted Failure',
            'value': _format_first_failure_year(summary),
            'description': 'Earliest failure expected'
        }
    }
    
    return metrics


def _format_first_failure_year(summary: dict) -> str:
    """Format the first failure year for display."""
    
    erf_year = summary.get('first_erf_failure_year')
    depth_year = summary.get('first_depth_failure_year')
    
    if erf_year is None and depth_year is None:
        return "No failures predicted"
    elif erf_year is None:
        return f"Year {depth_year} (Depth)"
    elif depth_year is None:
        return f"Year {erf_year} (ERF)"
    else:
        earliest = min(erf_year, depth_year)
        failure_type = "ERF" if earliest == erf_year else "Depth"
        return f"Year {earliest} ({failure_type})"


def create_failure_details_table(results: dict, max_year: int = 5) -> pd.DataFrame:
    """
    Create a detailed table of failures for the first few years.
    
    Parameters:
    - results: Results dictionary from predict_joint_failures_over_time
    - max_year: Maximum year to include in table
    
    Returns:
    - DataFrame with failure details
    """
    
    details_data = []
    
    for year_data in results['failure_details']:
        year = year_data['year']
        if year > max_year:
            continue
            
        # ERF failures
        for joint_num in year_data['erf_failed_joints']:
            joint_details = year_data['joint_failure_details'].get(joint_num, [])
            erf_failures = [d for d in joint_details if d['failure_type'] == 'erf']
            
            if erf_failures:
                worst_erf = min(erf_failures, key=lambda x: x['erf'])
                details_data.append({
                    'Year': year,
                    'Joint Number': joint_num,
                    'Failure Type': 'ERF < 1.0',
                    'ERF Value': f"{worst_erf['erf']:.3f}",
                    'Failure Pressure (MPa)': f"{worst_erf['failure_pressure_mpa']:.1f}",
                    'Location (m)': f"{worst_erf['location_m']:.2f}",
                    'Details': f"Operating pressure exceeds safe capacity"
                })
        
        # Depth failures
        for joint_num in year_data['depth_failed_joints']:
            joint_details = year_data['joint_failure_details'].get(joint_num, [])
            depth_failures = [d for d in joint_details if d['failure_type'] == 'depth']
            
            if depth_failures:
                worst_depth = max(depth_failures, key=lambda x: x['depth_pct'])
                details_data.append({
                    'Year': year,
                    'Joint Number': joint_num,
                    'Failure Type': 'Depth > 80%',
                    'ERF Value': 'N/A',
                    'Failure Pressure (MPa)': 'N/A',
                    'Location (m)': f"{worst_depth['location_m']:.2f}",
                    'Details': f"Depth: {worst_depth['depth_pct']:.1f}%"
                })
    
    if details_data:
        df = pd.DataFrame(details_data)
        return df.sort_values(['Year', 'Joint Number']).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['Year', 'Joint Number', 'Failure Type', 'ERF Value', 
                                   'Failure Pressure (MPa)', 'Location (m)', 'Details'])


def create_failure_comparison_chart(results_dict: dict) -> go.Figure:
    """
    Create a comparison chart showing different assessment methods side by side.
    
    Parameters:
    - results_dict: Dictionary with results from multiple assessment methods
                   Format: {'b31g': results, 'modified_b31g': results, ...}
    
    Returns:
    - Plotly figure comparing methods
    """
    
    methods = list(results_dict.keys())
    years = results_dict[methods[0]]['years']  # Assume all have same years
    
    fig = go.Figure()
    
    colors = {
        'b31g': '#E74C3C',
        'modified_b31g': '#3498DB', 
        'simplified_eff_area': '#27AE60'
    }
    
    method_names = {
        'b31g': 'B31G Original',
        'modified_b31g': 'Modified B31G',
        'simplified_eff_area': 'RSTRENG'
    }
    
    for method in methods:
        results = results_dict[method]
        total_failures = [erf + depth for erf, depth in 
                         zip(results['erf_failures_by_year'], results['depth_failures_by_year'])]
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=total_failures,
                mode='lines+markers',
                name=method_names.get(method, method),
                line=dict(width=3, color=colors.get(method, '#95A5A6')),
                marker=dict(size=8),
                hovertemplate=(
                    f"<b>{method_names.get(method, method)}</b><br>"
                    "Year: %{x}<br>"
                    "Total Failures: %{y} joints<br>"
                    "<extra></extra>"
                )
            )
        )
    
    fig.update_layout(
        title="Failure Prediction Comparison by Assessment Method",
        xaxis_title="Years from Now",
        yaxis_title="Total Failed Joints (Annual)",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom", 
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(
        gridcolor='rgba(128,128,128,0.2)',
        showgrid=True,
        tick0=1,
        dtick=1
    )
    
    fig.update_yaxes(
        gridcolor='rgba(128,128,128,0.2)',
        showgrid=True
    )
    
    return fig