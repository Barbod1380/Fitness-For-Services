import numpy as np
import plotly.graph_objects as go
from typing import Dict

def create_survival_curve(simulation_results: Dict) -> go.Figure:
    """Create survival curve showing joint survival over time."""
    
    annual_results = simulation_results['annual_results']
    years = [r['year'] for r in annual_results]
    surviving_joints = [r['surviving_joints'] for r in annual_results]
    total_joints = annual_results[0]['total_joints']
    
    # Calculate survival percentage
    survival_pct = [(s / total_joints * 100) for s in surviving_joints]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=survival_pct,
        mode='lines+markers',
        name='Joint Survival Rate',
        line=dict(color='rgba(50, 200, 50, 0.8)', width=4),
        marker=dict(size=10),
        fill='tonexty',
        fillcolor='rgba(50, 200, 50, 0.1)',
        hovertemplate='<b>Year %{x}</b><br>Survival Rate: %{y:.1f}%<extra></extra>'
    ))
    
    # Add 50% survival line
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="50% Survival Threshold")
    
    fig.update_layout(
        title='Pipeline Joint Survival Curve',
        xaxis_title='Simulation Year',
        yaxis_title='Survival Rate (%)',
        yaxis=dict(range=[0, 105]),
        height=400,
        showlegend=True
    )
    
    return fig

def create_erf_evolution_plot(simulation_results: Dict) -> go.Figure:
    """Create plot showing ERF evolution over time."""
    
    annual_results = simulation_results['annual_results']
    years = [r['year'] for r in annual_results]
    max_erf = [r['max_erf'] for r in annual_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=max_erf,
        mode='lines+markers',
        name='Maximum ERF',
        line=dict(color='rgba(255, 150, 0, 0.8)', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Year %{x}</b><br>Max ERF: %{y:.3f}<extra></extra>'
    ))
    
    # Add ERF threshold line
    erf_threshold = simulation_results['simulation_params'].erf_threshold
    fig.add_hline(y=erf_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"ERF Threshold ({erf_threshold})")
    
    fig.update_layout(
        title='Maximum ERF Evolution Over Time',
        xaxis_title='Simulation Year',
        yaxis_title='ERF Value',
        height=400,
        showlegend=True
    )
    
    return fig



"""
Simple visualization functions for failure prediction results.
"""

def create_failure_timeline_histogram(simulation_results: dict) -> go.Figure:
    """Create histogram of joint failures by year."""
    
    timeline = simulation_results['failure_timeline']
    
    # Separate current failures from future predictions
    current_failures = timeline.get(0, 0)
    future_timeline = {k: v for k, v in timeline.items() if k > 0}
    
    fig = go.Figure()
    
    # Add current failures as special bar
    if current_failures > 0:
        fig.add_trace(go.Bar(
            x=[0],
            y=[current_failures],
            name='Current Failures',
            marker_color='rgba(255, 0, 0, 0.9)',  # Red for immediate
            text=[current_failures],
            textposition='outside',
            hovertemplate='<b>Current State</b><br>Immediate Failures: %{y}<extra></extra>'
        ))
    
    # Add future predictions
    if future_timeline:
        years = list(future_timeline.keys())
        failures = list(future_timeline.values())
        
        fig.add_trace(go.Bar(
            x=years,
            y=failures,
            name='Predicted Failures',
            marker_color='rgba(255, 99, 71, 0.8)',
            text=failures,
            textposition='outside',
            hovertemplate='<b>Year %{x}</b><br>Predicted Failures: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Pipeline Failure Timeline: Current + Predictions',
        xaxis_title='Years from Now (0 = Current State)',
        yaxis_title='Joint Failures',
        height=500
    )
    
    return fig