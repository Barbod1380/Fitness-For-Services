from plotly.subplots import make_subplots
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
    
    fig.add_trace(go.Scattergl(
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
    
    fig.add_trace(go.Scattergl(
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


def create_failure_timeline_histogram(simulation_results: dict) -> go.Figure:
    # —————————————————————————————
    # 1. Safely sanitize the timeline data:
    timeline = simulation_results.get('failure_timeline', {})
    safe_timeline = {}
    for year, count in timeline.items():
        try:
            # Handle array-like values
            if hasattr(count, '__iter__') and not isinstance(count, (str, bytes)):
                count = int(sum(count)) if len(count) > 1 else int(count[0])
            else:
                count = int(count)

            # Handle array-like years
            if hasattr(year, '__iter__') and not isinstance(year, (str, bytes)):
                year = int(year[0])
            else:
                year = int(year)

            safe_timeline[year] = safe_timeline.get(year, 0) + count
        except Exception:
            continue

    if not safe_timeline:
        fig = go.Figure()
        fig.add_annotation(text="No valid timeline data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False,
                           font=dict(size=16, color='red'))
        fig.update_layout(title="Defect Failure Timeline – Data Error")
        return fig

    # Sort and build annual + cumulative arrays
    years = sorted(safe_timeline.keys())
    counts = [safe_timeline[y] for y in years]
    cum_counts = np.cumsum(counts)

    # —————————————————————————————
    # 2. Create the dual-axis figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Assign bar colors and label
    colors = ['red' if y == 0 else 'lightsalmon' for y in years]

    fig.add_trace(
        go.Bar(
            x=years,
            y=counts,
            name='Annual Failures',
            marker_color=colors,
            text=[str(c) for c in counts],
            textposition='outside',
            hovertemplate='<b>Year %{x}</b><br>Annual: %{y}<extra></extra>'
        ),
        secondary_y=False
    )

    # Add cumulative line
    fig.add_trace(
        go.Scatter(
            x=years,
            y=cum_counts,
            mode='lines+markers+text',
            name='Cumulative Failures',
            line=dict(color='blue', dash='dash', width=2),
            text=[str(int(v)) for v in cum_counts],
            textposition='top center',
            hovertemplate='<b>Year %{x}</b><br>Cum Total: %{y}<extra></extra>'
        ),
        secondary_y=True
    )

    # —————————————————————————————
    # 3. Update overall layout and interactions:
    fig.update_layout(
        title=f"Defect Failures & Cumulative Trend: {int(cum_counts[-1])} total over {years[-1]} years",
        barmode='group',
        height=600,
        hovermode='x unified',  # unified hover with both traces :contentReference[oaicite:1]{index=1}
        template='simple_white',
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center')
    )

    # Grids and spikes
    fig.update_xaxes(title_text='Years from Now (0 = Current)',
                     showgrid=True,
                     gridcolor='rgba(128,128,128,0.2)',
                     tickmode='linear',
                     showspikes=True)
    fig.update_yaxes(title_text='Annual Failures',
                     secondary_y=False,
                     showgrid=True,
                     gridcolor='rgba(128,128,128,0.2)',
                     showspikes=True)
    fig.update_yaxes(title_text='Cumulative Failures',
                     secondary_y=True,
                     showgrid=False,
                     showspikes=True)

    # —————————————————————————————
    # 4. Final annotation of the cumulative endpoint:
    fig.add_annotation(
        x=years[-1],
        y=cum_counts[-1],
        text=f"Total: {int(cum_counts[-1])}",
        showarrow=True,
        arrowhead=2,
        ax=0, ay=-40,
        font=dict(color='blue')
    )

    return fig