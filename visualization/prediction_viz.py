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

# Replace the entire function in visualization/prediction_viz.py if needed

def create_failure_timeline_histogram(simulation_results: dict) -> go.Figure:
    """Simple, bulletproof timeline showing defect and joint failures."""
    
    try:
        # Get defect failures
        defect_timeline = simulation_results.get('failure_timeline', {})
        joint_timeline = simulation_results.get('joint_failure_timeline', {})
        
        # Simple data extraction
        years = []
        defect_failures = []
        joint_failures = []
        
        # Get all years from both timelines
        all_years = set()
        if defect_timeline:
            all_years.update(defect_timeline.keys())
        if joint_timeline:
            all_years.update(joint_timeline.keys())
        
        if not all_years:
            fig = go.Figure()
            fig.add_annotation(text="No failure data available",
                             xref="paper", yref="paper", x=0.5, y=0.5,
                             showarrow=False, font=dict(size=16, color='orange'))
            fig.update_layout(title="Failure Timeline – No Data")
            return fig
        
        # Convert to sorted list
        years = sorted([int(y) for y in all_years])
        
        # Extract counts for each year
        for year in years:
            # Defect failures
            defect_count = defect_timeline.get(year, 0)
            if isinstance(defect_count, (list, tuple)):
                defect_count = sum(defect_count) if defect_count else 0
            defect_failures.append(int(defect_count))
            
            # Joint failures  
            joint_count = joint_timeline.get(year, 0)
            if isinstance(joint_count, (list, tuple)):
                joint_count = sum(joint_count) if joint_count else 0
            joint_failures.append(int(joint_count))
        
        # Calculate cumulative (simple Python sum)
        defect_cumulative = []
        joint_cumulative = []
        defect_total = 0
        joint_total = 0
        
        for i in range(len(years)):
            defect_total += defect_failures[i]
            joint_total += joint_failures[i]
            defect_cumulative.append(defect_total)
            joint_cumulative.append(joint_total)
        
        # Create figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Annual bars
        fig.add_trace(
            go.Bar(
                x=years,
                y=defect_failures,
                name='Annual Defect Failures',
                marker_color='lightsalmon',
                offsetgroup=1,
                hovertemplate='<b>Year %{x}</b><br>Defects: %{y}<extra></extra>'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=years,
                y=joint_failures,
                name='Annual Joint Failures',
                marker_color='red',
                offsetgroup=2,
                hovertemplate='<b>Year %{x}</b><br>Joints: %{y}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Cumulative lines
        fig.add_trace(
            go.Scatter(
                x=years,
                y=defect_cumulative,
                mode='lines+markers',
                name='Total Defects Failed',
                line=dict(color='orange', dash='dash', width=2),
                hovertemplate='<b>Year %{x}</b><br>Total Defects: %{y}<extra></extra>'
            ),
            secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=joint_cumulative,
                mode='lines+markers',
                name='Total Joints Failed',
                line=dict(color='darkred', dash='dot', width=3),
                hovertemplate='<b>Year %{x}</b><br>Total Joints: %{y}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Layout
        max_year = max(years) if years else 0
        final_defects = defect_cumulative[-1] if defect_cumulative else 0
        final_joints = joint_cumulative[-1] if joint_cumulative else 0
        
        fig.update_layout(
            title=f"Failure Timeline: {final_defects} defects, {final_joints} joints over {max_year} years",
            barmode='group',
            height=600,
            hovermode='x unified',
            template='simple_white',
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center')
        )
        
        # Axes
        fig.update_xaxes(title_text='Years from Now', showgrid=True)
        fig.update_yaxes(title_text='Annual Failures', secondary_y=False, showgrid=True)
        fig.update_yaxes(title_text='Cumulative Failures', secondary_y=True, showgrid=False)
        
        return fig
        
    except Exception as e:
        # Fallback error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart Error: {str(e)[:100]}...",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color='red')
        )
        fig.update_layout(title="Timeline Chart Error")
        return fig