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

"""
Simple visualization functions for failure prediction results.
"""
def create_failure_timeline_histogram(simulation_results: dict) -> go.Figure:
    """
    FIXED: Create histogram of defect failures by year - handles pandas arrays properly.
    """
    
    try:
        timeline = simulation_results.get('failure_timeline', {})
        
        if not timeline:
            # Empty timeline case
            fig = go.Figure()
            fig.add_annotation(
                text="No failures predicted in simulation timeframe",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color='#2C3E50')
            )
            fig.update_layout(title="Defect Failure Timeline - No Failures Predicted")
            return fig
        
        # SOLUTION 1: Convert all timeline values to safe integers
        safe_timeline = {}
        for year, count in timeline.items():
            # Handle pandas arrays/Series safely
            try:
                if hasattr(count, '__iter__') and not isinstance(count, (str, bytes)):
                    # If it's an array-like object, take the first value or sum
                    if hasattr(count, '__len__') and len(count) > 0:
                        safe_count = int(count[0]) if len(count) == 1 else int(sum(count))
                    else:
                        safe_count = 0
                else:
                    safe_count = int(count)
                
                # Convert year safely too
                if hasattr(year, '__iter__') and not isinstance(year, (str, bytes)):
                    safe_year = int(year[0]) if hasattr(year, '__len__') and len(year) > 0 else int(year)
                else:
                    safe_year = int(year)
                    
                safe_timeline[safe_year] = safe_count
                
            except (ValueError, TypeError, IndexError):
                # Skip problematic entries
                continue
        
        if not safe_timeline:
            # No valid data case
            fig = go.Figure()
            fig.add_annotation(
                text="No valid timeline data available",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color='#E74C3C')
            )
            fig.update_layout(title="Defect Failure Timeline - Data Error")
            return fig
        
        # SOLUTION 2: Safe separation of current vs future failures
        current_failures = safe_timeline.get(0, 0)  # Now guaranteed to be int
        future_timeline = {k: v for k, v in safe_timeline.items() if k > 0}  # Safe comparison
        
        fig = go.Figure()
        
        # SOLUTION 3: Safe boolean check for current failures
        if current_failures > 0:  # Now safe - current_failures is definitely an int
            fig.add_trace(go.Bar(
                x=[0],
                y=[current_failures],
                name='Current Failures',
                marker_color='rgba(255, 0, 0, 0.9)',
                text=[str(current_failures)],
                textposition='outside',
                hovertemplate='<b>Current State</b><br>Immediate Defect Failures: %{y}<extra></extra>'  # Updated text
            ))
        
        # SOLUTION 4: Safe processing of future timeline
        if future_timeline:  # Safe check - future_timeline is a regular dict
            years = sorted(future_timeline.keys())  # Sort for better display
            failures = [future_timeline[year] for year in years]
            
            # Additional safety check
            valid_data = [(y, f) for y, f in zip(years, failures) if f > 0]
            
            if valid_data:
                valid_years, valid_failures = zip(*valid_data)
                
                fig.add_trace(go.Bar(
                    x=valid_years,
                    y=valid_failures,
                    name='Predicted Defect Failures',  # Updated text
                    marker_color='rgba(255, 99, 71, 0.8)',
                    text=[str(f) for f in valid_failures],
                    textposition='outside',
                    hovertemplate='<b>Year %{x}</b><br>Predicted Defect Failures: %{y}<extra></extra>'  # Updated text
                ))
        
        # SOLUTION 5: Enhanced layout with better error handling
        total_failures = sum(safe_timeline.values())
        simulation_years = max(safe_timeline.keys()) if safe_timeline else 0
        
        fig.update_layout(
            title=f'Defect Failure Timeline: {total_failures} failures over {simulation_years} years',  # Updated title
            xaxis_title='Years from Now (0 = Current State)',
            yaxis_title='Defect Failures',  # Updated label
            height=500,
            showlegend=True,
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )
        )
        
        return fig
        
    except Exception as e:
        # SOLUTION 6: Comprehensive error handling with diagnostic info
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating timeline chart: {str(e)[:100]}...",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color='#E74C3C')
        )
        fig.update_layout(title="Defect Failure Timeline - Error")
        return fig