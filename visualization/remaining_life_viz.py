"""
Visualization functions for remaining life analysis of pipeline defects.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_remaining_life_pipeline_visualization(remaining_life_results):
    """
    Create an interactive pipeline visualization colored by remaining life years.
    
    Parameters:
    - remaining_life_results: Dictionary from calculate_remaining_life_analysis
    
    Returns:
    - Plotly figure object
    """
    if not remaining_life_results.get('analysis_possible', False):
        fig = go.Figure()
        fig.add_annotation(
            text="Remaining life analysis not possible with current data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Combine all defect analyses
    all_analyses = (remaining_life_results['matched_defects_analysis'] + 
                   remaining_life_results['new_defects_analysis'])
    
    if not all_analyses:
        fig = go.Figure()
        fig.add_annotation(
            text="No defects available for remaining life analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(all_analyses)
    
    # Handle infinite values for visualization
    df['remaining_life_display'] = df['remaining_life_years'].replace([np.inf], 100)  # Cap at 100 years for visualization
    
    # Create color mapping based on risk status
    color_map = {
        'CRITICAL': 'red',
        'HIGH_RISK': 'orange', 
        'MEDIUM_RISK': 'yellow',
        'LOW_RISK': 'green',
        'STABLE': 'blue',
        'ERROR': 'gray'
    }
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each risk category
    for status, color in color_map.items():
        status_data = df[df['status'] == status]
        if not status_data.empty:
            # Handle infinite values in hover text
            hover_text = []
            for _, row in status_data.iterrows():
                if np.isinf(row['remaining_life_years']):
                    life_text = "Stable (>100 years)"
                else:
                    life_text = f"{row['remaining_life_years']:.1f} years"
                
                growth_source = "📊 Measured" if row['growth_rate_source'] == 'MEASURED' else "📈 Estimated"
                
                hover_text.append(
                    f"<b>Location:</b> {row['log_dist']:.2f}m<br>"
                    f"<b>Remaining Life:</b> {life_text}<br>"
                    f"<b>Current Depth:</b> {row['current_depth_pct']:.1f}%<br>"
                    f"<b>Growth Rate:</b> {row['growth_rate_pct_per_year']:.2f}%/year<br>"
                    f"<b>Defect Type:</b> {row['defect_type']}<br>"
                    f"<b>Joint:</b> {row['joint_number']}<br>"
                    f"<b>Growth Data:</b> {growth_source}<br>"
                    f"<b>Status:</b> {status.replace('_', ' ').title()}"
                )
            
            fig.add_trace(go.Scatter(
                x=status_data['log_dist'],
                y=[1] * len(status_data),  # All at same y-level for pipeline view
                mode='markers',
                marker=dict(
                    size=12,
                    color=color,
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                name=f'{status.replace("_", " ").title()} ({len(status_data)})',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title="Pipeline Remaining Life Analysis<br><sub>Hover over points to see detailed information</sub>",
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title="Pipeline Representation",
        height=500,
        hovermode='closest',
        yaxis=dict(
            range=[0.5, 1.5],
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)"
        ),
        plot_bgcolor="white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        )
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
    if not remaining_life_results.get('analysis_possible', False):
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for histogram",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Combine all analyses
    all_analyses = (remaining_life_results['matched_defects_analysis'] + 
                   remaining_life_results['new_defects_analysis'])
    
    if not all_analyses:
        return go.Figure()
    
    df = pd.DataFrame(all_analyses)
    
    # Filter out infinite values and errors for histogram
    finite_data = df[np.isfinite(df['remaining_life_years']) & (df['status'] != 'ERROR')]
    
    if finite_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No finite remaining life values to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create histogram with different colors for measured vs estimated
    fig = go.Figure()
    
    # Measured growth rates
    measured_data = finite_data[finite_data['growth_rate_source'] == 'MEASURED']
    if not measured_data.empty:
        fig.add_trace(go.Histogram(
            x=measured_data['remaining_life_years'],
            nbinsx=20,
            name="Measured Growth Rates",
            marker=dict(
                color='rgba(0, 100, 200, 0.7)',
                line=dict(color='rgba(0, 100, 200, 1)', width=1)
            ),
            opacity=0.7
        ))
    
    # Estimated growth rates  
    estimated_data = finite_data[finite_data['growth_rate_source'] == 'ESTIMATED']
    if not estimated_data.empty:
        fig.add_trace(go.Histogram(
            x=estimated_data['remaining_life_years'],
            nbinsx=20,
            name="Estimated Growth Rates",
            marker=dict(
                color='rgba(200, 100, 0, 0.7)',
                line=dict(color='rgba(200, 100, 0, 1)', width=1)
            ),
            opacity=0.7
        ))
    
    # Add vertical lines for risk thresholds
    fig.add_shape(
        type="line",
        x0=2, x1=2,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=10, x1=10,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="orange", width=2, dash="dash"),
    )
    
    # Add annotations for thresholds
    fig.add_annotation(
        x=2, y=1, yref="paper",
        text="High Risk<br>Threshold",
        showarrow=True, arrowhead=1,
        ax=-30, ay=-30, font=dict(color="red", size=10)
    )
    
    fig.add_annotation(
        x=10, y=1, yref="paper", 
        text="Medium Risk<br>Threshold",
        showarrow=True, arrowhead=1,
        ax=30, ay=-30, font=dict(color="orange", size=10)
    )
    
    # Layout
    fig.update_layout(
        title="Distribution of Remaining Life Until Critical Depth (80%)",
        xaxis_title="Remaining Life (Years)",
        yaxis_title="Number of Defects",
        barmode='overlay',
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right", 
            x=0.99
        )
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
    if not remaining_life_results.get('analysis_possible', False):
        return pd.DataFrame([{'Metric': 'Analysis Status', 'Value': 'Not Possible - Missing Data'}])
    
    summary_stats = remaining_life_results.get('summary_statistics', {})
    
    if not summary_stats:
        return pd.DataFrame([{'Metric': 'Analysis Status', 'Value': 'No Data Available'}])
    
    # Create summary rows
    rows = []
    
    rows.append({
        'Metric': 'Total Defects Analyzed',
        'Value': f"{summary_stats.get('total_defects_analyzed', 0)}"
    })
    
    rows.append({
        'Metric': 'Defects with Measured Growth',
        'Value': f"{summary_stats.get('defects_with_measured_growth', 0)}"
    })
    
    rows.append({
        'Metric': 'Defects with Estimated Growth', 
        'Value': f"{summary_stats.get('defects_with_estimated_growth', 0)}"
    })
    
    # Add remaining life statistics if available
    avg_life = summary_stats.get('average_remaining_life_years', float('nan'))
    if not np.isnan(avg_life):
        rows.append({
            'Metric': 'Average Remaining Life',
            'Value': f"{avg_life:.1f} years"
        })
    
    median_life = summary_stats.get('median_remaining_life_years', float('nan'))
    if not np.isnan(median_life):
        rows.append({
            'Metric': 'Median Remaining Life',
            'Value': f"{median_life:.1f} years"
        })
    
    min_life = summary_stats.get('min_remaining_life_years', float('nan'))
    if not np.isnan(min_life):
        rows.append({
            'Metric': 'Shortest Remaining Life',
            'Value': f"{min_life:.1f} years"
        })
    
    # Add risk status distribution
    status_dist = summary_stats.get('status_distribution', {})
    for status, count in status_dist.items():
        rows.append({
            'Metric': f'{status.replace("_", " ").title()} Defects',
            'Value': f"{count}"
        })
    
    return pd.DataFrame(rows)

def create_remaining_life_risk_matrix(remaining_life_results):
    """
    Create a risk matrix visualization showing current condition vs remaining life.
    
    Parameters:
    - remaining_life_results: Dictionary from calculate_remaining_life_analysis
    
    Returns:
    - Plotly figure object
    """
    if not remaining_life_results.get('analysis_possible', False):
        fig = go.Figure()
        fig.add_annotation(
            text="Risk matrix not available with current data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Combine all analyses
    all_analyses = (remaining_life_results['matched_defects_analysis'] + 
                   remaining_life_results['new_defects_analysis'])
    
    if not all_analyses:
        return go.Figure()
    
    df = pd.DataFrame(all_analyses)
    
    # Filter out errors and infinite values
    valid_data = df[(df['status'] != 'ERROR') & np.isfinite(df['remaining_life_years'])]
    
    if valid_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data for risk matrix",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Cap remaining life at 50 years for better visualization
    valid_data['remaining_life_capped'] = valid_data['remaining_life_years'].clip(upper=50)
    
    # Create color mapping
    color_map = {
        'CRITICAL': 'red',
        'HIGH_RISK': 'orange',
        'MEDIUM_RISK': 'yellow', 
        'LOW_RISK': 'green',
        'STABLE': 'blue'
    }
    
    fig = go.Figure()
    
    # Add scatter points colored by status
    for status, color in color_map.items():
        status_data = valid_data[valid_data['status'] == status]
        if not status_data.empty:
            hover_text = [
                f"<b>Location:</b> {row['log_dist']:.2f}m<br>"
                f"<b>Current Depth:</b> {row['current_depth_pct']:.1f}%<br>"
                f"<b>Remaining Life:</b> {row['remaining_life_years']:.1f} years<br>"
                f"<b>Defect Type:</b> {row['defect_type']}<br>"
                f"<b>Growth Source:</b> {row['growth_rate_source']}"
                for _, row in status_data.iterrows()
            ]
            
            fig.add_trace(go.Scatter(
                x=status_data['current_depth_pct'],
                y=status_data['remaining_life_capped'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                name=f'{status.replace("_", " ").title()}',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
    
    # Add risk zone backgrounds
    fig.add_shape(
        type="rect",
        x0=0, x1=100, y0=0, y1=2,
        fillcolor="rgba(255, 0, 0, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="rect", 
        x0=0, x1=100, y0=2, y1=10,
        fillcolor="rgba(255, 165, 0, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    # Layout
    fig.update_layout(
        title="Risk Matrix: Current Condition vs Remaining Life",
        xaxis_title="Current Depth (% of Wall Thickness)",
        yaxis_title="Remaining Life (Years, capped at 50)",
        height=500,
        hovermode='closest',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 50]),
        annotations=[
            dict(
                x=50, y=1,
                text="HIGH RISK ZONE",
                showarrow=False,
                font=dict(color="red", size=12, family="Arial Black")
            ),
            dict(
                x=50, y=6,
                text="MEDIUM RISK ZONE", 
                showarrow=False,
                font=dict(color="orange", size=12, family="Arial Black")
            )
        ]
    )
    
    return fig