import plotly.graph_objects as go

def create_pressure_assessment_visualization(enhanced_df, method='b31g'):
    """
    Create visualization showing pressure-based assessment results.
    """

    # Filter valid data
    valid_data = enhanced_df[enhanced_df[f'{method}_safe'] == True].copy()
    
    if valid_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data for pressure assessment visualization",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Check if we should enable hover (disable for > 2000 points)
    enable_hover = len(valid_data) <= 2000
    
    # Color mapping for operational status
    color_map = {
        'ACCEPTABLE': 'green',
        'PRESSURE_DERATION_REQUIRED': 'orange', 
        'IMMEDIATE_REPAIR_REQUIRED': 'red',
        'UNKNOWN': 'gray'
    }
    
    # Create scatter plot
    fig = go.Figure()
    
    for status, color in color_map.items():
        status_data = valid_data[valid_data[f'{method}_operational_status'] == status]
        if status_data.empty:
            continue
        
        # Prepare scatter trace parameters
        trace_params = {
            'x': status_data['log dist. [m]'],
            'y': status_data[f'{method}_safe_pressure_mpa'],
            'mode': 'markers',
            'marker': dict(size=10, color=color, opacity=0.7),
            'name': f"{status.replace('_', ' ').title()} ({len(status_data)})"
        }
        
        # Add hover functionality only if points <= 2000
        if enable_hover:
            hover_text = []
            for _, row in status_data.iterrows():
                hover_text.append(
                    f"<b>Location:</b> {row['log dist. [m]']:.2f}m<br>"
                    f"<b>Safe Pressure:</b> {row[f'{method}_safe_pressure_mpa']:.1f} MPa<br>"
                    f"<b>Max Safe Operating:</b> {row[f'{method}_max_safe_operating_pressure_mpa']:.1f} MPa<br>"
                    f"<b>Pressure Margin:</b> {row[f'{method}_pressure_margin_pct']:.1f}%<br>"
                    f"<b>Status:</b> {status.replace('_', ' ').title()}<br>"
                    f"<b>Action:</b> {row[f'{method}_recommended_action']}"
                )
            trace_params['text'] = hover_text
            trace_params['hovertemplate'] = '%{text}<extra></extra>'
        else:
            trace_params['hoverinfo'] = 'skip'
        
        fig.add_trace(go.Scattergl(**trace_params))
    
    # Add horizontal lines for pressure references
    analysis_pressure = enhanced_df['analysis_pressure_mpa'].iloc[0]
    max_allowable = enhanced_df['max_allowable_pressure_mpa'].iloc[0]
    
    fig.add_hline(
        y=analysis_pressure, 
        line_dash="dash", 
        line_color="blue",
        annotation_text=f"Analysis Pressure: {analysis_pressure:.1f} MPa"
    )
    
    fig.add_hline(
        y=max_allowable,
        line_dash="dot", 
        line_color="purple",
        annotation_text=f"Max Allowable: {max_allowable:.1f} MPa"
    )
    
    # Update layout with conditional hovermode
    layout_params = {
        'title': f"Pressure-Based Assessment Results - {method.replace('_', ' ').title()}",
        'xaxis_title': "Distance Along Pipeline (m)",
        'yaxis_title': "Safe Operating Pressure (MPa)",
        'height': 500,
        'legend': dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    }
    
    if enable_hover:
        layout_params['hovermode'] = 'closest'
    else:
        layout_params['hovermode'] = False
    
    fig.update_layout(**layout_params)
    
    return fig