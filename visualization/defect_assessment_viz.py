"""
Defect assessment visualization showing defect population against B31G criteria.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def create_defect_assessment_scatter_plot(enhanced_df, pipe_diameter_mm, smys_mpa, safety_factor=1.39):
    """
    Create defect assessment scatter plot with performance optimization for large datasets.
    Disables hover and uses WebGL rendering for datasets > 2000 points.
    """
    
    # Validate inputs
    if enhanced_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No defect data available for assessment plot",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#2C3E50')
        )
        return fig
    
    # Required columns check
    required_cols = ['length [mm]', 'depth [%]', 'wall_thickness_used_mm']
    missing_cols = [col for col in required_cols if col not in enhanced_df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns: {', '.join(missing_cols)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#E74C3C')
        )
        return fig
    
    # Prepare data
    plot_df = enhanced_df.copy()
    
    # Convert depth from percentage to mm using wall thickness
    plot_df['depth_mm'] = plot_df['depth [%]'] * plot_df['wall_thickness_used_mm'] / 100
    
    # Filter out invalid data
    valid_mask = (
        (plot_df['length [mm]'] > 0) & 
        (plot_df['depth_mm'] > 0) &
        (plot_df['depth_mm'] <= plot_df['wall_thickness_used_mm'])
    )
    plot_df = plot_df[valid_mask].copy()
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid defect data for plotting",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#E74C3C')
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping for surface locations
    surface_colors = {
        'NON-INT': '#3498DB',  # Blue for external
        'INT': '#E74C3C',      # Red for internal  
        'Combined': '#27AE60', # Green for combined/clustered
        'Unknown': '#95A5A6'   # Gray for unknown
    }
    
    surface_labels = {
        'NON-INT': 'Features - External',
        'INT': 'Features - Internal',
        'Combined': 'Features - Combined', 
        'Unknown': 'Features - Unknown'
    }
    
    # Determine surface location categories
    if 'surface location' in plot_df.columns:
        plot_df['surface_category'] = plot_df['surface location'].fillna('Unknown')
        if 'is_combined' in plot_df.columns:
            plot_df.loc[plot_df['is_combined'] == True, 'surface_category'] = 'Combined'
    else:
        plot_df['surface_category'] = 'Unknown'
    
    # Choose scatter type based on performance mode
    scatter_type = go.Scattergl
    
    # Add scatter plots for each surface location category
    for category in plot_df['surface_category'].unique():
        if pd.isna(category):
            category = 'Unknown'
            
        category_data = plot_df[plot_df['surface_category'] == category]
        
        if len(category_data) > 0:
            # PERFORMANCE MODE: No hover, simplified markers
            trace_config = {
                'x': category_data['length [mm]'],
                'y': category_data['depth_mm'],
                'mode': 'markers',
                'marker': dict(
                    color=surface_colors.get(category, '#95A5A6'),
                    size=4,  # Smaller markers for performance
                    opacity=0.6,
                    line=dict(color='white', width=0.5)
                ),
                'name': surface_labels.get(category, f'Features - {category}'),
                'hoverinfo': 'skip',  # Disable hover completely
                'showlegend': True
            }
            fig.add_trace(scatter_type(**trace_config))
    
    # Generate B31G curves (unchanged)
    length_range = np.logspace(0, 4, 200)
    avg_wall_thickness = plot_df['wall_thickness_used_mm'].mean()
    
    b31g_depths = _calculate_b31g_allowable_curve(
        length_range, pipe_diameter_mm, avg_wall_thickness, smys_mpa, 'original', safety_factor
    )
    
    modified_b31g_depths = _calculate_b31g_allowable_curve(
        length_range, pipe_diameter_mm, avg_wall_thickness, smys_mpa, 'modified', safety_factor
    )
    
    # Add B31G curves (these remain interactive since they're just 2 lines)
    fig.add_trace(go.Scattergl(
        x=length_range,
        y=b31g_depths,
        mode='lines',
        line=dict(color='#F1C40F', width=3, dash='dash'),
        name='ASME B31G',
        hovertemplate='<b>ASME B31G Limit</b><br>Length: %{x:.1f}mm<br>Max Depth: %{y:.2f}mm<extra></extra>',
        showlegend=True
    ))
    
    fig.add_trace(go.Scattergl(
        x=length_range,
        y=modified_b31g_depths,
        mode='lines',
        line=dict(color='#E67E22', width=3),
        name='Modified B31G',
        hovertemplate='<b>Modified B31G Limit</b><br>Length: %{x:.1f}mm<br>Max Depth: %{y:.2f}mm<extra></extra>',
        showlegend=True
    ))
    
    # Add wall thickness indicators (from previous fix)
    min_wall_thickness = plot_df['wall_thickness_used_mm'].min()
    max_wall_thickness = plot_df['wall_thickness_used_mm'].max()

    if min_wall_thickness == max_wall_thickness:
        fig.add_hline(
            y=max_wall_thickness,
            line=dict(color='#2C3E50', width=3),
            annotation_text=f"Wall Thickness: {max_wall_thickness:.1f}mm",
            annotation_position="top right"
        )
    else:
        fig.add_hline(
            y=max_wall_thickness,
            line=dict(color='#2C3E50', width=2, dash='solid'),
            annotation_text=f"Max WT: {max_wall_thickness:.1f}mm",
            annotation_position="top right"
        )
        
        fig.add_hline(
            y=min_wall_thickness,
            line=dict(color='#2C3E50', width=2, dash='dash'),
            annotation_text=f"Min WT: {min_wall_thickness:.1f}mm",
            annotation_position="bottom right"
        )
    
    # Update layout with performance considerations
    fig.update_layout(
        title=dict(
            text=f"Defect Assessment: Population vs B31G Criteria",
            font=dict(size=16, family="Inter, Arial, sans-serif"),
            x=0.02
        ),
        xaxis=dict(
            title="Axial Length (mm)",
            type="log",
            range=[0, 4],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linecolor='rgba(128,128,128,0.5)',
            ticks="outside",
            tickformat=",",
            title_font=dict(size=12, color='#2C3E50'),
            tickfont=dict(size=10, color='#2C3E50')
        ),
        yaxis=dict(
            title="Depth (mm)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linecolor='rgba(128,128,128,0.5)',
            ticks="outside",
            title_font=dict(size=12, color='#2C3E50'),
            tickfont=dict(size=10, color='#2C3E50'),
            range=[0, max_wall_thickness * 1.1]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=11)
        ),
    )
    
    return fig


def _calculate_b31g_allowable_curve(length_range_mm, pipe_diameter_mm, wall_thickness_mm, smys_mpa, method, safety_factor):
    """
    Calculate the maximum allowable defect depth for a range of defect lengths
    based on B31G or Modified B31G methodology using flow stress and pressure equations.

    Parameters:
    - length_range_mm: Iterable of defect lengths in mm
    - pipe_diameter_mm: Pipe outside diameter in mm
    - wall_thickness_mm: Wall thickness in mm
    - smys_mpa: Specified Minimum Yield Strength (SMYS) in MPa
    - method: 'original' or 'modified' B31G
    - safety_factor: Safety factor (default 1.39 for gas pipelines)

    Returns:
    - Numpy array of maximum allowable defect depths (in mm) for each length
    """
    
    allowable_depths = []

    # Calculate flow stress
    if method == 'original':
        flow_stress_mpa = 1.1 * smys_mpa
    elif method == 'modified':
        flow_stress_mpa = smys_mpa + 68.95
    else:
        raise ValueError("Method must be either 'original' or 'modified'")
    
    # Calculate design pressure (Po) using thin-wall hoop stress formula
    Po_mpa = (2 * wall_thickness_mm * flow_stress_mpa) / pipe_diameter_mm

    for length_mm in length_range_mm:
        try:
            # Dimensionless length parameter z = L^2 / (D * t)
            z = (length_mm ** 2) / (pipe_diameter_mm * wall_thickness_mm)

            # Limit B31G applicability to z ≤ 50
            if z > 50:
                allowable_depths.append(np.nan)
                continue

            # Compute Folias factor M
            if method == 'original':
                M = np.sqrt(1.0 + 0.8 * z)
            else:  # Modified B31G
                m_squared = 1.0 + 0.6275 * z - 0.003375 * z**2
                if m_squared <= 0:
                    allowable_depths.append(np.nan)
                    continue
                M = np.sqrt(m_squared)

            # Minimum acceptable pressure ratio (Pf/Po)
            min_pressure_ratio = 1.0 / safety_factor

            # Solve for A/A0 using:
            # Pf/Po = (1 - A/A0) / (1 - (A/A0)/M) >= 1/safety_factor
            # Let x = A/A0 → (1 - x) / (1 - x/M) = 1/safety_factor
            a = 1.0 / M - 1.0
            b = M - min_pressure_ratio
            c = -min_pressure_ratio

            discriminant = b**2 - 4*a*c
            if discriminant < 0 or abs(a) < 1e-12:
                allowable_depths.append(np.nan)
                continue

            # Solve quadratic
            root1 = (-b + np.sqrt(discriminant)) / (2 * a)
            root2 = (-b - np.sqrt(discriminant)) / (2 * a)

            # Choose smallest valid root in [0, 1]
            A_A0_max = None
            for root in (root1, root2):
                if 0 <= root <= 1:
                    if A_A0_max is None or root < A_A0_max:
                        A_A0_max = root

            if A_A0_max is None:
                allowable_depths.append(np.nan)
                continue

            # Convert A/A0 to depth ratio: A/A0 ≈ 0.85 * (d/t)
            d_over_t = A_A0_max / 0.85

            # Cap at 80% wall thickness (B31G applicability)
            d_over_t = min(d_over_t, 0.8)

            max_depth_mm = d_over_t * wall_thickness_mm
            allowable_depths.append(max_depth_mm)

        except Exception:
            allowable_depths.append(np.nan)

    # Convert to numpy array
    allowable_depths = np.array(allowable_depths)

    # Fill NaNs conservatively for plotting
    for i in range(len(allowable_depths)):
        if np.isnan(allowable_depths[i]):
            if i > 0 and not np.isnan(allowable_depths[i-1]):
                allowable_depths[i] = allowable_depths[i-1]
            else:
                allowable_depths[i] = 0.8 * wall_thickness_mm

    return allowable_depths


def create_defect_assessment_summary_table(enhanced_df):
    """
    Create a summary table of defect assessment results by surface location.
    
    Parameters:
    - enhanced_df: DataFrame with enhanced corrosion assessment results
    
    Returns:
    - DataFrame with summary statistics
    """
    if enhanced_df.empty or 'surface location' not in enhanced_df.columns:
        return pd.DataFrame()
    
    # Prepare data
    plot_df = enhanced_df.copy()
    plot_df['depth_mm'] = plot_df['depth [%]'] * plot_df['wall_thickness_used_mm'] / 100
    
    # Group by surface location
    summary_data = []
    
    for surface_loc in plot_df['surface location'].unique():
        if pd.isna(surface_loc):
            surface_loc = 'Unknown'
            
        surface_data = plot_df[plot_df['surface location'] == surface_loc]
        
        if len(surface_data) > 0:
            summary_data.append({
                'Surface Location': surface_loc,
                'Count': len(surface_data),
                'Avg Depth (mm)': surface_data['depth_mm'].mean(),
                'Max Depth (mm)': surface_data['depth_mm'].max(),
                'Avg Length (mm)': surface_data['length [mm]'].mean(),
                'Max Length (mm)': surface_data['length [mm]'].max(),
                'Percentage': len(surface_data) / len(plot_df) * 100
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by count descending
    if not summary_df.empty:
        summary_df = summary_df.sort_values('Count', ascending=False).reset_index(drop=True)
        
        # Format numeric columns
        for col in ['Avg Depth (mm)', 'Max Depth (mm)', 'Avg Length (mm)', 'Max Length (mm)']:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].round(2)
        
        summary_df['Percentage'] = summary_df['Percentage'].round(1)
    
    return summary_df


def create_rstreng_envelope_plot(enhanced_df, pipe_diameter_mm, smys_mpa, safety_factor, maop):
    """
    Create RSTRENG Fitness-for-Service envelope plot showing allowable defect dimensions.
    
    Parameters:
    - enhanced_df: DataFrame with defect data and assessment results
    - pipe_diameter_mm: Pipe outside diameter in mm
    - smys_mpa: Specified Minimum Yield Strength in MPa
    - safety_factor: Safety factor applied
    - maop: Maximum Allowed Operating Pressure
    Returns:
    - Plotly figure object
    """
    
    # Import the RSTRENG calculation function
    from analysis.ffs_calculations import calculate_rstreng_level1
    
    # Filter defects with valid dimensions
    valid_defects = enhanced_df[
        (enhanced_df['depth [%]'].notna()) & 
        (enhanced_df['length [mm]'].notna()) & 
        (enhanced_df['width [mm]'].notna()) & 
        (enhanced_df['depth [%]'] > 0) & 
        (enhanced_df['length [mm]'] > 0) & 
        (enhanced_df['width [mm]'] > 0)
    ].copy()
    
    if len(valid_defects) == 0:
        # Return empty plot if no valid data
        fig = go.Figure()
        fig.add_annotation(
            text="No valid defect data available for RSTRENG envelope analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color='#2C3E50')
        )
        fig.update_layout(title="RSTRENG Envelope - No Data")
        return fig
    
    # Get wall thickness (use average if multiple values)
    if 'wall_thickness_used_mm' in valid_defects.columns:
        avg_wall_thickness = valid_defects['wall_thickness_used_mm'].mean()
    else:
        # Fallback to a typical value
        avg_wall_thickness = 10.0
    
    # Calculate RSTRENG envelope curve
    length_range = np.logspace(np.log10(5), np.log10(1000), 50)  # 5mm to 1000mm
    envelope_depths = []
    
    for length_mm in length_range:
        # Find maximum allowable depth for this length
        max_allowable_depth = find_max_allowable_depth_rstreng(length_mm, avg_wall_thickness, pipe_diameter_mm, smys_mpa, safety_factor, maop)
        envelope_depths.append(max_allowable_depth)
    
    # Create the plot
    fig = go.Figure()
    
    # Color mapping for surface location
    color_map = {
        'INT': '#87CEEB',        # Light blue for internal
        'NON-INT': '#90EE90',    # Light green for external  
        'INTERNAL': '#87CEEB',   # Alternative internal naming
        'EXTERNAL': '#90EE90',   # Alternative external naming
        'N/A': '#FFB6C1',       # Light pink for unknown
        'UNKNOWN': '#FFB6C1'     # Alternative unknown naming
    }
    
    # Add defect data points by surface location
    surface_locations = valid_defects['surface location'].fillna('N/A').unique()
    
    for surface in surface_locations:
        surface_data = valid_defects[valid_defects['surface location'].fillna('N/A') == surface]
        
        if len(surface_data) == 0:
            continue
            
        # Determine color and symbol
        color = color_map.get(surface, '#FFB6C1')
        symbol = 'circle' if surface in ['INT', 'INTERNAL'] else 'triangle-up' if surface in ['NON-INT', 'EXTERNAL'] else 'diamond'
        
        # Create hover text with defect information
        hover_text = (
            "Length: " + surface_data['length [mm]'].round(1).astype(str) + " mm<br>" +
            "Depth: " + surface_data['depth [%]'].round(1).astype(str) + "%<br>" +
            "Surface: " + surface
        )
        
        fig.add_trace(go.Scattergl(
            x=surface_data['length [mm]'],
            y=surface_data['depth [%]'],
            mode='markers',
            name=surface,
            marker=dict(
                color=color,
                size=8,
                symbol=symbol,
                line=dict(color='white', width=1),
                opacity=0.7
            ),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text
        ))
    
    # Add RSTRENG envelope curve
    fig.add_trace(go.Scattergl(
        x=length_range,
        y=envelope_depths,
        mode='lines',
        name='RSTRENG Envelope',
        line=dict(color='red', width=3),
        hovertemplate="Length: %{x:.1f} mm<br>Max Allowable Depth: %{y:.1f}%<extra></extra>"
    ))
    
    # Update layout for professional appearance
    fig.update_layout(
        title={
            'text': 'RSTRENG Fitness-for-Service Assessment Envelope',
            'x': 0.5,
            'font': {'size': 16, 'color': '#2C3E50'}
        },
        xaxis=dict(
            title='Defect Length (mm)',
            type='log',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            range=[np.log10(5), np.log10(1000)]
        ),
        yaxis=dict(
            title='Defect Depth (%)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            range=[0, 100]
        ),
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        width=1000,
        height=600,
        margin=dict(l=80, r=150, t=80, b=80)
    )
    
    # Add pipe parameters annotation
    params_text = (
        f"<b>Pipe Parameters:</b><br>"
        f"Wall thickness: {avg_wall_thickness:.1f} mm<br>"
        f"Outside diameter: {pipe_diameter_mm:.1f} mm<br>"
        f"SMYS: {smys_mpa:.0f} MPa<br>"
        f"Safety factor: {safety_factor:.2f}"
    )
    
    fig.add_annotation(
        text=params_text,
        xref="paper", yref="paper",
        x=1.02, y=0.5,
        showarrow=False,
        font=dict(size=10, color='#2C3E50'),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        align="left"
    )
    
    return fig


def find_max_allowable_depth_rstreng(length_mm, wall_thickness_mm, pipe_diameter_mm, smys_mpa, safety_factor, maop):
    """
    Find maximum allowable depth for given defect length using RSTRENG method.
    Uses binary search to find the depth that gives ERF ≈ 1.0.
    """
    from analysis.ffs_calculations import calculate_rstreng_level1
    
    # Binary search parameters
    min_depth = 0.1  # Start from 0.1%
    max_depth = 80.0  # RSTRENG limit
    tolerance = 0.1  # Depth tolerance in %
    max_iterations = 50
    
    # Assume typical width (can be refined)
    typical_width_mm = min(length_mm * 0.5, 50.0)  # Conservative assumption
    
    for _ in range(max_iterations):
        test_depth = (min_depth + max_depth) / 2.0
        
        try:
            result = calculate_rstreng_level1(
                defect_depth_pct=test_depth,
                defect_length_mm=length_mm,
                defect_width_mm=typical_width_mm,
                pipe_diameter_mm=pipe_diameter_mm,
                wall_thickness_mm=wall_thickness_mm,
                maop_mpa=maop,
                smys_mpa=smys_mpa,
                safety_factor=safety_factor
            )
            
            if result['safe'] and result['failure_pressure_mpa'] > 0:
                # Calculate ERF (using user's definition: MAOP/Safe_Pressure)
                # We want to find where this equals 1.0 (i.e., safe_pressure = MAOP)
                # For envelope, we can assume MAOP = design pressure
                erf = maop / result['safe_pressure_mpa']
                
                if abs(erf - 1.0) < 0.05:  # Within 5% of limit
                    return test_depth
                elif erf > 1.0:  # Too deep, reduce depth
                    max_depth = test_depth
                else:  # Too shallow, increase depth
                    min_depth = test_depth
            else:
                # Calculation failed, reduce depth
                max_depth = test_depth
                
        except:
            # Error in calculation, reduce depth
            max_depth = test_depth
        
        if abs(max_depth - min_depth) < tolerance:
            break
    
    return (min_depth + max_depth) / 2.0