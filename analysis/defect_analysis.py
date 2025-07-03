import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import streamlit as st

def feature_cat_normalized(length_norm, width_norm):
    """
    Categorizes features based on normalized length and width (already divided by GeoParA).
    Since data is normalized by A, we use 1 instead of GeoParA in conditions.
    
    Parameters:
    - length_norm: Normalized length (length_mm / A)
    - width_norm: Normalized width (width_mm / A)
    
    Returns:
    - Category string
    """
    if length_norm == 0 or width_norm == 0:
        return ""
    
    if (width_norm >= 3 and length_norm >= 3):
        return "General"
    elif (width_norm < 1 and length_norm < 1):
        return "PinHole"
    elif (width_norm >= 1 and width_norm <= 3 and (length_norm / width_norm >= 2)):
        return "AxialGroove"
    elif (width_norm < 1 and length_norm >= 1):
        return "AxialSlot"
    elif ((length_norm / width_norm) <= 0.5 and (length_norm > 1 and length_norm < 3)):
        return "CircGroove"
    elif (width_norm >= 1 and length_norm < 1):
        return "CircSlot"
    else:
        return "Pitting"


def create_clean_combined_defect_plot(defects_df, joints_df, title_suffix = ""):
    """
    Create a clean combined plot with defect categorization map and frequency chart side by side.
    No annotations or hover - just the pure visualization.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - joints_df: DataFrame containing joint information with wall thickness
    
    Returns:
    - Plotly figure object with two clean subplots
    """
    # Check required columns
    required_defect_cols = ['length [mm]', 'width [mm]', 'joint number']
    required_joint_cols = ['joint number', 'wt nom [mm]']
    
    missing_defect_cols = [col for col in required_defect_cols if col not in defects_df.columns]
    missing_joint_cols = [col for col in required_joint_cols if col not in joints_df.columns]
    
    if missing_defect_cols or missing_joint_cols:
        # Create subplots with dynamic titles
        left_title = f"🔍 Defect Categorization Map{title_suffix}"
        right_title = f"📊 Category Frequency{title_suffix}"
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[left_title, right_title],
            specs=[[{"type": "xy"}, {"type": "xy"}]],
            horizontal_spacing=0.05,
            column_widths=[0.5, 0.5]
        )
        
        error_text = "Missing required columns"
        fig.add_annotation(
            text=error_text,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#2C3E50')
        )
        fig.update_layout(
            title="Defect Categorization - Missing Data",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        return fig
    
    # Create wall thickness lookup
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    # Filter defects with valid dimensions and joint numbers
    valid_defects = defects_df.dropna(subset=required_defect_cols).copy()
    
    if len(valid_defects) == 0:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Categorization Map", "Frequency Chart"],
            specs=[[{"type": "xy"}, {"type": "xy"}]],
            horizontal_spacing=0.05,
            column_widths=[0.5, 0.5]
        )
        
        fig.add_annotation(
            text="No valid defect data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#2C3E50')
        )
        fig.update_layout(
            title="Defect Categorization - No Valid Data",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        return fig
    
    # Calculate GeoParA and normalize dimensions
    def get_geo_para(joint_number):
        """Calculate GeoParA = max(wall_thickness, 10mm)"""
        wt = wt_lookup.get(joint_number, 10.0)
        if pd.isna(wt):
            wt = 10.0
        return max(wt, 10.0)
    
    valid_defects['geo_para_A'] = valid_defects['joint number'].apply(get_geo_para)
    valid_defects['length_normalized'] = valid_defects['length [mm]'] / valid_defects['geo_para_A']
    valid_defects['width_normalized'] = valid_defects['width [mm]'] / valid_defects['geo_para_A']
    
    # Apply categorization
    valid_defects['defect_category'] = valid_defects.apply(
        lambda row: feature_cat_normalized(row['length_normalized'], row['width_normalized']), 
        axis=1
    )
    
    # Create the theoretical grid for background
    length_values = np.linspace(0, 10, 150)  # Optimized for performance
    width_values = np.linspace(0, 10, 150)
    L, W = np.meshgrid(length_values, width_values)
    
    # Calculate categories for each point
    categories_grid = np.vectorize(feature_cat_normalized)(L, W)
    
    # Define category mapping and modern color palette
    category_map = {
        "PinHole": 0,
        "AxialSlot": 1, 
        "CircSlot": 2,
        "AxialGroove": 3,
        "CircGroove": 4,
        "Pitting": 5,
        "General": 6,
        "": 7
    }
    
    # Convert categories to numbers for the background
    category_numbers = np.zeros_like(categories_grid, dtype=int)
    for cat, num in category_map.items():
        category_numbers[categories_grid == cat] = num
    
    # Modern color schemes
    background_colors = [
        '#E8F6F3',  # Light mint for PinHole
        '#F8C8DC',  # Light pink for AxialSlot
        '#FFE5CC',  # Light peach for CircSlot  
        '#E1F5FE',  # Light blue for AxialGroove
        '#F3E5F5',  # Light purple for CircGroove
        '#FFEBEE',  # Light red for Pitting
        '#E8F5E8',  # Light green for General
        '#FAFAFA'   # Light gray for empty
    ]
    
    color_discrete_map = {
        "PinHole": "#00BCD4",     # Cyan
        "AxialSlot": "#E91E63",   # Pink
        "CircSlot": "#FF9800",    # Orange
        "AxialGroove": "#2196F3", # Blue
        "CircGroove": "#9C27B0",  # Purple
        "Pitting": "#F44336",     # Red
        "General": "#4CAF50"      # Green
    }
    
    # Create frequency data
    summary_df = valid_defects['defect_category'].value_counts().reset_index()
    summary_df.columns = ['Category', 'Count']
    summary_df['Percentage'] = (summary_df['Count'] / summary_df['Count'].sum() * 100).round(1)
    summary_df = summary_df.sort_values('Count', ascending=False)
    
    total_defects = len(valid_defects)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "🔍 Defect Categorization Map", 
            "📊 Category Frequency"
        ],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.05,
        column_widths=[0.5, 0.5]
    )
    
    # =============================================================================
    # LEFT SUBPLOT: Categorization Map
    # =============================================================================
    
    # Add background heatmap
    fig.add_trace(
        go.Heatmap(
            z=category_numbers,
            x=length_values,
            y=width_values,
            colorscale=[[i/(len(background_colors)-1), background_colors[i]] for i in range(len(background_colors))],
            showscale=False,
            hoverinfo='skip',
            name='Background',
            opacity=0.6
        ),
        row=1, col=1
    )
    
    # Add actual defect data points (no hover, clean)
    for category in valid_defects['defect_category'].unique():
        if category and category in color_discrete_map:
            cat_data = valid_defects[valid_defects['defect_category'] == category]
            
            fig.add_trace(
                go.Scatter(
                    x=cat_data['length_normalized'],
                    y=cat_data['width_normalized'],
                    mode='markers',
                    marker=dict(
                        color=color_discrete_map[category],
                        size=7,  # Small markers for clean look
                        symbol='circle',
                        line=dict(color='white', width=1),
                        opacity=0.7
                    ),
                    name=f'{category}',
                    hoverinfo='skip',  # No hover
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # =============================================================================
    # RIGHT SUBPLOT: Frequency Bar Chart
    # =============================================================================
    
    # Map colors to categories
    bar_colors = [color_discrete_map.get(cat, "#95A5A6") for cat in summary_df['Category']]
    
    fig.add_trace(
        go.Bar(
            x=summary_df['Category'],
            y=summary_df['Count'],
            text=[f"{count}" for count in summary_df['Count']],  # Just show count, clean
            textposition='outside',
            textfont=dict(size=11, color='#2C3E50', family="Inter, Arial, sans-serif"),
            marker=dict(
                color=bar_colors,
                line=dict(color='white', width=1.5),
                opacity=0.8
            ),
            hoverinfo='skip',  # No hover
            name="",
            showlegend=False
        ),
        row=1, col=2
    )
    
    # =============================================================================
    # Update Layout - Clean and Professional
    # =============================================================================
    
    fig.update_layout(
        width=1400,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif"),
        margin=dict(l=0, r=20, t=80, b=80),
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=0.1,
            font=dict(size=11)
        )
    )
    
    # Update left subplot (categorization map) axes
    fig.update_xaxes(
        title_text="Normalized Length (Length ÷ A)",
        range=[0, 10],
        constrain='domain',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        linecolor='rgba(128,128,128,0.5)',
        title_font=dict(size=12, color='#2C3E50'),
        tickfont=dict(size=10, color='#2C3E50'),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Normalized Width (Width ÷ A)",
        range=[0, 10],
        scaleanchor="x",
        scaleratio=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        linecolor='rgba(128,128,128,0.5)',
        title_font=dict(size=12, color='#2C3E50'),
        tickfont=dict(size=10, color='#2C3E50'),
        row=1, col=1
    )
    
    # Update right subplot (frequency chart) axes
    fig.update_xaxes(
        title_text="Category",
        showgrid=False,
        linecolor='rgba(128,128,128,0.5)',
        title_font=dict(size=12, color='#2C3E50'),
        tickangle=45,
        tickfont=dict(size=10, color='#2C3E50'),
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Count",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        linecolor='rgba(128,128,128,0.5)',
        title_font=dict(size=12, color='#2C3E50'),
        tickfont=dict(size=10, color='#2C3E50'),
        row=1, col=2
    )
    
    return fig


def create_defect_categorization_summary_table(defects_df, joints_df):
    """
    Create a summary table of defect categories.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - joints_df: DataFrame containing joint information
    
    Returns:
    - DataFrame with category summary
    """
    # Check required columns
    required_defect_cols = ['length [mm]', 'width [mm]', 'joint number']
    if not all(col in defects_df.columns for col in required_defect_cols):
        return pd.DataFrame()
    
    if 'wt nom [mm]' not in joints_df.columns:
        return pd.DataFrame()
    
    # Create wall thickness lookup
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    # Filter valid defects
    valid_defects = defects_df.dropna(subset=required_defect_cols).copy()
    
    if len(valid_defects) == 0:
        return pd.DataFrame()
    
    # Calculate normalized dimensions and categories
    def get_geo_para(joint_number):
        wt = wt_lookup.get(joint_number, 10.0)
        if pd.isna(wt):
            wt = 10.0
        return max(wt, 10.0)
    
    valid_defects['geo_para_A'] = valid_defects['joint number'].apply(get_geo_para)
    valid_defects['length_normalized'] = valid_defects['length [mm]'] / valid_defects['geo_para_A']
    valid_defects['width_normalized'] = valid_defects['width [mm]'] / valid_defects['geo_para_A']
    
    valid_defects['defect_category'] = valid_defects.apply(
        lambda row: feature_cat_normalized(row['length_normalized'], row['width_normalized']), 
        axis=1
    )
    
    # Create summary
    summary = valid_defects['defect_category'].value_counts().reset_index()
    summary.columns = ['Category', 'Count']
    summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(1)
    
    # Add category descriptions with emojis
    category_descriptions = {
        "PinHole": "🔹 Small localized defects",
        "AxialSlot": "↕️ Narrow axial features", 
        "CircSlot": "↔️ Narrow circumferential features",
        "AxialGroove": "📏 Elongated axial patterns",
        "CircGroove": "🔄 Elongated circumferential patterns",
        "General": "⚪ Large area defects",
        "Pitting": "🔴 General corrosion patterns"
    }
    
    summary['Description'] = summary['Category'].map(category_descriptions).fillna('❓ Unknown category')
    
    # Sort by count descending
    summary = summary.sort_values('Count', ascending=False).reset_index(drop=True)
    
    # Add rank
    summary['Rank'] = range(1, len(summary) + 1)
    
    # Reorder columns for better presentation
    summary = summary[['Rank', 'Category', 'Description', 'Count', 'Percentage']]
    
    return summary

def create_dimension_distribution_plots(defects_df, dimension_columns=None):
    """
    Create and display a combined Plotly figure with histograms and box plots for defect dimensions.

    Parameters:
    - defects_df: DataFrame containing defect information
    - dimension_columns: dict mapping column names to display titles

    Returns:
    - Combined Plotly Figure object (or None if no valid data)
    """
    if dimension_columns is None:
        dimension_columns = {
            'length [mm]': 'Defect Length (mm)',
            'width [mm]': 'Defect Width (mm)',
            'depth [%]': 'Defect Depth (%)'
        }

    valid_dims = []
    for col, title in dimension_columns.items():
        if col not in defects_df.columns:
            continue
        series = pd.to_numeric(defects_df[col], errors='coerce').dropna()
        if series.empty:
            continue
        valid_dims.append((col, title, series))

    if not valid_dims:
        st.warning("No valid dimension data to plot.")
        return None

    # 2) Create subplots with TWO rows, N columns
    n = len(valid_dims)
    fig = make_subplots(
        rows=2,
        cols=n,
        subplot_titles=[title for _, title, _ in valid_dims],
        vertical_spacing=0.08,  # Space between rows
        row_heights=[0.3, 0.7]  # Box plots smaller, histograms larger
    )

    # 3) Add box plots in the top row and histograms in the bottom row
    for idx, (col, title, series) in enumerate(valid_dims, start=1):
        # Add box plot in top row
        fig.add_trace(
            go.Box(
                x=series,
                name='',
                marker=dict(color='rgba(0,128,255,0.6)'),
                showlegend=False
            ),
            row=1,  # Top row
            col=idx
        )

        # Add histogram (row 2)
        fig.add_trace(
            go.Histogram(
                x=series,
                nbinsx=20,
                marker=dict(color='rgba(0,128,255,0.6)'),
                showlegend=False
            ),
            row=2,
            col=idx
        )

        # Axis label per subplot
        fig.update_xaxes(title_text=title, row=2, col=idx)

    # 4) Layout tweaks
    fig.update_layout(
        title_text="Distribution of Defect Dimensions",
        height=600,  # Increased height for 2 rows
        width=300 * n,
        bargap=0.1,
        showlegend=False
    )
    return {"combined_dimensions": fig} if fig else {}


def create_combined_dimensions_plot(defects_df):
    """
    Create a scatter plot showing the relationship between length, width, and depth.

    Parameters:
    - defects_df: DataFrame containing defect information

    Returns:
    - Plotly figure object
    """
    required_cols = ['length [mm]', 'width [mm]']
    has_depth = 'depth [%]' in defects_df.columns

    # Check if required columns exist
    if not all(col in defects_df.columns for col in required_cols):
        fig = go.Figure()
        fig.add_annotation(
            text='Required dimension columns not available',
            xref='paper',
            yref='paper',
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Filter out invalid or NaN values for required columns
    valid_data = defects_df.copy()
    for col in required_cols:
        valid_data = valid_data[
            pd.to_numeric(valid_data[col], errors='coerce').notna()
        ]

    if has_depth:
        valid_data = valid_data[
            pd.to_numeric(valid_data['depth [%]'], errors='coerce').notna()
        ]

    if valid_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text='No valid dimension data available',
            xref='paper',
            yref='paper',
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Calculate defect area
    valid_data['area [mm²]'] = (
        valid_data['length [mm]'] * valid_data['width [mm]']
    )

    # Create scatter plot
    if has_depth:
        fig = px.scatter(
            valid_data,
            x='length [mm]',
            y='width [mm]',
            color='depth [%]',
            size='area [mm²]',
            hover_name='component / anomaly identification',
            color_continuous_scale=px.colors.sequential.Viridis,
            title='Defect Dimensions Relationship',
            labels={
                'length [mm]': 'Length (mm)',
                'width [mm]': 'Width (mm)',
                'depth [%]': 'Depth (%)',
                'area [mm²]': 'Area (mm²)'
            }
        )
    else:
        hover_field = (
            'component / anomaly identification'
            if 'component / anomaly identification' in valid_data.columns
            else None
        )
        fig = px.scatter(
            valid_data,
            x='length [mm]',
            y='width [mm]',
            size='area [mm²]',
            hover_name=hover_field,
            title='Defect Dimensions Relationship',
            labels={
                'length [mm]': 'Length (mm)',
                'width [mm]': 'Width (mm)',
                'area [mm²]': 'Area (mm²)'
            }
        )

    # Add buttons to control marker size
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='left',
                buttons=[
                    dict(
                        args=[{'marker.size': valid_data['area [mm²]'] * 1}],
                        label='Small',
                        method='restyle'
                    ),
                    dict(
                        args=[{'marker.size': valid_data['area [mm²]'] * 2}],
                        label='Medium',
                        method='restyle'
                    ),
                    dict(
                        args=[{'marker.size': valid_data['area [mm²]'] * 4}],
                        label='Large',
                        method='restyle'
                    )
                ],
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.11,
                xanchor='left',
                y=1.1,
                yanchor='top'
            )
        ]
    )

    # Add explanation for bubble size and color
    legend_text = 'Bubble size represents defect area (mm²)'
    if has_depth:
        legend_text += ', color represents depth (%)'

    fig.add_annotation(
        text=legend_text,
        x=-0.0,
        y=-0.25,
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=12)
    )

    return fig


def create_dimension_statistics_table(defects_df):
    """
    Create a statistics summary table for defect dimensions.

    Parameters:
    - defects_df: DataFrame containing defect information

    Returns:
    - DataFrame with dimension statistics
    """
    dimension_cols = ['length [mm]', 'width [mm]', 'depth [%]']
    available_cols = [col for col in dimension_cols if col in defects_df.columns]

    if not available_cols:
        return pd.DataFrame()

    stats = []
    for col in available_cols:
        values = pd.to_numeric(defects_df[col], errors='coerce')

        if values.isna().all():
            continue

        stat = {
            'Dimension': col,
            'Mean': values.mean(),
            'Median': values.median(),
            'Min': values.min(),
            'Max': values.max(),
            'Std Dev': values.std(),
            'Count': values.count()
        }
        stats.append(stat)

    return pd.DataFrame(stats)


def create_joint_summary(defects_df, joints_df, selected_joint):
    """
    Create a summary of a selected joint with defect count, types, length, and severity ranking.

    Parameters:
    - defects_df: DataFrame containing defect information
    - joints_df: DataFrame containing joint information
    - selected_joint: The joint number to analyze

    Returns:
    - dict: Dictionary with summary information
    """
    # Get joint data
    joint_data = joints_df[joints_df['joint number'] == selected_joint]

    if joint_data.empty:
        return {
            'defect_count': 0,
            'defect_types': {},
            'joint_length': 'N/A',
            'joint_position': 'N/A',
            'severity_rank': 'N/A'
        }

    joint_length = joint_data.iloc[0]['joint length [m]']
    joint_position = joint_data.iloc[0]['log dist. [m]']

    # Get defects for this joint
    joint_defects = defects_df[defects_df['joint number'] == selected_joint]
    defect_count = len(joint_defects)

    # Count defect types if available
    defect_types = {}
    if defect_count > 0 and 'component / anomaly identification' in joint_defects.columns:
        defect_types = joint_defects[
            'component / anomaly identification'
        ].value_counts().to_dict()

    # Calculate severity metric for each joint (max depth or defect count)
    all_joints = defects_df['joint number'].unique()
    joint_severity = []

    for joint in all_joints:
        joint_def = defects_df[defects_df['joint number'] == joint]

        if 'depth [%]' in joint_def.columns and not joint_def['depth [%]'].empty:
            max_depth = joint_def['depth [%]'].max()
        else:
            max_depth = len(joint_def)

        joint_severity.append({'joint': joint, 'severity': max_depth})

    severity_df = pd.DataFrame(joint_severity)

    if not severity_df.empty:
        severity_df = severity_df.sort_values('severity', ascending=False)
        severity_df['rank'] = range(1, len(severity_df) + 1)

        joint_rank_rows = severity_df[severity_df['joint'] == selected_joint]
        if not joint_rank_rows.empty:
            joint_rank = joint_rank_rows['rank'].iloc[0]
            rank_text = f'{int(joint_rank)} of {len(all_joints)}'
        else:
            rank_text = 'N/A (no defects)'
    else:
        rank_text = 'N/A'

    return {
        'defect_count': defect_count,
        'defect_types': defect_types,
        'joint_length': joint_length,
        'joint_position': joint_position,
        'severity_rank': rank_text
    }