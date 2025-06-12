import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import plotly.express as px

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
        
        # Add histogram in bottom row
        fig.add_trace(
            go.Histogram(
                x=series,
                nbinsx=20,
                marker=dict(color='rgba(0,128,255,0.6)'),
                showlegend=False
            ),
            row=2,  # Bottom row
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