"""
Functions for analyzing pipeline defect dimensions and statistics.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dimension_distribution_plots(defects_df, dimension_columns=None):
    """
    Create distribution plots for defect dimensions.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - dimension_columns: Dictionary mapping column names to display titles
    
    Returns:
    - Dictionary of plotly figures keyed by dimension name
    """
    if dimension_columns is None:
        dimension_columns = {
            'length [mm]': 'Defect Length (mm)',
            'width [mm]': 'Defect Width (mm)',
            'depth [%]': 'Defect Depth (%)'
        }
    
    figures = {}
    
    for col, title in dimension_columns.items():
        if col in defects_df.columns:
            # Filter out non-numeric values and NaNs
            valid_data = defects_df[pd.to_numeric(defects_df[col], errors='coerce').notna()]
            
            if not valid_data.empty:
                # Create histogram
                fig = px.histogram(
                    valid_data,
                    x=col,
                    nbins=20,
                    marginal="box",  # Add a box plot to show quartiles
                    title=f"Distribution of {title}",
                    labels={col: title},
                    color_discrete_sequence=['rgba(0, 128, 255, 0.6)']
                )
                
                # Format
                fig.update_layout(
                    xaxis_title=title,
                    yaxis_title="Count",
                    bargap=0.1,
                    height=400
                )
                
                figures[col] = fig
    return figures

def create_combined_dimensions_plot(defects_df):
    """
    Create a scatter plot showing the relationship between length, width and depth.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    
    Returns:
    - Plotly figure object
    """
    # Check if we have the necessary columns
    req_cols = ['length [mm]', 'width [mm]']
    has_depth = 'depth [%]' in defects_df.columns
    
    if not all(col in defects_df.columns for col in req_cols):
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Required dimension columns not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Filter out NaNs and invalid values
    valid_data = defects_df.copy()
    for col in req_cols:
        valid_data = valid_data[pd.to_numeric(valid_data[col], errors='coerce').notna()]
    
    if has_depth:
        valid_data = valid_data[pd.to_numeric(valid_data['depth [%]'], errors='coerce').notna()]
    
    if valid_data.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No valid dimension data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calculate area
    valid_data['area [mm²]'] = valid_data['length [mm]'] * valid_data['width [mm]']
    
    # Create scatter plot
    if has_depth:
        fig = px.scatter(
            valid_data,
            x='length [mm]',
            y='width [mm]',
            color='depth [%]',
            size='area [mm²]',
            hover_name='component / anomaly identification' if 'component / anomaly identification' in valid_data.columns else None,
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Defect Dimensions Relationship",
            labels={
                'length [mm]': 'Length (mm)',
                'width [mm]': 'Width (mm)',
                'depth [%]': 'Depth (%)',
                'area [mm²]': 'Area (mm²)'
            }
        )
    else:
        fig = px.scatter(
            valid_data,
            x='length [mm]',
            y='width [mm]',
            size='area [mm²]',
            hover_name='component / anomaly identification' if 'component / anomaly identification' in valid_data.columns else None,
            title="Defect Dimensions Relationship",
            labels={
                'length [mm]': 'Length (mm)',
                'width [mm]': 'Width (mm)',
                'area [mm²]': 'Area (mm²)'
            }
        )    

    fig.update_layout(
        height=500,
        xaxis=dict(
            title="Defect Length (mm)",
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title="Defect Width (mm)", 
            tickfont=dict(size=14)
        )
    )
    
    # Add explanation for bubble size and color
    legend_text = "Bubble size represents defect area (mm²)"
    if has_depth:
        legend_text += ", color represents depth (%)"
    
    fig.add_annotation(
        text=legend_text,
        x=-0.0,
        y=-0.25,
        xref="paper",
        yref="paper",
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
    
    # Calculate statistics for each dimension
    stats = []
    for col in available_cols:
        # Convert to numeric, coercing errors to NaN
        values = pd.to_numeric(defects_df[col], errors='coerce')
        
        # Skip if all values are NaN
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
    joint_data = joints_df[joints_df["joint number"] == selected_joint]
    
    # Check if joint exists
    if joint_data.empty:
        return {
            "defect_count": 0,
            "defect_types": {},
            "joint_length": "N/A",
            "joint_position": "N/A",
            "severity_rank": "N/A"
        }
    
    joint_length = joint_data.iloc[0]["joint length [m]"]
    joint_position = joint_data.iloc[0]["log dist. [m]"]
    
    # Get defects for this joint
    joint_defects = defects_df[defects_df["joint number"] == selected_joint]
    defect_count = len(joint_defects)
    
    # Get defect types
    defect_types = {}
    if defect_count > 0 and "component / anomaly identification" in joint_defects.columns:
        defect_types = joint_defects["component / anomaly identification"].value_counts().to_dict()
    
    # Calculate severity metrics for all joints to get ranking
    all_joints = defects_df["joint number"].unique()
    joint_severity = []
    
    for joint in all_joints:
        joint_def = defects_df[defects_df["joint number"] == joint]
        
        # Use maximum depth as severity metric if available
        if "depth [%]" in joint_def.columns and not joint_def["depth [%]"].empty:
            max_depth = joint_def["depth [%]"].max()
        else:
            # Fallback to defect count if depth not available
            max_depth = len(joint_def)
            
        joint_severity.append({"joint": joint, "severity": max_depth})
    
    # Create DataFrame and sort by severity descending
    severity_df = pd.DataFrame(joint_severity)
    
    # Handle ranking
    if not severity_df.empty:
        severity_df = severity_df.sort_values("severity", ascending=False)
        severity_df["rank"] = range(1, len(severity_df) + 1)
        
        # Get rank of selected joint - safely
        joint_rank_rows = severity_df[severity_df["joint"] == selected_joint]
        if not joint_rank_rows.empty:
            joint_rank = joint_rank_rows["rank"].iloc[0]
            rank_text = f"{int(joint_rank)} of {len(all_joints)}"
        else:
            # Joint has no defects, so it's not in the severity dataframe
            rank_text = f"N/A (no defects)"
    else:
        rank_text = "N/A"
    
    return {
        "defect_count": defect_count,
        "defect_types": defect_types,
        "joint_length": joint_length,
        "joint_position": joint_position,
        "severity_rank": rank_text
    }