import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import streamlit as st


def feature_cat_normalized(length_norm, width_norm):
    """
    Categorizes features based on normalized length and width (already divided by GeoParA).
    Follows ROSEN specifications exactly as defined in the PDF document.
    
    Parameters:
    - length_norm: Normalized length (length_mm / A)
    - width_norm: Normalized width (width_mm / A)
    
    Returns:
    - Category string
    """
    # Handle invalid/zero dimensions
    if length_norm <= 0 or width_norm <= 0:
        return ""
    
    # Calculate L/W ratio for conditions that need it
    l_w_ratio = length_norm / width_norm if width_norm > 0 else float('inf')
    
    # Apply categorization rules EXACTLY as per ROSEN PDF:
    
    # General: [W ≥ 3A] and [L ≥ 3A]
    if width_norm >= 3 and length_norm >= 3:
        return "General"
    
    # Pinhole: [0 < W < 1A] and [0 < L < 1A]
    elif (0 < width_norm < 1) and (0 < length_norm < 1):
        return "PinHole"
    
    # Axial grooving: [1A ≤ W < 3A] and [L/W ≥ 2]
    elif (1 <= width_norm < 3) and (l_w_ratio >= 2):
        return "AxialGroove"
    
    # Circumferential grooving: [L/W ≤ 0.5] and [1A ≤ L < 3A]
    elif (l_w_ratio <= 0.5) and (1 <= length_norm < 3):
        return "CircGroove"
    
    # Axial slotting: [0 < W < 1A] and [L ≥ 1A]
    elif (0 < width_norm < 1) and (length_norm >= 1):
        return "AxialSlot"
    
    # Circumferential slotting: [W ≥ 1A] and [0 < L < 1A]
    elif (width_norm >= 1) and (0 < length_norm < 1):
        return "CircSlot"
    
    # Pitting: {([1A ≤ W < 6A] and [1A ≤ L < 6A] and [0.5 < L/W < 2]) 
    #          and not ([W ≥ 3A] and [L ≥ 3A])}
    elif ((1 <= width_norm < 6) and 
          (1 <= length_norm < 6) and 
          (0.5 < l_w_ratio < 2) and 
          not (width_norm >= 3 and length_norm >= 3)):
        return "Pitting"
    
    # If none of the above conditions are met, return empty string
    else:
        return ""
    

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
    length_values = np.linspace(0, 10, 100)  # Optimized for performance
    width_values = np.linspace(0, 10, 100)
    L, W = np.meshgrid(length_values, width_values)
    
    # Calculate categories for each point
    L_flat, W_flat = L.flatten(), W.flatten()
    categories_flat = np.array([feature_cat_normalized(l, w) for l, w in zip(L_flat, W_flat)])
    categories_grid = categories_flat.reshape(L.shape)
        
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
    all_categories = ["PinHole", "AxialSlot", "CircSlot", "AxialGroove", "CircGroove", "Pitting", "General"]
    
    # Create frequency data with all categories
    category_counts = valid_defects['defect_category'].value_counts()
    
    # Create complete summary with all categories (including 0 counts)
    summary_data = []
    for category in all_categories:
        count = category_counts.get(category, 0)  # Get count or 0 if category doesn't exist
        summary_data.append({'Category': category, 'Count': count})
    
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate percentages (handle case where total is 0)
    total_count = summary_df['Count'].sum()
    if total_count > 0:
        summary_df['Percentage'] = (summary_df['Count'] / total_count * 100).round(1)
    else:
        summary_df['Percentage'] = 0.0
    
    # Sort by count descending, but keep all categories
    summary_df = summary_df.sort_values('Count', ascending=False)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "🔍 Defect Categorization Map", 
            "📊 Category Frequency"
        ],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.05,
        column_widths=[0.55, 0.45]
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
    all_categories = ["PinHole", "AxialSlot", "CircSlot", "AxialGroove", "CircGroove", "Pitting", "General"]
    
    # Add traces for ALL categories (even those with 0 defects)
    for category in all_categories:
        if category in color_discrete_map:
            # Get data for this category
            cat_data = valid_defects[valid_defects['defect_category'] == category]
            
            if len(cat_data) > 0:
                # Category has data - add normal trace
                fig.add_trace(
                    go.Scattergl(
                        x=cat_data['length_normalized'],
                        y=cat_data['width_normalized'],
                        mode='markers',
                        marker=dict(
                            color=color_discrete_map[category],
                            size=7,
                            symbol='circle',
                            line=dict(color='white', width=1),
                            opacity=0.7
                        ),
                        name=f'{category}',
                        hoverinfo='skip',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            else:
                # Category has no data - add invisible trace for legend only
                fig.add_trace(
                    go.Scattergl(
                        x=[None],  # No data points
                        y=[None],  # No data points
                        mode='markers',
                        marker=dict(
                            color=color_discrete_map[category],
                            size=7,
                            symbol='circle',
                            line=dict(color='white', width=1),
                            opacity=0.7
                        ),
                        name=f'{category}',
                        hoverinfo='skip',
                        showlegend=True,
                        visible='legendonly'  # Only show in legend, not on plot
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
            text=[f"{count}<br>({pct}%)" for count, pct in zip(summary_df['Count'], summary_df['Percentage'])],  # Show both count and percentage
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
    Uses Freedman-Diaconis rule for optimal bin selection.
    """
    if dimension_columns is None:
        dimension_columns = {
            'length [mm]': 'Defect Length (mm)',
            'width [mm]': 'Defect Width (mm)',
            'depth [%]': 'Defect Depth (%)'
        }

    def calculate_optimal_bins(data):
        """
        Calculate optimal number of bins using Freedman-Diaconis rule.
        Formula: bin_width = 2 * IQR / n^(1/3)
        """
        n = len(data)
        if n < 2:
            return 10  # Default for very small datasets
        
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        
        if iqr == 0:
            return int(np.sqrt(n))  # Fallback to square root rule if no spread
        
        bin_width = 2 * iqr / (n ** (1/3))
        data_range = data.max() - data.min()
        
        if data_range == 0:
            return 10  # Default if no range
        
        n_bins = int(np.ceil(data_range / bin_width))
        
        # Apply reasonable bounds (5-100 bins)
        return max(5, min(100, n_bins))

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

    # Create subplots with TWO rows, N columns
    n = len(valid_dims)
    fig = make_subplots(
        rows=2,
        cols=n,
        subplot_titles=[title for _, title, _ in valid_dims],
        vertical_spacing=0.08,
        row_heights=[0.3, 0.7]
    )

    # Track bin counts for annotation
    bin_info = []

    # Add box plots in the top row and histograms in the bottom row
    for idx, (col, title, series) in enumerate(valid_dims, start=1):
        # Calculate optimal bin count using Freedman-Diaconis rule
        nbins = calculate_optimal_bins(series)
        bin_info.append(f"{title.split()[1]}: {nbins}")
        
        # Add box plot in top row
        fig.add_trace(
            go.Box(
                x=series,
                name='',
                marker=dict(color='rgba(0,128,255,0.6)'),
                showlegend=False
            ),
            row=1,
            col=idx
        )

        # Add histogram in bottom row with optimal binning
        fig.add_trace(
            go.Histogram(
                x=series,
                nbinsx=nbins,
                marker=dict(color='rgba(0,128,255,0.6)'),
                showlegend=False,
                name=f"{title} Distribution"
            ),
            row=2,
            col=idx
        )

        # Update x-axis label per subplot
        fig.update_xaxes(title_text=title, row=2, col=idx)

    # Enhanced layout with dynamic bin information
    fig.update_layout(
        title_text="Distribution of Defect Dimensions - Freedman-Diaconis Optimal Binning",
        height=600,
        width=300 * n,
        bargap=0.1,
        showlegend=False,
        annotations=[
            dict(
                text=f"📊 Optimal bins (Freedman-Diaconis): {' | '.join(bin_info)}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                xanchor='center',
                font=dict(size=10, color='gray')
            )
        ]
    )
    return {"combined_dimensions": fig} if fig else {}

def create_combined_dimensions_plot(defects_df, joints_df):
    """
    Create a scatter plot showing Defect Aspect Ratio vs Volume Loss.
    More informative for engineering analysis than simple length vs width.

    Parameters:
    - defects_df: DataFrame containing defect information
    - joints_df: DataFrame containing joint information with wall thickness

    Returns:
    - Plotly figure object
    """
    # Step 1: Check required columns
    required_defect_cols = ['length [mm]', 'width [mm]', 'depth [%]', 'joint number']
    required_joint_cols = ['joint number', 'wt nom [mm]']
    
    missing_defect_cols = [col for col in required_defect_cols if col not in defects_df.columns]
    missing_joint_cols = [col for col in required_joint_cols if col not in joints_df.columns]
    
    if missing_defect_cols or missing_joint_cols:
        fig = go.Figure()
        error_message = f"Missing columns: {missing_defect_cols + missing_joint_cols}"
        fig.add_annotation(
            text=error_message,
            xref='paper', yref='paper',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#2C3E50')
        )
        fig.update_layout(
            title='Defect Aspect Ratio vs Volume Loss - Missing Data',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        return fig

    # Step 2: Filter valid data and create wall thickness lookup
    valid_data = defects_df.copy()
    
    # Remove rows with invalid dimensions
    for col in ['length [mm]', 'width [mm]', 'depth [%]']:
        valid_data = valid_data[pd.to_numeric(valid_data[col], errors='coerce').notna()]
        valid_data = valid_data[pd.to_numeric(valid_data[col], errors='coerce') > 0]
    
    if valid_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text='No valid defect dimension data available',
            xref='paper', yref='paper',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#2C3E50')
        )
        fig.update_layout(
            title='Defect Aspect Ratio vs Volume Loss - No Valid Data',
            height=600
        )
        return fig

    # Create wall thickness lookup
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    # Step 3: Calculate engineering parameters
    def get_wall_thickness(joint_number):
        """Get wall thickness with fallback to 10mm if missing"""
        wt = wt_lookup.get(joint_number, 10.0)
        if pd.isna(wt) or wt <= 0:
            wt = 10.0  # Fallback value
        return wt
    
    # Add wall thickness to data
    valid_data['wall_thickness'] = valid_data['joint number'].apply(get_wall_thickness)
    
    # Calculate Aspect Ratio (Length/Width) with zero-division protection
    valid_data['aspect_ratio'] = valid_data.apply(
        lambda row: row['length [mm]'] / max(row['width [mm]'], 0.1),  # Prevent division by zero
        axis=1
    )
    
    # Calculate Volume Loss = Length × Width × (Depth%/100) × Wall_Thickness
    valid_data['volume_loss_mm3'] = (
        valid_data['length [mm]'] * 
        valid_data['width [mm]'] * 
        (valid_data['depth [%]'] / 100.0) * 
        valid_data['wall_thickness']
    )
    
    # Step 4: Calculate defect categories using existing function
    def get_geo_para(joint_number):
        """Calculate GeoParA = max(wall_thickness, 10mm)"""
        wt = wt_lookup.get(joint_number, 10.0)
        if pd.isna(wt):
            wt = 10.0
        return max(wt, 10.0)
    
    valid_data['geo_para_A'] = valid_data['joint number'].apply(get_geo_para)
    valid_data['length_normalized'] = valid_data['length [mm]'] / valid_data['geo_para_A']
    valid_data['width_normalized'] = valid_data['width [mm]'] / valid_data['geo_para_A']
    
    # Apply defect categorization
    valid_data['defect_category'] = valid_data.apply(
        lambda row: feature_cat_normalized(row['length_normalized'], row['width_normalized']), 
        axis=1
    )
    
    # Filter out uncategorized defects for cleaner visualization
    categorized_data = valid_data[valid_data['defect_category'] != ''].copy()
    
    if categorized_data.empty:
        # Fall back to using all data if no categories found
        categorized_data = valid_data.copy()
        categorized_data['defect_category'] = 'Unclassified'
    
    # Step 5: Create color mapping for categories
    color_discrete_map = {
        "PinHole": "#00BCD4",     # Cyan
        "AxialSlot": "#E91E63",   # Pink
        "CircSlot": "#FF9800",    # Orange
        "AxialGroove": "#2196F3", # Blue
        "CircGroove": "#9C27B0",  # Purple
        "Pitting": "#F44336",     # Red
        "General": "#4CAF50",     # Green
        "Unclassified": "#95A5A6" # Gray
    }
    
    # Step 6: Create the enhanced scatter plot
    fig = px.scatter(
        categorized_data,
        x='aspect_ratio',
        y='volume_loss_mm3',
        color='defect_category',
        color_discrete_map=color_discrete_map,
        hover_data={
            'length [mm]': ':.1f',
            'width [mm]': ':.1f', 
            'depth [%]': ':.1f',
            'wall_thickness': ':.1f',
            'aspect_ratio': ':.2f',
            'volume_loss_mm3': ':.1f'
        },
        title='Defect Aspect Ratio vs Volume Loss Analysis',
        labels={
            'aspect_ratio': 'Aspect Ratio (Length ÷ Width)',
            'volume_loss_mm3': 'Volume Loss (mm³)',
            'defect_category': 'Defect Category'
        },
        opacity = 0.7
    )

    # Step 7: Enhanced layout with engineering context
    fig.update_layout(
        width=1000,
        height=600,
        plot_bgcolor='white',
        font=dict(family="Inter, Arial, sans-serif"),
        margin=dict(l=50, r=50, t=80, b=80),
        legend=dict(
            title="Defect Categories",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        )
    )
    
    # Update axes with engineering context
    fig.update_xaxes(
        title_text="Aspect Ratio (Length ÷ Width)<br>",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    
    fig.update_yaxes(
        title_text="Volume Loss (mm³)<br><sub>Material removed by corrosion</sub>",
        type="log", 
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    
    # Add summary statistics
    avg_aspect = categorized_data['aspect_ratio'].mean()
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f"Average Aspect Ratio: {avg_aspect:.2f}",
        showarrow=False,
        font=dict(size=10, color="#2C3E50"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        align="left"
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