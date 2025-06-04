"""
Core UI elements for the Pipeline Analysis application.
"""
import streamlit as st
import base64

def card(title, content):
    """
    Create a custom card with a title and content.
    
    Parameters:
    - title: Card title
    - content: HTML content for the card
    
    Returns:
    - Streamlit markdown element
    """
    card_html = f"""
    <div class="card-container">
        <div class="section-header">{title}</div>
        {content}
    </div>
    """
    return st.markdown(card_html, unsafe_allow_html=True)

def custom_metric(label, value, description=None):
    """
    Create a custom metric display with a value and label.
    
    Parameters:
    - label: Metric name
    - value: Metric value
    - description: Optional description text
    
    Returns:
    - HTML string for the metric
    """
    metric_html = f"""
    <div class="custom-metric">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {f"<div style='font-size:12px;color:#95a5a6;'>{description}</div>" if description else ""}
    </div>
    """
    return metric_html

def status_badge(text, status):
    """
    Create a colored status badge.
    
    Parameters:
    - text: Badge text
    - status: Badge color/status (green, yellow, red)
    
    Returns:
    - HTML string for the badge
    """
    badge_html = f"""
    <span class="status-badge {status}">{text}</span>
    """
    return badge_html

def info_box(text, box_type="info"):
    """
    Create an info, warning, or success box.
    
    Parameters:
    - text: Box content
    - box_type: Box style (info, warning, success)
    
    Returns:
    - Streamlit markdown element
    """
    box_class = f"{box_type}-box"
    box_html = f"""
    <div class="{box_class}">
        {text}
    </div>
    """
    return st.markdown(box_html, unsafe_allow_html=True)

def show_step_indicator(active_step):
    """
    Display a step progress indicator.
    
    Parameters:
    - active_step: Current active step (1-based index)
    """
    steps = ["Upload File", "Map Columns", "Process Data"]
    cols = st.columns(len(steps))
    
    for i, (col, step_label) in enumerate(zip(cols, steps), 1):
        with col:
            if i < active_step:
                emoji = "✅"  # Completed
                color = "green"
            elif i == active_step:
                emoji = "🔵"  # Active
                color = "blue"
            else:
                emoji = "⚪"  # Not started
                color = "gray"
            
            st.markdown(f"### {emoji} **Step {i}**", unsafe_allow_html=True)
            st.caption(step_label)

def create_data_download_links(df, prefix, year):
    """
    Create download links for dataframes.
    
    Parameters:
    - df: DataFrame to download
    - prefix: Prefix for the filename
    - year: Year to include in the filename
    
    Returns:
    - HTML string with the download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{prefix}_{year}.csv" class="custom-button" style="display:inline-block;text-decoration:none;margin-top:10px;font-size:0.8em;padding:5px 10px;">Download {prefix} CSV</a>'
    return href