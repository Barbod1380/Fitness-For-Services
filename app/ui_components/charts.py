"""
Form components for the Pipeline Analysis application.
"""
import streamlit as st
from app.ui_components.ui_elements import custom_metric

def create_metrics_row(metrics_data):
    """
    Create a row of metrics with values.
    
    Parameters:
    - metrics_data: List of (label, value, description) tuples
    
    Returns:
    - Streamlit element with metrics
    """
    num_metrics = len(metrics_data)
    cols = st.columns(num_metrics)
    
    for i, (label, value, description) in enumerate(metrics_data):
        with cols[i]:
            st.markdown(custom_metric(label, value, description), unsafe_allow_html=True)
            
    return cols

def create_comparison_metrics(comparison_results):
    """
    Create a row of metrics for comparison results.
    
    Parameters:
    - comparison_results: Dictionary with comparison results
    
    Returns:
    - Streamlit element with metrics
    """
    metrics_data = [
        ("Total Defects", comparison_results['total_defects'], None),
        ("Common Defects", comparison_results['common_defects_count'], None),
        ("New Defects", comparison_results['new_defects_count'], None),
        ("% New Defects", f"{comparison_results['pct_new']:.1f}%", None)
    ]
    
    return create_metrics_row(metrics_data)