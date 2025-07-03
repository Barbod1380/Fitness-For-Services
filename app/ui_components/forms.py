"""
Form components for the Pipeline Analysis application.
"""
import streamlit as st
from app.ui_components.ui_elements import info_box
from core.column_mapping import get_missing_required_columns, STANDARD_COLUMNS, REQUIRED_COLUMNS

def create_column_mapping_form(df, year, suggested_mapping):
    """
    Create a form for mapping columns from the uploaded file to standard column names.
    
    Parameters:
    - df: DataFrame with the uploaded data
    - year: Year for the data
    - suggested_mapping: Dict with suggested column mappings
    
    Returns:
    - Dict with the confirmed column mappings
    """
    st.info("""
        **Column Mapping Instructions:**
        Match your file's columns to standard column names. Required fields are marked with *.
        This mapping ensures consistent analysis across different data formats.
    """)
    
    # Create UI for mapping confirmation
    st.write("Confirm the mapping between your file's columns and standard columns:")
    
    confirmed_mapping = {}
    all_columns = [None] + df.columns.tolist()
    
    # Create three columns for the mapping UI to save space
    col1, col2, col3 = st.columns(3)
    
    # Split the standard columns into three groups
    third = len(STANDARD_COLUMNS) // 3
    remaining = len(STANDARD_COLUMNS) % 3
    
    # Calculate split points for columns
    if remaining == 1:
        # First column gets one extra
        split1 = third + 1
        split2 = split1 + third
    elif remaining == 2:
        # First and second columns get one extra each
        split1 = third + 1
        split2 = split1 + third + 1
    else:
        # Even distribution
        split1 = third
        split2 = split1 + third
    
    # First column of mappings
    with col1:
        for std_col in STANDARD_COLUMNS[:split1]:
            suggested = suggested_mapping.get(std_col)
            index = 0 if suggested is None else all_columns.index(suggested)
            
            is_required = std_col in REQUIRED_COLUMNS
            label = f"{std_col}" + (" *" if is_required else "")
            
            selected = st.selectbox(
                label,
                options=all_columns,
                index=index,
                key=f"map_{year}_{std_col}"
            )
            confirmed_mapping[std_col] = selected
    
    # Second column of mappings
    with col2:
        for std_col in STANDARD_COLUMNS[split1:split2]:
            suggested = suggested_mapping.get(std_col)
            index = 0 if suggested is None else all_columns.index(suggested)
            
            is_required = std_col in REQUIRED_COLUMNS
            label = f"{std_col}" + (" *" if is_required else "")
            
            selected = st.selectbox(
                label,
                options=all_columns,
                index=index,
                key=f"map_{year}_{std_col}_col2"
            )
            confirmed_mapping[std_col] = selected
    
    # Third column of mappings
    with col3:
        for std_col in STANDARD_COLUMNS[split2:]:
            suggested = suggested_mapping.get(std_col)
            index = 0 if suggested is None else all_columns.index(suggested)
            
            is_required = std_col in REQUIRED_COLUMNS
            label = f"{std_col}" + (" *" if is_required else "")
            
            selected = st.selectbox(
                label,
                options=all_columns,
                index=index,
                key=f"map_{year}_{std_col}_col3"
            )
            confirmed_mapping[std_col] = selected
    
    # Add note about required fields
    st.markdown('<div style="margin-top:10px;font-size:0.8em;">* Required fields</div>', unsafe_allow_html=True)
    
    # Check for missing required columns
    missing_cols = get_missing_required_columns(confirmed_mapping)
    if missing_cols:
        info_box(f"Missing required columns: {', '.join(missing_cols)}. You may proceed, but functionality may be limited.", "warning")
    
    return confirmed_mapping