"""
Improved form components for the Pipeline Analysis application.
Enhanced column mapping with better UX while maintaining lightweight design.
"""

import streamlit as st
from app.ui_components.ui_elements import info_box
from core.data_pipeline import get_missing_required_columns, STANDARD_COLUMNS, REQUIRED_COLUMNS

def create_column_mapping_form(df, year, suggested_mapping):
    """
    Professionally styled Streamlit form for column mapping.
    """
    st.markdown("""
        <style>
        .mapping-card {
            background-color: #FFFFFF;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 15px 10px 15px 10px;
            margin-bottom: 15px;
            transition: box-shadow 0.3s ease;
        }
        .mapping-card:hover {
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }
        .mapping-header {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #111827;
        }
        .required-label {
            color: #DC2626;
            font-weight: 500;
            font-size: 14px;
        }
        .optional-label {
            color: #6B7280;
            font-weight: 500;
            font-size: 14px;
        }
        .preview-text {
            color: #047857;
            font-size: 13px;
            font-style: italic;
        }
        .suggestion-text {
            color: #4B5563;
            font-size: 12px;
            font-style: italic;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### Column Mapping")
    required_mapped = sum(1 for col in REQUIRED_COLUMNS if suggested_mapping.get(col))
    required_total = len(REQUIRED_COLUMNS)
    can_proceed = required_mapped == required_total

    if can_proceed:
        st.markdown(f"<div style='color:#059669;font-weight:600;'>All required columns mapped ({required_mapped}/{required_total})</div>", unsafe_allow_html=True)
    else:
        progress = required_mapped / required_total
        st.progress(progress)
        st.markdown(f"<div style='color:#DC2626;font-weight:600;'>{required_total - required_mapped} required columns left</div>", unsafe_allow_html=True)

    with st.expander("Instructions", expanded=False):
        st.markdown("""
        - <span style='color:#DC2626;'>Required fields</span> must be mapped to proceed.
        - <span style='color:#6B7280;'>Optional fields</span> enhance analysis but aren't mandatory.
        - Suggestions may guide accurate mappings.
        """, unsafe_allow_html=True)

    all_columns = [None] + df.columns.tolist()
    display_columns = REQUIRED_COLUMNS + [c for c in STANDARD_COLUMNS if c not in REQUIRED_COLUMNS]
    confirmed_mapping = {}

    MAPPINGS_PER_ROW = 3
    rows = [display_columns[i:i + MAPPINGS_PER_ROW] for i in range(0, len(display_columns), MAPPINGS_PER_ROW)]

    for row_cols in rows:
        cols = st.columns(len(row_cols))
        for idx, std_col in enumerate(row_cols):
            with cols[idx]:
                is_required = std_col in REQUIRED_COLUMNS
                suggested = suggested_mapping.get(std_col)

                # Card container
                st.markdown(f"<div class='mapping-card'>", unsafe_allow_html=True)

                # Header and required/optional label
                label = "Required" if is_required else "Optional"
                label_class = "required-label" if is_required else "optional-label"
                st.markdown(f"""
                    <div class='mapping-header'>{std_col}</div>
                    <div class='{label_class}'>{label}</div>
                """, unsafe_allow_html=True)

                # Mapping selection
                default_index = all_columns.index(suggested) if suggested in all_columns else 0
                selected = st.selectbox(
                    f"Select column for {std_col}",  # <-- this label is hidden
                    options=all_columns,
                    index=default_index,
                    key=f"map_{year}_{std_col}",
                    label_visibility="collapsed",
                    help=f"Map '{std_col}' to a column from your file"
                )
                confirmed_mapping[std_col] = selected

                # Preview selected column value
                if selected and selected in df.columns:
                    preview_val = df[selected].iloc[0]
                    st.markdown(f"<div class='preview-text'>Preview: {preview_val}</div>", unsafe_allow_html=True)

                # Suggestion notice
                if suggested and selected != suggested and suggested in all_columns:
                    st.markdown(f"<div class='suggestion-text'>Suggestion: {suggested}</div>", unsafe_allow_html=True)
                elif selected == suggested and selected:
                    st.markdown(f"<div class='suggestion-text'>Using suggestion</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # Final validation messages
    missing_cols = get_missing_required_columns(confirmed_mapping)
    mapped_files = [v for v in confirmed_mapping.values() if v]
    duplicates = set([x for x in mapped_files if mapped_files.count(x) > 1])

    if missing_cols:
        info_box(f"Missing required columns: {', '.join(missing_cols)}", "warning")
    if duplicates:
        info_box(f"Duplicate mappings detected: {', '.join(duplicates)}", "warning")

    return confirmed_mapping