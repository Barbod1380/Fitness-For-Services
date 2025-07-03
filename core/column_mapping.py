# column_mapping.py
import pandas as pd
from fuzzywuzzy import process
import re

# ------------------------------------------------------------------------
# Canonical column names the rest of the pipeline will rely on
STANDARD_COLUMNS = [
    "log dist. [m]",
    "component / anomaly identification",
    "joint number",
    "joint length [m]",
    "wt nom [mm]",
    "up weld dist. [m]",
    "clock",
    "depth [%]",
    "ERF B31G",
    "length [mm]",
    "width [mm]",
    "surface location",
]

# ------------------------------------------------------------------------
# Common alternative names for each standard column
COLUMN_VARIANTS = {
    "log dist. [m]": ["log distance", "distance", "chainage", "position", "log_dist", "log dist"],
    "component / anomaly identification": [
        "event",
        "anomaly",
        "defect type",
        "feature",
        "feature type",
        "anomaly type",
        "defect_type"
    ],
    "joint number": ["J. no.", "J.no.", "joint #", "Joint No", "joint_number", "joint no", "joint_no"],
    "joint length [m]": [
        "J. len [m]",
        "joint length",
        "length [m]",
        "Joint Length",
        "pipe length",
        "joint_length"
    ],
    "wt nom [mm]": ["t [mm]", "thickness", "wall thickness", "nominal thickness", "WT", "wall_thickness", "wt_nom"],
    "up weld dist. [m]": [
        "to u/s w. [m]",
        "upstream weld distance",
        "distance to upstream weld",
        "up weld",
        "up_weld_dist",
        "upstream_dist"
    ],
    "clock": ["o'clock", "oclock", "clock position", "angular position", "clock_position"],
    "depth [%]": ["depth percent", "depth percentage", "defect depth", "depth_%", "depth_pct"],
    "ERF B31G": ["ERF_AS2885", "ERF", "B31G", "expansion_ratio_factor", "erf_b31g"],
    "length [mm]": ["defect length", "anomaly length", "feature length", "length_mm", "defect_length"],
    "width [mm]": ["defect width", "anomaly width", "feature width", "width_mm", "defect_width"],
    "surface location": ["internal", "external", "location", "orientation", "surface_loc", "surface"],
}

# ------------------------------------------------------------------------
# Known direct mappings from alternate names to standard names
KNOWN_MAPPINGS = {
    "event": "component / anomaly identification",
    "J. no.": "joint number",
    "J. len [m]": "joint length [m]",
    "t [mm]": "wt nom [mm]",
    "to u/s w. [m]": "up weld dist. [m]",
    "o'clock": "clock",
    "ERF_AS2885": "ERF B31G",
    "internal": "surface location",
}

# ------------------------------------------------------------------------
# Columns that must be present (after mapping) for downstream processing
REQUIRED_COLUMNS = [
    "log dist. [m]",
    "joint number",
    "joint length [m]",
    "wt nom [mm]",
    "component / anomaly identification",
    "up weld dist. [m]",
    "clock",
    "depth [%]",
    "length [mm]",
    "width [mm]",
    "surface location",
]

def clean_column_name(col_name):
    """Clean column name for better matching"""
    # Convert to lowercase
    cleaned = col_name.lower().strip()
    # Replace common separators with spaces
    cleaned = re.sub(r'[_\-\.]', ' ', cleaned)
    # Remove extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

def suggest_column_mapping(df):
    """
    Suggest a mapping from each STANDARD_COLUMN to the best candidate
    in df.columns, using:
      1. Exact match
      2. KNOWN_MAPPINGS
      3. COLUMN_VARIANTS
      4. Fuzzy matching (score > 70)
    
    Enhanced to handle files with many columns better.
    
    Returns:
        mapping: dict where keys are STANDARD_COLUMNS and values are
                 the matched column from df.columns, or None if no match.
    """
    file_columns = list(df.columns)
    mapping = {}
    used_columns = set()  # Track columns already mapped to avoid duplicates

    for std_col in STANDARD_COLUMNS:
        # 1. Exact match
        if std_col in file_columns and std_col not in used_columns:
            mapping[std_col] = std_col
            used_columns.add(std_col)
            continue

        # 2. Check KNOWN_MAPPINGS
        mapped = False
        for alt_name, target in KNOWN_MAPPINGS.items():
            if target == std_col:
                # Check both exact and case-insensitive match
                for file_col in file_columns:
                    if file_col not in used_columns and (
                        file_col == alt_name or 
                        file_col.lower() == alt_name.lower()
                    ):
                        mapping[std_col] = file_col
                        used_columns.add(file_col)
                        mapped = True
                        break
                if mapped:
                    break

        # 3. Check COLUMN_VARIANTS
        if std_col not in mapping and std_col in COLUMN_VARIANTS:
            for variant in COLUMN_VARIANTS[std_col]:
                for file_col in file_columns:
                    if file_col not in used_columns:
                        # Check exact, case-insensitive, and cleaned matches
                        if (file_col == variant or 
                            file_col.lower() == variant.lower() or
                            clean_column_name(file_col) == clean_column_name(variant)):
                            mapping[std_col] = file_col
                            used_columns.add(file_col)
                            mapped = True
                            break
                if mapped:
                    break

        # 4. Fuzzy matching fallback
        if std_col not in mapping:
            # Filter out already used columns
            available_columns = [col for col in file_columns if col not in used_columns]
            
            if available_columns:
                # Try fuzzy matching on cleaned column names
                cleaned_std = clean_column_name(std_col)
                cleaned_available = [(clean_column_name(col), col) for col in available_columns]
                
                # First try with cleaned names
                match, score = process.extractOne(cleaned_std, [c[0] for c in cleaned_available])
                
                if score and score > 80:  # Higher threshold for many columns
                    # Find the original column name
                    for cleaned, original in cleaned_available:
                        if cleaned == match:
                            mapping[std_col] = original
                            used_columns.add(original)
                            break
                else:
                    # Try with original names if cleaned didn't work
                    match, score = process.extractOne(std_col, available_columns)
                    if score and score > 85:  # Even higher threshold
                        mapping[std_col] = match
                        used_columns.add(match)
                    else:
                        mapping[std_col] = None
            else:
                mapping[std_col] = None

    return mapping


def apply_column_mapping(df, mapping):
    """
    Rename or add columns to match STANDARD_COLUMNS using mapping.
    For each std_col in mapping:
      - if mapping[std_col] is not None and exists in df, create
        a new column named std_col = df[mapping[std_col]].
    Returns a new DataFrame with the renamed columns added.
    """
    # Start with only the mapped columns to avoid extra columns
    renamed_df = pd.DataFrame()
    
    # First, add all mapped standard columns
    for std_col, file_col in mapping.items():
        if file_col is not None and file_col in df.columns:
            renamed_df[std_col] = df[file_col]
    
    # Add any columns that were directly mapped (same name)
    for col in df.columns:
        if col in STANDARD_COLUMNS and col not in renamed_df.columns:
            renamed_df[col] = df[col]
    
    # Preserve the original index
    renamed_df.index = df.index
    
    return renamed_df


def get_missing_required_columns(mapping):
    """
    Identify which REQUIRED_COLUMNS are not mapped (i.e., mapping[col] is None).
    Returns:
        missing: list of required columns missing in mapping
    """
    missing = []
    for col in REQUIRED_COLUMNS:
        if col not in mapping or mapping[col] is None:
            missing.append(col)
    return missing