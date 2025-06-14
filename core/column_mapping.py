# column_mapping.py

from fuzzywuzzy import process

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
    "log dist. [m]": ["log distance", "distance", "chainage", "position"],
    "component / anomaly identification": [
        "event",
        "anomaly",
        "defect type",
        "feature",
        "feature type",
    ],
    "joint number": ["J. no.", "J.no.", "joint #", "Joint No", "joint_number"],
    "joint length [m]": [
        "J. len [m]",
        "joint length",
        "length [m]",
        "Joint Length",
        "pipe length",
    ],
    "wt nom [mm]": ["t [mm]", "thickness", "wall thickness", "nominal thickness", "WT"],
    "up weld dist. [m]": [
        "to u/s w. [m]",
        "upstream weld distance",
        "distance to upstream weld",
        "up weld",
    ],
    "clock": ["o'clock", "oclock", "clock position", "angular position"],
    "depth [%]": ["depth percent", "depth percentage", "defect depth"],
    "ERF B31G": ["ERF_AS2885", "ERF", "B31G", "expansion_ratio_factor"],
    "length [mm]": ["defect length", "anomaly length", "feature length"],
    "width [mm]": ["defect width", "anomaly width", "feature width"],
    "surface location": ["internal", "external", "location", "orientation"],
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


def suggest_column_mapping(df):
    """
    Suggest a mapping from each STANDARD_COLUMN to the best candidate
    in df.columns, using:
      1. Exact match
      2. KNOWN_MAPPINGS
      3. COLUMN_VARIANTS
      4. Fuzzy matching (score > 70)
    Returns:
        mapping: dict where keys are STANDARD_COLUMNS and values are
                 the matched column from df.columns, or None if no match.
    """
    file_columns = list(df.columns)
    mapping = {}

    for std_col in STANDARD_COLUMNS:
        # 1. Exact match
        if std_col in file_columns:
            mapping[std_col] = std_col
            continue

        # 2. Check KNOWN_MAPPINGS
        for alt_name, target in KNOWN_MAPPINGS.items():
            if target == std_col and alt_name in file_columns:
                mapping[std_col] = alt_name
                break

        # 3. Check COLUMN_VARIANTS
        if std_col not in mapping and std_col in COLUMN_VARIANTS:
            for variant in COLUMN_VARIANTS[std_col]:
                if variant in file_columns:
                    mapping[std_col] = variant
                    break

        # 4. Fuzzy matching fallback
        if std_col not in mapping:
            match, score = process.extractOne(std_col, file_columns)
            if score and score > 70:
                mapping[std_col] = match
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
    renamed_df = df.copy()

    for std_col, file_col in mapping.items():
        if file_col is not None and file_col in df.columns:
            renamed_df[std_col] = df[file_col]

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