import logging
import numpy as np
import pandas as pd

# Set up logging for debugging (can be configured externally)
logger = logging.getLogger(__name__)


def compare_defects(old_defects_df, new_defects_df, old_joints_df=None, new_joints_df=None, old_year=None, new_year=None, distance_tolerance=0.1, clock_tolerance_minutes=20):
    """
    Compare defects between two pipeline inspection years to track defect evolution.

    This function matches defects from an older inspection to a newer inspection based on:
    - Physical location (distance along pipeline)
    - Clock position (circumferential position around pipe)

    For matched defects, it calculates growth rates in depth, length, and width.

    Parameters:
    -----------
    old_defects_df : pandas.DataFrame
        Defects from the earlier inspection. Must contain 'log dist. [m]' column.
    new_defects_df : pandas.DataFrame
        Defects from the newer inspection. Must contain 'log dist. [m]' column.
    old_joints_df : pandas.DataFrame, optional
        Joint data from earlier inspection. Used to get wall thickness for depth calculations.
        Should contain 'joint number' and 'wt nom [mm]' columns.
    new_joints_df : pandas.DataFrame, optional
        Joint data from newer inspection. Used to get wall thickness for depth calculations.
    old_year : int, optional
        Year of the earlier inspection. Required for growth rate calculations.
    new_year : int, optional
        Year of the newer inspection. Required for growth rate calculations.
    distance_tolerance : float, default=0.1
        Maximum distance difference (in meters) to consider defects as the same location.
    clock_tolerance_minutes : float, default=20
        Maximum clock position difference (in minutes) to consider defects at same position.
        Clock positions are on a 12-hour dial (e.g., 3 o'clock = 3.0 hours).

    Returns:
    --------
    dict
        Dictionary containing:
        - 'matches_df': DataFrame of matched defects with growth calculations
        - 'new_defects': DataFrame of defects that couldn't be matched (truly new)
        - 'common_defects_count': Number of matched defects
        - 'new_defects_count': Number of unmatched defects
        - 'total_defects': Total defects in new inspection
        - 'pct_common': Percentage of defects that were matched
        - 'pct_new': Percentage of defects that are new
        - 'defect_type_distribution': Distribution of defect types for new defects
        - 'growth_stats': Statistics about growth rates (if years provided)
        - Various flags indicating data availability

    Raises:
    -------
    ValueError
        If required columns are missing or input data is invalid
    """
    try:
        # === STEP 1: Validate inputs ===
        if old_defects_df is None or new_defects_df is None:
            raise ValueError("Both old and new defect dataframes are required")

        # Create working copies to avoid modifying originals
        old_df = old_defects_df.copy()
        new_df = new_defects_df.copy()

        logger.debug(f"Old defects columns: {old_df.columns.tolist()}")
        logger.debug(f"New defects columns: {new_df.columns.tolist()}")
        if old_joints_df is not None and new_joints_df is not None:
            logger.debug(f"Old joints columns: {old_joints_df.columns.tolist()}")
            logger.debug(f"New joints columns: {new_joints_df.columns.tolist()}")

        # === STEP 2: Check data availability for growth calculations ===
        calculate_growth = False
        year_difference = None
        if old_year is not None and new_year is not None and new_year > old_year:
            calculate_growth = True
            year_difference = new_year - old_year
            logger.info(f"Growth calculation enabled: {year_difference} years between inspections")

        has_depth_data = ("depth [%]" in old_df.columns) and ("depth [%]" in new_df.columns)
        has_length_data = ("length [mm]" in old_df.columns) and ("length [mm]" in new_df.columns)
        has_width_data = ("width [mm]" in old_df.columns) and ("width [mm]" in new_df.columns)

        # Check if we can link defects to joints
        has_joint_num = ("joint number" in old_df.columns) and ("joint number" in new_df.columns)
        if not has_joint_num:
            logger.warning("'joint number' column missing - cannot calculate mm-based depth growth")

        # Check if wall thickness data is available
        has_wt_data = False
        if (
            has_joint_num
            and old_joints_df is not None
            and new_joints_df is not None
            and "wt nom [mm]" in old_joints_df.columns
            and "wt nom [mm]" in new_joints_df.columns
        ):
            has_wt_data = True
        else:
            logger.warning("Wall thickness data unavailable - depth growth will be in % only")

        # Check clock position data availability
        has_clock_float = ("clock_float" in old_df.columns) and ("clock_float" in new_df.columns)
        clock_tolerance_hours = clock_tolerance_minutes / 60.0

        # === STEP 3: Validate required columns ===
        required_columns = ["log dist. [m]"]
        for col in required_columns:
            if col not in old_df.columns or col not in new_df.columns:
                raise ValueError(f"Required column missing: '{col}'")

        # === STEP 4: Prepare data for efficient matching ===
        old_df = old_df.sort_values("log dist. [m]").reset_index(drop=True)
        new_df = new_df.sort_values("log dist. [m]").reset_index(drop=True)

        old_df["defect_id"] = range(len(old_df))
        new_df["defect_id"] = range(len(new_df))

        # === STEP 5: Create wall thickness lookup tables ===
        old_wt_lookup = {}
        new_wt_lookup = {}
        if has_wt_data:
            try:
                old_wt_lookup = dict(zip(old_joints_df["joint number"], old_joints_df["wt nom [mm]"]))
                new_wt_lookup = dict(zip(new_joints_df["joint number"], new_joints_df["wt nom [mm]"])) 
                logger.info(
                    f"Created wall thickness lookups: {len(old_wt_lookup)} old joints, {len(new_wt_lookup)} new joints"
                )
            except Exception as e:
                logger.error(f"Failed to create wall thickness lookup: {e}")
                has_wt_data = False

        # === STEP 6: Main matching algorithm ===
        matches = []
        matched_old_indices = set()
        matched_new_indices = set()

        old_index = 0
        old_max = len(old_df)

        for new_idx, new_defect in new_df.iterrows():
            new_dist = new_defect["log dist. [m]"]
            new_clock = new_defect.get("clock_float") if has_clock_float else None

            # Slide window to first candidate within distance tolerance
            while old_index < old_max and old_df.loc[old_index, "log dist. [m]"] < new_dist - distance_tolerance:
                old_index += 1

            temp_old_index = old_index
            potential_matches = []

            while temp_old_index < old_max and old_df.loc[temp_old_index, "log dist. [m]"] <= new_dist + distance_tolerance:
                if temp_old_index not in matched_old_indices:
                    old_defect = old_df.loc[temp_old_index]
                    dist_diff = abs(old_defect["log dist. [m]"] - new_dist)

                    if dist_diff <= distance_tolerance:
                        clock_match = True
                        clock_diff = 0

                        if has_clock_float and new_clock is not None:
                            old_clock = old_defect.get("clock_float")
                            if old_clock is not None:
                                raw_diff = abs(old_clock - new_clock)
                                clock_diff = min(raw_diff, 12 - raw_diff)
                                clock_match = clock_diff <= clock_tolerance_hours

                        if clock_match:
                            potential_matches.append((temp_old_index, dist_diff, clock_diff))

                temp_old_index += 1

            if potential_matches:
                weight = 1.5
                if has_clock_float:
                    potential_matches.sort(key=lambda x: x[1] + weight * x[2])
                else:
                    potential_matches.sort(key=lambda x: x[1])

                best_old_idx, distance_diff, clock_diff = potential_matches[0]
                closest_old = old_df.loc[best_old_idx]

                match_data = _build_match_record(
                    new_defect,
                    closest_old,
                    distance_diff,
                    clock_diff,
                    has_clock_float,
                    has_joint_num,
                    calculate_growth,
                    year_difference,
                    has_depth_data,
                    has_length_data,
                    has_width_data,
                    has_wt_data,
                    old_wt_lookup,
                    new_wt_lookup,
                )

                matches.append(match_data)
                matched_old_indices.add(best_old_idx)
                matched_new_indices.add(new_idx)

        # === STEP 7: Create results DataFrames ===
        columns = _get_result_columns(
            new_defects_df,
            has_joint_num,
            has_clock_float,
            calculate_growth,
            has_depth_data,
            has_length_data,
            has_width_data,
            has_wt_data,
        )
        matches_df = pd.DataFrame(matches, columns=columns) if matches else pd.DataFrame(columns=columns)

        if calculate_growth and not matches_df.empty:
            matches_df["old_year"] = old_year
            matches_df["new_year"] = new_year

        new_defects = new_df.loc[~new_df.index.isin(matched_new_indices)].copy()

        # === STEP 8: Calculate summary statistics ===
        total = len(new_df)
        common = len(matches_df)
        new_count = len(new_defects)

        pct_common = (common / total * 100) if total > 0 else 0
        pct_new = (new_count / total * 100) if total > 0 else 0

        if new_count > 0 and "component / anomaly identification" in new_defects.columns:
            dist = (
                new_defects["component / anomaly identification"]
                .value_counts()
                .rename_axis("defect_type")
                .reset_index(name="count")
            )
            dist["percentage"] = dist["count"] / new_count * 100
        else:
            dist = pd.DataFrame(columns=["defect_type", "count", "percentage"])

        # === STEP 9: Calculate growth statistics ===
        growth_stats = None
        if calculate_growth and not matches_df.empty and any([has_depth_data, has_length_data, has_width_data]):
            growth_stats = _calculate_growth_statistics(matches_df, has_depth_data, has_length_data, has_width_data, has_wt_data)

        # === STEP 10: Return comprehensive results ===
        return {
            "matches_df": matches_df,
            "new_defects": new_defects,
            "common_defects_count": common,
            "new_defects_count": new_count,
            "total_defects": total,
            "pct_common": pct_common,
            "pct_new": pct_new,
            "defect_type_distribution": dist,
            "growth_stats": growth_stats,
            "has_depth_data": has_depth_data,
            "has_length_data": has_length_data,
            "has_width_data": has_width_data,
            "has_wt_data": has_wt_data,
            "has_joint_num": has_joint_num,
            "calculate_growth": calculate_growth,
        }

    except KeyError as ke:
        column_name = str(ke).strip("'")
        if column_name == "joint number":
            raise ValueError(
                "Missing 'joint number' column in the defects data. "
                "Please ensure both datasets have this column for wall thickness calculations."
            )
        else:
            raise ValueError(f"Missing required column: {column_name}")
    except Exception as e:
        raise ValueError(f"Error in compare_defects: {e}") from e


def _build_match_record(
    new_defect,
    old_defect,
    distance_diff,
    clock_diff,
    has_clock_float,
    has_joint_num,
    calculate_growth,
    year_difference,
    has_depth_data,
    has_length_data,
    has_width_data,
    has_wt_data,
    old_wt_lookup,
    new_wt_lookup,
):
    """
    Constructs a comprehensive match record for paired pipeline defects across inspections.

    Consolidates defect attributes, calculates dimensional changes, and computes growth rates
    for depth, length, and width dimensions. Handles unit conversions and data validation.

    Parameters
    ----------
    new_defect : pandas.Series
        Defect data from recent inspection. Must contain 'defect_id', 'log dist. [m]',
        and dimensional columns as available.
    old_defect : pandas.Series
        Corresponding defect data from earlier inspection (same structure as new_defect).
    distance_diff : float
        Absolute longitudinal distance difference between matched defects (meters).
    clock_diff : float
        Minimum circumferential clock position difference (hours) accounting for circularity.
    has_clock_float : bool
        Flag indicating availability of clock position data.
    has_joint_num : bool
        Flag indicating availability of joint number data.
    calculate_growth : bool
        Flag enabling growth rate calculations.
    year_difference : int
        Time elapsed between inspections (years) for growth rate normalization.
    has_depth_data : bool
        Flag indicating depth data availability.
    has_length_data : bool
        Flag indicating length data availability.
    has_width_data : bool
        Flag indicating width data availability.
    has_wt_data : bool
        Flag indicating wall thickness data availability.
    old_wt_lookup : dict
        Mapping of joint numbers to wall thickness values (mm) for old inspection.
    new_wt_lookup : dict
        Mapping of joint numbers to wall thickness values (mm) for new inspection.

    Returns
    -------
    dict
        Unified defect record containing:
        - Core identifiers: new_defect_id, old_defect_id, defect_type
        - Location data: log_dist, old_log_dist, distance_diff
        - Clock differences (if available): clock_diff_hours, clock_diff_minutes
        - Dimensional measurements: length [mm], width [mm] (as available)
        - Growth metrics (if calculated):
            * Depth: old/new depth (%), depth change, annual growth rate (%/year)
            * Depth (mm): old/new depth (mm), mm change, annual growth (mm/year)
            * Length/Width: old/new dimensions, changes, annual growth rates
        - Negative growth flags for each dimension

    Notes
    -----
    - Depth values are clamped to 0-100% range to handle measurement outliers
    - Wall thickness conversion uses average of old/new wall thickness when available
    - Negative growth flags indicate possible measurement error or repair
    - Missing data dimensions are silently excluded from the record
    """


    match_data = {
        "new_defect_id": new_defect["defect_id"],      
        "old_defect_id": old_defect["defect_id"],     
        "distance_diff": distance_diff,
        "log_dist": new_defect["log dist. [m]"],
        "old_log_dist": old_defect["log dist. [m]"],
        "defect_type": new_defect.get("component / anomaly identification", "Unknown"),
    }

    if has_joint_num:
        match_data["joint number"] = new_defect["joint number"]

    for dim_col in ["length [mm]", "width [mm]"]:
        if dim_col in new_defect:
            match_data[dim_col] = new_defect[dim_col]

    if has_clock_float:
        match_data["clock_diff_hours"] = clock_diff
        match_data["clock_diff_minutes"] = clock_diff * 60

    if year_difference > 0 and calculate_growth:
        # Depth growth
        if has_depth_data:
            old_depth = old_defect["depth [%]"]
            new_depth = new_defect["depth [%]"]

            if not (0 <= old_depth <= 100):
                logger.warning(f"Invalid old depth {old_depth}% for defect at {old_defect['log dist. [m]']}m")
                old_depth = max(0, min(100, old_depth))  # Clamp to valid range
                
            if not (0 <= new_depth <= 100):
                logger.warning(f"Invalid new depth {new_depth}% for defect at {new_defect['log dist. [m]']}m")
                new_depth = max(0, min(100, new_depth))  # Clamp to valid range
            
            depth_change = new_depth - old_depth

            match_data.update(
                {
                    "old_depth_pct": old_depth,
                    "new_depth_pct": new_depth,
                    "depth_change_pct": depth_change,
                    "growth_rate_pct_per_year": depth_change / year_difference,
                    "is_negative_growth": depth_change < 0,
                }
            )

            if has_wt_data and has_joint_num:
                try:
                    old_joint = old_defect["joint number"]
                    new_joint = new_defect["joint number"]
                    old_wt = old_wt_lookup.get(old_joint)
                    new_wt = new_wt_lookup.get(new_joint)
                    if old_wt is not None and new_wt is not None:
                        old_depth_mm = old_depth * old_wt / 100
                        new_depth_mm = new_depth * new_wt / 100

                        match_data.update(
                            {
                                "old_wt_mm": old_wt,
                                "new_wt_mm": new_wt,
                                "old_depth_mm": old_depth_mm,
                                "new_depth_mm": new_depth_mm,
                                "depth_change_mm": new_depth_mm - old_depth_mm,
                                "growth_rate_mm_per_year": (new_depth_mm - old_depth_mm) / year_difference,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Could not calculate mm-based depth growth for joint {new_joint}: {e}")

    else:
        logger.warning(f"Invalid year difference: {year_difference}. Growth rates will not be calculated.")
        calculate_growth = False

    # Length growth
    if has_length_data:
        old_length = old_defect["length [mm]"]
        new_length = new_defect["length [mm]"]
        length_change = new_length - old_length

        match_data.update(
            {
                "old_length_mm": old_length,
                "new_length_mm": new_length,
                "length_change_mm": length_change,
                "length_growth_rate_mm_per_year": length_change / year_difference,
                "is_negative_length_growth": length_change < 0,
            }
        )

    # Width growth
    if has_width_data:
        old_width = old_defect["width [mm]"]
        new_width = new_defect["width [mm]"]
        width_change = new_width - old_width

        match_data.update(
            {
                "old_width_mm": old_width,
                "new_width_mm": new_width,
                "width_change_mm": width_change,
                "width_growth_rate_mm_per_year": width_change / year_difference,
                "is_negative_width_growth": width_change < 0,
            }
        )

    return match_data


def _get_result_columns(
    new_defects_df,
    has_joint_num,
    has_clock_float,
    calculate_growth,
    has_depth_data,
    has_length_data,
    has_width_data,
    has_wt_data,
):
    """
    Dynamically generates output columns for matched defects based on data availability.

    Constructs the optimal result schema by inspecting dataset capabilities and
    calculation requirements. Ensures only relevant columns are included.

    Parameters
    ----------
    new_defects_df : pandas.DataFrame
        New inspection defects data used to validate column existence.
    has_joint_num : bool
        Indicates if joint number data is available.
    has_clock_float : bool
        Indicates if clock position data is available.
    calculate_growth : bool
        Flag determining if growth columns should be included.
    has_depth_data : bool
        Indicates depth measurement availability.
    has_length_data : bool
        Indicates length measurement availability.
    has_width_data : bool
        Indicates width measurement availability.
    has_wt_data : bool
        Indicates wall thickness data availability.

    Returns
    -------
    list
        Ordered list of column names for the result DataFrame, containing:
        - Mandatory columns: defect IDs, location data, defect type
        - Conditional columns: joint number, clock differences
        - Growth columns (if enabled and data available):
            * Depth: percentage and mm-based metrics
            * Length: mm-based metrics
            * Width: mm-based metrics
        - Negative growth indicators for each dimension

    Notes
    -----
    - Column order: Core metadata → Location → Dimensions → Growth metrics
    - Growth columns are only added if:
        1. calculate_growth = True
        2. Corresponding dimension data exists
        3. For mm-depth: has_wt_data = True
    - Maintains consistent column naming across analyses
    """

    columns = [
        "new_defect_id",
        "old_defect_id",
        "distance_diff",
        "log_dist",
        "old_log_dist",
        "defect_type",
    ]

    for dim_col in ["length [mm]", "width [mm]"]:
        if dim_col in new_defects_df.columns:
            columns.append(dim_col)

    if has_joint_num:
        columns.append("joint number")

    if has_clock_float:
        columns.extend(["clock_diff_hours", "clock_diff_minutes"])

    if calculate_growth:
        if has_depth_data:
            columns.extend(
                [
                    "old_depth_pct",
                    "new_depth_pct",
                    "depth_change_pct",
                    "growth_rate_pct_per_year",
                    "is_negative_growth",
                ]
            )
            if has_wt_data:
                columns.extend(
                    [
                        "old_wt_mm",
                        "new_wt_mm",
                        "old_depth_mm",
                        "new_depth_mm",
                        "depth_change_mm",
                        "growth_rate_mm_per_year",
                    ]
                )

        if has_length_data:
            columns.extend(
                [
                    "old_length_mm",
                    "new_length_mm",
                    "length_change_mm",
                    "length_growth_rate_mm_per_year",
                    "is_negative_length_growth",
                ]
            )

        if has_width_data:
            columns.extend(
                [
                    "old_width_mm",
                    "new_width_mm",
                    "width_change_mm",
                    "width_growth_rate_mm_per_year",
                    "is_negative_width_growth",
                ]
            )
    return columns


def _calculate_growth_statistics(matches_df, has_depth_data, has_length_data, has_width_data, has_wt_data):
    """
    Computes aggregate growth statistics from matched pipeline defects.

    Analyzes dimensional changes across inspections to provide engineering insights
    about corrosion progression. Separates positive and negative growth for
    anomaly detection.

    Parameters
    ----------
    matches_df : pandas.DataFrame
        Matched defects from compare_defects() containing growth metrics.
    has_depth_data : bool
        Indicates depth growth metrics are available.
    has_length_data : bool
        Indicates length growth metrics are available.
    has_width_data : bool
        Indicates width growth metrics are available.
    has_wt_data : bool
        Indicates mm-based depth metrics are available.

    Returns
    -------
    dict
        Comprehensive growth statistics including:
        - Total matched defect count
        - For each dimension (depth/length/width):
            * Negative growth count and percentage
            * Average growth rate (all defects)
            * Average positive growth rate
            * Maximum positive growth rate
        - Depth-specific metrics in both % and mm (if has_wt_data)

    Notes
    -----
    - Negative growth indicates possible measurement error or repair
    - Positive growth rates exclude negative/zero values
    - Depth statistics include both % and mm-based metrics when available
    - Handles missing data gracefully by excluding unavailable dimensions
    - Returns empty statistics if matches_df contains no growth columns
    """
    
    growth_stats = {"total_matched_defects": len(matches_df)}

    # Depth statistics
    if has_depth_data:
        neg_depth_count = matches_df["is_negative_growth"].sum()
        pct_neg_depth = (neg_depth_count / len(matches_df)) * 100
        pos_depth = matches_df[~matches_df["is_negative_growth"]]

        growth_stats.update(
            {
                "depth_negative_growth_count": int(neg_depth_count),
                "depth_pct_negative_growth": pct_neg_depth,
                "avg_growth_rate_pct": matches_df["growth_rate_pct_per_year"].mean(),
                "avg_positive_growth_rate_pct": pos_depth["growth_rate_pct_per_year"].mean()
                if not pos_depth.empty
                else 0,
                "max_growth_rate_pct": pos_depth["growth_rate_pct_per_year"].max()
                if not pos_depth.empty
                else 0,
            }
        )

        if has_wt_data and "growth_rate_mm_per_year" in matches_df.columns:
            mm_data = matches_df.dropna(subset=["growth_rate_mm_per_year"])
            pos_mm = mm_data[mm_data["growth_rate_mm_per_year"] > 0]

            growth_stats.update(
                {
                    "avg_growth_rate_mm": mm_data["growth_rate_mm_per_year"].mean()
                    if not mm_data.empty
                    else np.nan,
                    "avg_positive_growth_rate_mm": pos_mm["growth_rate_mm_per_year"].mean()
                    if not pos_mm.empty
                    else 0,
                    "max_growth_rate_mm": pos_mm["growth_rate_mm_per_year"].max()
                    if not pos_mm.empty
                    else 0,
                } 
            ) 

    # Length statistics
    if has_length_data:
        neg_len_count = matches_df["is_negative_length_growth"].sum()
        pct_neg_len = (neg_len_count / len(matches_df)) * 100
        pos_len = matches_df[~matches_df["is_negative_length_growth"]]

        growth_stats.update(
            {
                "length_negative_growth_count": int(neg_len_count),
                "length_pct_negative_growth": pct_neg_len,
                "avg_length_growth_rate_mm": matches_df["length_growth_rate_mm_per_year"].mean(),
                "avg_positive_length_growth_rate_mm": pos_len["length_growth_rate_mm_per_year"].mean()
                if not pos_len.empty
                else 0,
                "max_length_growth_rate_mm": pos_len["length_growth_rate_mm_per_year"].max()
                if not pos_len.empty
                else 0,
            }
        )

    # Width statistics
    if has_width_data:
        neg_wid_count = matches_df["is_negative_width_growth"].sum()
        pct_neg_wid = (neg_wid_count / len(matches_df)) * 100
        pos_wid = matches_df[~matches_df["is_negative_width_growth"]]

        growth_stats.update(
            {
                "width_negative_growth_count": int(neg_wid_count),
                "width_pct_negative_growth": pct_neg_wid,
                "avg_width_growth_rate_mm": matches_df["width_growth_rate_mm_per_year"].mean(),
                "avg_positive_width_growth_rate_mm": pos_wid["width_growth_rate_mm_per_year"].mean()
                if not pos_wid.empty
                else 0,
                "max_width_growth_rate_mm": pos_wid["width_growth_rate_mm_per_year"].max()
                if not pos_wid.empty
                else 0,
            }
        )

    return growth_stats