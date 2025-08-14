import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def correct_negative_growth_rates(matches_df, k=3, joint_tolerance=20):
    """
    Optimized correction of negative growth rates using a batched K-Nearest Neighbors approach.
    This avoids refitting the KNN model for every single defect.
    """
    df = matches_df.copy()

    # --- 1. Initial Setup and Validation ---
    required_cols = ['joint number', 'old_depth_pct', 'new_depth_pct', 'is_negative_growth']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return df, {"error": f"Missing required columns: {missing}", "success": False}

    dimension_cols = ['length [mm]', 'width [mm]']
    available_dimension_cols = [col for col in dimension_cols if col in df.columns]
    if not available_dimension_cols:
        return df, {"error": "Missing at least one dimension column for features", "success": False}

    features = available_dimension_cols + ['old_depth_pct']
    has_mm_data = 'growth_rate_mm_per_year' in df.columns

    negative_growth = df[df['is_negative_growth']].copy()
    positive_growth = df[~df['is_negative_growth']].copy()

    if negative_growth.empty or positive_growth.empty:
        return df, {"error": "No negative or positive depth growth defects found", "success": False}

    df['is_corrected'] = False
    corrections = {}
    uncorrected_count = 0
    uncorrected_joints = set()

    # --- 2. Group Negative Defects by Joint Proximity ---
    # This allows batching KNN for defects that share the same candidate pool.
    # We group by a 'zone' defined by the joint tolerance.
    zone_size = joint_tolerance * 2
    negative_growth['zone'] = (negative_growth['joint number'] // zone_size)

    for zone, group in negative_growth.groupby('zone'):
        # --- 3. Define a single, larger candidate pool for the entire group ---
        min_joint = group['joint number'].min() - joint_tolerance
        max_joint = group['joint number'].max() + joint_tolerance

        nearby_positive = positive_growth[
            (positive_growth['joint number'] >= min_joint) &
            (positive_growth['joint number'] <= max_joint)
        ]

        if len(nearby_positive) < k:
            uncorrected_count += len(group)
            uncorrected_joints.update(group['joint number'].unique())
            continue

        # --- 4. Fit KNN Model ONCE for the group ---
        scaler = StandardScaler()
        X_positive_scaled = scaler.fit_transform(nearby_positive[features])

        k_value = min(k, len(nearby_positive))
        knn = NearestNeighbors(n_neighbors=k_value)
        knn.fit(X_positive_scaled)

        # --- 5. Batch Query for all defects in the group ---
        X_negative_scaled = scaler.transform(group[features])
        all_distances, all_indices = knn.kneighbors(X_negative_scaled)

        # --- 6. Process Results for each defect in the group ---
        for i, (neg_idx, neg_defect) in enumerate(group.iterrows()):
            neighbor_indices_in_nearby = all_indices[i]
            similar_defect_indices = nearby_positive.index[neighbor_indices_in_nearby]
            
            # Calculate weights based on individual proximity
            weights = []
            for similar_idx in similar_defect_indices:
                similar_defect = nearby_positive.loc[similar_idx]
                joint_dist = abs(similar_defect['joint number'] - neg_defect['joint number'])
                weight_result = calculate_engineering_weights(neg_defect, similar_defect, joint_dist)
                weights.append(weight_result['combined_weight'])
            
            weights = np.array(weights)
            if weights.sum() > 0:
                weights /= weights.sum()
            else:
                weights = np.ones(len(weights)) / len(weights)

            # Calculate weighted average growth rate
            growth_rates_pct = nearby_positive.loc[similar_defect_indices, 'growth_rate_pct_per_year'].values
            avg_growth_pct = np.average(growth_rates_pct, weights=weights)
            
            year_diff = neg_defect['new_year'] - neg_defect['old_year']

            # Store correction data
            corrections[neg_idx] = {
                'growth_rate_pct_per_year': float(avg_growth_pct),
                'depth_change_pct': float(avg_growth_pct * year_diff),
                'new_depth_pct': float(neg_defect['old_depth_pct'] + (avg_growth_pct * year_diff)),
                'is_negative_growth': False,
                'is_corrected': True
            }

            if has_mm_data:
                growth_rates_mm = nearby_positive.loc[similar_defect_indices, 'growth_rate_mm_per_year'].values
                avg_growth_mm = np.average(growth_rates_mm, weights=weights)
                corrections[neg_idx].update({
                    'growth_rate_mm_per_year': float(avg_growth_mm),
                    'depth_change_mm': float(avg_growth_mm * year_diff),
                    'new_depth_mm': float(neg_defect['old_depth_mm'] + (avg_growth_mm * year_diff))
                })

    # --- 7. Apply Corrections and Finalize ---
    for idx, correction in corrections.items():
        for col, value in correction.items():
            df.loc[idx, col] = value

    corrected_count = len(corrections)
    total_negative = len(negative_growth)

    # Compute updated growth statistics if corrections were made
    updated_growth_stats = None
    if corrected_count > 0:
        # Recalculate post-correction stats
        final_positive_growth = df[~df['is_negative_growth']]
        updated_growth_stats = {
            'total_matched_defects': len(df),
            'negative_growth_count': df['is_negative_growth'].sum(),
            'pct_negative_growth': (df['is_negative_growth'].sum() / len(df) * 100) if len(df) > 0 else 0,
            'avg_growth_rate_pct': df['growth_rate_pct_per_year'].mean(),
            'avg_positive_growth_rate_pct': final_positive_growth['growth_rate_pct_per_year'].mean(),
            'max_growth_rate_pct': final_positive_growth['growth_rate_pct_per_year'].max(),
        }
        if has_mm_data:
            updated_growth_stats.update({
                'avg_growth_rate_mm': df['growth_rate_mm_per_year'].mean(),
                'avg_positive_growth_rate_mm': final_positive_growth['growth_rate_mm_per_year'].mean(),
                'max_growth_rate_mm': final_positive_growth['growth_rate_mm_per_year'].max(),
            })

    correction_info = {
        'total_negative': total_negative,
        'corrected_count': corrected_count,
        'uncorrected_count': uncorrected_count,
        'uncorrected_joints': list(uncorrected_joints),
        'joint_tolerance_used': joint_tolerance,
        'success': corrected_count > 0,
    }
    if updated_growth_stats:
        correction_info['updated_growth_stats'] = updated_growth_stats

    return df, correction_info



def calculate_engineering_weights(target_defect, similar_defect, joint_distance):
    """
    Calculate similarity weights based on pipeline integrity engineering principles.    

    References:
    - API 579-1 Part 4: Defect proximity and interaction methodology 
    - NACE MR0175: Environment-dependent corrosion mechanisms 
    - Pipeline integrity engineering principles (remove incorrect NACE SP0169 reference)
    - General measurement uncertainty principles (remove specific ASME B31.8S reference)
    
    Parameters:
    - target_defect: Dict with defect properties needing correction
    - similar_defect: Dict with reference defect properties
    - joint_distance: Number of joints between defects
    
    Returns:
    - Dict with individual and combined weights
    """
    

    # 1. JOINT PROXIMITY WEIGHT
    # Characteristic correlation length, 2 joints for pipeline environments.
    # Defects in nearby pipe joints share similar corrosion environments
    characteristic_joints = 2.0
    joint_weight = math.exp(-abs(joint_distance) / characteristic_joints)
    

    # 2. DEPTH SIMILARITY WEIGHT (Most critical per NACE MR0175)
    # Depth difference indicates different corrosion mechanisms
    # Defects in nearby pipe joints share similar corrosion environments
    depth_diff = abs(similar_defect['old_depth_pct'] - target_defect['old_depth_pct'])
    characteristic_depth_diff = 8.0  # 8% depth difference as characteristic
    depth_weight = math.exp(-depth_diff / characteristic_depth_diff)
    

    # 3. SIZE SIMILARITY WEIGHT (Measurement confidence per ASME B31.8S) 
    # Ratio approach - similar sizes have similar measurement uncertainty
    # Similar-sized defects have similar measurement confidence
    length_target = target_defect.get('length [mm]')
    length_similar = similar_defect.get('length [mm]')
    
    # Avoid division by zero
    if length_target > 0 and length_similar > 0:
        size_ratio = min(length_target, length_similar) / max(length_target, length_similar)
        size_weight = size_ratio ** 0.5  # Square root for diminishing returns
    else:
        size_weight = 0.1  # Low weight for invalid sizes
    

    # 4. ENVIRONMENTAL SIMILARITY (Wall thickness indicates vintage/environment) 
    # Similar wall thickness suggests similar installation vintage and environment
    # Similar wall thickness suggests similar vintage/installation conditions
    wt_target = target_defect.get('new_wt_mm', target_defect.get('old_wt_mm'))
    wt_similar = similar_defect.get('new_wt_mm', similar_defect.get('old_wt_mm'))
    
    if wt_target > 0 and wt_similar > 0:
        wt_ratio = min(wt_target, wt_similar) / max(wt_target, wt_similar)
        environment_weight = wt_ratio
    else:
        environment_weight = 1.0
    

    # COMBINED WEIGHT - Engineering-based priorities:
    # Depth similarity is most critical (40%) - indicates corrosion mechanism
    # Joint proximity is very important (30%) - shared environment  
    # Size similarity is moderate (20%) - measurement confidence
    # Environment similarity is supporting (10%) - vintage/material
    combined_weight = (
        0.40 * depth_weight +       # Highest: corrosion mechanism similarity
        0.30 * joint_weight +       # High: environmental similarity
        0.20 * size_weight +        # Moderate: measurement confidence
        0.10 * environment_weight   # Supporting: material/vintage similarity
    )
    
    return {
        'joint_weight': joint_weight,
        'depth_weight': depth_weight,
        'size_weight': size_weight,
        'environment_weight': environment_weight,
        'combined_weight': combined_weight,
        'weighting_basis': 'NACE_SP0169_API579_engineering_principles'
    }