import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def correct_negative_growth_rates(matches_df, k=3, joint_tolerance=20):
    """
    Correct negative depth growth rates using K-Nearest Neighbors from similar defects 
    in the same joint AND nearby joints (±2 joints).

    Parameters:
    - matches_df: DataFrame with matched defects from two inspections
    - k: Number of neighbors to consider (default: 3)
    - joint_tolerance: Number of joints before/after to include in search (default: 2)

    Returns:
    - Tuple of (corrected_df, correction_info) where correction_info contains statistics
    """
    
    df = matches_df.copy()

    required_cols = ['joint number', 'old_depth_pct', 'new_depth_pct', 'is_negative_growth']
    dimension_cols = ['length [mm]', 'width [mm]']

    # Check for required columns
    for col in required_cols:
        if col not in df.columns:
            return df, {"error": f"Missing required column: {col}", "success": False}

    available_dimension_cols = [col for col in dimension_cols if col in df.columns]
    if not available_dimension_cols:
        return df, {
            "error": "Missing at least one dimension column (length [mm] or width [mm]) for features",
            "success": False
        }

    has_mm_data = 'growth_rate_mm_per_year' in df.columns

    # Separate negative and positive growth defects
    negative_growth = df[df['is_negative_growth']].copy()
    positive_growth = df[~df['is_negative_growth']].copy()

    if negative_growth.empty or positive_growth.empty:
        return df, {
            "error": "No negative or positive depth growth defects found",
            "success": False
        }

    df['is_corrected'] = False

    total_negative = len(negative_growth)
    corrected_count = 0
    uncorrected_count = 0
    uncorrected_joints = set()
    corrections = {}

    try:
        # NEW: Process each negative growth defect individually with nearby joint search
        for neg_idx, neg_defect in negative_growth.iterrows():
            target_joint = neg_defect['joint number']
            
            # Define joint search range
            joint_min = target_joint - joint_tolerance
            joint_max = target_joint + joint_tolerance
            
            # Find positive growth defects in nearby joints
            nearby_positive = positive_growth[
                (positive_growth['joint number'] >= joint_min) & 
                (positive_growth['joint number'] <= joint_max)
            ]
            
            if len(nearby_positive) < k:
                uncorrected_count += 1
                uncorrected_joints.add(target_joint)
                continue

            # Features for KNN: dimensions plus old depth
            features = available_dimension_cols + ['old_depth_pct']

            # Prepare feature data - FIX: Handle Series correctly
            scaler = StandardScaler()

            # Ensure we have DataFrame with proper column names for fitting
            nearby_positive_features = nearby_positive[features].copy()

            # Fit scaler with DataFrame (preserves feature names)
            X_positive = scaler.fit_transform(nearby_positive_features)

            neg_features_dict = {feature: neg_defect[feature] for feature in features}
            neg_features_df = pd.DataFrame([neg_features_dict], columns=features)
            X_negative = scaler.transform(neg_features_df)


            # Find k nearest neighbors
            k_value = min(k, len(nearby_positive))
            knn = NearestNeighbors(n_neighbors=k_value)
            knn.fit(X_positive)

            distances, indices = knn.kneighbors(X_negative)
            
            # Get the actual indices of similar defects
            similar_defect_indices = nearby_positive.index[indices[0]]
            
            # ENHANCEMENT: Weight by joint distance (closer joints get higher weight)
            weights = []
            for i, similar_idx in enumerate(similar_defect_indices):
                similar_defect = nearby_positive.loc[similar_idx]
                joint_distance = abs(similar_defect['joint number'] - target_joint)
                
                # Use engineering-based weight calculation
                weight_result = calculate_engineering_weights(neg_defect, similar_defect, joint_distance)
                weights.append(weight_result['combined_weight'])
            
            # Normalize weights
            weights = np.array(weights)
            if weights.sum() > 0:  # Avoid division by zero
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(weights)) / len(weights)  # Equal weights fallback
            
            # Calculate weighted average growth rate
            growth_rates = nearby_positive.loc[similar_defect_indices, 'growth_rate_pct_per_year'].values
            
            avg_growth_pct = np.average(growth_rates, weights=weights)
            
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

            corrected_count += 1

    except Exception as e:
        return df, {
            "error": f"Error during correction: {str(e)}",
            "success": False
        }

    # Apply all computed corrections to the DataFrame
    for idx, correction in corrections.items():
        for col, value in correction.items():
            df.loc[idx, col] = value

    # Compute updated growth statistics if corrections were made
    updated_growth_stats = None
    if corrected_count > 0:
        negative_growth_count = df['is_negative_growth'].sum()
        pct_negative_growth = (
            (negative_growth_count / len(df)) * 100 if len(df) > 0 else 0
        )
        positive_growth = df[~df['is_negative_growth']]

        updated_growth_stats = {
            'total_matched_defects': len(df),
            'negative_growth_count': negative_growth_count,
            'pct_negative_growth': pct_negative_growth,
            'avg_growth_rate_pct': df['growth_rate_pct_per_year'].mean(),
            'avg_positive_growth_rate_pct': (
                positive_growth['growth_rate_pct_per_year'].mean()
                if not positive_growth.empty else 0
            ),
            'max_growth_rate_pct': (
                positive_growth['growth_rate_pct_per_year'].max()
                if not positive_growth.empty else 0
            )
        }

        if has_mm_data:
            updated_growth_stats.update({
                'avg_growth_rate_mm': df['growth_rate_mm_per_year'].mean(),
                'avg_positive_growth_rate_mm': (
                    positive_growth['growth_rate_mm_per_year'].mean()
                    if not positive_growth.empty else 0
                ),
                'max_growth_rate_mm': (
                    positive_growth['growth_rate_mm_per_year'].max()
                    if not positive_growth.empty else 0
                )
            })

    correction_info = {
        'total_negative': total_negative,
        'corrected_count': corrected_count,
        'uncorrected_count': uncorrected_count,
        'uncorrected_joints': list(uncorrected_joints),
        'joint_tolerance_used': joint_tolerance,
        'success': corrected_count > 0
    }

    if updated_growth_stats is not None:
        correction_info['updated_growth_stats'] = updated_growth_stats

    return df, correction_info



def calculate_engineering_weights(target_defect, similar_defect, joint_distance):
    """
    Calculate similarity weights based on pipeline integrity engineering principles.
    
    References:
    - NACE SP0169-2013: Local corrosion environment correlation
    - API 579-1 Part 4: Proximity effects in pipeline defects  
    - NACE MR0175: Environment-dependent corrosion mechanisms
    - ASME B31.8S: ILI measurement uncertainty principles
    
    Parameters:
    - target_defect: Dict with defect properties needing correction
    - similar_defect: Dict with reference defect properties
    - joint_distance: Number of joints between defects
    
    Returns:
    - Dict with individual and combined weights
    """
    
    # 1. JOINT PROXIMITY WEIGHT (Exponential decay per NACE SP0169)
    # Characteristic correlation length: 2 joints for pipeline environments
    characteristic_joints = 2.0
    joint_weight = math.exp(-abs(joint_distance) / characteristic_joints)
    
    # 2. DEPTH SIMILARITY WEIGHT (Most critical per NACE MR0175)
    # Depth difference indicates different corrosion mechanisms
    depth_diff = abs(similar_defect['old_depth_pct'] - target_defect['old_depth_pct'])
    characteristic_depth_diff = 8.0  # 8% depth difference as characteristic
    depth_weight = math.exp(-depth_diff / characteristic_depth_diff)
    
    # 3. SIZE SIMILARITY WEIGHT (Measurement confidence per ASME B31.8S)
    # Ratio approach - similar sizes have similar measurement uncertainty
    length_target = target_defect.get('length [mm]', 50.0)
    length_similar = similar_defect.get('length [mm]', 50.0)
    
    # Avoid division by zero
    if length_target > 0 and length_similar > 0:
        size_ratio = min(length_target, length_similar) / max(length_target, length_similar)
        size_weight = size_ratio ** 0.5  # Square root for diminishing returns
    else:
        size_weight = 0.1  # Low weight for invalid sizes
    
    # 4. ENVIRONMENTAL SIMILARITY (Wall thickness indicates vintage/environment)
    # Similar wall thickness suggests similar installation vintage and environment
    wt_target = target_defect.get('wall_thickness_mm', 10.0)
    wt_similar = similar_defect.get('wall_thickness_mm', 10.0)
    
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