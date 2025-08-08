# core/standards_compliant_clustering.py
import numpy as np
import streamlit as st
import pandas as pd
import math
from dataclasses import dataclass
from enum import Enum


class ClusteringStandard(Enum):
    """Supported industry standards for defect clustering"""
    DNV_RP_F101 = "DNV-RP-F101 Composite Defects"
    ROSEN_15 = "ROSEN#15"


@dataclass
class InteractionCriteria:
    """Standard-specific interaction criteria"""
    axial_distance_mm: float
    circumferential_distance_mm: float
    depth_interaction_factor: float
    standard_name: str
    applicability_notes: str


class StandardsCompliantClusterer:
    """
    Industry-standards compliant defect clustering implementation.
    Replaces proprietary parameters with validated industry criteria.
    """
    
    def __init__(self, standard: ClusteringStandard, pipe_diameter_mm: float, conservative_factor: float, use_projection_sweep: bool = True, projection_step_deg: int = 5, allow_across_girth_welds: bool = True):
        
        """
        Create a standards-aware clusterer for pipeline corrosion features.

        Parameters
        ----------
        standard : ClusteringStandard
            Which published methodology to use for *interaction screening* and clustering
            (e.g., B31G/Modified B31G-style 3t/6t rules, DNV-RP-F101 composite defect pre-screen).
        pipe_diameter_mm : float
            Pipe outside diameter [mm]; used for circumferential arc-length calculations from clock positions.
        conservative_factor : float, optional
            Multiplier ≥ 1.0 applied to computed interaction distances as an engineering margin.
            This does *not* replace the standard’s assessment rules (e.g., DNV combined-defect pressure check).
        """
        
        self.standard = standard
        self.pipe_diameter_mm = pipe_diameter_mm
        self.conservative_factor = conservative_factor
                
        # Projection sweep applies to DNV only; ROSEN uses pairwise only
        self.use_projection_sweep = (self.standard == ClusteringStandard.DNV_RP_F101)
        self.projection_step_deg = int(projection_step_deg)
        self.allow_across_girth_welds = bool(allow_across_girth_welds)
        
        # Initialize standard-specific parameters
        self._initialize_standard_parameters()

    def _clock_to_deg(self, clock_float) -> float | None:
        """Convert 'clock' (0..12) to degrees (0..360)."""
        if clock_float is None or (isinstance(clock_float, float) and math.isnan(clock_float)):
            return None
        return (float(clock_float) % 12.0) * 30.0


    def _angle_dist_deg(self, a: float, b: float) -> float:
        """Smallest angular distance in degrees."""
        d = abs((a - b) % 360.0)
        return min(d, 360.0 - d)


    def _halfwidth_deg_from_width_mm(self, width_mm: float) -> float:
        """Half the angular span (deg) from circumferential width in mm."""
        # φ (deg) = 360 * width / (π D)  → half-angle = φ/2
        return 180.0 * float(width_mm) / (math.pi * self.pipe_diameter_mm)
    

    def _merge_overlapping_groups(self, groups):
        """Union overlapping index sets: [[1,2],[2,3],[10,11]] → [[1,2,3],[10,11]]."""
        if not groups:
            return []
        parent = {}
        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for g in groups:
            for i in range(1, len(g)):
                union(g[0], g[i])
        buckets = {}
        for g in groups:
            for i in g:
                buckets.setdefault(find(i), set()).add(i)
        return [sorted(list(s)) for s in buckets.values()]


    def _projection_sweep_clusters(self, defects_df: pd.DataFrame) -> list[list[int]]:
        """
        DNV projection-line sweep:
        - For each generator angle θg (0..360 in self.projection_step_deg),
        select defects whose circumferential span covers θg.
        - Merge defects along θg if axial intervals overlap or if the gap ≤ 2√(D·t_local).
        - Returns clusters as lists of *row positions* (0..N-1) in defects_df.
        """
        if defects_df.empty:
            return []

        # Map DataFrame row index → positional index (what find_interacting_defects returns today)
        pos_of = {idx: pos for pos, idx in enumerate(defects_df.index)}

        items = []  # row-level cache for projection
        for idx, row in defects_df.iterrows():
            start_mm = float(row.get("log dist. [m]", 0.0)) * 1000.0   # defect START in meters → mm
            length_mm = row.get("current_length_mm", row.get("length [mm]", 0.0)) or 0.0
            end_mm = start_mm + float(length_mm)

            theta = self._clock_to_deg(row.get("clock_float"))
            width_mm = row.get("current_width_mm", row.get("width [mm]", None))
            if theta is None or width_mm is None:
                # no clock or no width → skip; pairwise will still handle these
                continue

            halfw_deg = self._halfwidth_deg_from_width_mm(float(width_mm))
            t_mm = row.get("t_mm")  # will be added in find_interacting_defects
            items.append({
                "row_idx": idx,
                "pos": pos_of[idx],
                "start_mm": start_mm,
                "end_mm": end_mm,
                "theta_deg": theta,
                "halfw_deg": halfw_deg,
                "t_mm": float(t_mm) if t_mm is not None else None,
                "joint": row.get("joint number", None),
            })

        if len(items) < 2:
            return []

        # Union-Find keyed by DataFrame row index
        parent = {it["row_idx"]: it["row_idx"] for it in items}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        min_width_deg = min((2*it['halfw_deg'] for it in items), default=float(self.projection_step_deg))
        step = max(1, int(min(self.projection_step_deg, max(1.0, 0.5*min_width_deg))))
        for theta_g in range(0, 360, step):
            # Defects whose circumferential span covers this generator
            cand = [it for it in items if self._angle_dist_deg(it["theta_deg"], theta_g) <= it["halfw_deg"]]
            if len(cand) < 2:
                continue

            # Sort along axis
            cand.sort(key=lambda x: (x["start_mm"], x["end_mm"]))
            prev = cand[0].copy()
            for it in cand[1:]:
                # local thickness for the gap between prev & it
                if prev["t_mm"] is not None and it["t_mm"] is not None:
                    t_local = min(prev["t_mm"], it["t_mm"])
                else:
                    # fall back to dataset median thickness if available; otherwise use 10 mm nominal
                    t_series = defects_df['t_mm'] if 't_mm' in defects_df.columns else None
                    t_med = float(pd.Series(t_series).median(skipna=True)) if t_series is not None else float('nan')
                    t_local = t_med if (t_med == t_med) else 10.0

                same_joint_ok = (
                    self.allow_across_girth_welds
                    or (prev['joint'] is None or it['joint'] is None)
                    or (prev['joint'] == it['joint'])
                )
                
                s_th = 2.0 * math.sqrt(self.pipe_diameter_mm * max(t_local, 0.0)) * self.conservative_factor

                if same_joint_ok and (it["start_mm"] - prev["end_mm"] <= s_th):
                    union(prev["row_idx"], it["row_idx"])
                    prev["end_mm"] = max(prev["end_mm"], it["end_mm"])  # extend merged interval
                else:
                    prev = it.copy()

        # Collect groups (only multi-member)
        groups = {}
        for it in items:
            r = find(it["row_idx"])
            groups.setdefault(r, set()).add(it["row_idx"])

        clusters = []
        for members in groups.values():
            if len(members) >= 2:
                clusters.append(sorted(pos_of[m] for m in members))
        return clusters


    def _initialize_standard_parameters(self):
        """
        Initialize internal parameters for the selected standard.

        Notes
        -----
        This sets *screening* parameters (e.g., default spacing factors) used to pre-select
        candidate interacting defects. Standards like DNV-RP-F101 ultimately require
        a combined-defect pressure check; these parameters alone do not constitute
        the full standard assessment.
        """

        if self.standard == ClusteringStandard.DNV_RP_F101:
            self._setup_dnv_parameters()
        elif self.standard == ClusteringStandard.ROSEN_15:
            self._setup_rosen15_parameters()
        else:
            raise ValueError(f"Unsupported standard: {self.standard}")
    
    def _setup_dnv_parameters(self):
        """
        Configures DNV-RP-F101 composite defect parameters.
        
        References:
        - DNV-RP-F101:2019 Section 5.3.4 (Interaction Rules)
        - Equation 5.18 (Composite Defect Length)
        """
        self.standard_name = "DNV-RP-F101:2019"
        self.reference = "DNV-RP-F101:2019 Section 5.3"
        self.applicability_notes = (
            "Defects interact unless isolation screens are satisfied:\n"
            "- Axial spacing s > 2√(D·t)\n"
            "- Circumferential spacing φ > 360·t/D  (arc length = π·t)"
        )

    def _setup_rosen15_parameters(self):
        """
        Set defaults for ROSEN #15 clustering (pairwise-only).
        Thresholds are pair-dependent:
            Axial: IL = min(6 t, (L1 + L2)/2)
            Circ.: IW = min(6 t, (W1 + W2)/2)
        All distances in mm.
        """
        # Pairwise only; disable projection sweep
        self.use_projection_sweep = False

        # Optional behavior flags (keep consistent with DNV defaults unless you decide otherwise)
        # If you want to *require* clock+width for circ check, flip to True.
        self.require_clock_for_circ = False

        # Across-girth-weld behavior stays whatever the instance was constructed with
        # (your factory/UI decides this).
        # self.allow_across_girth_welds = self.allow_across_girth_welds

        # Notes for debugging/UX
        self.standard_name = "ROSEN#15"
        self.applicability_notes = (
            "ROSEN #15 screening: axial IL = min(6 t, (L1+L2)/2), "
            "circum IW = min(6 t, (W1+W2)/2). Pair-dependent thresholds."
        )


    def calculate_interaction_criteria(self, wall_thickness_mm: float) -> InteractionCriteria:
        """
        Calculate interaction criteria for given wall thickness.
        
        Parameters:
        - wall_thickness_mm: Nominal wall thickness in mm
        
        Returns:
        - InteractionCriteria object with calculated distances
        """
        if self.standard == ClusteringStandard.DNV_RP_F101:
            return self._calculate_dnv_criteria(wall_thickness_mm)
        
        if self.standard == ClusteringStandard.ROSEN_15:
            return InteractionCriteria(
                axial_distance_mm=None,
                circumferential_distance_mm=None,
                depth_interaction_factor=None,
                standard_name=self.standard_name,
                applicability_notes=self.applicability_notes
            )

    def _calculate_dnv_criteria(self, wall_thickness_mm: float) -> InteractionCriteria:
        """
        Calculates DNV-RP-F101:2019 interaction screening criteria for corrosion defects.

        References
        ----------
        - DNV-RP-F101:2019 Section 5.3.4 and Table 5-5
        - Equation 5.18 (Composite Defect Length)

        These criteria are intended for initial grouping. For full standard compliance,
        colonies should be assessed using the combined-profile method (Sec 5.3.5).

        Returns
        -------
        InteractionCriteria
            Contains axial, circumferential, and depth ratio thresholds.
        """

        axial_distance = 2.0 * math.sqrt(self.pipe_diameter_mm * wall_thickness_mm) 
        circumferential_distance = math.pi * wall_thickness_mm           

        # Apply engineering conservatism (≥1) to make interaction more likely
        axial_distance *= self.conservative_factor
        circumferential_distance *= self.conservative_factor

        return InteractionCriteria(
            axial_distance_mm = axial_distance,
            circumferential_distance_mm = circumferential_distance,
            depth_interaction_factor = None,  
            standard_name = self.standard_name,
            applicability_notes = self.applicability_notes
        )
    

    def _pairwise_interacting_clusters(self, defects_df: pd.DataFrame, joints_df: pd.DataFrame, show_progress: bool = True) -> list[list[int]]:
        """
        Existing pairwise/vectorized method you already had, unchanged logic.
        Must return clusters as lists of *row positions* in defects_df.
        """
                
        # Pre-compute wall thickness lookup once
        wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
        
        # Pre-sort defects by location for spatial indexing
        defects_sorted = defects_df.sort_values('log dist. [m]').reset_index()
        original_indices = defects_sorted['index'].tolist()  # Keep track of original indices

        n_defects = len(defects_sorted)
        clusters = []
        processed_indices = set()
        
        # Progress bar setup
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Batch process defects in chunks to update progress
        batch_size = max(1, n_defects // 20)  # 20 progress updates
        
        for batch_start in range(0, n_defects, batch_size):
            batch_end = min(batch_start + batch_size, n_defects)
            
            # Update progress
            if show_progress:
                progress = batch_end / n_defects
                progress_bar.progress(progress)
                status_text.text(f"Processing defects {batch_end}/{n_defects} ({progress*100:.1f}%)")
            
            # Process batch
            for i in range(batch_start, batch_end):
                original_i = original_indices[i]
                
                if original_i in processed_indices:
                    continue
                
                defect_i = defects_sorted.iloc[i]
                
                # Get wall thickness once per defect
                wall_thickness = wt_lookup.get(defect_i['joint number'], 10.0)
                criteria = self.calculate_interaction_criteria(wall_thickness)
                
                # Spatial indexing - only check nearby defects
                cluster = [original_i]
                location_i = defect_i['log dist. [m]']
                
                if self.standard == ClusteringStandard.DNV_RP_F101:
                    max_distance_m = criteria.axial_distance_mm / 1000.0  # fixed DNV screen
                elif self.standard == ClusteringStandard.ROSEN_15:
                    # Upper bound for Rosen axial threshold:
                    # IL = min(6*t, (L1+L2)/2) <= 6*t  → use a single global window of 6*max(t)
                    t_max_mm = float(defects_df['t_mm'].max())
                    max_distance_m = (6.0 * t_max_mm) / 1000.0
                
                # Binary search for nearby defects (much faster than checking all)
                start_idx = self._find_nearby_start(defects_sorted, location_i - max_distance_m)
                end_idx = self._find_nearby_end(defects_sorted, location_i + max_distance_m)

                # Only check defects within spatial range
                for j in range(start_idx, end_idx + 1):
                    original_j = original_indices[j]
                    
                    if i == j or original_j in processed_indices:
                        continue
                    
                    defect_j = defects_sorted.iloc[j]
                    
                    # Use optimized interaction check
                    if self._defects_interact_vectorized(defect_i, defect_j, criteria):
                        cluster.append(original_j)                

                # If cluster has more than one defect, it's meaningful
                if len(cluster) > 1:
                    clusters.append(cluster)
                    processed_indices.update(cluster)
                else:
                    processed_indices.add(original_i)
        
        # Cleanup progress display
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text(f"✅ Clustering complete: {len(clusters)} clusters found")
        return clusters


    def find_interacting_defects(self, defects_df: pd.DataFrame, joints_df: pd.DataFrame, show_progress: bool = True) -> list[list[int]]:
        """
        Orchestrate DNV projection sweep (DNV only) + pairwise logic (DNV/ROSEN).
        Returns clusters as lists of positional indices into defects_df.
        """

        # --- Required columns vary by standard ---
        base_required = ['log dist. [m]', 'joint number', 'depth [%]']
        if self.standard == ClusteringStandard.DNV_RP_F101:
            required_defect_cols = base_required + ['length [mm]', 'width [mm]']   # DNV uses both in sweep & circ check
        elif self.standard == ClusteringStandard.ROSEN_15:
            # Rosen IL/IW depend on lengths/widths for each PAIR
            required_defect_cols = base_required + ['length [mm]', 'width [mm]']
        else:
            required_defect_cols = base_required

        required_joint_cols = ['joint number', 'wt nom [mm]']
        missing_defect_cols = [c for c in required_defect_cols if c not in defects_df.columns]
        missing_joint_cols = [c for c in required_joint_cols if c not in joints_df.columns]
        if missing_defect_cols or missing_joint_cols:
            raise ValueError(f"Missing required columns: {missing_defect_cols + missing_joint_cols}")

        # --- Working copy + merge nominal thickness as 't_mm' (mm) ---
        df = defects_df.copy()
        t_map = joints_df[['joint number', 'wt nom [mm]']].rename(columns={'wt nom [mm]': 't_mm'})
        df = df.merge(t_map, on='joint number', how='left')

        # Sanity: diameter unit & thickness presence
        if not (50.0 <= float(self.pipe_diameter_mm) <= 3500.0):
            raise ValueError(f"pipe_diameter_mm looks wrong ({self.pipe_diameter_mm}). Expect mm.")
        if df['t_mm'].isna().all() or (df['t_mm'] <= 0).all():
            raise ValueError("All t_mm missing/<=0. Check joints_df['wt nom [mm]'] and units.")

        all_groups: list[list[int]] = []

        # --- DNV-only helpers (effective sweep step; optional axial gap debug) ---
        if self.standard == ClusteringStandard.DNV_RP_F101 and self.use_projection_sweep:
            widths_deg = 180.0 * df['width [mm]'] / (math.pi * self.pipe_diameter_mm)
            min_width_deg = float(widths_deg.min())
            eff_step = int(min(self.projection_step_deg, max(1.0, 0.5 * min_width_deg)))

        # --- DNV projection sweep (DNV only) ---
        if self.standard == ClusteringStandard.DNV_RP_F101 and self.use_projection_sweep:
            proj_groups = self._projection_sweep_clusters(df)
            if proj_groups:
                all_groups.extend(proj_groups)

        # --- Pairwise/vectorized logic (both standards) ---
        pair_groups = self._pairwise_interacting_clusters(df, joints_df, show_progress=show_progress)

        # Convert original index clusters to positional indices (simulator uses iloc)
        pos_of = {idx: pos for pos, idx in enumerate(df.index)}
        pair_groups = [[pos_of[o] for o in group if o in pos_of] for group in pair_groups]
        if pair_groups:
            all_groups.extend(pair_groups)

        # --- Union overlapping groups and return ---
        final_groups = self._merge_overlapping_groups(all_groups)
        return final_groups


    def _defects_interact_vectorized(self, defect1: pd.Series, defect2: pd.Series, criteria: InteractionCriteria) -> bool:
        # Respect across-girth-weld setting
        if (not self.allow_across_girth_welds) and (defect1['joint number'] != defect2['joint number']):
            return False

        # Pair-local thickness for thresholds (mm)
        t1 = defect1.get('t_mm')
        t2 = defect2.get('t_mm')
        if (pd.isna(t1) and pd.isna(t2)):
            return False  # cannot evaluate without thickness
        t_pair = min([t for t in [t1, t2] if not pd.isna(t)])

        # Axial ligament (edge-to-edge) in mm
        z1 = float(defect1['log dist. [m]']) * 1000.0
        L1 = float(defect1.get('length [mm]', defect1.get('current_length_mm', 0.0)))
        z2 = float(defect2['log dist. [m]']) * 1000.0
        L2 = float(defect2.get('length [mm]', defect2.get('current_length_mm', 0.0)))
        start1, end1 = z1, z1 + L1
        start2, end2 = z2, z2 + L2
        s_axial = max(0.0, max(start2 - end1, start1 - end2))

        # Circumferential ligament (edge-to-edge) in mm (if clock & width available)
        w1 = defect1.get('width [mm]', defect1.get('current_width_mm', None))
        w2 = defect2.get('width [mm]', defect2.get('current_width_mm', None))
        c1 = defect1.get('clock_float', None)
        c2 = defect2.get('clock_float', None)
        s_circ = None
        if (w1 is not None and w2 is not None and pd.notna(c1) and pd.notna(c2)):
            clock_diff = min(abs(c1 - c2), 12 - abs(c1 - c2))  # in "hours"
            center_arc = (clock_diff / 12.0) * math.pi * self.pipe_diameter_mm  # mm
            s_circ = max(0.0, center_arc - 0.5*(float(w1) + float(w2)))

        if self.standard == ClusteringStandard.DNV_RP_F101:
            # DNV screens
            s_axial_th = 2.0 * math.sqrt(self.pipe_diameter_mm * t_pair) * self.conservative_factor
            if s_axial > s_axial_th:
                return False
            if s_circ is not None:
                s_circ_th = math.pi * t_pair * self.conservative_factor
                if s_circ > s_circ_th:
                    return False
            return True

        if self.standard == ClusteringStandard.ROSEN_15:
            # Rosen #15: IL = min(6t, (L1+L2)/2), IW = min(6t, (W1+W2)/2)
            IL = min(6.0 * t_pair, 0.5 * (L1 + L2))
            if s_axial >= IL:
                return False
            if s_circ is not None and (w1 is not None and w2 is not None):
                IW = min(6.0 * t_pair, 0.5 * (float(w1) + float(w2)))
                if s_circ >= IW:
                    return False
            return True

        # Default: no interaction
        return False

    def _find_nearby_start(self, sorted_defects: pd.DataFrame, target_location: float) -> int:
        """
        Binary search for spatial indexing - 10x faster than linear search
        """
        locations = sorted_defects['log dist. [m]'].values
        left, right = 0, len(locations) - 1
        
        while left < right:
            mid = (left + right) // 2
            if locations[mid] < target_location:
                left = mid + 1
            else:
                right = mid
        return max(0, left - 1)  # Include one before for safety


    def _find_nearby_end(self, sorted_defects: pd.DataFrame, target_location: float) -> int:
        """Binary search for end of spatial range"""
        locations = sorted_defects['log dist. [m]'].values
        left, right = 0, len(locations) - 1
        
        while left < right:
            mid = (left + right + 1) // 2
            if locations[mid] <= target_location:
                left = mid
            else:
                right = mid - 1
        return min(len(locations) - 1, right + 1)  # Include one after for safety


def create_standards_compliant_clusterer(standard_name: str, pipe_diameter_mm: float, conservative_factor: float = 1.2) -> StandardsCompliantClusterer:
    """
    Factory function to create standards-compliant clusterer.
    
    Parameters:
    - standard_name: Name of standard ("DNV")
    - pipe_diameter_mm: Pipeline diameter in mm
    - conservative_factor: Additional conservatism factor
    
    Returns:
    - Configured StandardsCompliantClusterer instance
    """
    
    standard_mapping = {
        "DNV": ClusteringStandard.DNV_RP_F101,
        "DNV-RP-F101": ClusteringStandard.DNV_RP_F101,
        "DNV-RP-F101 Composite Defects": ClusteringStandard.DNV_RP_F101,
        "ROSEN#15": ClusteringStandard.ROSEN_15,  
        "ROSEN15": ClusteringStandard.ROSEN_15,
        "ROSEN": ClusteringStandard.ROSEN_15,
    }
    
    if standard_name not in standard_mapping:
        raise ValueError(f"Unknown standard: {standard_name}. Choose from: {list(standard_mapping.keys())}")
    
    return StandardsCompliantClusterer(
        standard=standard_mapping[standard_name],
        pipe_diameter_mm=pipe_diameter_mm,
        conservative_factor=conservative_factor
    )