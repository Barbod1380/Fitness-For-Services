"""
Multi-year comparison view for the Pipeline Analysis application.
"""
import streamlit as st
import pandas as pd
from analysis.growth_analysis import correct_negative_growth_rates
from visualization.comparison_viz import *
from app.services.state_manager import *


def display_comparison_visualization_tabs(comparison_results, earlier_year, later_year):
    """Display the consolidated visualization tabs for comparison results."""
    
    viz_tabs = st.tabs(["Defect Overview", "Growth Analysis & Correction"])

    with viz_tabs[0]:
        # Combined overview showing both pie chart and defect types
        col1, col2 = st.columns(2)
        
        with col1:
            pie_fig = create_comparison_stats_plot(comparison_results)
            st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            bar_fig = create_new_defect_types_plot(comparison_results)
            st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Growth Analysis & Correction tab
    with viz_tabs[1]:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">
                🔬 Growth Analysis & Automated Correction
            </h2>
            <p style="color: #e2e8f0; margin: 5px 0 0 0; font-size: 0.9rem;">
                Analyze defect growth patterns with intelligent negative growth correction
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # === STEP 1: DIMENSION SELECTION ===
        with st.container():
            st.markdown("### 📊 Step 1: Select Analysis Dimension")
            
            # Get available dimensions
            available_dimensions = []
            if comparison_results.get('has_depth_data', False):
                available_dimensions.append('depth')
            if comparison_results.get('has_length_data', False):
                available_dimensions.append('length')
            if comparison_results.get('has_width_data', False):
                available_dimensions.append('width')
            
            if not available_dimensions:
                st.error("❌ **No growth data available**")
                st.info("Please ensure both datasets have dimension measurements (depth, length, or width).")
                return
            
            # Dimension selection with enhanced UI
            col1, col2 = st.columns([2, 1])
            
            with col1:
                select_key = f"analysis_dimension_{earlier_year}_{later_year}"
                if select_key not in st.session_state:
                    st.session_state[select_key] = 'depth' if 'depth' in available_dimensions else available_dimensions[0]
                
                if st.session_state[select_key] not in available_dimensions:
                    st.session_state[select_key] = available_dimensions[0]
                
                selected_dimension = st.selectbox(
                    "Choose dimension for analysis",
                    options=available_dimensions,
                    key=select_key,
                    help="Select which defect dimension to analyze for growth patterns",
                    format_func=lambda x: f"🔍 {x.title()} Growth Analysis"
                )
            
            with col2:
                # Show dimension status badges
                status_badges = []
                for dim in ['depth', 'length', 'width']:
                    if dim in available_dimensions:
                        if dim == selected_dimension:
                            status_badges.append(f"🟢 {dim.title()} (Active)")
                        else:
                            status_badges.append(f"🔵 {dim.title()} (Available)")
                    else:
                        status_badges.append(f"⚪ {dim.title()} (N/A)")
                
                st.markdown("**Available Dimensions:**")
                for badge in status_badges:
                    st.markdown(f"- {badge}")
        
        st.markdown("---")
        
        # === STEP 2: VISUAL DATA PRESENTATION ===
        with st.container():
            st.markdown(f"### 📈 Step 2: {selected_dimension.title()} Growth Visualization")
            
            # Show the growth plot in a nice container
            with st.expander(f"📊 {selected_dimension.title()} Growth Data Plot", expanded=True):
                original_plot = create_negative_growth_plot(comparison_results, dimension=selected_dimension)
                st.plotly_chart(original_plot, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        
        # === STEP 3: ANALYSIS & CORRECTION (DEPTH ONLY) ===
        if selected_dimension == 'depth':
            # Check if depth data is available for correction
            if not (comparison_results.get('has_depth_data', False) and 'is_negative_growth' in comparison_results['matches_df'].columns):
                st.error("❌ **Depth Analysis Unavailable**")
                st.warning("No depth growth data available for correction. Make sure both datasets have depth measurements.")
            else:
                st.markdown("### 🎯 Step 3: Depth Analysis & Intelligent Correction")
                
                # Calculate key metrics
                neg_count = comparison_results['matches_df']['is_negative_growth'].sum()
                total_count = len(comparison_results['matches_df'])
                pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                
                # === ANALYSIS SUMMARY ===
                st.markdown("#### 📋 Analysis Summary")
                
                # Enhanced metrics display
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                        <div style="font-size: 2rem; font-weight: bold; color: #1e40af;">{total_count}</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Total Defects</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    color = "#ef4444" if neg_count > 0 else "#10b981"
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid {color};">
                        <div style="font-size: 2rem; font-weight: bold; color: {color};">{neg_count}</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Negative Growth</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    color = "#ef4444" if pct > 10 else "#f59e0b" if pct > 5 else "#10b981"
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid {color};">
                        <div style="font-size: 2rem; font-weight: bold; color: {color};">{pct:.1f}%</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Anomaly Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col4:
                    # Correction status
                    corrected = st.session_state.corrected_results is not None
                    if corrected:
                        correction_info = st.session_state.corrected_results.get('correction_info', {})
                        corrected_count = correction_info.get('corrected_count', 0)
                        status_text = f"{corrected_count}"
                        status_color = "#10b981"
                        label = "Corrected"
                    else:
                        status_text = "Pending"
                        status_color = "#f59e0b"
                        label = "Status"
                    
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid {status_color};">
                        <div style="font-size: 2rem; font-weight: bold; color: {status_color};">{status_text}</div>
                        <div style="color: #64748b; font-size: 0.9rem;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # === CORRECTION WORKFLOW ===
                if neg_count > 0:
                    st.markdown("#### 🔧 KNN Correction Workflow")
                    
                    # Check if joint numbers are available
                    has_joint_num = comparison_results.get('has_joint_num', False)
                    if not has_joint_num:
                        st.error("""
                        ❌ **Prerequisites Missing**
                        
                        KNN correction requires 'joint number' column in your defect data.
                        Please ensure both datasets have this column properly mapped.
                        """)
                    else:
                        # Show correction status and controls
                        if st.session_state.corrected_results is None:
                            # Pre-correction state
                            st.info("""
                            🤖 **Intelligent KNN Correction Ready**
                            
                            - **Algorithm**: K-Nearest Neighbors (K=5)
                            - **Method**: Match negative growth defects to similar nearby defects
                            - **Criteria**: Defect type, depth, dimensions, and location
                            - **Safety**: Conservative engineering approach using measured data
                            """)
                            
                            # Correction button with enhanced styling
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
                            with button_col2:
                                if st.button(
                                    "🚀 Apply KNN Correction (K=5)", 
                                    use_container_width=True, 
                                    type="primary",
                                    help="Apply intelligent correction to negative depth growth anomalies"
                                ):
                                    with st.spinner("🔄 Applying KNN correction with K=5 neighbors..."):
                                        try:
                                            corrected_results = st.session_state.comparison_results.copy()
                                            
                                            # Apply correction
                                            corrected_df, correction_info = correct_negative_growth_rates(
                                                st.session_state.comparison_results['matches_df'], 
                                                k=5
                                            )
                                            
                                            corrected_results['matches_df'] = corrected_df
                                            corrected_results['correction_info'] = correction_info
                                            
                                            # Update growth stats
                                            if correction_info.get("updated_growth_stats"):
                                                corrected_results['growth_stats'] = correction_info['updated_growth_stats']
                                            
                                            st.session_state.corrected_results = corrected_results
                                            
                                            if correction_info.get("success", False):
                                                st.success(f"✅ Successfully corrected {correction_info['corrected_count']} out of {correction_info['total_negative']} defects!")
                                                if correction_info['uncorrected_count'] > 0:
                                                    st.warning(f"⚠️ {correction_info['uncorrected_count']} defects couldn't be corrected (insufficient similar defects)")
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Correction failed: {correction_info.get('error', 'Unknown error')}")
                                        except Exception as e:
                                            st.error(f"❌ Error during correction: {str(e)}")
                                            st.info("💡 Check data requirements: joint number, length, width, and depth columns")
                        else:
                            # Post-correction state
                            correction_info = st.session_state.corrected_results.get('correction_info', {})
                            
                            if correction_info.get("success", False):
                                st.success(f"""
                                ✅ **Correction Complete**
                                
                                - **Corrected**: {correction_info['corrected_count']} out of {correction_info['total_negative']} negative growth defects
                                - **Method**: K-Nearest Neighbors (K=5)
                                - **Success Rate**: {(correction_info['corrected_count']/correction_info['total_negative']*100):.1f}%
                                """)
                                
                                if correction_info['uncorrected_count'] > 0:
                                    st.warning(f"""
                                    ⚠️ **Partial Correction**
                                    
                                    {correction_info['uncorrected_count']} defects in {len(correction_info['uncorrected_joints'])} joints 
                                    could not be corrected due to insufficient similar defects for reliable estimation.
                                    """)
                                
                                # Show corrected visualization
                                st.markdown("#### 🎨 Corrected Growth Visualization")
                                with st.expander("📊 View Corrected Data Plot", expanded=True):
                                    corrected_plot = create_negative_growth_plot(st.session_state.corrected_results, dimension='depth')
                                    st.plotly_chart(corrected_plot, use_container_width=True, config={'displayModeBar': False})
                                    
                                    # Legend
                                    st.markdown("""
                                    **Legend:**
                                    - 🔵 **Blue Circles**: Natural positive growth (unchanged)
                                    - 🔴 **Red Triangles**: Negative growth (uncorrected anomalies)
                                    - 🟢 **Green Diamonds**: Corrected growth (formerly negative)
                                    """)
                else:
                    st.success("""
                    ✅ **Excellent Data Quality**
                    
                    No negative depth growth detected! Your data shows consistent, realistic growth patterns.
                    """)
                
                # === GROWTH STATISTICS ===
                st.markdown("#### 📊 Growth Rate Statistics")
                
                # Use corrected data if available
                results_to_use = st.session_state.corrected_results if st.session_state.corrected_results is not None else comparison_results
                matches_df = results_to_use.get('matches_df', pd.DataFrame())
                
                if not matches_df.empty and 'growth_rate_pct_per_year' in matches_df.columns:
                    # Calculate statistics
                    negative_count = matches_df['is_negative_growth'].sum()
                    positive_growth = matches_df[~matches_df['is_negative_growth']]
                    avg_growth = positive_growth['growth_rate_pct_per_year'].mean() if len(positive_growth) > 0 else 0
                    max_growth = positive_growth['growth_rate_pct_per_year'].max() if len(positive_growth) > 0 else 0
                    
                    # Statistics display
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    unit = '%/year' if not comparison_results.get('has_wt_data', False) else 'mm/year'
                    
                    with stats_col1:
                        st.markdown(f"""
                        <div style="background: #f0f9ff; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #0369a1;">{avg_growth:.3f}</div>
                            <div style="color: #0284c7; font-size: 0.9rem;">Average Growth ({unit})</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stats_col2:
                        st.markdown(f"""
                        <div style="background: #fef3c7; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #d97706;">{max_growth:.3f}</div>
                            <div style="color: #ea580c; font-size: 0.9rem;">Maximum Growth ({unit})</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stats_col3:
                        remaining_neg = matches_df['is_negative_growth'].sum()
                        color = "#ef4444" if remaining_neg > 0 else "#10b981"
                        st.markdown(f"""
                        <div style="background: #f1f5f9; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{remaining_neg}</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Remaining Anomalies</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Growth distribution histogram
                    st.markdown("#### 📈 Growth Rate Distribution")
                    with st.expander("📊 View Distribution Histogram", expanded=False):
                        try:
                            growth_hist_fig = create_growth_rate_histogram(results_to_use, dimension='depth')
                            st.plotly_chart(growth_hist_fig, use_container_width=True, config={'displayModeBar': False})
                        except Exception as e:
                            st.warning(f"Could not generate histogram: {str(e)}")
        
        else:
            # === NON-DEPTH DIMENSIONS ===
            st.markdown(f"### 📊 Step 3: {selected_dimension.title()} Analysis")
            
            st.info(f"""
            **📏 {selected_dimension.title()} Growth Analysis**
            
            You are viewing {selected_dimension} growth analysis. The visualization above shows how {selected_dimension} measurements 
            changed between inspections.
            
            **ℹ️ Note**: Automated KNN correction is only available for depth measurements, as depth is the primary 
            safety-critical dimension in pipeline integrity assessment. {selected_dimension.title()} variations may be 
            due to measurement differences between inspections.
            """)
            
            # Show basic statistics for length/width
            matches_df = comparison_results['matches_df']
            
            # Length statistics
            if selected_dimension == 'length' and comparison_results.get('has_length_data', False):
                if 'is_negative_length_growth' in matches_df.columns:
                    neg_count = matches_df['is_negative_length_growth'].sum()
                    total_count = len(matches_df)
                    pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                    
                    st.markdown("#### 📊 Length Growth Summary")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #3b82f6;">{total_count}</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Total Defects</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col2:
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #f59e0b;">{neg_count}</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Negative Growth</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col3:
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #8b5cf6;">{pct:.1f}%</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Percentage</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Width statistics
            elif selected_dimension == 'width' and comparison_results.get('has_width_data', False):
                if 'is_negative_width_growth' in matches_df.columns:
                    neg_count = matches_df['is_negative_width_growth'].sum()
                    total_count = len(matches_df)
                    pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                    
                    st.markdown("#### 📊 Width Growth Summary")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #3b82f6;">{total_count}</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Total Defects</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col2:
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #f59e0b;">{neg_count}</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Negative Growth</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col3:
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #8b5cf6;">{pct:.1f}%</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Percentage</div>
                        </div>
                        """, unsafe_allow_html=True)


def render_comparison_view():
    """
    Main function to render the multi-year comparison view.
    """
    st.title("Multi-Year Comparison")
    st.markdown("Compare defect growth between inspection years")
    
    # Check if we have enough datasets
    datasets = get_state('datasets', {})
    available_years = sorted(datasets.keys())
    
    if len(available_years) < 2:
        st.error("**Insufficient Data for Multi-Year Analysis**")
        st.info(f"""
        You need at least **2 datasets** from different years to perform multi-year comparison.
        
        **Current Status:**
        - Available datasets: {len(available_years)}
        - Required: 2 or more
        
        **Next Steps:**
        1. Upload additional inspection data from different years
        2. Return to this page to start the comparison
        """)
        
        if available_years:
            st.write("**Currently loaded years:**", ", ".join(map(str, available_years)))
        return
    
    # Year selection
    st.subheader("📅 Select Years for Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        earlier_year = st.selectbox(
            "Earlier Inspection Year",
            options=available_years[:-1],
            key="earlier_year_select"
        )
    
    with col2:
        later_years = [year for year in available_years if year > earlier_year]
        if not later_years:
            st.error("No later years available for the selected earlier year")
            return
            
        later_year = st.selectbox(
            "Later Inspection Year", 
            options=later_years,
            key="later_year_select"
        )
    
    # Show basic dataset info
    st.subheader("📊 Dataset Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{earlier_year} Dataset:**")
        earlier_data = datasets[earlier_year]
        st.write(f"- Defects: {len(earlier_data['defects_df'])}")
        st.write(f"- Joints: {len(earlier_data['joints_df'])}")
    
    with col2:
        st.write(f"**{later_year} Dataset:**")
        later_data = datasets[later_year]
        st.write(f"- Defects: {len(later_data['defects_df'])}")
        st.write(f"- Joints: {len(later_data['joints_df'])}")
    
    # Simple analysis button
    if st.button("Start Comparison Analysis", type="primary"):
        st.success(f"Analysis started for {earlier_year} vs {later_year}")
        st.info("Multi-year analysis functionality is under development.")
        
        # Store the selected years in session state
        update_state('comparison_years', (earlier_year, later_year), validate=False)
    
    # Show results with visualizations
    comparison_years = get_state('comparison_years')
    if comparison_years:
        st.subheader("📈 Analysis Results")
        
        # Create two columns for the visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            donut_chart = create_defect_status_donut(earlier_data, later_data)
            st.plotly_chart(donut_chart, use_container_width=True)
        
        with col2:
            bar_chart = create_new_defects_by_type_bar(later_data)
            st.plotly_chart(bar_chart, use_container_width=True)