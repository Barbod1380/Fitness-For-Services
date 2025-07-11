"""
Multi-year comparison view for the Pipeline Analysis application.
"""
import streamlit as st
from visualization.comparison_viz import *
from app.services.state_manager import *
from core.multi_year_analysis import compare_defects

def render_comparison_view():
    """
    Main function to render the multi-year comparison view.
    """
    st.title("🔄 Multi-Year Comparison")
    st.markdown("**Compare defect growth patterns between inspection years**")
    
    # Check if we have enough datasets
    datasets = get_state('datasets', {})
    available_years = sorted(datasets.keys())
    
    if len(available_years) < 2:
        st.error("**⚠️ Insufficient Data for Multi-Year Analysis**")
        
        with st.container():
            st.info(f"""
            📋 **Requirements Check:**
            - **Current datasets:** {len(available_years)}
            - **Required:** 2 or more
            - **Status:** {'✅ Ready' if len(available_years) >= 2 else '❌ Need more data'}
            
            **Next Steps:**
            1. Upload additional inspection data from different years
            2. Ensure data contains matching defects with location information
            3. Return to this page to start the comparison
            """)
        
        if available_years:
            st.markdown("**📅 Currently loaded years:**")
            cols = st.columns(min(len(available_years), 4))
            for i, year in enumerate(available_years):
                with cols[i % 4]:
                    defect_count = len(datasets[year]['defects_df'])
                    st.metric(f"Year {year}", f"{defect_count} defects")
        return
    
    # Improved layout with sections
    with st.container():
        st.markdown("---")
        st.subheader("📅 Year Selection")
        
        # Year selection in organized columns
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            earlier_year = st.selectbox(
                "**Baseline Year** (Earlier)",
                options=available_years[:-1],
                key="earlier_year_select",
                help="Select the baseline inspection year for comparison"
            )
        
        with col2:
            later_years = [year for year in available_years if year > earlier_year]
            if not later_years:
                st.error("No later years available")
                return
                
            later_year = st.selectbox(
                "**Comparison Year** (Later)", 
                options=later_years,
                key="later_year_select",
                help="Select the year to compare against baseline"
            )
        
        with col3:
            year_gap = later_year - earlier_year
            st.metric("Time Gap", f"{year_gap} years")
    
    # Enhanced dataset summary
    st.markdown("---")
    st.subheader("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📋 {earlier_year} Dataset (Baseline)")
        earlier_data = datasets[earlier_year]
        earlier_defects = len(earlier_data['defects_df'])
        earlier_joints = len(earlier_data['joints_df'])
        
        # Metrics for earlier data
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Defects", earlier_defects)
        with metrics_col2:
            st.metric("Joints", earlier_joints)
        
        st.write(f"**Pipe Diameter:** {earlier_data['pipe_diameter']}mm")
    
    with col2:
        st.markdown(f"### 📋 {later_year} Dataset (Comparison)")
        later_data = datasets[later_year]
        later_defects = len(later_data['defects_df'])
        later_joints = len(later_data['joints_df'])
        
        # Metrics for later data
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Defects", later_defects, delta=later_defects - earlier_defects)
        with metrics_col2:
            st.metric("Joints", later_joints, delta=later_joints - earlier_joints)
        
        st.write(f"**Pipe Diameter:** {later_data['pipe_diameter']}mm")
    
    # Matching parameters
    st.markdown("---")
    st.subheader("🎯 Matching Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        distance_tolerance = st.number_input(
            "**Distance Tolerance (meters)**",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Maximum distance difference to consider defects as the same location"
        )
    
    with col2:
        clock_tolerance = st.number_input(
            "**Clock Tolerance (minutes)**",
            min_value=1,
            max_value=120,
            value=20,
            step=5,
            help="Maximum clock position difference to consider defects at same position"
        )
    
    # Analysis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Start Comparison Analysis", type="primary", use_container_width=True):
            st.success(f"✅ Analysis initialized for {earlier_year} vs {later_year}")
            
            with st.spinner("🔄 Processing multi-year comparison..."):
                # Perform actual defect matching
                try:
                    comparison_results = compare_defects(
                        earlier_data, later_data, earlier_year, later_year,
                        distance_tolerance, clock_tolerance
                    )
                    
                    # Store results in session state
                    update_state('comparison_results', comparison_results, validate=False)
                    update_state('comparison_years', (earlier_year, later_year), validate=False)
                    
                    st.success(f"✅ Matched {comparison_results['common_defects_count']} defects, found {comparison_results['new_defects_count']} new defects")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 **Troubleshooting Tips:**\n- Ensure both datasets have 'log dist. [m]' columns\n- Check that defects have valid location data")
                    return
            
            st.rerun()
    
    # Display data preview and results
    comparison_years = get_state('comparison_years')
    if comparison_years and comparison_years == (earlier_year, later_year):
        display_data_preview_and_results(earlier_data, later_data)


def get_display_columns(df):
    """
    Get the most relevant columns for display in the data preview.
    """
    # Priority columns to show
    priority_cols = [
        'log dist. [m]', 'joint number', 'depth [%]', 
        'length [mm]', 'width [mm]', 'clock',
        'component / anomaly identification'
    ]
    
    # Get columns that exist in the dataframe
    available_cols = [col for col in priority_cols if col in df.columns]
    
    # If we have less than 4 columns, add others
    if len(available_cols) < 4:
        other_cols = [col for col in df.columns if col not in available_cols]
        available_cols.extend(other_cols[:4-len(available_cols)])
    
    return available_cols[:6]  # Limit to 6 columns max

    
    # Combined dataset info
    st.info(f"""
    **📊 {year} Dataset Summary:**
    - Defects: {len(data['defects_df'])}
    - Joints: {len(data['joints_df'])}
    - Pipe Diameter: {data['pipe_diameter']}mm
    """)


def display_data_preview_and_results(earlier_data, later_data):
    """
    Display data preview and analysis results.
    """
    st.markdown("---")
    st.subheader("📈 Analysis Results")
    
    st.markdown("### Comparison Charts")
    
    # Create two columns for the visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        donut_chart = create_defect_status_donut(earlier_data, later_data)
        st.plotly_chart(donut_chart, use_container_width=True)
    
    with col2:
        bar_chart = create_new_defects_by_type_bar(later_data)
        st.plotly_chart(bar_chart, use_container_width=True)