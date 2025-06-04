import streamlit as st

def load_css():
    """Apply custom CSS styling to the app."""
    css = """
    <style>
    /* Main container styling */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Custom title and header styling */
    .custom-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E5288;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        text-align: center;
        border-bottom: 2px solid #3A8B94;
    }
    
    .custom-subtitle {
        font-size: 1.2rem;
        color: #5D6D7E;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Card-like container styling */
    .card-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        border-top: 4px solid #3A8B94;
    }
    
    /* Data visualization container */
    .viz-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E5288;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #E4EBF5;
        border-left: 4px solid #3A8B94;
        padding: 12px 15px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 15px;
    }
    
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #E67E22;
        padding: 12px 15px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 15px;
    }
    
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid #27AE60;
        padding: 12px 15px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 15px;
    }
    
    .critical-box {
        background-color: #FFEBEE;
        border-left: 4px solid #922B21;
        padding: 12px 15px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 15px;
    }
    
    /* Data table styling */
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .styled-table thead tr {
        background-color: #1E5288;
        color: #ffffff;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }

    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f8fa;
    }
    
    /* Row hover effect */
    .styled-table tbody tr:hover {
        background-color: #E4EBF5;
    }

    /* Sidebar specific styling */
    .sidebar .sidebar-content {
        background-color: #2C3E50;
    }
    
    /* Button styling */
    .custom-button {
        background-color: #3A8B94;
        color: white;
        padding: 10px 18px;
        border-radius: 5px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        border: none;
        margin: 5px 0;
        font-weight: 500;
    }
    
    .custom-button:hover {
        background-color: #1E5288;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .secondary-button {
        background-color: #F5F7FA;
        color: #1E5288;
        border: 1px solid #1E5288;
        padding: 9px 18px;
        border-radius: 5px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        margin: 5px 0;
        font-weight: 500;
    }
    
    .secondary-button:hover {
        background-color: #E4EBF5;
    }
    
    /* Custom metric styling */
    .custom-metric {
        background-color: white;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s;
    }
    
    .custom-metric:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1E5288;
    }
    
    .metric-label {
        font-size: 14px;
        color: #5D6D7E;
        margin-top: 5px;
    }
    
    /* Tab styling - override Streamlit's default */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f1f1;
        border-radius: 4px 4px 0 0;
        padding-left: 20px;
        padding-right: 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E5288;
        color: white;
    }
    
    /* Progress bar step indicator */
    .step-progress {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
        position: relative;
    }
    
    .step-progress:before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 2px;
        background: #e0e0e0;
        transform: translateY(-50%);
        z-index: 1;
    }
    
    .step {
        width: 32px;
        height: 32px;
        background-color: white;
        border: 2px solid #e0e0e0;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        z-index: 2;
        position: relative;
    }
    
    .step.active {
        background-color: #3A8B94;
        border-color: #3A8B94;
        color: white;
    }
    
    .step.completed {
        background-color: #27AE60;
        border-color: #27AE60;
        color: white;
    }
    
    .step-label {
        position: absolute;
        top: 38px;
        font-size: 12px;
        width: 100px;
        text-align: center;
        left: -35px;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .logo {
        width: 80px;
        height: 80px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #5D6D7E;
        font-size: 12px;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #e0e0e0;
    }

    /* Custom badges for pipe status */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .status-badge.pass {
        background-color: #E8F5E9;
        color: #27AE60;
        border: 1px solid #27AE60;
    }

    .status-badge.caution {
        background-color: #FFF3E0;
        color: #E67E22;
        border: 1px solid #E67E22;
    }

    .status-badge.fail {
        background-color: #FFEBEE;
        color: #922B21;
        border: 1px solid #922B21;
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }

    .tooltip .tooltip-text {
        visibility: hidden;
        width: 220px;
        background-color: #34495E;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        line-height: 1.4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .tooltip .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #34495E transparent transparent transparent;
    }

    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /* Data visualization enhancements */
    .chart-title {
        font-size: 16px;
        font-weight: 600;
        color: #1E5288;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .chart-description {
        font-size: 13px;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 15px;
    }
    
    /* Make dataframes prettier */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    /* Specific for pipe defect analysis */
    .defect-metrics-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 15px;
        margin: 20px 0;
    }
    
    .defect-metric {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        flex: 1;
        min-width: 150px;
        border-bottom: 3px solid #3A8B94;
    }
    
    .defect-metric.critical {
        border-bottom-color: #922B21;
    }
    
    .defect-metric.warning {
        border-bottom-color: #E67E22;
    }
    
    .defect-metric.good {
        border-bottom-color: #27AE60;
    }
    
    /* Grid layout for dashboards */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .grid-item {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Special styling for the pipe integrity score */
    .integrity-score-container {
        text-align: center;
        padding: 20px;
        position: relative;
    }
    
    .integrity-score {
        font-size: 48px;
        font-weight: bold;
        color: #1E5288;
        margin: 10px 0;
    }
    
    .integrity-label {
        font-size: 16px;
        color: #5D6D7E;
        margin-bottom: 5px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
        
        .defect-metrics-container {
            flex-direction: column;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)