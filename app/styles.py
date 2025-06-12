import streamlit as st

def load_css():
    """Apply custom CSS styling to the app."""
    css = """
    <style>
    /*---------------------------------------------
      1) Import Google Font: "Inter"
    ---------------------------------------------*/
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /*---------------------------------------------
      2) Root Variables (colors, spacing, shadows)
    ---------------------------------------------*/
    :root {
        --font-family-base: 'Inter', sans-serif;
        --primary-color: #1E5288;       /* deep blue */
        --secondary-color: #3A8B94;     /* teal-ish */
        --accent-color: #27AE60;        /* green */
        --warning-color: #E67E22;       /* orange */
        --danger-color: #922B21;        /* dark red */
        --bg-light: #F5F7FA;            /* very light gray */
        --bg-white: #FFFFFF;
        --text-color: #2C3E50;          /* dark charcoal */
        --subtext-color: #5D6D7E;       /* muted slate */
        --border-color: #E0E0E0;        /* light gray border */
        --shadow-light: rgba(0, 0, 0, 0.05);
        --shadow-medium: rgba(0, 0, 0, 0.1);
        --shadow-strong: rgba(0, 0, 0, 0.15);
        --radius-base: 8px;
        --spacing-unit: 1rem;
    }
    
    html, body, [class*="css"]  {
        font-family: var(--font-family-base) !important;
        color: var(--text-color) !important;
        background-color: var(--bg-light) !important;
    }

    /*---------------------------------------------
      3) Scrollbar Customization (Webkit-based)
    ---------------------------------------------*/
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-light);
    }
    ::-webkit-scrollbar-thumb {
        background-color: var(--border-color);
        border-radius: var(--radius-base);
    }
    ::-webkit-scrollbar-thumb:hover {
        background-color: var(--secondary-color);
    }

    /*---------------------------------------------
      4) Main Container and Sidebar
    ---------------------------------------------*/
    .main {
        background-color: var(--bg-light);
    }
    .sidebar .sidebar-content {
        background-color: var(--primary-color);
        padding-top: var(--spacing-unit);
    }
    .sidebar .sidebar-content .css-1lcbmhc { /* collapse the default padding */
        padding-top: 0px;
    }

    /* Sidebar text & inputs */
    .sidebar .streamlit-expanderHeader {
        color: var(--bg-white) !important;
        background-color: var(--secondary-color) !important;
        font-weight: 500 !important;
    }
    .sidebar .st-bf {
        background-color: var(--bg-white) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-base) !important;
        padding: 8px 12px !important;
        margin-bottom: 0.75rem !important;
    }
    .sidebar .stSelectbox [data-baseweb="select"] {
        background-color: var(--bg-white) !important;
        border-radius: var(--radius-base) !important;
    }
    .sidebar .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--accent-color) !important;
        background-color: var(--bg-white) !important;
        padding: 0.75rem !important;
        border-radius: var(--radius-base) !important;
    }
    .sidebar .stButton>button {
        background-color: var(--accent-color) !important;
        color: var(--bg-white) !important;
        border-radius: var(--radius-base) !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .sidebar .stButton>button:hover {
        background-color: darken(var(--accent-color), 10%) !important;
        transition: background-color 0.2s ease-in-out;
    }

    /*---------------------------------------------
      5) Typography (Titles, Headers, Captions)
    ---------------------------------------------*/
    .custom-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
        text-align: center !important;
        margin-bottom: 0.75rem !important;
        border-bottom: 2px solid var(--secondary-color) !important;
        padding-bottom: 0.5rem !important;
    }

    .custom-subtitle {
        font-size: 1.2rem !important;
        color: var(--subtext-color) !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
    }

    .section-header {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: var(--primary-color) !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid var(--border-color) !important;
    }

    /*---------------------------------------------
      6) Card‐Like Containers & Elevation
    ---------------------------------------------*/
    .card-container,
    .viz-container,
    .dataframe-container,
    .grid-item {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: calc(var(--spacing-unit) * 1.25);
        margin-bottom: calc(var(--spacing-unit) * 1.25);
        box-shadow: 0 2px 8px var(--shadow-light);
        border-top: 4px solid var(--secondary-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card-container:hover,
    .viz-container:hover,
    .grid-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--shadow-medium);
    }

    /*---------------------------------------------
      7) Buttons (Primary, Secondary)
    ---------------------------------------------*/
    .custom-button {
        background-color: var(--secondary-color);
        color: var(--bg-white);
        padding: 0.75rem 1.5rem;
        border-radius: var(--radius-base);
        font-weight: 500;
        border: none;
        cursor: pointer;
        transition: background-color 0.2s ease, box-shadow 0.2s ease;
        text-align: center;
        margin: 0.5rem 0;
        display: inline-block;
    }
    .custom-button:hover {
        background-color: var(--primary-color);
        box-shadow: 0 4px 12px var(--shadow-light);
    }

    .secondary-button {
        background-color: var(--bg-white);
        color: var(--primary-color);
        border: 1px solid var(--primary-color);
        padding: 0.65rem 1.35rem;
        border-radius: var(--radius-base);
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s ease;
        margin: 0.5rem 0;
        display: inline-block;
    }
    .secondary-button:hover {
        background-color: var(--bg-light);
    }

    /*---------------------------------------------
      8) Info/Warning/Success Boxes
    ---------------------------------------------*/
    .info-box {
        background-color: #E4EBF5;
        border-left: 4px solid var(--secondary-color);
        padding: 0.75rem 1rem;
        border-radius: 0  var(--radius-base) var(--radius-base) 0;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid var(--warning-color);
        padding: 0.75rem 1rem;
        border-radius: 0  var(--radius-base) var(--radius-base) 0;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid var(--accent-color);
        padding: 0.75rem 1rem;
        border-radius: 0  var(--radius-base) var(--radius-base) 0;
        margin-bottom: 1rem;
    }
    .critical-box {
        background-color: #FFEBEE;
        border-left: 4px solid var(--danger-color);
        padding: 0.75rem 1rem;
        border-radius: 0  var(--radius-base) var(--radius-base) 0;
        margin-bottom: 1rem;
    }

    /*---------------------------------------------
      9) Custom Metric Cards
    ---------------------------------------------*/
    .custom-metric {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px var(--shadow-light);
        transition: transform 0.2s ease;
    }
    .custom-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow-light);
    }
    .metric-value {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: var(--primary-color) !important;
    }
    .metric-label {
        font-size: 0.875rem !important;
        color: var(--subtext-color) !important;
        margin-top: 0.25rem !important;
    }

    /*---------------------------------------------
      10) Styled Tables (DataFrames)
    ---------------------------------------------*/
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9rem;
        font-family: var(--font-family-base);
        min-width: 400px;
        box-shadow: 0 0 20px var(--shadow-light);
        border-radius: var(--radius-base);
        overflow: hidden;
    }
    .styled-table thead tr {
        background-color: var(--primary-color);
        color: var(--bg-white);
        text-align: left;
    }
    .styled-table th,
    .styled-table td {
        padding: 0.75rem 1rem;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid var(--border-color);
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #F3F8FA;
    }
    .styled-table tbody tr:hover {
        background-color: #E4EBF5;
    }

    /*---------------------------------------------
      11) Tabs (override Streamlit default)
    ---------------------------------------------*/
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #f1f1f1;
        border-radius: var(--radius-base) var(--radius-base) 0 0;
        padding: 0 1rem;
        font-weight: 500;
        color: var(--text-color);
        transition: background-color 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: var(--bg-white);
    }

    /*---------------------------------------------
      12) Progress Step Indicator
    ---------------------------------------------*/
    .step-progress {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        position: relative;
    }
    .step-progress::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--border-color);
        transform: translateY(-50%);
        z-index: 1;
    }
    .step {
        width: 32px;
        height: 32px;
        background-color: var(--bg-white);
        border: 2px solid var(--border-color);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: var(--subtext-color);
        z-index: 2;
        position: relative;
        transition: background-color 0.2s, border-color 0.2s, color 0.2s;
    }
    .step.active {
        background-color: var(--secondary-color);
        border-color: var(--secondary-color);
        color: var(--bg-white);
    }
    .step.completed {
        background-color: var(--accent-color);
        border-color: var(--accent-color);
        color: var(--bg-white);
    }
    .step-label {
        position: absolute;
        top: 40px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.75rem;
        width: 100px;
        text-align: center;
        color: var(--subtext-color);
    }

    /*---------------------------------------------
      13) Status Badges
    ---------------------------------------------*/
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-align: center;
        margin-right: 0.25rem;
    }
    .status-badge.pass {
        background-color: #E8F5E9;
        color: var(--accent-color);
        border: 1px solid var(--accent-color);
    }
    .status-badge.caution {
        background-color: #FFF3E0;
        color: var(--warning-color);
        border: 1px solid var(--warning-color);
    }
    .status-badge.fail {
        background-color: #FFEBEE;
        color: var(--danger-color);
        border: 1px solid var(--danger-color);
    }

    /*---------------------------------------------
      14) Tooltips
    ---------------------------------------------*/
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltip-text {
        visibility: hidden;
        width: 220px;
        background-color: var(--text-color);
        color: var(--bg-white);
        text-align: center;
        border-radius: var(--radius-base);
        padding: 0.5rem;
        position: absolute;
        z-index: 10;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s ease;
        font-size: 0.75rem;
        line-height: 1.4;
        box-shadow: 0 2px 8px var(--shadow-light);
    }
    .tooltip .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: var(--text-color) transparent transparent transparent;
    }
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /*---------------------------------------------
      15) Chart Titles & Descriptions
    ---------------------------------------------*/
    .chart-title {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--primary-color) !important;
        margin-bottom: 0.5rem !important;
        text-align: center !important;
    }
    .chart-description {
        font-size: 0.875rem !important;
        color: var(--subtext-color) !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }

    /*---------------------------------------------
      16) Grid / Dashboard Layouts
    ---------------------------------------------*/
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 1.25rem 0;
    }
    .grid-item {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: 1rem;
        box-shadow: 0 2px 8px var(--shadow-light);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .grid-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--shadow-medium);
    }

    /*---------------------------------------------
      17) Defect Metrics Container
    ---------------------------------------------*/
    .defect-metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1.25rem 0;
    }
    .defect-metric {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px var(--shadow-light);
        flex: 1 1 150px;
        border-bottom: 4px solid var(--secondary-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .defect-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--shadow-medium);
    }
    .defect-metric.critical {
        border-bottom-color: var(--danger-color);
    }
    .defect-metric.warning {
        border-bottom-color: var(--warning-color);
    }
    .defect-metric.good {
        border-bottom-color: var(--accent-color);
    }

    /*---------------------------------------------
      18) Integrity Score
    ---------------------------------------------*/
    .integrity-score-container {
        text-align: center;
        padding: 1.25rem 0;
        position: relative;
    }
    .integrity-score {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
        margin: 0.5rem 0 !important;
    }
    .integrity-label {
        font-size: 1rem !important;
        color: var(--subtext-color) !important;
        margin-bottom: 0.25rem !important;
    }

    /*---------------------------------------------
      19) Responsive Breakpoints
    ---------------------------------------------*/
    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
        .defect-metrics-container {
            flex-direction: column;
        }
        .step-label {
            display: none;  /* hide step labels on small screens */
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)