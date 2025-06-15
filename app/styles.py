import streamlit as st

def load_css():
    """Apply industrial-grade CSS styling for pipeline analysis app."""
    css = """
    <style>
    /*---------------------------------------------
      1) Import Google Font: "Inter" + "Roboto Condensed"
    ---------------------------------------------*/
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Condensed:wght@400;700&display=swap');
    
    /*---------------------------------------------
      2) Industrial Color Scheme
    ---------------------------------------------*/
    :root {
        --font-family-base: 'Inter', sans-serif;
        --font-family-headings: 'Roboto Condensed', sans-serif;
        --primary-color: #0d1b2a;       /* Deep industrial navy */
        --secondary-color: #415a77;     /* Steel blue */
        --accent-color: #e63946;        /* Alert red (pipeline integrity) */
        --success-color: #2a9d8f;       /* Industrial teal */
        --warning-color: #e9c46a;       /* Safety amber */
        --danger-color: #9d0208;        /* Critical burgundy */
        --info-color: #1d3557;          /* Technical blue */
        --bg-light: #f0f2f5;            /* Neutral light gray */
        --bg-white: #FFFFFF;
        --text-color: #1d3557;          /* Industrial dark blue */
        --subtext-color: #6c757d;       /* Technical gray */
        --border-color: #ced4da;        /* Metallic border */
        --shadow-light: rgba(0, 0, 0, 0.05);
        --shadow-medium: rgba(0, 0, 0, 0.1);
        --shadow-strong: rgba(0, 0, 0, 0.15);
        --radius-base: 4px;             /* Sharp industrial edges */
        --spacing-unit: 1rem;
        --pipeline-gradient: linear-gradient(90deg, #0d1b2a 0%, #1d3557 100%);
    }
    
    html, body, [class*="css"]  {
        font-family: var(--font-family-base) !important;
        color: var(--text-color) !important;
        background-color: var(--bg-light) !important;
    }

    /*---------------------------------------------
      3) Industrial Scrollbar
    ---------------------------------------------*/
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #e9ecef;
    }
    ::-webkit-scrollbar-thumb {
        background-color: var(--secondary-color);
        border-radius: 3px;
    }

    /*---------------------------------------------
      4) Technical Sidebar
    ---------------------------------------------*/
    .main {
        background-color: var(--bg-light);
    }
    .sidebar .sidebar-content {
        background: var(--pipeline-gradient);
        padding-top: var(--spacing-unit);
        border-right: 1px solid var(--border-color);
    }
    .sidebar .sidebar-content .css-1lcbmhc {
        padding-top: 0px;
    }

    /* Sidebar elements */
    .sidebar .streamlit-expanderHeader {
        font-family: var(--font-family-headings) !important;
        color: var(--bg-white) !important;
        background-color: rgba(0,0,0,0.2) !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .sidebar .st-bf {
        background-color: rgba(255,255,255,0.9) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-base) !important;
        padding: 10px !important;
    }
    .sidebar .stSelectbox [data-baseweb="select"] {
        background-color: var(--bg-white) !important;
    }
    .sidebar .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--accent-color) !important;
        background-color: rgba(230, 57, 70, 0.05) !important;
        padding: 0.75rem !important;
    }
    .sidebar .stButton>button {
        background: var(--pipeline-gradient) !important;
        color: var(--bg-white) !important;
        border-radius: var(--radius-base) !important;
        padding: 0.7rem 1.3rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        border: none !important;
        transition: all 0.3s ease;
    }
    .sidebar .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /*---------------------------------------------
      5) Industrial Typography
    ---------------------------------------------*/
    .custom-title {
        font-family: var(--font-family-headings) !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 3px solid var(--secondary-color) !important;
    }

    .custom-subtitle {
        font-size: 1.1rem !important;
        color: var(--subtext-color) !important;
        margin-bottom: 2rem !important;
    }

    .section-header {
        font-family: var(--font-family-headings) !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: var(--primary-color) !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid var(--border-color) !important;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    /*---------------------------------------------
      6) Technical Data Cards
    ---------------------------------------------*/
    .card-container,
    .viz-container,
    .dataframe-container,
    .grid-item {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 6px var(--shadow-light);
        border-left: 4px solid var(--secondary-color);
        transition: all 0.3s ease;
    }
    .card-container:hover,
    .viz-container:hover,
    .grid-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px var(--shadow-medium);
    }

    /*---------------------------------------------
      7) Industrial Buttons
    ---------------------------------------------*/
    .custom-button {
        background: var(--pipeline-gradient);
        color: var(--bg-white);
        padding: 0.8rem 1.8rem;
        border-radius: var(--radius-base);
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        display: inline-block;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--shadow-light);
    }

    /*---------------------------------------------
      8) Pipeline Status Boxes
    ---------------------------------------------*/
    .info-box {
        background-color: #e7f5ff;
        border-left: 4px solid var(--info-color);
        padding: 1rem;
        border-radius: 0 var(--radius-base) var(--radius-base) 0;
    }
    .warning-box {
        background-color: #fff8e6;
        border-left: 4px solid var(--warning-color);
        padding: 1rem;
    }
    .success-box {
        background-color: #e6f7ee;
        border-left: 4px solid var(--success-color);
        padding: 1rem;
    }
    .critical-box {
        background-color: #ffe6e9;
        border-left: 4px solid var(--danger-color);
        padding: 1rem;
    }

    /*---------------------------------------------
      9) Technical Metric Cards
    ---------------------------------------------*/
    .custom-metric {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px var(--shadow-light);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    .custom-metric:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px var(--shadow-medium);
    }
    .metric-value {
        font-family: var(--font-family-headings) !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.9rem !important;
        color: var(--subtext-color) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /*---------------------------------------------
      10) Technical Data Tables
    ---------------------------------------------*/
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        font-size: 0.9rem;
        font-family: var(--font-family-base);
        box-shadow: 0 0 10px var(--shadow-light);
        border-radius: var(--radius-base);
        overflow: hidden;
    }
    .styled-table thead tr {
        background: var(--pipeline-gradient);
        color: var(--bg-white);
        text-align: left;
        font-weight: 600;
        text-transform: uppercase;
    }
    .styled-table th {
        padding: 1rem;
        letter-spacing: 0.3px;
    }
    .styled-table td {
        padding: 0.9rem 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    .styled-table tbody tr {
        transition: background-color 0.2s;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f8f9fa;
    }
    .styled-table tbody tr:hover {
        background-color: #e9ecef;
    }

    /*---------------------------------------------
      11) Technical Tabs
    ---------------------------------------------*/
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: var(--subtext-color);
        border: none;
        border-radius: 0;
        margin: 0;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--accent-color) !important;
        box-shadow: inset 0 -3px 0 var(--accent-color);
    }

    /*---------------------------------------------
      12) Pipeline Progress Indicator
    ---------------------------------------------*/
    .step-progress {
        display: flex;
        justify-content: space-between;
        position: relative;
        margin: 2rem 0;
    }
    .step-progress::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--border-color);
        transform: translateY(-50%);
        z-index: 1;
    }
    .step {
        width: 36px;
        height: 36px;
        background-color: var(--bg-white);
        border: 2px solid var(--border-color);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: var(--subtext-color);
        z-index: 2;
        position: relative;
        transition: all 0.3s ease;
    }
    .step.active {
        background-color: var(--warning-color);
        border-color: var(--warning-color);
        color: var(--primary-color);
        transform: scale(1.1);
    }
    .step.completed {
        background-color: var(--success-color);
        border-color: var(--success-color);
        color: var(--bg-white);
    }
    .step-label {
        position: absolute;
        top: 42px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        color: var(--text-color);
        white-space: nowrap;
    }

    /*---------------------------------------------
      13) Pipeline Status Badges
    ---------------------------------------------*/
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .status-badge.pass {
        background-color: rgba(42, 157, 143, 0.15);
        color: var(--success-color);
    }
    .status-badge.caution {
        background-color: rgba(233, 196, 106, 0.15);
        color: var(--warning-color);
    }
    .status-badge.fail {
        background-color: rgba(157, 2, 8, 0.15);
        color: var(--danger-color);
    }

    /*---------------------------------------------
      14) Technical Tooltips
    ---------------------------------------------*/
    .tooltip {
        position: relative;
        border-bottom: 1px dashed var(--subtext-color);
        cursor: help;
    }
    .tooltip .tooltip-text {
        visibility: hidden;
        width: 260px;
        background-color: var(--primary-color);
        color: var(--bg-white);
        text-align: left;
        border-radius: var(--radius-base);
        padding: 1rem;
        position: absolute;
        z-index: 100;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
        line-height: 1.5;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .tooltip .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: var(--primary-color) transparent transparent transparent;
    }
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /*---------------------------------------------
      15) Technical Chart Elements
    ---------------------------------------------*/
    .chart-title {
        font-family: var(--font-family-headings) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--primary-color) !important;
        text-align: left !important;
        margin-bottom: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .chart-description {
        font-size: 0.9rem !important;
        color: var(--subtext-color) !important;
        text-align: left !important;
        margin-bottom: 1.5rem !important;
        line-height: 1.6;
    }

    /*---------------------------------------------
      16) Dashboard Grid Layout
    ---------------------------------------------*/
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.8rem;
        margin: 1.5rem 0;
    }
    .grid-item {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: 1.5rem;
        box-shadow: 0 3px 8px var(--shadow-light);
        border-left: 4px solid var(--info-color);
        transition: all 0.3s ease;
    }

    /*---------------------------------------------
      17) Pipeline Defect Metrics
    ---------------------------------------------*/
    .defect-metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .defect-metric {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 3px 8px var(--shadow-light);
        border-top: 4px solid var(--secondary-color);
        transition: all 0.3s ease;
    }
    .defect-metric.critical {
        border-top-color: var(--danger-color);
    }
    .defect-metric.warning {
        border-top-color: var(--warning-color);
    }
    .defect-metric.good {
        border-top-color: var(--success-color);
    }

    /*---------------------------------------------
      18) Pipeline Integrity Score
    ---------------------------------------------*/
    .integrity-score-container {
        background-color: var(--bg-white);
        border-radius: var(--radius-base);
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px var(--shadow-light);
        border: 1px solid var(--border-color);
    }
    .integrity-score {
        font-family: var(--font-family-headings) !important;
        font-size: 4rem !important;
        font-weight: 700 !important;
        margin: 1rem 0 !important;
        background: var(--pipeline-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .integrity-label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--text-color) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .integrity-description {
        font-size: 0.95rem !important;
        color: var(--subtext-color) !important;
        max-width: 600px;
        margin: 1rem auto !important;
        line-height: 1.6;
    }

    /*---------------------------------------------
      19) Technical Responsive Design
    ---------------------------------------------*/
    @media (max-width: 992px) {
        .grid-container {
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
        .integrity-score {
            font-size: 3.5rem !important;
        }
    }
    
    @media (max-width: 768px) {
        .custom-title {
            font-size: 1.8rem !important;
        }
        .grid-container {
            grid-template-columns: 1fr;
        }
        .defect-metrics-container {
            grid-template-columns: 1fr 1fr;
        }
        .step-label {
            font-size: 0.7rem;
        }
    }
    
    @media (max-width: 576px) {
        .defect-metrics-container {
            grid-template-columns: 1fr;
        }
        .step-label {
            display: none;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)