import streamlit as st

def load_css():
    """Apply professional enterprise-grade CSS styling for Pipeline FFS application."""
    css = """
    <style>
    /*---------------------------------------------
      1) Professional Typography & Font System
    ---------------------------------------------*/
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Roboto:wght@300;400;500;700&display=swap');
    
    /*---------------------------------------------
      2) Professional Color System - Oil & Gas Industry
    ---------------------------------------------*/
    :root {
        /* Primary Brand Colors */
        --primary-navy: #0A1628;           /* Deep professional navy */
        --primary-blue: #1E40AF;           /* Corporate blue */
        --secondary-steel: #374151;        /* Steel gray */
        --accent-orange: #EA580C;          /* Energy sector orange */
        --accent-gold: #D97706;            /* Pipeline gold */
        
        /* Status Colors - Industry Standard */
        --status-safe: #059669;            /* Safe operation green */
        --status-caution: #D97706;         /* Caution amber */
        --status-warning: #DC2626;         /* Warning red */
        --status-critical: #7F1D1D;        /* Critical dark red */
        --status-info: #1D4ED8;            /* Information blue */
        
        /* Neutral Palette */
        --neutral-50: #F8FAFC;
        --neutral-100: #F1F5F9;
        --neutral-200: #E2E8F0;
        --neutral-300: #CBD5E1;
        --neutral-400: #94A3B8;
        --neutral-500: #64748B;
        --neutral-600: #475569;
        --neutral-700: #334155;
        --neutral-800: #1E293B;
        --neutral-900: #0F172A;
        
        /* Typography */
        --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Monaco', 'Consolas', monospace;
        --font-display: 'Roboto', sans-serif;
        
        /* Spacing & Layout */
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
        --space-2xl: 3rem;
        
        /* Border Radius */
        --radius-sm: 4px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        
        /* Gradients */
        --gradient-primary: linear-gradient(135deg, var(--primary-navy) 0%, var(--primary-blue) 100%);
        --gradient-steel: linear-gradient(135deg, var(--neutral-700) 0%, var(--neutral-600) 100%);
        --gradient-surface: linear-gradient(145deg, var(--neutral-50) 0%, var(--neutral-100) 100%);
    }

    /*---------------------------------------------
      3) Global Base Styles
    ---------------------------------------------*/
    html, body, [class*="css"] {
        font-family: var(--font-primary) !important;
        background-color: var(--neutral-100) !important;
        color: var(--neutral-800) !important;
        line-height: 1.6;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--neutral-200);
        border-radius: var(--radius-sm);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--neutral-400), var(--neutral-500));
        border-radius: var(--radius-sm);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--neutral-500), var(--neutral-600));
    }

    /*---------------------------------------------
      4) Professional Header & Navigation
    ---------------------------------------------*/
    .main-header {
        background: var(--gradient-primary);
        color: white;
        padding: var(--space-lg) 0;
        margin-bottom: var(--space-xl);
        box-shadow: var(--shadow-lg);
        border-bottom: 3px solid var(--accent-orange);
    }

    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: var(--space-md);
    }

    .custom-title {
        font-family: var(--font-display) !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .custom-subtitle {
        font-size: 1.1rem !important;
        color: var(--neutral-200) !important;
        margin-top: var(--space-sm) !important;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /*---------------------------------------------
      5) Professional Sidebar Navigation
    ---------------------------------------------*/
    .sidebar .sidebar-content {
        background: var(--gradient-steel) !important;
        border-right: 3px solid var(--accent-orange);
        box-shadow: var(--shadow-xl);
    }

    /* Navigation Items */
    .nav-item {
        display: block !important;
        padding: var(--space-md) var(--space-lg) !important;
        margin: var(--space-xs) 0 !important;
        border-radius: var(--radius-md) !important;
        background: transparent !important;
        color: var(--neutral-200) !important;
        text-decoration: none !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        border: 2px solid transparent !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    .nav-item:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-color: var(--accent-orange) !important;
        transform: translateX(4px) !important;
    }

    .nav-item.active {
        background: var(--accent-orange) !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: var(--shadow-md) !important;
    }

    .nav-item.disabled {
        background: var(--neutral-600) !important;
        color: var(--neutral-400) !important;
        cursor: not-allowed !important;
        opacity: 0.6 !important;
    }

    /* Sidebar Sections */
    .sidebar-section-header {
        color: var(--accent-gold) !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin: var(--space-lg) 0 var(--space-md) 0 !important;
        padding-bottom: var(--space-sm) !important;
        border-bottom: 2px solid var(--accent-gold) !important;
    }

    /* File Uploader Styling */
    .sidebar .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--accent-orange) !important;
        background: rgba(234, 88, 12, 0.05) !important;
        border-radius: var(--radius-md) !important;
        padding: var(--space-lg) !important;
        transition: all 0.3s ease !important;
    }

    .sidebar .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-gold) !important;
        background: rgba(234, 88, 12, 0.1) !important;
    }

    /* Sidebar Form Elements */
    .sidebar .stSelectbox [data-baseweb="select"] {
        background: white !important;
        border-radius: var(--radius-md) !important;
        border: 2px solid var(--neutral-300) !important;
    }

    .sidebar .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border-radius: var(--radius-md) !important;
        padding: var(--space-md) var(--space-lg) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        box-shadow: var(--shadow-md) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .sidebar .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--accent-orange) 100%) !important;
    }

    /*---------------------------------------------
      6) Professional Content Cards & Containers
    ---------------------------------------------*/
    .card-container,
    .viz-container,
    .dataframe-container {
        background: white;
        border-radius: var(--radius-lg);
        padding: var(--space-xl);
        margin-bottom: var(--space-xl);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--neutral-200);
        border-left: 4px solid var(--primary-blue);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .card-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-primary);
    }

    .card-container:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-left-color: var(--accent-orange);
    }

    /* Section Headers */
    .section-header {
        font-family: var(--font-display) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--primary-navy) !important;
        margin-bottom: var(--space-lg) !important;
        padding-bottom: var(--space-md) !important;
        border-bottom: 3px solid var(--neutral-200) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
    }

    .section-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: var(--gradient-primary);
    }

    /*---------------------------------------------
      7) Professional Metrics & KPIs
    ---------------------------------------------*/
    .custom-metric {
        background: white;
        border-radius: var(--radius-lg);
        padding: var(--space-xl);
        text-align: center;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--neutral-200);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .custom-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }

    .custom-metric:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }

    .metric-value {
        font-family: var(--font-mono) !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: var(--primary-navy) !important;
        line-height: 1.2;
        margin-bottom: var(--space-sm) !important;
    }

    .metric-label {
        font-size: 0.9rem !important;
        color: var(--neutral-600) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Status-Specific Metrics */
    .metric-safe { border-left: 6px solid var(--status-safe); }
    .metric-caution { border-left: 6px solid var(--status-caution); }
    .metric-warning { border-left: 6px solid var(--status-warning); }
    .metric-critical { border-left: 6px solid var(--status-critical); }

    /*---------------------------------------------
      8) Professional Data Tables
    ---------------------------------------------*/
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: var(--space-lg) 0;
        font-size: 0.9rem;
        border-radius: var(--radius-lg);
        overflow: hidden;
        box-shadow: var(--shadow-md);
        background: white;
    }

    .styled-table thead tr {
        background: var(--gradient-primary);
        color: white;
        text-align: left;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .styled-table th {
        padding: var(--space-lg);
        font-size: 0.85rem;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    .styled-table td {
        padding: var(--space-md) var(--space-lg);
        border-bottom: 1px solid var(--neutral-200);
        border-right: 1px solid var(--neutral-100);
    }

    .styled-table tbody tr {
        transition: background-color 0.2s ease;
    }

    .styled-table tbody tr:nth-of-type(even) {
        background-color: var(--neutral-50);
    }

    .styled-table tbody tr:hover {
        background-color: rgba(30, 64, 175, 0.05);
        transform: scale(1.001);
    }

    /*---------------------------------------------
      9) Professional Status Indicators
    ---------------------------------------------*/
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: var(--space-xs) var(--space-md);
        border-radius: var(--radius-xl);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: var(--shadow-sm);
    }

    .status-badge.safe {
        background: var(--status-safe);
        color: white;
    }

    .status-badge.caution {
        background: var(--status-caution);
        color: white;
    }

    .status-badge.warning {
        background: var(--status-warning);
        color: white;
    }

    .status-badge.critical {
        background: var(--status-critical);
        color: white;
    }

    /*---------------------------------------------
      10) Professional Alert Boxes
    ---------------------------------------------*/
    .alert-box {
        padding: var(--space-lg);
        border-radius: var(--radius-md);
        margin: var(--space-lg) 0;
        border-left: 4px solid;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }

    .alert-info {
        background: rgba(29, 78, 216, 0.05);
        border-left-color: var(--status-info);
        color: var(--status-info);
    }

    .alert-success {
        background: rgba(5, 150, 105, 0.05);
        border-left-color: var(--status-safe);
        color: var(--status-safe);
    }

    .alert-warning {
        background: rgba(217, 119, 6, 0.05);
        border-left-color: var(--status-caution);
        color: var(--status-caution);
    }

    .alert-error {
        background: rgba(220, 38, 38, 0.05);
        border-left-color: var(--status-warning);
        color: var(--status-warning);
    }

    /*---------------------------------------------
      11) Professional Tabs
    ---------------------------------------------*/
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--neutral-100);
        border-radius: var(--radius-md);
        padding: var(--space-xs);
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        padding: var(--space-md) var(--space-lg);
        font-weight: 600;
        color: var(--neutral-600);
        border-radius: var(--radius-sm);
        margin: 0;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        color: var(--primary-blue) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /*---------------------------------------------
      12) Professional Buttons
    ---------------------------------------------*/
    .custom-button, .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        padding: var(--space-md) var(--space-xl) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        border: none !important;
        box-shadow: var(--shadow-md) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    .custom-button:hover, .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--accent-orange) 100%) !important;
    }

    .custom-button:active, .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /*---------------------------------------------
      13) Progress Indicators
    ---------------------------------------------*/
    .step-progress {
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        margin: var(--space-xl) 0;
        padding: var(--space-lg) 0;
    }

    .step-progress::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 10%;
        right: 10%;
        height: 3px;
        background: var(--neutral-300);
        transform: translateY(-50%);
        z-index: 1;
        border-radius: var(--radius-sm);
    }

    .step {
        width: 48px;
        height: 48px;
        background: white;
        border: 3px solid var(--neutral-300);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: var(--neutral-500);
        z-index: 2;
        position: relative;
        font-size: 1.1rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }

    .step.active {
        background: var(--accent-orange);
        border-color: var(--accent-orange);
        color: white;
        transform: scale(1.1);
        box-shadow: var(--shadow-lg);
    }

    .step.completed {
        background: var(--status-safe);
        border-color: var(--status-safe);
        color: white;
    }

    /*---------------------------------------------
      14) Professional Charts & Visualizations
    ---------------------------------------------*/
    .chart-container {
        background: white;
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--neutral-200);
        margin: var(--space-lg) 0;
    }

    .chart-title {
        font-family: var(--font-display) !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: var(--primary-navy) !important;
        margin-bottom: var(--space-md) !important;
        text-align: left !important;
    }

    /*---------------------------------------------
      15) Responsive Design
    ---------------------------------------------*/
    @media (max-width: 1200px) {
        .custom-title { font-size: 2rem !important; }
        .card-container { padding: var(--space-lg); }
    }

    @media (max-width: 768px) {
        .custom-title { font-size: 1.6rem !important; }
        .section-header { font-size: 1.2rem !important; }
        .metric-value { font-size: 2rem !important; }
        .card-container { padding: var(--space-md); }
    }

    /*---------------------------------------------
      16) Loading States & Animations
    ---------------------------------------------*/
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animate-in {
        animation: slideIn 0.6s ease-out;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def apply_navigation_styles():
    """Apply professional navigation-specific styling."""
    nav_css = """
    <style>
    /* Enhanced Navigation Specific Styles */
    .nav-container {
        background: var(--gradient-steel);
        border-radius: var(--radius-md);
        padding: var(--space-sm);
        margin-bottom: var(--space-lg);
        box-shadow: var(--shadow-md);
    }
    
    .breadcrumb {
        display: flex;
        align-items: center;
        padding: var(--space-md) 0;
        font-size: 0.9rem;
        color: var(--neutral-600);
    }
    
    .breadcrumb-item {
        color: var(--neutral-600);
        text-decoration: none;
        transition: color 0.2s ease;
    }
    
    .breadcrumb-item:hover {
        color: var(--primary-blue);
    }
    
    .breadcrumb-separator {
        margin: 0 var(--space-sm);
        color: var(--neutral-400);
        font-weight: bold;
    }
    
    /* Status indicators for loaded datasets */
    .dataset-status {
        display: inline-flex;
        align-items: center;
        background: rgba(5, 150, 105, 0.1);
        color: var(--status-safe);
        padding: var(--space-xs) var(--space-md);
        border-radius: var(--radius-xl);
        font-size: 0.8rem;
        font-weight: 600;
        margin: var(--space-xs) 0;
        border: 1px solid rgba(5, 150, 105, 0.2);
    }
    
    .dataset-status::before {
        content: '✓';
        margin-right: var(--space-xs);
        font-weight: bold;
    }
    </style>
    """
    st.markdown(nav_css, unsafe_allow_html=True)


def get_status_indicator(status_type):
    """Generate HTML for professional status indicators."""
    status_map = {
        'safe': {'class': 'safe', 'icon': '✓', 'text': 'SAFE'},
        'caution': {'class': 'caution', 'icon': '⚠', 'text': 'CAUTION'},
        'warning': {'class': 'warning', 'icon': '⚠', 'text': 'WARNING'},
        'critical': {'class': 'critical', 'icon': '✕', 'text': 'CRITICAL'}
    }
    
    if status_type in status_map:
        s = status_map[status_type]
        return f'<span class="status-badge {s["class"]}">{s["icon"]} {s["text"]}</span>'
    else:
        return f'<span class="status-badge">{status_type.upper()}</span>'


def professional_metric_card(title, value, status=None, description=None):
    """Generate professional metric card HTML."""
    status_class = f"metric-{status}" if status else ""
    desc_html = f'<div class="metric-description">{description}</div>' if description else ""
    
    return f"""
    <div class="custom-metric {status_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {desc_html}
    </div>
    """


def professional_alert(message, alert_type="info"):
    """Generate professional alert box HTML."""
    return f'<div class="alert-box alert-{alert_type}">{message}</div>'