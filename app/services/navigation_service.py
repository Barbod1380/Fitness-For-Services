"""
Navigation service for the Pipeline Analysis application.
"""
from app.services.state_manager import get_state, update_state
from app.views.failure_prediction import render_failure_prediction_view

# Define page navigation structure
PAGE_STRUCTURE = {
    'home': {
        'title': 'Home',
        'requires_data': False,
        'icon': '🏠'
    },
    'upload': {
        'title': 'Data Upload',
        'requires_data': False,
        'icon': '📤'
    },
    'single_analysis': {
        'title': 'Single Year Analysis',
        'requires_data': True,
        'icon': '📊'
    },
    'comparison': {
        'title': 'Multi-Year Comparison',
        'requires_data': True,
        'min_datasets': 2,
        'icon': '📈'
    },
    'corrosion_assessment': { 
        'title': 'Corrosion Assessment',
        'requires_data': True,
        'icon': '🔩'
    },
    'failure_prediction': {  # ADD THIS NEW ENTRY
        'title': 'Failure Prediction',
        'requires_data': True,
        'icon': '🔮'
    }
}

def get_current_page():
    """
    Get the current page from session state.
    
    Returns:
    - Current page name
    """
    return get_state('current_page', 'home')

def set_current_page(page_name):
    """
    Set the current page in session state.
    
    Parameters:
    - page_name: Name of the page to navigate to
    
    Returns:
    - Boolean indicating success
    """
    if page_name in PAGE_STRUCTURE:
        update_state('current_page', page_name)
        return True
    return False

def get_navigation_items():
    """
    Get a list of navigation items with availability based on current state.
    
    Returns:
    - List of dictionaries with navigation items
    """
    # Check if we have datasets
    datasets = get_state('datasets', {})
    has_data = len(datasets) > 0

    # Build navigation items
    nav_items = []
    for page_id, page_info in PAGE_STRUCTURE.items():
        # Check if page is available
        available = True
        if page_info.get('requires_data', False) and not has_data:
            available = False
        if page_info.get('min_datasets', 1) > len(datasets):
            available = False
            
        nav_items.append({
            'id': page_id,
            'title': page_info['title'],
            'icon': page_info.get('icon', '📄'),
            'available': available,
            'active': page_id == get_current_page()
        })
    
    return nav_items

def get_page_title(page_id=None):
    """
    Get the title for a specific page or the current page.
    
    Parameters:
    - page_id: Page ID or None for current page
    
    Returns:
    - Page title
    """
    if page_id is None:
        page_id = get_current_page()
        
    if page_id in PAGE_STRUCTURE:
        return PAGE_STRUCTURE[page_id]['title']
    
    return 'Unknown Page'

def get_breadcrumb_items():
    """
    Get breadcrumb items for the current page.
    
    Returns:
    - List of (label, active) tuples for breadcrumb
    """
    current_page = get_current_page()
    
    # Basic breadcrumb with Home -> Current Page
    items = [
        ('Home', current_page == 'home')
    ]
    
    # Add current page if not home
    if current_page != 'home':
        items.append((get_page_title(current_page), True))
    
    return items