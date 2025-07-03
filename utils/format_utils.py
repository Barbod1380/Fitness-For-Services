"""
Utility functions for data formatting and conversion.
"""
import pandas as pd
import numpy as np

def float_to_clock(time_float):
    """
    Convert a float to a clock time string (HH:MM format).
    
    Parameters:
    - time_float: Float time value
    
    Returns:
    - String in HH:MM format, or None if input is NaN
    """
    if pd.isna(time_float):
        return None  # or return "NaN" or ""

    total_minutes = time_float * 24 * 60
    hours = int(total_minutes // 60)
    minutes = int(round(total_minutes % 60))
    return f"{hours:02d}:{minutes:02d}"


def parse_clock(clock_str):
    """
    Parse clock string format (e.g. "5:30" or "5:30:45") to decimal hours (e.g. 5.5).
    Now handles both HH:MM and HH:MM:SS formats.
    
    Parameters:
    - clock_str: String in HH:MM or HH:MM:SS format
    
    Returns:
    - Float representing hours (e.g. 5.5 for 5:30)
    """
    try:
        # Remove any whitespace
        clock_str = str(clock_str).strip()
        
        # Split by colon
        parts = clock_str.split(":")
        
        if len(parts) == 2:
            # HH:MM format
            hours, minutes = map(int, parts)
            return hours + minutes / 60
        elif len(parts) == 3:
            # HH:MM:SS format - ignore seconds
            hours, minutes, _ = map(int, parts)
            return hours + minutes / 60
        else:
            return np.nan
    except Exception:
        return np.nan


def clean_clock_format(clock_value):
    """
    Clean clock format by removing seconds if present.
    Handles various input types and formats.
    
    Parameters:
    - clock_value: Clock value in various formats
    
    Returns:
    - String in HH:MM format or None
    """
    if pd.isna(clock_value):
        return None
    
    # Convert to string
    clock_str = str(clock_value).strip()
    
    # If it's already in HH:MM format, return as is
    if ':' in clock_str:
        parts = clock_str.split(':')
        if len(parts) == 2:
            # Already in HH:MM format
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                return f"{hours:02d}:{minutes:02d}"
            except ValueError:
                return None
        elif len(parts) == 3:
            # HH:MM:SS format - remove seconds
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                return f"{hours:02d}:{minutes:02d}"
            except ValueError:
                return None
    
    # If it's a numeric value, try to convert it
    try:
        float_val = float(clock_str)
        return float_to_clock(float_val)
    except ValueError:
        return None


def decimal_to_clock_str(decimal_hours):
    """
    Convert decimal hours to clock format string.
    Example: 5.9 → "5:54"
    
    Parameters:
    - decimal_hours: Clock position in decimal format
    
    Returns:
    - String in clock format "H:MM"
    """
    if pd.isna(decimal_hours):
        return "Unknown"
    
    # Ensure the value is between 1 and 12
    if decimal_hours < 1:
        decimal_hours += 12
    elif decimal_hours > 12:
        decimal_hours = decimal_hours % 12
        if decimal_hours == 0:
            decimal_hours = 12
    
    hours = int(decimal_hours)
    minutes = int((decimal_hours - hours) * 60)
    
    return f"{hours}:{minutes:02d}"


def standardize_surface_location(value):
    """
    Standardize different surface location values to INT/NON-INT format.
    
    This function handles various common formats for internal/external surface location:
    - Internal: INT, I, INTERNAL, INSIDE, YES, INTERNE, 1, TRUE
    - External: NON-INT, E, EXT, EXTERNAL, OUTSIDE, NO, NON INT, EXTERNE, 0, FALSE
    
    Parameters:
    - value: The original surface location value
    
    Returns:
    - Standardized value: either "INT", "NON-INT", or None for invalid/missing values
    """
    if pd.isna(value) or value is None:
        return None
    
    # Convert to uppercase string for consistent comparison
    value_str = str(value).strip().upper()
    
    # Remove common punctuation and extra spaces
    value_str = value_str.replace('-', '').replace('_', '').replace('.', '').replace(' ', '')
    
    # Comprehensive mapping for internal defects
    internal_variants = [
        'INT', 'I', 'INTERNAL', 'INSIDE', 'INTERIOR', 'INNER',
        'YES', 'Y', 'TRUE', 'T', '1', 
        'INTERNE', 'INTERNO', 'INTERIEUR',  # Other languages
        'IN', 'INTERN'
    ]
    
    # Comprehensive mapping for external defects
    external_variants = [
        'NONINT', 'NON-INT', 'NONINT', 'E', 'EXT', 'EXTERNAL', 'EXTERIOR', 
        'OUTSIDE', 'OUTER', 'NO', 'N', 'FALSE', 'F', '0',
        'EXTERNE', 'EXTERNO', 'EXTERIEUR',  # Other languages
        'OUT', 'EXTERN'
    ]
    
    # Check for internal defects
    if value_str in internal_variants:
        return 'INT'
    
    # Check for external defects  
    elif value_str in external_variants:
        return 'NON-INT'
    
    # Try partial matching for common patterns
    elif any(variant in value_str for variant in ['INT', 'INTERN']):
        return 'INT'
    elif any(variant in value_str for variant in ['EXT', 'EXTERN', 'OUT']):
        return 'NON-INT'
    
    else:
        # For unknown values, return None to indicate need for manual review
        import warnings
        warnings.warn(f"Unknown surface location value: '{value}'. Please review and standardize manually.")
        return None