<<<<<<< HEAD
# Pipeline Inspection Data Visualization

This is a Streamlit application for visualizing pipeline inspection data. The original code has been refactored into a modular structure for better organization and maintainability.

## File Structure

- **main.py**: The entry point of the application. Contains the main Streamlit UI components and application flow.
- **data_processing.py**: Contains functions for processing and transforming the pipeline data.
- **utils.py**: Contains utility functions for data conversion, such as clock format parsing.
- **visualizations.py**: Contains the visualization creation functions for both complete pipeline and joint-specific views.

## How to Run

1. Make sure all the files are in the same directory
2. Install required packages:
   ```
   pip install streamlit pandas numpy plotly
   ```
3. Run the application:
   ```
   streamlit run main.py
   ```

## App Features

- Upload pipeline inspection CSV data
- View processed data tables (joints and defects)
- Visualize the entire pipeline with defects
- Visualize specific joints with detailed defect information

## Visualization Types

1. **Complete Pipeline View**: An unwrapped cylinder visualization showing all defects with different color modes:
   - Depth (%)
   - Surface Location
   - Area (mm²)

2. **Joint-by-Joint View**: Detailed visualization of defects within a specific joint

## Data Processing

The application processes raw CSV data into two main tables:
- Joints table: Contains joint information (distance, joint number, length, etc.)
- Defects table: Contains defect information with references to associated joints

## Notes

This code reorganization maintains all original functionality while improving maintainability. No functionality has been changed - only the organization of the code into separate files.
=======
# Fitness-For-Services
>>>>>>> 2a5f3b9a1b8d5833b152163627342a03f49d85cc
