# Fitness-for-Service Analysis Platform for Pipelines

This is a comprehensive Streamlit application designed for the Fitness-for-Service (FFS) analysis of pipeline integrity. The platform allows for in-depth analysis of pipeline status, defect analysis, and visualization of inspection data.

## Key Features

*   **Data Upload & Processing**: Upload pipeline inspection data in CSV format. The application processes the data to identify joints and defects.
*   **Pipeline Visualization**: Visualize the entire pipeline as an unwrapped cylinder, with defects highlighted. Color-code defects by depth, surface location, or area.
*   **Joint-Specific View**: Isolate and visualize specific joints for a more detailed inspection of defects.
*   **Defect Analysis**: Perform defect analysis, including clustering of defects based on industry standards (e.g., DNV).
*   **Multi-Year Comparison**: Compare pipeline data from different years to track defect growth and changes in pipeline integrity.
*   **FFS Calculations**: Conduct Fitness-for-Service calculations to assess the integrity of the pipeline.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```
3.  Install the required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

Once the dependencies are installed, you can run the Streamlit application with the following command:

```bash
streamlit run main.py
```

The application will open in your default web browser.

## Project Structure

The project is organized into a modular structure for better maintainability and scalability:

-   `main.py`: The entry point for the Streamlit application.
-   `requirements.txt`: A list of the Python packages required to run the project.
-   `app/`: Contains the main application logic, including the Streamlit UI, styling, and page routing.
-   `analysis/`: Houses the modules for performing FFS calculations and defect analysis.
-   `core/`: Contains the core data processing pipeline, including defect matching and clustering algorithms.
-   `utils/`: A package for utility functions, such as data formatting and validation.
-   `visualization/`: Contains modules for creating the various data visualizations used in the application.
-   `assets/`: Contains static assets like images and logos.
