# Extracting Time‑Series Properties of Glucose (CGM/Insulin)
Extracts performance metrics of an Artificial Pancreas system from
sensor data.

# Technologies Used
- **Python**
- **pandas**
- **NumPy**

# Program description
This project implements a data processing and analysis workflow for continuous glucose monitoring (CGM) sensor data from Medtronic 670G artificial pancreas systems. The workflow processes 8+ months of sensor data (~55K samples) to extract clinical performance metrics and evaluate glucose control effectiveness across manual and automated insulin delivery modes.

## Key Technical Implementation:

- **Data preprocessing**: Handles missing values through linear interpolation and synchronizes asynchronous sensor timestamps
- **Feature engineering**: Creates temporal classifications (day/night periods) and operational mode assignments based on device state changes
- **Time series analysis**: Segments continuous glucose readings into daily windows and applies threshold-based categorization for clinical range classification
- **Statistical aggregation**: Computes percentage-based metrics using pandas groupby operations across multiple dimensional hierarchies (mode, day, period)
- **Multi-sensor data integration**: Processes insulin pump control signals to determine operational state transitions and align with CGM measurements


## to Run:
    ```bash
    python3 main.py

Reads *CGMData.csv* and *InsulinData.csv* from root folder; exports results to *Results.csv*