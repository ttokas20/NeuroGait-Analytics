NeuroGait Analytics: Real-Time IMU Gait Analysis Dashboard

Originally developed to advance neuro-rehabilitation for patients with motor disabilities, specifically Parkinson's disease, this platform handles the entire data-to-visualization pipeline. It transforms raw biomechanical signals from wearable bands into actionable clinical metrics, significantly reducing manual analysis time and improving patient outcomes.

## Key Features

Synchronized Multi-Modal Playback: Multithreaded GUI (built with `tkinter` and `OpenCV`) that synchronously plays clinical video alongside dual-leg IMU data plots.
Personalized Step Modeling: Dynamically calculates expected step length and speed based on user demographics (height, age, gender) and clinical status.(Parkinson's diagnosis)

Advanced Signal Processing:
    * Timestamp correction and sensor fusion.
    * Peak detection via `scipy.signal` for accurate step counting from Gyroscope-Z data.
    * Roll angle calculation using vectorized operations on 3-axis accelerometer data.
    * Integrated Gyro-Y tracking for rotational tracking.
    
Clinical Export: One-click generation of comprehensive, presentation-ready gait analysis plots.


## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/NeuroGait-Analytics.git](https://github.com/yourusername/NeuroGait-Analytics.git)
   cd NeuroGait-Analytics
Install the required dependencies:

Bash
pip install pandas numpy scipy matplotlib opencv-python Pillow
📁 Input Data Requirements
To utilize the dashboard, you need synchronized video and IMU data files.

1. Video File: Standard formats accepted (.mp4, .avi, .mov).
2. IMU Data (Left and Right Leg): Two separate .csv files. The data must contain the following columns (case-sensitive) to fully utilize the angle and speed calculations:

TimeStamp: Datetime string (e.g., YYYY-MM-DD HH:MM:SS.ms)

AX, AY, AZ: Accelerometer data (X, Y, Z axes)

GY: Gyroscope Y-axis data

GZ (or a variation like Gyro_Z): Gyroscope Z-axis data used for step peak detection

Note: The application automatically appends -L and -R suffixes internally when you load the respective CSVs.


* Usage
  Run the application:

  Bash
  python zebbs.py
  
* Configure User: Click User Data to set the patient's height, age, gender, and Parkinson's diagnosis. This calibrates the PersonalizedStepModel, so it accurately adjusts to their demographics.

* Load Media: Click Upload Video to load the visual reference.

* Load Data: Click Upload Left Leg Data and Upload Right Leg Data to ingest the IMU CSV files. The app will automatically merge them using nearest-neighbor timestamp matching (50ms tolerance).

* Select Metrics: Click Select Column Groups to choose which sensor streams, calculated speeds, or derived angles (Roll, Theta Y) to display.

* Analyze: Use the Play/Pause controls to review the synchronized data. Click Save Graph to export the entire session's time-series data as an image.


This project also supported a broader comparative review of 9 public IMU datasets (including HuGaDB), evaluating sampling rates, metadata integrity, and sensor placement. Additionally, the underlying algorithms were utilized to process 31 Functional Reach Sway Test videos to validate core balance metrics.
