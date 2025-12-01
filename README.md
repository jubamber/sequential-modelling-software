# Time Series Analysis Workstation

[![Chinese Version](https://img.shields.io/badge/Language-中文-blue.svg)](./README_zh.md)

Github Repository: https://github.com/jubamber/sequential-modelling-software

## Project Introduction

**Time Series Analysis Workstation** is a lightweight, local desktop application developed in Python. This project aims to provide learners, data scientists, and business personnel with an intuitive and easy-to-use platform for exploring, training, and evaluating various **Time Series Forecasting Models**. Currently, it supports **univariate** time series data analysis.

The backend is integrated with **FastAPI**, the frontend uses **Gradio** to build the interactive interface, and the underlying algorithms combine **TensorFlow/Keras** (for Deep Learning models) and **Sktime** (for Statistical models). The system has been specifically optimized so that most models and datasets can run smoothly in a standard CPU environment without requiring complex GPU configurations.



## Core Features

*   **Multi-Model Support**:
    *   **Deep Learning**: LSTM (Long Short-Term Memory), MLP (Multilayer Perceptron).
    *   **Statistical Models**: ARIMA, SARIMA, Exponential Smoothing.
*   **Flexible Data Sources**: Supports built-in classic datasets (e.g., AirPassengers, Sunspots) and allows users to upload local CSV data for analysis.
*   **Automatic Parameter Search**: Supports AutoARIMA, which automatically derives the optimal p, d, q, and seasonal parameters.
*   **Interactive Visualization**: Integrated with Plotly to provide scalable, interactive curves for training set fitting, test set predictions, and future trend forecasting.
*   **Model Persistence**: Supports automatic saving of trained models, loading historical models, and backtracking parameter configurations (Metadata).
*   **Future Forecasting**: In addition to validating the test set, it supports generating inference data for a specified number of future steps.



## Environment & Installation

It is recommended to use **Conda** or **Python venv** to create an independent virtual environment to run this **Gradio** application.

### 1. System Requirements

*   **Python Version**: 3.12.10 (Recommended)
*   **Operating System**: Windows / macOS / Linux
*   **Hardware**: Standard CPU supported (Project enables OneDNN optimization by default); GPU is optional (requires manual code configuration).

### 2. Installation Steps

1.  **Clone or Download Project Code**:
    Place all source code files in the project root directory.

2.  **Create and Activate Virtual Environment**:

    ```bash
    # Using conda
    conda create -n ts_station python=3.12.10
    conda activate ts_station
    
    # Or using venv
    python -m venv venv
    # Activate on Windows
    venv\Scripts\activate
    # Activate on Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Ensure the `requirements.txt` file is in the directory and run the following command:

    ```bash
    pip install -r requirements.txt
    ```



## Quick Start

1.  **Launch the Application**:
    Run the entry script in the project root directory:

    ```bash
    python main.py
    ```

2.  **Access the Interface**:
    The program will automatically clear port 7860 upon startup. When the console shows `Running on local URL: http://127.0.0.1:7860`, open your browser and visit that address.

3.  **Stop the Application**:
    Press `Ctrl+C` in the console. To completely shut down the service, you can close the terminal window or call the `/shutdown` API.



## Data Format Specifications

This system has strict formatting requirements for local data uploaded by users to ensure accurate parsing.

*   **File Format**: CSV (.csv)
*   **Column Structure**: Preferably a two-column format; other formats may not parse correctly.
    *   **1st Column (Index)**: Time data (Date/Time). Standard datetime format is recommended (e.g., `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`).
    *   **2nd Column (Value)**: Target observation value. Must be a pure numeric type.
*   **Header**: The system automatically identifies if a header exists. If the first row is not numeric, it will be treated as a header and automatically excluded.

**CSV Example**:

```csv
Date,Value
2023-01-01,100.5
2023-01-02,102.3
2023-01-03,99.8
...
```



## User Guide

The interface is mainly divided into the **Visualization Window** (Left/Top) and the **Control Panel** (Right/Bottom).

### 1. Data Selection

*   **Preset Datasets**: Select classic cases like "Sine Wave" or "AirPassengers" from the dropdown menu to experience the app immediately.
*   **Local Data**: Check "Upload Local Data (CSV)" and drag or click to upload a compliant file. The system will automatically parse and update the preview graph.

### 2. Model Configuration

*   **Deep Learning (LSTM/MLP)**:
    *   **Epochs**: Number of training iterations.
    *   **Batch Size**: Size of the training batch.
    *   **Look Back**: Time window size (i.e., how many past time steps are used to predict the next one).
    *   **Training Set Ratio**: The ratio for splitting the training and testing sets.
*   **Statistical Models (ARIMA/SARIMA)**:
    *   **Auto Parameter**: Check "Use Auto (S)ARIMA Parameter Inference" to let the algorithm automatically find optimal parameters (takes longer but is more accurate).
    *   **Manual Parameters**: Uncheck the auto option to manually adjust p, d, q (ARIMA) and P, D, Q, s (SARIMA seasonal parameters).

### 3. Run & Evaluate

Click the **Start Training & Evaluation** button:

1.  **Training**: The interface shows a progress bar (Epoch progress for Keras models, Fitting status for Sktime models).
2.  **Evaluation**: Calculates MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) on the test set.
3.  **Plotting**:
    *   **Gray Line**: Ground Truth (Actual Data).
    *   **Blue Line**: Training Set Fitting.
    *   **Red Line**: Test Set Prediction (Recursive prediction, using only predicted data).
    *   **Green Dashed Line**: Future Forecast Data (if enabled).

### 4. Stop Current Task

Some tasks running only on CPU may take a long time (e.g., AutoARIMA searching for optimal parameters with large S values on a large dataset). Click the **Stop Current Task** button on the interface to halt the current calculation.

### 5. Model Loading & Reuse

The system automatically saves trained models to the `saved_models` directory.

*   Check "Use Saved Model".
*   Select a historical model file from the dropdown box.
*   The system will automatically lock parameters related to the model structure (e.g., Look Back, Model Type), but allows modification of "Extra Forecast Steps".
*   Click Run to perform inference using the old model.
*   **Note**: It is best to use the saved model on the same dataset used during training. Changing datasets while using a saved model is likely to cause parameter conflicts and errors.

### 6. Clear Cache

Click the **Clear Saved Models** button to delete all `.keras`, `.pkl`, and `.json` files in the `saved_models` directory with one click, freeing up disk space.



## Note on Release Files

We used PyInstaller to package the original Python files and dependencies into a folder format (Dir), embedded it into an Electron app, and packaged it as a Portable version and an NSIS-based Installer version.

*   **Sequential Modelling App 1.0.0.exe**: Portable version. Double-click to use. Convenient, but due to the large package size, the loading time is long every time it starts.
*   **Sequential Modelling App Setup 1.0.0.exe**: Installer version. Requires installation. Loads faster on startup than the portable version.

Both packaging methods are relatively immature and **are not highly recommended**. Detailed packaging code is not provided. We recommend setting up the corresponding Python environment yourself and running the source code directly.



## Project Structure

```text
Project_Root/
├── main.py             # Entry Layer: App startup, port management, Gradio mounting
├── config.py           # Config Layer: Env variables, global constants (CPU optimization)
├── interface.py        # Interaction Layer: Gradio UI layout, event callbacks, logic control
├── model_engine.py     # Model Layer: LSTM/ARIMA definitions, training, prediction logic
├── data_processor.py   # Data Layer: Local/Preset data reading, cleaning, normalization
├── visualizer.py       # Visualization Layer: Plotly-based chart generation
├── utils.py            # Utils Layer: Port detection, path handling, system operations
├── datasets            # Source files for all built-in datasets
└── saved_models/       # Directory for storing models generated at runtime
```



## Built-in Datasets: References and Intro

### Arctic Oscillation (AO)

[CPC - Teleconnections: Arctic Oscillation](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml)

The daily Arctic Oscillation (AO) index is constructed by projecting the daily (00Z) 1000mb height anomalies north of 20°N onto the loading pattern of the AO. Please note that year-round monthly mean anomaly data have been used to obtain the loading pattern of the AO. Since the AO has the largest variability during the cold season, the loading pattern primarily captures characteristics of the cold season AO.



### AirPassengers

[✈️Air Passengers Dataset✈️](https://www.kaggle.com/datasets/brmil07/air-passengers-dataset)

This dataset provides monthly totals of US airline passengers from 1949 to 1960. This dataset is taken from an inbuilt dataset of R called AirPassengers. Analysts typically use various statistical techniques, such as decomposition, smoothing, and forecasting models, to analyze patterns, trends, and seasonal fluctuations in the data. Due to its historical nature and consistent time granularity, the AirPassengers dataset is a valuable resource for researchers, practitioners, and students in statistics, econometrics, and transportation planning.



### Sunspots

[Sunspots](https://www.kaggle.com/datasets/robervalt/sunspots)

Sunspots are temporary phenomena on the Sun's photosphere that appear as spots darker than the surrounding areas. They are regions of reduced surface temperature caused by concentrations of magnetic field flux that inhibit convection. Sunspots usually appear in pairs of opposite magnetic polarity. Their number varies according to the approximately 11-year solar cycle.



### Daily Minimum Temperatures in Melbourne

[Daily Minimum Temperatures in Melbourne](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne)



### CO2 Mauna Loa Weekly

[CO2 Mauna Loa Weekly](https://www.kaggle.com/datasets/dan3dewey/co2-mauna-loa-weekly)

Data from NOAA Earth System Research Laboratory (NOAA ESRL):

"These data are made freely available to the public and the scientific community in the belief that their wide dissemination will lead to greater understanding and new scientific insights. The availability of these data does not constitute publication of the data. NOAA relies on the ethics and integrity of the user to ensure that ESRL receives appropriate credit..."



## Disclaimer

This version of the software may still contain some unresolved bugs, but testing indicates they do not affect simple usage.

Parts of the code were generated with the assistance of Large Language Models (LLMs). The software icon was generated by AI. No copyright issues are involved.



## FAQ

**Q: Why does it say the port is occupied at startup?**
A: The program attempts to automatically clear port `7860` on startup. If it fails, please manually check if other Python processes or Gradio applications are using that port.

**Q: Why is the ARIMA model training speed so slow?**
A: If "Auto Parameter Inference (AutoARIMA)" is checked, the algorithm needs to iterate through various parameter combinations to find the optimal solution, which usually takes a long time. for quick verification, please uncheck it and manually specify p, d, q.

**Q: Is GPU acceleration supported?**
A: The project defaults to CPU optimization (`TF_ENABLE_ONEDNN_OPTS=1`) to ensure portability and compatibility. To use an NVIDIA GPU, please remove the relevant CPU enforcement settings in `config.py` and ensure `tensorflow-gpu` or corresponding CUDA dependencies are installed.

**Q: How does the "Future Forecast" feature work?**
A:
*   **LSTM/MLP**: Uses a recursive prediction strategy, where the prediction from the previous step is used as the input for the next step. Errors may accumulate as steps increase.
*   **ARIMA/SARIMA**: Calculates the expected values for future time steps directly based on statistical laws.

**Q: Why do I get an error when loading a saved ARIMA model?**
A: Statistical models (ARIMA/SARIMA) are very sensitive to input data length and indexing. If you load an old model using a dataset that is completely different (or has a vastly different length) from the training one, sktime may throw an "earlier to train starting point" error. It is recommended to use a data source consistent with the training data when loading models.



## License

This project is open-source under the MIT License. For more details, please refer to the [LICENSE](LICENSE) file in the project root directory.

Copyright (c) 2025 Jubamber

