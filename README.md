# ThreatEye: An Intrusion Detection System

## Project Overview

ThreatEye is a robust Intrusion Detection System (IDS) designed to analyze network traffic data, preprocess it, and use machine learning models to detect malicious activity. The system offers functionalities to:

- Load and preprocess network traffic datasets.

- Train and evaluate multiple machine learning models.

- Use pre-trained models for anomaly detection with majority voting for robust predictions.

- Generate detailed insights into model performance and data statistics.

## Features

### Dataset Handling:

- Supports CSV file uploads.

- Preprocesses datasets by handling missing values, dropping irrelevant columns, and scaling features.

### Machine Learning Models:

- Pre-trained models: Logistic Regression, Decision Tree, Random Forest, and KNN.

- On-the-fly model training with performance evaluation.

### Class Imbalance Resolution:

- Implements SMOTE (Synthetic Minority Oversampling Technique) to balance datasets.

## How to Run the Code

Clone the repository or download the scripts.
```bash
git clone <repository-link>
cd ThreatEye
```
create a venv and activate it(Mac/linux)
```bash
python -m venv .venv
source .venv/bin/activate
```

create a venv and activate it(Windows)
```bash
python -m venv .venv
.venv/Scripts/activate
```

install the requirements provided
```bash
python install -r requirements.txt
```
Start the Streamlit application:

  For Threateye.py (Pre-trained Models):
  ```bash
  streamlit run Threateye.py
  ```
  (Optional) For threatbye.py (Training and Evaluation):
  ```bash
  streamlit run threatbye.py
  ```

Open the application in your browser using the URL provided by Streamlit.

Sample datasets are provided.Click and download them.

Use the file uploader in the UI to upload your network traffic dataset (CSV format).

## Outputs

### Model Predictions:

Predictions are displayed in the UI, along with probabilities and majority voting results.
