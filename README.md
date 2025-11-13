# Loan Default Prediction - MLOps Project

This project demonstrates a complete MLOps workflow for a loan default prediction model, including model training, A/B testing, data drift monitoring, and model serving using MLflow.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Components](#components)

## 🎯 Project Overview

This project implements a machine learning system for predicting loan defaults with the following features:

- **Part 1**: Train and compare two models (Logistic Regression vs Random Forest)
- **Part 2**: A/B testing between model versions with statistical validation
- **Part 3**: Data drift monitoring using Evidently
- **Part 4**: Model serving via MLflow REST API

## 🔧 Prerequisites

- Python 3.8 or higher
- Git (for version control)
- Internet connection (for downloading packages)

## 🚀 Setup Instructions

### 1. Clone the Repository (if applicable)

```bash
git clone https://github.com/wavekanit/CPE393-MLOps-Practical-Exam-2.git
cd CPE393-MLOps-Practical-Exam-2
```

### 2. Create Virtual Environment

Create a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt indicating the virtual environment is active.

### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install:
- MLflow (≥2.12.2) - For experiment tracking and model registry
- scikit-learn (≥1.3.0) - For machine learning models
- pandas (≥2.0.3) - For data manipulation
- numpy (≥1.24.0) - For numerical operations
- matplotlib (≥3.7.0) - For visualizations
- evidently (0.4.33) - For data drift monitoring

### 4. Start MLflow Tracking Server

Open a **separate terminal window** and start the MLflow tracking server:

```bash
# Activate virtual environment first
source venv/bin/activate

# Start MLflow server on port 8080
mlflow server --host 127.0.0.1 --port 8080
```

Keep this terminal running. The MLflow UI will be available at: http://127.0.0.1:8080

## 📁 Project Structure

```
q1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── train_loan_default_model.py       # Part 1: Model training
├── abtest_models.py                  # Part 2: A/B testing
├── monitor_drift.py                  # Part 3: Drift monitoring
├── prediction_service.py             # Part 4: Test predictions
├── data/
│   ├── fintech_train_1000.csv        # Training dataset
│   ├── fintech_abtest_1000_with_GT.csv  # A/B test dataset
│   └── fintech_new_1000.csv          # New data for drift monitoring
├── mlruns/                           # MLflow experiment tracking
└── mlartifacts/                      # MLflow model artifacts
```

## 📖 Usage Guide

Follow these steps in order to complete the full MLOps workflow:

### Part 1: Train Models

Train and compare two models, then register the best one:

```bash
python train_loan_default_model.py
```

**What this does:**
- Loads training data from `data/fintech_train_1000.csv`
- Encodes categorical features
- Trains two models: Logistic Regression and Random Forest
- Logs metrics (accuracy, precision, recall, F1) to MLflow
- Registers the best performing model as `loan_default_model`

**Expected Output:**
- Model runs visible in MLflow UI
- Registered model in MLflow Model Registry
- Console output showing training results

### Part 2: A/B Testing

Compare two model versions using A/B testing:

```bash
python abtest_models.py
```

**What this does:**
- Loads A/B test data with ground truth labels
- Evaluates both model versions (version 1 and 2)
- Performs paired t-test for statistical significance
- Logs comparison metrics to MLflow
- Assigns "Production" alias to the winning model

**Expected Output:**
- Comparison metrics for both versions
- Statistical test results (p-value)
- Production alias set to best model
- Comparison run logged in MLflow

### Part 3: Monitor Data Drift

Detect data drift between training and new production data:

```bash
python monitor_drift.py
```

**What this does:**
- Compares reference data (training) with current data (new)
- Generates Evidently drift report (HTML)
- Detects drift in features and target variable
- Logs drift metrics and report to MLflow

**Expected Output:**
- `data_drift_report.html` saved locally
- Drift metrics logged to MLflow
- Console output showing number of drifted columns

**View the report:**
- Open `data_drift_report.html` in your browser, OR
- View it in MLflow UI under the run artifacts

### Part 4: Model Serving

#### 4.1 Start Model Server

Open a **new terminal window** and start the model serving endpoint:

```bash
# Activate virtual environment
source venv/bin/activate

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080

# Serve the production model on port 5002
mlflow models serve \
  -m "models:/loan_default_model@Production" \
  -p 5002 \
  --no-conda
```

Keep this terminal running. The model API will be available at: http://127.0.0.1:5002

#### 4.2 Test Predictions

In your **original terminal**, send a test prediction request:

```bash
python prediction_service.py
```

**What this does:**
- Sends sample customer data to the model server
- Receives prediction (0 = No Default, 1 = Default)
- Displays the result

**Expected Output:**
```
Status Code: 200
Prediction Result: {'predictions': [0]}
Customer Default Prediction: No (0)
```

## 🔍 Components

### 1. `train_loan_default_model.py`
- Trains Logistic Regression and Random Forest models
- Logs all metrics and parameters to MLflow
- Registers the best model in MLflow Model Registry

### 2. `abtest_models.py`
- Loads two model versions from MLflow
- Evaluates both on A/B test dataset
- Performs statistical significance testing
- Sets Production alias to winning model

### 3. `monitor_drift.py`
- Uses Evidently library for drift detection
- Generates comprehensive HTML report
- Logs drift metrics to MLflow
- Monitors both feature and target drift

### 4. `prediction_service.py`
- Client script to test model serving
- Sends HTTP POST requests to model endpoint
- Demonstrates real-time inference

## 🎓 MLflow UI Navigation

Access the MLflow UI at http://127.0.0.1:8080

### View Experiments
1. Click on "Loan_Default_Prediction" experiment
2. Browse all runs with their metrics and parameters
3. Compare runs side-by-side

### View Models
1. Click "Models" in the left sidebar
2. Select "loan_default_model"
3. See all registered versions
4. Check which version has "Production" alias

### View Artifacts
1. Click on any run
2. Scroll to "Artifacts" section
3. View drift reports, model files, etc.

## 🛠️ Troubleshooting

### Issue: MLflow connection errors
**Solution:** Ensure MLflow server is running on port 8080
```bash
mlflow server --host 127.0.0.1 --port 8080
```

### Issue: Model serving port already in use
**Solution:** Change the port number or kill the existing process
```bash
# Use a different port
mlflow models serve -m "models:/loan_default_model@Production" -p 5003 --no-conda
```

### Issue: Module not found errors
**Solution:** Ensure virtual environment is activated and dependencies are installed
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: File not found errors
**Solution:** Ensure you're running scripts from the project root directory and data files exist

## 📊 Expected Results

### Model Performance
- **Random Forest**: Typically achieves higher accuracy (~75-85%)
- **Logistic Regression**: Baseline performance (~70-80%)

### A/B Testing
- Statistical comparison between model versions
- Clear winner selection based on accuracy
- Production alias assignment

### Drift Detection
- Identifies columns with significant distribution changes
- Provides detailed statistical tests
- Visual reports for analysis

## 📝 Notes

- Always activate the virtual environment before running any scripts
- Keep MLflow server running throughout the workflow
- Model serving requires a separate terminal window
- Check MLflow UI frequently to monitor progress
- Data files must be in the `data/` directory

## 🤝 Contributing

To contribute or modify this project:

1. Create a new branch for your changes
2. Make modifications
3. Test thoroughly
4. Commit with descriptive messages
5. Push and create a pull request

## 📄 License

This project is for educational purposes as part of an MLOps practical exam.

---

**Last Updated:** November 2025  
**Author:** Kanit Bunyinkgool
