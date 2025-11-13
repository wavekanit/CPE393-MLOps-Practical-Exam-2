# CPE393 MLOps: Practical Exam 2 - Loan Default Prediction

This repository contains the solution for the **CPE393 Machine Learning Operations** practical examination. The project implements an end-to-end MLOps pipeline for a Fintech Loan Default Prediction scenario, utilizing **MLflow** for experiment tracking/model registry and **Evidently AI** for data drift monitoring.

## 📂 Project Structure

```text
.
├── data/
│   ├── fintech_train_1000.csv          # Historical training data
│   ├── fintech_abtest_1000_with_GT.csv # A/B testing data with Ground Truth
│   └── fintech_new_1000.csv            # New production data for monitoring
├── mlartifacts/                        # MLflow artifacts storage
├── mlruns/                             # MLflow tracking logs
├── abtest_models.py                    # Part 2: A/B Testing & Evaluation script
├── monitor_drift.py                    # Part 3: Data Drift Monitoring script
├── prediction_service.py               # Part 4: Client script for Model Serving
├── train_loan_default_model.py         # Part 1: Training & Experiment Tracking script
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## 🚀 Prerequisites & Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/wavekanit/CPE393-MLOps-Practical-Exam-2.git
   cd CPE393-MLOps-Practical-Exam-2
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the MLflow Tracking Server:**
   Open a new terminal and run:
   ```bash
   mlflow ui --host 127.0.0.1 --port 8080
   ```
   Access the UI at: [http://127.0.0.1:8080](http://127.0.0.1:8080)

---

## 🛠️ Usage Instructions

### Part 1: Experiment Tracking & Training
Train two models (Logistic Regression & Random Forest), log metrics to MLflow, and register the best performing model.
```bash
python train_loan_default_model.py
```

### Part 2: Model Evaluation & A/B Testing
Load model versions, evaluate on the A/B test dataset, perform statistical testing (Paired T-test), and assign the "Production" alias.
```bash
python abtest_models.py
```

### Part 3: Model Monitoring (Evidently AI)
Generate a Data Drift report comparing training data vs. new data and log the HTML report to MLflow.
```bash
python monitor_drift.py
```
*Note: Check the MLflow UI "Artifacts" tab to view the generated `data_drift_report.html`.*

### Part 4: Model Serving & Prediction

**Step 1: Serve the Model**
Open a **NEW terminal** (do not close MLflow UI) and run:
```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=[http://127.0.0.1:8080](http://127.0.0.1:8080)  # macOS/Linux
# set MLFLOW_TRACKING_URI=[http://127.0.0.1:8080](http://127.0.0.1:8080)   # Windows CMD
# $env:MLFLOW_TRACKING_URI="[http://127.0.0.1:8080](http://127.0.0.1:8080)" # Windows PowerShell

# Start serving the Production model on port 5002
mlflow models serve -m "models:/loan_default_model@Production" -p 5002 --no-conda
```
*Wait until you see "Listening at: http://127.0.0.1:5002"*

**Step 2: Send Prediction Request**
In your original terminal, run the client script to send sample data:
```bash
python prediction_service.py
```

---

## 📊 Deliverables Checklist

- [x] **Part 1:** Models trained and logged in MLflow with metrics/params.
- [x] **Part 2:** Statistical A/B test performed; "Production" alias assigned.
- [x] **Part 3:** Evidently AI Data Drift Report generated and logged.
- [x] **Part 4:** Model served via REST API and prediction verified.

---
**Author:** Kanit Bunyinkgool 65070501064
**Course:** CPE393 Machine Learning Operations