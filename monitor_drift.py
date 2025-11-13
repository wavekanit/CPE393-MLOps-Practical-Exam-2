import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

# ==========================================
# 1. Setup & Load Data
# ==========================================
# Configure MLflow (use same tracking URI as training)
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Loan_Default_Prediction")

print("Loading data for monitoring...")
# Reference Data: Training dataset (historical baseline)
ref_data = pd.read_csv('data/fintech_train_1000.csv')

# Current Data: New production dataset (current period)
try:
    curr_data = pd.read_csv('data/fintech_new_1000.csv')
except FileNotFoundError:
    print("Error: 'fintech_new_1000.csv' not found. Make sure it's in the data/ folder.")
    exit()

# ==========================================
# 2. Configure Column Mapping
# ==========================================
# Define column types for Evidently to calculate drift correctly
# Categorical features: home_ownership, purpose, state, term_months (typically 36/60 months)
# Numerical features: age, annual_income, loan_amount, interest_rate, dti, etc.

column_mapping = ColumnMapping()

# Specify target column (if available in current data)
if 'default' in curr_data.columns:
    column_mapping.target = 'default'
else:
    column_mapping.target = None  # No target drift if labels are not available

# Explicitly define categorical features
column_mapping.categorical_features = ['home_ownership', 'purpose', 'state', 'term_months']
# Remaining numerical columns will be treated as numerical features automatically
# Note: We should exclude non-feature columns like customer_id

# Remove ID column as it should not be used for drift detection (IDs are unique)
ref_data = ref_data.drop(columns=['customer_id'], errors='ignore')
curr_data = curr_data.drop(columns=['customer_id'], errors='ignore')

# ==========================================
# 3. Generate Evidently Drift Report
# ==========================================
print("Generating Drift Report...")

# Create report with two components as required:
# 1. Data Drift (check feature distributions)
# 2. Target Drift (check label distribution)
drift_report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])

drift_report.run(
    reference_data=ref_data,
    current_data=curr_data,
    column_mapping=column_mapping
)

# Save report as HTML
report_path = "data_drift_report.html"
drift_report.save_html(report_path)
print(f"Report saved locally to {report_path}")

# ==========================================
# 4. Log Report to MLflow
# ==========================================
print("Logging report to MLflow...")

with mlflow.start_run(run_name="Evidently_Drift_Monitoring"):
    # Log the HTML report file as an MLflow artifact
    mlflow.log_artifact(report_path)
    
    # Log summary metrics (optional but recommended for tracking)
    # Extract drift statistics from the report
    result = drift_report.as_dict()
    drift_share = result['metrics'][0]['result']['drift_share']
    number_of_drifted_columns = result['metrics'][0]['result']['number_of_drifted_columns']
    
    mlflow.log_metric("drift_share", drift_share)
    mlflow.log_metric("drifted_columns_count", number_of_drifted_columns)
    
    print(f"Logged artifact to MLflow Run ID: {mlflow.active_run().info.run_id}")
    print(f"Drift Summary: {number_of_drifted_columns} columns drifted.")

print("Part 3 Completed successfully.")