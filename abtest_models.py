import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np

# ==========================================
# 1. Setup & Load Data
# ==========================================
mlflow.set_tracking_uri("http://127.0.0.1:8080")
client = MlflowClient()

# Model name from Part 1
MODEL_NAME = "loan_default_model"
EXPERIMENT_NAME = "Loan_Default_Prediction"  # Use the same experiment name from training
mlflow.set_experiment(EXPERIMENT_NAME)

print("Loading A/B Test Data...")
# Load A/B test data with ground truth labels
data_path = 'data/fintech_abtest_1000_with_GT.csv' 
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    exit()

# Preprocessing
# WARNING: In production, you should load the same encoder used during training
# For this exam, we're refitting the encoder assuming the data has the same categories
categorical_cols = ['home_ownership', 'purpose', 'state']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        # Convert to string first to ensure consistent sorting
        df[col] = le.fit_transform(df[col].astype(str))

X_ab = df.drop(columns=['default', 'customer_id'], errors='ignore')
y_ab = df['default']

# ==========================================
# 2. Load Models & Evaluate Performance
# ==========================================
# Specify the versions to compare (check MLflow UI for correct version numbers)
version_a = "1"
version_b = "2"

def evaluate_model(version):
    """
    Load a model version and evaluate it on the A/B test data.
    
    Args:
        version: Model version number to evaluate
        
    Returns:
        dict: Dictionary containing metrics and predictions
    """
    print(f"Loading model version {version}...")
    model_uri = f"models:/{MODEL_NAME}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Generate predictions
    y_pred = model.predict(X_ab)
    
    # Calculate metrics
    acc = accuracy_score(y_ab, y_pred)
    precision = precision_score(y_ab, y_pred, average='binary')
    recall = recall_score(y_ab, y_pred, average='binary')
    f1 = f1_score(y_ab, y_pred, average='binary')
    
    # Create correctness array (1=correct, 0=incorrect) for statistical testing
    correctness = (y_pred == y_ab).astype(int)
    
    return {
        "version": version,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correctness": correctness,
        "predictions": y_pred
    }

# Evaluate both models
print("\n--- Evaluating Models ---")
result_a = evaluate_model(version_a)
result_b = evaluate_model(version_b)

# Display comparison results
print(f"\nModel V{result_a['version']} -> Acc: {result_a['accuracy']:.4f}, F1: {result_a['f1']:.4f}")
print(f"Model V{result_b['version']} -> Acc: {result_b['accuracy']:.4f}, F1: {result_b['f1']:.4f}")

# Compare predictions between the two models
diff_count = np.sum(result_a['predictions'] != result_b['predictions'])
total_count = len(y_ab)
print(f"\n[Comparison] The two models predicted differently on {diff_count} out of {total_count} samples.")

# Log comparison results to MLflow
# Log metrics (precision, recall, f1) for both versions
with mlflow.start_run(run_name="AB_Test_Comparison_Run"):
    # Log all metrics for both versions
    metrics_list = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics_list:
        mlflow.log_metric(f"v{version_a}_{metric}", result_a[metric])
        mlflow.log_metric(f"v{version_b}_{metric}", result_b[metric])
        
    mlflow.log_param("compared_version_a", version_a)
    mlflow.log_param("compared_version_b", version_b)
    print("Metrics logged to MLflow.")

# ==========================================
# 3. Statistical Test (A/B Testing)
# ==========================================
print("\n--- Performing Statistical Test (Paired T-test) ---")
# Paired T-test on correctness arrays to determine statistical significance
t_stat, p_value = stats.ttest_rel(result_a['correctness'], result_b['correctness'])

print(f"P-value: {p_value:.5f}")
alpha = 0.05

if p_value < alpha:
    print("Result: Statistically Significant Difference! (Reject H0)")
else:
    print("Result: No Significant Difference. (Fail to reject H0)")

# ==========================================
# 4. Select Winner & Set Production Alias
# ==========================================
# Determine winner based on accuracy
if result_a['accuracy'] > result_b['accuracy']:
    winner_version = result_a['version']
    print(f"\nWinner is Version {winner_version} (Higher Accuracy)")
elif result_b['accuracy'] > result_a['accuracy']:
    winner_version = result_b['version']
    print(f"\nWinner is Version {winner_version} (Higher Accuracy)")
else:
    # If tied, select the latest version
    winner_version = str(max(int(version_a), int(version_b)))
    print(f"\nTie! Selecting Version {winner_version} (Latest version) as winner.")

# Set "Production" alias to the winning model
print(f"Assigning alias 'Production' to Version {winner_version}...")
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Production",
    version=winner_version
)

print("Part 2 Completed successfully.")