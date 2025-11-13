from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# 1. Setup Paths & MLflow Configuration
# ==========================================
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"

# Set MLflow tracking URI
TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(TRACKING_URI)

# Set experiment name
EXP_NAME = "Loan_Default_Prediction"
mlflow.set_experiment(EXP_NAME)

# ==========================================
# 2. Helper Functions
# ==========================================
def load_and_prep_data():
    """
    Load data from CSV, encode categorical features, and split into train/test sets.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test (75/25 split)
    """
    print("Loading data...")
    csv_path = DATA_DIR / "fintech_train_1000.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Preprocessing: Encode categorical columns using Label Encoding
    # Convert string categorical features to numerical values for model training
    categorical_cols = ['home_ownership', 'purpose', 'state']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
    # Define features and target variable
    target = "default"
    # Drop customer_id as it's not useful for prediction
    X = df.drop(columns=[target, "customer_id"], errors='ignore')
    y = df[target]
    
    # Split data: 75% training, 25% testing
    print("Splitting data 75/25...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(model_obj, model_name, X_train, X_test, y_train, y_test):
    """
    Train a model and log all metrics and artifacts to MLflow.
    
    Args:
        model_obj: Sklearn model or pipeline to train
        model_name: Name for the MLflow run
        X_train, X_test: Feature data for training and testing
        y_train, y_test: Target labels for training and testing
        
    Returns:
        tuple: (run_id, test_accuracy)
    """
    with mlflow.start_run(run_name=model_name) as run:
        print(f"--- Starting run: {model_name} ---")
        
        # 1. Log model parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(model_obj.get_params())
        
        # 2. Train the model
        model_obj.fit(X_train, y_train)
        
        # ==========================================
        # 3. Calculate and Log Metrics
        # ==========================================
        
        # A. Training performance metrics
        y_train_pred = model_obj.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        mlflow.log_metric("train_accuracy", train_acc)
        
        # B. Testing performance metrics
        y_test_pred = model_obj.predict(X_test)
        
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='binary')
        recall = recall_score(y_test, y_test_pred, average='binary')
        f1 = f1_score(y_test, y_test_pred, average='binary')
        
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Display results
        print(f"  [Result] {model_name}")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Testing Accuracy:  {test_acc:.4f}")
        
        # 4. Log model artifact with signature and example
        signature = infer_signature(X_train, model_obj.predict(X_train))
        input_example = X_train.head(5)
        
        mlflow.sklearn.log_model(
            sk_model=model_obj,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        return run.info.run_id, test_acc

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    """Main function to train models, compare them, and register the best one."""
    # 1. Prepare data
    X_train, X_test, y_train, y_test = load_and_prep_data()
    
    # 2. Define two models for comparison
    # Model A: Logistic Regression (with StandardScaler pipeline)
    model_a = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Model B: Random Forest
    model_b = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # 3. Train and track both models
    run_id_a, acc_a = train_model(model_a, "Logistic_Regression", X_train, X_test, y_train, y_test)
    run_id_b, acc_b = train_model(model_b, "Random_Forest", X_train, X_test, y_train, y_test)
    
    # 4. Compare models and register the best one
    print("\n--- Comparing Models ---")
    best_model_name = "loan_default_model"
    
    if acc_b > acc_a:
        print(f"Random Forest Wins ({acc_b:.4f} vs {acc_a:.4f})")
        best_run_id = run_id_b
    else:
        print(f"Logistic Regression Wins ({acc_a:.4f} vs {acc_b:.4f})")
        best_run_id = run_id_a
        
    print(f"Registering model from run_id: {best_run_id} as '{best_model_name}'...")
    
    # Register the best model in MLflow Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, best_model_name)
    
    print("Part 1 Complete. Check MLflow UI.")

if __name__ == "__main__":
    main()