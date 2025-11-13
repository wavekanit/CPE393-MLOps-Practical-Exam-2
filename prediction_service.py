import requests
import pandas as pd
import json

# ==========================================
# Configuration
# ==========================================
# URL of the MLflow model server running on Port 5002
url = "http://127.0.0.1:5002/invocations"

headers = {
    "Content-Type": "application/json",
}

# ==========================================
# Prepare Sample Data
# ==========================================
# Sample customer data for prediction
# Note: Categorical features must be encoded as integers because the model was trained with LabelEncoder
# The numeric values here are assumed encodings for demonstration purposes
sample_data = {
    "age": 29,
    "annual_income": 35000,     
    "employment_years": 3,
    "home_ownership": 3,         # Encoded categorical value
    "loan_amount": 9000,         
    "term_months": 36,
    "interest_rate": 17.3,       
    "dti": 0.58,
    "credit_score": 620,         
    "num_open_accounts": 6,
    "delinquencies_2y": 2,
    "inquiries_6m": 4,
    "purpose": 9,                # Encoded categorical value
    "state": 10                  # Encoded categorical value
}

# Convert data to MLflow-compatible format
# Must be wrapped in "dataframe_records" or "dataframe_split" key
payload = {
    "dataframe_records": [sample_data]
}

# ==========================================
# Send Prediction Request
# ==========================================
print(f"Sending request to {url}...")
print("Data:", json.dumps(payload, indent=2))

try:
    response = requests.post(url, json=payload, headers=headers)
    
    print("\n--- Response ---")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:", result)
        # Format the result for better readability (typically returns [0] or [1])
        prediction = result['predictions'][0]
        print(f"Customer Default Prediction: {'Yes (1)' if prediction == 1 else 'No (0)'}")
    else:
        print("Error:", response.text)

except Exception as e:
    print(f"Connection Failed: {e}")
    print("Did you start the 'mlflow models serve' command in another terminal?")