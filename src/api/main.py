"""from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

from .pydantic_models import CustomerData, PredictionResponse

# === Load model ===
MODEL_NAME = "credit_scoring_model"
ALIAS = "production"

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

# === Create FastAPI app ===
app = FastAPI()

# === Root route for testing ===
@app.get("/")
def read_root():
    return {"message": "✅ Credit Scoring Model API is running!"}

# === Prediction endpoint ===
@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    probability = model.predict_proba(df)[:, 1][0]
    return PredictionResponse(risk_probability=probability)"""


from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import os
import mlflow

from .pydantic_models import CustomerData, PredictionResponse

# === Load model ===
MODEL_NAME = "credit_scoring_model"
ALIAS = "production"

mlflow.set_tracking_uri("file:/app/mlruns")
# Load feature names from training
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/model_ready_with_target.csv')
with open(FEATURES_PATH, 'r') as f:
    header = f.readline().strip().split(',')
    trained_features = header[1:-1]  # Exclude CustomerId and is_high_risk

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

# === Create FastAPI app ===
app = FastAPI()

# === Root route for testing ===
@app.get("/")
def read_root():
    return {"message": "✅ Credit Scoring Model API is running!"}

# === Prediction endpoint with preprocessing ===
@app.post("/predict")
def predict(data: CustomerData):
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # Validate TransactionStartTime
    try:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    except Exception as e:
        return {"error": f"Invalid or missing TransactionStartTime. Please provide a valid ISO datetime string. Details: {str(e)}"}

    df['Hour'] = df['TransactionStartTime'].dt.hour
    df = df.drop(columns=['TransactionStartTime'])
    
    # Apply one-hot encoding for categorical variables (placeholder)
    categorical_cols = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Scale numerical features (placeholder - use trained scaler)
    numerical_cols = ['Amount', 'Value', 'PricingStrategy', 'CountryCode', 'Hour']
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()

    # Reindex to match training features
    df = df.reindex(columns=trained_features, fill_value=0)

    # Make prediction
    probability = model.predict_proba(df)[:, 1][0]
    label = "Has probability of default" if probability >= 0.5 else "Does not have probability of default"
    return {"result": label, "risk_probability": probability}


"""from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

from .pydantic_models import CustomerData, PredictionResponse

# === Load model ===
MODEL_NAME = "credit_scoring_model"
ALIAS = "production"

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

# === Create FastAPI app ===
app = FastAPI()

# === Root route for testing ===
@app.get("/")
def read_root():
    return {"message": "✅ Credit Scoring Model API is running!"}

# === Prediction endpoint ===
@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    df = pd.DataFrame([data.features], columns=[f"feature_{i}" for i in range(51)])
    probability = model.predict_proba(df)[:, 1][0]
    return PredictionResponse(risk_probability=probability)
    """