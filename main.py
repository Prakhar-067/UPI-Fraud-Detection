from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin

# Custom frequency encoder class for proper serialization
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_counts_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.feature_counts_[col] = X[col].value_counts().to_dict()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X[col].map(self.feature_counts_[col]).fillna(0)
        return X_transformed

# Define input data model
class Transaction(BaseModel):
    step: int
    transaction_type: str
    amount: float
    sender_id: str
    oldbalanceOrg: float
    newbalanceOrig: float
    receiver_id: str
    oldbalanceDest: float
    newbalanceDest: float
    timestamp: str
    transaction_hour: int
    transaction_day: int
    transaction_weekday: int
    location: str
    device_type: str
    risk_score: Optional[float] = None
    new_location_flag: int
    txn_count_last_hour: int
    total_amount_last_hour: float
    balance_change_ratio: float
    receiver_balance_impact: float
    night_transaction: int
    new_recipient_flag: int

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor with proper error handling
try:
    model = joblib.load("fraud_detection_xgb.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model/preprocessor: {str(e)}")

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # Convert input to DataFrame
        input_data = transaction.dict()
        input_df = pd.DataFrame([input_data])

        # Preprocess input
        processed_data = preprocessor.transform(input_df)

        # Handle potential sparse matrix output
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1]

        return {
            "is_fraud": bool(prediction[0]),
            "fraud_probability": float(probability),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}