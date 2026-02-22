from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Literal, Dict, Optional

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="XGBoost model for predicting customer churn probability",
    version="1.0.0"
)


MODEL_PATH = "../models/model_vfinal.json"
PREPROC_PATH = "../models/preprocessor.joblib"

try:
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or preprocessor: {e}")

print("Model and preprocessor loaded successfully.")

class CustomerFeatures(BaseModel):
    gender: Optional[Literal["Female", "Male"]] = None
    SeniorCitizen: Optional[Literal[0, 1]] = None
    Partner: Optional[Literal["Yes", "No"]] = None
    Dependents: Optional[Literal["Yes", "No"]] = None
    tenure: float
    PhoneService: Optional[Literal["Yes", "No"]] = None
    MultipleLines: Optional[Literal["Yes", "No", "No phone service"]] = None
    InternetService: Optional[Literal["DSL", "Fiber optic", "No"]] = None
    OnlineSecurity: Optional[Literal["Yes", "No", "No internet service"]] = None
    OnlineBackup: Optional[Literal["Yes", "No", "No internet service"]] = None
    DeviceProtection: Optional[Literal["Yes", "No", "No internet service"]] = None
    TechSupport: Optional[Literal["Yes", "No", "No internet service"]] = None
    StreamingTV: Optional[Literal["Yes", "No", "No internet service"]] = None
    StreamingMovies: Optional[Literal["Yes", "No", "No internet service"]] = None
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Optional[Literal["Yes", "No"]] = None
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float
    
@app.post("/predict", response_model=Dict[str, float | bool | str])
async def predict_churn(features: CustomerFeatures):
    try:
        input_dict = features.model_dump()
        
        defaults = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'No',
            'OnlineSecurity': 'No internet service',
            'OnlineBackup': 'No internet service',
            'DeviceProtection': 'No internet service',
            'TechSupport': 'No internet service',
            'StreamingTV': 'No internet service',
            'StreamingMovies': 'No internet service',
            'PaperlessBilling': 'Yes',
        }
        
        full_input = {**defaults, **input_dict}
        input_df = pd.DataFrame([full_input])
        expected_cols = preprocessor.feature_names_in_ 
        input_df = input_df[expected_cols]

        X_input = preprocessor.transform(input_df)
        dmatrix = xgb.DMatrix(X_input)
        prob = booster.predict(dmatrix)[0] 

        return {
            "churn_probability": round(float(prob), 4),
            "predicted_churn": prob > 0.5,
            "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Input validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
