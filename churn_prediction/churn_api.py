from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Customer Retention API",
    description="An ML-powered API that predicts the probability of a customer canceling their subscription.",
    version="1.0.0"
)

# --- LOAD ML ARTIFACTS ---
try:
    model = joblib.load("xgboost_churn_model.pkl")
    expected_features = joblib.load("model_features.pkl")
except FileNotFoundError:
    raise RuntimeError("Model files not found. Please run Day 2's training script first.")

# --- DEFINE INPUT SCHEMA ---
# Pydantic ensures the incoming JSON matches exactly what we expect.
class CustomerProfile(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure_months: int = 5
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    MonthlyCharges: float = 85.50
    TotalCharges: float = 427.50

# --- CORE ENDPOINT ---
@app.post("/predict")
def predict_churn(customer: CustomerProfile):
    try:
        # 1. Convert incoming JSON payload to a Pandas DataFrame
        df = pd.DataFrame([customer.dict()])
        
        # 2. Re-apply Day 1 Data Preprocessing (Encoding)
        binary_cols = ['Partner', 'Dependents', 'PaperlessBilling']
        for col in binary_cols:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        
        # Manually One-Hot Encode the Contract column to match training
        df['Contract_One year'] = 1 if customer.Contract == 'One year' else 0
        df['Contract_Two year'] = 1 if customer.Contract == 'Two year' else 0
        df.drop('Contract', axis=1, inplace=True)
        
        # 3. Ensure columns align perfectly with the trained model
        # If any columns are missing, this will fill them with 0 to prevent crashes
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_features]
        
        # 4. Generate Predictions
        # .predict() gives 0 or 1. .predict_proba() gives the exact percentage (e.g., 0.85)
        churn_class = int(model.predict(df)[0])
        churn_probability = float(model.predict_proba(df)[0][1])
        
        # 5. Formulate Business Response
        risk_level = "High" if churn_probability >= 0.5 else "Low"
        recommended_action = "Trigger 20% Retention Discount" if risk_level == "High" else "No action needed"
        
        return {
            "status": "success",
            "prediction": {
                "will_churn": bool(churn_class),
                "churn_probability_score": round(churn_probability, 3),
                "risk_level": risk_level
            },
            "business_logic": {
                "recommended_action": recommended_action
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# --- RUN SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)