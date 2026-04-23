📉 Customer Retention API (Churn Predictor)

🚀 The Goal

Customer acquisition is 5x more expensive than retention. This project is an end-to-end Machine Learning pipeline that predicts subscriber churn before it happens, allowing Customer Success teams to trigger proactive discounts and save revenue.

🛠️ Tech Stack

Data Engineering: Pandas, NumPy

Machine Learning: XGBoost, Scikit-Learn

Backend API: FastAPI, Uvicorn, Pydantic

Frontend UI: Streamlit

📂 Project Structure

churn_eda.py: Data ingestion, cleaning, imputation, and exploratory data analysis (EDA).

churn_train.py: XGBoost model training, hyperparameter tuning, and feature importance analysis. Outputs .pkl artifacts.

churn_api.py: The production REST API that serves the model predictions.

churn_frontend.py: The Agent Dashboard built with Streamlit.

🏃‍♂️ How to Run the System

This project uses a microservices architecture. You must run the backend and the frontend simultaneously.

1. Start the AI Backend (Terminal 1)

uvicorn churn_api:app --reload


The API will run locally at http://127.0.0.1:8000
Interactive API Docs: http://127.0.0.1:8000/docs

2. Start the Agent Dashboard (Terminal 2)

streamlit run churn_frontend.py


📊 Business Impact

The model not only predicts the binary outcome (Will Churn / Will Stay), but also calculates the exact probability score. The frontend integrates this probability with the customer's billing data to display Value at Risk and Net Revenue Preserved, bridging the gap between Data Science and ROI.