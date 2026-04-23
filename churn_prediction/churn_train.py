import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor - Training", page_icon="🧠", layout="wide")

st.title("🧠 Customer Retention: XGBoost Training Studio")
st.markdown("**Project 1, Day 2: Machine Learning & Feature Importance**")

# --- 1. DATA PREP (Importing Day 1 Logic) ---
@st.cache_data
def load_and_prep_data():
    """Generates and cleans the data exactly as we did in Day 1."""
    # 1. Generate Raw Data
    np.random.seed(42)
    n = 2000
    tenure = np.random.randint(0, 72, n)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, n), 2)
    total_charges = tenure * monthly_charges
    total_charges = [str(val) if np.random.rand() > 0.05 else " " for val in total_charges]
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    
    churn_prob = np.zeros(n)
    churn_prob += np.where(tenure < 12, 0.4, 0.1)
    churn_prob += np.where(contracts == "Month-to-month", 0.3, 0.05)
    churn_prob += np.where(monthly_charges > 80, 0.15, 0.0)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = [1 if np.random.rand() < p else 0 for p in churn_prob]
    
    df = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure_months": tenure,
        "Contract": contracts,
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": ["Yes" if c == 1 else "No" for c in churn]
    })
    
    # 2. Clean Data (Day 1 logic)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 3. Encode Data (Day 1 logic)
    binary_cols = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    df_ml = pd.get_dummies(df, columns=['Contract'], drop_first=True)
    return df_ml

df_ml = load_and_prep_data()

# --- SIDEBAR: HYPERPARAMETERS ---
st.sidebar.header("🎛️ Model Hyperparameters")
st.sidebar.write("Tune the XGBoost algorithm.")

n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, step=50)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 10, 4)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)

# --- CORE TRAINING LOGIC ---
tab1, tab2, tab3 = st.tabs(["⚙️ Training Setup", "📊 Model Evaluation", "💡 Business Insights"])

with tab1:
    st.subheader("Data Split & Training")
    
    # 1. Define Features (X) and Target (y)
    X = df_ml.drop('Churn', axis=1)
    y = df_ml['Churn']
    
    # 2. Train/Test Split (80% Training, 20% Testing)
    test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.write(f"**Training Samples:** {len(X_train)} | **Testing Samples:** {len(X_test)}")
    st.dataframe(X_train.head(), use_container_width=True)
    
    if st.button("🚀 Train XGBoost Model", type="primary"):
        with st.spinner("Training the Gradient Boosting Algorithm..."):
            
            # Initialize XGBoost
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # Train the Model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            
            # Save Model and Metrics to Session State
            st.session_state['model'] = model
            st.session_state['metrics'] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec}
            st.session_state['X_cols'] = X.columns
            
            # Save physically to disk for Day 3 (API)
            joblib.dump(model, "xgboost_churn_model.pkl")
            # Also save feature names so our API knows what order to expect
            joblib.dump(list(X.columns), "model_features.pkl")
            
            st.success("✅ Model Trained & Saved as `xgboost_churn_model.pkl`!")

with tab2:
    if 'model' in st.session_state:
        st.subheader("Model Performance")
        st.write("How well did the AI predict churn on the unseen test data?")
        
        m = st.session_state['metrics']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['Accuracy']*100:.1f}%", help="Total correct predictions")
        c2.metric("Precision", f"{m['Precision']*100:.1f}%", help="When it predicted 'Churn', how often was it right?")
        c3.metric("Recall", f"{m['Recall']*100:.1f}%", help="Out of all ACTUAL churners, how many did it find?")
        
        if m['Recall'] < 0.6:
            st.warning("⚠️ Low Recall: The model is missing a lot of churning customers. Try increasing max_depth or n_estimators in the sidebar.")
        else:
            st.success("👍 Good Recall: The model is effectively catching at-risk customers.")

    else:
        st.info("👈 Click 'Train XGBoost Model' on the first tab to see metrics.")

with tab3:
    if 'model' in st.session_state:
        st.subheader("Feature Importance (Why are they leaving?)")
        st.write("This chart shows which data points have the biggest impact on a customer's decision to churn. This is pure Business Intelligence.")
        
        model = st.session_state['model']
        feature_names = st.session_state['X_cols']
        
        # Get importance scores
        importances = model.feature_importances_
        
        # Create a DataFrame for plotting
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=True)
        
        # Plot using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='#ff4b4b')
        ax.set_xlabel('Relative Importance')
        ax.set_title('XGBoost Feature Importance')
        
        st.pyplot(fig)
        
        st.markdown("""
        **Business Action Plan based on this data:**
        * If **Contract_Month-to-month** is at the top, Marketing should aggressively offer discounts to lock people into 1-year contracts.
        * If **tenure_months** is high, it means new customers are churning early. We need better onboarding sequences.
        """)
    else:
        st.info("👈 Train the model first to see business insights.")