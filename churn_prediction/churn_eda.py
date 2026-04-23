import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor - Data Engine", page_icon="📉", layout="wide")

st.title("📉 Customer Retention: Data Engineering Pipeline")
st.markdown("**Project 1, Day 1: Data Preprocessing & EDA**")

# --- 1. DATA INGESTION (Mocking a SQL/CSV pull) ---
@st.cache_data
def generate_telco_data(n=2000):
    """Generates a realistic synthetic Telco dataset with intentional 'messiness'."""
    np.random.seed(42)
    
    tenure = np.random.randint(0, 72, n)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, n), 2)
    
    # Calculate Total Charges but randomly inject some blanks (simulating new customers or database errors)
    total_charges = tenure * monthly_charges
    total_charges = [str(val) if np.random.rand() > 0.05 else " " for val in total_charges]
    
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    
    # Create realistic churn probability logic
    churn_prob = np.zeros(n)
    churn_prob += np.where(tenure < 12, 0.4, 0.1) # New customers churn more
    churn_prob += np.where(contracts == "Month-to-month", 0.3, 0.05) # Month-to-month churns more
    churn_prob += np.where(monthly_charges > 80, 0.15, 0.0) # High bills churn more
    
    # Normalize and apply randomness
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = [1 if np.random.rand() < p else 0 for p in churn_prob]
    
    df = pd.DataFrame({
        "customerID": [f"CUST_{i:04d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure_months": tenure,
        "Contract": contracts,
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": np.random.choice(["Yes", "No"], n) # Let's make it text so we have to clean it
    })
    
    # Force the logic we built above into the text column
    df["Churn"] = ["Yes" if c == 1 else "No" for c in churn]
    return df

raw_df = generate_telco_data()

# --- APP LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📊 Raw Data", "🧹 Data Cleaning", "📈 EDA & Features"])

# --- TAB 1: RAW DATA ---
with tab1:
    st.subheader("Step 1: Data Ingestion")
    st.write(f"Loaded **{raw_df.shape[0]}** rows and **{raw_df.shape[1]}** columns.")
    st.dataframe(raw_df.head(10), use_container_width=True)
    
    st.info("Notice the 'TotalCharges' column? Sometimes it has empty spaces instead of numbers. Machine Learning models hate that. We need to clean it.")

# --- TAB 2: DATA CLEANING ---
with tab2:
    st.subheader("Step 2: Data Cleaning & Imputation")
    
    df_clean = raw_df.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Handling Invalid Datatypes**")
        st.code("df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')", language="python")
        # Convert empty strings to NaN
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        missing_count = df_clean['TotalCharges'].isnull().sum()
        st.warning(f"Found {missing_count} missing values in TotalCharges.")
        
    with col2:
        st.markdown("**2. Imputation (Filling the Blanks)**")
        st.code("df['TotalCharges'].fillna(0, inplace=True)", language="python")
        # Fill NaN with 0 (since they are likely brand new customers with 0 tenure)
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
        st.success("Missing values filled with 0.0")

# --- TAB 3: EDA & ENCODING ---
with tab3:
    st.subheader("Step 3: Exploratory Data Analysis (EDA)")
    
    # 1. Churn Distribution
    churn_counts = df_clean['Churn'].value_counts()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", len(df_clean))
    m2.metric("Churned Customers", churn_counts.get("Yes", 0), delta="-Lost Revenue", delta_color="inverse")
    
    churn_rate = (churn_counts.get("Yes", 0) / len(df_clean)) * 100
    m3.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.divider()
    
    # 2. Visualizations
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("**Churn by Contract Type**")
        # Groupby and plot
        contract_churn = df_clean[df_clean['Churn'] == 'Yes'].groupby('Contract').size()
        st.bar_chart(contract_churn, color="#ff4b4b")
        st.caption("Insight: Month-to-month contracts have massive churn.")
        
    with col_c2:
        st.markdown("**Average Tenure by Churn Status**")
        tenure_churn = df_clean.groupby('Churn')['tenure_months'].mean()
        st.bar_chart(tenure_churn, color="#29b5e8")
        st.caption("Insight: Customers who stay longer (high tenure) rarely churn.")

    st.divider()
    
    st.subheader("Step 4: Categorical Encoding")
    st.write("XGBoost needs numbers, not text. We must convert 'Yes/No' to '1/0' and One-Hot Encode categories like Contract.")
    
    # Binary Encoding
    binary_cols = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
        
    df_clean['gender'] = df_clean['gender'].map({'Male': 1, 'Female': 0})
    
    # One-Hot Encoding
    df_ml = pd.get_dummies(df_clean, columns=['Contract'], drop_first=True)
    
    # Drop ID column as it holds no predictive value
    df_ml = df_ml.drop('customerID', axis=1)
    
    st.dataframe(df_ml.head(), use_container_width=True)
    st.success("✅ Dataset is fully numeric and ready for Machine Learning!")
    
    # Save to session state so Day 2 can access it (simulating saving to a CSV/DB)
    st.session_state['ml_data'] = df_ml