import streamlit as st
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Success Dashboard", page_icon="🎧", layout="wide")

st.title("🎧 Customer Success Portal")
st.markdown("**Project 1, Day 5: Business Impact & ROI**")

# The URL where your FastAPI server is running
API_URL = "https://churn-api-0s8x.onrender.com/predict"

st.sidebar.header("Agent Controls")
st.sidebar.info("Enter the customer's details while on the phone with them, then click 'Check Risk'.")

# --- UI: CUSTOMER DATA FORM ---
with st.form("customer_form"):
    st.subheader("Customer Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        
    with col2:
        st.markdown("**Account Details**")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=5)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        
    with col3:
        st.markdown("**Billing**")
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=85.50)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=427.50)
        
    submit_btn = st.form_submit_button("🔍 Check Retention Risk", type="primary", use_container_width=True)

# --- API INTEGRATION ---
if submit_btn:
    # 1. Format the data to match the FastAPI Pydantic schema perfectly
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure_months": tenure,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    
    with st.spinner("Connecting to AI Brain..."):
        try:
            # 2. Send HTTP POST request to the API
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                pred = data['prediction']
                biz = data['business_logic']
                
                st.divider()
                st.subheader("AI Analysis Result")
                
                # 3. Display the results beautifully
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    if pred['risk_level'] == "High":
                        st.error(f"⚠️ Risk Level: {pred['risk_level']}")
                    else:
                        st.success(f"✅ Risk Level: {pred['risk_level']}")
                        
                with res_col2:
                    st.metric("Churn Probability", f"{pred['churn_probability_score'] * 100:.1f}%")
                    
                with res_col3:
                    if pred['risk_level'] == "High":
                        st.warning(f"🎯 Action: {biz['recommended_action']}")
                    else:
                        st.info(f"🎯 Action: {biz['recommended_action']}")
                        
                # --- DAY 5: BUSINESS IMPACT METRICS ---
                st.divider()
                st.subheader("💼 Business Impact Analysis")
                
                annual_revenue = monthly * 12
                
                b1, b2 = st.columns(2)
                
                with b1:
                    if pred['risk_level'] == "High":
                        st.metric("Annual Value at Risk", f"${annual_revenue:,.2f}", delta="-High Risk", delta_color="inverse")
                    else:
                        st.metric("Secured Annual Revenue", f"${annual_revenue:,.2f}", delta="Low Risk", delta_color="normal")
                        
                with b2:
                    if pred['risk_level'] == "High":
                        cost_of_action = annual_revenue * 0.20
                        revenue_saved = annual_revenue * 0.80
                        st.write(f"📉 **Cost of 20% Discount:** ${cost_of_action:,.2f}")
                        st.write(f"📈 **Net Revenue Preserved:** ${revenue_saved:,.2f}")
                    else:
                        st.write("✅ **Customer is stable.**")
                        st.write("No discount required. Maintain standard engagement protocol.")
                        
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Could not connect to the API.")
            st.warning("**Troubleshooting:** Is your FastAPI server running? Open a separate terminal and run: `uvicorn churn_api:app --reload`")