import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="📈", layout="wide")

st.title("📈 Dynamic Pricing ML Engine")
st.markdown("**Project 3, Day 2: Demand Curve Modeling & Price Optimization**")

DB_NAME = "ecommerce_pricing.db"

# --- 1. DATA EXTRACTION ---
@st.cache_data
def load_data():
    if not os.path.exists(DB_NAME):
        return None
    
    conn = sqlite3.connect(DB_NAME)
    # Pull data from our warehouse
    query = "SELECT * FROM historical_pricing"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

df_raw = load_data()

if df_raw is None or df_raw.empty:
    st.error(f"🚨 Database `{DB_NAME}` not found or is empty. Please run Day 1's `pricing_db_setup.py` first!")
    st.stop()

# --- APP LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🧠 Model Training", "📊 Feature Importance", "💸 Dynamic Pricing Simulator"])

# --- TAB 1: MODEL TRAINING ---
with tab1:
    st.subheader("Train the Demand Prediction Model")
    st.write("We are training a Random Forest to predict `units_sold` based on the market conditions and price.")
    
    # Preprocessing
    # We want to predict Units Sold. The price we actually sold it for is a feature.
    features = ['actual_price_sold', 'competitor_price', 'inventory_level', 'demand_score']
    
    # One-hot encode the product_id
    df_ml = pd.get_dummies(df_raw, columns=['product_id'], drop_first=False)
    
    # Ensure boolean columns are integers (0/1) for scikit-learn
    for col in df_ml.columns:
        if df_ml[col].dtype == bool:
            df_ml[col] = df_ml[col].astype(int)
            
    # Define X and y
    feature_cols = [c for c in df_ml.columns if c in features or 'product_id_' in c]
    X = df_ml[feature_cols]
    y = df_ml['units_sold']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.dataframe(X.head(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of Trees in Forest", 50, 300, 100)
    with col2:
        max_depth = st.slider("Max Tree Depth", 3, 20, 10)
        
    if st.button("🚀 Train Random Forest", type="primary"):
        with st.spinner("Training ML Model..."):
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Save to session state
            st.session_state['model'] = model
            st.session_state['feature_cols'] = feature_cols
            st.session_state['metrics'] = {'mae': mae, 'r2': r2}
            
            st.success("✅ Model Trained Successfully!")
            st.metric("R² Score (Accuracy)", f"{r2*100:.1f}%", help="Closer to 100% means the model perfectly understands the demand curve.")
            st.metric("Mean Absolute Error", f"{mae:.2f} units", help="On average, how many units off is our prediction?")

# --- TAB 2: FEATURE IMPORTANCE ---
with tab2:
    st.subheader("What drives customer demand?")
    if 'model' in st.session_state:
        model = st.session_state['model']
        feature_cols = st.session_state['feature_cols']
        
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='#29b5e8')
        ax.set_title("Random Forest Feature Importance")
        st.pyplot(fig)
        
        st.info("💡 **Insight:** Usually, `actual_price_sold` and `demand_score` are the biggest drivers of units sold. The model learns that lowering the price increases sales!")
    else:
        st.warning("👈 Train the model in Tab 1 first.")

# --- TAB 3: THE OPTIMIZER ---
with tab3:
    st.subheader("💸 Dynamic Pricing Optimizer")
    st.write("Now we use the trained AI to simulate different price points and find the one that maximizes Total Revenue.")
    
    if 'model' in st.session_state:
        model = st.session_state['model']
        feature_cols = st.session_state['feature_cols']
        
        # User Inputs for Market Simulation
        st.markdown("### Current Market Conditions")
        c1, c2, c3, c4 = st.columns(4)
        
        target_product = c1.selectbox("Product", ["SKU_101 (Laptop)", "SKU_202 (Headphones)", "SKU_303 (Monitor)"])
        base_p = 999.99 if "101" in target_product else (149.99 if "202" in target_product else 299.99)
        
        comp_price = c2.number_input("Competitor Price ($)", value=base_p)
        inventory = c3.number_input("Our Inventory Level", value=150)
        demand = c4.slider("Market Demand Score", 0.0, 1.0, 0.7)
        
        if st.button("🔍 Find Optimal Price", type="primary"):
            with st.spinner("Simulating hundreds of price points..."):
                # Generate test prices: -20% to +20% of base price
                test_prices = np.linspace(base_p * 0.8, base_p * 1.2, 50)
                
                # Prepare a dataframe of identical scenarios, just varying the price
                sim_data = pd.DataFrame(columns=feature_cols)
                for i, price in enumerate(test_prices):
                    sim_data.loc[i, 'actual_price_sold'] = price
                    sim_data.loc[i, 'competitor_price'] = comp_price
                    sim_data.loc[i, 'inventory_level'] = inventory
                    sim_data.loc[i, 'demand_score'] = demand
                    
                    # Set the specific product column to 1, others to 0
                    for col in feature_cols:
                        if 'product_id_' in col:
                            if target_product in col:
                                sim_data.loc[i, col] = 1
                            else:
                                sim_data.loc[i, col] = 0
                
                # Predict Units Sold for every price point
                predicted_units = model.predict(sim_data)
                
                # Calculate Revenue = Price * Units Sold
                predicted_revenue = test_prices * predicted_units
                
                # Find the Best Scenario
                best_idx = np.argmax(predicted_revenue)
                best_price = test_prices[best_idx]
                max_revenue = predicted_revenue[best_idx]
                expected_sales = predicted_units[best_idx]
                
                st.divider()
                st.markdown(f"### 🏆 Recommended Price: **${best_price:.2f}**")
                
                r1, r2, r3 = st.columns(3)
                r1.metric("Projected Units Sold", f"{int(expected_sales)} units")
                r2.metric("Projected Revenue", f"${max_revenue:,.2f}")
                r3.metric("Competitor Difference", f"${best_price - comp_price:.2f}")
                
                # Plot the Revenue Curve
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(test_prices, predicted_revenue, color='green', linewidth=3, label="Projected Revenue")
                ax.axvline(best_price, color='red', linestyle='--', label=f"Optimal Price (${best_price:.2f})")
                ax.set_xlabel("Our Price ($)")
                ax.set_ylabel("Expected Revenue ($)")
                ax.set_title("Price vs. Revenue Curve")
                ax.legend()
                
                st.pyplot(fig)
                st.success("The AI mapped the demand curve and calculated the mathematically perfect price point to maximize your revenue!")

    else:
        st.info("👈 Please train the model in Tab 1 first to unlock the Optimizer.")