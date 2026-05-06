import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pipeline Orchestrator", page_icon="⚙️", layout="wide")

st.title("⚙️ Apache Airflow Simulator")
st.markdown("**Project 3, Day 3: Automated Data Pipelines (DAGs)**")

DB_NAME = "ecommerce_pricing.db"

# --- INIT NEW DATABASE TABLE ---
def init_update_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_price_updates (
        update_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        product_id TEXT NOT NULL,
        old_price REAL NOT NULL,
        new_optimized_price REAL NOT NULL,
        expected_revenue REAL NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

init_update_table()

# --- AIRFLOW TASKS (THE PIPELINE) ---

def task_1_extract_and_train():
    """Extracts raw data from SQL and trains the ML model."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM historical_pricing", conn)
    conn.close()
    
    if df.empty:
        raise ValueError("Database is empty!")

    features = ['actual_price_sold', 'competitor_price', 'inventory_level', 'demand_score']
    df_ml = pd.get_dummies(df, columns=['product_id'])
    
    for col in df_ml.columns:
        if df_ml[col].dtype == bool:
            df_ml[col] = df_ml[col].astype(int)
            
    feature_cols = [c for c in df_ml.columns if c in features or 'product_id_' in c]
    
    X = df_ml[feature_cols]
    y = df_ml['units_sold']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols, df

def task_2_optimize_prices(model, feature_cols, raw_df):
    """Loops through all products and finds the optimal price."""
    # Get the latest market conditions for each product
    latest_data = raw_df.sort_values('date').groupby('product_id').last().reset_index()
    
    recommendations = []
    
    for _, row in latest_data.iterrows():
        product = row['product_id']
        base_p = row['base_price']
        comp_price = row['competitor_price']
        inventory = row['inventory_level']
        demand = row['demand_score']
        
        test_prices = np.linspace(base_p * 0.8, base_p * 1.2, 50)
        sim_data = pd.DataFrame(columns=feature_cols)
        
        for i, price in enumerate(test_prices):
            sim_data.loc[i, 'actual_price_sold'] = price
            sim_data.loc[i, 'competitor_price'] = comp_price
            sim_data.loc[i, 'inventory_level'] = inventory
            sim_data.loc[i, 'demand_score'] = demand
            
            for col in feature_cols:
                if 'product_id_' in col:
                    sim_data.loc[i, col] = 1 if product in col else 0
                    
        # Predict & Optimize
        predicted_units = model.predict(sim_data)
        predicted_revenue = test_prices * predicted_units
        best_idx = np.argmax(predicted_revenue)
        
        recommendations.append({
            "product_id": product,
            "old_price": row['actual_price_sold'],
            "new_optimized_price": round(test_prices[best_idx], 2),
            "expected_revenue": round(predicted_revenue[best_idx], 2)
        })
        
    return pd.DataFrame(recommendations)

def task_3_load_to_db(recommendations_df):
    """Loads the new optimized prices back into the SQL database."""
    conn = sqlite3.connect(DB_NAME)
    
    # Add timestamp
    recommendations_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write to new table
    recommendations_df.to_sql('daily_price_updates', conn, if_exists='append', index=False)
    conn.close()

# --- APP UI ---
st.subheader("DAG: Daily Pricing Optimization")
st.write("This dashboard simulates an Apache Airflow Directed Acyclic Graph (DAG) scheduled to run every midnight.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Execution Panel")
    if st.button("▶️ Trigger Pipeline Run", type="primary", use_container_width=True):
        
        # UI Elements for Task Tracking
        st.divider()
        t1 = st.empty()
        t2 = st.empty()
        t3 = st.empty()
        
        try:
            # --- TASK 1 ---
            t1.info("⏳ Task 1: Extracting DB & Retraining Model...")
            time.sleep(1.5) # Simulate processing time
            model, features, raw_df = task_1_extract_and_train()
            t1.success("✅ Task 1 Complete: Model Synced.")
            
            # --- TASK 2 ---
            t2.info("⏳ Task 2: Running Optimization Simulations...")
            time.sleep(2) # Simulate processing time
            recs_df = task_2_optimize_prices(model, features, raw_df)
            t2.success("✅ Task 2 Complete: Prices Optimized.")
            
            # --- TASK 3 ---
            t3.info("⏳ Task 3: Loading Updates to SQL Warehouse...")
            time.sleep(1) # Simulate processing time
            task_3_load_to_db(recs_df)
            t3.success("✅ Task 3 Complete: Database Updated.")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Pipeline Failed: {e}")

with col2:
    st.markdown("### 🗄️ Database Live View (`daily_price_updates`)")
    st.caption("This table shows the pipeline's output. Refresh the app to see historical logs.")
    
    try:
        conn = sqlite3.connect(DB_NAME)
        logs_df = pd.read_sql("SELECT * FROM daily_price_updates ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True, hide_index=True)
            
            st.success(f"Pipeline has successfully executed **{len(logs_df) // 3}** times.")
        else:
            st.info("Table is empty. Awaiting first pipeline run.")
            
    except Exception as e:
        st.warning("Database not initialized yet.")