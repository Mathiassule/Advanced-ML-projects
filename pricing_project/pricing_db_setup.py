import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Data Warehouse Manager", page_icon="🗄️", layout="wide")

st.title("🗄️ Data Warehouse Manager")
st.markdown("**Project 3, Day 1: Relational Database & Data Engineering**")

DB_NAME = "ecommerce_pricing.db"

# --- CORE DATABASE FUNCTIONS ---
def init_db():
    """Connects to SQLite and creates the table schema if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create the schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historical_pricing (
        transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        date TEXT NOT NULL,
        base_price REAL NOT NULL,
        competitor_price REAL NOT NULL,
        inventory_level INTEGER NOT NULL,
        demand_score REAL NOT NULL,
        actual_price_sold REAL NOT NULL,
        units_sold INTEGER NOT NULL
    )
    ''')
    conn.commit()
    return conn

def generate_ecommerce_data(rows=1000):
    """Generates synthetic e-commerce supply & demand data."""
    np.random.seed(42)
    
    # Generate dates over the last 90 days
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=np.random.randint(0, 90))).strftime("%Y-%m-%d") for _ in range(rows)]
    
    # Synthetic Features
    product_ids = np.random.choice(["SKU_101 (Laptop)", "SKU_202 (Headphones)", "SKU_303 (Monitor)"], rows)
    base_prices = np.where(product_ids == "SKU_101 (Laptop)", 999.99, 
                  np.where(product_ids == "SKU_202 (Headphones)", 149.99, 299.99))
    
    # Competitors usually price slightly above or below the base price
    competitor_prices = base_prices * np.random.uniform(0.90, 1.10, rows)
    
    # Inventory from 0 (stockout) to 500 (overstocked)
    inventory_levels = np.random.randint(0, 500, rows)
    
    # Demand score (0.0 to 1.0) - High demand when competitor price is high
    demand_score = np.clip(np.random.normal(0.5, 0.2, rows) + (competitor_prices > base_prices) * 0.2, 0.0, 1.0)
    
    # The actual price the item was sold for historically (what our ML will try to optimize)
    actual_price_sold = base_prices * np.random.uniform(0.85, 1.15, rows)
    
    # Units sold heavily depend on how cheap we were compared to competitors and demand
    units_sold = np.maximum(1, np.int32((demand_score * 50) + ((competitor_prices - actual_price_sold) / 10)))
    
    df = pd.DataFrame({
        "product_id": product_ids,
        "date": dates,
        "base_price": np.round(base_prices, 2),
        "competitor_price": np.round(competitor_prices, 2),
        "inventory_level": inventory_levels,
        "demand_score": np.round(demand_score, 2),
        "actual_price_sold": np.round(actual_price_sold, 2),
        "units_sold": units_sold
    })
    
    # Sort by date for realism
    df = df.sort_values("date").reset_index(drop=True)
    return df

# --- APP LAYOUT ---
conn = init_db()

tab1, tab2, tab3 = st.tabs(["⚙️ Database Setup", "📊 Data Ingestion", "🔍 SQL Playground"])

with tab1:
    st.subheader("Database Status")
    
    db_exists = os.path.exists(DB_NAME)
    if db_exists:
        st.success(f"✅ Connected to local database: `{DB_NAME}`")
        
        # Check row count
        count = pd.read_sql("SELECT COUNT(*) as count FROM historical_pricing", conn).iloc[0]['count']
        st.metric("Total Rows in Database", f"{count:,}")
        
        if st.button("🗑️ Nuke Database (Reset)"):
            conn.cursor().execute("DROP TABLE historical_pricing")
            conn.commit()
            st.rerun()
    else:
        st.warning("Database not initialized.")

with tab2:
    st.subheader("ETL Pipeline (Extract, Transform, Load)")
    st.write("Generate synthetic e-commerce data and load it into the SQL Database.")
    
    num_rows = st.slider("Rows to Generate", 100, 5000, 1000, step=100)
    
    if st.button("🚀 Run Data Generation & Load to DB", type="primary"):
        with st.spinner("Generating and injecting data..."):
            # 1. Generate Pandas DataFrame
            df_new = generate_ecommerce_data(num_rows)
            
            # 2. SQL Injection (The Load step of ETL)
            df_new.to_sql('historical_pricing', conn, if_exists='append', index=False)
            
            st.success(f"✅ Successfully injected {num_rows} rows into `historical_pricing` table!")
            st.dataframe(df_new.head(), use_container_width=True)

with tab3:
    st.subheader("SQL Query Playground")
    st.info("Write raw SQL queries to extract data from your new relational database.")
    
    default_query = """
    SELECT 
        product_id, 
        ROUND(AVG(actual_price_sold), 2) as avg_price, 
        SUM(units_sold) as total_units_sold
    FROM historical_pricing
    GROUP BY product_id
    ORDER BY total_units_sold DESC;
    """
    
    sql_query = st.text_area("SQL Query", value=default_query.strip(), height=150)
    
    if st.button("▶️ Execute Query"):
        try:
            result_df = pd.read_sql(sql_query, conn)
            st.dataframe(result_df, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")