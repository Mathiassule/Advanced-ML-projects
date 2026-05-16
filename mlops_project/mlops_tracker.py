import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="MLOps Tracker", page_icon="🧪", layout="wide")

st.title("🧪 MLOps: Experiment Tracking")
st.markdown("**Project 4, Day 1: MLflow Integration**")

# --- 1. SETUP MLFLOW ---
# We tell MLflow to create a local database (a folder named 'mlruns') to store all our logs.
EXPERIMENT_NAME = "Predictive_Maintenance_Optimization"
mlflow.set_tracking_uri("sqlite:///mlruns.db") # Use a local SQLite DB for MLflow

# Create experiment if it doesn't exist
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(EXPERIMENT_NAME)

mlflow.set_experiment(EXPERIMENT_NAME)

# --- 2. DATA GENERATION ---
@st.cache_data
def generate_factory_data(n=1000):
    """Simulates factory machine sensors (Temperature, Vibration, Pressure)."""
    np.random.seed(42)
    temp = np.random.normal(70, 10, n)
    vibration = np.random.normal(5, 2, n)
    pressure = np.random.normal(100, 15, n)
    
    # Machine fails if it gets too hot and vibrates too much
    failure_prob = np.where((temp > 85) & (vibration > 7), 0.8, 0.05)
    failure = [1 if np.random.rand() < p else 0 for p in failure_prob]
    
    return pd.DataFrame({
        "temperature_c": temp,
        "vibration_hz": vibration,
        "pressure_psi": pressure,
        "machine_failure": failure
    })

df = generate_factory_data()
X = df.drop("machine_failure", axis=1)
y = df["machine_failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- APP LAYOUT ---
tab1, tab2 = st.tabs(["🏋️‍♂️ Train & Log Model", "📊 MLflow Registry"])

with tab1:
    st.subheader("Model Training Dashboard")
    st.write("Every time you click train, MLflow will secretly record everything in the background.")
    
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 200, 50, step=10)
    with col2:
        max_depth = st.slider("Max Depth", 2, 20, 5)
        
    run_name = st.text_input("Experiment Run Name (Optional):", placeholder="e.g. baseline_test")
    
    if st.button("🚀 Train & Log to MLflow", type="primary"):
        with st.spinner("Training model and syncing with MLflow..."):
            
            # --- THE MLOPS MAGIC ---
            # We wrap our training inside an MLflow "run"
            with mlflow.start_run(run_name=run_name if run_name else None) as run:
                
                # 1. Train Model
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                
                # 2. Evaluate
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds)
                rec = recall_score(y_test, preds)
                
                # 3. Log Hyperparameters (The settings)
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                
                # 4. Log Metrics (The results)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                
                # 5. Log the Model itself (The .pkl artifact)
                mlflow.sklearn.log_model(model, "random_forest_model")
                
                run_id = run.info.run_id
                
            st.success(f"✅ Experiment Logged Successfully! Run ID: `{run_id}`")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc*100:.1f}%")
            m2.metric("Precision", f"{prec*100:.1f}%")
            m3.metric("Recall", f"{rec*100:.1f}%")

with tab2:
    st.subheader("Experiment History")
    st.info("💡 **Pro Tip:** While this table shows a summary, MLflow has its own beautiful web dashboard! Check the instructions below to open it.")
    
    # Pull data directly from MLflow's database
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if not runs.empty:
            # Clean up the dataframe for display
            display_cols = [col for col in runs.columns if 'metrics.' in col or 'params.' in col or col in ['run_id', 'start_time', 'tags.mlflow.runName']]
            clean_runs = runs[display_cols].copy()
            clean_runs.columns = [c.replace('metrics.', '').replace('params.', '').replace('tags.mlflow.runName', 'Run Name') for c in clean_runs.columns]
            
            st.dataframe(clean_runs, use_container_width=True)
        else:
            st.write("No runs logged yet. Go to Tab 1 and train a model!")
    except Exception as e:
        st.write("Awaiting first experiment...")
        
    st.divider()
    st.markdown("""
    ### 🌐 How to open the official MLflow UI:
    To see the full power of MLflow (including charting, graphs, and model downloading), open a **new terminal window** in your project folder and run:
    
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlruns.db
    ```
    Then, open your browser to **http://127.0.0.1:5000**
    """)  