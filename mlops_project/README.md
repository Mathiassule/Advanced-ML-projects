🏭 Production-Grade MLOps Pipeline

🚀 Overview

Deploying a Machine Learning model is only 10% of the work; maintaining it in production is the other 90%. This project is a complete MLOps pipeline designed to track experiments, version massive datasets, monitor production data for drift, and automate testing via CI/CD.

🧠 System Architecture

This project implements four core pillars of Machine Learning Operations:

Experiment Tracking (mlops_tracker.py): Integrates MLflow to automatically log hyperparameters, evaluation metrics (Accuracy, Precision, Recall), and serialize .pkl model artifacts into a centralized SQLite tracking database.

Data Versioning (DVC): Utilizes Data Version Control (DVC) to track shifts in the factory_data.csv dataset, keeping the Git repository lightweight while allowing instant "time travel" to previous data states.

Data Drift Detection (mlops_drift_detector.py): A custom, statistical monitoring engine that continuously compares the Reference (Training) dataset against the Current (Production) dataset to flag feature drift before it degrades model performance.

Continuous Integration (CI/CD): A GitHub Actions workflow (ci_cd.yml) that automatically provisions an Ubuntu server, installs dependencies, and executes a pytest suite to validate data distributions and mathematical logic on every git push.

⚙️ Core Technologies

MLOps Tools: MLflow, DVC (Data Version Control)

Testing & Automation: Pytest, GitHub Actions (CI/CD)

Machine Learning: Scikit-Learn (Random Forest)

Frontend Dashboard: Streamlit, Pandas, NumPy

💻 Running the Pipeline Locally

1. Run the Experiment Tracker:

streamlit run mlops_tracker.py


(To view the MLflow UI, open a second terminal and run: mlflow ui --backend-store-uri sqlite:///mlruns.db)

2. Test Data Versioning (DVC):
Toggle the "Simulate Data Drift" button in the tracker, then use DVC to track the file changes:

dvc add factory_data.csv
git add factory_data.csv.dvc
git commit -m "Track drifted dataset"


3. Run the Drift Detector:

streamlit run mlops_drift_detector.py


4. Run the Unit Tests:

pytest test_mlops.py
