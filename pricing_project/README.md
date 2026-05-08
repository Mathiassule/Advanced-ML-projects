💸 Dynamic Pricing Engine Pipeline

🚀 Overview

In modern e-commerce, pricing is not static. This project is an end-to-end Machine Learning and Data Engineering pipeline that autonomously adjusts product prices based on competitor behavior, inventory levels, and historical demand. Instead of simply predicting a price, the AI predicts the Demand Curve and simulates multiple price points to find the exact dollar amount that mathematically maximizes total revenue.

🧠 System Architecture

This project implements a complete Extract, Transform, Load (ETL) and ML workflow:

Data Warehouse (pricing_db_setup.py): A foundational SQLite relational database holding thousands of rows of historical supply, demand, and transactional data.

The ML Brain (pricing_ml_engine.py): A Random Forest Regressor trained on the database to predict units_sold. It features a built-in optimizer that simulates 50 different price points to isolate the highest-revenue scenario.

The Orchestrator (pricing_orchestrator.py): A simulated Apache Airflow DAG designed to run on a schedule. It automatically extracts fresh data, retrains the model, optimizes all product prices, and loads the new pricing strategy back into a live SQL table.

⚙️ Core Technologies

Data Engineering: SQLite, SQL (SELECT, GROUP BY, INSERT), Pandas

Machine Learning: Scikit-Learn (Random Forest Regressor), Feature Importance Analysis

Orchestration: Directed Acyclic Graphs (DAGs), Automated ETL Pipelines

Version Control: Git (with strict .gitignore rules to protect data/model artifacts)

💻 Running the Pipeline Locally

1. Initialize the Database:
Generate the synthetic e-commerce data and populate the SQL tables.

streamlit run pricing_db_setup.py


2. Explore the Machine Learning Model:
Test the pricing simulator and view feature importances.

streamlit run pricing_ml_engine.py


3. Run the Automated Orchestrator:
Simulate the daily Airflow job that updates prices without human intervention.

streamlit run pricing_orchestrator.py


📊 Business Impact

By shifting from "cost-plus pricing" to "dynamic machine learning pricing," businesses can automatically react to competitor price drops or low inventory scenarios. The orchestrator ensures that the company's pricing strategy is re-aligned every single day, maximizing gross revenue with zero manual overhead.