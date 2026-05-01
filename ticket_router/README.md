🎫 Automated Support Ticket Router

🚀 Overview

Customer support teams waste countless hours manually reading and assigning incoming tickets. This project is an end-to-end Natural Language Processing (NLP) pipeline that ingests unstructured customer emails, analyzes the semantic context, and automatically routes them to the correct department (Billing, Technical Support, Sales, or Refunds).

🧠 Model Architecture

This project implements a progressive AI architecture:

The Baseline (ticket_router_baseline.py): A lightweight statistical model using TF-IDF Vectorization and Logistic Regression (with balanced class weights) to establish a performance benchmark.

The Production Engine (ticket_router_pipeline.py): A state-of-the-art Deep Learning approach using Hugging Face's facebook/bart-large-mnli model for Zero-Shot Classification. This allows the system to accurately categorize complex tickets without requiring extensive domain-specific training data.

⚙️ Features

Intelligent Preprocessing: RegEx-based sanitization to strip email noise, headers, and signatures.

JSON API Formatting: Converts raw text into structured JSON payloads containing routing decisions, confidence scores, and execution times.

Human-in-the-Loop: Automatically flags ambiguous tickets (confidence < 60%) for manual review.

Batch Processing Engine: Ingests CSV files of historical tickets and processes them in bulk for database tagging.

💻 Running Locally

This application runs natively using a standard Python environment.

1. Install Dependencies:
Ensure your virtual environment is activated, then navigate to the project directory and install the required libraries:

pip install -r requirements.txt


2. Run the Application:
Start the Streamlit dashboard by running:

streamlit run ticket_router_pipeline.py


3. Access the Dashboard:
Open your web browser and navigate to http://localhost:8501.

🛠️ Future Maintenance & Model Swapping

The Zero-Shot pipeline is designed to be highly modular. To upgrade or swap the AI model in the future:

Open ticket_router_pipeline.py.

Locate the @st.cache_resource function.

Change the model="facebook/bart-large-mnli" string to any compatible Hugging Face model.