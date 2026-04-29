import streamlit as st
import pandas as pd
from transformers import pipeline
import time
import json
import re
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ticket Pipeline", page_icon="⚙️", layout="wide")

st.title("⚙️ Support Ticket Router: Automated Pipeline")
st.markdown("**Project 2, Day 3: Data Cleaning, JSON Formatting & Batch Processing**")

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_transformer():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

try:
    with st.spinner("Waking up the NLP Brain..."):
        classifier = load_transformer()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

DEPARTMENTS = ["Billing", "Technical Support", "Sales", "Refunds"]

# --- 2. PIPELINE FUNCTIONS ---
def clean_text(text):
    """Sanitize raw email text before inference."""
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove potential email signatures/footer artifacts (simple heuristic)
    text = text.split("Best regards")[0].split("Thanks,")[0]
    return text

def process_ticket(raw_text):
    """The core pipeline: Clean -> Infer -> JSON Structure"""
    start_time = time.time()
    
    # 1. Clean
    cleaned_text = clean_text(raw_text)
    
    # 2. Infer
    result = classifier(
        cleaned_text, 
        candidate_labels=DEPARTMENTS,
        hypothesis_template="This customer support ticket is about {}."
    )
    
    inference_time = round(time.time() - start_time, 2)
    
    # 3. Structure JSON
    top_dept = result['labels'][0]
    top_score = result['scores'][0]
    
    # Flag for human review if the model is unsure
    requires_review = bool(top_score < 0.6)
    
    structured_output = {
        "metadata": {
            "processing_time_sec": inference_time,
            "requires_human_review": requires_review
        },
        "input": {
            "raw_text_length": len(raw_text),
            "cleaned_text": cleaned_text
        },
        "routing_decision": {
            "assigned_department": top_dept,
            "confidence_score": round(top_score, 3)
        },
        "alternative_routes": {
            result['labels'][1]: round(result['scores'][1], 3),
            result['labels'][2]: round(result['scores'][2], 3)
        }
    }
    return structured_output

# --- APP LAYOUT ---
tab1, tab2 = st.tabs(["💻 JSON API Simulator", "📦 Batch CSV Processor"])

# --- TAB 1: API SIMULATOR ---
with tab1:
    st.subheader("Single Ticket API Simulator")
    st.info("Watch how the pipeline transforms a messy, unstructured email into a strict, machine-readable JSON object.")
    
    sample_email = """
    Hello team,
    
    I've been trying to download my invoice for the past 3 hours but the page keeps throwing a 504 Gateway Timeout error. 
    Can someone please look into this? 
    
    Thanks,
    John Doe
    CEO, ExampleCorp
    """
    
    user_input = st.text_area("Incoming Customer Email:", value=sample_email.strip(), height=150)
    
    if st.button("⚙️ Run Pipeline", type="primary"):
        if user_input:
            with st.spinner("Processing..."):
                json_result = process_ticket(user_input)
                
                col_ui, col_json = st.columns(2)
                
                with col_ui:
                    st.success("Pipeline Execution Complete.")
                    st.metric("Assigned Route", json_result['routing_decision']['assigned_department'])
                    st.metric("Confidence", f"{json_result['routing_decision']['confidence_score']*100:.1f}%")
                    if json_result['metadata']['requires_human_review']:
                        st.warning("⚠️ Flagged for Human Review")
                        
                with col_json:
                    st.write("**Structured API Payload (JSON Output):**")
                    st.json(json_result)

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.subheader("Batch Ticket Routing")
    st.write("Upload a CSV file containing a column of support tickets to route them all at once.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Look for a likely text column
        text_col = None
        for col in df.columns:
            if 'text' in col.lower() or 'ticket' in col.lower() or 'email' in col.lower() or 'message' in col.lower():
                text_col = col
                break
                
        if text_col:
            st.success(f"Detected text column: `{text_col}`")
            st.dataframe(df.head(3))
            
            if st.button("🚀 Process Batch"):
                processed_data = []
                progress_bar = st.progress(0)
                
                # Limit to 10 for demonstration to prevent long loading times
                max_process = min(10, len(df)) 
                
                with st.spinner(f"Routing {max_process} tickets..."):
                    for i in range(max_process):
                        text = str(df.iloc[i][text_col])
                        result = process_ticket(text)
                        
                        processed_data.append({
                            "Original_Text": text,
                            "Assigned_Department": result['routing_decision']['assigned_department'],
                            "Confidence": result['routing_decision']['confidence_score'],
                            "Needs_Review": result['metadata']['requires_human_review']
                        })
                        progress_bar.progress((i + 1) / max_process)
                        
                result_df = pd.DataFrame(processed_data)
                st.success("Batch Processing Complete!")
                st.dataframe(result_df, use_container_width=True)
                
                # Download button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Routed Tickets", csv, "routed_tickets.csv", "text/csv")
        else:
            st.error("Could not automatically detect a text column. Please ensure your CSV has a column named 'ticket' or 'message'.")