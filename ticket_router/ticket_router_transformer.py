import streamlit as st
import pandas as pd
from transformers import pipeline
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Transformer Router", page_icon="🤖", layout="wide")

st.title("🤖 Support Ticket Router: NLP Pipeline")
st.markdown("**Project 2, Day 2: Hugging Face Transformer Integration**")

# --- 1. MODEL LOADING (THE BRAIN) ---
@st.cache_resource
def load_transformer():
    """
    Loads a pre-trained Zero-Shot Classification model.
    We upgraded to BART-large-MNLI, which is the industry standard for highly accurate zero-shot routing.
    """
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

try:
    with st.spinner("Downloading Hugging Face Transformer... (This takes a minute on the first run)"):
        classifier = load_transformer()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- APP LAYOUT ---
tab1, tab2 = st.tabs(["🧪 Zero-Shot Live Test", "🧠 How it Works"])

# Define our specific business departments
DEPARTMENTS = ["Billing", "Technical Support", "Sales", "Refunds"]

# --- TAB 1: LIVE TESTING ---
with tab1:
    st.subheader("Simulate Incoming Email (Transformer Engine)")
    st.info("Unlike yesterday, this model hasn't been trained on our specific data. It relies on its vast understanding of the English language to route the ticket.")
    
    user_input = st.text_area(
        "Copy/Paste a customer email here:", 
        "Hello! I am trying to add 3 new team members to our workspace but I can't find the pricing tier options."
    )
    
    if st.button("🚀 Route via Transformer", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter an email text.")
        else:
            with st.spinner("Analyzing semantic context..."):
                start_time = time.time()
                
                # 1. Run the Zero-Shot Pipeline
                # NEW: Added a hypothesis template to give the AI context about the task!
                result = classifier(
                    user_input, 
                    candidate_labels=DEPARTMENTS,
                    hypothesis_template="This customer support ticket is about {}."
                )
                
                inference_time = time.time() - start_time
                
                # Extract Results
                top_department = result['labels'][0]
                top_score = result['scores'][0]
                
                # 2. Display Results
                st.divider()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"### 🎯 Assigned Route:")
                    if top_score > 0.6:
                        st.success(f"**{top_department}**")
                    else:
                        # If the model is unsure, flag for human review
                        st.warning(f"**{top_department}** (Low Confidence - Needs Human Review)")
                        
                    st.metric("Confidence Score", f"{top_score * 100:.1f}%")
                    st.caption(f"Inference Time: {inference_time:.2f}s")
                    
                with col2:
                    st.write("**Confidence Distribution:**")
                    # Create a dataframe for the bar chart
                    prob_df = pd.DataFrame({
                        "Department": result['labels'],
                        "Probability": result['scores']
                    })
                    st.bar_chart(prob_df.set_index("Department"))

# --- TAB 2: ARCHITECTURE EXPLANATION ---
with tab2:
    st.subheader("TF-IDF Baseline vs. Transformers")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 📉 Day 1: TF-IDF (Statistical)")
        st.markdown("""
        * **How it works:** Counts how often a word appears. 
        * **Pros:** Blazing fast. Uses almost no memory.
        * **Cons:** Has zero understanding of context. If it hasn't seen the word 'invoice' mapped to 'Billing' during training, it will fail. Cannot understand sarcasm or complex phrasing.
        * **Training:** Required a dataset to learn from.
        """)
        
    with col_b:
        st.markdown("### 🧠 Day 2: Transformers (Semantic)")
        st.markdown("""
        * **How it works:** Uses 'Attention Mechanisms' to understand how words relate to each other in a sentence.
        * **Pros:** Deep contextual understanding. Handles synonyms and complex sentence structures effortlessly. 
        * **Cons:** Computationally expensive (slower). Takes up hundreds of MBs in memory.
        * **Training:** Zero-Shot. It works out-of-the-box without us providing any training data!
        """)