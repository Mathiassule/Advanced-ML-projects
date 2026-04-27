import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Support Ticket Router", page_icon="🎫", layout="wide")

st.title("🎫 Support Ticket Router: NLP Pipeline")
st.markdown("**Project 2, Day 1: TF-IDF Baseline Architecture**")

# --- 1. DATA GENERATION ENGINE ---
@st.cache_data
def generate_ticket_data(n=1500):
    """Generates synthetic customer support tickets mapped to specific departments."""
    np.random.seed(42)
    
    departments = ["Billing", "Technical Support", "Sales", "Refunds"]
    
    # Keyword templates to simulate real-world semantic patterns
    templates = {
        "Billing": [
            "I was overcharged on my last invoice.",
            "Update my credit card information.",
            "Why is my monthly bill higher than usual?",
            "I need a receipt for last month's transaction.",
            "My payment failed to process."
        ],
        "Technical Support": [
            "The application keeps crashing on startup.",
            "I can't log into my account. Password reset isn't working.",
            "Error code 500 when I try to save my profile.",
            "The dashboard is not loading on Chrome.",
            "API endpoint is timing out continuously."
        ],
        "Sales": [
            "I want to upgrade to the Enterprise tier.",
            "Can I get a demo of the new features?",
            "What are your volume pricing discounts?",
            "I need to add 5 more seats to our team plan.",
            "Does the Pro plan include priority support?"
        ],
        "Refunds": [
            "I want my money back, I forgot to cancel.",
            "Cancel my subscription and issue a refund.",
            "I am not satisfied with the product, please refund.",
            "Chargeback requested. Please return my funds.",
            "I accidentally bought the wrong tier, need a refund."
        ]
    }
    
    data = []
    for _ in range(n):
        dept = np.random.choice(departments, p=[0.3, 0.4, 0.15, 0.15])
        base_text = np.random.choice(templates[dept])
        
        # Inject some noise/variations
        noise = np.random.choice([" Hello, ", " Hi team, ", " Urgent: ", "", " Please help: "])
        text = noise + base_text
        data.append({"ticket_text": text, "department": dept})
        
    return pd.DataFrame(data)

df = generate_ticket_data()

# --- 2. PIPELINE FUNCTIONS ---
def clean_text(text):
    """Standardizes text: lowercase, removes special chars."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['clean_text'] = df['ticket_text'].apply(clean_text)

# --- APP LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📊 Dataset & EDA", "⚙️ Baseline Training", "🧪 Live Routing Test"])

# --- TAB 1: EDA ---
with tab1:
    st.subheader("Data Ingestion & Cleaning")
    st.write(f"Ingested {len(df)} support tickets.")
    st.dataframe(df[['ticket_text', 'clean_text', 'department']].head(13), use_container_width=True)
    
    st.divider()
    st.subheader("Class Distribution")
    dist = df['department'].value_counts()
    st.bar_chart(dist)
    st.caption("Note: Imbalanced classes (Tech Support dominates). We'll handle this in the ML pipeline.")

# --- TAB 2: MODEL TRAINING ---
with tab2:
    st.subheader("TF-IDF + Logistic Regression Baseline")
    
    X = df['clean_text']
    y = df['department']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if st.button("🚀 Train Baseline Model", type="primary"):
        with st.spinner("Vectorizing text and training LogReg..."):
            
            # 1. Vectorization (Text to Numbers)
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # 2. Model Training
            model = LogisticRegression(class_weight='balanced', max_iter=1000)
            model.fit(X_train_vec, y_train)
            
            # 3. Evaluation
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            
            # Save artifacts to session state for testing
            st.session_state['vectorizer'] = vectorizer
            st.session_state['lr_model'] = model
            
            st.success(f"✅ Baseline trained successfully! Overall Accuracy: {acc*100:.2f}%")
            
            st.markdown("**Detailed Classification Report**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format(precision=3), use_container_width=True)

# --- TAB 3: LIVE TESTING ---
with tab3:
    st.subheader("Simulate Incoming Email")
    
    if 'lr_model' in st.session_state:
        user_input = st.text_area("Copy/Paste a customer email here:", "Hey, my screen goes completely blank when I try to open the dashboard. Help!")
        
        if st.button("Route Ticket"):
            # 1. Clean
            cleaned = clean_text(user_input)
            # 2. Vectorize
            vec = st.session_state['vectorizer'].transform([cleaned])
            # 3. Predict
            pred_dept = st.session_state['lr_model'].predict(vec)[0]
            probs = st.session_state['lr_model'].predict_proba(vec)[0]
            classes = st.session_state['lr_model'].classes_
            
            st.divider()
            st.markdown(f"### 🎯 Assigned Route: **{pred_dept}**")
            
            # Show confidence scores
            st.write("**Confidence Scores by Department:**")
            prob_df = pd.DataFrame({"Department": classes, "Probability": probs})
            prob_df = prob_df.sort_values(by="Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Department"))
            
    else:
        st.info("👈 Please train the baseline model in Tab 2 first.")